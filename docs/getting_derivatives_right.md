# Getting derivatives right

Moju audits governing-law residuals. Those residuals are only as meaningful as the
derivatives that enter them. This page covers how to supply or compute derivatives,
how to keep every term at one instant, and how to read the per-term breakdown.

## Prefer solver (or autodiff) derivatives when you have them

When your surrogate or CFD export already provides gradients, Laplacians, and time
derivatives at the evaluation points, pass them in `state_pred` (Path B) or produce
them with a `state_builder` (Path A / PINN autodiff). Moju records those keys as
`supplied` in derivative provenance and does not overwrite them.

Finite-difference or spectral fill (`auto_path_b_derivatives` / `fill_law_fd`) is a
fallback for structured grids when a derivative is missing. It introduces truncation
error that can dominate the residual on coarse meshes.

## Keep every term at one instant

A residual such as Navier–Stokes momentum mixes $\partial_t\mathbf{u}$,
$\mathbf{u}\cdot\nabla\mathbf{u}$, $\nabla p$, and viscous terms. If
$\partial_t\mathbf{u}$ was differenced across a time stencil centred at $t+\Delta$
while the spatial fields are frozen at $t$, the residual is not the PDE residual at
either time.

Optional per-key **derivative metadata** documents source, stencil order, spacing,
`evaluation_time`, and (for time derivatives) `time_stencil`. When keys used by the
same law disagree on `evaluation_time`, or a time derivative lacks a stencil, Moju
emits a **diagnostic warning**. Warnings never change admissibility tiers.

```python
engine.compute_residuals(
    state_pred,
    derivative_metadata={
        "u_t": {
            "source": "finite_difference",
            "evaluation_time": 0.2,
            "time_stencil": [0.19, 0.2, 0.21],
            "stencil_order": 2,
        },
        "u_grad": {"source": "solver", "evaluation_time": 0.2},
        "u_laplacian": {"source": "solver", "evaluation_time": 0.2},
        "p_grad": {"source": "solver", "evaluation_time": 0.2},
    },
)
```

## Time-sampling density

For unsteady problems, choose $\Delta t$ so that the dominant period $T$ is resolved
by many samples (a practical starting point is $\Delta t \lesssim T/20$ for smooth
fields; sharper fronts need more). If you must difference in time yourself, use a
centred stencil of known order and record it in metadata. Prefer exporting
$\partial_t$ from the solver when available.

## Finite-difference order (Path B)

`PathBGridConfig.fd_order` selects explicit central stencils of order **2, 4, or 6**
with matching Fornberg one-sided boundaries. **Moju 2.0 defaults to `fd_order=4`**
(and `fd_order_time=4`). Pass `fd_order="legacy"` or `None` for the 1.x
`jnp.gradient` path. Second derivatives use **dedicated** stencils (`diff2_uniform`).

On **non-uniform** structured axes, the same `fd_order` uses Fornberg weights.

### Periodic axes (per-axis declaration)

`PathBGridConfig.periodic` is **never inferred**. Pass:

- `False` (default) — every spatial axis uses boundary closures;
- `True` — every spatial axis uses wrap-around central FD (or FFT when spectral);
- a per-axis sequence, e.g. `periodic=(False, True)` for Kovasznay (`x` bounded, `y` periodic).

Misdeclaring a periodic physical axis as bounded applies one-sided closures on a
wrap-around field and **inflates boundary residuals**. Spectral fill
(`diff_method="spectral"`) still requires **all** spatial axes declared periodic
(`periodic=True` or a sequence of all `True`); requesting spectral on a
non-periodic axis raises `ValueError`.

On **noisy** field exports, higher FD orders amplify noise: prefer `fd_order=2`,
or supply solver/autodiff derivatives. Avoid order 6 on noisy data.

## Weighted residual norms (2.0 default)

**Default:** area/volume-weighted R_eff (trapezoid from coordinates, or
`point_weights=`) plus a **95th-percentile safeguard**:

`R_score = max(weighted_R_eff, p95(|r|))`

equivalent to `min(score(weighted), score(p95))`. This keeps scores fair on
clustered grids while still detecting localized near-wall violations.

- `safeguard_percentile=` (default **95**) or
  `weighted_safeguard={"method": "percentile", "percentile": …}`.
- `safeguard_error_aware=True` — opt-in residual / derivative-error normalization
  (Phase A4/A5 evidence; **not** the 2.0 default).
- `weighted_norms=False` / `"unweighted"` — 1.x-style unweighted scoring.
- `weighted_norms="report"` — unweighted scoring; attach `rms_weighted` sidecar.
- `weighted_safeguard=False` — weighted RMS only (no percentile).
- `weighted_safeguard={"method": "max"|"percentile"|"worst_region", ...}` — choose
  another candidate (see the Phase A3–A5 impact reports).

**Detection floor:** violations confined to fewer than about **5%** of evaluation
points may not drop the tier under p95 (need larger magnitude or a wider band).
Use `eval_mask` or region scoring (`weighted_safeguard="worst_region"`) for small
critical regions. The resolution sensor does **not** catch narrow-Gaussian
perturbations; it targets grid-scale / checkerboard energy.

**Coarse bounded grids:** boundary rows can approach the p95 tail and lower tiers
on otherwise clean Path B data. Prefer a finer grid, supplied derivatives, or a
boundary-band `eval_mask`. Always declare periodic axes explicitly — misdeclaring
a periodic axis as bounded inflates boundary residuals.

When residual max or a worst-region statistic would drop below High while the
percentile safeguard keeps High, Moju emits a `safeguard_diagnostic` warning
(log + report). Material boundary-stencil spike fractions also warn, and note
coincidence with derivative-error or resolution-sensor flags. **Diagnostics never
change tiers.**

Scattered points: always pass explicit `point_weights`.

## Curvilinear structured grids

For non-rectilinear structured meshes, differentiate in computational coordinates and
transform with mapping metrics via `moju.monitor.curvilinear` (polar / Taylor–Couette
example in the Phase A2 tests). This is opt-in and separate from rectilinear Path B
fill until an auto layout is approved.

## Resolution sensor (diagnostic only)

`resolution_sensor=True` compares two stencil orders on each field and warns when
interior disagreement is large (derivatives may be unreliable). It also warns when
Nyquist-bin spectral energy exceeds the documented grid-scale threshold (`0.05` by
default) — a detection floor for checkerboard / grid-scale artifacts. **Warnings
never change admissibility tiers.**

## Boundary closures and masking

One-sided closures match the selected `fd_order` but remain the accuracy limiter on
wall-normal layers (e.g. exponential profiles). When the physics of interest is
interior, prefer `eval_mask` / `boundary_distance_mask` to exclude a band of width
roughly `fd_order` points at non-periodic edges.

## Derivative-error estimates (opt-in)

Set `estimate_derivative_error=True` (or `PathBGridConfig.estimate_derivative_error=True`)
to compare two FD approximations (different stencil orders) and attach per-key error
estimates. For supplied derivatives, pass a second set as `state_pred_alt` (for example
a coarser spacing). When the estimated derivative contribution is comparable to the
residual (default ratio ≥ 0.5, configurable via `derivative_error_threshold`), the
report flags `residual_may_be_dominated_by_derivative_error` for that law. **Flags do
not change tiers.** Methods: `"stencil_order"` (default), `"dual_spacing"`, or `"both"`.

## Reading the per-term breakdown

Every audit report includes `term_breakdown` for each law that ran:

| Field | Meaning |
|-------|---------|
| `terms.<name>.rms` / `.max` | Magnitude of each PDE term over evaluation points |
| `residual.rms` / `.max` | Residual magnitude |
| `residual_over_largest_term` | Residual RMS divided by the largest term RMS |
| `scale_k` / `scale_source` | Normalizing scale from the law's scale recipe |

If one term is orders of magnitude larger than the others, a moderate residual may
still be a small relative imbalance — or the opposite, if cancellation hides a large
error. Use the breakdown together with `derivative_error` and provenance, not the
tier alone.

## Evaluation masks

Pass `eval_mask` (boolean, True = keep) to exclude points (for example near walls)
from the RMS/max used in that call. Mask statistics appear under `eval_mask` in the
report. Omitting the mask preserves prior scoring behaviour.

## Implied properties need an identifiable constitutive term

Law-linked implied audits (e.g. viscosity from momentum, \(\alpha\) from Fourier)
recover a material coefficient from the same derivatives as the law. That recovery
is well-posed only where the constitutive term is not negligible in the local
balance. For NS/Stokes/Burgers viscosity, Moju keeps points where
\(|\nu\nabla^2\mathbf u|\) is at least `implied_term_fraction` (default 0.1) of the
largest balance term **and** the LSQ fit quality clears `implied_fit_quality_fraction`
(same default); otherwise the point is excluded or the row is **Not identifiable**
(could not evaluate — not a pass). See
[`law_implied_audits.md`](law_implied_audits.md). Poor derivatives make implied rows
noisier than law residuals — prefer supplied or spectral derivatives when those
audits lead the study.

## Example

See [`examples/cookbook_taylor_green_diagnostics.py`](../examples/cookbook_taylor_green_diagnostics.py)
for a Taylor–Green vortex walkthrough (exact derivatives, FD fill, and metadata).
