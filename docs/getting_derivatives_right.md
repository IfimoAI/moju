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
with matching one-sided boundaries on **uniform** grids. The default `fd_order=None`
keeps the legacy `jnp.gradient` path (unchanged from 1.3). Temporal fill uses
`fd_order_time` the same way.

Non-uniform spacing is only supported on the legacy path; set `fd_order=None` or
supply derivatives. Spectral spatial fill still requires `periodic=True` and uniform
spacing; Moju warns when those guards fail.

## Derivative-error estimates (opt-in)

Set `estimate_derivative_error=True` (or `PathBGridConfig.estimate_derivative_error=True`)
to compare two FD approximations (different stencil orders) and attach per-key error
estimates. For supplied derivatives, pass a second set as `state_pred_alt` (for example
a coarser spacing). When the estimated derivative contribution is comparable to the
residual (default ratio ≥ 0.5, configurable via `derivative_error_threshold`), the
report flags `residual_may_be_dominated_by_derivative_error` for that law. **Flags do
not change tiers.**

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

## Example

See [`examples/cookbook_taylor_green_diagnostics.py`](../examples/cookbook_taylor_green_diagnostics.py)
for a Taylor–Green vortex walkthrough (exact derivatives, FD fill, and metadata).
