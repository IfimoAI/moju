# Migrating from Moju 1.x to 2.0

## What changed and why

Moju 2.0 improves Path B finite-difference derivatives and makes residual norms
**area-fair** on clustered grids, without changing admissibility *thresholds*.

| Topic | 1.x | 2.0 default |
|-------|-----|-------------|
| Path B `fd_order` | `None` → `jnp.gradient` | **4** (Fornberg boundaries; dedicated ∂²) |
| `fd_order_time` | `None` | **4** |
| Residual RMS / scores | Unweighted mean over points | **Weighted** trapezoid norms + **95th-percentile safeguard** |
| Curvilinear grids | Not supported | Explicit `moju.monitor.curvilinear` API |
| Resolution sensor | — | Opt-in diagnostic (never changes tiers); grid-scale Nyquist warning |
| Periodic axes | Not declared for FD | Explicit `periodic=bool|(…)` per axis (never inferred); spectral still needs all-True |

Expected score shifts (from the Phase A–A3 impact reports):

- Smooth periodic Path B (Taylor–Green): residuals drop sharply under fd4; tier usually stays High.
- Wall-bounded Path B (Kovasznay): often **Moderate → High** under fd4 (truncation reduced).
- Clustered grids: unweighted scores were biased by point density; weighted+p95 restores area fairness while still flagging localized near-wall violations.
- Noisy exports: higher FD orders amplify noise — prefer supplied derivatives or `fd_order=2`; avoid 6.

## Reproduce 1.x behaviour

Any one of:

```python
from moju.monitor.path_b_derivatives import PathBGridConfig

grid = PathBGridConfig(fd_order="legacy", fd_order_time="legacy")  # or None
engine.compute_residuals(
    state,
    auto_path_b_derivatives=grid,
    fill_law_fd=True,
    weighted_norms=False,  # unweighted 1.x scoring
)
```

Or pin the package: `moju<2`.

## Weighted norms and the safeguard

Default scoring residual per key is:

```text
R_score = max(weighted_R_eff, percentile_95(|r|))
```

which is equivalent to `min(score(weighted_R_eff), score(p95))`.  
Opt out with `weighted_norms=False` / `"unweighted"`.  
Configure via `safeguard_percentile=` (default 95) or
`weighted_safeguard={"method": "max"|"percentile"|"worst_region", ...}`.
Derivative-error-aware normalization is **opt-in** (`safeguard_error_aware=True`;
see Phase A4/A5 reports) — not the 2.0 default.

**Detection floor (p95):** localized violations on fewer than about **5%** of
points may not lower the tier; use `eval_mask` or region scoring for small
critical regions. Narrow Gaussians are not flagged by the resolution sensor.

**Known limitations on coarse bounded Path B grids:** wall FD rows can approach
the p95 tail and lower tiers on clean data. Prefer finer grids, supplied
derivatives, or a boundary-band `eval_mask`. Always declare `periodic=` per axis;
misdeclaring a periodic axis as bounded inflates boundary residuals.

## Paper reproducibility

The published slab-cooling Path B notebook used **moju 1.x** with **supplied**
PINN derivatives (`T_xx` in state). That path does not use `fd_order`; scores
remain High under 2.0 when derivatives are still supplied. To match 1.x FD refill
behaviour, use `fd_order="legacy"` and `weighted_norms=False`.

See [`phase_a_fd_impact_report.md`](phase_a_fd_impact_report.md) for side-by-side numbers.
