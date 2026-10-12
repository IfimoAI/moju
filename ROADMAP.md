# Moju roadmap

## 2.0.0 (`feature/2.0.0-derivative-layer`)

Derivative layer: selectable FD orders, Fornberg non-uniform weights, dedicated
second-derivative stencils, per-axis periodic FD, weighted residual norms + p95
safeguard, opt-in error-aware safeguard, boundary closures, curvilinear API,
resolution sensor (incl. grid-scale Nyquist warning).

**Approved defaults:** `fd_order=4`, `boundary_closure="matching"`,
`safeguard_percentile=95`, `safeguard_error_aware=False` (opt-in), periodic axes
declared explicitly, resolution-sensor grid-scale threshold `0.05` (diagnostic only).

See [`docs/phase_a_fd_impact_report.md`](docs/phase_a_fd_impact_report.md),
[`docs/phase_a5_periodic_safeguard.md`](docs/phase_a5_periodic_safeguard.md), and
[`docs/MIGRATION_1x_to_2.0.md`](docs/MIGRATION_1x_to_2.0.md).

## 2.1 (deferred — do not implement in 2.0)

- **Region-aware thresholds for boundary bands** — so wall FD truncation on coarse
  bounded grids does not dominate global percentile safeguards.
- **Revisit the error-aware safeguard and a higher default percentile** using the
  Phase A4 / A5 evidence (order-difference ε unlocks p97.5 on some suites when
  periodic axes are declared correctly; still blocked on physical walls).
- **Element-wise polynomial differentiation** for data on element nodes
  (e.g. Gauss–Lobatto points), with averaging at shared nodes.
- **Opt-in weak-form residuals** against smooth test functions, with their own
  scale recipe.
