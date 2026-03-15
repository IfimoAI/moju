# Changelog

All notable changes to moju are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.2.1] - 2025-03-09

### Changed

- Single README: removed high-level architecture section from README (diagram remains in docs only) so PyPI and repo display cleanly.
- Package info: "What's included" now explicitly lists **moju.piratio** and **moju.monitor** in README and docs.
- GitHub Pages docs: light theme and readable colors (style.css, index.html); Mermaid diagram theme set to default in overview.

---

## [0.2.0] - 2025-03-08

### Added

- **`moju.monitor`** — New package for residuals, physics loss, and monitoring:
  - **ResidualEngine** — Single entry point: configure laws, groups, models, and constants; call `compute_residuals(state_pred, state_ref=None, key_ref=None)` to get a residual dict and log per-key RMS.
  - **build_loss** — Physics-only loss (cascaded over laws); user adds data loss in JAX or PyTorch.
  - **audit** — Computes R_norm, admissibility score, and overall admissibility score from the log; writes metrics back into the same log.
  - **visualize** — Plots RMS and metrics per key (optional matplotlib).
- Documentation: high-level architecture diagram (Mermaid) in docs; Training and monitoring (ResidualEngine) section in overview.

### Changed

- None for released APIs (0.1.x on PyPI is unchanged).

---

## [0.1.3] - Previous release

PiRatio: Groups, Models, Laws, Operators (dimensionless scaling, physical models, conservation-law residuals, differential operators). JAX-native, JIT-compiled, differentiable.
