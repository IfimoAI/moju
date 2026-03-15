# Changelog

All notable changes to moju are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.2.0] - Unreleased

### Added

- **`moju.monitor`** — New package for residuals, physics loss, and monitoring:
  - **ResidualEngine** — Single entry point: configure laws, groups, models, and constants; call `compute_residuals(state_pred, state_ref=None, key_ref=None)` to get a residual dict and log per-key RMS.
  - **build_loss** — Physics-only loss (cascaded over laws); user adds data loss in JAX or PyTorch.
  - **audit** — Computes R_norm, S, and overall physics score from the log; writes metrics back into the same log.
  - **visualize** — Plots RMS and metrics per key (optional matplotlib).
- Documentation: high-level architecture diagram (Mermaid) in README and docs; Training and monitoring (ResidualEngine) section in overview.

### Changed

- None for released APIs (0.1.x on PyPI is unchanged).

---

## [0.1.3] - Previous release

PiRatio: Groups, Models, Laws, Operators (dimensionless scaling, physical models, conservation-law residuals, differential operators). JAX-native, JIT-compiled, differentiable.
