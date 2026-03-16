# Changelog

All notable changes to moju are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.2.2] - 2026-03-16

### Added

- **Custom physics hooks**: `ResidualEngine` now accepts an optional `"fn"` in law/group/model specs so users can plug in their own JAX-differentiable Models, Groups, and Laws. Specs like `{"name": "my_law", "state_map": {...}, "fn": my_residual}` use the custom callable instead of the built-in `Laws.name`, with kwargs built from `state_map`.
- **Physical Admissibility Report**: new `moju.monitor.report.write_audit_pdf` helper and `audit(..., export_dir=...)` integration to generate a Physical Admissibility Report PDF (plus optional `residuals.json` and a zipped session folder). Sample report script lives in `scripts/generate_sample_audit_pdf.py`, with output in `examples/sample_physical_admissibility_report.pdf`.

### Changed

- **Audit report wording and layout**: the report now uses the title “Physical Admissibility Report”, writes “Moju is developed by Ifimo Lab at Ifimo Analytics” in the footer (left), and the disclaimer “This report is a heuristic and not a certification.” in the footer (right).
- **Docs and GitHub Pages**: clarified terminology (Physical vs Physics Admissibility), highlighted **moju.monitor** in the landing page and overview, and documented how to use custom Models, Groups, and Laws via the optional `fn` parameter.

---

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
