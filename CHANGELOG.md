# Changelog

All notable changes to moju are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.4.1] - 2026-03-19

### Changed

- **README**: High-conversion structure (hero, why moju, big idea, 5-minute example, what you get, use cases, core concepts, philosophy); correct moju API throughout; smoke test for 5-minute example.
- **Terminology**: "Physical Admissibility" replaced with "Physics Admissibility" in report title, docstrings, sample script, README, and CHANGELOG; sample PDF output renamed to `sample_physics_admissibility_report.pdf`.

---

## [0.4.0] - 2026-03-15

### Added

- **Scale-based R_norm**: R_norm(k) = RMS(r_k)/scale_k; scale is state-derived by default (from merged state/specs per key). Optional `r_ref` in `audit(log, r_ref=...)` overrides scale for given keys. Each log entry stores `entry["scale"]`; fallback to first-entry RMS when `scale` is missing (backward compatibility).
- **state_ref loaders**: `from_vtk`, `from_vtu`, `from_openfoam`, `from_hdf5` in `moju.monitor.state_ref`; optional extras `ref_vtk`, `ref_foam`, `ref_hdf5`, `ref_all`. Examples: `monitor_state_ref_from_vtu_demo.py`, `monitor_state_ref_from_openfoam_demo.py`, `monitor_state_ref_from_hdf5_demo.py`. Tests: `test_state_ref_meshio.py`, `test_state_ref_hdf5.py`. CI jobs for ref_vtk, ref_hdf5, ref_foam.

### Changed

- **audit()**: Uses scale_k from `r_ref` or `entry["scale"]` or fallback; R_norm(k) = RMS(r_k)/scale_k; admissibility unchanged.
- **ResidualEngine.compute_residuals**: Computes state-derived scale per key and stores in log entry.
- README and docs overview document R_norm = RMS/scale, state-derived scale, and r_ref override.
- `scripts/generate_sample_audit_pdf.py`: comment that r_norm is RMS/scale.

---

## [0.3.0] - 2026-03-09

### Breaking

- **ResidualEngine** no longer accepts `models` or `key_ref`. Group/model “distance to reference scalar” residuals are removed. Use **constitutive_audit** / **scaling_audit** tied to `Models.*` / `Groups.*` with `ref_delta`, `chain_dx`, `chain_dt` (and optional **constitutive_custom** / **scaling_custom**). **build_loss** is unchanged (laws only).

### Added

- **Model/Group audit registry**: audits are tied to `Models.*` and `Groups.*` functions (ref_delta, chain_dx, chain_dt). Helpers: `list_constitutive_models()`, `list_scaling_closure_ids()`.
- PDF/report categories for **constitutive** and **scaling**; disclaimer clarifies metrics are heuristic indicators, not certification.

### Changed

- **examples/slab_cooling_demo.py** updated for the new API (no engine `models`, no `key_ref`).
- README, overview (Mermaid), and landing copy describe closure-based audit.

---

## [0.2.2] - 2025-03-16

### Added

- **Custom physics hooks**: `ResidualEngine` now accepts an optional `"fn"` in law/group/model specs so users can plug in their own JAX-differentiable Models, Groups, and Laws. Specs like `{"name": "my_law", "state_map": {...}, "fn": my_residual}` use the custom callable instead of the built-in `Laws.name`, with kwargs built from `state_map`.
- **Physics Admissibility Report**: new `moju.monitor.report.write_audit_pdf` helper and `audit(..., export_dir=...)` integration to generate a Physics Admissibility Report PDF (plus optional `residuals.json` and a zipped session folder). Sample report script lives in `scripts/generate_sample_audit_pdf.py`, with output in `examples/sample_physics_admissibility_report.pdf`.

### Changed

- **Audit report wording and layout**: the report now uses the title “Physics Admissibility Report”, writes “Moju is developed by Ifimo Lab at Ifimo Analytics” in the footer (left), and the disclaimer “This report is a heuristic and not a certification.” in the footer (right).
- **Docs and GitHub Pages**: clarified terminology (Physics Admissibility), highlighted **moju.monitor** in the landing page and overview, and documented how to use custom Models, Groups, and Laws via the optional `fn` parameter.

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
  - **ResidualEngine** — Single entry point: laws, groups, models, constants; `compute_residuals` with optional `state_ref`. *(Superseded in 0.3.0: no engine `models` or `key_ref`; use constitutive/scaling closure audits.)*
  - **build_loss** — Physics-only loss (cascaded over laws); user adds data loss in JAX or PyTorch.
  - **audit** — Computes R_norm, admissibility score, and overall admissibility score from the log; writes metrics back into the same log.
  - **visualize** — Plots RMS and metrics per key (optional matplotlib).
- Documentation: high-level architecture diagram (Mermaid) in docs; Training and monitoring (ResidualEngine) section in overview.

### Changed

- None for released APIs (0.1.x on PyPI is unchanged).

---

## [0.1.3] - Previous release

PiRatio: Groups, Models, Laws, Operators (dimensionless scaling, physical models, conservation-law residuals, differential operators). JAX-native, JIT-compiled, differentiable.
