# Changelog

All notable changes to moju are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Changed

- **Breaking — constitutive/scaling `implied_delta` and `ref_delta`:** Always **nondimensional** (symmetric discrepancy, optional `|ref|` denominator via `implied_delta_ref_key` / `ref_delta_ref_key` or `{output_key}_ref`). Dimensional raw-difference behavior and its configuration **are removed**. See `moju.monitor.closure_registry.apply_closure_discrepancy_normalize`, README “Residual conventions”, `docs/law_implied_audits.md`.

- **Breaking — `Laws.schrodinger_steady`:** Signature is now `(psi_laplacian, V, E, psi, sch_kin_l2)` with `psi_laplacian = L²∇²ψ` and `sch_kin_l2 = 2mL²/ℏ²` from `Groups.schrodinger_kinetic_length_squared`. Residual: `-psi_laplacian + sch_kin_l2 * (V - E) * psi`. Law FD (`law_fd_recipes`) multiplies the grid Laplacian by `L**2` when `L` is in merged state/constants. Replaces `(…, m, h_bar=…)`.

- **Audit admissibility (breaking):** Per-category geometric mean requires **every** per-key admissibility in that category (`laws` / `constitutive` / `scaling` / `data`) to be finite; otherwise the **category score is 0** and overall admissibility becomes **0** (Non-Admissible). Per-key scores remain non-finite where RMS/R_norm are invalid; **`admissibility_level`** still returns **Unknown** for those keys. **RMS** (`_rms_scalar`) and weak-form closure RMS use **NaN-tolerant** reductions (`nanmean` / `nansum`); π-constant scale uses `nanmean` on absolute compare values.

### Added

- **π-constant recipes** for **`Groups.poisson_rhs_pi`** and **`Groups.schrodinger_kinetic_length_squared`** (registry parity with `GROUP_FNS`).

- **Law-linked implied audits:** `moju.monitor.law_implied_diagnostics` prepends `constitutive_audit` / `scaling_audit` rows for selected laws (e.g. `fourier_conduction` → `thermal_diffusivity` vs α from `T_t`/`T_laplacian`; `fick_diffusion` → `fo_mass`; `wave_equation` → `st_wave`; `advection_diffusion` → `pe`; Navier–Stokes / Stokes / Burgers → `re` when `rho`, `μ`, … are in state). Unique log keys via `residual_basename`; optional `include_ref_delta` (default true) gates `ref_delta` when `state_ref` is set. `MonitorConfig.law_implied_audits` (default true) and `ResidualEngine(..., law_implied_audits=...)`. Studio `build_studio_auto_fragment` prepends the same rows and sets `law_implied_audits: true`; dependency planner uses `effective_audit_specs_for_fragment`. Docs: `docs/law_implied_audits.md`. Tests: `tests/test_law_implied_diagnostics.py`.
- **Moju Studio — model-derived derived state:** [`studio_model_derived_registry`](apps/moju_studio/studio_model_derived_registry.py) + `enrich_fragment_from_model_audits()` appends ordered derived-state preprocessing steps on **`MonitorConfig`** when a constitutive audit matches a hand-maintained bridge (first entry: **`thermal_diffusivity`** → `alpha = k/(rho*cp)` for any **`groups`** spec that needs `alpha`, e.g. **`fo`**). Applied on Run and Dependency preview after config merge. Tests: `tests/test_studio_model_derived_registry.py`.
- **Derived state DSL:** `exp` and `pow` nodes in the monitor derived-state expression package for nonlinear material laws (e.g. Arrhenius-style factors). **Moju Studio README** documents **Fourier conduction** (`fo` / `alpha`); Audit **Config** captions point to it.
- **Ordered derived-state preprocessing (Phase B):** ordered `{"output_key", "expr"}` steps on **`MonitorConfig`**, evaluated **before** `groups` / Path B FD / laws using a JSON-safe expression DSL. `ResidualEngine.required_state_keys` and Studio dependency planning union `ref` dependencies and exclude keys produced by those steps. Studio auto fragment + JSON override support the same field. Tests under `tests/` cover derived-state preprocessing.
- **Moju Studio — Audit Data tab:** expander + caption suggesting expected **`T`** / coord shapes (`n_t`, `n_x`, …) from **sidebar Path B — FD grid** (spatial dimension, steady/transient, meshgrid vs separable).
- **Moju Studio — implied dimensionless groups:** `build_studio_auto_fragment` prepends `groups` specs so law arguments that match a registered `Groups.*` name (e.g. `fo`, `re`, `pe`, `da`, `eu`, `ec`, `fo_mass`, `kL` via `wavenumber`) are computed from primitive state/constants before laws. For `advection_diffusion`, `re`, `pr`, and `pe` are injected in dependency order. User-selected scaling groups with the same `output_key` replace the implied row. See `apps/moju_studio/studio_implied_groups.py` and `collect_group_output_keys_from_fragment` in the dependency planner (keys satisfied by `groups` are not flagged as missing NPZ keys).
- **Moju Studio — dependency planner:** `apps/moju_studio/studio_dependency_planner.py` derives required state keys and law-FD prerequisites from the merged config fragment; **Config** tab **Dependency preview** expander; **Run** preflight + downloadable checklist append planner markdown (`preflight_checklist_with_dependency_plan`, `dependency_plan_for_path_b_run`). Built-in **NPZ key aliases** (e.g. `temperature` → `T`) with warning text. Tests: `tests/test_studio_dependency_planner.py`.
- **Test:** `fill_path_b_derivatives` + `fill_law_recipes` fills `T_laplacian` for `fourier_conduction` from `T` and `x` on a 1D meshgrid (`tests/test_law_fd_recipes.py`).
- **`ResidualEngine.clear_log()`** — clear logged steps and reset the step counter between runs.
- **`moju.monitor.visualize_labels`** — `pretty_residual_key`, `pretty_category_name` for publication-style residual and category labels.

### Fixed

- **`Laws.fourier_conduction`:** **`alpha = fo*L²/t`** now **broadcasts** over trailing spatial axes when **`t`** is **`(n_t,)`** and **`T_t`** / **`T_laplacian`** are **`(n_t, n_x, …)`**, fixing JAX **`ValueError: Incompatible shapes for broadcasting`** on **`alpha * T_laplacian`**. Test: `test_broadcasts_alpha_when_t_1d_and_fields_2d`.
- **Path B / law FD:** `_is_steady_leading_time_stack` now treats **square** `(n_t, n_x)` with `n_t == n_x` and matching `t(n_t,)` + `x(n_x,)` as a **time stack**, fixing misclassified 2D spatial Laplacian and `jnp.gradient` failures (`T_laplacian` for `fourier_conduction`). Test: `test_fourier_conduction_t_laplacian_square_nt_equals_nx_time_stack`.

### Changed

- **Moju Studio — Audit preflight:** Run and report warnings use the **dependency planner** (`missing_state_direct`, law/audit FD blocked lists) instead of comparing **`required_state_keys`** to **NPZ keys only** — avoids false “missing” for **`T_t`**, **`T_laplacian`**, **`fo`**, and Constants-only keys. Downloadable checklist **`[x]`** marks can include **NPZ ∪ Constants** when `available_keys` from the planner is passed (`format_planner_preflight_warning`, `preflight_engine_with_available_keys` in `studio_core`).
- **Moju Studio — scaling audits:** Inferred temporal prediction lists no longer treat mesh coordinates as extra prediction axes in ways that produced spurious temporal derivative keys when **`t`** is in NPZ. README: transient **`T`** / **`t`** shape note for law FD fill.
- **Moju Studio — group `output_key`:** User-selected **Groups** (auto builder) now default **`output_key`** to the **lowercase registry name** (`fo`, `re`, `pe`, …), matching **Laws.*** arguments and implied groups, so the dependency planner no longer asks for **`Fo`** when the law expects **`fo`**. Exceptions remain for curated keys (**`Nu`**, **`Pe_m`**, **`k_wave`**, …). README Config bullet updated.
- **Moju Studio / dependency planner:** When **`t`** is among **missing state keys**, the dependency plan markdown adds a note that **`t`** for **`Groups.fo`** / Fourier is usually the **mesh time coordinate** in **`state_pred`** (Path B **`key_t`**), not Constants. README + Audit captions updated similarly.
- **`ResidualEngine.compute_residuals`:** If a governing-law input is still missing after Path B FD + law-FD fill, **`KeyError` now appends** the recent `fill_path_b_derivatives` warning lines (e.g. missing `x`, Laplacian failures). **`law_fd_recipes`:** `T_laplacian`-style keys present in **constants** only as **`null` / placeholder strings** no longer block FD fill (constants-only real arrays still skip overwrite).
- **`PathBGridConfig` / FD:** `spatial_dimension="auto"` now treats steady **row-shaped** fields `(1, N)` as **1D** (same as `(N, 1)`), avoiding false 2D Laplacian paths that require `y`. Studio sidebar adds a short Laplacian troubleshooting caption and a **Path B FD messages** expander after Run (full `inferred` log + warning when Laplacian/law_fd lines appear).
- **Path B merge order (breaking for duplicate keys):** `state_pred` / group-built state now **wins** over **`constants`** when the same key exists in both (`compute_residuals` merged dict, `_build_state`, and `_merged` in `path_b_derivatives`). Previously constants could overwrite FD-filled tensors (e.g. a placeholder `T_laplacian: null` in Studio **constants** wiping a filled Laplacian), causing `KeyError` for laws.
- **Moju Studio — Audit:** **Law FD prerequisites** expander for selected laws (`studio_law_fd_hints`); **preflight `st.warning`** before `compute_residuals` on Path B when required keys are missing from `state_pred`; Expert JSON caption for FD primitives/coords.
- **`ResidualEngine` / laws:** Missing merged state keys for **governing laws** now raise **`KeyError`** with an extra sentence for derived-looking keys (`*_laplacian`, `*_grad`, `*_t`, `u_grad`) pointing to Path B FD + `LAW_FD_RECIPES`.
- **Moju Studio — Audit page (breaking UX):** Replaced the per-field simple builder with **allowlisted** multiselects (**Laws** = FD-supported subset, **Models** → constitutive audits, **Groups** → `groups` + scaling audits), **identity** `state_map` (NPZ/constants keys must match API argument names), optional JSON override, and **Expert** mode for full `MonitorConfig` JSON. **Run** tab defaults **`auto_path_b_derivatives`** and **`fill_law_fd`** to **on**. See `apps/moju_studio/studio_auto_config.py` and `tests/test_studio_auto_config.py`.
- **`visualize`:** Default **figure title** is mode-specific (training vs state-prediction test audit); larger title font in Plotly. Optional **`figure_title`** still overrides. Plotly titles may include an HTML **subtitle** with final overall admissibility and HIGH/MODERATE/LOW status.
- **`visualize` (breaking):** Replaced the old multi-panel dashboard with **training** vs **test** modes. **Training (multi-step):** top row = overall admissibility vs step (with last-point marker) and **horizontal category admissibility bars** (laws / constitutive / scaling); second row = three **Normalized Governing / Constitutive / Scaling** line panels (**`r_norm_scale="log"`** default: `log10(R_norm + ε)`; use **`"linear"`** for raw R_norm; `data/` keys omitted); optional full-width **spatial** row when **`spatial_law_panel`** is set. **Test / single-step training:** horizontal R_norm bars, same **category admissibility** bar panel, optional spatial. Matplotlib and Plotly mirror the same layouts.

### Removed

- **Spatial/temporal derivative-consistency closures** for constitutive/scaling audits and related legacy `AuditSpec` fields (`predicted_spatial`, `predicted_temporal`, weak-form closure options, and associated configuration removed from `AuditSpec.from_dict` validation). Path B FD remains for **registered law inputs** only (`fill_law_fd` + `law_fd_recipes`).
- **Standalone diagnostics Plotly API** (`moju.monitor.diagnostics_plotly` and exports): **`plot_moju_diagnostics`**, **`plot_diagnostics`**, **`diagnostics_data_from_log`**, **`list_diagnostic_plot_keys`**. Use **`visualize(..., backend="plotly"|"matplotlib")`** only.

## [0.5.0] - 2026-03-21

### Added

- **Moju Studio (Streamlit):** optional app under `apps/moju_studio/` — upload `state_pred` / `state_ref` as `.npz`, **form builder** for laws/groups/constitutive/scaling audits (or full JSON), Path B / Path A shim, **PathBGridConfig** for FD, `audit`/`visualize` **r_ref** and weights, session log append + Plotly key subset redraw, **pred−ref** spatial view, preflight checklist download, optional **PDF ZIP** (`moju[report]`). **UX:** sidebar navigation + `page_link`, run **form**, **`st.status`** pipeline, **`st.toast`**, **`@st.fragment`** spatial/redraw, **`st.dialog`** clear-log confirm, RMS **`column_config`**, Plotly **`on_select`** where supported; **Quick start** / **Help** pages; repo-root **`.streamlit/config.toml`**. Requires **streamlit >= 1.33**. Install: `pip install "moju[studio,viz]"` (editable from repo). See `apps/moju_studio/README.md`.
- **Monitor `visualize` dashboard:** multi-panel matplotlib figure from engine logs alone—per-key RMS/admissibility, overall + category trajectories (including **data/** keys), heatmap, top R_norm keys, closure-type bars, omitted/inferred counts, twin RMS/R_norm, category radar, and per-category worst-key traces. Optional `r_ref` and `max_legend_keys`. **Audit** category scores now include **`data`** (geometric mean over `data/...` keys) so overall admissibility matches PDF groupings.
- **Monitor `visualize(..., backend="plotly")`:** interactive Plotly dashboard with the same panels (zoom/pan/hover). Install with `pip install plotly` or `pip install "moju[viz]"` (includes matplotlib + plotly). If plotly is missing, returns `None` like the matplotlib path without matplotlib.
- **Path B structured-grid FD:** `PathBGridConfig`, `fill_path_b_derivatives` in `moju.monitor` for finite differences on rectilinear grids. Optional `compute_residuals(..., auto_path_b_derivatives=True|PathBGridConfig)` with `fill_law_fd=True` fills **registered** missing `Laws.*` inputs. Tests: `tests/test_path_b_derivatives.py`, `tests/test_law_fd_recipes.py`.
- **Path B law FD (optional):** `fill_path_b_derivatives(..., fill_law_recipes=True, laws_spec=...)` and `compute_residuals(..., auto_path_b_derivatives=..., fill_law_fd=True)` fill **registered** missing `Laws.*` inputs (e.g. `phi_laplacian`, `u_grad`, `T_t`) from primitives on the same structured grid (`moju.monitor.law_fd_recipes`, `list_law_fd_supported_laws()`). Rectilinear 2D/3D meshgrid coordinates are detected for JAX-compatible spacing. Tests: `tests/test_law_fd_recipes.py`. Example: `examples/cookbook_path_b_fd_law_laplace.py`.
- **Constitutive `implied_delta`:** optional `AuditSpec.implied_value_key` or `implied_fn(merged_state, constants)`; residual `F(pred args) − implied`, omitted when implied is missing (same as other closures). `ref_delta` no longer requires `predicted_spatial`/`predicted_temporal`. Helper `audit_spec_to_engine_dict` for advanced specs. Examples: `examples/cookbook_constitutive_implied_ideal_gas_rho.py`, `examples/cookbook_constitutive_implied_power_law_fn.py`; tests in `tests/test_auditor.py`, `tests/test_examples_implied_cookbooks.py`.
- **π-constant scaling closure (Path A):** optional `AuditSpec` fields `invariance_pi_constant`, `invariance_compare_keys`, `invariance_scale_c` for `scaling_audit`. Second forward with built-in constant scaling so the audited group stays fixed; residual logged as `scaling/<name>/pi_constant` with R_norm scale `ε + mean(|scaled compare keys|)`. Built-in recipes for **every** registered `Groups.*` name (`moju.monitor.pi_constant_recipes`, `list_pi_constant_group_names()`); rules may use integer powers of `c` (e.g. `We`, `Gr`, `Da`).
- **Admissibility docs:** `audit()` docstring and README clarify three reporting levels (per key, per-category geometric means, overall geometric mean). PDF category section notes category scores are geometric means.
- **Examples:** `examples/cookbook_pi_constant_reynolds.py` and `examples/cookbook_pi_constant_prandtl.py` (Path A π-constant audit end-to-end); tests in `tests/test_examples_pi_constant_cookbooks.py`.
- **Models / examples:** `Models.smagorinsky_nu_t` (LES eddy viscosity template); cookbooks `cookbook_turbulence_law_of_wall.py`, `cookbook_turbulence_colebrook.py`, `cookbook_constitutive_smagorinsky.py` with `tests/test_examples_turbulence_cookbooks.py`.
- **RANS eddy viscosity (algebraic νₜ):** `Models.k_epsilon_nu_t` and `Models.k_omega_nu_t` with dissipation floors for stable AD; cookbooks `examples/cookbook_constitutive_k_epsilon.py`, `examples/cookbook_constitutive_k_omega.py` (Path B); tests in `tests/test_models.py` and `tests/test_examples_turbulence_cookbooks.py`.
- **Studio data formats:** optional extra `moju[studio-science]` for HDF5 / NetCDF / `.npy` state uploads in Moju Studio (`h5py`, `xarray`, `netCDF4`).
- **Studio π-constant gating:** π-constant scaling audits require a recomputing `state_builder` and non-empty `invariance_compare_keys`; default NPZ Path A shim is rejected when π is enabled (clear errors + tests + docs).

### Fixed

- **Path B / law FD + JAX:** detect approximately uniform 1D grid spacing and use scalar `jnp.gradient(..., h)` so float32 `linspace` and stricter JAX versions do not raise "Non-constant spacing not implemented" or skip law residuals on Python 3.9 CI. Tests use the same scalar-spacing references where `jnp.gradient(f, coord)` is unsupported.
- **Python 3.9:** avoid PEP 604 `X | Y` type hints in Studio modules so sources parse on 3.9.

### Changed

- **CI / docs:** minimal-install and `studio-science` import smoke jobs; README optional-extras table and import troubleshooting; root + Studio README updates for science extras.

## [0.4.3] - 2026-03-20

### Changed

- **PyPI metadata**: update package description wording to clarify moju as a physics-informed ML framework for enforcing governing equations and auditing physical consistency.

## [0.4.2] - 2026-03-20

### Added

- **Docs**: show how to discover valid `AuditSpec.name` ids for `Models.*` and `Groups.*` using the registry listing helpers (`list_constitutive_models()`, `list_scaling_closure_ids()`).

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

- **ResidualEngine** no longer accepts `models` or `key_ref`. Group/model “distance to reference scalar” residuals are removed. Use **constitutive_audit** / **scaling_audit** tied to `Models.*` / `Groups.*` with `ref_delta` and related closure options. **build_loss** is unchanged (laws only).

### Added

- **Model/Group audit registry**: audits are tied to `Models.*` and `Groups.*` functions (`ref_delta` and related). Helpers: `list_constitutive_models()`, `list_scaling_closure_ids()`.
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
