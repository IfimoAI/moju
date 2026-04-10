# Moju Studio

Interactive **Streamlit** app to upload `state_pred` (and optional `state_ref`) as **`.npz`**, **`.npy`**, **HDF5**, or **NetCDF** (see install notes below), pick **Laws / Models / Groups** from Studio allowlists (or **Expert** JSON), run `ResidualEngine.compute_residuals`, view `audit()` summaries, Plotly `visualize`, and explore arrays in space/time.

## Install

From the repository root:

```bash
pip install -e ".[studio,viz]"
```

(`viz` adds Plotly for the multi-panel dashboard; `studio` adds Streamlit **>= 1.33** + Plotly.)

For **HDF5** (`.h5` / `.hdf5`) and **NetCDF** (`.nc` / `.nc4`) uploads on the Data tab, also install:

```bash
pip install -e ".[studio-science]"
```

(`studio-science` adds `h5py`, `xarray`, and `netCDF4`. **NumPy `.npy`** and **`.npz`** work with the base `studio` extra only.)

**Import issues:** If `h5py` or `xarray` fails to import (missing extra vs. broken NumPy/wheel ABI), see **Troubleshooting import errors** in the repository root [`README.md`](../../README.md). Typical fixes: install `moju[studio-science]`, or use a fresh venv and reinstall `numpy` plus the failing package.

## Run

From the **repository root** (so `.streamlit/config.toml` is picked up):

```bash
streamlit run apps/moju_studio/Home.py
```

Use the **Audit** page to:

- **Config (default)** — set **Constants JSON**; multiselect **Laws** (FD-supported subset), **Models** (→ constitutive audits), and **Groups** (→ `groups` + scaling audits). Uses **identity** `state_map`: NPZ / constants keys must match `Laws.*` / `Models.*` / `Groups.*` argument names (e.g. `k_solid` for `bi`). Dimensionless group outputs use the same names as law arguments (**`fo`**, **`re`**, **`pe`**, …), not title-cased **`Fo`/`Re`**, so Studio does not ask for a duplicate key. Optional small JSON override. On **Run**, **Compute state derivatives (finite difference)** and **Compute law derivatives (finite difference)** default **on** (API: `auto_path_b_derivatives`, `fill_law_fd`). Expand **Dependency preview** to see required state keys and law-FD targets (assuming Run FD defaults); a small **built-in alias** map (e.g. `temperature` → `T`) is honored with a warning to prefer canonical names.
- **Expert mode** — checkbox to edit the full **MonitorConfig** JSON instead of the auto builder.
- **Path B** — pass uploaded tensors directly; optional **PathBGridConfig** when customizing the FD grid.
- **Path A (shim)** — same upload, but the engine uses a `state_builder` that returns your NPZ tensors (constants are **not** applied to those tensors in the shim). **π-constant** scale-invariance residuals are **disabled** for this default shim because `state_pred` does not recompute under scaled constants. To run π-constant in Studio, set `st.session_state["studio_recomputing_state_builder"]` to a callable `state_builder(model, params, collocation, constants)` that actually depends on `constants` (or use the Python API). Each π scaling audit must set non-empty **`invariance_compare_keys`**. Adjust **`invariance_scale_c`** (`c > 1`) in the Config tab when π is enabled.
- **Dashboard** — card-based Plotly charts (law / category bars; law vs constitutive spatial **R_norm** heatmaps); optional `r_ref` / weights / `max_legend_keys` / π slider for the **next** run. Full **single-figure** `visualize(..., mode="training")` uses **two** KPI cards (Governing / Constitutive); **`mode="eval"`** (or legacy **`mode="test"`**) can show **Scaling** and **Data** when those categories exist (see **`docs/monitor_training_vs_eval.md`** and **`run_mode`** on `compute_residuals`). Sidebar **heatmap colorscale** and **spatial axis** (`x` / `y` / `z`) apply to spatial heatmaps; session log append (sidebar) for multi-step runs; redraw expander with a subset of keys. The Python API documents `visualize(..., mode="training"|"eval")`; **`test`** remains a silent alias for **`eval`**. The Audit page caption uses **`format_admissibility_status_label`**, which matches **`admissibility_level`** (four bands on the score in `[0, 1]`; see root **README**).
- **Export** — JSON reports, optional **PDF ZIP** if `moju[report]` is installed.

## Model-derived derived-state steps (auto)

Some **`Groups.*` inputs** (e.g. **`alpha`** for **`Groups.fo`**) can be filled automatically when you add the matching **constitutive audit** (`Models.*`): Studio runs [`enrich_fragment_from_model_audits`](apps/moju_studio/studio_model_derived_registry.py) after merging constants and **appends** JSON-safe preprocessing steps to the ordered derived-state field on **`MonitorConfig`** before `groups` / FD / laws.

**Current registry** (extend in [`studio_model_derived_registry.py`](apps/moju_studio/studio_model_derived_registry.py)):

| Model audit `name`   | Appends (same closed form as `Models.*`)   | Needs group input key = audit `output_key` |
|---------------------|---------------------------------------------|---------------------------------------------|
| `thermal_diffusivity` | `alpha = k / (rho * cp)`                  | `alpha` (for `fo`, etc.)                    |

The audit’s **`state_map`** supplies the state keys for `k`, `rho`, `cp` (identity map by default). If you already define the `output_key` in **Expert derived-state JSON** or NPZ, nothing is duplicated. Other **`fo`** inputs (**`t`**, **`L`**) are unchanged — see **Time `t`** below.

**Time `t` for `Groups.fo`:** use **`t` in `state_pred`** as the elapsed-time / mesh time coordinate (same idea as **`x`** for space). Align the NPZ key with **sidebar → Path B — FD grid** **`key_t`** (default `t`). For transient runs, avoid relying on **Constants JSON** for `t` unless you intentionally want a **scalar** that broadcasts. Aliases **`time`**, **`coords_t`** map to canonical **`t`** in the dependency planner.

## Fourier conduction: `fo`, `alpha`, and `rho(T)`, `k(T)`

`Laws.fourier_conduction` takes **`T_t`**, **`T_laplacian`**, **`fo`**, **`t`**, **`L`** — not `rho` or `k` directly. Studio injects **`Groups.fo`**, which builds **`fo`** from **`alpha`**, **`t`**, **`L`**.

**Easiest:** add **`thermal_diffusivity`** under Models — Studio auto-appends **`alpha`** from **`k`**, **`rho`**, **`cp`** when `fo` needs `alpha`, and the audit still checks consistency. Alternatively put **`alpha`** in the NPZ, or use manual derived-state steps in **Expert** **MonitorConfig** JSON (same top-level key as in the example block below).

**Time `t` (not a “constant” by default):** `fo` and the law need a **`t`** field — typically the **mesh time coordinate** in **`state_pred`**, matching **`key_t`** on the Path B FD grid (default `t`). Use **Constants JSON** for `t` only if you want a single elapsed time broadcast everywhere. See **Model-derived** section above for aliases.

**Transient + FD:** for **`T_t`** and **`T_laplacian`** auto-fill, **`T`**’s **leading dimension must match** the length of **`t`** (e.g. **`T` shape `(n_t, n_x, …)`** with **`t` shape `(n_t,)`**). **`Laws.fourier_conduction`** broadcasts **`alpha = fo·L²/t`** over spatial axes so **`t(n_t,)`** with **`T(n_t, n_x, …)`** is valid. If you see `t length must match K leading dimension` or `unsteady laplacian: t must match leading dim`, set **sidebar Path B — FD grid** to **Transient** and reshape **`T`** / **`t`** accordingly (or use **Steady** only when there is no time axis in the data).

**Law-linked Fourier vs `thermal_diffusivity`:** the prepended row compares **α implied from the heat equation** to **`Models.thermal_diffusivity(k, ρ, cₚ)`** via **`implied_delta`** when those fields are available. Selecting **`thermal_diffusivity`** under Models supplies **`alpha`** from **`k`**, **`rho`**, **`cp`** and avoids duplicate constitutive rows.

**Temperature-dependent properties:** use derived-state preprocessing in Expert **MonitorConfig** JSON or in the **Optional JSON override** (auto mode). Steps run **before** `groups`, so `alpha` is available when `Groups.fo` runs. Put coefficients in **Constants JSON** and reference them with `{"op": "ref", "key": "cp"}`.

Linear example (`rho = rho0 + c_rho * (T - T_ref)`, `k = k0 + c_k * (T - T_ref)`, `alpha = k/(rho*cp)` — adjust names to match your NPZ):

```json
"derived_state_chain": [
  {"output_key": "rho", "expr": {"op": "add", "a": {"op": "ref", "key": "rho0"}, "b": {"op": "mul", "a": {"op": "ref", "key": "c_rho"}, "b": {"op": "sub", "a": {"op": "ref", "key": "T"}, "b": {"op": "ref", "key": "T_ref"}}}}},
  {"output_key": "k", "expr": {"op": "add", "a": {"op": "ref", "key": "k0"}, "b": {"op": "mul", "a": {"op": "ref", "key": "c_k"}, "b": {"op": "sub", "a": {"op": "ref", "key": "T"}, "b": {"op": "ref", "key": "T_ref"}}}}},
  {"output_key": "alpha", "expr": {"op": "div", "a": {"op": "ref", "key": "k"}, "b": {"op": "mul", "a": {"op": "ref", "key": "rho"}, "b": {"op": "ref", "key": "cp"}}}}
]
```

The DSL also supports **`exp`** and **`pow`** in `moju.monitor` for nonlinear laws such as Arrhenius-style factors.

## Notes

- `MonitorConfig` JSON must not rely on `implied_fn` (not serializable); use the Python API for that.
- π-constant audits require non-empty `invariance_compare_keys` and (outside the NPZ shim) a `state_builder` that recomputes from scaled `constants`; see Path A notes above.
- JAX on shared hosting (e.g. Streamlit Cloud) may need extra setup; local or VM is recommended.
