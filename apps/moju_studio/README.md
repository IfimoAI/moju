# Moju Studio

Interactive **Streamlit** app to upload `state_pred` (and optional `state_ref`) as **`.npz`**, **`.npy`**, **HDF5**, or **NetCDF** (see install notes below), edit a `MonitorConfig` fragment as JSON, run `ResidualEngine.compute_residuals`, view `audit()` summaries, Plotly `visualize`, and explore arrays in space/time.

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

- **Simple (form) builder** — add laws, groups, constitutive/scaling audits with dropdowns mapped to NPZ keys; optional JSON override per section. Or switch off to edit the full **MonitorConfig** JSON.
- **Path B** — pass uploaded `.npz` tensors directly; optional **PathBGridConfig** when using FD fills.
- **Path A (shim)** — same upload, but the engine uses a `state_builder` that returns your NPZ tensors (constants are **not** applied to those tensors in the shim). **π-constant** scale-invariance residuals are **disabled** for this default shim because `state_pred` does not recompute under scaled constants. To run π-constant in Studio, set `st.session_state["studio_recomputing_state_builder"]` to a callable `state_builder(model, params, collocation, constants)` that actually depends on `constants` (or use the Python API). Each π scaling audit must set non-empty **`invariance_compare_keys`**. Adjust **`invariance_scale_c`** (`c > 1`) in the Config tab when π is enabled.
- **Audit / visualize** — optional `r_ref` and weights JSON; session log append for multi-step Plotly dashboards; redraw with a subset of RMS keys.
- **Export** — JSON reports, optional **PDF ZIP** if `moju[report]` is installed.

## Notes

- `MonitorConfig` JSON must not rely on `implied_fn` (not serializable); use the Python API for that.
- π-constant audits require non-empty `invariance_compare_keys` and (outside the NPZ shim) a `state_builder` that recomputes from scaled `constants`; see Path A notes above.
- JAX on shared hosting (e.g. Streamlit Cloud) may need extra setup; local or VM is recommended.
