# moju — Physics-AI supervision for engineering-grade simulations

```bash
pip install moju
```

**Moju makes AI models physically admissible and auditable.** It is a lightweight framework for enforcing physics constraints during training, composing dimensionless groups and constitutive models with governing laws, and auditing how well predictions satisfy physics.

*Physics you know, in the AI you train. Dimensionless scaling, constitutive models, and equation residuals in one JAX library.*

---

## Why moju?

Most Physics AI tools focus on adding a physics loss. Moju goes further:

- **Structured physics** — Models, Groups, and Laws as composable building blocks (Reynolds number, viscosity, conservation equations).
- **Automatic residual construction** — `ResidualEngine.compute_residuals(...)` builds law, constitutive, and scaling residuals from your state.
- **Physics admissibility scoring** — `audit(log)` returns per-key and per-category scores; **training** logs also get a single **overall** score, while **eval** logs omit that rollup (use categories and per-key metrics instead).
- **Works across PINNs, CFD surrogates, and other state predictors** — Differentiable end-to-end; use in training loops or as a standalone audit toolkit.

---

## The big idea

Moju treats physics as composable building blocks:

```
Predictions (state_pred)
        ↓
Constitutive models (Models.*) + Dimensionless groups (Groups.*)
        ↓
Governing laws (Laws.*)
        ↓
ResidualEngine.compute_residuals(...)  →  residuals
        ↓
loss = build_loss(residuals)     report = audit(engine.log)
```

Built-in `Laws.*` residuals are **nondimensional**: supply fields and derivatives in each law’s documented scaled sense, and use `Groups.*` / `Models.*` for dimensionless groups and constitutive recovery.

**Residual conventions (ND-first):**

- **Governing laws** (`Laws.*`): PDE balance residuals in the documented nondimensional sense.
- **Constitutive `implied_delta` and `ref_delta`:** always **nondimensional**—by default
  \((F - \tilde F) / (\varepsilon + |F| + |\tilde F|)\) where \(\tilde F\) is the implied value or \(F(\text{ref})\).
  Catalog **`Models.*` / `Groups.*`** still evaluate physical formulas with your state keys; the **logged closure tensors** use that discrepancy only. Optional denominator \((\varepsilon + |\text{ref}|)\) when **`implied_delta_ref_key`** / **`ref_delta_ref_key`** or **`{output_key}_ref`** is present in merged state/constants (see `moju.monitor.closure_registry.apply_closure_discrepancy_normalize`).
- **Scaling audits:** group values are dimensionless; π-constant checks compare scaled states.

Instead of hand-wiring `loss = data_loss + physics_loss`, you get residuals from the engine, a physics loss from `build_loss(residuals)`, and an admissibility report from `audit(engine.log)`.

---

## 5-minute example

Run this after `pip install moju`:

```python
import jax.numpy as jnp
from moju.monitor import ResidualEngine, build_loss, audit, MonitorConfig, AuditSpec
from moju.piratio import Models, Groups

mu0 = jnp.array(1.8e-5)
T0 = jnp.array(273.0)
S = jnp.array(110.4)

T = jnp.array(300.0)
mu = Models.sutherland_mu(T=T, mu0=mu0, T0=T0, S=S)

Re = jnp.array(10.0)
Pr = jnp.array(2.0)
Pe = Groups.pe(re=Re, pr=Pr)

cfg = MonitorConfig(
    laws=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_xx"}}],
    constitutive_audit=[
        AuditSpec(
            name="sutherland_mu",
            output_key="mu",
            state_map={"T": "T", "mu0": "mu0", "T0": "T0", "S": "S"},
        )
    ],
    scaling_audit=[
        AuditSpec(
            name="pe",
            output_key="Pe",
            state_map={"re": "Re", "pr": "Pr"},
        )
    ],
)

engine = ResidualEngine(config=cfg)

state_pred = {
    "phi_xx": jnp.array(0.0),
    "T": T,
    "mu0": mu0,
    "T0": T0,
    "S": S,
    "mu": mu * 1.01,
    "Re": Re,
    "Pr": Pr,
    "Pe": Pe,
}
state_ref = {
    "T": T,
    "mu0": mu0,
    "T0": T0,
    "S": S,
    "mu": mu,
    "Re": Re,
    "Pr": Pr,
    "Pe": Pe,
}

residuals = engine.compute_residuals(
    state_pred, state_ref=state_ref, run_mode="eval"
)
loss = build_loss(residuals)
report = audit(engine.log)

print("Physics loss:", float(loss))
print("Overall (NaN in eval):", report["overall_admissibility_score"], report["overall_admissibility_level"])
print("Per category:", report["per_category"])
```

---

## What you get

Moju gives you physics diagnostics, not just a loss. The audit report looks like this:

| Category              | Score |
| --------------------- | ----- |
| Governing laws        | 0.92  |
| Constitutive         | 0.94  |
| Scaling and similarity | 0.96 |

**Overall admissibility score** — for **training** logs (`compute_residuals(..., run_mode="training")`, the default), the geometric mean of **governing laws** and **constitutive** only. For **eval** logs (`run_mode="eval"`), **not defined**: `report["overall_admissibility_score"]` is **`nan`** and `report["overall_admissibility_level"]` is **Unknown**. Legacy logs **without** `run_mode` still use the geometric mean of every present category (`laws`, `constitutive`, `scaling`, `data`).  
**Overall admissibility level** — for a finite overall score, derived in **`[0, 1]`** by `admissibility_level`: **&lt; 0.5** Non-Admissible; **0.5–0.75** Low Admissibility; **0.75–0.95** Moderate Admissibility; **&gt; 0.95** High Admissibility (same bands for per-key scores in `per_key`).

Report keys: `report["per_category"]` (includes **`data`** when those keys exist), `report["overall_admissibility_score"]`, `report["overall_admissibility_level"]`, `report["monitor_run_mode"]`. Per-key **R_eff** (logged as `rms`), R_norm, and admissibility are in `report["per_key"]`.

**Admissibility levels:** (1) each residual key has its own score in `per_key`; (2) each category score in `per_category` is the **geometric mean** of **finite** per-key scores in that category (NaN/inf keys are skipped; categories with no finite keys are omitted); (3) the **overall** score (training or legacy logs only) is the geometric mean of the relevant **finite** category scores as above. Logged **`rms`** is **R_eff** = √(mean(r²)+δ²)·**Q^0.5** (tiny **δ²** = **`R_EFF_RMS_JITTER_SQ`**, AD-smooth at **r→0**) with **Q** = RMS(m)/mean(m), **m_i** = √(r_i²+ε²); **Q = 1** when |r| is uniform across collocation points (and for single-point tensors). Default logged **`scale_k`** for **laws/** and nondimensional **implied_delta** / **ref_delta** is **2×10⁻²** (see **`DEFAULT_NONDIM_R_NORM_SCALE_K`** in `moju.monitor.auditor`). Optional **`audit(..., r_ref=...)`** / **`visualize(..., r_ref=...)`** overrides **`scale_k`** per flat key. **NaN-tolerant** reductions apply where applicable. New metrics use the same pipeline—for example optional **π-constant** checks on a scaling audit add a key `scaling/<name>/pi_constant` (eval + Path A only; see `docs/monitor_training_vs_eval.md`) and are included in the scaling category mean when you run **`run_mode="eval"`**. π-constant recipes exist for **every** registered dimensionless group (`list_pi_constant_group_names()`); each recipe scales selected inputs by powers of `c>1` so the group value is unchanged (see `moju.monitor.pi_constant_recipes`). For **Grashof** (`gr`), `g` is fixed inside `Groups.gr`; the recipe only varies the other arguments. End-to-end π-constant examples: `examples/cookbook_pi_constant_reynolds.py`, `examples/cookbook_pi_constant_prandtl.py`. Turbulence-related constitutive audit cookbooks: `examples/cookbook_turbulence_law_of_wall.py`, `examples/cookbook_turbulence_colebrook.py`, `examples/cookbook_constitutive_smagorinsky.py`, `examples/cookbook_constitutive_k_epsilon.py` (k–ε νₜ), `examples/cookbook_constitutive_k_omega.py` (k–ω νₜ). These νₜ closures are algebraic only; full k–ε/k–ω transport belongs in `Laws.*` if you need PDE residuals. **Implied constitutive audit** (`constitutive/<name>/implied_delta`): compare `Models.*` to an alternate value in `state_pred` via `AuditSpec.implied_value_key`, or to `implied_fn(state, constants)` (Python-only; omitted from `to_dict()`). Cookbooks: `examples/cookbook_constitutive_implied_ideal_gas_rho.py`, `examples/cookbook_constitutive_implied_power_law_fn.py`.

**Law-linked implied audits (default on)** — For several `Laws.*` entries, Moju **prepends** matching `constitutive_audit` / `scaling_audit` rows whose `implied_fn` recomputes a quantity by rearranging the law using your law `state_map` (e.g. **Fourier conduction** → `Models.thermal_diffusivity(k,rho,cp)` vs **α\_implied = T_t / T_laplacian**). Logged **`implied_delta`** / **`ref_delta`** tensors use the **default nondimensional symmetric normalization** (see “Residual conventions” above). Residual keys look like `constitutive/thermal_diffusivity/law_fourier_conduction/implied_delta`. We **do not** add a separate implied row for **Fo** when **α** is already checked (same information given fixed `t`, `L`). Toggle with `MonitorConfig(law_implied_audits=False)` or `ResidualEngine(..., law_implied_audits=False)`. With **`state_ref`** and **`run_mode="eval"`**, **`ref_delta`** runs for those rows (unless a spec sets `include_ref_delta: false`). In **`run_mode="training"`** (default), **`state_ref`** is ignored for **`ref_delta`** and for the **`data/`** pred−ref block—use a separate eval pass for ground-truth comparison. Registry: `list_laws_with_implied_diagnostics()`, `merge_law_implied_audit_specs`, `moju.monitor.law_implied_diagnostics`; Studio prepends the same rows in `build_studio_auto_fragment`. Details: [docs/law_implied_audits.md](docs/law_implied_audits.md).

---

## Use cases

- **Physics-Informed Neural Networks (PINNs)** — Residuals and loss from governing equations; audit score each step.
- **CFD surrogate models** — Compare to high-fidelity data via `state_ref`; constitutive and scaling audits.
- **Digital twins** — Continuous audit of predictions against physics and data.
- **Scale-invariant modeling** — Dimensionless groups (Re, Pr, Pe, …) and scaling-similarity audits.

---

## Core concepts

| Concept         | Meaning |
| --------------- | ------- |
| **Models**      | Constitutive relationships (e.g. viscosity μ(T), density ρ(P,T)). |
| **Groups**      | Dimensionless quantities (Re, Pr, Pe, Ma, …). |
| **Laws**        | Governing equations (mass, momentum, energy, …); residuals go into `build_loss`. |
| **ResidualEngine** | Builds state from config and optional predictions; runs laws and optional constitutive/scaling audits (`implied_delta` always when configured; `ref_delta` / `data/` / π-constant only with **`run_mode="eval"`**); produces residuals and a log. |
| **build_loss**  | Builds a scalar physics loss from law residuals using the same **R_eff** reduction as the log’s `rms`. |
| **audit**       | Takes the engine log; returns per-key and per-category admissibility and overall score. |

---

## Installation

```bash
pip install moju
```

Optional extras:

- `pip install moju[ref]` — xarray-based `state_ref` loaders and interpolation.
- `pip install moju[ref_vtk]` — VTK/VTU loaders (meshio).
- `pip install moju[ref_foam]` — OpenFOAM snapshot loaders (meshio).
- `pip install moju[ref_hdf5]` — HDF5 loaders (h5py).
- `pip install moju[report]` — PDF Physics Admissibility Report from `audit(..., export_dir=...)`.
- `pip install moju[viz]` — **plotly** for **`visualize(engine.log, backend="plotly"|"none")`** (default **`plotly`**), with `mode="training"|"eval"` (legacy **`mode="test"`** is a silent alias for **`eval`**), optional **`spatial_law_panel`**, **`spatial_heatmap_colorscale`** (default **Viridis**), **`step_label`**, **`r_norm_scale="log"|"linear"`**, **`figure_title`**, **`dashboard_mode`**, **`theme="light"`**, **`baseline_score`**, **`show_branding`** (optional watermark; default off), **`visualize_layout="single"|"split"`** ( **`split`** adds a **`worst_keys`** table figure), **`worst_keys_top_n`**, **`density="comfortable"|"compact"`**, and **`ResidualEngine.clear_log()`** between runs. The single-figure output is a decision-oriented **Physics Admissibility Report**. **Training:** two KPI cards (Governing / Constitutive), overall admissibility subtitle when defined, vs-step trend, category breakdown, residuals, spatial row, summary with troubleshooting hints when a category lags. **Eval:** three KPI cards; when **`run_mode="eval"`** in the log, the title explains that roll-up overall is undefined and points to category scores. **`dashboard_mode="dash-tabs"`:** the eval **KPI** tab mirrors those category indicators plus a **`run_mode`** note (no empty placeholder). Pass **`keys=[...]`** or **`r_ref=...`** to subset or rescale like `audit`. In **Jupyter or Colab**, **restart the kernel** after upgrading `moju` so `visualize` loads the matching `visualize_plotly` code.
- `pip install moju[studio]` — Streamlit + Plotly for **Moju Studio** (`streamlit run apps/moju_studio/Home.py` from a source checkout; see `apps/moju_studio/README.md`).
- `pip install moju[studio-science]` — optional **HDF5 / NetCDF** state uploads in Studio (`h5py`, `xarray`, `netCDF4`); `.npz` / `.npy` work with `studio` alone.

| If you need… | Extra | Install |
| ------------- | ----- | ------- |
| Reference grids / NetCDF → `state_ref` | `ref` | `pip install moju[ref]` |
| VTK/VTU reference | `ref_vtk` | `pip install moju[ref_vtk]` |
| OpenFOAM reference | `ref_foam` | `pip install moju[ref_foam]` |
| HDF5 reference | `ref_hdf5` | `pip install moju[ref_hdf5]` |
| PDF report export | `report` | `pip install moju[report]` |
| Plotly monitoring dashboards | `viz` | `pip install moju[viz]` |
| Moju Studio (Streamlit) | `studio` | `pip install moju[studio]` |
| Studio HDF5 / NetCDF uploads | `studio-science` | `pip install moju[studio-science]` |
| PyTorch ↔ JAX law bridge | `torch` | `pip install moju[torch]` |

### Troubleshooting import errors

- **`ImportError` for xarray, h5py, plotly, streamlit, reportlab, …**  
  Install the matching extra from the table above (e.g. `moju[ref]` for xarray loaders, `moju[studio-science]` for Studio HDF5/NetCDF). Core `pip install moju` only pulls JAX and NumPy.

- **`ValueError: numpy.dtype size changed` or similar when importing an optional package**  
  Usually a **binary wheel mismatch** after upgrading NumPy (e.g. NumPy 2 vs extensions built for NumPy 1). Use a **clean virtual environment**, align versions (`pip install -U numpy h5py xarray` / the failing package), or reinstall the optional stack. `moju.monitor.state_ref` catches a broken xarray import so `import moju.monitor.state_ref` still loads; xarray-based helpers then raise a clear error until the environment is fixed.

---

## Philosophy

Moju does not define physics. Moju provides a structured way to **enforce** and **audit** it. You bring your governing equations, constitutive models, and dimensionless groups; moju gives you residuals, a differentiable loss, and an admissibility score. JAX-native and fully differentiable so it fits into training loops and high-stakes workflows.

---

## Learn more

**API at a glance** — Two namespaces: **moju.piratio** (Groups, Models, Laws, Operators) and **moju.monitor** (ResidualEngine, `MonitorConfig`, `AuditSpec`, `audit_spec_to_engine_dict`, `PathBGridConfig`, `fill_path_b_derivatives`, `fill_law_fd_from_primitives`, `list_law_fd_supported_laws`, **`merge_law_implied_audit_specs`**, **`list_laws_with_implied_diagnostics`**, **`law_implied_unsupported_reasons`**, **`effective_audit_specs_for_fragment`**, build_loss, audit, **`visualize(..., backend="plotly"|"none", mode="training"|"eval", spatial_law_panel=..., r_norm_scale=...)`** for training/eval dashboards (`test` still accepted as alias for `eval`), `pretty_residual_key` / `pretty_category_name` for display). Law-linked implied rows follow a strict constitutive-only policy; use `law_implied_unsupported_reasons()` for laws pending constitutive target/model support. Constitutive/scaling closure keys include `ref_delta`, `implied_delta`, and scaling `pi_constant` when configured. Path B optional FD: `compute_residuals(..., auto_path_b_derivatives=...)` with `fill_law_fd=True` fills missing **registered** `Laws.*` inputs on structured grids. Use `engine.required_state_keys()` for introspection.

**Examples**

- Quick scaling and laws: `Groups.re(...)`, `Models.ideal_gas_rho(...)`, `Laws.mass_incompressible(u_grad)` — see snippets in the full docs.
- End-to-end NN → residuals → PDF: `python examples/monitor_heat_end_to_end.py`, `python examples/monitor_burgers_end_to_end.py`.
- CFD snapshot → state_ref → audit: `examples/cfd_snapshot_cookbook_heat_1d.py`; reference loaders: `examples/monitor_state_ref_from_vtu_demo.py`, `from_openfoam`, `from_hdf5`.
- Path B auto-FD (law inputs): `examples/cookbook_path_b_fd_law_laplace.py` (`phi_laplacian` fill for `laplace_equation`).
- Implied constitutive audit (`implied_delta`): `examples/cookbook_constitutive_implied_ideal_gas_rho.py`, `examples/cookbook_constitutive_implied_power_law_fn.py`.

**Paths** — Path A: pass `(model, params, collocation)` and a `state_builder` to build `state_pred`. Path B: pass `state_pred` directly (e.g. from CFD or finite differences). Optional **structured-grid FD**: `compute_residuals(..., auto_path_b_derivatives=True|PathBGridConfig)` with `fill_law_fd=True` fills missing **registered** `Laws.*` inputs (e.g. `phi_laplacian`, `u_grad`) via `law_fd_recipes`. Constitutive and scaling audits use specs tied to `Models.*` and `Groups.*`: **ref_delta** (needs `state_ref`), **implied_delta** on constitutive specs only (`AuditSpec.implied_value_key` or `implied_fn`; `implied_fn` is omitted from `MonitorConfig.to_dict()`—use in-memory `AuditSpec` + `ResidualEngine(config=...)` or `audit_spec_to_engine_dict`), and **π-constant** checks on scaling (Path A). **R_norm** = **R_eff**/scale_k where **R_eff** is logged as `rms` (see admissibility section above); default scale_k **2×10⁻²** for **laws/** and nondimensional **implied_delta** / **ref_delta**; other audit keys and **data/** use state/reference-derived scales. Optional `audit(log, r_ref=...)` overrides scale_k per key. Admissibility uses 1/(1+R_norm) per key.

**Docs** — [VERSIONING.md](VERSIONING.md). Online docs: overview, Groups, Models, Laws, Operators.

---

## License

MIT License. Developed by Ifimo Lab, a division of Ifimo Analytics.
