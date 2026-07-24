# moju

Physics supervision and audit tools for SciML and Physics AI.

```bash
pip install moju
```

## Paper (Zenodo)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20519331.svg)](https://doi.org/10.5281/zenodo.20519331)

**[Moju: A Physics Admissibility Auditing Framework for Scientific Machine Learning Surrogates](https://zenodo.org/records/20519331)** — preprint PDF on Zenodo. Software and reproduction notebooks are on GitHub (see below). Canonical links: [docs/paper_metadata.md](https://github.com/IfimoAI/moju/blob/main/docs/paper_metadata.md).

## Paper and demo notebooks (Colab)

| Notebook | Description |
|----------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IfimoAI/moju/blob/main/examples/Notebooks/moju_slab_cooling_paper.ipynb) [Path A — paper reproduction](https://github.com/IfimoAI/moju/blob/main/examples/Notebooks/moju_slab_cooling_paper.ipynb) | Train `32×32×32` PINN, Moju audit, verify vs paper reference |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IfimoAI/moju/blob/main/examples/Notebooks/media/moju_slab_cooling_path_b.ipynb) [Path B — instant audit](https://github.com/IfimoAI/moju/blob/main/examples/Notebooks/media/moju_slab_cooling_path_b.ipynb) | Load pre-exported w2 states, audit and visualize (no training) |

Open-in-Colab badges require the notebooks on GitHub **`main`**. See [examples/Notebooks/README.md](https://github.com/IfimoAI/moju/blob/main/examples/Notebooks/README.md).

Moju helps you turn predicted state fields into governing-law residuals, physics losses, constitutive consistency checks, and audit reports. It is JAX-native at the core, with a PyTorch-facing interface available through `moju.torch`.

## What Moju Does

- Builds residuals from composable `Laws`, `Groups`, and `Models`.
- Turns governing-law residuals into a differentiable training loss with `build_loss`.
- Audits predictions with per-key, per-category, and overall admissibility scores.
- Infers law-linked constitutive checks where the governing equation implies a material property.
- Visualizes training/eval diagnostics, spatial residuals, and constitutive divergence and consistency.

Moju is not a training framework or a solver. It is a physics supervision layer you can use with PINNs, CFD surrogates, neural operators, digital twins, or any workflow that can provide a `state_pred` dictionary.

## 5-Minute Example: 1D Slab Cooling

This is the minimal Path B flow: pass `state_pred` directly. The derivatives are already in the state, so no finite-difference inference is needed. The Fourier law automatically adds the law-linked `thermal_diffusivity` implied audit.

```python
import jax.numpy as jnp

from moju.monitor import audit, build_loss, build_minimal_residual_engine, visualize

L = 0.02
rho = 2700.0
cp = 900.0
k = 200.0
alpha = k / (rho * cp)

x = jnp.linspace(0.0, L, 64)
t = jnp.ones_like(x) * 10.0

# A toy variable temperature profile with supplied derivatives.
T = 300.0 + 20.0 * (1.0 - x / L) ** 2
T_laplacian = jnp.ones_like(x) * (40.0 / (L**2))
T_t = alpha * T_laplacian

state_pred = {
    "x": x,
    "t": t,
    "T": T,
    "T_t": T_t,
    "T_laplacian": T_laplacian,
    "L": jnp.ones_like(x) * L,
    "k": jnp.ones_like(x) * k,
    "rho": jnp.ones_like(x) * rho,
    "cp": jnp.ones_like(x) * cp,
    "alpha": jnp.ones_like(x) * alpha,
}

engine = build_minimal_residual_engine(
    law_names=["fourier_conduction"],
    coord_dimension=1,
)

residuals = engine.compute_residuals(state_pred, run_mode="training")
loss = build_loss(residuals)
report = audit(engine.log)
fig = visualize(engine.log, engine=engine)

print("Physics loss:", float(loss))
print("Overall admissibility:", report["overall_admissibility_score"])
print("Categories:", report["per_category"].keys())
fig.show()
```

What happens here:

- `build_minimal_residual_engine(...)` creates the Fourier conduction law and the needed `fo` group row.
- `state_pred` supplies the variable field `T`, its derivatives `T_t` and `T_laplacian`, coordinates, and material properties.
- `law_implied_audits=True` is the default, so Moju adds `constitutive/thermal_diffusivity/law_fourier_conduction/implied_delta`.
- `build_loss` uses governing-law residuals for training.
- `audit` and `visualize` use the log plus `engine.last_residuals` to report physics diagnostics.

## Core Concepts

- `moju.piratio.Models` - constitutive relationships such as viscosity, density, diffusivity, wave speed, and turbulence closures (including `kinematic_viscosity_from_re` for Burgers).
- `moju.piratio.Groups` - dimensionless quantities such as `re`, `pr`, `pe`, `fo`, `ma`, and `bi`, materialized into state for laws.
- `moju.piratio.Laws` - governing-equation residuals for heat, diffusion, wave, momentum, mass, Darcy/Brinkman, Poisson, Burgers, and related equations.
- `moju.piratio.Operators` - JAX autodiff helpers such as gradients, divergence, Laplacian, curl, and time derivatives.
- `moju.monitor.ResidualEngine` - runs laws, groups, constitutive audits, optional data comparisons, and records audit logs. With default **`law_implied_audits=True`**, selected laws prepend constitutive implied rows (e.g. Fourier → `thermal_diffusivity`; **Burgers → `kinematic_viscosity_from_re`** with implied **ν**). Path B can optionally fill missing law derivatives with **FD** (`auto_path_b_derivatives=True`) or **periodic spectral** (`PathBGridConfig(diff_method="spectral", periodic=True)` / `fill_path_b_spectral`); see [`docs/path_b_derivatives.md`](https://github.com/IfimoAI/moju/blob/main/docs/path_b_derivatives.md).
- `moju.monitor.audit` - converts logs into R_norm, admissibility scores, category summaries, and report data.
- `moju.monitor.visualize` - Plotly dashboards for training/eval residuals, category scores, spatial fields, and constitutive diagnostics. The constitutive row shows a **Divergence** heatmap (normalised as `(model − implied) / (|model| + ε)`) alongside a **Constitutive Consistency** line plot with spatially varying **±0.1 % / ±0.5 % / ±1 %** acceptability bands and tier boundary markers centred on the model prediction.

## Training vs Eval

`compute_residuals(..., run_mode="training")` is the default for optimization loops. It runs laws, groups, and constitutive implied audits. `state_ref` is ignored in training mode.

Use `run_mode="eval"` when you want reference comparisons. In eval mode, `state_ref` enables constitutive `ref_delta` and `data/` residuals.

Overall admissibility is the **minimum** of the finite category scores participating in the current run mode. Training rolls up **laws** and **constitutive** categories. Eval also includes `data` when present. Legacy logs may still contain a historical `scaling` category.

### Why two admissibility metrics?

Governing **laws** are scored with **RMS `R_eff`** (average compliance across collocation points). Constitutive **`implied_delta`** / **`ref_delta`** closure keys use **worst-point** `max |δ|` for admissibility — a PINN can satisfy the PDE on average while cheating closure at a few hotspots. Logged **`rms`** and **`build_loss`** stay RMS everywhere; audit/reporting uses the split. Constitutive category rolls up with **minimum** (not geometric mean). See **Admissibility metrics** in [`docs/monitor_training_vs_eval.md`](https://github.com/IfimoAI/moju/blob/main/docs/monitor_training_vs_eval.md).

For **multi-scale or high-Re** problems, governing laws use **auto term-balance `scale_k`** by default (floored at **`≈ 1e-2`**). Use **`law_scale_mode="fixed"`** or per-key **`audit(..., r_ref={...})`** when you want a fixed gauge — see **Calibrating scale_k** in that doc. Path B SI uploads: **`state_units="dimensional"`** (Studio checkbox **State in physical units (SI)**).

Details: [`docs/monitor_training_vs_eval.md`](https://github.com/IfimoAI/moju/blob/main/docs/monitor_training_vs_eval.md).

## PyTorch Support

Install the PyTorch extra:

```bash
pip install "moju[torch]"
```

`moju.torch` provides:

- `TorchResidualEngine` - PyTorch-facing residual engine with parity-oriented behavior. Optional Path B fill: `path_b_fill=True` with `path_b_diff_method="fd"|"spectral"` and `path_b_periodic` (spectral requires periodic grids; see [`docs/path_b_derivatives.md`](https://github.com/IfimoAI/moju/blob/main/docs/path_b_derivatives.md)).
- `build_loss_torch` and `r_eff_scalar_torch` - Torch-native R_eff loss helpers.
- `wrap_law_torch` - wrap JAX `Laws.*` functions for use with Torch tensors through `jax2torch`.
- Torch-native nondimensionalization helpers.

Start with [`scripts/torch_laws_jax2torch_example.py`](https://github.com/IfimoAI/moju/blob/main/scripts/torch_laws_jax2torch_example.py). The implementation is covered by `tests/test_torch_engine.py` and `tests/test_torch_interop.py`.

## Installation profiles

| Install | Use when |
|---------|----------|
| `pip install moju` | Default: residuals, audit, Plotly `visualize()`, PDF export |
| `pip install "moju[io]"` | Science file I/O for `state_ref` loaders (xarray, HDF5, VTK, NetCDF, …) |
| `pip install "moju[studio]"` | Moju Studio Streamlit app (+ HDF5/NetCDF uploads) |
| `pip install "moju[torch,io]"` | PyTorch training + file loaders |
| `pip install "moju[dev]"` | Development (pytest, black, ruff) |

Training demos that use **optax** (e.g. [`examples/slab_cooling_demo.py`](https://github.com/IfimoAI/moju/blob/main/examples/slab_cooling_demo.py)) require `pip install optax` in addition to `moju`.

## Documentation

- GitHub Pages source and API overview: [`docs/`](https://github.com/IfimoAI/moju/tree/main/docs)
- Training vs eval behavior: [`docs/monitor_training_vs_eval.md`](https://github.com/IfimoAI/moju/blob/main/docs/monitor_training_vs_eval.md)
- Path B FD / spectral derivative fill: [`docs/path_b_derivatives.md`](https://github.com/IfimoAI/moju/blob/main/docs/path_b_derivatives.md)
- Law-linked constitutive implied audits: [`docs/law_implied_audits.md`](https://github.com/IfimoAI/moju/blob/main/docs/law_implied_audits.md)
- Moju Studio: [`apps/moju_studio/README.md`](https://github.com/IfimoAI/moju/blob/main/apps/moju_studio/README.md)
- Versioning policy: [`VERSIONING.md`](https://github.com/IfimoAI/moju/blob/main/VERSIONING.md)
- Changelog: [`CHANGELOG.md`](https://github.com/IfimoAI/moju/blob/main/CHANGELOG.md)

## Examples

- Full 1D slab cooling demo: [`examples/slab_cooling_demo.py`](https://github.com/IfimoAI/moju/blob/main/examples/slab_cooling_demo.py)
- Burgers end-to-end (law + implied **ν**): [`examples/monitor_burgers_end_to_end.py`](https://github.com/IfimoAI/moju/blob/main/examples/monitor_burgers_end_to_end.py)
- CFD snapshot audit: [`examples/cfd_snapshot_cookbook_heat_1d.py`](https://github.com/IfimoAI/moju/blob/main/examples/cfd_snapshot_cookbook_heat_1d.py)
- Path B finite-difference law fill: [`examples/cookbook_path_b_fd_law_laplace.py`](https://github.com/IfimoAI/moju/blob/main/examples/cookbook_path_b_fd_law_laplace.py)
- Path B spectral (periodic) Burgers fill: [`examples/cookbook_path_b_spectral_burgers.py`](https://github.com/IfimoAI/moju/blob/main/examples/cookbook_path_b_spectral_burgers.py)
- Constitutive divergence dashboard: [`examples/cookbook_constitutive_divergence.py`](https://github.com/IfimoAI/moju/blob/main/examples/cookbook_constitutive_divergence.py)
- Torch interop: [`scripts/torch_laws_jax2torch_example.py`](https://github.com/IfimoAI/moju/blob/main/scripts/torch_laws_jax2torch_example.py)

## Philosophy

Moju does not define physics for you. It gives you a structured way to apply the physics you already trust, measure residuals consistently, and surface where a model agrees or disagrees with governing laws and constitutive assumptions.

## License

MIT License. Developed by Ifimo Lab, a division of Ifimo Analytics.
