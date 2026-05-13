# moju

Physics supervision and audit tools for SciML and Physics AI.

```bash
pip install moju
```

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

- `moju.piratio.Models` - constitutive relationships such as viscosity, density, diffusivity, wave speed, and turbulence closures.
- `moju.piratio.Groups` - dimensionless quantities such as `re`, `pr`, `pe`, `fo`, `ma`, and `bi`, materialized into state for laws.
- `moju.piratio.Laws` - governing-equation residuals for heat, diffusion, wave, momentum, mass, Darcy/Brinkman, Poisson, Burgers, and related equations.
- `moju.piratio.Operators` - JAX autodiff helpers such as gradients, divergence, Laplacian, curl, and time derivatives.
- `moju.monitor.ResidualEngine` - runs laws, groups, constitutive audits, optional data comparisons, and records audit logs.
- `moju.monitor.audit` - converts logs into R_norm, admissibility scores, category summaries, and report data.
- `moju.monitor.visualize` - Plotly dashboards for training/eval residuals, category scores, spatial fields, and constitutive diagnostics. The constitutive row shows a **Divergence** heatmap (normalised as `(model − implied) / (|model| + ε)`) alongside a **Constitutive Consistency** line plot with spatially varying ±1 % / ±5 % acceptability bands and tier boundary markers centred on the model prediction.

## Training vs Eval

`compute_residuals(..., run_mode="training")` is the default for optimization loops. It runs laws, groups, and constitutive implied audits. `state_ref` is ignored in training mode.

Use `run_mode="eval"` when you want reference comparisons. In eval mode, `state_ref` enables constitutive `ref_delta` and `data/` residuals.

Overall admissibility is the minimum of the finite category scores participating in the current run mode. Training rolls up laws and constitutive categories. Eval also includes `data` when present. Legacy logs may still contain a historical `scaling` category.

Details: [`docs/monitor_training_vs_eval.md`](docs/monitor_training_vs_eval.md).

## PyTorch Support

Install the PyTorch extra:

```bash
pip install "moju[torch]"
```

`moju.torch` provides:

- `TorchResidualEngine` - PyTorch-facing residual engine with parity-oriented behavior.
- `build_loss_torch` and `r_eff_scalar_torch` - Torch-native R_eff loss helpers.
- `wrap_law_torch` - wrap JAX `Laws.*` functions for use with Torch tensors through `jax2torch`.
- Torch-native nondimensionalization helpers.

Start with [`scripts/torch_laws_jax2torch_example.py`](scripts/torch_laws_jax2torch_example.py). The implementation is covered by `tests/test_torch_engine.py` and `tests/test_torch_interop.py`.

## Installation Extras

- `moju[viz]` - Plotly dashboards.
- `moju[report]` - PDF report export.
- `moju[ref]` - xarray-based reference loaders.
- `moju[ref_vtk]` - VTK/VTU reference loaders.
- `moju[ref_foam]` - OpenFOAM snapshot loaders.
- `moju[ref_hdf5]` - HDF5 reference loaders.
- `moju[studio]` - Streamlit-based Moju Studio.
- `moju[studio-science]` - HDF5/NetCDF upload support for Studio.
- `moju[torch]` - PyTorch and `jax2torch` integration.
- `moju[dev]` - test and formatting tools.

## Documentation

- GitHub Pages source and API overview: [`docs/`](docs/)
- Training vs eval behavior: [`docs/monitor_training_vs_eval.md`](docs/monitor_training_vs_eval.md)
- Law-linked constitutive implied audits: [`docs/law_implied_audits.md`](docs/law_implied_audits.md)
- Moju Studio: [`apps/moju_studio/README.md`](apps/moju_studio/README.md)
- Versioning policy: [`VERSIONING.md`](VERSIONING.md)
- Changelog: [`CHANGELOG.md`](CHANGELOG.md)

## Examples

- Full 1D slab cooling demo: [`examples/slab_cooling_demo.py`](examples/slab_cooling_demo.py)
- CFD snapshot audit: [`examples/cfd_snapshot_cookbook_heat_1d.py`](examples/cfd_snapshot_cookbook_heat_1d.py)
- Path B finite-difference law fill: [`examples/cookbook_path_b_fd_law_laplace.py`](examples/cookbook_path_b_fd_law_laplace.py)
- Constitutive divergence dashboard: [`examples/cookbook_constitutive_divergence.py`](examples/cookbook_constitutive_divergence.py)
- Torch interop: [`scripts/torch_laws_jax2torch_example.py`](scripts/torch_laws_jax2torch_example.py)

## Philosophy

Moju does not define physics for you. It gives you a structured way to apply the physics you already trust, measure residuals consistently, and surface where a model agrees or disagrees with governing laws and constitutive assumptions.

## License

MIT License. Developed by Ifimo Lab, a division of Ifimo Analytics.
