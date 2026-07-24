# Path B derivative fill (FD and spectral)

Path B passes a ready `state_pred` into `ResidualEngine`. When law inputs such as `u_grad`, `u_laplacian`, or `T_t` are missing, Moju can **optionally** fill them from primitive fields on a structured grid. This page is the canonical guide for that fill path.

**Related:** [monitor_training_vs_eval.md](monitor_training_vs_eval.md) (training/eval and Path B units) · [law_implied_audits.md](law_implied_audits.md) (constitutive implied δ) · cookbooks below.

## When to fill vs supply derivatives yourself

| Situation | Recommendation |
|-----------|----------------|
| Live PINN with autodiff | **Path A** (`state_builder` + Operators) — do not use Path B spectral |
| Field dump already has `*_grad` / `*_laplacian` / `*_t` | Pass them in `state_pred`; Moju **never overwrites** non-`None` keys |
| Periodic Fourier / FNO-style grid, missing spatial derivatives | **Path B spectral** (`diff_method="spectral"`, `periodic=True`) |
| Non-periodic / wall BCs / irregular structured grids | **Path B FD** (default) |
| Neural-operator audit and you can differentiate in the FNO basis | Prefer **precomputing spectral ∂ outside Moju** and putting them in `state_pred` (best for implied ν-δ); use Moju spectral only when you have fields + coords but no model handle |

Implied constitutive audits (especially Burgers **ν-δ**) are **Laplacian-sensitive**. Prefer accurate spatial derivatives (spectral or model-native) over coarse FD when those audits lead the analysis.

## API surface

| Piece | Role |
|-------|------|
| [`PathBGridConfig`](../moju/monitor/path_b_derivatives.py) | Layout, dimension, `diff_method`, `periodic` |
| `fill_path_b_derivatives(..., fill_law_recipes=True)` | Generic fill; honors `grid.diff_method` |
| `fill_path_b_spectral(...)` | Convenience: forces `diff_method="spectral"`, `periodic=True`, recipes on |
| `fill_law_fd_from_primitives` | Law-recipe fill used under the hood |
| `ResidualEngine.compute_residuals(..., auto_path_b_derivatives=..., fill_law_fd=True)` | Engine wiring |
| Torch: `fill_path_b_derivatives_torch(..., diff_method=..., periodic=...)` | Same idea for `moju.torch` |
| Torch: `TorchResidualEngine(path_b_fill=True, path_b_diff_method="spectral", path_b_periodic=True)` | Engine-level spectral |

Registered law arguments and kinds live in `moju.monitor.law_fd_recipes.LAW_FD_RECIPES` (`laplacian`, `vector_laplacian`, `grad_scalar`, `jacobian`, `dt`, `dtt`).

## Defaults (important)

- **`diff_method="fd"`** is the default.
- Bare **`auto_path_b_derivatives=True`** always uses **FD** — spectral is **never** silent.
- Spectral requires **`periodic=True`** and **uniform** rectilinear spacing; otherwise Moju raises `ValueError`.
- **Temporal** `dt` / `dtt` always use FD (time stacks are rarely periodic), even when spatial fill is spectral.

## Spectral usage (JAX)

```python
from moju.monitor import PathBGridConfig, ResidualEngine, fill_path_b_spectral

# Standalone fill
state, warns = fill_path_b_spectral(
    {"u": u, "x": x},  # periodic uniform x, e.g. linspace(0, L, n, endpoint=False)
    laws_spec=[{"name": "burgers_equation", "state_map": {...}}],
    constants={"re": Re, "U": U, "L": L},
    grid=PathBGridConfig(spatial_dimension=1, layout="separable", steady=True),
)

# Via ResidualEngine
grid = PathBGridConfig(
    diff_method="spectral",
    periodic=True,
    spatial_dimension=1,
    layout="separable",
    steady=True,
)
residuals = engine.compute_residuals(
    state_pred,
    auto_path_b_derivatives=grid,
    fill_law_fd=True,
)
```

Coordinates: use **separable** 1D axis vectors (`x` length `nx`, …) or rectilinear meshgrids Moju can reduce to 1D axes. Period length is `L = n * dx` per axis.

## Spectral usage (Torch)

```python
from moju.torch import TorchResidualEngine
from moju.torch._path_b import fill_path_b_derivatives_torch

state, warns = fill_path_b_derivatives_torch(
    {"u": u, "x": x},
    diff_method="spectral",
    periodic=True,
)

engine = TorchResidualEngine(
    laws=[{"name": "burgers_equation"}],
    path_b_fill=True,
    path_b_diff_method="spectral",
    path_b_periodic=True,
)
```

## FD usage (default)

```python
# True → PathBGridConfig with diff_method="fd"
residuals = engine.compute_residuals(
    state_pred,
    auto_path_b_derivatives=True,
    fill_law_fd=True,
)
```

Cookbook: [`examples/cookbook_path_b_fd_law_laplace.py`](../examples/cookbook_path_b_fd_law_laplace.py).

## Spectral cookbook

[`examples/cookbook_path_b_spectral_burgers.py`](../examples/cookbook_path_b_spectral_burgers.py) — periodic 1D `sin(x)`, spectral fill of Burgers `u_grad` / `u_laplacian`, ResidualEngine smoke.

## Requirements and failure modes

| Requirement | If violated |
|-------------|-------------|
| `periodic=True` with spectral | `ValueError` |
| Uniform spacing per spatial axis | `ValueError` / warning; use FD or resample |
| Primitive field present (e.g. `u` for `u_laplacian`) | Skip that key; warning in fill list |
| Target already in `state_pred` | Left unchanged |
| Curvilinear / non-rectilinear mesh | Unsupported for spectral (and limited for FD) |

## Module map

- `moju.monitor.path_b_derivatives` — `PathBGridConfig`, `fill_path_b_derivatives`, `fill_path_b_spectral`
- `moju.monitor.path_b_spectral` — FFT helpers and validation
- `moju.monitor.law_fd_recipes` — law argument recipes + FD/spectral dispatch
- `moju.torch._path_b` — Torch FD/spectral fill
