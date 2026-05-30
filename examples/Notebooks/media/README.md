# Path B — media / instant audit assets

Pre-exported PINN states for quick Moju audits and figures (social posts, demos) without retraining.

## Bundled demo (`data/`)

The repo ships one canonical **128×128×128 (w2)** slab bundle (`DEFAULT_SLAB_PREFIX` in `export_state_zips.py`):

| File | Contents |
|------|----------|
| `wide2_const_prop_1D_cooling_slab_final_training_state.json.zip` | Training collocation state (`T`, `T_t`, `T_x`, `T_xx`, `t`, `x`) |
| `wide2_const_prop_1D_cooling_slab_test_state_pred.json.zip` | Eval-grid `state_pred` (same keys) |
| `wide2_const_prop_1D_cooling_slab.csv` | Optional training monitor export for trajectory plots |

Raw `.json` dumps stay local (gitignored). Only the bundled demo zips above are versioned; export other architectures locally unless you add gitignore allowlist entries.

## Naming convention

For any architecture or run tag `{prefix}`:

```text
{prefix}_final_training_state.json.zip
{prefix}_test_state_pred.json.zip
```

Example prefixes: `wide2_const_prop_1D_cooling_slab`, `32x32x32_opt_const_prop_1D_cooling_slab`.

## Load and audit (minimal)

```python
import jax.numpy as jnp
from moju.monitor import ResidualEngine, audit, implied_group_specs_for_laws
from moju.piratio import Models

from export_state_zips import DEFAULT_SLAB_PREFIX, load_state_from_json_zip, state_bundle_paths

L, cp, h, k, rho = 0.1, 900.0, 500.0, 200.0, 2700.0
engine_kw = {
    "constants": {
        "L": L, "cp": cp, "h": h, "k": k, "rho": rho,
        "alpha": Models.thermal_diffusivity(k, rho, cp),
    },
    "laws": [{
        "name": "fourier_conduction",
        "state_map": {"T_t": "T_t", "T_laplacian": "T_xx", "fo": "fo", "t": "t", "L": "L"},
    }],
    "groups": implied_group_specs_for_laws(["fourier_conduction"]),
}
engine = ResidualEngine(**engine_kw)

_, test_zip = state_bundle_paths("data", DEFAULT_SLAB_PREFIX)
state = load_state_from_json_zip(test_zip)
state_jax = {k: jnp.asarray(v) for k, v in state.items()}
residuals = engine.compute_residuals(state_jax, log_to_python=True)
print(audit(engine.log, last_residual_dict=residuals))
```

## Exporting bundles (any architecture)

After training, when `state_final` and `state_pred` are in memory:

```python
from export_state_zips import DEFAULT_SLAB_PREFIX, export_state_bundle

# Bundled w2 demo (default prefix)
export_state_bundle(state_final, state_pred, "examples/Notebooks/media/data")

# Any other architecture / run tag
export_state_bundle(
    state_final,
    state_pred,
    "examples/Notebooks/media/data",
    prefix="32x32x32_opt_const_prop_1D_cooling_slab",
)
```

Or repack an existing zip (cleans archive metadata):

```bash
python examples/Notebooks/media/export_state_zips.py --repack examples/Notebooks/media/data/wide2_*.json.zip
```

Do not hand-edit zip contents.
