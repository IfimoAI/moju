# Path B — media / instant audit assets

Pre-exported PINN states for the wide `128×128×128` (w2) slab-cooling run. Use these for quick Moju audits and figures (social posts, demos) without retraining.

## Files in `data/`

| File | Contents |
|------|----------|
| `wide2_const_prop_1D_cooling_slab_final_training_state.json.zip` | Training collocation state (`T`, `T_t`, `T_x`, `T_xx`, `t`, `x`) |
| `wide2_const_prop_1D_cooling_slab_test_state_pred.json.zip` | Eval-grid `state_pred` (same keys) |
| `wide2_const_prop_1D_cooling_slab.csv` | Optional training monitor export for trajectory plots |

Raw `.json` dumps stay local (gitignored). Only `wide2*.json.zip` bundles are versioned.

## Load and audit (minimal)

```python
import jax.numpy as jnp
from moju.monitor import ResidualEngine, audit, implied_group_specs_for_laws
from moju.piratio import Models

from examples.Notebooks.media.export_state_zips import load_state_from_json_zip

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

state = load_state_from_json_zip("data/wide2_const_prop_1D_cooling_slab_test_state_pred.json.zip")
state_jax = {k: jnp.asarray(v) for k, v in state.items()}
residuals = engine.compute_residuals(state_jax, log_to_python=True)
print(audit(engine.log, last_residual_dict=residuals))
```

## Regenerating bundles

1. Run the playground w2 notebook through L-BFGS training and eval (`state_final`, `state_pred` in memory):

   `playground/.../moju_1D_Heat_Simulation_lbfgs_128x128x128_w2.ipynb`

2. Export with the shared helper:

   ```python
   from examples.Notebooks.media.export_state_zips import export_wide2_states
   export_wide2_states(state_final, state_pred, "examples/Notebooks/media/data")
   ```

   Or repack an existing zip (cleans archive metadata):

   ```bash
   python examples/Notebooks/media/export_state_zips.py --repack examples/Notebooks/media/data/wide2_*.json.zip
   ```

Do not hand-edit zip contents.
