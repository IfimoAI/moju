"""
Demo: ingest CFD-like reference data (xarray) into state_ref and audit.

Run:
  pip install moju[ref] moju[report]
  python examples/monitor_state_ref_from_xarray_demo.py
"""

import numpy as np
import xarray as xr

import jax.numpy as jnp

from moju.monitor import MonitorConfig, ResidualEngine, audit
from moju.monitor.state_ref import from_xarray


def main():
    # Minimal laws-only monitor; data residual comes from state_ref.
    cfg = MonitorConfig(
        laws=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_xx"}}],
    )
    engine = ResidualEngine(config=cfg)

    # Predicted state (Path B) at collocation points (pretend: model output + derivatives).
    # Here, phi_xx is just a scalar residual carrier for the Laplace demo.
    state_pred = {"phi_xx": jnp.array(0.5)}

    # Reference data on a labeled grid (CFD snapshot).
    ds = xr.Dataset(
        data_vars={"phi_xx_cfd": (("t", "x"), np.zeros((5, 8), dtype=float))},
        coords={"t": np.linspace(0.0, 1.0, 5), "x": np.linspace(0.0, 1.0, 8)},
    )

    # Ingest and interpolate reference data onto a target grid if needed.
    state_ref = from_xarray(
        ds,
        var_map={"phi_xx": "phi_xx_cfd"},
        target={"t": np.array([0.0]), "x": np.array([0.0])},
        method="linear",
    )

    engine.compute_residuals(state_pred, state_ref=state_ref)
    report = audit(engine.log, export_dir="exports", model_name="xarray_state_ref_demo")
    print("Wrote report with overall admissibility:", report.get("overall_admissibility_score"))


if __name__ == "__main__":
    main()

