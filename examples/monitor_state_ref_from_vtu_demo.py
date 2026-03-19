"""
Demo template: ingest a VTU snapshot into state_ref using meshio.

Run:
  pip install moju[ref_vtk]
  python examples/monitor_state_ref_from_vtu_demo.py /path/to/snapshot.vtu
"""

import sys

import jax.numpy as jnp

from moju.monitor import MonitorConfig, ResidualEngine
from moju.monitor.state_ref import from_vtu


def main(vtu_path: str):
    # Minimal engine: compute a data residual on a shared key.
    cfg = MonitorConfig(laws=[])
    engine = ResidualEngine(config=cfg)

    # Map a VTU field name to a moju state_ref key.
    # Example: var_map={"T": "Temperature"} if your VTU point_data has "Temperature".
    state_ref = from_vtu(vtu_path, var_map={"phi": "phi"}, cell_or_point="point", strict=False)

    # Dummy prediction (same key must exist to produce a data residual).
    state_pred = {"phi": jnp.asarray(0.0)}
    residuals = engine.compute_residuals(state_pred, state_ref=state_ref)
    print("Computed keys:", sorted(residuals.keys()))
    if "data" in residuals:
        print("Data residual keys:", sorted(residuals["data"].keys()))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python examples/monitor_state_ref_from_vtu_demo.py /path/to/snapshot.vtu")
    main(sys.argv[1])

