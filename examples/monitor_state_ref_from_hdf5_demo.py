"""
Demo template: ingest an HDF5 snapshot into state_ref.

Run:
  pip install moju[ref_hdf5]
  python examples/monitor_state_ref_from_hdf5_demo.py /path/to/data.h5
"""

import sys

import jax.numpy as jnp

from moju.monitor import MonitorConfig, ResidualEngine
from moju.monitor.state_ref import from_hdf5


def main(h5_path: str):
    cfg = MonitorConfig(laws=[])
    engine = ResidualEngine(config=cfg)

    # Map moju keys to dataset paths inside the HDF5 file.
    # Example var_map={"T": "fields/T"}.
    state_ref = from_hdf5(h5_path, var_map={"phi": "phi"}, strict=False)

    state_pred = {"phi": jnp.asarray(0.0)}
    residuals = engine.compute_residuals(state_pred, state_ref=state_ref)
    print("Computed keys:", sorted(residuals.keys()))
    if "data" in residuals:
        print("Data residual keys:", sorted(residuals["data"].keys()))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python examples/monitor_state_ref_from_hdf5_demo.py /path/to/data.h5")
    main(sys.argv[1])

