"""
Demo template: ingest an OpenFOAM snapshot into state_ref.

Many OpenFOAM workflows export to VTK/VTU first (e.g. via foamToVTK) and then use `from_vtu`.
This script shows both options conceptually.

Run:
  pip install moju[ref_foam]
  python examples/monitor_state_ref_from_openfoam_demo.py /path/to/case_or_export
"""

import sys

import jax.numpy as jnp

from moju.monitor import MonitorConfig, ResidualEngine
from moju.monitor.state_ref import from_openfoam


def main(case_path: str):
    cfg = MonitorConfig(laws=[])
    engine = ResidualEngine(config=cfg)

    # For OpenFOAM, field names depend on what’s stored/readable by the reader.
    # Start with strict=False and inspect returned keys.
    state_ref = from_openfoam(case_path, var_map={"U": "U", "p": "p"}, strict=False)

    state_pred = {"p": jnp.asarray(0.0)}
    residuals = engine.compute_residuals(state_pred, state_ref=state_ref)
    print("Computed keys:", sorted(residuals.keys()))
    if "data" in residuals:
        print("Data residual keys:", sorted(residuals["data"].keys()))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python examples/monitor_state_ref_from_openfoam_demo.py /path/to/case_or_export")
    main(sys.argv[1])

