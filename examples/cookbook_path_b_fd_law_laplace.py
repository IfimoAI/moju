#!/usr/bin/env python3
"""
Path B cookbook: auto finite-difference fill for **governing law** inputs.

``compute_residuals(..., auto_path_b_derivatives=..., fill_law_fd=True)`` fills missing
**registered** ``Laws.*`` arguments (e.g. ``phi_laplacian`` for ``laplace_equation``)
from primitives on the same grid. ``fill_law_fd`` requires ``auto_path_b_derivatives``
so a shared ``PathBGridConfig`` applies.

This script uses a linear ``phi(x)``, so ``d²phi/dx² = 0`` in the interior and the
Laplace residual is small up to FD error. See ``list_law_fd_supported_laws()`` for
other built-in laws with FD recipes.

Run::

    python examples/cookbook_path_b_fd_law_laplace.py
"""

from __future__ import annotations

import jax.numpy as jnp

from moju.monitor import PathBGridConfig, ResidualEngine, audit, list_law_fd_supported_laws


def main() -> None:
    supported = list_law_fd_supported_laws()
    print("Sample laws with FD recipes (first 8):", supported[:8])

    engine = ResidualEngine(
        laws=[
            {
                "name": "laplace_equation",
                "state_map": {"phi_laplacian": "phi_laplacian"},
            }
        ],
    )

    grid = PathBGridConfig(layout="meshgrid", spatial_dimension=1, steady=True)
    x = jnp.linspace(0.0, 1.0, 48)
    phi = x  # Laplacian 0 (interior FD)

    state_pred = {"phi": phi, "x": x}
    # No phi_laplacian — filled by law FD

    residuals = engine.compute_residuals(
        state_pred,
        auto_path_b_derivatives=grid,
        fill_law_fd=True,
        log_to_python=True,
    )
    lap = residuals["laws"]["laplace_equation"]
    print("laplace_equation residual max |r| (interior):", float(jnp.max(jnp.abs(lap[1:-1]))))

    if engine.log and "inferred" in engine.log[-1]:
        inf = engine.log[-1]["inferred"]
        fd_msgs = [s for s in inf if "path_b" in s or "law_fd" in s]
        if fd_msgs:
            print("Log inferred (FD, first 4):", fd_msgs[:4])

    rep = audit(engine.log)
    print("Overall admissibility:", rep["overall_admissibility_score"])
    lk = "laws/laplace_equation"
    if lk in rep.get("per_key", {}):
        print(lk, "RMS:", rep["per_key"][lk].get("rms"))


if __name__ == "__main__":
    main()
