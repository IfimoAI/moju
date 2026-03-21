#!/usr/bin/env python3
"""
Path B cookbook: auto finite-difference fill for **audit chain** derivatives.

``compute_residuals(..., auto_path_b_derivatives=PathBGridConfig(...))`` fills missing
``d_*_dx`` / ``d_*_dy`` / … keys required by ``constitutive_audit`` / ``scaling_audit``
from structured coordinates (here 1D ``x``) and the primitive fields.

We audit ``Groups.pe``: ``Pe = Re * Pr`` with ``Pr`` constant and ``Re(x)`` varying.
The chain closure needs ``d_Re_dx`` and ``d_Pe_dx``; we omit them from ``state_pred``
and let moju fill them via FD.

Run::

    python examples/cookbook_path_b_fd_audit_pe.py
"""

from __future__ import annotations

import jax.numpy as jnp

from moju.monitor import PathBGridConfig, ResidualEngine, audit, fill_path_b_derivatives


def main() -> None:
    engine = ResidualEngine(
        laws=[],
        scaling_audit=[
            {
                "name": "pe",
                "output_key": "Pe",
                "state_map": {"re": "Re", "pr": "Pr"},
                "predicted_spatial": ["Re"],
            }
        ],
    )

    grid = PathBGridConfig(layout="meshgrid", spatial_dimension=1, steady=True)
    x = jnp.linspace(0.0, 1.0, 40)
    Re = x**2
    Pr = jnp.array(0.75)
    Pe = Re * Pr

    # Only primitives + coordinates — no d_Re_dx / d_Pe_dx
    state_pred = {"Re": Re, "Pe": Pe, "Pr": Pr, "x": x}

    # Optional: same fill without running the full engine
    filled, warnings = fill_path_b_derivatives(
        dict(state_pred),
        scaling_audit=engine.scaling_audit,
        grid=grid,
        fill_law_recipes=False,
    )
    print("fill_path_b_derivatives warnings:", warnings[:3], "..." if len(warnings) > 3 else "")
    print("Filled keys include d_Re_dx:", "d_Re_dx" in filled, "| d_Pe_dx:", "d_Pe_dx" in filled)

    residuals = engine.compute_residuals(
        state_pred,
        auto_path_b_derivatives=grid,
        log_to_python=True,
    )
    chain = residuals["scaling"]["pe/chain_dx"]
    print("pe/chain_dx (max abs, interior):", float(jnp.max(jnp.abs(chain[1:-1]))))

    rep = audit(engine.log)
    print("Overall admissibility:", rep["overall_admissibility_score"])
    pe_key = "scaling/pe/chain_dx"
    if pe_key in rep.get("per_key", {}):
        print(pe_key, "RMS:", rep["per_key"][pe_key].get("rms"))


if __name__ == "__main__":
    main()
