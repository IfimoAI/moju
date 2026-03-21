#!/usr/bin/env python3
"""
Cookbook: constitutive chain audit for ``Models.law_of_the_wall`` (log-law u⁺ vs y⁺).

Checks that the predicted **u⁺** field and **d_u_plus_dx** are consistent with the
analytic closure **u⁺ = 2.5 ln(y⁺) + 5** and the chain rule in a wall-normal direction
(``predicted_spatial`` on ``y_plus``). This is an **algebraic wall-function** diagnostic,
not a full RANS closure.

Uses Path B: supply ``state_pred`` with ``d_<key>_dx`` for varying keys. See
``moju.monitor.closure_registry`` for chain residual semantics.
"""

from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp

from moju.monitor import AuditSpec, MonitorConfig, ResidualEngine, audit
from moju.piratio.models import Models


def main() -> Dict[str, Any]:
    # Log-law region (y+ not too close to the wall; avoids buffer-layer physics).
    y_plus = jnp.linspace(50.0, 500.0, 64)
    u_plus = Models.law_of_the_wall(y_plus)
    d_y_plus_dx = jnp.ones_like(y_plus)
    # du+/dx = (du+/dy+) * dy+/dx ; u+ = 2.5*ln(y+) + 5  =>  du+/dy+ = 2.5/y+
    d_u_plus_dx = (2.5 / y_plus) * d_y_plus_dx

    state_pred = {
        "y_plus": y_plus,
        "u_plus": u_plus,
        "d_y_plus_dx": d_y_plus_dx,
        "d_u_plus_dx": d_u_plus_dx,
    }

    cfg = MonitorConfig(
        constants={},
        laws=[],
        constitutive_audit=[
            AuditSpec(
                name="law_of_the_wall",
                output_key="u_plus",
                state_map={"y_plus": "y_plus"},
                predicted_spatial=["y_plus"],
            )
        ],
    )
    engine = ResidualEngine(config=cfg)
    residuals = engine.compute_residuals(state_pred)
    report = audit(engine.log)
    chain_key = "constitutive/law_of_the_wall/chain_dx"
    rms = engine.log[-1]["rms"][chain_key]
    return {"report": report, "chain_rms": rms, "residuals": residuals, "engine": engine}


if __name__ == "__main__":
    out = main()
    key = "constitutive/law_of_the_wall/chain_dx"
    print("Chain-rule RMS (", key, "):", out["chain_rms"])
    pk = out["report"]["per_key"].get(key, {})
    if pk:
        print("Admissibility (per_key):", pk.get("admissibility_score"), pk.get("admissibility_level"))
