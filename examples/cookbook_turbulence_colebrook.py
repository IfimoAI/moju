#!/usr/bin/env python3
"""
Cookbook: constitutive chain audit for ``Models.colebrook_friction`` (Darcy f vs Re, ε/D).

Synthetic 1D sweep: **Re** varies along a notional streamwise coordinate, relative roughness
**epsilon_d** is uniform. The predicted **f** matches the Haaland-style correlation, and
**d_f_dx** matches **(∂f/∂Re)(dRe/dx)** from the chain rule. Useful as a **turbulent
pipe friction** consistency check for surrogates that output **f** and **Re**.

Path B with ``predicted_spatial=["Re"]``.
"""

from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp

from moju.monitor import AuditSpec, MonitorConfig, ResidualEngine, audit
from moju.piratio.models import Models


def main() -> Dict[str, Any]:
    Re = jnp.linspace(5_000.0, 2.0e6, 80)
    eps_scalar = jnp.array(1e-4)
    eps_rel = jnp.broadcast_to(eps_scalar, Re.shape)
    f = Models.colebrook_friction(Re, eps_rel)

    d_Re_dx = jnp.ones_like(Re)

    def f_wrt_re(r: jnp.ndarray) -> jnp.ndarray:
        return Models.colebrook_friction(r, eps_scalar)

    d_f_d_Re = jax.vmap(lambda r: jax.grad(lambda x: f_wrt_re(x))(r))(Re)
    d_f_dx = d_f_d_Re * d_Re_dx

    state_pred = {
        "Re": Re,
        "eps_rel": eps_rel,
        "f": f,
        "d_Re_dx": d_Re_dx,
        "d_f_dx": d_f_dx,
    }

    cfg = MonitorConfig(
        constants={},
        laws=[],
        constitutive_audit=[
            AuditSpec(
                name="colebrook_friction",
                output_key="f",
                state_map={"re": "Re", "epsilon_d": "eps_rel"},
                predicted_spatial=["Re"],
            )
        ],
    )
    engine = ResidualEngine(config=cfg)
    residuals = engine.compute_residuals(state_pred)
    report = audit(engine.log)
    chain_key = "constitutive/colebrook_friction/chain_dx"
    rms = engine.log[-1]["rms"][chain_key]
    return {"report": report, "chain_rms": rms, "residuals": residuals, "engine": engine}


if __name__ == "__main__":
    out = main()
    key = "constitutive/colebrook_friction/chain_dx"
    print("Chain-rule RMS (", key, "):", out["chain_rms"])
    pk = out["report"]["per_key"].get(key, {})
    if pk:
        print("Admissibility (per_key):", pk.get("admissibility_score"), pk.get("admissibility_level"))
