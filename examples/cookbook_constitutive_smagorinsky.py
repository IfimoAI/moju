#!/usr/bin/env python3
"""
Cookbook: constitutive chain audit for ``Models.smagorinsky_nu_t`` (LES eddy viscosity).

Template for **differentiable** auditing of a simple SGS closure:
``nu_t = (Cs * Delta)^2 |S|``. ``Cs`` and ``Delta`` live in ``ResidualEngine.constants``;
the resolved strain magnitude ``S`` varies along a 1D line so ``chain_dx`` checks
``d_nu_t/dx`` against ``jax.grad`` of the closure w.r.t. ``S``.

For full LES, replace the synthetic ``S`` field with |S| from your velocity gradients
and wire the same ``AuditSpec`` pattern.
"""

from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp

from moju.monitor import AuditSpec, MonitorConfig, ResidualEngine, audit
from moju.piratio.models import Models


def main() -> Dict[str, Any]:
    Cs = jnp.array(0.17)
    Delta = jnp.array(0.01)
    S = jnp.linspace(0.1, 50.0, 48)
    nu_t = Models.smagorinsky_nu_t(Cs, Delta, S)
    d_S_dx = jnp.ones_like(S)
    d_nu_t_dx = (Cs * Delta) ** 2 * d_S_dx

    state_pred = {
        "S": S,
        "nu_t": nu_t,
        "d_S_dx": d_S_dx,
        "d_nu_t_dx": d_nu_t_dx,
    }

    cfg = MonitorConfig(
        constants={"Cs": Cs, "Delta": Delta},
        laws=[],
        constitutive_audit=[
            AuditSpec(
                name="smagorinsky_nu_t",
                output_key="nu_t",
                state_map={
                    "Cs": "Cs",
                    "Delta": "Delta",
                    "strain_rate_magnitude": "S",
                },
                predicted_spatial=["S"],
            )
        ],
    )
    engine = ResidualEngine(config=cfg)
    residuals = engine.compute_residuals(state_pred)
    report = audit(engine.log)
    chain_key = "constitutive/smagorinsky_nu_t/chain_dx"
    rms = engine.log[-1]["rms"][chain_key]
    return {"report": report, "chain_rms": rms, "residuals": residuals, "engine": engine}


if __name__ == "__main__":
    out = main()
    key = "constitutive/smagorinsky_nu_t/chain_dx"
    print("Chain-rule RMS (", key, "):", out["chain_rms"])
    pk = out["report"]["per_key"].get(key, {})
    if pk:
        print("Admissibility (per_key):", pk.get("admissibility_score"), pk.get("admissibility_level"))
