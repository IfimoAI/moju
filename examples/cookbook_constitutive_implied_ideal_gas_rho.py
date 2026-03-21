#!/usr/bin/env python3
"""
Cookbook: **implied_delta** constitutive audit (no ``state_ref``).

Compares ``Models.ideal_gas_rho(P, R, T)`` to an **implied** density carried under
``rho_implied`` in ``state_pred`` (e.g. from a surrogate, experiment, or alternate EOS path).

- Residual: ``constitutive/ideal_gas_rho/implied_delta`` = F(P,R,T) − ``rho_implied``.
- If ``rho_implied`` is missing, the closure is **omitted** (same policy as ``ref_delta`` /
  ``chain_*`` when prerequisites are missing).

Path B: user supplies ``state_pred`` only.
"""

from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp

from moju.monitor import AuditSpec, MonitorConfig, ResidualEngine, audit
from moju.piratio.models import Models


def main() -> Dict[str, Any]:
    P = jnp.array(101_325.0)
    R = jnp.array(287.0)
    T = jnp.linspace(250.0, 320.0, 40)
    rho = Models.ideal_gas_rho(P, R, T)
    # Scenario A: implied matches EOS → implied_delta ~ 0
    rho_implied = rho
    # Scenario B (uncomment to demo mismatch): rho_implied = rho * 1.02

    state_pred = {"P": P, "R": R, "T": T, "rho": rho, "rho_implied": rho_implied}

    cfg = MonitorConfig(
        laws=[],
        constitutive_audit=[
            AuditSpec(
                name="ideal_gas_rho",
                output_key="rho",
                state_map={"P": "P", "R": "R", "T": "T"},
                predicted_spatial=[],
                predicted_temporal=[],
                implied_value_key="rho_implied",
            )
        ],
    )
    engine = ResidualEngine(config=cfg)
    residuals = engine.compute_residuals(state_pred)
    report = audit(engine.log)
    flat_key = "constitutive/ideal_gas_rho/implied_delta"
    rms = engine.log[-1]["rms"][flat_key]
    return {
        "report": report,
        "implied_rms": rms,
        "residuals": residuals,
        "engine": engine,
        "flat_key": flat_key,
    }


if __name__ == "__main__":
    out = main()
    print("implied_delta RMS (", out["flat_key"], "):", out["implied_rms"])
    pk = out["report"]["per_key"].get(out["flat_key"], {})
    if pk:
        print("Admissibility (per_key):", pk.get("admissibility_score"), pk.get("admissibility_level"))
