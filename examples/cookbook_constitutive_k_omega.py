#!/usr/bin/env python3
"""
Cookbook: constitutive **ref_delta** for ``Models.k_omega_nu_t`` (Path B).

Synthetic 1D **k** and **ω**; ``omega0`` in ``ResidualEngine.constants``.
With matching ``state_ref``, ``constitutive/k_omega_nu_t/ref_delta`` is ~0.
"""

from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp

from moju.monitor import AuditSpec, MonitorConfig, ResidualEngine, audit
from moju.piratio.models import Models


def main() -> Dict[str, Any]:
    omega0 = jnp.array(1e-12)
    x = jnp.linspace(0.0, 1.0, 48)
    k = 0.3 + 0.2 * x
    omega = 0.5 + 0.3 * x
    nu_t = Models.k_omega_nu_t(k, omega, omega0)

    state_pred = {
        "k": k,
        "omega": omega,
        "nu_t": nu_t,
    }

    cfg = MonitorConfig(
        constants={"omega0": omega0},
        laws=[],
        constitutive_audit=[
            AuditSpec(
                name="k_omega_nu_t",
                output_key="nu_t",
                state_map={
                    "k": "k",
                    "omega": "omega",
                    "omega0": "omega0",
                },
            )
        ],
    )
    engine = ResidualEngine(config=cfg)
    residuals = engine.compute_residuals(
        state_pred, state_ref=dict(state_pred), run_mode="eval"
    )
    report = audit(engine.log)
    flat_key = "constitutive/k_omega_nu_t/ref_delta"
    rms = engine.log[-1]["rms"][flat_key]
    return {"report": report, "ref_rms": float(rms), "residuals": residuals, "engine": engine, "flat_key": flat_key}


if __name__ == "__main__":
    out = main()
    print("ref_delta RMS (", out["flat_key"], "):", out["ref_rms"])
    pk = out["report"]["per_key"].get(out["flat_key"], {})
    if pk:
        print("Admissibility (per_key):", pk.get("admissibility_score"), pk.get("admissibility_level"))
