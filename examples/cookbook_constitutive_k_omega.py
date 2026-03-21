#!/usr/bin/env python3
"""
Cookbook: constitutive chain audit for ``Models.k_omega_nu_t`` (k–ω eddy viscosity).

Path B: synthetic 1D **k** and **ω**; ``omega0`` in ``ResidualEngine.constants``.
Analytic ``d_nu_t/dx`` so ``constitutive/k_omega_nu_t/chain_dx`` RMS ~ 0.

Transport equations for k and ω are **not** included—only νₜ(k, ω).
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
    d_k_dx = jnp.full_like(k, 0.2)
    d_omega_dx = jnp.full_like(omega, 0.3)

    nu_t = Models.k_omega_nu_t(k, omega, omega0)
    den = omega + omega0
    d_nu_dk = 1.0 / den
    d_nu_domega = -k / (den**2)
    d_nu_t_dx = d_nu_dk * d_k_dx + d_nu_domega * d_omega_dx

    state_pred = {
        "k": k,
        "omega": omega,
        "nu_t": nu_t,
        "d_k_dx": d_k_dx,
        "d_omega_dx": d_omega_dx,
        "d_nu_t_dx": d_nu_t_dx,
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
                predicted_spatial=["k", "omega"],
            )
        ],
    )
    engine = ResidualEngine(config=cfg)
    residuals = engine.compute_residuals(state_pred)
    report = audit(engine.log)
    chain_key = "constitutive/k_omega_nu_t/chain_dx"
    rms = engine.log[-1]["rms"][chain_key]
    return {"report": report, "chain_rms": rms, "residuals": residuals, "engine": engine}


if __name__ == "__main__":
    out = main()
    key = "constitutive/k_omega_nu_t/chain_dx"
    print("Chain-rule RMS (", key, "):", out["chain_rms"])
    pk = out["report"]["per_key"].get(key, {})
    if pk:
        print("Admissibility (per_key):", pk.get("admissibility_score"), pk.get("admissibility_level"))
