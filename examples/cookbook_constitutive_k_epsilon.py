#!/usr/bin/env python3
"""
Cookbook: constitutive chain audit for ``Models.k_epsilon_nu_t`` (RANS eddy viscosity).

Path B: synthetic 1D profiles for **k** and **ε**; ``C_mu`` and ``eps0`` in
``ResidualEngine.constants``. Analytic ``d_nu_t/dx`` from the chain rule matches
``constitutive/k_epsilon_nu_t/chain_dx`` (RMS ~ 0).

Full k–ε transport is **not** modeled here—only νₜ(k, ε). Add PDE residuals under
``Laws.*`` if needed.
"""

from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp

from moju.monitor import AuditSpec, MonitorConfig, ResidualEngine, audit
from moju.piratio.models import Models


def main() -> Dict[str, Any]:
    C_mu = jnp.array(0.09)
    eps0 = jnp.array(1e-12)
    x = jnp.linspace(0.0, 1.0, 48)
    k = 0.3 + 0.2 * x
    epsilon = 0.05 + 0.1 * x
    d_k_dx = jnp.full_like(k, 0.2)
    d_epsilon_dx = jnp.full_like(epsilon, 0.1)

    nu_t = Models.k_epsilon_nu_t(C_mu, k, epsilon, eps0)
    den = epsilon + eps0
    d_nu_dk = 2.0 * C_mu * k / den
    d_nu_depsilon = -C_mu * k**2 / (den**2)
    d_nu_t_dx = d_nu_dk * d_k_dx + d_nu_depsilon * d_epsilon_dx

    state_pred = {
        "k": k,
        "epsilon": epsilon,
        "nu_t": nu_t,
        "d_k_dx": d_k_dx,
        "d_epsilon_dx": d_epsilon_dx,
        "d_nu_t_dx": d_nu_t_dx,
    }

    cfg = MonitorConfig(
        constants={"C_mu": C_mu, "eps0": eps0},
        laws=[],
        constitutive_audit=[
            AuditSpec(
                name="k_epsilon_nu_t",
                output_key="nu_t",
                state_map={
                    "C_mu": "C_mu",
                    "k": "k",
                    "epsilon": "epsilon",
                    "eps0": "eps0",
                },
                predicted_spatial=["k", "epsilon"],
            )
        ],
    )
    engine = ResidualEngine(config=cfg)
    residuals = engine.compute_residuals(state_pred)
    report = audit(engine.log)
    chain_key = "constitutive/k_epsilon_nu_t/chain_dx"
    rms = engine.log[-1]["rms"][chain_key]
    return {"report": report, "chain_rms": rms, "residuals": residuals, "engine": engine}


if __name__ == "__main__":
    out = main()
    key = "constitutive/k_epsilon_nu_t/chain_dx"
    print("Chain-rule RMS (", key, "):", out["chain_rms"])
    pk = out["report"]["per_key"].get(key, {})
    if pk:
        print("Admissibility (per_key):", pk.get("admissibility_score"), pk.get("admissibility_level"))
