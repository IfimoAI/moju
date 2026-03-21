#!/usr/bin/env python3
"""
Cookbook: **implied_delta** with ``implied_fn`` (Python callable).

For ``Models.power_law_mu``, shear stress for the power-law fluid is
``tau = K * gamma_dot^n``. An **effective** apparent viscosity can be defined as
``mu_effective = tau / gamma_dot = K * gamma_dot^(n-1)``, matching ``power_law_mu``.

Here ``implied_fn(merged_state, constants)`` returns ``tau_effective / gamma_dot``.
If your network predicts a different ``tau_effective`` than the consistent stress,
``implied_delta`` becomes non-zero.

``implied_fn`` is **not** JSON-serializable; ``MonitorConfig.to_dict()`` omits it.
``ResidualEngine`` restores it from the ``AuditSpec`` object when loading ``MonitorConfig``.

Path B: synthetic 1D ``gamma_dot``; no ``state_ref``; no ``chain_*`` unless you add
``predicted_spatial``.
"""

from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp

from moju.monitor import AuditSpec, MonitorConfig, ResidualEngine, audit
from moju.piratio.models import Models


def _implied_mu_from_tau(state: Dict[str, Any], _constants: Dict[str, Any]):
    return state["tau_effective"] / state["gamma_dot"]


def main() -> Dict[str, Any]:
    gamma_dot = jnp.linspace(5.0, 80.0, 36)
    K = jnp.array(0.08)
    n = jnp.array(0.75)
    mu = Models.power_law_mu(gamma_dot, K, n)
    tau_consistent = K * gamma_dot**n
    # Match power law → zero residual
    tau_effective = tau_consistent
    # Mismatch example: tau_effective = tau_consistent * 1.03

    state_pred = {
        "gamma_dot": gamma_dot,
        "K": K,
        "n": n,
        "mu": mu,
        "tau_effective": tau_effective,
    }

    cfg = MonitorConfig(
        laws=[],
        constitutive_audit=[
            AuditSpec(
                name="power_law_mu",
                output_key="mu",
                state_map={"gamma_dot": "gamma_dot", "K": "K", "n": "n"},
                predicted_spatial=[],
                predicted_temporal=[],
                implied_fn=_implied_mu_from_tau,
            )
        ],
    )
    # ResidualEngine merges implied_fn onto the spec dict (omitted from JSON to_dict).
    engine = ResidualEngine(config=cfg)
    residuals = engine.compute_residuals(state_pred)
    report = audit(engine.log)
    flat_key = "constitutive/power_law_mu/implied_delta"
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
