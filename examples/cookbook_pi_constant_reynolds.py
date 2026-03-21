#!/usr/bin/env python3
"""
Cookbook: π-constant (Path A) audit for the Reynolds number ``Groups.re``.

Moju scales selected entries in ``ResidualEngine.constants`` by a built-in recipe
(see ``moju.monitor.pi_constant_recipes`` / ``list_pi_constant_group_names()``) so
``Re = rho*u*L/mu`` stays numerically fixed, runs a second ``state_builder`` forward,
and forms a residual on ``invariance_compare_keys``. The flat RMS key is
``scaling/re/pi_constant`` and feeds the same admissibility pipeline as other scaling
metrics.

Requirements: recipe arguments touched by the recipe must live in ``constants``;
use Path A (``compute_residuals`` without passing ``state_pred``, with non-``None``
``model`` / ``params`` / ``collocation`` placeholders as required by the engine).
Chain-rule closures (``chain_dx`` / ``chain_dt``) are optional for this check.
"""

from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp

from moju.monitor import AuditSpec, MonitorConfig, ResidualEngine, audit


def state_builder(model: Any, params: Any, collocation: Any, constants: Dict[str, Any]):
    """Stub surrogate: prediction equals the Reynolds number from constants."""
    rho = constants["rho"]
    u = constants["u"]
    L = constants["L"]
    mu = constants["mu"]
    Re_pred = rho * u * L / mu
    return {"Re_pred": Re_pred}


def main() -> Dict[str, Any]:
    constants = {
        "u": jnp.array(2.0),
        "L": jnp.array(0.05),
        "rho": jnp.array(1.2),
        "mu": jnp.array(1.8e-5),
    }

    cfg = MonitorConfig(
        constants=constants,
        scaling_audit=[
            AuditSpec(
                name="re",
                output_key="Re",
                state_map={"u": "u", "L": "L", "rho": "rho", "mu": "mu"},
                invariance_pi_constant=True,
                invariance_compare_keys=["Re_pred"],
                invariance_scale_c=10.0,
            )
        ],
        state_builder=state_builder,
    )
    engine = ResidualEngine(config=cfg)
    # Path A requires non-None placeholders; this stub ignores model/params/collocation.
    engine.compute_residuals(None, model=0, params=0, collocation={})
    report = audit(engine.log)
    pi_key = "scaling/re/pi_constant"
    return {
        "report": report,
        "pi_rms": engine.log[-1]["rms"][pi_key],
        "engine": engine,
    }


if __name__ == "__main__":
    out = main()
    pi_key = "scaling/re/pi_constant"
    print("π-constant RMS (scaling/re/pi_constant):", out["pi_rms"])
    pk = out["report"]["per_key"].get(pi_key, {})
    if pk:
        print("Admissibility (per_key):", pk.get("admissibility_score"), pk.get("admissibility_level"))
    print("Overall admissibility:", out["report"]["overall_admissibility_score"])

    # If ``state_builder`` returned something not Re-invariant under the recipe (e.g. only
    # ``constants["L"]``), this RMS would generally be non-zero even when Re is unchanged.
