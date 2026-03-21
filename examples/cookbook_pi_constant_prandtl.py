#!/usr/bin/env python3
"""
Cookbook: π-constant (Path A) audit for the Prandtl number ``Groups.pr``.

Same pattern as ``cookbook_pi_constant_reynolds.py``: built-in recipe scales ``mu`` and
``k`` by the same stretch ``c > 1`` so ``Pr = cp*mu/k`` is unchanged; a second forward
compares ``invariance_compare_keys``. Path A with non-``None`` ``model`` / ``params`` /
``collocation`` placeholders. See ``moju.monitor.pi_constant_recipes`` and
``list_pi_constant_group_names()`` for all supported groups.
"""

from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp

from moju.monitor import AuditSpec, MonitorConfig, ResidualEngine, audit


def state_builder(model: Any, params: Any, collocation: Any, constants: Dict[str, Any]):
    """Stub surrogate: prediction equals Pr from material constants."""
    mu = constants["mu"]
    cp = constants["cp"]
    k = constants["k"]
    Pr_pred = cp * mu / k
    return {"Pr_pred": Pr_pred}


def main() -> Dict[str, Any]:
    constants = {
        "mu": jnp.array(1.8e-5),
        "cp": jnp.array(1005.0),
        "k": jnp.array(0.026),
    }

    cfg = MonitorConfig(
        constants=constants,
        scaling_audit=[
            AuditSpec(
                name="pr",
                output_key="Pr",
                state_map={"mu": "mu", "cp": "cp", "k": "k"},
                invariance_pi_constant=True,
                invariance_compare_keys=["Pr_pred"],
                invariance_scale_c=10.0,
            )
        ],
        state_builder=state_builder,
    )
    engine = ResidualEngine(config=cfg)
    engine.compute_residuals(None, model=0, params=0, collocation={})
    report = audit(engine.log)
    pi_key = "scaling/pr/pi_constant"
    return {
        "report": report,
        "pi_rms": engine.log[-1]["rms"][pi_key],
        "engine": engine,
    }


if __name__ == "__main__":
    out = main()
    pi_key = "scaling/pr/pi_constant"
    print("π-constant RMS (scaling/pr/pi_constant):", out["pi_rms"])
    pk = out["report"]["per_key"].get(pi_key, {})
    if pk:
        print("Admissibility (per_key):", pk.get("admissibility_score"), pk.get("admissibility_level"))
    print("Overall admissibility:", out["report"]["overall_admissibility_score"])
