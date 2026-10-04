#!/usr/bin/env python3
"""
Register a governing law by name: 1D heat residual ``T_t - alpha * T_xx``.

``moju.registry.register_law`` makes ``{"name": "heat_1d"}`` resolve like a built-in ``Laws.*``
entry. ``required_keys`` is the default identity ``state_map``. The log and the audit report record
``law_sources`` as ``registered``.

Run::

    python examples/cookbook_register_law_heat_1d.py
"""

from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp

from moju.monitor import ResidualEngine, audit
from moju.registry import register_law, unregister_law

LAW = "heat_1d"


def heat_1d(T_t, T_xx, alpha):
    """1D heat residual. Zero when ``T_t = alpha * T_xx``."""
    return T_t - alpha * T_xx


def heat_scale(merged, constants, law_spec, nondim_scales):
    from moju.monitor.law_scale_recipes import law_arg_value, term_max_rms

    rate = law_arg_value(merged, constants, law_spec, "T_t")
    diffusion = law_arg_value(merged, constants, law_spec, "alpha") * law_arg_value(
        merged, constants, law_spec, "T_xx"
    )
    return term_max_rms(rate, diffusion)


def _state(n: int = 64, alpha: float = 0.2) -> Dict[str, Any]:
    x = jnp.linspace(0.0, 1.0, n)
    temperature = jnp.sin(jnp.pi * x)
    curvature = -(jnp.pi**2) * temperature
    return {"x": x, "T_t": alpha * curvature, "T_xx": curvature, "alpha": alpha}


def main() -> Dict[str, Any]:
    register_law(
        LAW,
        heat_1d,
        required_keys=("T_t", "T_xx", "alpha"),
        scale_recipe=heat_scale,
        time_scale="fourier",
        overwrite=True,
    )
    try:
        engine = ResidualEngine(laws=[{"name": LAW}], law_implied_audits=False)
        engine.compute_residuals(_state())
        report = audit(engine.log)
    finally:
        unregister_law(LAW)
    source = report["law_sources"][LAW]
    row = report["per_key"][f"laws/{LAW}"]
    print(f"{LAW}: source={source['source']}, score={row['admissibility_score']:.3f} ({row['admissibility_level']})")
    return {"report": report, "flat_key": f"laws/{LAW}"}


if __name__ == "__main__":
    main()
