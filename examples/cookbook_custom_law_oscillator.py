#!/usr/bin/env python3
"""
Custom-law cookbook: damped oscillator ``m q_tt + c q_t + k q = 0``.

Shows the three hooks a custom governing law needs to be scored like a built-in:

1. **Scale recipe** (``law_scale_mode="auto"``): ``scale_k = max(rms(m q_tt), rms(c q_t), rms(k q))``
   via :func:`moju.registry.register_law_scale_recipe` (or a spec ``"scale_recipe"`` key).
2. **Time-scale hint** (``state_units="dimensional"``): ``t_ref = sqrt(m / k)`` via the spec
   ``"time_scale"`` key (or :func:`moju.registry.register_law_time_scale`).
3. **Key declarations**: ``q`` is a length, ``q_t`` its first and ``q_tt`` its second time derivative,
   so SI inputs are nondimensionalised consistently.

Run::

    python examples/cookbook_custom_law_oscillator.py
"""

from __future__ import annotations

import math
from typing import Any, Dict

import jax.numpy as jnp

from moju.monitor import KeyDeclaration, LawSpec, ResidualEngine, audit
from moju.monitor.law_scale_recipes import law_arg_value, term_max_rms
from moju.registry import register_law_scale_recipe, unregister_law_scale_recipe

M, C, K = 2.0, 0.4, 50.0  # kg, N s/m, N/m
AMPLITUDE = 0.05  # m


def oscillator_si(q_tt, q_t, q, m, c, k):
    """Dimensional residual [N]."""
    return m * q_tt + c * q_t + k * q


def oscillator_nd(q_tt, q_t, q, m, c, k):
    """Nondimensional residual with t* = t / sqrt(m/k), q* = q / L_ref: q*'' + 2 zeta q*' + q*."""
    return q_tt + (c / jnp.sqrt(m * k)) * q_t + q


def oscillator_si_recipe(merged, constants, law_spec, nondim_scales):
    v = lambda a: law_arg_value(merged, constants, law_spec, a)  # noqa: E731
    return term_max_rms(v("m") * v("q_tt"), v("c") * v("q_t"), v("k") * v("q"))


def oscillator_nd_recipe(merged, constants, law_spec, nondim_scales):
    v = lambda a: law_arg_value(merged, constants, law_spec, a)  # noqa: E731
    zeta2 = v("c") / jnp.sqrt(v("m") * v("k"))
    return term_max_rms(v("q_tt"), zeta2 * v("q_t"), v("q"))


def trajectory(n: int = 400, t_end: float = 3.0) -> Dict[str, Any]:
    """Exact underdamped solution and its time derivatives (SI)."""
    wn = math.sqrt(K / M)
    zeta = C / (2.0 * math.sqrt(M * K))
    wd = wn * math.sqrt(1.0 - zeta**2)
    t = jnp.linspace(0.0, t_end, n)
    e = jnp.exp(-zeta * wn * t)
    q = AMPLITUDE * e * jnp.cos(wd * t)
    q_t = AMPLITUDE * e * (-zeta * wn * jnp.cos(wd * t) - wd * jnp.sin(wd * t))
    q_tt = -2.0 * zeta * wn * q_t - wn**2 * q
    return {"t": t, "q": q, "q_t": q_t, "q_tt": q_tt}


_STATE_MAP = {a: a for a in ("q_tt", "q_t", "q", "m", "c", "k")}
CONSTANTS = {"m": M, "c": C, "k": K, "L": AMPLITUDE}


def si_engine() -> ResidualEngine:
    """Residual evaluated directly in SI; scale_k from the registered recipe."""
    return ResidualEngine(
        constants=CONSTANTS,
        laws=[LawSpec(name="damped_oscillator", fn=oscillator_si, state_map=_STATE_MAP)],
        law_implied_audits=False,
    )


def dimensional_engine() -> ResidualEngine:
    """SI input, nondimensionalised by Moju with t_ref = sqrt(m/k) and declared derivative keys."""
    return ResidualEngine(
        constants=CONSTANTS,
        laws=[
            LawSpec(
                name="damped_oscillator_nd",
                fn=oscillator_nd,
                state_map=_STATE_MAP,
                time_scale=lambda constants, scales: math.sqrt(constants["m"] / constants["k"]),
                scale_recipe=oscillator_nd_recipe,
            )
        ],
        law_implied_audits=False,
        state_units="dimensional",
        state_declarations={
            "q": KeyDeclaration("length"),
            "q_t": KeyDeclaration("length", time_order=1),
            "q_tt": KeyDeclaration("length", time_order=2),
        },
    )


def main() -> Dict[str, Any]:
    state = trajectory()
    register_law_scale_recipe("damped_oscillator", oscillator_si_recipe, overwrite=True)
    try:
        eng_si = si_engine()
        eng_si.compute_residuals(state)
    finally:
        unregister_law_scale_recipe("damped_oscillator")
    rep_si = audit(eng_si.log)

    eng_nd = dimensional_engine()
    eng_nd.compute_residuals(state)
    rep_nd = audit(eng_nd.log)

    nd = eng_nd.log[-1]["nondim_scales"]
    print(f"SI law: scale_k = {eng_si.log[-1]['scale']['laws/damped_oscillator']:.4g} N, "
          f"score {rep_si['overall_admissibility_score']:.3f}")
    print(f"ND law: time_scale = {nd['time_scale']}, t_ref = {nd['t_ref_override']:.4f} s, "
          f"score {rep_nd['overall_admissibility_score']:.3f}")
    return {"si": (eng_si, rep_si), "nd": (eng_nd, rep_nd), "state": state}


if __name__ == "__main__":
    main()
