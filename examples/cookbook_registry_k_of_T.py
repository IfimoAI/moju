#!/usr/bin/env python3
"""
Registry cookbook: a user constitutive model, temperature-dependent conductivity
``k(T) = k0 * (1 + beta * (T - T_lin))``, audited by name inside a Fourier-conduction audit.

``moju.registry.register_model`` makes ``AuditSpec(name="k_of_T", ...)`` resolve exactly like a
built-in ``Models.*`` function. The surrogate's predicted ``k`` field is checked against the model
through ``implied_delta`` (worst-point, nondimensional).

Run::

    python examples/cookbook_registry_k_of_T.py
"""

from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp

from moju.monitor import AuditSpec, LawSpec, ResidualEngine, audit
from moju.registry import register_model, unregister_model

MODEL_NAME = "k_of_T"


def k_of_T(T, k0, beta, T_lin):
    """Linear-in-temperature thermal conductivity [W/(m K)]."""
    return k0 * (1.0 + beta * (T - T_lin))


def _state(n: int = 64, k_noise: float = 0.0) -> Dict[str, Any]:
    L, rho, cp = 0.02, 2700.0, 900.0
    k0, beta, T_lin = 200.0, 1e-3, 300.0
    x = jnp.linspace(0.0, L, n)
    T = 300.0 + 20.0 * (1.0 - x / L) ** 2
    k = k_of_T(T, k0, beta, T_lin) * (1.0 + k_noise * jnp.sin(40.0 * x / L))
    alpha = k / (rho * cp)
    T_lap = jnp.full_like(x, 40.0 / L**2)
    t = jnp.full_like(x, 5.0)
    return {
        "x": x,
        "t": t,
        "T": T,
        "T_t": alpha * T_lap,
        "T_laplacian": T_lap,
        "k": k,
        "rho": jnp.full_like(x, rho),
        "cp": jnp.full_like(x, cp),
        "alpha": alpha,
        "L": jnp.full_like(x, L),
        "k0": k0,
        "beta": beta,
        "T_lin": T_lin,
    }


def _engine() -> ResidualEngine:
    return ResidualEngine(
        laws=[
            LawSpec(
                name="fourier_conduction",
                state_map={a: a for a in ("T_t", "T_laplacian", "fo", "t", "L")},
            )
        ],
        groups=[{"name": "fo", "output_key": "fo", "state_map": {"alpha": "alpha", "t": "t", "L": "L"}}],
        constitutive_audit=[
            AuditSpec(
                name=MODEL_NAME,
                output_key="k",
                state_map={"T": "T", "k0": "k0", "beta": "beta", "T_lin": "T_lin"},
                implied_value_key="k",
            )
        ],
        law_implied_audits=False,
    )


def main() -> Dict[str, Any]:
    register_model(MODEL_NAME, k_of_T, overwrite=True)
    try:
        flat_key = f"constitutive/{MODEL_NAME}/implied_delta"
        good = _engine()
        good.compute_residuals(_state())
        report_good = audit(good.log)

        drifted = _engine()
        drifted.compute_residuals(_state(k_noise=0.05))
        report_drifted = audit(drifted.log)
    finally:
        unregister_model(MODEL_NAME)

    for label, rep in (("consistent k", report_good), ("k drifts 5%", report_drifted)):
        row = rep["per_key"][flat_key]
        print(f"{label:>13}: {flat_key} -> {row['admissibility_score']:.3f} ({row['admissibility_level']})")
    return {"flat_key": flat_key, "report": report_good, "report_drifted": report_drifted}


if __name__ == "__main__":
    main()
