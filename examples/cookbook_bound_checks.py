#!/usr/bin/env python3
"""
Bound-check cookbook: inequality constraints scored like constitutive checks.

Four generic constraints on a 1-D conduction surrogate that predicts temperature ``T``, heat flux
``q``, viscosity ``mu``, density ``rho``, and a volume fraction ``phi_v``:

1. Clausius-Duhem: entropy production ``sigma = -q * dT/dx / T**2 >= 0``.
2. Heat flows down the temperature gradient: ``q * dT/dx <= 0``.
3. Positive material properties: ``mu > 0`` and ``rho > 0``.
4. Volume fraction in ``[0, 1]``.

Each check emits ``constitutive/bound/<name>/violation = (relu(lower - v) + relu(v - upper)) / scale``
(zero where the bound holds), scored worst-point on the 1e-2 gauge: a violation of 0.1 % of ``scale``
at a single point sits at the High-tier boundary.

Run::

    python examples/cookbook_bound_checks.py
"""

from __future__ import annotations

from typing import Any, Dict, List

import jax.numpy as jnp

from moju.monitor import BoundCheck, ResidualEngine, audit

K = 2.0  # W/(m K)


def surrogate_state(n: int = 101, defect: bool = False) -> Dict[str, Any]:
    x = jnp.linspace(0.0, 1.0, n)
    T = 300.0 + 50.0 * x
    T_x = jnp.full_like(x, 50.0)
    q = -K * T_x
    phi_v = 0.2 + 0.6 * x
    mu = jnp.full_like(x, 1e-3)
    rho = jnp.full_like(x, 1000.0)
    if defect:
        q = q.at[-5:].set(+K * 5.0)  # wrong-sign flux near the hot wall
        phi_v = phi_v + 0.25 * x**8  # overshoots 1 near x = 1
    return {"x": x, "T": T, "T_x": T_x, "q": q, "phi_v": phi_v, "mu": mu, "rho": rho}


def entropy_production(state, constants):
    return -state["q"] * state["T_x"] / state["T"] ** 2


def flux_alignment(state, constants):
    return state["q"] * state["T_x"]


BOUND_CHECKS: List[BoundCheck] = [
    # Typical sigma = k (dT/dx)^2 / T^2 ~ 2 * 2500 / 350^2 ~ 0.04 W/(m^3 K).
    BoundCheck(name="clausius_duhem", value_fn=entropy_production, lower=0.0, scale=0.04),
    # Typical |q dT/dx| = k (dT/dx)^2 = 5000 W/m^3 per K/m.
    BoundCheck(name="heat_flux_down_gradient", value_fn=flux_alignment, upper=0.0, scale=5000.0),
    BoundCheck(name="viscosity_positive", value_key="mu", lower=0.0, scale=1e-3),
    BoundCheck(name="density_positive", value_key="rho", lower=0.0, scale=1000.0),
    BoundCheck(name="volume_fraction", value_key="phi_v", lower=0.0, upper=1.0, scale=1.0),
]


def run(defect: bool) -> Dict[str, Any]:
    eng = ResidualEngine(bound_checks=BOUND_CHECKS)
    eng.compute_residuals(surrogate_state(defect=defect))
    return audit(eng.log)


def main() -> Dict[str, Any]:
    reports = {"clean": run(False), "defect": run(True)}
    for label, rep in reports.items():
        print(f"[{label}] constitutive score {rep['per_category']['constitutive']:.3f}")
        for b in BOUND_CHECKS:
            row = rep["per_key"][f"constitutive/bound/{b.name}/violation"]
            print(f"   {b.name:<24} {row['admissibility_score']:.3f}  {row['admissibility_level']}")
    return reports


if __name__ == "__main__":
    main()
