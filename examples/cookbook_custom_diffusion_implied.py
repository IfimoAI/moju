#!/usr/bin/env python3
"""
Law-linked implied check for a custom law.

A user diffusion law ``c_t - D * c_laplacian = 0`` implies the diffusivity ``D = c_t / c_laplacian``.
Registering a :class:`~moju.monitor.law_implied_diagnostics.LawImpliedCheck` ties that implied value
to a registered constitutive model (Arrhenius ``D(T) = D0 * exp(-E_R / T)``), so every audit of the
law also scores whether the surrogate's dynamics agree with the material model — exactly like the
built-in Fourier -> thermal-diffusivity row.

Run::

    python examples/cookbook_custom_diffusion_implied.py
"""

from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp

from moju.monitor import LawImpliedCheck, LawSpec, ResidualEngine, audit, implied_by_projection
from moju.registry import (
    register_law_implied_check,
    register_model,
    unregister_law_implied_checks,
    unregister_model,
)

LAW = "species_diffusion"
MODEL = "arrhenius_diffusivity"
FLAT_KEY = f"constitutive/{MODEL}/law_{LAW}/implied_delta"


def species_diffusion(c_t, c_laplacian, D):
    return c_t - D * c_laplacian


def arrhenius_diffusivity(T, D0, E_R):
    return D0 * jnp.exp(-E_R / T)


def _state(n: int = 64, rate_error: float = 0.0) -> Dict[str, Any]:
    x = jnp.linspace(0.0, 1.0, n)
    T = 600.0 + 100.0 * x
    D = arrhenius_diffusivity(T, 1e-4, 2000.0)
    c_lap = 2.0 + jnp.cos(3.0 * x)
    c_t = D * c_lap * (1.0 + rate_error * jnp.sin(7.0 * x))
    return {"x": x, "T": T, "D": D, "c_t": c_t, "c_laplacian": c_lap, "D0": 1e-4, "E_R": 2000.0}


def _engine() -> ResidualEngine:
    return ResidualEngine(
        laws=[
            LawSpec(
                name=LAW,
                fn=species_diffusion,
                state_map={"c_t": "c_t", "c_laplacian": "c_laplacian", "D": "D"},
            )
        ]
    )


def main() -> Dict[str, Any]:
    register_model(MODEL, arrhenius_diffusivity, overwrite=True)
    register_law_implied_check(
        LAW,
        LawImpliedCheck(
            model=MODEL,
            output_key="D",
            state_map={"T": "T", "D0": "D0", "E_R": "E_R"},
            implied_maker=implied_by_projection("c_t", "c_laplacian"),
        ),
    )
    try:
        reports = {}
        for label, err in (("consistent", 0.0), ("rate off 3%", 0.03)):
            eng = _engine()
            eng.compute_residuals(_state(rate_error=err))
            reports[label] = audit(eng.log)
    finally:
        unregister_law_implied_checks(LAW)
        unregister_model(MODEL)

    for label, rep in reports.items():
        row = rep["per_key"][FLAT_KEY]
        print(f"{label:>12}: {FLAT_KEY} -> {row['admissibility_score']:.3f} ({row['admissibility_level']})")
    return {"flat_key": FLAT_KEY, "reports": reports}


if __name__ == "__main__":
    main()
