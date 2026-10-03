#!/usr/bin/env python3
"""
Derivative provenance cookbook: heat-equation surrogate with autodiff derivatives.

A surrogate ``T(x, t)`` supplies ``T_t`` and ``T_laplacian`` from ``jax.grad`` (Path A
``state_builder``). With ``derivatives="supplied_only"`` Moju never computes a derivative itself:
the report labels both inputs ``supplied``, and a missing derivative raises
:class:`~moju.monitor.derivative_provenance.MissingSuppliedDerivativeError` instead of being filled.
For contrast, the Path B run lets Moju fill ``T_laplacian`` by finite differences, and the report
says so.

Run::

    python examples/cookbook_supplied_derivatives_heat.py
"""

from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp

from moju.monitor import LawSpec, MissingSuppliedDerivativeError, PathBGridConfig, ResidualEngine, audit

T_SAMPLE = 0.05  # nondimensional time of the snapshot


def surrogate(params, x, t):
    """Exact nondimensional solution ``exp(-pi^2 t) sin(pi x)`` scaled by ``params``."""
    return params * jnp.exp(-(jnp.pi**2) * t) * jnp.sin(jnp.pi * x)


def state_builder(model, params, collocation, constants) -> Dict[str, Any]:
    x, t = collocation["x"], collocation["t"]
    T = jax.vmap(lambda xi: model(params, xi, t))(x)
    T_t = jax.vmap(jax.grad(lambda ti, xi: model(params, xi, ti)), in_axes=(None, 0))(t, x)
    T_xx = jax.vmap(jax.grad(jax.grad(lambda xi: model(params, xi, t))))(x)
    return {"x": x, "t": t, "T": T, "T_t": T_t, "T_laplacian": T_xx, "alpha": 1.0, "L": 1.0}


LAWS = [LawSpec(name="fourier_conduction", state_map={a: a for a in ("T_t", "T_laplacian", "fo", "t", "L")})]
GROUPS = [{"name": "fo", "output_key": "fo", "state_map": {"alpha": "alpha", "t": "t", "L": "L"}}]


def engine(derivatives: str = "supplied_only") -> ResidualEngine:
    return ResidualEngine(
        laws=LAWS,
        groups=GROUPS,
        state_builder=state_builder,
        law_implied_audits=False,
        derivatives=derivatives,
    )


def main() -> Dict[str, Any]:
    colloc = {"x": jnp.linspace(0.0, 1.0, 129), "t": jnp.asarray(T_SAMPLE)}

    eng = engine()
    eng.compute_residuals(model=surrogate, params=1.0, collocation=colloc)
    rep_supplied = audit(eng.log)

    path_b = state_builder(surrogate, 1.0, colloc, {})
    missing = dict(path_b)
    del missing["T_t"]
    try:
        engine().compute_residuals(missing)
        raised = None
    except MissingSuppliedDerivativeError as err:
        raised = err

    fd_state = dict(path_b)
    del fd_state["T_laplacian"]
    eng_fd = engine("auto")
    eng_fd.compute_residuals(
        fd_state, auto_path_b_derivatives=PathBGridConfig(spatial_dimension=1), fill_law_fd=True
    )
    rep_fd = audit(eng_fd.log)

    print("supplied_only:", rep_supplied["derivative_provenance"], f"score {rep_supplied['overall_admissibility_score']:.3f}")
    print("missing T_t  :", raised)
    print("auto + FD    :", rep_fd["derivative_provenance"], f"score {rep_fd['overall_admissibility_score']:.3f}")
    return {"supplied": rep_supplied, "fd": rep_fd, "missing_error": raised}


if __name__ == "__main__":
    main()
