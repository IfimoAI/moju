"""Built-in law scale recipes stay traceable, so same-shaped evaluate stays batched."""

import jax
import jax.numpy as jnp
import pytest

from moju.monitor import ResidualEngine, audit, evaluate
from moju.monitor.law_scale_recipes import (
    LAW_SCALE_RECIPES,
    _SCALE_EPS,
    _rms_mag,
    _term_max_rms,
)
from moju.registry import get_law, list_laws

_N = 8
_D = 1
_VECTOR = {
    "u",
    "u_t",
    "u_laplacian",
    "p_grad",
    "rho_grad",
    "phi_grad",
    "stress",
    "strain",
    "E_curl",
    "B_t",
}
_TENSOR = {"u_grad", "stiffness_tensor"}
_COEFFICIENT = {
    "re",
    "pe",
    "fo",
    "t",
    "L",
    "fo_mass",
    "epsilon",
    "st_wave",
    "omega",
    "kL",
    "da",
    "mu",
    "ec",
    "U",
    "sch_kin_l2",
    "eu",
}


def _historical_rms(arr):
    """Host reduction the built-in recipes used before they became traceable."""
    a = jnp.asarray(arr)
    if a.size == 0:
        return float("nan")
    if a.ndim >= 1 and a.shape[-1] > 1 and jnp.issubdtype(a.dtype, jnp.floating):
        sq = jnp.sum(a**2, axis=-1)
        return float(jnp.sqrt(jnp.mean(sq) + _SCALE_EPS))
    return float(jnp.sqrt(jnp.mean(a**2) + _SCALE_EPS))


def _value(name: str, magnitude: float):
    if name in _TENSOR:
        return jnp.full((_N, _D, _D), magnitude)
    if name in _VECTOR:
        return jnp.full((_N, _D), magnitude)
    if name in _COEFFICIENT:
        return jnp.asarray(magnitude)
    return jnp.full((_N,), magnitude)


def _state(name: str, magnitude: float):
    return {key: _value(key, magnitude) for key in get_law(name).required_keys}


def _engine(name: str) -> ResidualEngine:
    return ResidualEngine(laws=[{"name": name}], law_implied_audits=False)


BUILTIN_LAWS = [name for name in list_laws() if get_law(name).source == "builtin"]


@pytest.mark.parametrize("name", BUILTIN_LAWS)
def test_builtin_law_evaluate_is_batched_and_matches_loop(name):
    states = (_state(name, 0.25), _state(name, 1.5))
    reports = evaluate(_engine(name), states, run_mode="eval")
    assert [report["execution"] for report in reports] == ["batched", "batched"]
    for state, report in zip(states, reports):
        loop = _engine(name)
        loop.compute_residuals(state, run_mode="eval")
        expected = audit(loop.log)
        assert report["overall_admissibility_score"] == pytest.approx(
            expected["overall_admissibility_score"]
        )
        assert set(report["per_key"]) == set(expected["per_key"])
        for key, row in expected["per_key"].items():
            got = report["per_key"][key]
            assert got["scale_source"] == row["scale_source"]
            assert got["rms"] == pytest.approx(row["rms"], rel=1e-5, abs=1e-6)
            assert got["r_norm"] == pytest.approx(row["r_norm"], rel=1e-5, abs=1e-6)
            assert got["admissibility_score"] == pytest.approx(
                row["admissibility_score"], rel=1e-5, abs=1e-6
            )


def test_continuity_recipe_matches_historical_rms_and_traces():
    gradient = jnp.arange(32.0).reshape(_N, 2, 2)
    spec = {"name": "mass_incompressible", "state_map": {"u_grad": "u_grad"}}
    divergence = jnp.trace(gradient, axis1=-2, axis2=-1)
    host = LAW_SCALE_RECIPES["mass_incompressible"](
        {"u_grad": gradient}, {}, spec, None
    )
    traced = jax.jit(
        lambda field: LAW_SCALE_RECIPES["mass_incompressible"](
            {"u_grad": field}, {}, spec, None
        )
    )(gradient)
    assert float(host) == pytest.approx(_historical_rms(divergence))
    assert float(traced) == pytest.approx(float(host))
    vector = jnp.arange(12.0).reshape(4, 3)
    scalar = jnp.array([0.1, 0.0, 4.0])
    assert _rms_mag(vector) == pytest.approx(_historical_rms(vector))
    assert _rms_mag(scalar) == pytest.approx(_historical_rms(scalar))
    assert _term_max_rms(vector, scalar, None) == pytest.approx(
        max(_historical_rms(vector), _historical_rms(scalar))
    )
