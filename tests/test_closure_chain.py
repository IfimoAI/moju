"""Tests for chain-rule closures and ``chain_output=fd_on_composition``."""

import jax.numpy as jnp
import pytest

from moju.monitor.closure_registry import compute_chain


def _linear_fo(alpha, t, L):
    return alpha * 2.0 + t * 0.0 + L * 0.0


def test_compute_chain_fd_on_composition_matches_chain_rule_1d(rtol, atol):
    x = jnp.linspace(0.0, 1.0, 17)
    alpha = x**2
    t = jnp.ones_like(x)
    L = jnp.ones_like(x) * 0.5
    state = {
        "x": x,
        "alpha": alpha,
        "t": t,
        "L": L,
        "d_alpha_dx": 2.0 * x,
    }
    r = compute_chain(
        fn=_linear_fo,
        arg_names=["alpha", "t", "L"],
        output_key="fo",
        state_map={"alpha": "alpha", "t": "t", "L": "L"},
        state_pred=state,
        constants={},
        predicted_varying=["alpha"],
        deriv="x",
        chain_output="fd_on_composition",
    )
    assert r is not None
    # Endpoints: jnp.gradient vs chain rule can differ slightly on boundaries.
    assert jnp.allclose(r[1:-1], 0.0, rtol=1e-2, atol=1e-2)


def test_compute_chain_state_derivative_requires_d_out_dx():
    x = jnp.linspace(0.0, 1.0, 9)
    alpha = x**2
    t = jnp.ones_like(x)
    L = jnp.ones_like(x)
    state = {
        "x": x,
        "alpha": alpha,
        "t": t,
        "L": L,
        "d_alpha_dx": 2.0 * x,
    }
    r = compute_chain(
        fn=_linear_fo,
        arg_names=["alpha", "t", "L"],
        output_key="fo",
        state_map={"alpha": "alpha", "t": "t", "L": "L"},
        state_pred=state,
        constants={},
        predicted_varying=["alpha"],
        deriv="x",
        chain_output="state_derivative",
    )
    assert r is None


def test_compute_chain_invalid_chain_output_raises():
    with pytest.raises(ValueError, match="chain_output"):
        compute_chain(
            fn=_linear_fo,
            arg_names=["alpha", "t", "L"],
            output_key="fo",
            state_map={"alpha": "alpha", "t": "t", "L": "L"},
            state_pred={
                "alpha": jnp.array(1.0),
                "t": jnp.array(1.0),
                "L": jnp.array(1.0),
                "d_alpha_dx": jnp.array(0.0),
            },
            constants={},
            predicted_varying=["alpha"],
            deriv="x",
            chain_output="bogus",
        )
