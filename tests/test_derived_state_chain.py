"""Tests for JSON-safe derived state chain (Phase B)."""

import jax.numpy as jnp

from moju.monitor.derived_state_chain import (
    apply_derived_state_chain,
    collect_expr_ref_keys,
    eval_derived_expr,
    keys_produced_by_chain,
    all_ref_keys_from_chain,
)
from moju.monitor.auditor import ResidualEngine


def test_eval_exp():
    expr = {"op": "exp", "x": {"op": "ref", "key": "s"}}
    env = {"s": jnp.array([0.0, 1.0])}
    out = eval_derived_expr(expr, env)
    assert jnp.allclose(out, jnp.exp(jnp.array([0.0, 1.0])))


def test_eval_pow():
    expr = {
        "op": "pow",
        "a": {"op": "ref", "key": "T"},
        "b": {"op": "const", "value": 2.0},
    }
    env = {"T": jnp.array([2.0, 3.0])}
    out = eval_derived_expr(expr, env)
    assert jnp.allclose(out, jnp.array([4.0, 9.0]))


def test_collect_expr_ref_keys_exp_pow():
    expr = {
        "op": "mul",
        "a": {"op": "exp", "x": {"op": "ref", "key": "beta"}},
        "b": {"op": "pow", "left": {"op": "ref", "key": "T"}, "right": {"op": "const", "value": -1.0}},
    }
    assert collect_expr_ref_keys(expr) == {"beta", "T"}


def test_eval_mul_ref():
    expr = {
        "op": "mul",
        "a": {"op": "const", "value": 2.0},
        "b": {"op": "ref", "key": "T"},
    }
    env = {"T": jnp.array([1.0, 2.0])}
    out = eval_derived_expr(expr, env)
    assert jnp.allclose(out, jnp.array([2.0, 4.0]))


def test_chain_kappa_alpha():
    T = jnp.array([300.0])
    rho = jnp.array([1.2])
    cp = jnp.array([1005.0])
    state = {"T": T, "rho": rho, "cp": cp}
    constants = {}
    steps = [
        {
            "output_key": "kappa",
            "expr": {
                "op": "mul",
                "a": {"op": "const", "value": 0.001},
                "b": {"op": "ref", "key": "T"},
            },
        },
        {
            "output_key": "alpha",
            "expr": {
                "op": "div",
                "a": {"op": "ref", "key": "kappa"},
                "b": {
                    "op": "mul",
                    "a": {"op": "ref", "key": "rho"},
                    "b": {"op": "ref", "key": "cp"},
                },
            },
        },
    ]
    out, warn = apply_derived_state_chain(state, constants, steps)
    assert not warn
    assert jnp.allclose(out["kappa"], 0.001 * T)
    assert jnp.allclose(out["alpha"], out["kappa"] / (rho * cp))


def test_keys_produced_and_refs():
    steps = [
        {"output_key": "kappa", "expr": {"op": "ref", "key": "T"}},
        {"output_key": "alpha", "expr": {"op": "ref", "key": "kappa"}},
    ]
    assert keys_produced_by_chain(steps) == {"kappa", "alpha"}
    assert all_ref_keys_from_chain(steps) == {"T", "kappa"}


def test_residual_engine_required_state_keys_respects_chain():
    chain = [
        {
            "output_key": "kappa",
            "expr": {"op": "mul", "a": {"op": "const", "value": 0.001}, "b": {"op": "ref", "key": "T"}},
        },
        {
            "output_key": "alpha",
            "expr": {
                "op": "div",
                "a": {"op": "ref", "key": "kappa"},
                "b": {
                    "op": "mul",
                    "a": {"op": "ref", "key": "rho"},
                    "b": {"op": "ref", "key": "cp"},
                },
            },
        },
    ]
    engine = ResidualEngine(
        laws=[],
        groups=[],
        constitutive_audit=[],
        scaling_audit=[],
        derived_state_chain=chain,
    )
    keys = engine.required_state_keys()
    assert {"T", "rho", "cp"} <= keys
    assert "kappa" not in keys
    assert "alpha" not in keys


def test_apply_chain_warning_bad_step_continues():
    state = {"T": jnp.array([1.0])}
    steps = [
        {"output_key": "bad", "expr": {"op": "ref", "key": "missing"}},
        {
            "output_key": "ok",
            "expr": {"op": "mul", "a": {"op": "ref", "key": "T"}, "b": {"op": "const", "value": 2.0}},
        },
    ]
    out, w = apply_derived_state_chain(state, {}, steps)
    assert any("missing" in x for x in w)
    assert "ok" in out
    assert jnp.allclose(out["ok"], jnp.array([2.0]))
