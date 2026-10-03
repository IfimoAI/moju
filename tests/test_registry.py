"""Public model/group registry (moju.registry)."""

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import pytest

from moju.monitor import AuditSpec, ResidualEngine, audit, list_constitutive_models
from moju.monitor.closure_registry import GROUP_FNS, MODEL_FNS
from moju.registry import (
    get_group_fn,
    get_model_fn,
    list_registered_groups,
    list_registered_models,
    register_group,
    register_model,
    unregister_group,
    unregister_model,
)

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load(filename):
    spec = importlib.util.spec_from_file_location(filename[:-3], _EXAMPLES / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_register_and_resolve_model():
    def my_model(T, a):
        return a * T

    register_model("my_model_t", my_model)
    try:
        assert get_model_fn("my_model_t") is my_model
        assert "my_model_t" in MODEL_FNS
        assert "my_model_t" in list_registered_models()
        assert "my_model_t" in list_constitutive_models()
        with pytest.raises(ValueError):
            register_model("my_model_t", my_model)
    finally:
        unregister_model("my_model_t")
    assert "my_model_t" not in MODEL_FNS


def test_builtin_name_requires_overwrite():
    with pytest.raises(ValueError, match="overwrite=True"):
        register_model("sutherland_mu", lambda T: T)
    with pytest.raises(ValueError):
        register_group("re", lambda a: a)
    with pytest.raises(ValueError):
        unregister_model("sutherland_mu")


def test_overwrite_builtin_then_restore():
    orig = get_model_fn("sutherland_mu")
    register_model("sutherland_mu", lambda T, mu0, T0, S: T * 0 + 1.0, overwrite=True)
    try:
        assert get_model_fn("sutherland_mu") is not orig
    finally:
        unregister_model("sutherland_mu")
    assert get_model_fn("sutherland_mu") is orig


def test_register_group_used_by_engine():
    register_group("my_ratio", lambda a, b: a / b)
    try:
        assert "my_ratio" in GROUP_FNS and "my_ratio" in list_registered_groups()
        eng = ResidualEngine(
            laws=[{"name": "custom", "fn": lambda r: r - 2.0, "state_map": {"r": "r"}}],
            groups=[{"name": "my_ratio", "output_key": "r", "state_map": {"a": "a", "b": "b"}}],
            law_implied_audits=False,
        )
        res = eng.compute_residuals({"a": jnp.array([4.0, 6.0]), "b": jnp.array([2.0, 3.0])})
        assert jnp.allclose(res["laws"]["custom"], 0.0)
    finally:
        unregister_group("my_ratio")
    with pytest.raises(KeyError, match="Unknown group"):
        get_group_fn("my_ratio")


def test_unknown_model_errors_instead_of_silent_skip():
    with pytest.raises(ValueError, match="not registered"):
        ResidualEngine(
            constitutive_audit=[AuditSpec(name="nope_model", output_key="k", state_map={}, implied_value_key="k")]
        )


def test_entry_points_loaded_lazily(monkeypatch):
    from moju.monitor import closure_registry as cr

    class _EP:
        name = "ep_model"

        @staticmethod
        def load():
            return lambda T: 2.0 * T

    class _EPS:
        def select(self, group):
            return [_EP()] if group == "moju.models" else []

    table = cr.FnTable({}, "moju.models")
    monkeypatch.setattr("importlib.metadata.entry_points", lambda: _EPS())
    assert "ep_model" in table
    fn, args = table["ep_model"]
    assert args == ["T"] and float(fn(2.0)) == 4.0


def test_cookbook_registry_k_of_T():
    out = _load("cookbook_registry_k_of_T.py").main()
    good = out["report"]["per_key"][out["flat_key"]]
    bad = out["report_drifted"]["per_key"][out["flat_key"]]
    assert good["admissibility_level"] == "High Admissibility"
    assert bad["admissibility_score"] < good["admissibility_score"]
    assert "k_of_T" not in MODEL_FNS
