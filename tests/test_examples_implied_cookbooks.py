"""Smoke tests: implied_delta constitutive cookbooks."""

import importlib.util
from pathlib import Path

import jax.numpy as jnp

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load_example_module(filename: str):
    path = _EXAMPLES / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cookbook_constitutive_implied_ideal_gas_rho_main():
    mod = _load_example_module("cookbook_constitutive_implied_ideal_gas_rho.py")
    out = mod.main()
    assert jnp.allclose(out["implied_rms"], 0.0, atol=1e-9)
    assert out["flat_key"] in out["report"]["per_key"]


def test_cookbook_constitutive_implied_power_law_fn_main():
    mod = _load_example_module("cookbook_constitutive_implied_power_law_fn.py")
    out = mod.main()
    assert jnp.allclose(out["implied_rms"], 0.0, atol=1e-6)
    assert out["flat_key"] in out["report"]["per_key"]
