"""Smoke tests: turbulence-related constitutive cookbooks under examples/."""

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


def test_cookbook_turbulence_law_of_wall_main():
    mod = _load_example_module("cookbook_turbulence_law_of_wall.py")
    out = mod.main()
    assert out["chain_rms"] == 0.0
    assert "constitutive/law_of_the_wall/chain_dx" in out["report"]["per_key"]


def test_cookbook_turbulence_colebrook_main():
    mod = _load_example_module("cookbook_turbulence_colebrook.py")
    out = mod.main()
    assert out["chain_rms"] == 0.0
    assert "constitutive/colebrook_friction/chain_dx" in out["report"]["per_key"]


def test_cookbook_constitutive_smagorinsky_main():
    mod = _load_example_module("cookbook_constitutive_smagorinsky.py")
    out = mod.main()
    assert out["chain_rms"] == 0.0
    assert "constitutive/smagorinsky_nu_t/chain_dx" in out["report"]["per_key"]


def test_cookbook_constitutive_k_epsilon_main():
    mod = _load_example_module("cookbook_constitutive_k_epsilon.py")
    out = mod.main()
    # Hand-derived chain vs JAX grad in monitor can differ at ~1e-8 float noise.
    assert jnp.allclose(out["chain_rms"], 0.0, atol=1e-6)
    assert "constitutive/k_epsilon_nu_t/chain_dx" in out["report"]["per_key"]


def test_cookbook_constitutive_k_omega_main():
    mod = _load_example_module("cookbook_constitutive_k_omega.py")
    out = mod.main()
    assert jnp.allclose(out["chain_rms"], 0.0, atol=1e-6)
    assert "constitutive/k_omega_nu_t/chain_dx" in out["report"]["per_key"]
