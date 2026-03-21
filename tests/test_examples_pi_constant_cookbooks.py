"""Smoke tests: π-constant cookbooks under examples/."""

import importlib.util
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load_example_module(filename: str):
    path = _EXAMPLES / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cookbook_pi_constant_reynolds_main():
    mod = _load_example_module("cookbook_pi_constant_reynolds.py")
    out = mod.main()
    assert "scaling/re/pi_constant" in out["report"]["per_key"]
    assert out["pi_rms"] < 1e-3


def test_cookbook_pi_constant_prandtl_main():
    mod = _load_example_module("cookbook_pi_constant_prandtl.py")
    out = mod.main()
    assert "scaling/pr/pi_constant" in out["report"]["per_key"]
    assert out["pi_rms"] < 1e-5
