"""Named governing-law registry (moju.registry.register_law) and moju.laws entry points."""

import importlib.util
import warnings
from pathlib import Path

import pytest

from moju.monitor import LawImpliedCheck, ResidualEngine, audit, evaluate, implied_by_projection
from moju.monitor.law_scale_recipes import characteristic_law_scale_k
from moju.monitor.nondim_inference import law_time_scale_hint
from moju.registry import (
    RegisteredLaw,
    get_law,
    list_laws,
    register_law,
    register_model,
    unregister_law,
    unregister_model,
)

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load(filename):
    spec = importlib.util.spec_from_file_location(filename[:-3], _EXAMPLES / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


heat = _load("cookbook_register_law_heat_1d.py")


def _reset_entry_points():
    from moju import registry as reg

    reg._LAWS._entry_points_loaded = False


def test_list_laws_includes_builtins():
    names = list_laws()
    assert "fourier_conduction" in names
    record = get_law("fourier_conduction")
    assert record.source == "builtin"
    assert record.required_keys == ("T_t", "T_laplacian", "fo", "t", "L")


def test_cookbook_registered_heat_law():
    out = heat.main()
    row = out["report"]["per_key"][out["flat_key"]]
    assert row["admissibility_level"] == "High Admissibility"
    assert out["report"]["law_sources"]["heat_1d"] == {"source": "registered"}
    with pytest.raises(KeyError):
        get_law("heat_1d")


def test_registered_heat_log_source_and_default_state_map():
    register_law("heat_1d", heat.heat_1d, required_keys=("T_t", "T_xx", "alpha"), scale_recipe=heat.heat_scale)
    try:
        engine = ResidualEngine(laws=[{"name": "heat_1d"}], law_implied_audits=False)
        assert engine.laws_spec[0]["state_map"] == {"T_t": "T_t", "T_xx": "T_xx", "alpha": "alpha"}
        engine.compute_residuals(heat._state())
        report = audit(engine.log)
        assert engine.log[-1]["law_sources"]["heat_1d"]["source"] == "registered"
        assert report["law_sources"]["heat_1d"]["source"] == "registered"
        assert engine.log[-1]["scale_source"]["laws/heat_1d"] == "user_recipe"
    finally:
        unregister_law("heat_1d")


def test_evaluate_reports_law_source():
    register_law("heat_1d", heat.heat_1d, required_keys=("T_t", "T_xx", "alpha"))
    try:
        engine = ResidualEngine(laws=[{"name": "heat_1d"}], law_implied_audits=False)
        state = heat._state()
        reports = evaluate(engine, [state, state])
    finally:
        unregister_law("heat_1d")
    assert engine.log == []
    assert len(reports) == 2
    assert reports[0]["law_sources"]["heat_1d"]["source"] == "registered"
    assert reports[0]["execution"] == "batched"


def test_spec_fn_source_and_hook_precedence():
    def other_scale(*_args):
        return 0.2

    register_law(
        "heat_1d",
        heat.heat_1d,
        required_keys=("T_t", "T_xx", "alpha"),
        scale_recipe=heat.heat_scale,
        time_scale="fourier",
    )
    try:
        spec = {
            "name": "heat_1d",
            "fn": heat.heat_1d,
            "state_map": {"T_t": "T_t", "T_xx": "T_xx", "alpha": "alpha"},
            "scale_recipe": other_scale,
            "time_scale": "wave",
        }
        engine = ResidualEngine(laws=[spec], law_implied_audits=False)
        engine.compute_residuals(heat._state())
        assert engine.log[-1]["law_sources"]["heat_1d"] == {"source": "spec_fn"}
        assert engine.log[-1]["scale_source"]["laws/heat_1d"] == "user_recipe"
        assert law_time_scale_hint("heat_1d", spec) == "wave"
        assert law_time_scale_hint("heat_1d") == "fourier"
        bare = {"name": "heat_1d", "state_map": spec["state_map"]}
        scale, source = characteristic_law_scale_k(
            "heat_1d", merged=heat._state(), constants={}, law_spec=bare
        )
        assert source == "user_recipe"
        assert scale > 1e-2
    finally:
        unregister_law("heat_1d")


def test_implied_check_on_registered_law():
    def model_alpha(alpha):
        return alpha

    register_model("heat_alpha", model_alpha)
    register_law(
        "heat_1d",
        heat.heat_1d,
        required_keys=("T_t", "T_xx", "alpha"),
        implied_check=LawImpliedCheck(
            model="heat_alpha",
            output_key="alpha",
            state_map={"alpha": "alpha"},
            implied_maker=implied_by_projection("T_t", "T_xx"),
        ),
    )
    try:
        engine = ResidualEngine(laws=[{"name": "heat_1d"}])
        engine.compute_residuals(heat._state())
        report = audit(engine.log)
        row = report["per_key"]["constitutive/heat_alpha/law_heat_1d/implied_delta"]
        assert row["admissibility_level"] == "High Admissibility"
    finally:
        unregister_law("heat_1d")
        unregister_model("heat_alpha")


def test_collision_and_required_keys():
    with pytest.raises(ValueError, match="overwrite=True"):
        register_law("fourier_conduction", heat.heat_1d, required_keys=("T_t", "T_xx", "alpha"))
    with pytest.raises(ValueError, match="positional parameters"):
        register_law("heat_1d", heat.heat_1d, required_keys=("T_t",))
    with pytest.raises(ValueError, match="built-in"):
        unregister_law("fourier_conduction")
    register_law("heat_1d", heat.heat_1d, required_keys=("T_t", "T_xx", "alpha"))
    try:
        with pytest.raises(ValueError, match="missing required arguments"):
            ResidualEngine(
                laws=[{"name": "heat_1d", "state_map": {"T_t": "T_t"}}],
                law_implied_audits=False,
            )
    finally:
        unregister_law("heat_1d")


def test_entry_point_load_provenance_and_isolation(monkeypatch):
    class _Dist:
        name = "example-laws"
        version = "0.1.0"

    class _Good:
        name = "ep_heat"
        dist = _Dist()

        @staticmethod
        def load():
            return RegisteredLaw(heat.heat_1d, ("T_t", "T_xx", "alpha"), time_scale="fourier")

    class _Bad:
        name = "broken_law"
        dist = _Dist()

        @staticmethod
        def load():
            raise RuntimeError("plugin failed")

    class _Collide:
        name = "fourier_conduction"
        dist = _Dist()

        @staticmethod
        def load():
            return heat.heat_1d

    class _EPS:
        def select(self, group):
            return [_Good(), _Bad(), _Collide()] if group == "moju.laws" else []

    _reset_entry_points()
    monkeypatch.setattr("importlib.metadata.entry_points", lambda: _EPS())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        record = get_law("ep_heat")
    try:
        messages = " ".join(str(item.message) for item in caught)
        assert "broken_law" in messages
        assert "fourier_conduction" in messages and "collides" in messages
        assert record.source == "entry_point"
        assert record.package == "example-laws" and record.version == "0.1.0"
        assert get_law("fourier_conduction").source == "builtin"
        engine = ResidualEngine(laws=[{"name": "ep_heat"}], law_implied_audits=False)
        engine.compute_residuals(heat._state())
        report = audit(engine.log)
        assert report["law_sources"]["ep_heat"] == {
            "source": "entry_point",
            "package": "example-laws",
            "version": "0.1.0",
        }
        assert report["per_key"]["laws/ep_heat"]["admissibility_level"] == "High Admissibility"
    finally:
        unregister_law("ep_heat")
        _reset_entry_points()


def test_torch_registered_heat_matches_jax():
    torch = pytest.importorskip("torch")
    import numpy as np

    from moju.torch import TorchResidualEngine

    register_law("heat_1d", heat.heat_1d, required_keys=("T_t", "T_xx", "alpha"))
    try:
        jax_state = heat._state()
        state = {key: torch.as_tensor(np.asarray(value)) for key, value in jax_state.items()}
        torch_engine = TorchResidualEngine(laws=[{"name": "heat_1d"}], law_implied_audits=False)
        torch_residual = torch_engine.compute_residuals_torch(state)["laws"]["heat_1d"]
        jax_engine = ResidualEngine(laws=[{"name": "heat_1d"}], law_implied_audits=False)
        jax_residual = jax_engine.compute_residuals(jax_state)["laws"]["heat_1d"]
        assert torch.allclose(torch_residual.cpu(), torch.as_tensor(np.asarray(jax_residual)), atol=1e-5)
        report = torch_engine.audit(state)
        assert report["law_sources"]["heat_1d"]["source"] == "registered"
        assert report["per_key"]["laws/heat_1d"]["admissibility_level"] == "High Admissibility"
    finally:
        unregister_law("heat_1d")
