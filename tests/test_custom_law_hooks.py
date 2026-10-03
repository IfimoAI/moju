"""Custom-law hooks: scale recipes, time-scale hints, key declarations, Fourier ND consistency."""

import importlib.util
import math
import warnings
from pathlib import Path

import jax.numpy as jnp
import pytest

from moju.monitor import KeyDeclaration, ResidualEngine, build_law_spec_identity
from moju.monitor.key_declarations import declaration_factor
from moju.monitor.law_scale_recipes import (
    characteristic_law_scale_k,
    law_scale_coverage_report,
    register_law_scale_recipe,
    term_max_rms,
    term_rms,
    unregister_law_scale_recipe,
)
from moju.monitor.nondim_inference import (
    infer_nondim_scales,
    register_law_time_scale,
    resolve_time_scale_for_laws,
    unregister_law_time_scale,
)
from moju.piratio.nondim import NondimScales, dimensional_to_nd

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load(filename):
    spec = importlib.util.spec_from_file_location(filename[:-3], _EXAMPLES / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


osc = _load("cookbook_custom_law_oscillator.py")


# ---- law-linked implied checks -------------------------------------------------------------------


def test_custom_diffusion_implied_check():
    out = _load("cookbook_custom_diffusion_implied.py").main()
    good = out["reports"]["consistent"]["per_key"][out["flat_key"]]
    bad = out["reports"]["rate off 3%"]["per_key"][out["flat_key"]]
    assert good["admissibility_level"] == "High Admissibility"
    assert bad["admissibility_score"] < 0.5


def test_registered_implied_rows_merge_and_unregister():
    from moju.monitor import LawImpliedCheck, implied_by_projection, merge_law_implied_audit_specs
    from moju.monitor.law_implied_diagnostics import (
        list_laws_with_implied_diagnostics,
        register_law_implied_check,
        supported_auto_implied_laws_for,
        unregister_law_implied_checks,
    )

    chk = LawImpliedCheck("thermal_diffusivity", "alpha", {"k": "k"}, implied_by_projection("a", "b"))
    register_law_implied_check("my_law", chk)
    register_law_implied_check("my_law", chk)
    try:
        rows, _ = merge_law_implied_audit_specs([{"name": "my_law", "state_map": {"k": "kk"}}])
        assert len(rows) == 1
        assert rows[0]["residual_basename"] == "thermal_diffusivity/law_my_law"
        assert rows[0]["state_map"] == {"k": "kk"}
        assert "my_law" in list_laws_with_implied_diagnostics()
        assert supported_auto_implied_laws_for([{"name": "my_law"}])[0] == ["my_law"]
        assert merge_law_implied_audit_specs([{"name": "my_law"}], enabled=False) == ([], [])
    finally:
        unregister_law_implied_checks("my_law")
    assert merge_law_implied_audit_specs([{"name": "my_law", "state_map": {}}])[0] == []


def test_implied_by_projection_vector():
    from moju.monitor import implied_by_projection

    fn = implied_by_projection("lhs", "op", vector=True)({"lhs": "L1", "op": "O1"})
    op = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    out = fn({"L1": 2.5 * op, "O1": op}, {})
    assert jnp.allclose(out, 2.5)


# ---- scale recipes -------------------------------------------------------------------------------


def test_term_helpers_public():
    assert term_rms(jnp.array([3.0, -3.0])) == pytest.approx(3.0, rel=1e-6)
    assert term_rms(jnp.array([[3.0, 4.0]]), vector=True) == pytest.approx(5.0, rel=1e-6)
    assert term_max_rms(jnp.array([1.0]), None, jnp.array([2.0])) == pytest.approx(2.0, rel=1e-6)
    assert math.isnan(term_max_rms(None))


def test_registered_recipe_used_and_reported():
    spec = {"name": "my_law", "state_map": {"a": "a"}}
    register_law_scale_recipe("my_law", lambda m, c, s, n: 7.0)
    try:
        assert characteristic_law_scale_k("my_law", merged={}, constants={}, law_spec=spec) == (7.0, "auto")
        assert law_scale_coverage_report()["my_law"] == "user_recipe"
        with pytest.raises(ValueError):
            register_law_scale_recipe("my_law", lambda *a: 1.0)
    finally:
        unregister_law_scale_recipe("my_law")
    assert "my_law" not in law_scale_coverage_report()


def test_spec_recipe_beats_registered():
    spec = {"name": "my_law2", "state_map": {}, "scale_recipe": lambda *a: 3.0}
    register_law_scale_recipe("my_law2", lambda *a: 9.0)
    try:
        assert characteristic_law_scale_k("my_law2", merged={}, constants={}, law_spec=spec)[0] == 3.0
    finally:
        unregister_law_scale_recipe("my_law2")


def test_failing_user_recipe_warns_and_falls_back():
    def bad(*a):
        raise RuntimeError("boom")

    spec = {"name": "bad_law", "state_map": {"a": "a"}, "scale_recipe": bad}
    with pytest.warns(UserWarning, match="boom"):
        sk, src = characteristic_law_scale_k("bad_law", merged={"a": jnp.array([5.0])}, constants={}, law_spec=spec)
    assert sk == pytest.approx(5.0, rel=1e-6)


def test_oscillator_si_scale_is_term_balance():
    out = osc.main()
    eng, rep = out["si"]
    st = out["state"]
    expected = max(
        float(term_rms(osc.M * st["q_tt"])), float(term_rms(osc.C * st["q_t"])), float(term_rms(osc.K * st["q"]))
    )
    assert eng.log[-1]["scale"]["laws/damped_oscillator"] == pytest.approx(expected, rel=1e-5)
    assert rep["overall_admissibility_level"] == "High Admissibility"


# ---- time-scale hints ----------------------------------------------------------------------------


def test_oscillator_dimensional_uses_custom_t_ref():
    out = osc.main()
    eng, rep = out["nd"]
    nd = eng.log[-1]["nondim_scales"]
    assert nd["time_scale"] == "custom"
    assert nd["t_ref_override"] == pytest.approx(math.sqrt(osc.M / osc.K))
    assert eng.log[-1]["nondim_scale_source"]["t_ref_override"] == "law_hint"
    assert rep["overall_admissibility_level"] == "High Admissibility"


def test_registered_time_scale_kind_and_conflicts():
    register_law_time_scale("my_heat", "fourier")
    try:
        assert resolve_time_scale_for_laws(["my_heat"]) == "fourier"
        with pytest.raises(ValueError, match="incompatible"):
            resolve_time_scale_for_laws(["my_heat", "burgers_equation"])
        with pytest.raises(ValueError):
            register_law_time_scale("my_heat", "convective")
    finally:
        unregister_law_time_scale("my_heat")
    with pytest.raises(ValueError):
        register_law_time_scale("x", "weird")


def test_explicit_t_ref_dict_and_mixing_error():
    specs = [{"name": "a", "state_map": {}, "time_scale": {"t_ref": 2.0}}]
    sc, prov = infer_nondim_scales(["a"], {"L": 1.0}, {}, law_specs=specs)
    assert sc.t_ref == 2.0 and sc.time_scale == "custom"
    with pytest.raises(ValueError, match="mix"):
        resolve_time_scale_for_laws(["a", "fourier_conduction"], law_specs=specs)


def test_unhinted_custom_law_warns_in_dimensional_mode():
    eng = ResidualEngine(
        constants={"L": 1.0},
        laws=[{"name": "my_custom", "fn": lambda q: q, "state_map": {"q": "q"}}],
        law_implied_audits=False,
        state_units="dimensional",
        state_declarations={"q": "dimensionless"},
    )
    with pytest.warns(UserWarning, match="no time-scale hint"):
        eng.compute_residuals({"q": jnp.zeros(4)})


# ---- key declarations ----------------------------------------------------------------------------


def test_declaration_factor():
    s = NondimScales(L_ref=2.0, U_ref=4.0, time_scale="custom", t_ref_override=3.0)
    assert declaration_factor(KeyDeclaration("length", time_order=2), s) == pytest.approx(9.0 / 2.0)
    assert declaration_factor(KeyDeclaration("velocity", space_order=1), s) == pytest.approx(2.0 / 4.0)
    assert declaration_factor(KeyDeclaration("dimensionless", time_order=1), s) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        KeyDeclaration("mass")


def test_extra_rules_override_passthrough_keys():
    s = NondimScales(L_ref=0.5)
    assert float(dimensional_to_nd({"L": 0.5}, s, extra_rules={"L": 2.0})["L"]) == pytest.approx(1.0)
    assert dimensional_to_nd({"L": 0.5}, s)["L"] == 0.5


def _undeclared_engine(policy):
    return ResidualEngine(
        constants={"L": 1.0},
        laws=[{"name": "c", "fn": lambda q: q, "state_map": {"q": "q"}, "time_scale": "convective"}],
        law_implied_audits=False,
        state_units="dimensional",
        undeclared_keys=policy,
    )


def test_undeclared_keys_warn_once_and_error():
    eng = _undeclared_engine("warn")
    with pytest.warns(UserWarning, match=r"\['q'\]"):
        eng.compute_residuals({"q": jnp.zeros(3)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        eng.compute_residuals({"q": jnp.zeros(3)})
    with pytest.raises(ValueError, match="unscaled"):
        _undeclared_engine("error").compute_residuals({"q": jnp.zeros(3)})
    with pytest.raises(ValueError):
        _undeclared_engine("ignore")


def test_nondimensional_mode_never_checks_declarations():
    eng = ResidualEngine(
        laws=[{"name": "c", "fn": lambda q: q, "state_map": {"q": "q"}}],
        law_implied_audits=False,
        undeclared_keys="error",
    )
    eng.compute_residuals({"q": jnp.zeros(3)})


def test_fourier_dimensional_exact_solution_is_admissible():
    """Regression: Fourier ND form needs L* = L/L_ref alongside t* = t/t_ref."""
    L, alpha = 0.02, 8e-5
    x = jnp.linspace(0.0, L, 41)
    lap = jnp.full_like(x, 40.0 / L**2)
    st = {
        "x": x,
        "t": jnp.full_like(x, 5.0),
        "T": 300.0 + 20.0 * (1.0 - x / L) ** 2,
        "T_t": alpha * lap,
        "T_laplacian": lap,
        "L": jnp.full_like(x, L),
        "alpha": jnp.full_like(x, alpha),
    }
    eng = ResidualEngine(
        laws=[build_law_spec_identity("fourier_conduction")],
        groups=[{"name": "fo", "output_key": "fo", "state_map": {"alpha": "alpha", "t": "t", "L": "L"}}],
        law_implied_audits=False,
        state_units="dimensional",
    )
    res = eng.compute_residuals(st)
    assert float(jnp.max(jnp.abs(res["laws"]["fourier_conduction"]))) < 1e-5
