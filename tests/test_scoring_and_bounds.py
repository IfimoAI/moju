"""Scoring declarations and BoundCheck."""

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import pytest

from moju.monitor import (
    BoundCheck,
    ConstitutiveCustomSpec,
    LawSpec,
    MonitorConfig,
    ResidualEngine,
    Scoring,
    audit,
)
from moju.monitor.tiers import TIER_CUTOFFS

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load(filename):
    spec = importlib.util.spec_from_file_location(filename[:-3], _EXAMPLES / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- scoring declarations ------------------------------------------------------------------------


def test_custom_constitutive_without_scoring_unchanged():
    eng = ResidualEngine(constitutive_custom=[{"name": "c1", "fn": lambda s, c: s["r"]}])
    eng.compute_residuals({"r": jnp.array([0.1, -0.1])})
    e = eng.log[-1]
    assert e["scale_source"]["constitutive/custom/c1"] == "state_derived"
    assert "scoring" not in e
    assert "constitutive/custom/c1" not in e["r_max"]


def test_declared_scale_and_worst_point_metric():
    eng = ResidualEngine(
        constitutive_custom=[
            ConstitutiveCustomSpec(name="c1", fn=lambda s, c: s["r"], scoring=Scoring("worst_point", scale=2.0))
        ]
    )
    eng.compute_residuals({"r": jnp.array([0.0, 0.0, 0.2])})
    e = eng.log[-1]
    k = "constitutive/custom/c1"
    assert e["scale"][k] == 2.0 and e["scale_source"][k] == "declared"
    assert e["scoring"][k] == {"metric": "worst_point", "scale": 2.0, "dimensionless": False}
    assert e["r_max"][k] == pytest.approx(0.2, rel=1e-6)
    rep = audit(eng.log)
    row = rep["per_key"][k]
    assert row["admissibility_metric"] == "max"
    assert row["r_norm"] == pytest.approx(0.1, rel=1e-5)
    assert row["scale_source"] == "declared"
    assert "Declared by the user" in rep["audit_meta"]["plain_sections"]["scaling"]


def test_dimensionless_scoring_uses_default_gauge():
    eng = ResidualEngine(
        laws=[
            LawSpec(
                name="unit_law",
                fn=lambda r: r,
                state_map={"r": "r"},
                scoring=Scoring("rms", dimensionless=True),
            )
        ],
        law_implied_audits=False,
    )
    eng.compute_residuals({"r": jnp.full((8,), 1e-3)})
    e = eng.log[-1]
    assert e["scale"]["laws/unit_law"] == pytest.approx(1e-2, rel=1e-6)
    assert e["scale_source"]["laws/unit_law"] == "declared"
    assert audit(eng.log)["per_key"]["laws/unit_law"]["admissibility_score"] == pytest.approx(
        TIER_CUTOFFS["High Admissibility"], rel=1e-5
    )


def test_scoring_dict_form_accepted():
    eng = ResidualEngine(
        constitutive_custom=[{"name": "c", "fn": lambda s, c: s["r"], "scoring": {"metric": "rms", "scale": 4.0}}]
    )
    eng.compute_residuals({"r": jnp.array([1.0])})
    assert eng.log[-1]["scale"]["constitutive/custom/c"] == 4.0


# ---- bound checks --------------------------------------------------------------------------------


def test_bound_violation_values_and_high_boundary():
    b = BoundCheck(name="pos", value_key="v", lower=0.0, scale=10.0)
    eng = ResidualEngine(bound_checks=[b])
    res = eng.compute_residuals({"v": jnp.array([1.0, -0.01, 3.0])})
    viol = res["constitutive"]["bound/pos/violation"]
    assert jnp.allclose(viol, jnp.array([0.0, 0.001, 0.0]), atol=1e-7)
    rep = audit(eng.log)
    row = rep["per_key"]["constitutive/bound/pos/violation"]
    assert row["admissibility_score"] == pytest.approx(TIER_CUTOFFS["High Admissibility"], rel=1e-4)
    dbg = res["closure_debug"]["bound/pos/violation"]
    assert dbg["mode"] == "bound" and float(dbg["implied"][1]) == 0.0


def test_two_sided_and_callable_bounds():
    b = BoundCheck(
        name="frac",
        value_fn=lambda s, c: s["a"] * 2.0,
        lower=lambda s, c: jnp.zeros_like(s["a"]),
        upper=1.0,
        scale=1.0,
    )
    res = ResidualEngine(bound_checks=[b]).compute_residuals({"a": jnp.array([-0.5, 0.25, 0.75])})
    assert jnp.allclose(res["constitutive"]["bound/frac/violation"], jnp.array([1.0, 0.0, 0.5]))


def test_bound_validation_and_missing_key():
    with pytest.raises(ValueError):
        BoundCheck(name="x", value_key="v", scale=1.0)
    with pytest.raises(ValueError):
        BoundCheck(name="x", value_key="v", value_fn=lambda s, c: 0, lower=0, scale=1.0)
    with pytest.raises(ValueError):
        BoundCheck(name="x", value_key="v", lower=0, scale=0.0)
    with pytest.raises(ValueError, match="unique"):
        ResidualEngine(bound_checks=[BoundCheck("a", 1.0, "v", lower=0), BoundCheck("a", 1.0, "w", lower=0)])
    eng = ResidualEngine(bound_checks=[{"name": "b", "value_key": "missing", "lower": 0.0, "scale": 1.0}])
    with pytest.raises(KeyError, match="missing"):
        eng.compute_residuals({"v": jnp.zeros(2)})
    eng2 = ResidualEngine(bound_checks=[BoundCheck("b", 1.0, "missing", lower=0)], best_effort_partial=True)
    eng2.compute_residuals({"v": jnp.zeros(2)})
    assert eng2.log[-1]["unresolved_dependencies"][0]["stage"] == "bound"


def test_bounds_roll_up_with_minimum_and_config():
    cfg = MonitorConfig(
        bound_checks=[
            BoundCheck("good", 1.0, "v", lower=0.0),
            BoundCheck("bad", 1.0, "v", upper=0.5),
        ]
    )
    eng = ResidualEngine(cfg)
    eng.compute_residuals({"v": jnp.array([0.2, 0.6])})
    rep = audit(eng.log)
    bad = rep["per_key"]["constitutive/bound/bad/violation"]["admissibility_score"]
    assert rep["per_category"]["constitutive"] == pytest.approx(bad)
    assert eng.required_state_keys() >= {"v"}


def test_cookbook_bound_checks():
    reps = _load("cookbook_bound_checks.py").main()
    assert reps["clean"]["per_category"]["constitutive"] == pytest.approx(1.0)
    d = reps["defect"]["per_key"]
    for name in ("clausius_duhem", "heat_flux_down_gradient", "volume_fraction"):
        assert d[f"constitutive/bound/{name}/violation"]["admissibility_level"] == "Non-Admissible"
    assert d["constitutive/bound/viscosity_positive/violation"]["admissibility_level"] == "High Admissibility"


def test_bounds_visualize_smoke():
    pytest.importorskip("plotly")
    from moju.monitor import visualize

    mod = _load("cookbook_bound_checks.py")
    eng = ResidualEngine(bound_checks=mod.BOUND_CHECKS)
    eng.compute_residuals(mod.surrogate_state(defect=True))
    audit(eng.log)
    visualize(eng.log, engine=eng, backend="none")
    visualize(eng.log, engine=eng, mode="training")
