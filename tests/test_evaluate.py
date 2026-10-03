"""Stateless batch evaluation."""

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import pytest

import moju
from moju.monitor import (
    REPORT_SCHEMA_VERSION,
    BoundCheck,
    LawSpec,
    ResidualEngine,
    Scoring,
    audit,
    evaluate,
)
from moju.monitor.derivative_provenance import MissingSuppliedDerivativeError
from moju.registry import register_law_scale_recipe, unregister_law_scale_recipe


def _engine() -> ResidualEngine:
    return ResidualEngine(
        laws=[
            LawSpec(
                name="unit_law",
                fn=lambda r: r,
                state_map={"r": "r"},
                scoring=Scoring("rms", dimensionless=True),
            )
        ],
        bound_checks=[BoundCheck(name="pos", value_key="v", lower=0.0, scale=10.0)],
        law_implied_audits=False,
    )


def _candidates():
    return [
        {"r": jnp.full((4,), 1e-3), "v": jnp.array([1.0, -0.01])},
        {"r": jnp.full((4,), 0.2), "v": jnp.array([0.5, 0.5])},
    ]


def test_evaluate_does_not_touch_engine_log():
    eng = _engine()
    eng.compute_residuals(_candidates()[0])
    logged = eng.log[0]
    n = len(eng.log)
    evaluate(eng, _candidates(), run_mode="training")
    assert len(eng.log) == n
    assert eng.log[0] is logged
    assert eng._index == 1


def test_evaluate_matches_per_candidate_audit():
    candidates = _candidates()
    refs = [
        {"r": jnp.zeros(4), "v": jnp.zeros(2)},
        {"r": jnp.full((4,), 0.1), "v": jnp.ones(2)},
    ]
    reports = evaluate(_engine(), candidates, state_refs=refs, return_residuals=False)
    assert len(reports) == 2
    for state, ref, report in zip(candidates, refs, reports):
        eng = _engine()
        eng.compute_residuals(state, ref, run_mode="eval")
        expected = audit(eng.log)
        assert report["schema_version"] == REPORT_SCHEMA_VERSION == expected["schema_version"]
        assert report["overall_admissibility_score"] == pytest.approx(expected["overall_admissibility_score"])
        assert report["per_category"].keys() == expected["per_category"].keys()
        for cat, score in expected["per_category"].items():
            assert report["per_category"][cat] == pytest.approx(score, rel=1e-5, abs=1e-6)
        assert set(report["per_key"]) == set(expected["per_key"])
        for key, row in expected["per_key"].items():
            assert report["per_key"][key]["admissibility_score"] == pytest.approx(row["admissibility_score"])
            assert report["per_key"][key]["scale_source"] == row["scale_source"]


def test_evaluate_scoring_and_bounds_flow_through():
    report = evaluate(_engine(), [_candidates()[0]])[0]
    assert report["schema_version"] == REPORT_SCHEMA_VERSION
    law = report["per_key"]["laws/unit_law"]
    assert law["scale_source"] == "declared"
    bound = report["per_key"]["constitutive/bound/pos/violation"]
    assert bound["admissibility_metric"] == "max"
    assert bound["scale_source"] == "declared"


def test_evaluate_state_refs_length():
    with pytest.raises(ValueError, match="state_refs"):
        evaluate(_engine(), _candidates(), state_refs=[None])


def test_evaluate_can_return_residuals():
    report = evaluate(_engine(), [_candidates()[0]], return_residuals=True)[0]
    assert "bound/pos/violation" in report["residuals"]["constitutive"]


def _oscillator():
    path = Path(__file__).resolve().parents[1] / "examples" / "cookbook_custom_law_oscillator.py"
    spec = importlib.util.spec_from_file_location("cookbook_custom_law_oscillator", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _oscillator_states():
    osc = _oscillator()
    exact = osc.trajectory()
    wrong = dict(exact)
    wrong["q_tt"] = jnp.zeros_like(exact["q_tt"])
    return osc, exact, wrong


def test_evaluate_batches_exact_and_wrong_oscillator():
    osc, exact, wrong = _oscillator_states()
    register_law_scale_recipe("damped_oscillator", osc.oscillator_si_recipe, overwrite=True)
    try:
        eng = osc.si_engine()
        logged = list(eng.log)
        reports = evaluate(eng, [exact, wrong], return_residuals=True)
    finally:
        unregister_law_scale_recipe("damped_oscillator")
    assert eng.log == logged
    assert [r["execution"] for r in reports] == ["batched", "batched"]
    assert reports[0]["overall_admissibility_level"] == "High Admissibility"
    assert reports[1]["overall_admissibility_level"] != "High Admissibility"
    assert len(reports[0]["residuals"]) and len(reports[1]["residuals"])
    assert set(reports[0]["residuals"]) == set(reports[1]["residuals"])

    register_law_scale_recipe("damped_oscillator", osc.oscillator_si_recipe, overwrite=True)
    try:
        alone = [evaluate(osc.si_engine(), [state])[0] for state in (exact, wrong)]
    finally:
        unregister_law_scale_recipe("damped_oscillator")
    for report, one in zip(reports, alone):
        assert report["schema_version"] == one["schema_version"] == REPORT_SCHEMA_VERSION
        assert report["moju_version"] == one["moju_version"] == moju.__version__
        assert report["tier_definition"] == one["tier_definition"]
        assert report["derivative_provenance"] == one["derivative_provenance"]
        assert set(report["per_key"]) == set(one["per_key"])
        for key, row in one["per_key"].items():
            got = report["per_key"][key]
            assert got["scale_source"] == row["scale_source"]
            assert got["admissibility_score"] == pytest.approx(row["admissibility_score"])
            assert got["rms"] == pytest.approx(row["rms"])


def test_evaluate_loops_on_mismatched_shapes_and_fd_fill():
    osc, exact, _wrong = _oscillator_states()
    short = osc.trajectory(n=40)
    register_law_scale_recipe("damped_oscillator", osc.oscillator_si_recipe, overwrite=True)
    try:
        shaped = evaluate(osc.si_engine(), [exact, short])
        filled = evaluate(
            osc.si_engine(),
            [exact, exact],
            auto_path_b_derivatives=True,
            fill_law_fd=True,
        )
    finally:
        unregister_law_scale_recipe("damped_oscillator")
    assert [r["execution"] for r in shaped] == ["loop", "loop"]
    assert [r["execution"] for r in filled] == ["loop", "loop"]


def test_evaluate_supplied_only_raises_before_a_report():
    osc, exact, _wrong = _oscillator_states()
    missing = dict(exact)
    del missing["q_tt"]
    register_law_scale_recipe("damped_oscillator", osc.oscillator_si_recipe, overwrite=True)
    try:
        eng = osc.si_engine()
        eng.derivatives = "supplied_only"
        with pytest.raises(MissingSuppliedDerivativeError):
            evaluate(eng, [missing])
        assert eng.log == []
    finally:
        unregister_law_scale_recipe("damped_oscillator")
