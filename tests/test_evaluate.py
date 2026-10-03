"""Stateless batch evaluation."""

import jax.numpy as jnp
import pytest

from moju.monitor import (
    REPORT_SCHEMA_VERSION,
    BoundCheck,
    LawSpec,
    ResidualEngine,
    Scoring,
    audit,
    evaluate,
)


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
        assert report["per_category"] == expected["per_category"]
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
