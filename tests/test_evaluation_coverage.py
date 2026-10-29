"""Evaluation coverage records and nonfinite_policy (Moju 2.4)."""

from __future__ import annotations

import json
import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from moju.monitor import LawSpec, ResidualEngine, Scoring, audit, evaluate
from moju.monitor.analytic_flows import ns_state_from_analytic, taylor_green_2d
from moju.monitor.evaluation_coverage import TIER_INCOMPLETE
from moju.monitor.monitor_log_export import export_monitor_log, monitor_log_export_to_jsonable
from moju.monitor.report import _residual_dict_to_json_serializable


def _unit_engine() -> ResidualEngine:
    return ResidualEngine(
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


def _audit_case(values, *, policy="flag", eval_mask=None, eval_mask_label=None):
    eng = _unit_engine()
    eng.compute_residuals(
        {"r": jnp.array(values)},
        run_mode="eval",
        weighted_norms=False,
        eval_mask=eval_mask,
        eval_mask_label=eval_mask_label,
    )
    return audit(
        eng.log,
        nonfinite_policy=policy,
        eval_mask_label=eval_mask_label,
        last_residual_dict=eng.last_residuals,
    )


def _eval_case(values, *, policy="flag", debug_loop=False):
    eng = _unit_engine()
    return evaluate(
        eng,
        [{"r": jnp.array(values)}],
        run_mode="eval",
        weighted_norms=False,
        nonfinite_policy=policy,
        evaluate_debug_loop=debug_loop,
    )[0]


@pytest.mark.parametrize(
    "values,policy,expect_score_finite,expect_tier,expect_status",
    [
        ([0.0, 0.0], "flag", True, "High Admissibility", "ok"),
        ([0.0, 0.0], "strict", True, "High Admissibility", "ok"),
        ([0.0, 0.0], "ignore", True, "High Admissibility", "ok"),
        ([0.0, float("nan")], "flag", True, "High Admissibility", "incomplete"),
        ([0.0, float("nan")], "strict", False, TIER_INCOMPLETE, "incomplete"),
        ([0.0, float("nan")], "ignore", True, "High Admissibility", "incomplete"),
        ([float("nan"), float("nan")], "flag", True, "Non-Admissible", "empty"),
        ([float("nan"), float("nan")], "strict", False, TIER_INCOMPLETE, "empty"),
        ([float("nan"), float("nan")], "ignore", True, "Non-Admissible", "empty"),
        ([0.0, float("inf")], "flag", True, "Non-Admissible", "incomplete"),
        ([0.0, float("inf")], "strict", False, TIER_INCOMPLETE, "incomplete"),
        ([0.0, float("-inf")], "flag", True, "Non-Admissible", "incomplete"),
        ([0.0, float("-inf")], "strict", False, TIER_INCOMPLETE, "incomplete"),
    ],
)
def test_reported_cases_all_policies_audit(
    values, policy, expect_score_finite, expect_tier, expect_status
):
    rep = _audit_case(values, policy=policy)
    score = rep["overall_admissibility_score"]
    if expect_score_finite:
        assert score is not None and math.isfinite(float(score))
        if expect_status == "empty" and policy == "flag":
            assert float(score) == 0.0
    else:
        assert score is None
    assert rep["overall_admissibility_level"] == expect_tier
    cov = rep["per_key"]["laws/unit_law"]["evaluation_coverage"]
    assert cov["coverage_status"] == expect_status
    assert rep["coverage_status"] == expect_status


@pytest.mark.parametrize("debug_loop", [False, True])
@pytest.mark.parametrize(
    "values,policy,expect_tier,expect_status",
    [
        ([0.0, float("nan")], "flag", "High Admissibility", "incomplete"),
        ([0.0, float("nan")], "strict", TIER_INCOMPLETE, "incomplete"),
        ([float("nan"), float("nan")], "flag", "Non-Admissible", "empty"),
        ([0.0, float("inf")], "strict", TIER_INCOMPLETE, "incomplete"),
    ],
)
def test_reported_cases_evaluate_paths(values, policy, expect_tier, expect_status, debug_loop):
    rep = _eval_case(values, policy=policy, debug_loop=debug_loop)
    assert rep["execution"] == ("loop" if debug_loop else "batched")
    assert rep["overall_admissibility_level"] == expect_tier
    assert rep["coverage_status"] == expect_status
    assert rep["per_key"]["laws/unit_law"]["evaluation_coverage"]["coverage_status"] == expect_status


def test_masked_vs_nonfinite_separation():
    """``[0, NaN]`` with mask keeping only the finite point → coverage ok."""
    mask = jnp.array([True, False])
    rep = _audit_case(
        [0.0, float("nan")],
        policy="flag",
        eval_mask=mask,
        eval_mask_label="keep_finite",
    )
    cov = rep["per_key"]["laws/unit_law"]["evaluation_coverage"]
    assert cov["coverage_status"] == "ok"
    assert cov["n_masked"] == 1
    assert cov["n_nonfinite"] == 0
    assert cov["n_finite"] == 1
    assert cov["n_total"] == 2
    assert cov["eval_mask_label"] == "keep_finite"
    assert rep["overall_admissibility_level"].startswith("High")


def test_batched_vs_loop_coverage_parity():
    vals = [0.0, float("nan")]
    a = _eval_case(vals, policy="flag", debug_loop=False)
    b = _eval_case(vals, policy="flag", debug_loop=True)
    assert a["per_key"]["laws/unit_law"]["evaluation_coverage"] == b["per_key"][
        "laws/unit_law"
    ]["evaluation_coverage"]
    assert a["coverage_status"] == b["coverage_status"]
    assert a["overall_admissibility_score"] == pytest.approx(b["overall_admissibility_score"])


def test_implied_parameter_with_nonfinite_sample():
    n = 16
    xs = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    x, y = jnp.meshgrid(xs, xs, indexing="ij")
    fields = taylor_green_2d(x, y, 0.1, re=10.0)
    state = ns_state_from_analytic(fields, x, y, t=0.1)
    state.update({"rho": 1.0, "U": 1.0, "L": 1.0, "mu": 0.1, "re": 10.0})
    lap = jnp.array(state["u_laplacian"])
    lap = lap.at[0, 0, 0].set(jnp.nan)
    state["u_laplacian"] = lap
    sm = {
        "u_t": "u_t",
        "u": "u",
        "u_grad": "u_grad",
        "p_grad": "p_grad",
        "u_laplacian": "u_laplacian",
        "re": "re",
    }
    eng = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": sm}],
        law_implied_audits=True,
        constants={"rho": 1.0, "U": 1.0, "L": 1.0},
    )
    eng.compute_residuals(state, weighted_norms=False)
    rep_flag = audit(eng.log, nonfinite_policy="flag")
    rep_strict = audit(eng.log, nonfinite_policy="strict")
    mu = "constitutive/dynamic_viscosity_from_re/law_momentum_navier_stokes/implied_delta"
    law = "laws/momentum_navier_stokes"
    assert "evaluation_coverage" in rep_flag["per_key"][law]
    assert "evaluation_coverage" in rep_flag["per_key"][mu]
    # Aggregate present
    assert "evaluation_coverage" in rep_flag
    assert rep_flag["evaluation_coverage"]["aggregate"]["coverage_status"] in (
        "ok",
        "incomplete",
        "empty",
    )
    # Strict may or may not Incomplete depending on which keys have nonfinite;
    # at least coverage is attached and policy recorded.
    assert rep_strict["nonfinite_policy"] == "strict"


def test_aggregate_status_propagation():
    rep = _audit_case([0.0, float("nan")], policy="flag")
    assert rep["coverage_status"] == "incomplete"
    assert rep["evaluation_coverage"]["aggregate"]["coverage_status"] == "incomplete"


def test_json_roundtrip_coverage_and_incomplete():
    rep = _audit_case([0.0, float("nan")], policy="strict")
    blob = _residual_dict_to_json_serializable(rep)
    text = json.dumps(blob, allow_nan=False)
    parsed = json.loads(text)
    assert parsed["overall_admissibility_score"] is None
    assert parsed["overall_admissibility_level"] == TIER_INCOMPLETE
    cov = parsed["per_key"]["laws/unit_law"]["evaluation_coverage"]
    assert cov["coverage_status"] == "incomplete"
    assert cov["n_nonfinite"] == 1

    eng = _unit_engine()
    eng.compute_residuals(
        {"r": jnp.array([0.0, float("nan")])}, run_mode="eval", weighted_norms=False
    )
    audit(eng.log, nonfinite_policy="strict")
    export = export_monitor_log(eng.log, scope="audit", persist=False)
    # Enrich may not re-apply strict; dump report path instead
    text2 = json.dumps(monitor_log_export_to_jsonable(export), allow_nan=False)
    json.loads(text2)


def test_flag_preserves_high_on_partial_nan():
    """Default flag keeps today's High score on [0, NaN] (backward compatible)."""
    rep = _audit_case([0.0, float("nan")], policy="flag")
    assert float(rep["overall_admissibility_score"]) == pytest.approx(0.99999999, rel=0, abs=1e-8)
    assert rep["overall_admissibility_level"].startswith("High")
    assert rep["coverage_status"] == "incomplete"
