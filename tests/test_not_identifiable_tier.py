"""Not-identifiable implied laws must not drive the overall tier (2.1 semantics / 2.3.1 fix)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from moju.monitor import ResidualEngine, audit, evaluate
from moju.monitor.analytic_flows import ns_state_from_analytic, taylor_green_2d

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "goldens" / "not_identifiable" / "tg_re100.json"

_NI_KEY = "constitutive/dynamic_viscosity_from_re/law_momentum_navier_stokes/implied_delta"
_NI_BN = "dynamic_viscosity_from_re/law_momentum_navier_stokes"
_LAW_KEY = "laws/momentum_navier_stokes"


def _ns_sm():
    return {
        "u_t": "u_t",
        "u": "u",
        "u_grad": "u_grad",
        "p_grad": "p_grad",
        "u_laplacian": "u_laplacian",
        "re": "re",
    }


def _ni_state(*, re: float = 100.0, mu: float = 0.01, n: int = 32, t: float = 0.1):
    xs = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    x, y = jnp.meshgrid(xs, xs, indexing="ij")
    fields = taylor_green_2d(x, y, t, re=re)
    state = ns_state_from_analytic(fields, x, y, t=t)
    state.update({"rho": 1.0, "U": 1.0, "L": 1.0, "mu": mu, "re": re})
    return state, x


def _identifiable_state(*, re: float = 10.0, mu: float = 0.1, n: int = 32, t: float = 0.1):
    return _ni_state(re=re, mu=mu, n=n, t=t)


def _engine(**kwargs):
    return ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _ns_sm()}],
        law_implied_audits=True,
        constants={"rho": 1.0, "U": 1.0, "L": 1.0},
        **kwargs,
    )


def _assert_ni_semantics(report: dict, *, golden: dict | None = None) -> None:
    exp = (golden or json.loads(GOLDEN.read_text()))["expected"]
    pk = report["per_key"][_NI_KEY]
    assert pk["admissibility_level"] == exp["implied_admissibility_level"]
    assert pk.get("identifiability_status") == exp["identifiability_status"]
    assert pk["admissibility_score"] is None
    assert pk["admissibility_score"] == exp["implied_admissibility_score"]
    assert report["per_key"][_LAW_KEY]["admissibility_level"].startswith(
        exp["law_admissibility_level_prefix"]
    )
    assert report["overall_admissibility_level"].startswith(
        exp["overall_admissibility_level_prefix"]
    )
    id_map = report.get("implied_identifiability") or {}
    assert id_map[_NI_BN]["status"] == "not_identifiable"
    cats = report.get("per_category") or {}
    if exp["constitutive_category_absent_or_nonzero"]:
        assert "constitutive" not in cats or (
            math.isfinite(float(cats["constitutive"])) and float(cats["constitutive"]) != 0.0
        )


def test_golden_fixture_documents_expected_tier():
    golden = json.loads(GOLDEN.read_text())
    assert golden["keys"]["implied"] == _NI_KEY
    assert golden["expected"]["overall_admissibility_level_prefix"] == "High"


def test_audit_not_identifiable_matches_golden():
    golden = json.loads(GOLDEN.read_text())
    state, _ = _ni_state()
    eng = _engine()
    eng.compute_residuals(state, weighted_norms=False)
    _assert_ni_semantics(audit(eng.log), golden=golden)


def test_evaluate_loop_not_identifiable_matches_golden():
    golden = json.loads(GOLDEN.read_text())
    state, _ = _ni_state()
    reports = evaluate(_engine(), [state], weighted_norms=False, evaluate_debug_loop=True)
    assert reports[0].get("execution") == "loop"
    _assert_ni_semantics(reports[0], golden=golden)


def test_evaluate_batched_not_identifiable_matches_golden():
    golden = json.loads(GOLDEN.read_text())
    state, _ = _ni_state()
    reports = evaluate(_engine(), [state], weighted_norms=False)
    assert reports[0].get("execution") == "batched"
    _assert_ni_semantics(reports[0], golden=golden)


def test_evaluate_batched_with_regions_not_identifiable_matches_golden():
    golden = json.loads(GOLDEN.read_text())
    state, x = _ni_state()
    near = np.asarray(x < jnp.pi)
    regions = {"near": near, "far": ~near}
    eng = _engine(regions=regions, region_min_points=8)
    reports = evaluate(
        eng, [state], weighted_norms=False, regions=regions, region_min_points=8
    )
    assert reports[0].get("execution") == "batched"
    _assert_ni_semantics(reports[0], golden=golden)
    for name, blk in reports[0]["region_results"]["regions"].items():
        pk = blk["per_key"][_NI_KEY]
        assert pk["admissibility_level"] == "Not identifiable", name
        assert pk.get("status") == "not_identifiable", name
        assert pk["admissibility_score"] is None, name
        assert blk["viscosity"]["status"] == "not_identifiable", name


def test_batched_vs_loop_equality_including_not_identifiable():
    state, _ = _ni_state()
    eng = _engine()
    loop = evaluate(eng, [state], weighted_norms=False, evaluate_debug_loop=True)[0]
    bat = evaluate(eng, [state], weighted_norms=False)[0]
    assert bat["execution"] == "batched"
    assert loop["overall_admissibility_level"] == bat["overall_admissibility_level"]
    assert loop["overall_admissibility_score"] == pytest.approx(
        bat["overall_admissibility_score"], rel=0, abs=1e-12
    )
    assert loop["per_key"][_NI_KEY]["admissibility_level"] == bat["per_key"][_NI_KEY][
        "admissibility_level"
    ]
    assert loop["per_category"].keys() == bat["per_category"].keys()
    for cat in loop["per_category"]:
        assert loop["per_category"][cat] == pytest.approx(bat["per_category"][cat])


def test_mixed_batch_identifiable_and_not_identifiable():
    ni, _ = _ni_state()
    ok, _ = _identifiable_state()
    reports = evaluate(_engine(), [ni, ok], weighted_norms=False)
    assert reports[0]["execution"] == "batched"
    _assert_ni_semantics(reports[0])
    pk_ok = reports[1]["per_key"][_NI_KEY]
    assert pk_ok["admissibility_level"].startswith("High")
    assert pk_ok.get("identifiability_status") in (None, "ok")
    assert math.isfinite(float(pk_ok["admissibility_score"]))
    assert reports[1]["overall_admissibility_level"].startswith("High")


def test_insufficient_points_does_not_zero_overall_tier():
    """Region insufficient_points uses NaN scores locally; global tier stays from scored keys."""
    state, x = _identifiable_state()
    tiny = np.zeros(np.asarray(x).shape, dtype=bool)
    tiny.ravel()[:2] = True
    regions = {"tiny": tiny, "rest": ~tiny}
    eng = _engine(regions=regions, region_min_points=16)
    eng.compute_residuals(state, weighted_norms=False)
    rep = audit(eng.log)
    assert rep["overall_admissibility_level"].startswith("High")
    tiny_blk = rep["region_results"]["regions"]["tiny"]
    assert tiny_blk["status"] == "insufficient_points"
    assert tiny_blk["per_key"][_NI_KEY]["admissibility_level"] == "Insufficient points"
    assert tiny_blk["per_key"][_NI_KEY]["admissibility_score"] is None
    bat = evaluate(
        eng, [state], weighted_norms=False, regions=regions, region_min_points=16
    )[0]
    assert bat["overall_admissibility_level"].startswith("High")
    assert bat["region_results"]["regions"]["tiny"]["status"] == "insufficient_points"
    assert bat["region_results"]["regions"]["tiny"]["per_key"][_NI_KEY][
        "admissibility_score"
    ] is None


def test_json_export_ni_and_insufficient_scores_are_strict_null():
    """JSON export must use null (allow_nan=False); never NaN or 0.0."""
    from moju.monitor.monitor_log_export import (
        export_monitor_log,
        monitor_log_export_to_jsonable,
    )

    state, x = _ni_state()
    near = np.asarray(x < jnp.pi)
    tiny = np.zeros(np.asarray(x).shape, dtype=bool)
    tiny.ravel()[:2] = True
    regions = {"near": near & ~tiny, "tiny": tiny, "far": ~near}
    eng = _engine(regions=regions, region_min_points=16)
    eng.compute_residuals(state, weighted_norms=False)
    export = export_monitor_log(eng.log, scope="audit", persist=False)
    blob = monitor_log_export_to_jsonable(export)
    text = json.dumps(blob, allow_nan=False)
    parsed = json.loads(text)
    pk = parsed["summary"]["per_key"][_NI_KEY]
    assert pk["admissibility_level"] == "Not identifiable"
    assert pk["admissibility_score"] is None
    assert pk["admissibility_score"] != 0.0
    # Batched evaluate export path
    bat = evaluate(
        eng, [state], weighted_norms=False, regions=regions, region_min_points=16
    )[0]
    # Re-export via a one-step log-shaped summary: dump report through jsonable helper
    from moju.monitor.report import _residual_dict_to_json_serializable

    bat_json = _residual_dict_to_json_serializable(bat)
    bat_text = json.dumps(bat_json, allow_nan=False)
    bat_parsed = json.loads(bat_text)
    assert bat_parsed["per_key"][_NI_KEY]["admissibility_score"] is None
    tiny_score = bat_parsed["region_results"]["regions"]["tiny"]["per_key"][_NI_KEY][
        "admissibility_score"
    ]
    assert tiny_score is None


def test_equivalent_to_worst_point_does_not_zero_overall_tier():
    """percentile_status=equivalent_to_worst_point is advisory; scores stay finite."""
    state, x = _identifiable_state()
    small = np.zeros(np.asarray(x).shape, dtype=bool)
    small.ravel()[:10] = True
    regions = {"small": small, "big": ~small}
    eng = _engine(
        regions=regions,
        region_min_points=8,
        implied_metric="percentile",
        implied_percentile=95.0,
    )
    eng.compute_residuals(
        state, weighted_norms=False, implied_metric="percentile", implied_percentile=95.0
    )
    rep = audit(eng.log)
    assert rep["overall_admissibility_level"].startswith("High")
    small_blk = rep["region_results"]["regions"]["small"]
    assert small_blk.get("percentile_status") == "equivalent_to_worst_point"
    assert math.isfinite(float(small_blk["per_key"][_NI_KEY]["admissibility_score"]))
    assert "constitutive" in (rep.get("per_category") or {})
    assert float(rep["per_category"]["constitutive"]) != 0.0
