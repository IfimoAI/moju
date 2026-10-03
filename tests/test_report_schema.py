"""Versioned report: schema_version, moju_version, tier definition, typed specs."""

import json
import math

import jax.numpy as jnp
import pytest

import moju
from moju.monitor import (
    REPORT_SCHEMA_VERSION,
    TIER_CUTOFFS,
    TIER_DEFINITION,
    AuditSpec,
    LawSpec,
    ResidualEngine,
    Scoring,
    audit,
    report_json_schema,
    tier_for_score,
)
from moju.monitor.auditor import ADM_HIGH_THRESHOLD, ADM_LOW_THRESHOLD, ADM_MODERATE_THRESHOLD


def _fourier_state(n=32):
    L, rho, cp, k = 0.02, 2700.0, 900.0, 200.0
    alpha = k / (rho * cp)
    x = jnp.linspace(0.0, L, n)
    T_lap = jnp.ones_like(x) * (40.0 / L**2)
    return {
        "x": x,
        "t": jnp.ones_like(x) * 10.0,
        "T": 300.0 + 20.0 * (1.0 - x / L) ** 2,
        "T_t": alpha * T_lap,
        "T_laplacian": T_lap,
        "L": jnp.ones_like(x) * L,
        "k": jnp.ones_like(x) * k,
        "rho": jnp.ones_like(x) * rho,
        "cp": jnp.ones_like(x) * cp,
        "alpha": jnp.ones_like(x) * alpha,
    }


def _engine():
    from moju.monitor import build_minimal_residual_engine

    return build_minimal_residual_engine(law_names=["fourier_conduction"], coord_dimension=1)


def _check_required(report, schema):
    for key in schema["required"]:
        assert key in report, key
    for k, row in report["per_key"].items():
        for rk in schema["properties"]["per_key"]["additionalProperties"]["required"]:
            assert rk in row, (k, rk)


def test_report_header_fields():
    eng = _engine()
    eng.compute_residuals(_fourier_state())
    rep = audit(eng.log)
    assert rep["schema_version"] == REPORT_SCHEMA_VERSION
    assert rep["moju_version"] == moju.__version__
    assert rep["tier_definition"]["meaning"].endswith("not a guarantee of correctness.")
    names = [t["name"] for t in rep["tier_definition"]["tiers"]]
    assert names == ["High Admissibility", "Moderate Admissibility", "Low Admissibility", "Non-Admissible"]


def test_empty_log_report_has_header():
    rep = audit([])
    assert rep["schema_version"] == REPORT_SCHEMA_VERSION
    assert rep["moju_version"] == moju.__version__


def test_tier_cutoffs_match_auditor():
    assert TIER_CUTOFFS["High Admissibility"] == ADM_HIGH_THRESHOLD
    assert TIER_CUTOFFS["Moderate Admissibility"] == ADM_MODERATE_THRESHOLD
    assert TIER_CUTOFFS["Low Admissibility"] == ADM_LOW_THRESHOLD
    assert tier_for_score(0.95) == "High Admissibility"
    assert tier_for_score(float("nan")) == "Unknown"
    assert TIER_DEFINITION["default_scale_k"] == pytest.approx(1e-2)


def test_report_matches_json_schema():
    eng = _engine()
    eng.compute_residuals(_fourier_state())
    rep = audit(eng.log)
    schema = report_json_schema()
    _check_required(rep, schema)
    json_part = {k: rep[k] for k in schema["properties"] if k in rep}
    json.dumps(json_part)
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(json_part, schema)


def test_typed_specs_equal_dict_specs():
    st = _fourier_state()
    law = {"name": "fourier_conduction", "state_map": {a: a for a in ("T_t", "T_laplacian", "fo", "t", "L")}}
    grp = {"name": "fo", "output_key": "fo", "state_map": {"alpha": "alpha", "t": "t", "L": "L"}}
    e1 = ResidualEngine(constants={}, laws=[law], groups=[grp])
    e2 = ResidualEngine(
        constants={},
        laws=[LawSpec(name="fourier_conduction", state_map=law["state_map"])],
        groups=[grp],
    )
    r1 = audit(e1.compute_residuals(st) and e1.log)
    r2 = audit(e2.compute_residuals(st) and e2.log)
    assert math.isclose(r1["overall_admissibility_score"], r2["overall_admissibility_score"])


def test_audit_spec_pred_fn_roundtrip():
    s = AuditSpec(name="x", output_key="k", state_map={"T": "T"}, pred_fn_key="k_fn", pred_state_map={"T": "T"})
    d = s.to_dict()
    assert d["pred_fn_key"] == "k_fn"
    assert AuditSpec.from_dict(d) == s


def test_scoring_validation():
    with pytest.raises(ValueError):
        Scoring("rms")
    with pytest.raises(ValueError):
        Scoring("median", dimensionless=True)
    assert Scoring.coerce({"metric": "worst_point", "scale": 2.0}).scale == 2.0


def test_pdf_includes_version_and_tiers(tmp_path):
    pytest.importorskip("reportlab")
    from moju.monitor.report import write_audit_pdf

    eng = _engine()
    eng.compute_residuals(_fourier_state())
    rep = audit(eng.log)
    out = tmp_path / "r.pdf"
    write_audit_pdf(rep, str(out))
    assert out.read_bytes()[:5] == b"%PDF-"
