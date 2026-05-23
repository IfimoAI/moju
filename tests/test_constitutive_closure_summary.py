"""Tests for constitutive closure summary helpers and band labels."""

import math

import pytest

from moju.monitor.auditor import (
    ADM_HIGH_THRESHOLD,
    ADM_LOW_THRESHOLD,
    ADM_MODERATE_THRESHOLD,
    CONSTITUTIVE_BAND_FRAC_HIGH,
    CONSTITUTIVE_BAND_FRAC_LOW,
    CONSTITUTIVE_BAND_FRAC_MOD,
)
from moju.monitor.constitutive_closure_summary import (
    build_constitutive_closure_summary,
    constitutive_band_label,
    format_constitutive_closure_summary,
    select_worst_implied_delta,
)


def test_admissibility_thresholds_align_with_bands() -> None:
    assert ADM_LOW_THRESHOLD == pytest.approx(0.5)
    assert ADM_MODERATE_THRESHOLD == pytest.approx(2.0 / 3.0)
    assert ADM_HIGH_THRESHOLD == pytest.approx(1.0 / 1.1)


def test_constitutive_band_labels() -> None:
    assert constitutive_band_label(CONSTITUTIVE_BAND_FRAC_HIGH) == "±0.1% green band"
    assert constitutive_band_label(CONSTITUTIVE_BAND_FRAC_MOD) == "±0.1%–±0.5% amber band"
    assert constitutive_band_label(CONSTITUTIVE_BAND_FRAC_LOW) == "±0.5%–±1% red band"
    assert constitutive_band_label(0.02) == "beyond ±1% alarm band"


def test_format_constitutive_closure_summary_at_high_band() -> None:
    r_worst = 0.001
    r_rms = 0.0005
    adm = 1.0 / (1.0 + r_worst / 0.01)
    line = format_constitutive_closure_summary(
        r_worst=r_worst, admissibility_score=adm, r_rms=r_rms
    )
    assert "worst-point fractional error = 0.10%" in line
    assert "±0.1% green band" in line
    assert "RMS = 0.05%" in line
    assert "90.91%" in line
    assert "(High)" in line
    assert "Admissibility (worst-point)" in line


def test_select_worst_implied_delta_picks_largest_r_max() -> None:
    per_key = {
        "constitutive/a/implied_delta": {
            "rms": 0.001,
            "r_max": 0.001,
            "admissibility_score": 0.91,
        },
        "constitutive/b/implied_delta": {
            "rms": 0.002,
            "r_max": 0.008,
            "admissibility_score": 0.56,
        },
    }
    picked = select_worst_implied_delta(per_key)
    assert picked is not None
    key, r_worst, r_rms, adm = picked
    assert key == "constitutive/b/implied_delta"
    assert r_worst == pytest.approx(0.008)
    assert r_rms == pytest.approx(0.002)
    assert adm == pytest.approx(0.56)


def test_build_constitutive_closure_summary_none_without_implied_delta() -> None:
    assert build_constitutive_closure_summary({"laws/a": {"rms": 1.0}}) is None


def test_audit_report_includes_constitutive_closure_summary() -> None:
    from moju.monitor.auditor import audit

    log = [
        {
            "index": 0,
            "rms": {"constitutive/alpha/implied_delta": 0.001},
            "scale": {"constitutive/alpha/implied_delta": 0.01},
        }
    ]
    report = audit(log)
    summary = report.get("constitutive_closure_summary")
    assert summary is not None
    assert "worst-point" in summary
    assert "±0.1% green band" in summary


def test_write_audit_pdf_includes_closure_summary() -> None:
    pytest.importorskip("reportlab")
    import tempfile
    from pathlib import Path

    from moju.monitor.report import write_audit_pdf

    report = {
        "per_key": {},
        "per_category": {"constitutive": 0.91},
        "overall_admissibility_score": 0.91,
        "overall_admissibility_level": "High Admissibility",
        "constitutive_closure_summary": (
            "Constitutive worst-point fractional error = 0.10% (±0.1% green band). "
            "RMS = 0.05%. Admissibility (worst-point) = 90.91% (High)."
        ),
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "report.pdf"
        write_audit_pdf(report, str(path))
        assert path.is_file()
        assert path.stat().st_size > 400
