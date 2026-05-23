"""Tests for moju.monitor.audit_meta — plain-language audit calibration summaries."""

from __future__ import annotations

import jax.numpy as jnp

from moju.monitor.audit_meta import audit_meta, build_audit_meta, format_audit_meta_plain_summary
from moju.monitor.auditor import ResidualEngine, audit


def _synthetic_log_entry() -> dict:
    return {
        "index": 0,
        "run_mode": "training",
        "monitor_settings": {
            "law_scale_mode": "auto",
            "state_units": "dimensional",
        },
        "scale": {
            "laws/fourier_conduction": 0.042,
            "laws/laplace_equation": 0.01,
            "constitutive/ideal_gas_rho/implied_delta": 0.01,
        },
        "scale_source": {
            "laws/fourier_conduction": "auto",
            "laws/laplace_equation": "auto_fallback",
            "constitutive/ideal_gas_rho/implied_delta": "fixed",
        },
        "nondim_scales": {
            "L_ref": 0.05,
            "alpha_ref": 1e-6,
            "time_scale": "fourier",
        },
        "nondim_scale_source": {
            "L_ref": "state.L",
            "alpha_ref": "k/(rho*cp)",
        },
        "inferred": ["state_units=dimensional: inferred NondimScales (time_scale='fourier')"],
        "omitted": ["some closure omitted"],
        "rms": {
            "laws/fourier_conduction": 0.1,
            "laws/laplace_equation": 0.05,
            "constitutive/ideal_gas_rho/implied_delta": 0.001,
        },
    }


class TestBuildAuditMeta:
    def test_synthetic_log_structured_and_plain(self):
        log = [_synthetic_log_entry()]
        meta = build_audit_meta(log)
        assert meta["entry_index"] == 0
        assert meta["run_mode"] == "training"
        assert meta["monitor_settings"]["law_scale_mode"] == "auto"
        assert meta["monitor_settings"]["state_units"] == "dimensional"
        assert meta["nondim"]["applied"] is True
        assert meta["nondim"]["scales"]["L_ref"] == 0.05
        summary = meta["scale_calibration"]["summary"]
        assert summary["laws_auto"] == 1
        assert summary["laws_auto_fallback"] == 1
        assert summary["closure_fixed"] == 1
        assert meta["plain_summary"]
        assert "1 of 2" in meta["plain_summary"] or "term-balance" in meta["plain_summary"].lower()
        assert "r_ref" in meta["plain_summary"].lower()
        assert "laplace_equation" in meta["plain_summary"]
        per_key = {r["key"]: r for r in meta["scale_calibration"]["per_key"]}
        assert per_key["laws/fourier_conduction"]["scale_source"] == "auto"
        assert "Term-balance" in per_key["laws/fourier_conduction"]["plain"]

    def test_legacy_log_graceful_unknown(self):
        log = [
            {
                "index": 0,
                "scale": {"laws/a": 0.01},
                "rms": {"laws/a": 1.0},
            }
        ]
        meta = build_audit_meta(log)
        per_key = {r["key"]: r for r in meta["scale_calibration"]["per_key"]}
        assert per_key["laws/a"]["scale_source"] == "unknown"
        assert "legacy" in per_key["laws/a"]["plain"].lower()
        assert meta["plain_summary"]

    def test_r_ref_override_in_meta(self):
        log = [_synthetic_log_entry()]
        meta = build_audit_meta(log, r_ref={"laws/fourier_conduction": 0.5})
        per_key = {r["key"]: r for r in meta["scale_calibration"]["per_key"]}
        assert per_key["laws/fourier_conduction"]["scale_source"] == "r_ref"
        assert "Overridden" in per_key["laws/fourier_conduction"]["plain"]
        assert meta["scale_calibration"]["summary"]["r_ref_override"] == 1

    def test_audit_meta_alias_includes_plain_summary(self):
        meta = audit_meta([_synthetic_log_entry()])
        assert "plain_summary" in meta
        assert meta["plain_summary"] == format_audit_meta_plain_summary(meta)

    def test_empty_log(self):
        meta = build_audit_meta([])
        assert meta["entry_index"] is None
        assert "empty" in meta["plain_summary"].lower()

    def test_inferred_law_scale_mode_from_scale_source(self):
        log = [
            {
                "index": 0,
                "scale_source": {
                    "laws/a": "auto",
                    "laws/b": "auto_fallback",
                },
                "scale": {"laws/a": 0.1, "laws/b": 0.01},
                "rms": {"laws/a": 0.1, "laws/b": 0.01},
            }
        ]
        meta = build_audit_meta(log)
        assert meta["monitor_settings"]["law_scale_mode"] == "auto"

    def test_inferred_state_units_nondimensional(self):
        log = [
            {
                "index": 0,
                "scale_source": {"laws/a": "fixed"},
                "scale": {"laws/a": 0.01},
                "rms": {"laws/a": 0.1},
            }
        ]
        meta = build_audit_meta(log)
        assert meta["monitor_settings"]["state_units"] == "nondimensional"
        assert meta["nondim"]["applied"] is False


class TestAuditIntegration:
    def test_audit_includes_audit_meta(self):
        log = [_synthetic_log_entry()]
        report = audit(log)
        assert "audit_meta" in report
        assert report["audit_meta"]["plain_summary"]
        assert report["audit_meta"]["monitor_settings"]["law_scale_mode"] == "auto"
        assert report["audit_meta"]["admissibility"]["available"] is True

    def test_plain_summary_leads_with_admissibility_after_audit(self):
        log = [_synthetic_log_entry()]
        report = audit(log)
        summary = report["audit_meta"]["plain_summary"]
        assert summary.startswith("Overall admissibility:")
        assert "Law scaling" in summary

    def test_plain_summary_r_ref_hint_for_fallback(self):
        meta = build_audit_meta([_synthetic_log_entry()])
        assert "Consider supplying r_ref" in meta["plain_summary"]
        assert "laws/laplace_equation" in meta["plain_summary"]

    def test_plain_summary_no_r_ref_hint_when_overridden(self):
        meta = build_audit_meta(
            [_synthetic_log_entry()],
            r_ref={"laws/laplace_equation": 0.05},
        )
        assert "Consider supplying r_ref" not in meta["plain_summary"]

    def test_plain_summary_eval_mode_uses_categories(self):
        log = [
            {
                **_synthetic_log_entry(),
                "run_mode": "eval",
            }
        ]
        report = audit(log)
        summary = report["audit_meta"]["plain_summary"]
        assert "eval mode" in summary.lower()
        assert "Governing laws" in summary

    def test_plain_summary_without_audit_skips_admissibility(self):
        meta = build_audit_meta([_synthetic_log_entry()])
        assert meta["admissibility"]["available"] is False
        assert not meta["plain_summary"].startswith("Overall admissibility:")

    def test_engine_log_monitor_settings(self):
        core = ResidualEngine(
            laws=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_xx"}}],
            law_scale_mode="fixed",
        )
        core.compute_residuals({"phi_xx": jnp.array(1.0)})
        entry = core.log[-1]
        assert entry["monitor_settings"]["law_scale_mode"] == "fixed"
        assert entry["monitor_settings"]["state_units"] == "nondimensional"
        report = audit(core.log)
        assert report["audit_meta"]["monitor_settings"]["law_scale_mode"] == "fixed"
