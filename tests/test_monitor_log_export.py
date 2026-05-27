"""Tests for export_monitor_log (visualize / audit / both scopes)."""

from __future__ import annotations

import copy
import json
import math

import pytest

from moju.monitor.auditor import audit, build_monitor_visualize_bundle
from moju.monitor.monitor_log_export import (
    export_monitor_log,
    get_monitor_log_export,
    monitor_log_export_to_bundle,
    monitor_log_export_to_jsonable,
)


def _two_step_log():
    return [
        {
            "index": 0,
            "run_mode": "training",
            "rms": {"laws/a": 2.0, "constitutive/x/implied_delta": 1.0},
            "scale": {"laws/a": 1.0, "constitutive/x/implied_delta": 0.01},
        },
        {
            "index": 1,
            "run_mode": "training",
            "rms": {"laws/a": 1.0, "constitutive/x/implied_delta": 0.5},
            "scale": {"laws/a": 1.0, "constitutive/x/implied_delta": 0.01},
        },
    ]


def _eval_last_step_log():
    return [
        {
            "index": 0,
            "run_mode": "training",
            "rms": {"laws/a": 2.0, "constitutive/x/implied_delta": 1.0},
            "scale": {"laws/a": 1.0, "constitutive/x/implied_delta": 0.01},
        },
        {
            "index": 1,
            "run_mode": "eval",
            "rms": {
                "laws/a": 1.0,
                "constitutive/x/implied_delta": 0.5,
                "data/u": 0.2,
            },
            "scale": {
                "laws/a": 1.0,
                "constitutive/x/implied_delta": 0.01,
                "data/u": 1.0,
            },
        },
    ]


class TestExportScopes:
    def test_scope_visualize_default(self):
        log = _two_step_log()
        export = export_monitor_log(log, persist=False)
        assert export["scope"] == "visualize"
        assert "bundle" in export
        assert "plot_options" in export
        assert "steps" not in export
        assert "series" not in export
        assert "summary" not in export
        assert export["bundle"]["n"] == 2

    def test_scope_audit_full_log(self):
        log = _two_step_log()
        export = export_monitor_log(log, scope="audit", persist=False)
        assert export["scope"] == "audit"
        assert len(export["steps"]) == len(log)
        assert "series" in export
        assert "summary" in export
        assert "bundle" not in export

    def test_scope_both_has_all_blocks(self):
        log = _two_step_log()
        export = export_monitor_log(log, scope="both", persist=False)
        assert "bundle" in export
        assert "steps" in export
        assert "series" in export
        assert "summary" in export

    def test_audit_series_aligns_with_log(self):
        log = _two_step_log()
        export = export_monitor_log(log, scope="audit", persist=False)
        for i, step in enumerate(export["steps"]):
            assert export["series"]["overall_adm"][i] == step["overall_admissibility_score"]

    def test_eval_visualize_slices_bundle(self):
        log = _eval_last_step_log()
        export = export_monitor_log(log, scope="both", mode="eval", persist=False)
        assert export["bundle"]["n"] == 1
        assert len(export["bundle"]["log"]) == 1
        assert len(export["steps"]) == len(log)

    def test_training_export_n_matches_full_log(self):
        log = _two_step_log()
        export = export_monitor_log(log, mode="training", persist=False)
        assert export["bundle"]["n"] == len(log)
        assert len(export["bundle"]["overall_adm"]) == len(log)

    def test_eval_bar_chart_always(self):
        log = _eval_last_step_log()
        export = export_monitor_log(log, mode="eval", persist=False)
        assert export["bundle"]["use_bar_chart"] is True

    def test_training_bar_chart_only_when_n_eq_1(self):
        log_one = [_two_step_log()[0]]
        log_two = _two_step_log()
        exp_one = export_monitor_log(log_one, persist=False)
        exp_two = export_monitor_log(log_two, persist=False)
        assert exp_one["bundle"]["use_bar_chart"] is True
        assert exp_two["bundle"]["use_bar_chart"] is False

    def test_audit_summary_matches_audit(self):
        log = _two_step_log()
        export = export_monitor_log(log, scope="audit", persist=False)
        report = audit(copy.deepcopy(log))
        summary = export["summary"]
        assert summary["overall_admissibility_score"] == report["overall_admissibility_score"]
        assert summary["overall_admissibility_level"] == report["overall_admissibility_level"]
        assert summary["per_category"] == report["per_category"]
        assert summary["constitutive_closure_summary"] == report["constitutive_closure_summary"]

    def test_eval_cats_fin_includes_data_when_present(self):
        log = _eval_last_step_log()
        export = export_monitor_log(log, mode="eval", persist=False)
        cats = [name for name, _ in export["bundle"]["cats_fin"]]
        assert any("Data" in c or c == "Data" for c in cats)


class TestExportCacheAndEnrich:
    def test_persist_on_last_entry(self):
        log = _two_step_log()
        export_monitor_log(log, persist=True, force=True)
        assert get_monitor_log_export(log) is not None
        assert log[-1]["monitor_log_export"]["scope"] == "visualize"

    def test_cache_hit_skips_rebuild(self, monkeypatch):
        log = _two_step_log()
        calls = {"n": 0}
        from moju.monitor import monitor_log_export as mle

        orig = mle._build_audit_block

        def counted(*args, **kwargs):
            calls["n"] += 1
            return orig(*args, **kwargs)

        monkeypatch.setattr(mle, "_build_audit_block", counted)
        export_monitor_log(log, scope="audit", persist=True, force=True)
        export_monitor_log(log, scope="audit", persist=True, force=False)
        assert calls["n"] == 1

    def test_cache_invalidates_on_scope_change(self):
        log = _two_step_log()
        export_monitor_log(log, scope="visualize", persist=True, force=True)
        export_monitor_log(log, scope="audit", persist=True, force=False)
        cached = get_monitor_log_export(log)
        assert cached is not None
        assert cached["scope"] == "audit"

    def test_enrich_log_writes_scores(self):
        log = _two_step_log()
        for entry in log:
            entry.pop("r_norm", None)
            entry.pop("admissibility_score", None)
        export_monitor_log(log, scope="audit", enrich_log=True, persist=False)
        assert "r_norm" in log[0]
        assert "overall_admissibility_score" in log[1]


class TestExportSerialization:
    def test_jsonable_roundtrip_all_scopes(self):
        log = _two_step_log()
        for scope in ("visualize", "audit", "both"):
            export = export_monitor_log(log, scope=scope, persist=False)
            payload = monitor_log_export_to_jsonable(export)
            json.dumps(payload)

    def test_export_matches_build_monitor_visualize_bundle(self):
        log = _two_step_log()
        direct = build_monitor_visualize_bundle(log, mode="training")
        export = export_monitor_log(log, persist=False)
        bundle = monitor_log_export_to_bundle(export)
        assert bundle is not None
        assert int(bundle["n"]) == int(direct["n"])
        assert list(bundle["overall_adm"]) == list(direct["overall_adm"])
        assert bundle["plot_keys"] == direct["plot_keys"]

    def test_rehydrate_builds_plotly_figure(self):
        pytest.importorskip("plotly")
        from moju.monitor.visualize_plotly import build_plotly_monitor_figure

        log = _two_step_log()
        export = export_monitor_log(log, persist=False)
        bundle = monitor_log_export_to_bundle(export)
        opts = export["plot_options"]
        fig = build_plotly_monitor_figure(
            bundle,
            step_label=opts["step_label"],
            r_norm_scale=opts["r_norm_scale"],
            dashboard_mode=opts["dashboard_mode"],
        )
        assert fig is not None

    def test_monitor_log_export_to_bundle_none_for_audit_only(self):
        log = _two_step_log()
        export = export_monitor_log(log, scope="audit", persist=False)
        assert monitor_log_export_to_bundle(export) is None

    def test_invalid_scope_raises(self):
        log = _two_step_log()
        with pytest.raises(ValueError, match="scope must be"):
            export_monitor_log(log, scope="invalid", persist=False)

    def test_empty_log_visualize_raises(self):
        with pytest.raises(ValueError, match="empty"):
            export_monitor_log([], scope="visualize", persist=False)

    def test_empty_log_audit_ok(self):
        export = export_monitor_log([], scope="audit", persist=False)
        assert export["steps"] == []
        assert export["series"]["indices"] == []
