"""
Export monitor log data for external reuse (plots, analytics, audit summaries).

``export_monitor_log`` supports ``scope="visualize"`` (plot bundle), ``scope="audit"``
(full-log steps/series/summary), or ``scope="both"``. Does not replace :func:`audit` or
:func:`visualize`.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_VALID_SCOPES = frozenset({"visualize", "audit", "both"})
_MONITOR_LOG_EXPORT_KEY = "monitor_log_export"
_BUNDLE_NUMPY_KEYS = frozenset(
    {
        "indices",
        "r_norm_mat",
        "overall_adm",
        "bar_values",
        "bar_values_eff",
    }
)
_CATEGORY_TRAINING_ARRAY_KEYS = frozenset({"r_norm_mat", "r_eff_mat"})


def _normalize_scope(scope: str) -> str:
    s = str(scope).strip().lower()
    if s not in _VALID_SCOPES:
        raise ValueError(f"scope must be one of {sorted(_VALID_SCOPES)!r}, got {scope!r}")
    return s


def _fingerprint(
    *,
    scope: str,
    mode: Optional[str],
    r_ref: Optional[Dict[str, float]],
    keys: Optional[List[str]],
    max_legend_keys: int,
    show_state_overlay: bool,
    spatial_coord_key: str,
    spatial_prefer_last_t: bool,
    worst_keys_top_n: int,
    enrich_log: bool,
) -> Dict[str, Any]:
    return {
        "scope": scope,
        "mode": mode,
        "r_ref": dict(r_ref) if r_ref else None,
        "keys": list(keys) if keys is not None else None,
        "max_legend_keys": int(max_legend_keys),
        "show_state_overlay": bool(show_state_overlay),
        "spatial_coord_key": str(spatial_coord_key),
        "spatial_prefer_last_t": bool(spatial_prefer_last_t),
        "worst_keys_top_n": int(worst_keys_top_n),
        "enrich_log": bool(enrich_log),
    }


def _fingerprints_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return a == b


def get_monitor_log_export(log: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return persisted export on the last log entry, if any."""
    if not log:
        return None
    last = log[-1]
    if not isinstance(last, dict):
        return None
    exp = last.get(_MONITOR_LOG_EXPORT_KEY)
    return exp if isinstance(exp, dict) else None


def _cache_valid(
    cached: Dict[str, Any],
    *,
    log_len: int,
    fingerprint: Dict[str, Any],
) -> bool:
    if cached.get("version") != 1:
        return False
    if cached.get("source_log_len") != log_len:
        return False
    fp = cached.get("fingerprint")
    if not isinstance(fp, dict):
        return False
    return _fingerprints_equal(fp, fingerprint)


def _apply_step_metrics_to_log(
    log: List[Dict[str, Any]],
    metrics: List[Dict[str, Any]],
) -> None:
    for entry, m in zip(log, metrics):
        if not isinstance(entry, dict):
            continue
        entry["r_norm"] = dict(m.get("r_norm") or {})
        entry["admissibility_score"] = dict(m.get("admissibility_score") or {})
        entry["category_admissibility_score"] = dict(m.get("category_admissibility_score") or {})
        entry["overall_admissibility_score"] = m.get("overall_admissibility_score")


def _collect_flat_keys(log: List[Dict[str, Any]]) -> List[str]:
    keys: List[str] = []
    seen: set = set()
    for entry in log:
        rms = entry.get("rms") if isinstance(entry, dict) else None
        if not isinstance(rms, dict):
            continue
        for k in rms:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    return keys


def _build_steps(
    log: List[Dict[str, Any]],
    metrics: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    for entry, m in zip(log, metrics):
        idx = entry.get("index") if isinstance(entry, dict) else None
        if idx is None:
            idx = len(steps)
        steps.append(
            {
                "index": idx,
                "overall_admissibility_score": float(m.get("overall_admissibility_score", float("nan"))),
                "category_admissibility_score": dict(m.get("category_admissibility_score") or {}),
                "admissibility_score": dict(m.get("admissibility_score") or {}),
                "r_norm": dict(m.get("r_norm") or {}),
                "per_key_report": dict(m.get("per_key_report") or {}),
            }
        )
    return steps


def _build_series(
    log: List[Dict[str, Any]],
    metrics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    n = len(log)
    indices = list(range(n))
    overall_adm: List[float] = []
    category_adm: Dict[str, List[float]] = {}
    flat_keys = _collect_flat_keys(log)
    r_norm_series: Dict[str, List[float]] = {k: [] for k in flat_keys}
    r_eff_series: Dict[str, List[float]] = {k: [] for k in flat_keys}

    for j, (entry, m) in enumerate(zip(log, metrics)):
        overall_adm.append(float(m.get("overall_admissibility_score", float("nan"))))
        cat_scores = m.get("category_admissibility_score") or {}
        if isinstance(cat_scores, dict):
            for cat, val in cat_scores.items():
                category_adm.setdefault(str(cat), [])
                while len(category_adm[str(cat)]) < j:
                    category_adm[str(cat)].append(float("nan"))
                category_adm[str(cat)].append(float(val))
        rms_j = (entry.get("rms") or {}) if isinstance(entry, dict) else {}
        rn_j = m.get("r_norm") or {}
        for k in flat_keys:
            rv = rn_j.get(k)
            r_norm_series[k].append(float(rv) if rv is not None else float("nan"))
            rv_eff = rms_j.get(k)
            try:
                r_eff_series[k].append(float(rv_eff) if rv_eff is not None else float("nan"))
            except (TypeError, ValueError):
                r_eff_series[k].append(float("nan"))

    for cat in list(category_adm.keys()):
        while len(category_adm[cat]) < n:
            category_adm[cat].append(float("nan"))

    return {
        "indices": indices,
        "overall_adm": overall_adm,
        "category_adm": category_adm,
        "r_norm": r_norm_series,
        "r_eff": r_eff_series,
    }


def _build_summary(
    log: List[Dict[str, Any]],
    metrics: List[Dict[str, Any]],
    *,
    r_ref: Optional[Dict[str, float]],
    keys: Optional[List[str]],
    worst_keys_top_n: int,
) -> Dict[str, Any]:
    from moju.monitor.audit_meta import audit_meta
    from moju.monitor.auditor import _worst_keys_table_rows, admissibility_level
    from moju.monitor.constitutive_closure_summary import build_constitutive_closure_summary

    if not log or not metrics:
        return {
            "per_key": {},
            "per_category": {},
            "overall_admissibility_score": 0.0,
            "overall_admissibility_level": admissibility_level(0.0),
            "constitutive_closure_summary": "",
            "audit_meta": audit_meta(log, r_ref=r_ref) if log else {},
            "monitor_run_mode": None,
            "worst_keys_rows": [],
            "plot_keys": [],
        }

    last_m = metrics[-1]
    last_report_per_key = dict(last_m.get("per_key_report") or {})
    overall = float(last_m.get("overall_admissibility_score", 0.0))
    if not math.isfinite(overall):
        overall = 0.0
    per_category = dict(last_m.get("category_admissibility_score") or {})
    plot_keys = list(keys) if keys is not None else list((log[0].get("rms") or {}).keys())
    worst_rows = _worst_keys_table_rows(log, plot_keys, metrics, top_n=worst_keys_top_n)

    return {
        "per_key": last_report_per_key,
        "per_category": per_category,
        "overall_admissibility_score": overall,
        "overall_admissibility_level": admissibility_level(overall),
        "constitutive_closure_summary": build_constitutive_closure_summary(last_report_per_key),
        "audit_meta": audit_meta(log, r_ref=r_ref),
        "monitor_run_mode": log[-1].get("run_mode") if isinstance(log[-1], dict) else None,
        "worst_keys_rows": worst_rows,
        "plot_keys": plot_keys,
    }


def _build_audit_block(
    log: List[Dict[str, Any]],
    *,
    r_ref: Optional[Dict[str, float]],
    keys: Optional[List[str]],
    worst_keys_top_n: int,
    enrich_log: bool,
) -> Dict[str, Any]:
    from moju.monitor.auditor import _compute_log_step_metrics

    metrics = _compute_log_step_metrics(log, r_ref)
    if enrich_log:
        _apply_step_metrics_to_log(log, metrics)
    return {
        "steps": _build_steps(log, metrics),
        "series": _build_series(log, metrics),
        "summary": _build_summary(
            log,
            metrics,
            r_ref=r_ref,
            keys=keys,
            worst_keys_top_n=worst_keys_top_n,
        ),
    }


def _to_jsonable(obj: Any) -> Any:
    from moju.monitor.report import _residual_dict_to_json_serializable

    if obj is None:
        return None
    if isinstance(obj, dict):
        return _residual_dict_to_json_serializable(obj)
    if hasattr(obj, "tolist"):
        try:
            import numpy as np

            return np.asarray(obj).tolist()
        except Exception:
            return obj
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj


def _jsonify_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in bundle.items() if k not in ("np", "residuals")}
    return _to_jsonable(out)


def _rehydrate_category_training(cat_training: Any) -> Dict[str, Any]:
    import numpy as np

    if not isinstance(cat_training, dict):
        return {}
    out: Dict[str, Any] = {}
    for cat, info in cat_training.items():
        if not isinstance(info, dict):
            out[cat] = info
            continue
        block = dict(info)
        for key in _CATEGORY_TRAINING_ARRAY_KEYS:
            raw = block.get(key)
            if raw is not None:
                block[key] = np.asarray(raw, dtype=float)
        out[cat] = block
    return out


def monitor_log_export_to_bundle(export: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Rehydrate ``export["bundle"]`` with numpy arrays for :func:`moju.monitor.visualize`.

    Returns ``None`` when the export has no bundle block (``scope="audit"`` only).
    """
    import numpy as np

    raw = export.get("bundle")
    if not isinstance(raw, dict):
        return None
    bundle = copy.deepcopy(raw)
    for key in _BUNDLE_NUMPY_KEYS:
        if key in bundle and bundle[key] is not None:
            bundle[key] = np.asarray(bundle[key], dtype=float)
    if "category_training" in bundle:
        bundle["category_training"] = _rehydrate_category_training(bundle["category_training"])
    bundle["np"] = np
    n = int(bundle.get("n") or 0)
    if "indices" not in bundle or bundle["indices"] is None:
        bundle["indices"] = np.arange(n, dtype=float)
    return bundle


def monitor_log_export_to_jsonable(export: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-serializable copy of the export dict."""
    return _to_jsonable(export)


def export_monitor_log(
    log: List[Dict[str, Any]],
    *,
    scope: str = "visualize",
    r_ref: Optional[Dict[str, float]] = None,
    mode: str = "training",
    keys: Optional[List[str]] = None,
    max_legend_keys: int = 16,
    residuals: Optional[Dict[str, Any]] = None,
    state_pred: Optional[Mapping[str, Any]] = None,
    spatial_coord_key: str = "x",
    spatial_prefer_last_t: bool = True,
    engine: Optional[Any] = None,
    worst_keys_top_n: int = 12,
    show_state_overlay: bool = False,
    enrich_log: bool = False,
    step_label: str = "Step",
    r_norm_scale: str = "log",
    dashboard_mode: str = "single-figure",
    persist: bool = True,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Build monitor log export for external reuse.

    ``scope``:

    - ``"visualize"`` (default): plot-ready bundle (same rules as :func:`visualize`).
    - ``"audit"``: full-log ``steps``, ``series``, and ``summary`` (audit-compatible).
    - ``"both"``: visualize bundle and audit blocks together.

    When ``persist=True``, stores the result on ``log[-1]["monitor_log_export"]`` when the log
    is non-empty. A cached export is returned when ``source_log_len`` and ``fingerprint`` match.
    """
    scope_n = _normalize_scope(scope)
    from moju.monitor.auditor import _normalize_visualize_mode, build_monitor_visualize_bundle

    mode_eff = _normalize_visualize_mode(mode) if scope_n in ("visualize", "both") else None
    fingerprint = _fingerprint(
        scope=scope_n,
        mode=mode_eff,
        r_ref=r_ref,
        keys=keys,
        max_legend_keys=max_legend_keys,
        show_state_overlay=show_state_overlay,
        spatial_coord_key=spatial_coord_key,
        spatial_prefer_last_t=spatial_prefer_last_t,
        worst_keys_top_n=worst_keys_top_n,
        enrich_log=enrich_log,
    )

    if not force:
        cached = get_monitor_log_export(log)
        if cached is not None and _cache_valid(cached, log_len=len(log), fingerprint=fingerprint):
            return cached

    export: Dict[str, Any] = {
        "version": 1,
        "scope": scope_n,
        "source_log_len": len(log),
        "fingerprint": fingerprint,
    }

    if scope_n in ("audit", "both"):
        export.update(
            _build_audit_block(
                log,
                r_ref=r_ref,
                keys=keys,
                worst_keys_top_n=worst_keys_top_n,
                enrich_log=enrich_log,
            )
        )

    if scope_n in ("visualize", "both"):
        eff_residuals = residuals
        if eff_residuals is None and engine is not None:
            eff_residuals = getattr(engine, "last_residuals", None)
        bundle = build_monitor_visualize_bundle(
            log,
            keys,
            r_ref,
            max_legend_keys,
            mode=mode_eff or "training",
            residuals=eff_residuals,
            state_pred=state_pred,
            spatial_coord_key=spatial_coord_key,
            spatial_prefer_last_t=spatial_prefer_last_t,
            engine=engine,
            worst_keys_top_n=worst_keys_top_n,
            show_state_overlay=show_state_overlay,
        )
        if bundle is None:
            raise ValueError("Cannot build visualize export: log is empty or has no residual keys.")
        export["plot_options"] = {
            "step_label": step_label,
            "r_norm_scale": r_norm_scale,
            "dashboard_mode": dashboard_mode,
        }
        export["bundle"] = _jsonify_bundle(bundle)

    if persist and log:
        log[-1][_MONITOR_LOG_EXPORT_KEY] = export

    return export


__all__ = [
    "export_monitor_log",
    "get_monitor_log_export",
    "monitor_log_export_to_jsonable",
    "monitor_log_export_to_bundle",
]
