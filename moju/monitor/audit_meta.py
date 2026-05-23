"""
Plain-language summary of scaling and nondimensionalization decisions for a monitor log.

Use :func:`audit_meta` or :func:`build_audit_meta` to explain how ``scale_k`` and ND
conversion were chosen for a given ``compute_residuals`` step — without parsing raw log dicts.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from moju.monitor.auditor import admissibility_level
from moju.monitor.visualize_labels import format_admissibility_pct

# Match :data:`moju.monitor.auditor.DEFAULT_NONDIM_R_NORM_SCALE_K` (avoid circular import).
_DEFAULT_NONDIM_R_NORM_SCALE_K = 1e-2

_SCALE_SOURCE_PLAIN: Dict[str, str] = {
    "auto": "Term-balance RMS from merged state (governing law).",
    "auto_fallback": "Generic field RMS fallback; floored at the 1e-2 reference gauge.",
    "fixed": "Fixed 1e-2 reference gauge (closure-aligned tier calibration).",
    "state_derived": "RMS of related state or reference fields.",
    "r_ref": "Overridden by audit(..., r_ref=...) for this report.",
    "unknown": "Unknown (legacy log or missing scale_source).",
}

_CATEGORY_LABELS: Dict[str, str] = {
    "laws": "Governing laws",
    "constitutive": "Constitutive",
    "data": "Data",
    "scaling": "Scaling",
}


def _resolve_entry(log: List[Dict[str, Any]], entry_index: int) -> Optional[Dict[str, Any]]:
    if not log:
        return None
    idx = entry_index if entry_index >= 0 else len(log) + entry_index
    if idx < 0 or idx >= len(log):
        return None
    return log[idx]


def _is_closure_key(flat_key: str) -> bool:
    if not flat_key.startswith("constitutive/"):
        return False
    tail = flat_key.split("/", 1)[1]
    return tail.endswith("/implied_delta") or tail.endswith("/ref_delta") or tail.endswith(
        "implied_delta"
    ) or tail.endswith("ref_delta")


def _infer_law_scale_mode(entry: Dict[str, Any]) -> str:
    ms = entry.get("monitor_settings") or {}
    if ms.get("law_scale_mode"):
        return str(ms["law_scale_mode"])
    scale_src = entry.get("scale_source") or {}
    law_sources = [
        str(v)
        for k, v in scale_src.items()
        if k.startswith("laws/") and v is not None
    ]
    if not law_sources:
        return "unknown"
    if any(s in ("auto", "auto_fallback") for s in law_sources):
        return "auto"
    if all(s == "fixed" for s in law_sources):
        return "fixed"
    return "unknown"


def _infer_state_units(entry: Dict[str, Any]) -> str:
    ms = entry.get("monitor_settings") or {}
    if ms.get("state_units"):
        return str(ms["state_units"])
    if entry.get("nondim_scales") or entry.get("nondim_scale_source"):
        return "dimensional"
    return "nondimensional"


def _effective_scale_source(
    flat_key: str,
    entry: Dict[str, Any],
    r_ref: Optional[Dict[str, float]],
) -> str:
    if r_ref is not None and flat_key in r_ref and r_ref[flat_key] is not None and r_ref[flat_key] > 0:
        return "r_ref"
    scale_src = entry.get("scale_source") or {}
    if flat_key in scale_src and scale_src[flat_key]:
        return str(scale_src[flat_key])
    return "unknown"


def _plain_for_source(source: str) -> str:
    return _SCALE_SOURCE_PLAIN.get(source, _SCALE_SOURCE_PLAIN["unknown"])


def _build_nondim_block(entry: Dict[str, Any], state_units: str) -> Dict[str, Any]:
    scales = dict(entry.get("nondim_scales") or {})
    sources = dict(entry.get("nondim_scale_source") or {})
    applied = bool(scales or sources) or state_units == "dimensional"
    lines: List[str] = []
    time_scale = scales.get("time_scale")
    if applied and scales:
        lines.append(
            "Physical (SI) state was converted to nondimensional form before laws "
            "(groups ran on physical state first)."
        )
        if time_scale:
            lines.append(f"Time convention: {time_scale!r}.")
        for fld in ("L_ref", "U_ref", "dT_ref", "alpha_ref", "D_ref", "c_ref", "rho_ref", "T0"):
            if fld in scales:
                src = sources.get(fld, "inferred")
                lines.append(f"  {fld} = {scales[fld]:g} (from {src}).")
    elif state_units == "dimensional" and not scales:
        lines.append(
            "state_units was dimensional but no resolved nondim_scales were logged on this entry."
        )
    else:
        lines.append(
            "State treated as nondimensional; no dimensional-to-ND conversion was logged."
        )
    return {
        "applied": applied and bool(scales),
        "time_scale": time_scale,
        "scales": scales,
        "sources": sources,
        "lines": lines,
    }


def _build_scale_calibration(
    entry: Dict[str, Any],
    r_ref: Optional[Dict[str, float]],
) -> Dict[str, Any]:
    scales = entry.get("scale") or {}
    flat_keys = sorted(set(scales.keys()) | set((entry.get("scale_source") or {}).keys()))
    summary: Dict[str, int] = {
        "laws_auto": 0,
        "laws_auto_fallback": 0,
        "laws_fixed": 0,
        "closure_fixed": 0,
        "state_derived": 0,
        "r_ref_override": 0,
        "unknown": 0,
    }
    per_key: List[Dict[str, Any]] = []
    for k in flat_keys:
        src = _effective_scale_source(k, entry, r_ref)
        try:
            scale_k = float(scales.get(k, float("nan")))
        except (TypeError, ValueError):
            scale_k = float("nan")
        plain = _plain_for_source(src)
        row = {"key": k, "scale_k": scale_k, "scale_source": src, "plain": plain}
        per_key.append(row)
        if src == "r_ref":
            summary["r_ref_override"] += 1
        elif k.startswith("laws/"):
            if src == "auto":
                summary["laws_auto"] += 1
            elif src == "auto_fallback":
                summary["laws_auto_fallback"] += 1
            elif src == "fixed":
                summary["laws_fixed"] += 1
            else:
                summary["unknown"] += 1
        elif _is_closure_key(k) and src == "fixed":
            summary["closure_fixed"] += 1
        elif src == "state_derived":
            summary["state_derived"] += 1
        elif src == "unknown":
            summary["unknown"] += 1
    return {"summary": summary, "per_key": per_key}


def _build_pipeline_block(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "inferred": list(entry.get("inferred") or []),
        "omitted": list(entry.get("omitted") or []),
        "unresolved_dependencies": list(entry.get("unresolved_dependencies") or []),
    }


def _build_admissibility_block(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Admissibility scores from log entry after ``audit()`` (optional)."""
    overall_raw = entry.get("overall_admissibility_score")
    per_cat = dict(entry.get("category_admissibility_score") or {})
    run_mode = entry.get("run_mode")
    try:
        overall = float(overall_raw) if overall_raw is not None else float("nan")
    except (TypeError, ValueError):
        overall = float("nan")
    overall_finite = math.isfinite(overall)
    category_parts: List[str] = []
    for cat in ("laws", "constitutive", "data", "scaling"):
        if cat not in per_cat:
            continue
        try:
            score = float(per_cat[cat])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score):
            continue
        label = _CATEGORY_LABELS.get(cat, cat)
        category_parts.append(f"{label} {format_admissibility_pct(score)}")
    available = overall_finite or bool(category_parts)
    block: Dict[str, Any] = {
        "available": available,
        "run_mode": run_mode,
        "overall_score": overall if overall_finite else None,
        "overall_pct": format_admissibility_pct(overall) if overall_finite else None,
        "overall_level": admissibility_level(overall) if overall_finite else None,
        "overall_finite": overall_finite,
        "per_category": per_cat,
        "category_parts": category_parts,
    }
    return block


def _law_rows(per_key: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in per_key if str(r.get("key", "")).startswith("laws/")]


def format_audit_meta_plain_summary(meta: Dict[str, Any]) -> str:
    """Two-to-five sentence executive summary from :func:`build_audit_meta` output."""
    if not meta:
        return "No audit metadata available (empty log)."
    parts: List[str] = []

    adm = meta.get("admissibility") or {}
    run_mode = adm.get("run_mode") or meta.get("run_mode")
    if adm.get("available"):
        if run_mode == "eval":
            cat_parts = adm.get("category_parts") or []
            if cat_parts:
                parts.append(
                    "No single overall score in eval mode; "
                    + "; ".join(cat_parts)
                    + "."
                )
            elif adm.get("overall_finite"):
                parts.append(
                    f"Overall admissibility: {adm['overall_pct']} ({adm['overall_level']})."
                )
        elif adm.get("overall_finite"):
            parts.append(
                f"Overall admissibility: {adm['overall_pct']} ({adm['overall_level']})."
            )
        elif adm.get("category_parts"):
            parts.append("; ".join(adm["category_parts"]) + ".")

    ms = meta.get("monitor_settings") or {}
    law_mode = ms.get("law_scale_mode", "unknown")
    sc = meta.get("scale_calibration") or {}
    sm = sc.get("summary") or {}
    law_rows = _law_rows(sc.get("per_key") or [])
    n_laws = len(law_rows)
    n_auto = int(sm.get("laws_auto") or 0)
    n_fallback = int(sm.get("laws_auto_fallback") or 0)
    n_fixed_laws = int(sm.get("laws_fixed") or 0)

    if n_laws > 0:
        if law_mode == "auto":
            if n_fallback:
                fallback_keys = [
                    str(r["key"]) for r in law_rows if r.get("scale_source") == "auto_fallback"
                ]
                key_hint = f" ({', '.join(fallback_keys)})" if fallback_keys else ""
                parts.append(
                    f"Law scaling (auto mode): {n_auto} of {n_laws} governing law(s) used "
                    f"term-balance auto scaling; {n_fallback} fell back to the "
                    f"{_DEFAULT_NONDIM_R_NORM_SCALE_K:g} floor{key_hint}."
                )
                if fallback_keys and not int(sm.get("r_ref_override") or 0):
                    parts.append(
                        "Consider supplying r_ref for fallback law key(s): "
                        + ", ".join(fallback_keys)
                        + "."
                    )
            elif n_auto == n_laws:
                parts.append(
                    f"Law scaling (auto mode): all {n_laws} governing law(s) used "
                    "term-balance auto scaling."
                )
            else:
                bits: List[str] = []
                if n_auto:
                    bits.append(f"{n_auto} auto term-balance")
                if n_fixed_laws:
                    bits.append(f"{n_fixed_laws} fixed {_DEFAULT_NONDIM_R_NORM_SCALE_K:g} gauge")
                parts.append(
                    f"Law scaling (auto mode): {n_laws} governing law(s): "
                    + ", ".join(bits)
                    + "."
                )
        elif law_mode == "fixed":
            parts.append(
                f"Law scaling: fixed {_DEFAULT_NONDIM_R_NORM_SCALE_K:g} gauge on "
                f"{n_laws} governing law(s)."
            )
        else:
            parts.append(f"Law scaling mode: {law_mode!r} ({n_laws} governing law(s)).")

    if int(sm.get("r_ref_override") or 0):
        parts.append(
            f"{sm['r_ref_override']} key(s) use audit r_ref overrides for this report."
        )

    state_units = ms.get("state_units", "unknown")
    nondim = meta.get("nondim") or {}
    if nondim.get("applied"):
        nd_lines = nondim.get("lines") or []
        if nd_lines:
            parts.append(nd_lines[0])
    elif state_units == "dimensional":
        parts.append("State units: dimensional (SI).")

    pipeline = meta.get("pipeline") or {}
    n_om = len(pipeline.get("omitted") or [])
    if n_om:
        parts.append(
            f"Pipeline: {n_om} omitted closure(s) — check log omitted / inferred fields."
        )

    if not parts:
        parts.append(
            f"Governing-law scale mode: {law_mode!r}; state units: {state_units!r}."
        )
    return " ".join(parts)


def _build_plain_sections(
    meta: Dict[str, Any],
) -> Dict[str, str]:
    sc = meta.get("scale_calibration") or {}
    per_key = sc.get("per_key") or []
    scaling_lines = [
        f"- {row['key']}: scale_k ≈ {row['scale_k']:.4g} — {row['plain']}"
        for row in per_key
        if row.get("key", "").startswith(("laws/", "constitutive/", "data/"))
    ]
    nondim_lines = list((meta.get("nondim") or {}).get("lines") or [])
    pipeline = meta.get("pipeline") or {}
    pipe_lines: List[str] = []
    for msg in (pipeline.get("inferred") or [])[:20]:
        pipe_lines.append(f"- Inferred: {msg}")
    for msg in (pipeline.get("omitted") or [])[:20]:
        pipe_lines.append(f"- Omitted: {msg}")
    for item in (pipeline.get("unresolved_dependencies") or [])[:10]:
        if isinstance(item, dict):
            pipe_lines.append(
                f"- Unresolved {item.get('stage', '?')}/{item.get('name', '?')}: "
                f"missing {item.get('missing_keys', [])}"
            )
    return {
        "scaling": "\n".join(scaling_lines) if scaling_lines else "No per-key scale data.",
        "nondim": "\n".join(nondim_lines) if nondim_lines else "No nondimensional conversion logged.",
        "pipeline": "\n".join(pipe_lines) if pipe_lines else "No pipeline inferred/omitted messages.",
    }


def build_audit_meta(
    log: List[Dict[str, Any]],
    *,
    entry_index: int = -1,
    r_ref: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Build structured + plain-language metadata explaining scaling and ND decisions.

    Parameters
    ----------
    log:
        Session log from :meth:`ResidualEngine.compute_residuals`.
    entry_index:
        Which step to explain (default ``-1`` = last entry).
    r_ref:
        Optional audit-time overrides (same as :func:`audit`); keys with ``r_ref`` entries
        are reported with ``scale_source: \"r_ref\"``.
    """
    entry = _resolve_entry(log, entry_index)
    if entry is None:
        return {
            "entry_index": None,
            "run_mode": None,
            "monitor_settings": {},
            "admissibility": {"available": False},
            "nondim": {"applied": False, "lines": ["Empty log."]},
            "scale_calibration": {"summary": {}, "per_key": []},
            "pipeline": {"inferred": [], "omitted": [], "unresolved_dependencies": []},
            "plain_summary": "No audit metadata available (empty log).",
            "plain_sections": {},
        }
    idx = entry_index if entry_index >= 0 else len(log) + entry_index
    law_mode = _infer_law_scale_mode(entry)
    state_units = _infer_state_units(entry)
    monitor_settings = dict(entry.get("monitor_settings") or {})
    if not monitor_settings.get("law_scale_mode"):
        monitor_settings["law_scale_mode"] = law_mode
    if not monitor_settings.get("state_units"):
        monitor_settings["state_units"] = state_units

    meta: Dict[str, Any] = {
        "entry_index": idx,
        "run_mode": entry.get("run_mode"),
        "monitor_settings": monitor_settings,
        "admissibility": _build_admissibility_block(entry),
        "nondim": _build_nondim_block(entry, state_units),
        "scale_calibration": _build_scale_calibration(entry, r_ref),
        "pipeline": _build_pipeline_block(entry),
    }
    meta["plain_sections"] = _build_plain_sections(meta)
    meta["plain_summary"] = format_audit_meta_plain_summary(meta)
    return meta


def audit_meta(
    log: List[Dict[str, Any]],
    *,
    entry_index: int = -1,
    r_ref: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Convenience alias for :func:`build_audit_meta` (includes ``plain_summary``)."""
    return build_audit_meta(log, entry_index=entry_index, r_ref=r_ref)
