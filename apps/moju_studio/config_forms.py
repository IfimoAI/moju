"""
Pure helpers to build MonitorConfig fragments from form-like data (testable without Streamlit).

Signature-driven arg lists for Laws, Groups, Models, and full declarative AuditSpec dicts.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from moju.monitor.closure_registry import GROUP_FNS, MODEL_FNS
from moju.monitor.path_b_derivatives import PathBGridConfig
from moju.piratio.groups import Groups
from moju.piratio.laws import Laws


def _positional_param_names(fn: Any) -> List[str]:
    sig = inspect.signature(fn)
    names: List[str] = []
    for p in sig.parameters.values():
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            names.append(p.name)
    return names


def law_parameter_names(law_name: str) -> List[str]:
    fn = getattr(Laws, law_name)
    return _positional_param_names(fn)


def group_parameter_names(group_name: str) -> List[str]:
    fn = getattr(Groups, group_name)
    return _positional_param_names(fn)


def model_parameter_names(model_name: str) -> List[str]:
    if model_name not in MODEL_FNS:
        raise KeyError(f"unknown model {model_name!r}")
    fn, _ = MODEL_FNS[model_name]
    return _positional_param_names(fn)


def scaling_fn_parameter_names(group_name: str) -> List[str]:
    if group_name not in GROUP_FNS:
        raise KeyError(f"unknown group {group_name!r}")
    fn, _ = GROUP_FNS[group_name]
    return _positional_param_names(fn)


def build_law_spec(law_name: str, state_map: Dict[str, str]) -> Dict[str, Any]:
    return {"name": law_name, "state_map": dict(state_map)}


def build_group_spec(group_name: str, output_key: str, state_map: Dict[str, str]) -> Dict[str, Any]:
    return {"name": group_name, "output_key": output_key, "state_map": dict(state_map)}


def build_audit_spec_dict(
    *,
    category: str,
    name: str,
    output_key: str,
    state_map: Dict[str, str],
    implied_value_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a JSON-compatible dict for MonitorConfig.from_dict (constitutive row)."""
    if category != "constitutive":
        raise ValueError("category must be constitutive")
    d: Dict[str, Any] = {
        "name": name,
        "output_key": output_key,
        "state_map": dict(state_map),
    }
    if implied_value_key:
        d["implied_value_key"] = implied_value_key
    return d


def path_b_grid_from_options(
    *,
    layout: str = "meshgrid",
    spatial_dimension: Union[int, str] = "auto",
    steady: bool = True,
    key_x: str = "x",
    key_y: str = "y",
    key_z: str = "z",
    key_t: str = "t",
) -> PathBGridConfig:
    return PathBGridConfig(
        layout=layout,  # type: ignore[arg-type]
        spatial_dimension=spatial_dimension,  # type: ignore[arg-type]
        steady=bool(steady),
        key_x=key_x,
        key_y=key_y,
        key_z=key_z,
        key_t=key_t,
    )


def merge_simple_config_with_json_override(
    simple: Dict[str, Any],
    override_json: str,
) -> Dict[str, Any]:
    """
    Start from ``simple`` fragment. For each of ``laws``, ``groups``, ``constitutive_audit``,
    and ``derived_state_chain``, if that key is present in the parsed
    override JSON, the override list replaces the form-built list (including explicit ``[]``).
    ``constants`` are shallow-merged; ``primary_fields`` are replaced if present in the override.
    """
    out = {
        "laws": list(simple.get("laws") or []),
        "groups": list(simple.get("groups") or []),
        "constitutive_audit": list(simple.get("constitutive_audit") or []),
        "constants": dict(simple.get("constants") or {}),
        "primary_fields": list(simple.get("primary_fields") or []),
        "derived_state_chain": list(simple.get("derived_state_chain") or []),
        "law_implied_audits": bool(simple.get("law_implied_audits", True)),
    }
    raw = (override_json or "").strip()
    if not raw:
        return out
    j = json.loads(raw)
    if not isinstance(j, dict):
        raise ValueError("JSON override must be an object")

    for key in ("laws", "groups", "constitutive_audit", "derived_state_chain"):
        if key in j and j[key] is not None:
            out[key] = list(j[key])

    if "constants" in j and isinstance(j["constants"], dict):
        merged = dict(out["constants"])
        merged.update(j["constants"])
        out["constants"] = merged

    if "primary_fields" in j and j["primary_fields"] is not None:
        out["primary_fields"] = list(j["primary_fields"])

    if "law_implied_audits" in j and j["law_implied_audits"] is not None:
        out["law_implied_audits"] = bool(j["law_implied_audits"])

    return out


def reindex_log_entries(existing: List[Dict[str, Any]], new_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Append ``new_entries`` to ``existing``, reassigning monotonic ``index`` for Studio session viz."""
    base = max((int(e.get("index", -1)) for e in existing), default=-1) + 1
    out = list(existing)
    for i, e in enumerate(new_entries):
        e2 = dict(e)
        e2["index"] = base + i
        out.append(e2)
    return out


def preflight_checklist_text(
    required_state: Sequence[str],
    required_deriv: Sequence[str],
    npz_keys: Sequence[str],
    *,
    available_keys: Optional[Sequence[str]] = None,
) -> str:
    """Human-readable checklist for download.

    If ``available_keys`` is set (e.g. dependency planner ``effective_available_keys``), ``[x]``
    marks keys present in NPZ **or** Constants (plus alias expansion). Otherwise only ``npz_keys``
    are used for checkmarks.
    """
    nk = set(available_keys) if available_keys is not None else set(npz_keys)
    lines = ["# Moju Studio preflight checklist", ""]
    lines.append("## Required state keys")
    for k in sorted(required_state):
        lines.append(f"- {'[x]' if k in nk else '[ ]'} {k}")
    lines.append("")
    lines.append("## Required derivative keys (for configured audits)")
    for k in sorted(required_deriv):
        lines.append(f"- {'[x]' if k in nk else '[ ]'} {k}")
    lines.append("")
    if available_keys is not None:
        lines.append(
            "_Checkmarks use merged **NPZ ∪ Constants** key names (with built-in aliases to canonical keys)._"
        )
    else:
        lines.append(
            "_Checkmarks use **NPZ keys only**; pass ``available_keys`` from the dependency planner to include Constants._"
        )
    return "\n".join(lines)


def preflight_checklist_with_dependency_plan(
    required_state: Sequence[str],
    required_deriv: Sequence[str],
    npz_keys: Sequence[str],
    plan_markdown: str,
    *,
    available_keys: Optional[Sequence[str]] = None,
) -> str:
    """Append Studio dependency planner output (markdown) to the downloadable checklist."""
    base = preflight_checklist_text(
        required_state, required_deriv, npz_keys, available_keys=available_keys
    )
    return f"{base}\n\n## Dependency planner (FD / aliases)\n\n{plan_markdown.strip()}\n"
