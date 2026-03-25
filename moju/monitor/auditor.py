"""
ResidualEngine: residuals for governing laws, constitutive, and scaling/similarity audits.

- compute_residuals(state_pred, state_ref=None, *, log_to_python=True)
- build_loss: cascaded RMS over laws only (training).
- audit / visualize: same metrics (RMS, R_norm, admissibility) for all residual keys;
  visualize builds training/test dashboards (optional spatial law panel for x slices).

Constitutive and scaling/similarity audits are tied to Models.* and Groups.* functions via
standard closure types (ref_delta, implied_delta, chain_dx/chain_dy/chain_dz, chain_dt). Metrics are consistency
indicators, not certification.
"""

from __future__ import annotations

import datetime
import math
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Union

import jax
import jax.numpy as jnp

from moju.piratio.groups import Groups
from moju.piratio.laws import Laws
from moju.monitor.closure_registry import (
    GROUP_FNS,
    MODEL_FNS,
    compute_chain,
    compute_chain_weak,
    compute_implied_delta,
    compute_ref_delta,
)
from moju.monitor.derived_state_chain import all_ref_keys_from_chain, keys_produced_by_chain
from moju.monitor.derivative_keys import CHAIN_SPATIAL_DERIVS, collect_audit_derivative_keys
from moju.monitor.pi_constant_recipes import (
    GROUP_PI_CONSTANT_RECIPES,
    apply_pi_constant_recipe,
)
from moju.monitor.law_implied_diagnostics import (
    merge_fragment_law_implied_audit_specs,
    merge_law_implied_audit_specs,
)
from moju.monitor.visualize_labels import (
    category_adm_bar_x_range,
    format_admissibility_pct,
    pretty_category_name,
    pretty_residual_key,
    truncate_display_label,
)

DEFAULT_VISUALIZE_TITLE_TRAINING = "Physics admissibility audit (model training)"
DEFAULT_VISUALIZE_TITLE_TEST = "State prediction audit (physics residuals)"


def _resolve_visualize_figure_title(mode: str, figure_title: Optional[str]) -> str:
    """Figure title for :func:`visualize`; non-empty ``figure_title`` overrides mode default."""
    if figure_title is not None and str(figure_title).strip():
        return str(figure_title).strip()
    if mode == "training":
        return DEFAULT_VISUALIZE_TITLE_TRAINING
    return DEFAULT_VISUALIZE_TITLE_TEST


def _state_key_suggests_law_fd_fill(key: str) -> bool:
    """True if missing key may be fillable via fill_law_fd (derived law input)."""
    if key.endswith("_laplacian") or key.endswith("_grad") or key.endswith("_tt"):
        return True
    if key.endswith("_t") and not key.endswith("_tt"):
        return True
    return key == "u_grad"


def _string_placeholder_like(v: Any, *, key: str, arg_name: str) -> bool:
    """Best-effort check for UI placeholders accidentally saved as strings."""
    if not isinstance(v, str):
        return False
    s = v.strip()
    if not s:
        return True
    low = s.lower()
    if low in {"none", "null", "nan", "na"}:
        return True
    return s == key or s == arg_name


def _kwargs_from_state(
    state: Dict[str, Any],
    constants: Dict[str, Any],
    state_map: Dict[str, str],
    *,
    law_context: Optional[str] = None,
) -> Dict[str, Any]:
    """Build kwargs for a law/group from state_map (arg_name -> state_key)."""
    out = {}
    for arg_name, key in state_map.items():
        val = state.get(key)
        if _string_placeholder_like(val, key=key, arg_name=str(arg_name)):
            val = None
        if val is None:
            val = constants.get(key)
            if _string_placeholder_like(val, key=key, arg_name=str(arg_name)):
                val = None
        if val is None:
            msg = f"Key {key!r} not found in state or constants (arg {arg_name})"
            if law_context is not None:
                msg += f" (law {law_context!r})"
                if _state_key_suggests_law_fd_fill(key):
                    msg += (
                        " — on Path B, enable auto_path_b_derivatives=True and fill_law_fd=True, "
                        "provide the primitive field (e.g. T for T_laplacian) and mesh coordinates "
                        "(x, and y/z/t as needed); see moju.monitor.law_fd_recipes.LAW_FD_RECIPES. "
                        "The engine does not rename NPZ keys: use key `T` (or change law state_map), "
                        "not only Studio alias hints."
                    )
            raise KeyError(msg)
        if isinstance(val, str):
            msg = (
                f"Key {key!r} resolved to string value {val!r} (arg {arg_name})"
                " but laws expect numeric array-like inputs."
            )
            if law_context is not None:
                msg += f" (law {law_context!r})"
            raise TypeError(msg)
        out[arg_name] = val
    return out


def _get_fn(spec: Dict[str, Any], builtin_class: Any) -> Any:
    if "fn" in spec:
        return spec["fn"]
    return getattr(builtin_class, spec["name"])


def _build_state(
    state_pred: Dict[str, Any],
    constants: Dict[str, Any],
    groups_spec: List[Dict],
) -> Dict[str, Any]:
    """Run group specs in order; write output_key into state."""
    state = dict(state_pred)
    merged = {**constants, **state}
    for spec in groups_spec:
        state_map = spec["state_map"]
        output_key = spec["output_key"]
        kwargs = _kwargs_from_state(merged, constants, state_map)
        fn = _get_fn(spec, Groups)
        state[output_key] = fn(**kwargs)
        merged[output_key] = state[output_key]
    return state


def _rms_scalar(x: jnp.ndarray) -> jnp.ndarray:
    a = jnp.asarray(x)
    if a.size == 0:
        return jnp.asarray(float("nan"))
    return jnp.sqrt(jnp.nanmean(jnp.square(a)))


def admissibility_level(score: float) -> str:
    if not math.isfinite(score):
        return "Unknown"
    if score >= 0.90:
        return "High Admissibility"
    if score >= 0.70:
        return "Moderate Admissibility"
    if score >= 0.40:
        return "Low Admissibility"
    return "Non-Admissible"


def _geom_mean_admissibility(values: Sequence[float]) -> float:
    """
    Geometric mean over finite scores only. If any finite value is <= 0, returns 0.0.
    If no finite values, returns NaN.
    """
    finite: List[float] = []
    for x in values:
        try:
            xf = float(x)
        except (TypeError, ValueError):
            continue
        if math.isfinite(xf):
            finite.append(xf)
    if not finite:
        return float("nan")
    if min(finite) <= 0.0:
        return 0.0
    return float(math.exp(sum(math.log(x) for x in finite) / len(finite)))


def _rms_per_key(
    residuals_flat: Dict[str, jnp.ndarray],
    *,
    to_python: bool = True,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, arr in residuals_flat.items():
        r = _rms_scalar(arr)
        if to_python:
            out[key] = float(jax.device_get(r))
        else:
            out[key] = r
    return out


def _flatten_residual_dict(residuals: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
    flat: Dict[str, jnp.ndarray] = {}
    for category, content in residuals.items():
        if not isinstance(content, dict):
            if hasattr(content, "shape"):
                flat[category] = jnp.asarray(content)
            else:
                flat[category] = jnp.asarray(content)
            continue
        for name, arr in content.items():
            flat[f"{category}/{name}"] = jnp.asarray(arr)
    return flat


_SCALE_EPS = 1e-12


def _state_derived_scale_per_key(
    flat_keys: Iterable[str],
    merged: Dict[str, Any],
    laws_spec: List[Dict[str, Any]],
    constitutive_audit: List[Dict[str, Any]],
    scaling_audit: List[Dict[str, Any]],
    state_ref_built: Optional[Dict[str, Any]] = None,
    *,
    to_python: bool = True,
) -> Dict[str, float]:
    """
    State-derived scale per residual key for R_norm = RMS(r_k) / scale_k.
    Used when r_ref is not supplied; provides a scale relative to solution size.
    """
    out: Dict[str, float] = {}
    ref = state_ref_built if state_ref_built is not None else merged

    for k in flat_keys:
        if "/" not in k:
            scale = _SCALE_EPS + _rms_scalar(jnp.asarray(1.0))
            out[k] = float(jax.device_get(scale)) if to_python else float(scale)
            continue
        prefix, rest = k.split("/", 1)
        if prefix == "laws":
            name = rest
            spec = next((s for s in laws_spec if s.get("name") == name), None)
            if spec is None:
                scale = _SCALE_EPS + _rms_scalar(jnp.asarray(1.0))
            else:
                state_map = spec.get("state_map") or {}
                parts = []
                for sk in state_map.values():
                    if sk in merged:
                        v = jnp.asarray(merged[sk])
                        parts.append(jnp.ravel(v))
                if parts:
                    big = jnp.concatenate(parts)
                    scale = _SCALE_EPS + _rms_scalar(big)
                else:
                    scale = _SCALE_EPS + _rms_scalar(jnp.asarray(1.0))
            out[k] = float(jax.device_get(scale)) if to_python else float(scale)
            continue
        if prefix == "constitutive":
            name = rest.split("/")[0]
            spec = next(
                (
                    s
                    for s in constitutive_audit
                    if s.get("name") == name
                    or (s.get("residual_basename") or "").split("/")[0] == name
                ),
                None,
            )
            if spec is None:
                scale = _SCALE_EPS + _rms_scalar(jnp.asarray(1.0))
            else:
                state_map = spec.get("state_map") or {}
                output_key = spec.get("output_key")
                parts = []
                for sk in state_map.values():
                    if sk in merged:
                        parts.append(jnp.ravel(jnp.asarray(merged[sk])))
                if output_key and output_key in merged:
                    parts.append(jnp.ravel(jnp.asarray(merged[output_key])))
                ivk = spec.get("implied_value_key")
                if ivk and ivk in merged:
                    parts.append(jnp.ravel(jnp.asarray(merged[ivk])))
                if parts:
                    big = jnp.concatenate(parts)
                    scale = _SCALE_EPS + _rms_scalar(big)
                else:
                    scale = _SCALE_EPS + _rms_scalar(jnp.asarray(1.0))
            out[k] = float(jax.device_get(scale)) if to_python else float(scale)
            continue
        if prefix == "scaling":
            name = rest.split("/")[0]
            spec = next(
                (
                    s
                    for s in scaling_audit
                    if s.get("name") == name
                    or (s.get("residual_basename") or "").split("/")[0] == name
                ),
                None,
            )
            if spec is None:
                scale = _SCALE_EPS + _rms_scalar(jnp.asarray(1.0))
            else:
                state_map = spec.get("state_map") or {}
                output_key = spec.get("output_key")
                parts = []
                for sk in state_map.values():
                    if sk in merged:
                        parts.append(jnp.ravel(jnp.asarray(merged[sk])))
                if output_key and output_key in merged:
                    parts.append(jnp.ravel(jnp.asarray(merged[output_key])))
                ivk_s = spec.get("implied_value_key")
                if ivk_s and ivk_s in merged:
                    parts.append(jnp.ravel(jnp.asarray(merged[ivk_s])))
                if parts:
                    big = jnp.concatenate(parts)
                    scale = _SCALE_EPS + _rms_scalar(big)
                else:
                    scale = _SCALE_EPS + _rms_scalar(jnp.asarray(1.0))
            out[k] = float(jax.device_get(scale)) if to_python else float(scale)
            continue
        if prefix == "data":
            state_key = rest
            if state_key in ref:
                v = jnp.asarray(ref[state_key])
                scale = _SCALE_EPS + _rms_scalar(jnp.ravel(v))
            else:
                scale = _SCALE_EPS + _rms_scalar(jnp.asarray(1.0))
            out[k] = float(jax.device_get(scale)) if to_python else float(scale)
            continue
        scale = _SCALE_EPS + _rms_scalar(jnp.asarray(1.0))
        out[k] = float(jax.device_get(scale)) if to_python else float(scale)
    return out


def build_loss(
    residual_dict: Dict[str, Any],
    option: str = "cascaded",
    law_weights: Optional[Dict[str, float]] = None,
) -> jnp.ndarray:
    if option != "cascaded":
        raise ValueError(f"Only option='cascaded' is implemented, got {option!r}")
    laws = residual_dict.get("laws", {})
    if not laws:
        return jnp.array(0.0)
    names = list(laws.keys())
    n = len(names)
    weights = law_weights or {}
    w = jnp.array([weights.get(name, 1.0 / n) for name in names])
    rms_vals = jnp.array([_rms_scalar(jnp.asarray(laws[name])) for name in names])
    return jnp.sum(w * rms_vals)


def _compute_log_step_metrics(
    log: List[Dict[str, Any]],
    r_ref: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Per-log-entry admissibility metrics (same rules as ``audit``), without mutating ``log``.

    Returns one dict per entry with keys: ``r_norm``, ``admissibility_score``,
    ``category_admissibility_score``, ``overall_admissibility_score``, and
    ``per_key_report`` (flat key -> {rms, r_norm, admissibility_score, admissibility_level}).
    """
    if not log:
        return []
    first_rms = log[0].get("rms", {})
    out: List[Dict[str, Any]] = []
    category_buckets = ("laws", "constitutive", "scaling", "data")
    for entry in log:
        rms = entry.get("rms", {})
        entry_scale = entry.get("scale") or {}
        r_norm: Dict[str, float] = {}
        admissibility: Dict[str, float] = {}
        per_key_report: Dict[str, Any] = {}
        for k, v in rms.items():
            if r_ref is not None and k in r_ref and r_ref[k] is not None and r_ref[k] > 0:
                scale_k = r_ref[k]
            elif k in entry_scale and entry_scale[k] is not None and entry_scale[k] > 0:
                scale_k = entry_scale[k]
            elif k in first_rms and first_rms[k] is not None and first_rms[k] > 0:
                scale_k = first_rms[k]
            else:
                scale_k = 1.0
            if scale_k <= 0 or not math.isfinite(float(scale_k)):
                scale_k = 1.0
            try:
                v_f = float(v)
            except (TypeError, ValueError):
                v_f = float("nan")
            if not math.isfinite(v_f):
                r_norm[k] = float("nan")
                admissibility[k] = float("nan")
            else:
                r_norm[k] = v_f / scale_k
                if not math.isfinite(r_norm[k]):
                    admissibility[k] = float("nan")
                else:
                    admissibility[k] = 1.0 / (1.0 + r_norm[k])
            per_key_report[k] = {
                "rms": v,
                "r_norm": r_norm[k],
                "admissibility_score": admissibility[k],
                "admissibility_level": admissibility_level(admissibility[k]),
            }
        category_keys: Dict[str, List[str]] = {c: [] for c in category_buckets}
        for k in admissibility.keys():
            if "/" in k:
                prefix = k.split("/", 1)[0]
                if prefix in category_keys:
                    category_keys[prefix].append(k)
        category_scores: Dict[str, float] = {}
        for cat, keys in category_keys.items():
            if not keys:
                continue
            vals = [float(admissibility[kk]) for kk in keys]
            cat_gm = _geom_mean_admissibility(vals)
            if math.isfinite(cat_gm):
                category_scores[cat] = cat_gm
        cats_present = list(category_scores.keys())
        if not cats_present:
            overall = float("nan")
        else:
            overall = _geom_mean_admissibility([float(category_scores[c]) for c in cats_present])
        out.append(
            {
                "r_norm": r_norm,
                "admissibility_score": admissibility,
                "category_admissibility_score": category_scores,
                "overall_admissibility_score": overall,
                "per_key_report": per_key_report,
            }
        )
    return out


def audit(
    log: List[Dict[str, Any]],
    r_ref: Optional[Dict[str, float]] = None,
    weights: Optional[Dict[str, float]] = None,
    *,
    export_dir: Optional[str] = None,
    save_residuals: bool = False,
    last_residual_dict: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Physics admissibility from logged RMS and scales.

    Reporting uses three levels (no extra aggregation API): (1) per residual key in
    ``per_key``; (2) geometric mean within each category — ``per_category`` keys
    ``laws``, ``constitutive``, ``scaling``, ``data`` — using **finite** per-key
    admissibility only (non-finite keys are skipped; empty categories omitted); (3) overall
    score — geometric mean of **finite** category scores only.
    """
    if not log:
        return {"per_key": {}, "overall_admissibility_score": 0.0, "overall_admissibility_level": "Non-Admissible"}
    step_metrics = _compute_log_step_metrics(log, r_ref)
    last_report_per_key: Dict[str, Any] = {}
    for entry, m in zip(log, step_metrics):
        entry["r_norm"] = m["r_norm"]
        entry["admissibility_score"] = m["admissibility_score"]
        entry["category_admissibility_score"] = m["category_admissibility_score"]
        entry["overall_admissibility_score"] = m["overall_admissibility_score"]
        last_report_per_key = dict(m["per_key_report"])
    overall = log[-1].get("overall_admissibility_score", 0.0) if log else 0.0
    report = {
        "per_key": last_report_per_key,
        "per_category": log[-1].get("category_admissibility_score", {}) if log else {},
        "overall_admissibility_score": overall,
        "overall_admissibility_level": admissibility_level(overall),
    }

    if export_dir:
        import zipfile
        from pathlib import Path
        session_name = datetime.datetime.now().strftime("audit_%Y%m%d_%H%M")
        session_dir = Path(export_dir) / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        try:
            from moju.monitor.report import write_audit_pdf, write_residuals_json
            pdf_path = session_dir / "report.pdf"
            write_audit_pdf(report, str(pdf_path), model_name=model_name, model_id=model_id)
            if save_residuals and last_residual_dict is not None:
                json_path = session_dir / "residuals.json"
                write_residuals_json(last_residual_dict, str(json_path))
            zip_path = Path(export_dir) / f"{session_name}.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in session_dir.iterdir():
                    zf.write(f, f"{session_name}/{f.name}")
        except ImportError as e:
            raise ImportError(
                "PDF export requires reportlab. Install with: pip install moju[report] or pip install reportlab"
            ) from e

    return report


def _keys_by_category(plot_keys: Sequence[str]) -> Dict[str, List[str]]:
    order = ("laws", "constitutive", "scaling", "data")
    buckets: Dict[str, List[str]] = {c: [] for c in order}
    for k in plot_keys:
        if "/" not in k:
            continue
        p = k.split("/", 1)[0]
        if p in buckets:
            buckets[p].append(k)
    return buckets


def _parse_spatial_law_panel(spatial_law_panel: Optional[Any]) -> Optional[Dict[str, Any]]:
    """
    Validate optional spatial panel: ``{"x": 1D, "values": {law_key: 1D same length}}``.

    Law keys may be bare names or ``laws/<name>``. Returns ``x``, ``Z`` (n_laws × n_x),
    and ``row_labels`` (pretty strings) or ``None`` if invalid.
    """
    if spatial_law_panel is None:
        return None
    if not isinstance(spatial_law_panel, Mapping):
        return None
    x = spatial_law_panel.get("x")
    values = spatial_law_panel.get("values")
    if x is None or values is None or not isinstance(values, Mapping):
        return None
    import numpy as np

    x_arr = np.asarray(x, dtype=float).ravel()
    rows: List[Any] = []
    labels: List[str] = []
    for k in sorted(values.keys()):
        v = values[k]
        arr = np.asarray(v, dtype=float).ravel()
        if arr.shape != x_arr.shape:
            return None
        rows.append(arr)
        lk = str(k)
        flat = lk if lk.startswith("laws/") else f"laws/{lk}"
        labels.append(pretty_residual_key(flat))
    if not rows:
        return None
    pos_ax = spatial_law_panel.get("position_axis")
    out: Dict[str, Any] = {"x": x_arr, "Z": np.stack(rows, axis=0), "row_labels": labels}
    if pos_ax is not None and str(pos_ax).strip():
        out["position_axis"] = str(pos_ax).strip()
    return out


def _parse_spatial_rnorm_panel(spatial_rnorm_panel: Optional[Any]) -> Optional[Dict[str, Any]]:
    """
    Validate optional R_norm spatial panel: ``{"x": 1D, "values": {flat_key: 1D}}``.
    """
    if spatial_rnorm_panel is None:
        return None
    if not isinstance(spatial_rnorm_panel, Mapping):
        return None
    x = spatial_rnorm_panel.get("x")
    values = spatial_rnorm_panel.get("values")
    if x is None or values is None or not isinstance(values, Mapping):
        return None
    import numpy as np

    x_arr = np.asarray(x, dtype=float).ravel()
    rows: List[Any] = []
    labels: List[str] = []
    for k in sorted(values.keys()):
        arr = np.asarray(values[k], dtype=float).ravel()
        if arr.shape != x_arr.shape:
            return None
        rows.append(arr)
        labels.append(pretty_residual_key(str(k)))
    if not rows:
        return None
    pos_ax = spatial_rnorm_panel.get("position_axis")
    out: Dict[str, Any] = {"x": x_arr, "Z": np.stack(rows, axis=0), "row_labels": labels}
    if pos_ax is not None and str(pos_ax).strip():
        out["position_axis"] = str(pos_ax).strip()
    return out


def _build_visualize_bundle(
    log: List[Dict[str, Any]],
    keys: Optional[List[str]],
    r_ref: Optional[Dict[str, float]],
    max_legend_keys: int,
    *,
    spatial_parsed: Optional[Dict[str, Any]],
    spatial_rnorm_parsed: Optional[Dict[str, Any]] = None,
    mode: str,
) -> Optional[Dict[str, Any]]:
    """
    Shared arrays and metadata for :func:`visualize` (matplotlib or plotly).

    Does not mutate ``log``. Requires ``numpy`` (already a moju dependency).
    """
    import numpy as np

    if not log:
        return None
    first_rms = log[0].get("rms", {})
    plot_keys = list(keys) if keys is not None else list(first_rms.keys())
    if not plot_keys:
        return None
    metrics = _compute_log_step_metrics(log, r_ref)
    n = len(log)
    indices = np.arange(n, dtype=float)
    cap = max(1, int(max_legend_keys))
    use_bar_chart = mode == "test" or (mode == "training" and n == 1)
    bar_keys = plot_keys[: min(48, len(plot_keys))]

    buckets = _keys_by_category(plot_keys)
    category_training: Dict[str, Dict[str, Any]] = {}
    cat_order = ("laws", "constitutive") if mode == "training" else ("laws", "constitutive", "scaling")
    for cat in cat_order:
        cat_keys = sorted(buckets.get(cat, []))[:cap]
        nk = len(cat_keys)
        mat = np.zeros((nk, n)) if nk else np.zeros((0, n))
        for j in range(n):
            for i, kk in enumerate(cat_keys):
                v = metrics[j]["r_norm"].get(kk)
                mat[i, j] = float(v) if v is not None else float("nan")
        category_training[cat] = {
            "keys": cat_keys,
            "displays": [pretty_residual_key(k) for k in cat_keys],
            "r_norm_mat": mat,
        }

    legend_keys = plot_keys[: min(cap, len(plot_keys))]
    r_norm_mat = np.zeros((len(legend_keys), n))
    for j in range(n):
        for i, kk in enumerate(legend_keys):
            v = metrics[j]["r_norm"].get(kk)
            r_norm_mat[i, j] = float(v) if v is not None else float("nan")

    legend_display = [pretty_residual_key(k) for k in legend_keys]
    bar_display = [pretty_residual_key(k) for k in bar_keys]
    bar_values = []
    last_rn = metrics[-1]["r_norm"]
    for kk in bar_keys:
        v = last_rn.get(kk)
        bar_values.append(float(v) if v is not None else float("nan"))
    bar_values_arr = np.asarray(bar_values, dtype=float)

    cat_colors = {
        "laws": "#4e79a7",
        "constitutive": "#f28e2b",
        "scaling": "#59a14f",
        "data": "#b07aa1",
    }
    last_cat = metrics[-1]["category_admissibility_score"]
    cats_fin = [
        (pretty_category_name(c), float(last_cat[c]))
        for c in cat_colors
        if c in last_cat and np.isfinite(last_cat[c])
    ]
    nr_title = "Normalized Residuals"
    if not use_bar_chart and len(plot_keys) > len(legend_keys):
        nr_title = f"Normalized Residuals (showing {len(legend_keys)} of {len(plot_keys)} keys)"

    category_titles = {
        "laws": "Normalized Governing Laws Residuals",
        "constitutive": "Normalized Constitutive Residuals",
        "scaling": "Normalized Scaling Residuals",
    }

    return {
        "log": log,
        "metrics": metrics,
        "n": n,
        "indices": indices,
        "plot_keys": plot_keys,
        "legend_keys": legend_keys,
        "legend_display": legend_display,
        "r_norm_mat": r_norm_mat,
        "category_training": category_training,
        "category_titles": category_titles,
        "use_bar_chart": use_bar_chart,
        "bar_keys": bar_keys,
        "bar_display": bar_display,
        "bar_values": bar_values_arr,
        "overall_adm": [float(metrics[i]["overall_admissibility_score"]) for i in range(n)],
        "cats_fin": cats_fin,
        "cat_colors": cat_colors,
        "spatial": spatial_parsed,
        "spatial_rnorm": spatial_rnorm_parsed,
        "mode": mode,
        "nr_title": nr_title,
        "np": np,
    }


def _apply_visualize_style() -> Dict[str, Any]:
    """Publication-oriented matplotlib rcParams (restored by caller context)."""
    return {
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "axes.titlecolor": "#1a1a1a",
        "text.color": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "grid.color": "#cccccc",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "lines.linewidth": 1.6,
    }


R_NORM_LOG_EPS = 1e-12


def _apply_visualize_style_actionable() -> Dict[str, Any]:
    """Stronger typography and spines for :func:`visualize` dashboards."""
    base = _apply_visualize_style()
    extra: Dict[str, Any] = {
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "font.size": 10.5,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
    }
    return {**base, **extra}


def _admissibility_status_hml(score: float) -> str:
    """HIGH / MODERATE / LOW band for headline (same bands as former diagnostics plot)."""
    if not math.isfinite(score):
        return "N/A"
    if score >= 0.9:
        return "HIGH"
    if score >= 0.7:
        return "MODERATE"
    return "LOW"


def _category_adm_bar_color(score: float) -> str:
    if not math.isfinite(score):
        return "#bdc3c7"
    if score >= 0.9:
        return "#27ae60"
    if score >= 0.7:
        return "#e67e22"
    return "#c0392b"


def _transform_r_norm_y(ys: Any, np: Any, log_scale: bool) -> Any:
    """R_norm line-plot values: optional log10(R_norm + eps)."""
    arr = np.asarray(ys, dtype=float)
    if not log_scale:
        return arr
    return np.log10(np.maximum(arr, 0.0) + R_NORM_LOG_EPS)


def _category_three_pillar_scores(metrics: List[Dict[str, Any]]) -> tuple[List[str], List[float]]:
    """Labels and scores for laws / constitutive pillars (final step)."""
    last_cat = metrics[-1]["category_admissibility_score"]
    order = ("laws", "constitutive")
    labels = [pretty_category_name(c) for c in order]
    vals = [float(last_cat[c]) if c in last_cat and math.isfinite(float(last_cat[c])) else float("nan") for c in order]
    return list(labels), vals


def _matplotlib_draw_category_adm_three_pillar(ax: Any, metrics: List[Dict[str, Any]], np: Any) -> None:
    """Horizontal bar chart for laws/constitutive pillars (matplotlib)."""
    from matplotlib.ticker import PercentFormatter

    clabels, cvals = _category_three_pillar_scores(metrics)
    y_pos = np.arange(len(clabels))
    colors_b = [_category_adm_bar_color(v) for v in cvals]
    bar_lengths = np.array([v if math.isfinite(v) else 0.0 for v in cvals], dtype=float)
    ax.barh(
        y_pos,
        bar_lengths,
        color=colors_b,
        edgecolor="#333333",
        linewidth=0.9,
        height=0.55,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(clabels, fontsize=10)
    x0, x1 = category_adm_bar_x_range(list(cvals))
    ax.set_xlim(x0, x1)
    ax.set_xlabel("Admissibility", fontsize=10)
    ax.set_title("Category admissibility (final step)", fontsize=11, fontweight="600")
    for i, v in enumerate(cvals):
        if math.isfinite(v):
            tx = min(v + 0.02 * max(x1 - x0, 1e-6), x1 - 1e-6)
            ax.text(tx, i, format_admissibility_pct(v), va="center", fontsize=9, fontweight="600")
        else:
            ax.text(x0 + 0.02 * (x1 - x0), i, "N/A", va="center", fontsize=9, color="#666666")
    ax.grid(True, axis="x", alpha=0.35)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    for spine in ax.spines.values():
        spine.set_linewidth(1.05)


def build_monitor_visualize_bundle(
    log: List[Dict[str, Any]],
    keys: Optional[List[str]] = None,
    r_ref: Optional[Dict[str, float]] = None,
    max_legend_keys: int = 16,
    *,
    spatial_law_panel: Optional[Dict[str, Any]] = None,
    spatial_rnorm_panel: Optional[Dict[str, Any]] = None,
    mode: str = "training",
) -> Optional[Dict[str, Any]]:
    """
    Build the internal visualization bundle (same as :func:`visualize` uses for Plotly).

    Intended for Studio and other callers that want several small Plotly figures instead of
    one combined subplot grid.
    """
    work_log = list(log)
    if mode == "test" and len(work_log) > 1:
        work_log = work_log[-1:]
    spatial_parsed = _parse_spatial_law_panel(spatial_law_panel)
    spatial_rnorm_parsed = _parse_spatial_rnorm_panel(spatial_rnorm_panel)
    return _build_visualize_bundle(
        work_log,
        keys,
        r_ref,
        max_legend_keys,
        spatial_parsed=spatial_parsed,
        spatial_rnorm_parsed=spatial_rnorm_parsed,
        mode=mode,
    )


def visualize(
    log: List[Dict[str, Any]],
    keys: Optional[List[str]] = None,
    backend: str = "matplotlib",
    *,
    r_ref: Optional[Dict[str, float]] = None,
    max_legend_keys: int = 16,
    mode: str = "training",
    spatial_law_panel: Optional[Dict[str, Any]] = None,
    spatial_rnorm_panel: Optional[Dict[str, Any]] = None,
    figure_title: Optional[str] = None,
    step_label: str = "Step",
    r_norm_scale: str = "log",
    spatial_heatmap_colorscale: Optional[str] = None,
) -> Any:
    """
    Monitor dashboard from ``ResidualEngine`` log entries (``rms``, ``scale``).

    Uses the same R_norm / admissibility rules as :func:`audit` via
    :func:`_compute_log_step_metrics` and **does not mutate** ``log``.

    **Modes**

    - ``training`` (multi-step) — **Top row:** overall admissibility vs step (with final
      value marker) and **horizontal category admissibility bars** (laws / constitutive,
      final step). **Second row:** two panels — :math:`R_{\\mathrm{norm}}` vs
      step for **governing laws** and **constitutive** (``data/`` and ``scaling/`` omitted);
      **y-axis** is ``log10(R_{\\mathrm{norm}} + \\varepsilon)`` by default, or linear if
      ``r_norm_scale="linear"``. **Third row:** ``R_{\\mathrm{norm}}`` heatmaps vs step
      for laws and constitutive (Jet colormap). **Optional fourth row:** spatial heatmaps
      when ``spatial_law_panel`` or ``spatial_rnorm_panel`` is set.
    - ``training`` (single log entry) — horizontal bars for normalized residuals, category
      admissibility bars, optional spatial panel (compact layout).
    - ``test`` — Uses the **last** log entry only: horizontal bar chart of normalized
      residuals per key, category admissibility bars, and optional spatial panel.

    **Spatial panel**

    The log stores **scalar** RMS per key per step only. To plot law residuals vs
    ``x`` (e.g. at a fixed time slice), pass ``spatial_law_panel``::

        {"x": jax.numpy.linspace(...), "values": {"laplace_equation": r_norm_along_x, ...}}

    Each value array must match ``x`` in shape. Keys may be bare law names or
    ``laws/<name>``.
    For constitutive-only spatial slices, pass ``spatial_rnorm_panel`` with the same
    shape contract (Studio uses this for ``constitutive/...`` rows alongside laws).
    Optional ``position_axis`` (e.g. ``\"y\"``) on each panel dict sets the horizontal
    axis label; default is ``x``.

    **Backends**

    - ``matplotlib`` — static figure (requires ``matplotlib``).
    - ``plotly`` — interactive figure (requires ``pip install plotly`` or ``moju[viz]``).
    - ``none`` — returns ``None``.

    Parameters
    ----------
    log
        Entries from ``ResidualEngine.log`` (after ``compute_residuals``).
    keys
        Subset of flat residual keys to plot; default = all keys in the first entry.
    backend
        ``matplotlib``, ``plotly``, or ``none``.
    r_ref
        Optional per-key reference scale overrides (same as :func:`audit`).
    max_legend_keys
        Cap legend entries on per-key line plots for readability (training mode, multi-step).
    mode
        ``training`` or ``test``. Test mode slices to the last entry when ``len(log) > 1``.
    spatial_law_panel
        Optional ``dict`` with ``x`` and ``values`` (see above).
    spatial_rnorm_panel
        Optional ``dict`` with ``x`` and ``values`` for per-key R_norm spatial slices.
    figure_title
        Optional override for the figure title. If omitted or blank, a mode-specific
        default is used (training vs test).
    step_label
        X-axis label for training step axis (e.g. ``Iteration`` or ``Epoch``).
    r_norm_scale
        ``log`` (default) plots ``log10(R_norm + ε)`` on the three category residual
        panels; ``linear`` plots raw ``R_norm``. Does not affect the overall admissibility
        axis.
    spatial_heatmap_colorscale
        Plotly colorscale name for optional spatial heatmaps (e.g. ``\"Jet\"``, ``\"Viridis\"``).
        Default ``None`` uses ``Jet`` in the Plotly backend.
    """
    if backend == "none":
        return None
    if mode not in ("training", "test"):
        raise ValueError("mode must be 'training' or 'test'")
    if r_norm_scale not in ("log", "linear"):
        raise ValueError("r_norm_scale must be 'log' or 'linear'")

    work_log = list(log)
    if mode == "test" and len(work_log) > 1:
        work_log = work_log[-1:]

    spatial_parsed = _parse_spatial_law_panel(spatial_law_panel)
    spatial_rnorm_parsed = _parse_spatial_rnorm_panel(spatial_rnorm_panel)
    bundle = _build_visualize_bundle(
        work_log,
        keys,
        r_ref,
        max_legend_keys,
        spatial_parsed=spatial_parsed,
        spatial_rnorm_parsed=spatial_rnorm_parsed,
        mode=mode,
    )
    if bundle is None:
        return None

    resolved_title = _resolve_visualize_figure_title(mode, figure_title)

    if backend == "plotly":
        try:
            from moju.monitor.visualize_plotly import build_plotly_monitor_figure

            return build_plotly_monitor_figure(
                bundle,
                figure_title=resolved_title,
                step_label=step_label,
                r_norm_scale=r_norm_scale,
                spatial_heatmap_colorscale=spatial_heatmap_colorscale,
            )
        except ImportError:
            return None

    if backend != "matplotlib":
        return None

    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError:
        return None

    np = bundle["np"]
    n = bundle["n"]
    indices = bundle["indices"]
    use_bar_chart = bundle["use_bar_chart"]
    bar_display = bundle["bar_display"]
    bar_values = bundle["bar_values"]
    overall_adm = bundle["overall_adm"]
    cats_fin = bundle["cats_fin"]
    spatial = bundle["spatial"]
    spatial_rnorm = bundle.get("spatial_rnorm")
    mode_eff = bundle["mode"]

    has_spatial = spatial is not None
    has_spatial_rnorm = spatial_rnorm is not None
    category_training = bundle.get("category_training") or {}
    category_titles = bundle.get("category_titles") or {}
    metrics = bundle["metrics"]
    use_log_rnorm = r_norm_scale == "log"
    rnorm_ylabel = "log10(R_norm + ε)" if use_log_rnorm else "Normalized residual (R norm)"

    style = _apply_visualize_style_actionable()
    with plt.rc_context(rc=style):
        if mode_eff == "training" and not use_bar_chart:
            fig_h = 14.5 if (has_spatial or has_spatial_rnorm) else 12.5
            fig = plt.figure(figsize=(15.0, fig_h), constrained_layout=True)
            n_outer = 4 if (has_spatial or has_spatial_rnorm) else 3
            hratios = [1.0, 1.12, 1.06, 1.06] if (has_spatial or has_spatial_rnorm) else [1.0, 1.12, 1.06]
            outer = fig.add_gridspec(n_outer, 1, height_ratios=hratios)

            g_top = outer[0].subgridspec(1, 2, wspace=0.30)
            ax_overall = fig.add_subplot(g_top[0, 0])
            ax_cat = fig.add_subplot(g_top[0, 1])
            last_ov = float(overall_adm[-1]) if len(overall_adm) else float("nan")
            status_hml = _admissibility_status_hml(last_ov)
            if any(np.isfinite(overall_adm)):
                ax_overall.plot(
                    indices,
                    overall_adm,
                    color="#2c3e50",
                    linewidth=2.2,
                    label="Overall admissibility",
                )
                ax_overall.margins(y=0.12)
                ax_overall.relim()
                ax_overall.autoscale(axis="y")
                if math.isfinite(last_ov):
                    lix = float(indices[-1])
                    ax_overall.scatter(
                        [lix],
                        [last_ov],
                        s=95,
                        color="#c0392b",
                        zorder=5,
                        edgecolors="white",
                        linewidths=1.6,
                    )
                    ax_overall.annotate(
                        format_admissibility_pct(last_ov),
                        xy=(lix, last_ov),
                        xytext=(10, 4),
                        textcoords="offset points",
                        fontsize=10,
                        fontweight="600",
                        color="#2c3e50",
                    )
            ax_overall.set_xlabel(step_label)
            ax_overall.set_ylabel("Admissibility (%)")
            ax_overall.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax_overall.set_title("Overall admissibility", fontweight="600")
            ax_overall.grid(True, alpha=0.4)
            for spine in ax_overall.spines.values():
                spine.set_linewidth(1.05)
            if any(np.isfinite(overall_adm)):
                ax_overall.legend(
                    loc="upper left",
                    bbox_to_anchor=(0.02, 0.98),
                    framealpha=0.92,
                    fontsize=9,
                )

            _matplotlib_draw_category_adm_three_pillar(ax_cat, metrics, np)

            g_cat = outer[1].subgridspec(1, 2, wspace=0.52)
            ax_laws = fig.add_subplot(g_cat[0, 0])
            ax_const = fig.add_subplot(g_cat[0, 1], sharex=ax_laws, sharey=ax_laws)
            palette = plt.cm.tab10(np.linspace(0, 0.9, 10))
            cat_axes = (ax_laws, ax_const)
            cat_ids = ("laws", "constitutive")
            for ax_i, cat in enumerate(cat_ids):
                ax_c = cat_axes[ax_i]
                info = category_training.get(cat, {"keys": [], "displays": [], "r_norm_mat": np.zeros((0, n))})
                ckeys = info["keys"]
                displays = info["displays"]
                mat = info["r_norm_mat"]
                title_c = category_titles.get(
                    cat,
                    cat.replace("_", " ").title(),
                )
                if not ckeys:
                    ax_c.text(
                        0.5,
                        0.5,
                        "No keys in this category",
                        ha="center",
                        va="center",
                        transform=ax_c.transAxes,
                        fontsize=10,
                        color="#666666",
                    )
                    if ax_i == 0:
                        ax_c.set_xlabel(step_label)
                        ax_c.set_ylabel(rnorm_ylabel)
                    else:
                        ax_c.set_xlabel("")
                        plt.setp(ax_c.get_yticklabels(), visible=False)
                else:
                    for i, _kk in enumerate(ckeys):
                        ys = mat[i, :]
                        if np.all(np.isfinite(ys)):
                            y_plot = _transform_r_norm_y(ys, np, use_log_rnorm)
                            ax_c.plot(
                                indices,
                                y_plot,
                                label=displays[i],
                                color=palette[i % len(palette)],
                                alpha=0.9,
                                linewidth=1.8,
                            )
                    ax_c.legend(
                        loc="upper right",
                        framealpha=0.92,
                        fontsize=7,
                        labelspacing=0.28,
                        handlelength=1.05,
                        borderpad=0.35,
                    )
                ax_c.set_title(title_c, fontsize=11, fontweight="600")
                ax_c.grid(True, alpha=0.4)
                for spine in ax_c.spines.values():
                    spine.set_linewidth(1.05)
                if ax_i == 0:
                    ax_c.set_xlabel(step_label)
                    ax_c.set_ylabel(rnorm_ylabel)
                else:
                    ax_c.set_xlabel("")
                    plt.setp(ax_c.get_yticklabels(), visible=False)
                if ckeys and n > 15:
                    plt.setp(ax_c.get_xticklabels(), rotation=35, ha="right", fontsize=8.5)

            g_hm = outer[2].subgridspec(1, 2, wspace=0.50)
            ax_law_hm = fig.add_subplot(g_hm[0, 0])
            ax_const_hm = fig.add_subplot(g_hm[0, 1])
            for ax_hm, cat, title_hm in (
                (ax_law_hm, "laws", "Governing laws R_norm (vs step)"),
                (ax_const_hm, "constitutive", "Constitutive R_norm (vs step)"),
            ):
                info_h = category_training.get(cat, {"keys": [], "displays": [], "r_norm_mat": np.zeros((0, n))})
                ckeys_h = info_h["keys"]
                displays_h = info_h["displays"]
                mat_h = np.asarray(info_h["r_norm_mat"], dtype=float)
                ax_hm.set_title(title_hm, fontsize=11, fontweight="600")
                if not ckeys_h or mat_h.size == 0:
                    ax_hm.text(
                        0.5,
                        0.5,
                        "No keys in this category",
                        ha="center",
                        va="center",
                        transform=ax_hm.transAxes,
                        fontsize=10,
                        color="#666666",
                    )
                    ax_hm.set_axis_off()
                else:
                    zplot = _transform_r_norm_y(mat_h, np, use_log_rnorm)
                    imh = ax_hm.imshow(
                        zplot,
                        aspect="auto",
                        cmap="jet",
                        interpolation="nearest",
                        origin="upper",
                    )
                    ax_hm.set_xlabel(step_label, labelpad=6)
                    ax_hm.set_xticks(range(n))
                    ax_hm.set_yticks(range(len(displays_h)))
                    disp_hm = [truncate_display_label(d, 34) for d in displays_h]
                    ax_hm.set_yticklabels(disp_hm, fontsize=7.5)
                    ax_hm.tick_params(axis="y", pad=5)
                    fig.colorbar(imh, ax=ax_hm, fraction=0.034, pad=0.058, label=rnorm_ylabel)
                    if n > 12:
                        plt.setp(ax_hm.get_xticklabels(), rotation=40, ha="right", fontsize=8)

            ax_sp = None
            if has_spatial or has_spatial_rnorm:
                g_sp = outer[3].subgridspec(1, 2 if (has_spatial and has_spatial_rnorm) else 1, wspace=0.38)
                ax_idx = 0
                if has_spatial:
                    ax_sp = fig.add_subplot(g_sp[0, ax_idx])
                    ax_idx += 1
                    Z = spatial["Z"]
                    x_sp = spatial["x"]
                    row_labels = spatial["row_labels"]
                    pos_ax = spatial.get("position_axis") or "x"
                    im = ax_sp.imshow(
                        Z,
                        aspect="auto",
                        cmap="cividis",
                        extent=(float(x_sp[0]), float(x_sp[-1]), len(row_labels) - 0.5, -0.5),
                        interpolation="nearest",
                    )
                    ax_sp.set_yticks(range(len(row_labels)))
                    rl_t = [truncate_display_label(lb, 34) for lb in row_labels]
                    ax_sp.set_yticklabels(rl_t, fontsize=7.5)
                    ax_sp.tick_params(axis="y", pad=5)
                    ax_sp.set_xlabel(f"Position {pos_ax}", labelpad=7)
                    ax_sp.set_title("Law residuals (spatial slice)", fontsize=11, fontweight="600")
                    fig.colorbar(im, ax=ax_sp, fraction=0.032, pad=0.055, label="R norm")
                if has_spatial_rnorm:
                    ax_rn = fig.add_subplot(g_sp[0, ax_idx])
                    Zr = spatial_rnorm["Z"]
                    xr = spatial_rnorm["x"]
                    rl = spatial_rnorm["row_labels"]
                    pos_ax_r = spatial_rnorm.get("position_axis") or "x"
                    imr = ax_rn.imshow(
                        Zr,
                        aspect="auto",
                        cmap="magma",
                        extent=(float(xr[0]), float(xr[-1]), len(rl) - 0.5, -0.5),
                        interpolation="nearest",
                    )
                    ax_rn.set_yticks(range(len(rl)))
                    rl_rt = [truncate_display_label(lb, 34) for lb in rl]
                    ax_rn.set_yticklabels(rl_rt, fontsize=7.5)
                    ax_rn.tick_params(axis="y", pad=5)
                    ax_rn.set_xlabel(f"Position {pos_ax_r}", labelpad=7)
                    ax_rn.set_title("Implied constitutive R_norm (spatial slice)", fontsize=11, fontweight="600")
                    fig.colorbar(imr, ax=ax_rn, fraction=0.032, pad=0.055, label="R norm")

            left_for_align = [ax_laws, ax_law_hm]
            if ax_sp is not None:
                left_for_align.append(ax_sp)
            fig.align_ylabels(left_for_align)

            fig.suptitle(resolved_title, fontsize=17, fontweight="700", y=0.978)
            if math.isfinite(last_ov):
                fig.text(
                    0.5,
                    0.922,
                    f"Overall admissibility (final): {format_admissibility_pct(last_ov)} — {status_hml}",
                    ha="center",
                    fontsize=11,
                    fontweight="600",
                    color="#1a1a1a",
                    transform=fig.transFigure,
                )
            _le = fig.get_layout_engine()
            if _le is not None and hasattr(_le, "set"):
                _le.set(
                    h_pad=0.04,
                    w_pad=0.06,
                    hspace=0.09,
                    wspace=0.06,
                    rect=[0.07, 0.055, 0.97, 0.88],
                )
            return fig

        fig_h = 9.2 if has_spatial else 7.0
        nrows = 2
        fig = plt.figure(figsize=(10.5, fig_h), constrained_layout=True)
        height_ratios = [1.18, 1.08]
        gs = fig.add_gridspec(nrows, 2, height_ratios=height_ratios, wspace=0.34)

        ax0 = fig.add_subplot(gs[0, :])
        valid = np.isfinite(bar_values)
        if np.any(valid):
            y_pos = np.arange(len(bar_display))
            colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(bar_display)))
            ax0.barh(
                y_pos,
                np.where(valid, bar_values, 0.0),
                color=colors,
                edgecolor="white",
                linewidth=0.5,
            )
            ax0.set_yticks(y_pos)
            ax0.set_yticklabels([truncate_display_label(d, 44) for d in bar_display], fontsize=8)
            ax0.invert_yaxis()
        ax0.set_xlabel("Normalized residual (R norm)")
        ax0.set_title("Normalized Residuals")
        ax0.grid(True, axis="x", alpha=0.4)

        polar_row = 1
        polar_span_full = mode_eff == "test" and not has_spatial
        if polar_span_full:
            ax_cat2 = fig.add_subplot(gs[polar_row, :])
            ax_sp = None
        else:
            ax_cat2 = fig.add_subplot(gs[polar_row, 0])
            ax_sp = fig.add_subplot(gs[polar_row, 1])

        _matplotlib_draw_category_adm_three_pillar(ax_cat2, metrics, np)

        if ax_sp is not None:
            if spatial is not None:
                Z = spatial["Z"]
                x_sp = spatial["x"]
                row_labels = spatial["row_labels"]
                im = ax_sp.imshow(
                    Z,
                    aspect="auto",
                    cmap="cividis",
                    extent=(float(x_sp[0]), float(x_sp[-1]), len(row_labels) - 0.5, -0.5),
                    interpolation="nearest",
                )
                ax_sp.set_yticks(range(len(row_labels)))
                pos_ax_c = spatial.get("position_axis") or "x"
                ax_sp.set_yticklabels([truncate_display_label(lb, 34) for lb in row_labels], fontsize=7.5)
                ax_sp.tick_params(axis="y", pad=5)
                ax_sp.set_xlabel(f"Position {pos_ax_c}", labelpad=7)
                ax_sp.set_title("Law residuals (spatial slice)", fontsize=11, fontweight="600")
                fig.colorbar(im, ax=ax_sp, fraction=0.034, pad=0.055, label="R norm")
            else:
                ax_sp.text(
                    0.5,
                    0.5,
                    "Spatial panel: pass spatial_law_panel\nwith x and per-law values.",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#555555",
                    transform=ax_sp.transAxes,
                )
                ax_sp.set_axis_off()

        fig.suptitle(resolved_title, fontsize=17, fontweight="700", y=0.975)
        last_ov_c = float(overall_adm[-1]) if len(overall_adm) else float("nan")
        if math.isfinite(last_ov_c):
            fig.text(
                0.5,
                0.918,
                f"Overall admissibility (final): {format_admissibility_pct(last_ov_c)} — {_admissibility_status_hml(last_ov_c)}",
                ha="center",
                fontsize=10.5,
                fontweight="600",
                color="#1a1a1a",
                transform=fig.transFigure,
            )
        _le2 = fig.get_layout_engine()
        if _le2 is not None and hasattr(_le2, "set"):
            _le2.set(
                h_pad=0.04,
                w_pad=0.06,
                hspace=0.08,
                wspace=0.06,
                rect=[0.08, 0.08, 0.96, 0.86],
            )

    return fig


class ResidualEngine:
    """
    Governing laws (Laws.*), optional group specs to enrich state, and model/group closures.

    Entry points:
      - Path A (recommended): provide (model, params, collocation) and a state_builder
        so moju can build state_pred (and derivatives) internally.
      - Path B (advanced): provide state_pred directly.

    Closure policy:
      - chain_dx / chain_dt run only when predicted_spatial / predicted_temporal are non-empty.
      - ref_delta runs when state_ref is provided (independent of predicted_*), unless the spec sets
        include_ref_delta=False.
      - implied_delta runs when implied_value_key or implied_fn is set; omitted if implied is missing.
      - A spec with no chain, no ref_delta, and no implied_delta does nothing (optional omit log).
      - Law-linked implied rows (see ``moju.monitor.law_implied_diagnostics``) are prepended when
        ``law_implied_audits`` is true (``MonitorConfig`` default). Use optional ``residual_basename``
        for unique flat keys under each category.

    Audit spec shape (constitutive_audit / scaling_audit items):
      {
        "name": "sutherland_mu",               # Models.<name> or Groups.<name>
        "output_key": "mu",                    # state key for F output; expects d_mu_dx / d_mu_dt for chain
        "state_map": {"T": "T", "mu0": "mu0", "T0": "T0", "S": "S"},  # fn arg -> state key
        "predicted_spatial": ["T"],            # state keys varying in x
        "predicted_temporal": ["T"],           # state keys varying in t
      }

    Derivative convention in state_pred: d_<state_key>_dx, _dy, _dz, _dt as required by audits.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        constants: Optional[Dict[str, Any]] = None,
        laws: Optional[List[Dict[str, Any]]] = None,
        groups: Optional[List[Dict[str, Any]]] = None,
        *,
        constitutive_audit: Optional[List[Dict[str, Any]]] = None,
        scaling_audit: Optional[List[Dict[str, Any]]] = None,
        constitutive_custom: Optional[List[Dict[str, Any]]] = None,
        scaling_custom: Optional[List[Dict[str, Any]]] = None,
        derived_state_chain: Optional[List[Dict[str, Any]]] = None,
        state_builder: Optional[
            Callable[[Any, Any, Dict[str, Any], Dict[str, Any]], Dict[str, Any]]
        ] = None,
        enable_omit_messages: bool = True,
        primary_fields: Optional[List[str]] = None,
        law_implied_audits: bool = True,
    ):
        law_implied_enabled = bool(law_implied_audits)
        # MonitorConfig convenience
        if config is not None:
            from moju.monitor.config import MonitorConfig, audit_spec_to_engine_dict

            if isinstance(config, MonitorConfig):
                constants = config.constants
                laws = config.laws
                groups = config.groups
                constitutive_audit = [audit_spec_to_engine_dict(s) for s in config.constitutive_audit]
                scaling_audit = [audit_spec_to_engine_dict(s) for s in config.scaling_audit]
                constitutive_custom = config.constitutive_custom
                scaling_custom = config.scaling_custom
                derived_state_chain = list(config.derived_state_chain or [])
                primary_fields = list(config.primary_fields)
                law_implied_enabled = bool(config.law_implied_audits)
                if config.state_builder is not None and state_builder is None:
                    state_builder = config.state_builder
            else:
                raise TypeError("config must be a MonitorConfig")

        self.constants = dict(constants or {})
        self.laws_spec = list(laws or [])
        self.groups_spec = list(groups or [])
        self.constitutive_audit = list(constitutive_audit or [])
        self.scaling_audit = list(scaling_audit or [])
        li_c, li_s = merge_law_implied_audit_specs(self.laws_spec, enabled=law_implied_enabled)
        mc, rc = merge_fragment_law_implied_audit_specs(li_c, self.constitutive_audit)
        ms, rs = merge_fragment_law_implied_audit_specs(li_s, self.scaling_audit)
        self.constitutive_audit = mc + rc
        self.scaling_audit = ms + rs
        self.constitutive_custom = list(constitutive_custom or [])
        self.scaling_custom = list(scaling_custom or [])
        self.derived_state_chain = list(derived_state_chain or [])
        self.state_builder = state_builder
        self.enable_omit_messages = bool(enable_omit_messages)
        self.primary_fields = list(primary_fields or ["T", "u", "v", "w", "p", "rho"])

        # Config-time validation (low effort)
        def _validate_specs(specs: Sequence[Dict[str, Any]], registry: Dict[str, Any], category: str) -> None:
            for spec in specs:
                if "name" not in spec:
                    raise ValueError(f"{category} spec missing 'name'")
                name = spec["name"]
                reg = registry.get(name)
                if reg is None:
                    raise ValueError(f"{category} spec name {name!r} is not registered")
                if "output_key" not in spec:
                    raise ValueError(f"{category}:{name} missing 'output_key'")
                if "state_map" not in spec or not isinstance(spec["state_map"], dict):
                    raise ValueError(f"{category}:{name} missing 'state_map' dict")
                _, arg_names = reg
                missing_args = [an for an in arg_names if an not in spec["state_map"]]
                if missing_args:
                    raise ValueError(f"{category}:{name} state_map missing args: {missing_args}")
                sm_vals = set(spec["state_map"].values())
                out_k = spec.get("output_key")
                for k in spec.get("predicted_spatial", []) or []:
                    if k not in sm_vals and k != out_k:
                        raise ValueError(
                            f"{category}:{name} predicted_spatial key {k!r} must be a state_map value "
                            f"or match output_key {out_k!r}"
                        )
                for k in spec.get("predicted_temporal", []) or []:
                    if k not in sm_vals and k != out_k:
                        raise ValueError(
                            f"{category}:{name} predicted_temporal key {k!r} must be a state_map value "
                            f"or match output_key {out_k!r}"
                        )
                ivk = spec.get("implied_value_key")
                ifn = spec.get("implied_fn")
                if ivk and ifn is not None:
                    raise ValueError(
                        f"{category}:{name} use only one of implied_value_key and implied_fn, not both"
                    )
                csa = list(spec.get("chain_spatial_axes") or ["x"])
                allowed = set(CHAIN_SPATIAL_DERIVS)
                bad = [a for a in csa if a not in allowed]
                if bad:
                    raise ValueError(
                        f"{category}:{name} chain_spatial_axes must be subset of {sorted(allowed)}, "
                        f"invalid: {bad}"
                    )
                if not csa:
                    raise ValueError(f"{category}:{name} chain_spatial_axes must be non-empty")
                co = str(spec.get("chain_output") or "state_derivative")
                if co not in ("state_derivative", "fd_on_composition"):
                    raise ValueError(
                        f"{category}:{name} chain_output must be 'state_derivative' or "
                        f"'fd_on_composition', got {co!r}"
                    )

        _validate_specs(self.constitutive_audit, MODEL_FNS, "constitutive")
        _validate_specs(self.scaling_audit, GROUP_FNS, "scaling")
        self._validate_pi_constant_specs()

        self._log: List[Dict[str, Any]] = []
        self._index = 0

    @property
    def log(self) -> List[Dict[str, Any]]:
        return self._log

    def clear_log(self) -> None:
        """Remove all logged steps and reset the step counter.

        Safe to call between training runs; does not alter engine configuration.
        """
        self._log.clear()
        self._index = 0

    def _validate_pi_constant_specs(self) -> None:
        for spec in self.scaling_audit:
            if not spec.get("invariance_pi_constant"):
                continue
            name = spec["name"]
            if name not in GROUP_PI_CONSTANT_RECIPES:
                raise ValueError(
                    f"scaling:{name} invariance_pi_constant requires a built-in π-constant recipe; "
                    f"supported: {sorted(GROUP_PI_CONSTANT_RECIPES.keys())}"
                )
            recipe = GROUP_PI_CONSTANT_RECIPES[name]
            sm = spec.get("state_map") or {}
            for arg_name, _ in recipe:
                if arg_name not in sm:
                    raise ValueError(
                        f"scaling:{name} π-constant recipe needs state_map entry for arg {arg_name!r}"
                    )
            cmp_keys = list(spec.get("invariance_compare_keys") or [])
            if not cmp_keys:
                raise ValueError(
                    f"scaling:{name} invariance_pi_constant requires non-empty invariance_compare_keys"
                )
            c = float(spec.get("invariance_scale_c", 10.0))
            if c <= 1.0:
                raise ValueError(f"scaling:{name} invariance_scale_c must be > 1, got {c}")
            for arg_name, _ in recipe:
                sk = sm[arg_name]
                if sk not in self.constants:
                    raise ValueError(
                        f"scaling:{name} π-constant requires key {sk!r} in ResidualEngine.constants "
                        f"(arg {arg_name!r})"
                    )
            if self.state_builder is None:
                raise ValueError(
                    f"scaling:{name} invariance_pi_constant requires ResidualEngine(state_builder=...) (Path A only)"
                )

    def _state_builder(
        self,
        state_pred: Dict[str, Any],
        constants_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        c = self.constants if constants_override is None else constants_override
        return _build_state(state_pred, c, self.groups_spec)

    def compute_residuals(
        self,
        state_pred: Optional[Dict[str, Any]] = None,
        state_ref: Optional[Dict[str, Any]] = None,
        *,
        model: Any = None,
        params: Any = None,
        collocation: Optional[Dict[str, Any]] = None,
        log_to_python: bool = True,
        auto_path_b_derivatives: Any = False,
        fill_law_fd: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute residuals.

        Path A: pass (model, params, collocation) and configure engine.state_builder.
        Path B: pass state_pred directly.

        If ``auto_path_b_derivatives`` is True, uses default ``PathBGridConfig``; if a
        ``PathBGridConfig`` instance, uses that layout. Fills missing ``d_*_dx``/``_dy``/``_dz``/``_dt``
        keys required by audits via finite differences (after group state build, before laws).
        Warnings are appended to the log ``inferred`` list when enabled.

        If ``fill_law_fd`` is True, ``auto_path_b_derivatives`` must also be enabled; missing
        **registered** ``Laws.*`` inputs (e.g. ``phi_laplacian``, ``u_grad``) are filled from
        primitives on the same grid when possible (see ``law_fd_recipes``).
        """
        path_a = state_pred is None
        pi_specs = [s for s in self.scaling_audit if s.get("invariance_pi_constant")]
        if pi_specs and not path_a:
            raise ValueError(
                "π-constant scaling audit (invariance_pi_constant) requires Path A: "
                "call compute_residuals(..., model=..., params=..., collocation=...) "
                "without passing state_pred."
            )

        residuals: Dict[str, Any] = {"laws": {}}
        pb_warn: List[str] = []

        if state_pred is None:
            if self.state_builder is None:
                raise ValueError("Path A requires ResidualEngine(state_builder=...)")
            if model is None or params is None or collocation is None:
                raise ValueError("Path A requires model, params, and collocation")
            state_pred = self.state_builder(model, params, collocation, self.constants)

        omitted_msgs: List[str] = []
        inferred_msgs: List[str] = []

        def _maybe_log_omit(msg: str) -> None:
            if self.enable_omit_messages:
                omitted_msgs.append(msg)

        def _maybe_log_infer(msg: str) -> None:
            if self.enable_omit_messages:
                inferred_msgs.append(msg)

        state_for_groups = dict(state_pred)
        if self.derived_state_chain:
            from moju.monitor.derived_state_chain import apply_derived_state_chain

            state_for_groups, _dwarn = apply_derived_state_chain(
                state_for_groups,
                self.constants,
                self.derived_state_chain,
            )
            for w in _dwarn:
                _maybe_log_infer(f"derived_state: {w}")

        state_pred_built = self._state_builder(state_for_groups)
        merged = {**self.constants, **state_pred_built}

        def _state_ref_raw_after_derived(sr: Dict[str, Any]) -> Dict[str, Any]:
            s0 = dict(sr)
            if self.derived_state_chain:
                from moju.monitor.derived_state_chain import apply_derived_state_chain

                s0, wr = apply_derived_state_chain(s0, self.constants, self.derived_state_chain)
                for x in wr:
                    _maybe_log_infer(f"derived_state(ref): {x}")
            return s0

        def _merge_state_ref(sr: Dict[str, Any]) -> Dict[str, Any]:
            return {**self._state_builder(_state_ref_raw_after_derived(sr)), **self.constants}

        if fill_law_fd and not auto_path_b_derivatives:
            raise ValueError(
                "fill_law_fd=True requires auto_path_b_derivatives=True or a PathBGridConfig"
            )

        if auto_path_b_derivatives:
            from moju.monitor.path_b_derivatives import PathBGridConfig, fill_path_b_derivatives

            if auto_path_b_derivatives is True:
                grid = PathBGridConfig()
            elif isinstance(auto_path_b_derivatives, PathBGridConfig):
                grid = auto_path_b_derivatives
            else:
                raise TypeError(
                    "auto_path_b_derivatives must be False, True, or a PathBGridConfig instance"
                )
            state_pred_built, pb_warn = fill_path_b_derivatives(
                state_pred_built,
                constitutive_audit=self.constitutive_audit,
                scaling_audit=self.scaling_audit,
                laws_spec=self.laws_spec,
                constants=self.constants,
                grid=grid,
                copy=False,
                fill_law_recipes=bool(fill_law_fd),
            )
            merged = {**self.constants, **state_pred_built}
            for w in pb_warn:
                _maybe_log_infer(f"path_b_derivatives: {w}")

        for spec in self.laws_spec:
            name = spec["name"]
            state_map = spec["state_map"]
            try:
                kwargs = _kwargs_from_state(
                    merged, self.constants, state_map, law_context=str(name)
                )
            except KeyError as err:
                if pb_warn:
                    tail = "\n".join(f"  - {w}" for w in pb_warn[:32])
                    raise KeyError(
                        f"{err.args[0]}\n\nFinite-difference / law-FD fill reported:\n{tail}"
                    ) from err
                raise
            fn = _get_fn(spec, Laws)
            residuals["laws"][name] = fn(**kwargs)

        def _run_specs(
            specs: Sequence[Dict[str, Any]],
            *,
            registry: Dict[str, Any],
            category: str,
        ) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for spec in specs:
                name = spec["name"]
                output_key = spec.get("output_key")
                state_map = spec.get("state_map") or {}
                closure_mode = str(spec.get("closure_mode") or "pointwise")
                quadrature_weights = dict(spec.get("quadrature_weights") or {})
                chain_out = str(spec.get("chain_output") or "state_derivative")
                if closure_mode not in ("pointwise", "weak"):
                    raise ValueError(f"{category}:{name} closure_mode must be 'pointwise' or 'weak', got {closure_mode!r}")
                # Sensible defaults (medium effort): when not provided, infer from collocation and common keys
                if "predicted_spatial" in spec:
                    predicted_spatial = list(spec.get("predicted_spatial") or [])
                else:
                    predicted_spatial = []
                    if collocation is not None and "x" in collocation:
                        for k in self.primary_fields:
                            if k in state_map.values():
                                predicted_spatial = [k]
                                break
                    _maybe_log_infer(f"{category}:{name} inferred predicted_spatial={predicted_spatial}")

                if "predicted_temporal" in spec:
                    predicted_temporal = list(spec.get("predicted_temporal") or [])
                else:
                    predicted_temporal = []
                    if collocation is not None and "t" in collocation:
                        for k in self.primary_fields:
                            if k in state_map.values():
                                predicted_temporal = [k]
                                break
                    _maybe_log_infer(f"{category}:{name} inferred predicted_temporal={predicted_temporal}")

                has_implied = bool(spec.get("implied_value_key")) or spec.get("implied_fn") is not None
                if (
                    not predicted_spatial
                    and not predicted_temporal
                    and state_ref is None
                    and not has_implied
                ):
                    _maybe_log_omit(
                        f"{category}:{name} omitted: no chain, ref_delta, or implied_delta applicable"
                    )
                    continue

                reg = registry.get(name)
                if reg is None:
                    # unknown function name -> omit silently (config validation should catch)
                    continue
                fn, arg_names = reg
                base = spec.get("residual_basename") or name

                if (
                    state_ref is not None
                    and output_key is not None
                    and spec.get("include_ref_delta", True)
                ):
                    arr = compute_ref_delta(
                        fn=fn,
                        arg_names=arg_names,
                        output_key=output_key,
                        state_map=state_map,
                        state_pred=merged,
                        state_ref=_merge_state_ref(state_ref),
                        constants=self.constants,
                    )
                    if arr is not None:
                        out[f"{base}/ref_delta"] = jnp.asarray(arr)

                if has_implied:
                    arr = compute_implied_delta(
                        fn=fn,
                        arg_names=arg_names,
                        state_map=state_map,
                        state_pred=merged,
                        constants=self.constants,
                        implied_value_key=spec.get("implied_value_key"),
                        implied_fn=spec.get("implied_fn"),
                    )
                    if arr is not None:
                        out[f"{base}/implied_delta"] = jnp.asarray(arr)

                if predicted_spatial and output_key is not None:
                    spatial_axes = list(spec.get("chain_spatial_axes") or ["x"])
                    for spatial_axis in spatial_axes:
                        if spatial_axis not in CHAIN_SPATIAL_DERIVS:
                            continue
                        chain_key = f"chain_d{spatial_axis}"
                        if closure_mode == "weak":
                            arr = compute_chain_weak(
                                fn=fn,
                                arg_names=arg_names,
                                output_key=output_key,
                                state_map=state_map,
                                state_pred=merged,
                                constants=self.constants,
                                predicted_varying=predicted_spatial,
                                deriv=spatial_axis,
                                weight_key=quadrature_weights.get(spatial_axis),
                                chain_output=chain_out,
                            )
                            _maybe_log_infer(
                                f"{category}:{name} using weak {chain_key}"
                            )
                        else:
                            arr = compute_chain(
                                fn=fn,
                                arg_names=arg_names,
                                output_key=output_key,
                                state_map=state_map,
                                state_pred=merged,
                                constants=self.constants,
                                predicted_varying=predicted_spatial,
                                deriv=spatial_axis,
                                chain_output=chain_out,
                            )
                        if arr is not None:
                            out[f"{base}/{chain_key}"] = jnp.asarray(arr)

                if predicted_temporal and output_key is not None:
                    if closure_mode == "weak":
                        arr = compute_chain_weak(
                            fn=fn,
                            arg_names=arg_names,
                            output_key=output_key,
                            state_map=state_map,
                            state_pred=merged,
                            constants=self.constants,
                            predicted_varying=predicted_temporal,
                            deriv="t",
                            weight_key=quadrature_weights.get("t"),
                            chain_output=chain_out,
                        )
                        _maybe_log_infer(f"{category}:{name} using weak chain_dt")
                    else:
                        arr = compute_chain(
                            fn=fn,
                            arg_names=arg_names,
                            output_key=output_key,
                            state_map=state_map,
                            state_pred=merged,
                            constants=self.constants,
                            predicted_varying=predicted_temporal,
                            deriv="t",
                            chain_output=chain_out,
                        )
                    if arr is not None:
                        out[f"{base}/chain_dt"] = jnp.asarray(arr)

            return out

        if self.constitutive_audit or self.constitutive_custom:
            c = _run_specs(self.constitutive_audit, registry=MODEL_FNS, category="constitutive")
            if self.constitutive_custom:
                for spec in self.constitutive_custom:
                    cname = spec["name"]
                    arr = spec["fn"](merged, self.constants)
                    if arr is not None:
                        c[f"custom/{cname}"] = jnp.asarray(arr)
            if c:
                residuals["constitutive"] = c

        pi_constant_scales: Dict[str, float] = {}
        if self.scaling_audit or self.scaling_custom:
            s = _run_specs(self.scaling_audit, registry=GROUP_FNS, category="scaling")
            if self.scaling_custom:
                for spec in self.scaling_custom:
                    cname = spec["name"]
                    arr = spec["fn"](merged, self.constants)
                    if arr is not None:
                        s[f"custom/{cname}"] = jnp.asarray(arr)
            if path_a and self.state_builder is not None:
                for spec in self.scaling_audit:
                    if not spec.get("invariance_pi_constant"):
                        continue
                    name = spec["name"]
                    c = float(spec.get("invariance_scale_c", 10.0))
                    if c <= 1.0:
                        raise ValueError(f"scaling:{name} invariance_scale_c must be > 1, got {c}")
                    recipe = GROUP_PI_CONSTANT_RECIPES[name]
                    state_map = spec.get("state_map") or {}
                    constants_scaled = apply_pi_constant_recipe(
                        self.constants, recipe, state_map, c
                    )
                    state_pred_pi = self.state_builder(model, params, collocation, constants_scaled)
                    merged_scaled = {
                        **self._state_builder(state_pred_pi, constants_scaled),
                        **constants_scaled,
                    }
                    fn, arg_names = GROUP_FNS[name]
                    kb = _kwargs_from_state(merged, self.constants, state_map)
                    ks = _kwargs_from_state(merged_scaled, constants_scaled, state_map)
                    val_b = fn(**{an: kb[an] for an in arg_names})
                    val_s = fn(**{an: ks[an] for an in arg_names})
                    if not jnp.allclose(
                        jnp.asarray(val_b), jnp.asarray(val_s), rtol=1e-4, atol=1e-6
                    ):
                        raise ValueError(
                            f"scaling:{name} π-constant scaled inputs did not preserve group value "
                            f"(baseline {val_b!r} vs scaled {val_s!r})"
                        )
                    compare_keys = list(spec.get("invariance_compare_keys") or [])
                    parts_r: List[jnp.ndarray] = []
                    parts_scale: List[jnp.ndarray] = []
                    for ck in compare_keys:
                        if ck not in merged:
                            raise KeyError(
                                f"scaling:{name} invariance_compare_keys: {ck!r} missing from baseline merged state"
                            )
                        if ck not in merged_scaled:
                            raise KeyError(
                                f"scaling:{name} invariance_compare_keys: {ck!r} missing from scaled merged state"
                            )
                        vb = jnp.asarray(merged[ck])
                        vs = jnp.asarray(merged_scaled[ck])
                        parts_r.append(jnp.ravel(vs - vb))
                        parts_scale.append(jnp.ravel(jnp.abs(vs)))
                    r_pi = jnp.concatenate(parts_r) if parts_r else jnp.array([0.0])
                    stacked_abs = jnp.concatenate(parts_scale) if parts_scale else jnp.array([0.0])
                    flat_pi = f"{name}/pi_constant"
                    s[flat_pi] = r_pi
                    scale_key = f"scaling/{flat_pi}"
                    mean_abs = float(jax.device_get(jnp.nanmean(stacked_abs)))
                    pi_constant_scales[scale_key] = float(_SCALE_EPS + mean_abs)
            if s:
                residuals["scaling"] = s

        if state_ref is not None:
            state_ref_built = self._state_builder(_state_ref_raw_after_derived(state_ref))
            common = set(state_pred_built.keys()) & set(state_ref_built.keys())
            residuals["data"] = {
                k: jnp.asarray(state_ref_built[k]) - jnp.asarray(state_pred_built[k])
                for k in common
            }

        flat = _flatten_residual_dict(residuals)
        rms_per_key = _rms_per_key(flat, to_python=log_to_python)
        state_ref_built_for_scale = None
        if state_ref is not None:
            state_ref_built_for_scale = self._state_builder(_state_ref_raw_after_derived(state_ref))
        scale_per_key = _state_derived_scale_per_key(
            flat.keys(),
            merged,
            self.laws_spec,
            self.constitutive_audit,
            self.scaling_audit,
            state_ref_built_for_scale,
            to_python=log_to_python,
        )
        scale_per_key.update(pi_constant_scales)
        entry: Dict[str, Any] = {"index": self._index, "rms": rms_per_key, "scale": scale_per_key}
        if omitted_msgs:
            entry["omitted"] = omitted_msgs
        if inferred_msgs:
            entry["inferred"] = inferred_msgs
        self._log.append(entry)
        self._index += 1
        return residuals

    def required_state_keys(
        self,
        *,
        include_groups: bool = True,
        include_laws: bool = True,
        include_audits: bool = True,
    ) -> Set[str]:
        keys: Set[str] = set()
        if include_laws:
            for spec in self.laws_spec:
                keys |= set((spec.get("state_map") or {}).values())
        if include_groups:
            for spec in self.groups_spec:
                keys |= set((spec.get("state_map") or {}).values())
                ok = spec.get("output_key")
                if ok:
                    keys.add(ok)
        if include_audits:
            for spec in self.constitutive_audit + self.scaling_audit:
                keys |= set((spec.get("state_map") or {}).values())
                ok = spec.get("output_key")
                if ok:
                    keys.add(ok)
                ivk = spec.get("implied_value_key")
                if ivk:
                    keys.add(ivk)
        keys |= all_ref_keys_from_chain(self.derived_state_chain)
        keys -= keys_produced_by_chain(self.derived_state_chain)
        return keys

    def required_derivative_keys(self) -> Set[str]:
        sx, st = collect_audit_derivative_keys(
            list(self.constitutive_audit), list(self.scaling_audit)
        )
        return sx | st


def list_constitutive_models():
    from moju.monitor.closure_registry import list_models
    return list_models()


def list_scaling_closure_ids():
    from moju.monitor.closure_registry import list_groups
    return list_groups()
