"""
Constitutive Divergence card — where catalog ``F(pred)`` and the law-implied
quantity disagree.

For every constitutive audit row, the engine now stashes a sidecar entry on
``residuals["closure_debug"][basename]``::

    {
        "pred":     F(pred_args),                # always present
        "implied":  implied (subtract mode) | None,
        "raw":      pred - implied OR balance raw,
        "scale_a":  balance side a | None,
        "scale_b":  balance side b | None,
        "ref":      ref tensor (when normalisation uses it) | None,
        "mode":     "subtract" | "balance",
        "output_key": str,
        "law_name": str | None,
        "model_name": str,
    }

This module produces four visualisation modes plus a composite 2x2 dashboard.

Modes
-----

- ``"spatial"`` — three-panel heatmap row: model output, law-implied side,
  signed normalised divergence (diverging colormap centered on zero).
- ``"scatter"`` — pred vs implied scatter with the y = x identity line,
  colored by ``|divergence|``; reports % of points within ±1, ±5, ±10 %.
- ``"distribution"`` — histogram of normalised divergence with admissibility
  threshold bands (green / amber / red).
- ``"hotspot"`` — top-k worst points by ``|divergence|``, plotted in 2-D /
  3-D over collocation coordinates ``(x, y, [z], [t])``.

Composite entry: :func:`build_constitutive_divergence_dashboard` returns a
2x2 :class:`plotly.graph_objects.Figure` with one panel of each mode for
the basename whose final-step R_norm is largest (or the explicit basename
passed by the caller).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from moju.monitor.visualize_theme import (
    MOJU_LIGHT,
    apply_theme,
    get_theme,
    themed_axis_style,
    themed_colorbar,
)
from moju.monitor.visualize_components import _empty_card  # type: ignore[attr-defined]
from moju.monitor.visualize_labels import (
    pretty_residual_key,
    truncate_display_label,
)


_DIVERGENCE_EPS: float = 1e-30

# User-facing divergence panel wording (distinct from closure_debug JSON keys ``pred`` / ``implied``).
USER_CONSTIT_MODEL: str = "Model"
USER_CONSTIT_IMPLIED: str = "Implied"


def _user_constitutive_side_labels() -> Tuple[str, str]:
    return USER_CONSTIT_MODEL, USER_CONSTIT_IMPLIED


def divergence_y_quantity_label(debug_entry: Optional[Dict[str, Any]] = None) -> str:
    """Y-axis wording for spatial line panels."""
    ok = (debug_entry or {}).get("output_key")
    if ok:
        return f"Value ({ok})"
    return "Value"


def _spatial_position_axis_title(bundle_spatial: Any) -> str:
    """Match monitor spatial labelling."""
    sp = bundle_spatial if isinstance(bundle_spatial, dict) else None
    if not sp or sp.get("kind") != "1d":
        return "Position x"
    ax = sp.get("position_axis") or "x"
    return f"Position {ax}"


def infer_divergence_abscissa(
    bundle: Dict[str, Any],
    n: int,
    *,
    hint_axis: Optional[str] = None,
) -> Tuple[np.ndarray, str]:
    """
    Return ``(xs, x_title)`` aligned to length ``n`` for divergence line plots.

    Priority: when ``hint_axis`` / ``spatial_coord_hint`` is ``"t"`` and ``coord_snapshot["t"]``
    matches ``n``, use **Time t** (transient dashboards are not overwritten by spatial **x**);
    spatial 1-D ``bundle["spatial"]["x"]``; then remaining ``coord_snapshot`` axes (hint first
    among ``x``, ``y``, ``z``, ``t``, then defaults); fallback ``Sample index``.
    """
    if n <= 0:
        return np.asarray([], dtype=float), "Sample index"

    eff_hint: Optional[str] = None
    if isinstance(hint_axis, str) and hint_axis.strip():
        eff_hint = hint_axis.strip().lower()
    if eff_hint is None:
        bk = bundle.get("spatial_coord_hint")
        if isinstance(bk, str) and bk.strip():
            eff_hint = bk.strip().lower()

    cs: Any = {}
    log = bundle.get("log") or []
    if log and isinstance(log[-1], dict):
        cs = log[-1].get("coord_snapshot") or {}

    axes_order = ("x", "y", "z", "t")
    coord_titles = {"x": "Position x", "y": "Position y", "z": "Position z", "t": "Time t"}

    if isinstance(cs, dict) and eff_hint == "t":
        v = cs.get("t")
        arr = np.asarray(v, dtype=float).ravel() if v is not None else np.asarray([])
        if arr.shape[0] == n:
            return arr, coord_titles["t"]

    spatial_any = bundle.get("spatial") or {}
    if isinstance(spatial_any, dict) and spatial_any.get("kind") == "1d":
        x_sp = spatial_any.get("x")
        if x_sp is not None:
            arr = np.asarray(x_sp, dtype=float).ravel()
            if arr.shape[0] == n:
                return arr, _spatial_position_axis_title(spatial_any)
    if isinstance(cs, dict):
        probe = []
        if eff_hint and eff_hint in axes_order:
            probe.append(eff_hint)
        for ax in axes_order:
            if ax not in probe:
                probe.append(ax)
        for ax in probe:
            v = cs.get(ax)
            arr = np.asarray(v, dtype=float).ravel() if v is not None else np.asarray([])
            if arr.shape[0] == n:
                return arr, coord_titles.get(ax, f"Position {ax}")

    return np.arange(n, dtype=float), "Sample index"


def _closure_debug(bundle: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Resolve the ``closure_debug`` sidecar in a visualization bundle."""
    if not isinstance(bundle, dict):
        return {}
    cd = bundle.get("closure_debug")
    if isinstance(cd, dict) and cd:
        return cd
    residuals = bundle.get("residuals")
    if isinstance(residuals, dict):
        nested = residuals.get("closure_debug")
        if isinstance(nested, dict):
            return nested
    return {}


def _final_rnorm_by_basename(bundle: Dict[str, Any]) -> Dict[str, float]:
    """Map constitutive ``residual_basename`` → final-step R_norm."""
    plot_keys: Optional[List[str]] = bundle.get("plot_keys")
    r_norm_mat = bundle.get("r_norm_mat")
    if not plot_keys or r_norm_mat is None:
        return {}
    mat = np.asarray(r_norm_mat, dtype=float)
    if mat.ndim != 2 or mat.shape[1] != len(plot_keys):
        return {}
    finals = mat[-1, :]
    out: Dict[str, float] = {}
    for k, v in zip(plot_keys, finals):
        if not isinstance(k, str) or not k.startswith("constitutive/"):
            continue
        if not np.isfinite(v):
            continue
        # strip "constitutive/" prefix and "/implied_delta" / "/ref_delta" suffix
        bn = k[len("constitutive/"):]
        if bn.endswith("/implied_delta"):
            bn = bn[: -len("/implied_delta")]
        elif bn.endswith("/ref_delta"):
            bn = bn[: -len("/ref_delta")]
        out[bn] = max(out.get(bn, 0.0), float(v))
    return out


def _auto_select_basename(bundle: Dict[str, Any], debug: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """Pick the basename with the worst final-step R_norm; fall back to first."""
    finals = _final_rnorm_by_basename(bundle)
    if finals:
        candidates = [(bn, r) for bn, r in finals.items() if bn in debug]
        if candidates:
            candidates.sort(key=lambda kv: -kv[1])
            return candidates[0][0]
    return next(iter(debug.keys()), None)


def primary_closure_debug_field_length(bundle: Dict[str, Any]) -> Optional[int]:
    """Flattened sample count for selected ``closure_debug`` basename, if any."""
    debug = _closure_debug(bundle)
    if not debug:
        return None
    bn = _auto_select_basename(bundle, debug)
    if bn is None or bn not in debug:
        return None
    entry = debug[bn]
    if entry.get("mode") == "balance":
        arr = _coerce_to_numpy(entry.get("scale_a"))
    else:
        arr = _coerce_to_numpy(entry.get("pred"))
    arr = np.asarray(arr, dtype=float)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.size == 0:
        return None
    return int(arr.size) if arr.ndim <= 2 else None


def _coerce_to_numpy(x: Any) -> np.ndarray:
    try:
        return np.asarray(x, dtype=float)
    except Exception:  # noqa: BLE001
        return np.asarray([], dtype=float)


def _sides_for_divergence(
    debug_entry: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """
    Return ``(side_model, side_implied, label_model, label_implied)``.

    User-facing axis labels **Model** / **Implied** (closure_debug retains ``pred`` / ``implied`` keys).
    """
    lab_m, lab_i = _user_constitutive_side_labels()
    mode = debug_entry.get("mode")
    if mode == "balance":
        a = _coerce_to_numpy(debug_entry.get("scale_a"))
        b = _coerce_to_numpy(debug_entry.get("scale_b"))
        return a, b, lab_m, lab_i
    a = _coerce_to_numpy(debug_entry.get("pred"))
    b = _coerce_to_numpy(debug_entry.get("implied"))
    return a, b, lab_m, lab_i


def _normalized_divergence(a: np.ndarray, b: np.ndarray, ref: Optional[np.ndarray] = None) -> np.ndarray:
    """Mirror :func:`apply_closure_discrepancy_normalize` for numpy arrays."""
    raw = a - b
    if ref is not None and ref.size:
        denom = _DIVERGENCE_EPS + np.abs(ref)
    else:
        denom = _DIVERGENCE_EPS + np.abs(a) + np.abs(b)
    return raw / denom


def _flatten(a: np.ndarray) -> np.ndarray:
    """Flatten while broadcasting scalars to 1-element arrays."""
    if a.ndim == 0:
        return a.reshape(1)
    return a.reshape(-1)


def _coords_for_points(
    bundle: Dict[str, Any], n_points: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Best-effort flat (x, y, z, t) arrays of length ``n_points``."""
    log = bundle.get("log") or []
    if not log:
        return None, None, None, None
    cs = log[-1].get("coord_snapshot") if isinstance(log[-1], dict) else None
    if not isinstance(cs, dict):
        return None, None, None, None
    out: List[Optional[np.ndarray]] = []
    for axis in ("x", "y", "z", "t"):
        v = cs.get(axis)
        arr = np.asarray(v, dtype=float).ravel() if v is not None else None
        if arr is not None and arr.shape[0] != n_points:
            arr = None
        out.append(arr)
    return out[0], out[1], out[2], out[3]


# ---------------------------------------------------------------------------
# Mode builders
# ---------------------------------------------------------------------------


def build_spatial_divergence_panel(
    bundle: Dict[str, Any],
    residual_basename: Optional[str] = None,
    *,
    theme: Any = MOJU_LIGHT,
    height: Optional[int] = None,
    title: Optional[str] = None,
) -> Any:
    """Three-panel spatial heatmap: model, implied, normalised divergence."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    t = get_theme(theme)
    debug = _closure_debug(bundle)
    if not debug:
        return _empty_card(title or "Constitutive divergence (spatial)", "No closure_debug sidecar available", theme)
    bn = residual_basename or _auto_select_basename(bundle, debug)
    if bn is None or bn not in debug:
        return _empty_card(title or "Constitutive divergence (spatial)", "No constitutive basename to render", theme)
    entry = debug[bn]
    a_full = _coerce_to_numpy(entry.get("scale_a") if entry.get("mode") == "balance" else entry.get("pred"))
    b_full = _coerce_to_numpy(entry.get("scale_b") if entry.get("mode") == "balance" else entry.get("implied"))
    ref_full = entry.get("ref")
    ref_arr = _coerce_to_numpy(ref_full) if ref_full is not None else None
    if a_full.size == 0 or b_full.size == 0:
        return _empty_card(title or bn, "No spatial data for divergence", theme)
    if a_full.shape != b_full.shape:
        return _empty_card(title or bn, "Model and Implied shapes differ; cannot render side-by-side heatmaps", theme)

    label_model, label_implied = _user_constitutive_side_labels()
    hint_ax = bundle.get("spatial_coord_hint")

    # Squeeze leading singletons (e.g. batch dim)
    a = a_full
    b = b_full
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
        b = b[0]
    y_qty = divergence_y_quantity_label(entry)
    if a.ndim == 1:
        # Render as three 1-D line traces
        div = _normalized_divergence(a, b, ref=ref_arr.reshape(a.shape) if ref_arr is not None and ref_arr.size == a.size else None)
        xs, x_title = infer_divergence_abscissa(bundle, int(a.shape[0]), hint_axis=str(hint_ax) if hint_ax else None)
        fig = make_subplots(
            rows=1,
            cols=3,
            shared_yaxes=False,
            subplot_titles=(label_model, label_implied, "Normalised divergence"),
        )
        fig.add_trace(go.Scatter(x=xs, y=a, mode="lines", line=dict(color=t.palette.line_primary), name=label_model), row=1, col=1)
        fig.add_trace(go.Scatter(x=xs, y=b, mode="lines", line=dict(color=t.palette.cat_constitutive), name=label_implied), row=1, col=2)
        fig.add_trace(go.Scatter(x=xs, y=div, mode="lines", line=dict(color=t.palette.adm_low), name="Δ"), row=1, col=3)
        for col in (1, 2, 3):
            fig.update_xaxes(row=1, col=col, title_text=x_title, **themed_axis_style(theme, show_grid=False, zero_line=False))
            y_lab = "Normalised divergence" if col == 3 else y_qty
            fig.update_yaxes(row=1, col=col, title_text=y_lab, **themed_axis_style(theme))
        fig.update_layout(showlegend=False)
        return apply_theme(fig, theme, title=title or f"{pretty_residual_key(bn)} (divergence)", height=height or t.layout.card_height)

    if a.ndim != 2:
        return _empty_card(title or bn, f"Cannot render divergence for ndim={a.ndim} arrays", theme)

    div = _normalized_divergence(
        a,
        b,
        ref=ref_arr.reshape(a.shape) if ref_arr is not None and ref_arr.size == a.size else None,
    )
    abs_lim = float(np.nanpercentile(np.abs(div), 95)) if div.size else 1.0
    if not np.isfinite(abs_lim) or abs_lim == 0.0:
        abs_lim = 1.0

    coords = bundle.get("spatial") or {}
    cx = coords.get("coords", {}).get("x") if isinstance(coords, dict) else None
    cy = coords.get("coords", {}).get("y") if isinstance(coords, dict) else None
    cx = np.asarray(cx) if cx is not None and np.asarray(cx).shape[0] == a.shape[1] else None
    cy = np.asarray(cy) if cy is not None and np.asarray(cy).shape[0] == a.shape[0] else None

    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        subplot_titles=(label_model, label_implied, "Normalised divergence"),
    )
    fig.add_trace(
        go.Heatmap(
            z=a,
            x=cx,
            y=cy,
            colorscale=t.colorscales.sequential,
            colorbar=themed_colorbar(theme, title=label_model),
            hovertemplate="x=%{x:.3g}<br>y=%{y:.3g}<br>val=%{z:.3g}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=b,
            x=cx,
            y=cy,
            colorscale=t.colorscales.sequential_alt,
            colorbar=themed_colorbar(theme, title=label_implied),
            hovertemplate="x=%{x:.3g}<br>y=%{y:.3g}<br>val=%{z:.3g}<extra></extra>",
            showscale=True,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Heatmap(
            z=div,
            x=cx,
            y=cy,
            colorscale=t.colorscales.divergence,
            zmid=0,
            zmin=-abs_lim,
            zmax=abs_lim,
            colorbar=themed_colorbar(theme, title="(Model − Implied) / scale"),
            hovertemplate="x=%{x:.3g}<br>y=%{y:.3g}<br>Δ=%{z:.3g}<extra></extra>",
            showscale=True,
        ),
        row=1,
        col=3,
    )
    for col in (1, 2, 3):
        fig.update_xaxes(row=1, col=col, title_text="Position x", **themed_axis_style(theme, show_grid=False, zero_line=False))
        fig.update_yaxes(row=1, col=col, title_text="Position y", **themed_axis_style(theme, show_grid=False, zero_line=False))
    return apply_theme(fig, theme, title=title or f"{pretty_residual_key(bn)} (divergence)", height=height or t.layout.card_height)


def worst_div_mean_abs_row_index(div: np.ndarray) -> int:
    """Pick row ``y_index`` maximizing mean ``|Δ_norm|`` over ``x`` (monitor line slice)."""
    d = np.asarray(div, dtype=float)
    if d.ndim != 2 or d.shape[0] < 1:
        return 0
    row_means = np.nanmean(np.abs(d), axis=1)
    return int(np.nanargmax(row_means))


def _coords_pred_for_closure_slice(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Build coordinate dict compatible with ``_reduce_spatial_array`` (snapshot + spatial fallback)."""
    pred: Dict[str, Any] = {}
    log = bundle.get("log") or []
    if log and isinstance(log[-1], dict):
        cs = log[-1].get("coord_snapshot")
        if isinstance(cs, dict):
            for axis in ("x", "y", "z", "t"):
                v = cs.get(axis)
                if v is None:
                    continue
                pred[str(axis)] = np.asarray(v, dtype=float).ravel()
    sp_any = bundle.get("spatial")
    if isinstance(sp_any, dict):
        kind = sp_any.get("kind")
        if kind == "1d" and pred.get("x") is None and sp_any.get("x") is not None:
            pred["x"] = np.asarray(sp_any["x"], dtype=float).ravel()
        if kind == "2d":
            coords_dict = sp_any.get("coords") or {}
            cx = coords_dict.get("x") if coords_dict else sp_any.get("x")
            cy = coords_dict.get("y") if coords_dict else sp_any.get("y")
            if cx is not None:
                xv = np.asarray(cx, dtype=float).ravel()
                if pred.get("x") is None or xv.size > np.asarray(pred["x"]).size:
                    pred["x"] = xv
            if cy is not None and pred.get("y") is None:
                pred["y"] = np.asarray(cy, dtype=float).ravel()
    return pred


def _squeeze_matching_model_implied(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    while aa.ndim > 2 and aa.shape[0] == 1 and bb.ndim == aa.ndim and bb.shape[0] == 1:
        aa = aa[0]
        bb = bb[0]
    return aa, bb


def prepare_monitor_closure_divergence_embed(
    bundle: Dict[str, Any],
    *,
    prefer_last_t: bool = True,
    theme: Any = MOJU_LIGHT,
):
    """
    Plotly traces for the monitor divergence row only: normalized Δ (**left**) and Model+Implied
    vs ``x`` (**right**); last ``t`` slice when coordinates align.

    Returns ``dict`` or ``None`` if nothing to embed. Keys include ``left_traces``, ``right_traces``,
    ``left_kind`` (``\"heatmap\"`` | ``\"scatter\"``).
    """
    from moju.monitor.spatial_rnorm_panels import _reduce_spatial_array

    import plotly.graph_objects as go

    debug = _closure_debug(bundle)
    if not debug:
        return None
    bn = _auto_select_basename(bundle, debug)
    if bn is None or bn not in debug:
        return None
    entry = dict(debug[bn])
    a0, b0, lab_a, lab_b = _sides_for_divergence(entry)
    a0, b0 = _squeeze_matching_model_implied(a0, b0)
    if a0.size == 0 or b0.size == 0 or a0.shape != b0.shape:
        return None

    ref_np: Optional[np.ndarray] = None
    ref_raw = entry.get("ref")
    if ref_raw is not None:
        ref_candidate = np.asarray(ref_raw, dtype=float)
        if ref_candidate.shape == a0.shape:
            ref_np = ref_candidate

    coords_pred = _coords_pred_for_closure_slice(bundle)

    try:
        a_red = np.asarray(_reduce_spatial_array(a0, coords_pred, prefer_last_t=prefer_last_t), dtype=float)
        b_red = np.asarray(_reduce_spatial_array(b0, coords_pred, prefer_last_t=prefer_last_t), dtype=float)
        ref_red: Optional[np.ndarray] = None
        if ref_np is not None and ref_np.size > 0:
            ref_try = np.asarray(_reduce_spatial_array(ref_np, coords_pred, prefer_last_t=prefer_last_t), dtype=float)
            if ref_try.shape == a_red.shape:
                ref_red = ref_try
    except (TypeError, ValueError):
        return None

    if a_red.ndim == 0 or b_red.ndim == 0 or a_red.shape != b_red.shape:
        return None

    hint_ax = bundle.get("spatial_coord_hint")

    div = _normalized_divergence(
        a_red,
        b_red,
        ref=ref_red if ref_red is not None else None,
    )

    line_y_qty = divergence_y_quantity_label(entry)
    t = get_theme(theme)

    out: Dict[str, Any] = {"debug_entry": entry, "left_kind": "", "left_traces": [], "right_traces": []}

    def _fill_scatter_from_1d(xs: np.ndarray, x_lab: str) -> bool:
        d2 = np.asarray(div, dtype=float)
        aa = np.asarray(a_red, dtype=float)
        bb = np.asarray(b_red, dtype=float)
        if d2.ndim != 1 or aa.ndim != 1 or bb.ndim != 1:
            return False
        nx = int(xs.shape[0])
        if d2.shape[0] != nx or aa.shape[0] != nx or bb.shape[0] != nx:
            return False
        lo, hi = float(np.nanmin(xs)), float(np.nanmax(xs))
        xrange_kw: Dict[str, Any] = dict(range=[lo, hi]) if np.isfinite(lo) and np.isfinite(hi) and hi > lo else {}
        out["left_traces"] = [
            go.Scatter(
                x=np.asarray(xs, dtype=float),
                y=d2,
                mode="lines",
                line=dict(color=t.palette.adm_low, width=2),
                name="Normalised Δ",
                showlegend=False,
            )
        ]
        out["right_traces"] = [
            go.Scatter(
                x=np.asarray(xs, dtype=float),
                y=aa,
                mode="lines",
                line=dict(color=t.palette.line_primary, width=2),
                name=lab_a,
                showlegend=False,
                hovertemplate=f"%{{x:.4g}}<br>{lab_a}=%{{y:.4g}}<extra></extra>",
            ),
            go.Scatter(
                x=np.asarray(xs, dtype=float),
                y=bb,
                mode="lines",
                line=dict(color=t.palette.cat_constitutive, width=2),
                name=lab_b,
                showlegend=False,
                hovertemplate=f"%{{x:.4g}}<br>{lab_b}=%{{y:.4g}}<extra></extra>",
            ),
        ]
        out["left_kind"] = "scatter"
        out["line_x_title"] = x_lab
        out["line_y_title_delta"] = "Normalised divergence"
        out["line_y_title_mi"] = line_y_qty
        out["scatter_xrange"] = xrange_kw
        return True

    if np.asarray(div).ndim == 1:
        n = int(np.asarray(div).shape[0])
        xs, x_lab = infer_divergence_abscissa(bundle, n, hint_axis=str(hint_ax) if hint_ax else None)
        xv = np.asarray(xs, dtype=float).ravel()
        if xv.shape[0] == n:
            if _fill_scatter_from_1d(xv, x_lab):
                return out
        if _fill_scatter_from_1d(np.arange(n, dtype=float), x_lab):
            return out
        return None

    d2 = np.asarray(div, dtype=float)
    if d2.ndim != 2:
        return None

    coords = bundle.get("spatial")
    coords = coords if isinstance(coords, dict) else {}
    cxx: Optional[np.ndarray] = None
    cyy: Optional[np.ndarray] = None
    cds_raw = coords.get("coords") or {}
    xv0 = cds_raw.get("x") if cds_raw else coords.get("x")
    yv0 = cds_raw.get("y") if cds_raw else coords.get("y")
    if xv0 is not None:
        xv = np.asarray(xv0, dtype=float).ravel()
        if xv.size == d2.shape[1]:
            cxx = xv
    if yv0 is not None:
        yvv = np.asarray(yv0, dtype=float).ravel()
        if yvv.size == d2.shape[0]:
            cyy = yvv
    if cxx is None and coords_pred.get("x") is not None:
        px = np.asarray(coords_pred["x"], dtype=float).ravel()
        if px.size == d2.shape[1]:
            cxx = px
    if cyy is None and coords_pred.get("y") is not None:
        py = np.asarray(coords_pred["y"], dtype=float).ravel()
        if py.size == d2.shape[0]:
            cyy = py
    nx, ny = d2.shape[1], d2.shape[0]
    if cxx is None:
        cxx_infer, _ = infer_divergence_abscissa(bundle, nx, hint_axis=str(hint_ax) if hint_ax else None)
        cxx = np.asarray(cxx_infer, dtype=float).ravel()
        if cxx.size != nx:
            cxx = np.linspace(0.0, 1.0, nx, dtype=float)
    if cyy is None:
        cyy = np.linspace(0.0, 1.0, ny, dtype=float)

    abs_lim = float(np.nanpercentile(np.abs(d2), 95)) if d2.size else 1.0
    if not np.isfinite(abs_lim) or abs_lim == 0.0:
        abs_lim = 1.0
    out["left_traces"] = [
        go.Heatmap(
            z=d2,
            x=np.asarray(cxx, dtype=float),
            y=np.asarray(cyy, dtype=float),
            colorscale=t.colorscales.divergence,
            zmid=0,
            zmin=-abs_lim,
            zmax=abs_lim,
            colorbar=themed_colorbar(theme, title="Δ_norm"),
            hovertemplate="x=%{x:.3g}<br>y=%{y:.3g}<br>Δ=%{z:.3g}<extra></extra>",
        )
    ]
    out["left_kind"] = "heatmap"

    y_ix = worst_div_mean_abs_row_index(d2)
    y_pick = float(np.asarray(cyy[y_ix]))
    xm = np.asarray(np.asarray(a_red, dtype=float)[y_ix], dtype=float).ravel()
    xi = np.asarray(np.asarray(b_red, dtype=float)[y_ix], dtype=float).ravel()
    x_line = np.asarray(cxx, dtype=float).ravel()
    if xm.size != x_line.size or xi.size != x_line.size:
        return None
    lo, hi = float(np.nanmin(x_line)), float(np.nanmax(x_line))
    xrange_kw: Dict[str, Any] = dict(range=[lo, hi]) if np.isfinite(lo) and np.isfinite(hi) and hi > lo else {}

    stripe_note = "(final t, worst |Δ| stripe)" if prefer_last_t else "(worst |Δ| stripe)"
    out["right_traces"] = [
        go.Scatter(
            x=x_line,
            y=xm,
            mode="lines",
            line=dict(color=t.palette.line_primary, width=2),
            showlegend=False,
            customdata=np.full(x_line.shape, y_pick),
            hovertemplate=f"y*=%{{customdata:.4g}} {stripe_note}<br>%{{x:.4g}}<br>{lab_a}=%{{y:.4g}}<extra></extra>",
        ),
        go.Scatter(
            x=x_line,
            y=xi,
            mode="lines",
            line=dict(color=t.palette.cat_constitutive, width=2),
            showlegend=False,
            customdata=np.full(x_line.shape, y_pick),
            hovertemplate=f"y*=%{{customdata:.4g}} {stripe_note}<br>%{{x:.4g}}<br>{lab_b}=%{{y:.4g}}<extra></extra>",
        ),
    ]
    out["stripe_y_value"] = y_pick
    out["stripe_row_index"] = y_ix
    out["line_x_title"] = _spatial_position_axis_title(coords)
    out["line_y_title_mi"] = line_y_qty
    out["scatter_xrange"] = xrange_kw
    return out


def build_scatter_divergence_panel(
    bundle: Dict[str, Any],
    residual_basename: Optional[str] = None,
    *,
    theme: Any = MOJU_LIGHT,
    height: Optional[int] = None,
    title: Optional[str] = None,
) -> Any:
    """Scatter of model vs implied; y=x identity line; colored by |Δ|."""
    import plotly.graph_objects as go

    t = get_theme(theme)
    debug = _closure_debug(bundle)
    if not debug:
        return _empty_card(title or "Constitutive scatter", "No closure_debug sidecar available", theme)
    bn = residual_basename or _auto_select_basename(bundle, debug)
    if bn is None or bn not in debug:
        return _empty_card(title or "Constitutive scatter", "No constitutive basename to render", theme)
    entry = debug[bn]
    a, b, lab_a, lab_b = _sides_for_divergence(entry)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return _empty_card(title or bn, "Cannot align model / implied for scatter", theme)
    af = _flatten(a)
    bf = _flatten(b)
    mask = np.isfinite(af) & np.isfinite(bf)
    af = af[mask]
    bf = bf[mask]
    if af.size == 0:
        return _empty_card(title or bn, "No finite points", theme)
    div = _normalized_divergence(af, bf)
    abs_div = np.abs(div)

    # Identity line spanning the data extent
    lo = float(np.nanmin([np.nanmin(af), np.nanmin(bf)]))
    hi = float(np.nanmax([np.nanmax(af), np.nanmax(bf)]))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = 0.0, 1.0
    if lo == hi:
        hi = lo + 1.0
    pad = 0.05 * (hi - lo)

    # Tolerance band statistics
    pct_1 = float(np.mean(abs_div <= 0.01) * 100.0)
    pct_5 = float(np.mean(abs_div <= 0.05) * 100.0)
    pct_10 = float(np.mean(abs_div <= 0.10) * 100.0)
    cx, cy, cz, ct = _coords_for_points(bundle, af.size)
    custom = np.column_stack([
        cx if cx is not None else np.full(af.size, np.nan),
        cy if cy is not None else np.full(af.size, np.nan),
        cz if cz is not None else np.full(af.size, np.nan),
        ct if ct is not None else np.full(af.size, np.nan),
        div,
    ])
    hover = (
        f"<b>{lab_a}</b>=%{{x:.4g}}<br>"
        f"<b>{lab_b}</b>=%{{y:.4g}}<br>"
        "Δ_norm=%{customdata[4]:.3g}<br>"
        "x=%{customdata[0]:.3g} y=%{customdata[1]:.3g}"
        " z=%{customdata[2]:.3g} t=%{customdata[3]:.3g}"
        "<extra></extra>"
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[lo - pad, hi + pad],
            y=[lo - pad, hi + pad],
            mode="lines",
            line=dict(color=t.palette.muted, dash="dash", width=1.4),
            hoverinfo="skip",
            showlegend=False,
            name="y = x",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=af,
            y=bf,
            mode="markers",
            marker=dict(
                size=6,
                color=abs_div,
                colorscale=t.colorscales.divergence,
                cmin=0.0,
                cmax=float(np.nanpercentile(abs_div, 98)) if abs_div.size else 1.0,
                colorbar=themed_colorbar(theme, title="|Δ_norm|"),
                line=dict(color=t.palette.bar_line, width=0.4),
            ),
            customdata=custom,
            hovertemplate=hover,
            showlegend=False,
        )
    )
    annot = (
        f"Within ±1%: {pct_1:.1f}%<br>"
        f"Within ±5%: {pct_5:.1f}%<br>"
        f"Within ±10%: {pct_10:.1f}%"
    )
    fig.add_annotation(
        text=annot,
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        showarrow=False,
        align="left",
        font=t.font_dict(size=11, color=t.palette.muted),
        bgcolor=t.palette.summary_bg,
        bordercolor=t.palette.summary_border,
        borderwidth=1,
        borderpad=6,
    )
    fig.update_xaxes(title_text=lab_a, range=[lo - pad, hi + pad], **themed_axis_style(theme))
    fig.update_yaxes(title_text=lab_b, range=[lo - pad, hi + pad], **themed_axis_style(theme))
    return apply_theme(fig, theme, title=title or f"{pretty_residual_key(bn)} (scatter)", height=height or t.layout.card_height)


def build_distribution_divergence_panel(
    bundle: Dict[str, Any],
    residual_basename: Optional[str] = None,
    *,
    theme: Any = MOJU_LIGHT,
    height: Optional[int] = None,
    title: Optional[str] = None,
) -> Any:
    """Histogram of normalised divergence with admissibility threshold bands."""
    import plotly.graph_objects as go

    t = get_theme(theme)
    debug = _closure_debug(bundle)
    if not debug:
        return _empty_card(title or "Divergence distribution", "No closure_debug sidecar available", theme)
    bn = residual_basename or _auto_select_basename(bundle, debug)
    if bn is None or bn not in debug:
        return _empty_card(title or "Divergence distribution", "No constitutive basename to render", theme)
    entry = debug[bn]
    a, b, _, _ = _sides_for_divergence(entry)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return _empty_card(title or bn, "Cannot align model / implied for distribution", theme)
    div = _normalized_divergence(_flatten(a), _flatten(b))
    div = div[np.isfinite(div)]
    if div.size == 0:
        return _empty_card(title or bn, "No finite divergence values", theme)
    abs_lim = float(np.nanpercentile(np.abs(div), 99)) if div.size else 1.0
    if not np.isfinite(abs_lim) or abs_lim == 0.0:
        abs_lim = 1.0
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=div,
            nbinsx=72,
            marker=dict(color=t.palette.cat_constitutive, line=dict(color=t.palette.bar_line, width=0.4)),
            opacity=0.92,
            name="Δ_norm",
            hovertemplate="bin %{x:.3g}<br>count: %{y}<extra></extra>",
        )
    )
    # Threshold bands (mirrors admissibility green/amber/red zones)
    band_specs = [
        (-0.01, 0.01, t.palette.adm_high, "±1%"),
        (-0.05, -0.01, t.palette.adm_med, ""),
        (0.01, 0.05, t.palette.adm_med, "±5%"),
        (-0.10, -0.05, t.palette.adm_low, ""),
        (0.05, 0.10, t.palette.adm_low, "±10%"),
    ]
    for x0, x1, color, _label in band_specs:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.08, line_width=0)
    fig.add_vline(x=0.0, line=dict(color=t.palette.muted, width=1, dash="dot"))

    pct_band = lambda thresh: float(np.mean(np.abs(div) <= thresh) * 100.0)
    annot = (
        f"|Δ| ≤ 1%: {pct_band(0.01):.1f}%<br>"
        f"|Δ| ≤ 5%: {pct_band(0.05):.1f}%<br>"
        f"|Δ| ≤ 10%: {pct_band(0.10):.1f}%"
    )
    fig.add_annotation(
        text=annot,
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        showarrow=False,
        align="left",
        font=t.font_dict(size=11, color=t.palette.muted),
        bgcolor=t.palette.summary_bg,
        bordercolor=t.palette.summary_border,
        borderwidth=1,
        borderpad=6,
    )
    rng = max(abs_lim, 0.12)
    fig.update_xaxes(title_text="(Model − Implied) / scale", range=[-rng, rng], **themed_axis_style(theme))
    fig.update_yaxes(title_text="count", **themed_axis_style(theme))
    fig.update_layout(showlegend=False)
    return apply_theme(fig, theme, title=title or f"{pretty_residual_key(bn)} (distribution)", height=height or t.layout.card_height)


def build_hotspot_divergence_panel(
    bundle: Dict[str, Any],
    residual_basename: Optional[str] = None,
    *,
    theme: Any = MOJU_LIGHT,
    height: Optional[int] = None,
    title: Optional[str] = None,
    top_k: int = 32,
) -> Any:
    """Top-``k`` worst points spatially over collocation coordinates (2-D or 3-D)."""
    import plotly.graph_objects as go

    t = get_theme(theme)
    debug = _closure_debug(bundle)
    if not debug:
        return _empty_card(title or "Divergence hotspots", "No closure_debug sidecar available", theme)
    bn = residual_basename or _auto_select_basename(bundle, debug)
    if bn is None or bn not in debug:
        return _empty_card(title or "Divergence hotspots", "No constitutive basename to render", theme)
    entry = debug[bn]
    a, b, lab_a, lab_b = _sides_for_divergence(entry)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return _empty_card(title or bn, "Cannot align model / implied for hotspots", theme)
    af = _flatten(a)
    bf = _flatten(b)
    div = _normalized_divergence(af, bf)
    abs_div = np.abs(div)
    mask = np.isfinite(abs_div)
    if not mask.any():
        return _empty_card(title or bn, "No finite divergence", theme)
    af = af[mask]
    bf = bf[mask]
    div = div[mask]
    abs_div = abs_div[mask]
    cx, cy, cz, ct = _coords_for_points(bundle, af.size)

    order = np.argsort(-abs_div)[:top_k]
    af = af[order]
    bf = bf[order]
    div = div[order]
    abs_div = abs_div[order]
    cx_ = cx[order] if cx is not None else None
    cy_ = cy[order] if cy is not None else None
    cz_ = cz[order] if cz is not None else None
    ct_ = ct[order] if ct is not None else None

    size_scaled = 8 + 24 * (abs_div / (abs_div.max() + _DIVERGENCE_EPS))
    abs_lim = float(np.nanpercentile(abs_div, 95)) if abs_div.size else 1.0
    if not np.isfinite(abs_lim) or abs_lim == 0.0:
        abs_lim = 1.0

    has_2d = cx_ is not None and cy_ is not None
    has_3d = has_2d and cz_ is not None

    customdata = np.column_stack([
        cx_ if cx_ is not None else np.full(af.size, np.nan),
        cy_ if cy_ is not None else np.full(af.size, np.nan),
        cz_ if cz_ is not None else np.full(af.size, np.nan),
        ct_ if ct_ is not None else np.full(af.size, np.nan),
        af,
        bf,
        div,
    ])
    hover = (
        "x=%{customdata[0]:.3g} y=%{customdata[1]:.3g}"
        " z=%{customdata[2]:.3g} t=%{customdata[3]:.3g}<br>"
        f"{lab_a}=%{{customdata[4]:.3g}}<br>"
        f"{lab_b}=%{{customdata[5]:.3g}}<br>"
        "Δ_norm=%{customdata[6]:.3g}"
        "<extra></extra>"
    )

    fig = go.Figure()
    if has_3d:
        fig.add_trace(
            go.Scatter3d(
                x=cx_,
                y=cy_,
                z=cz_,
                mode="markers",
                marker=dict(
                    size=size_scaled,
                    color=div,
                    cmin=-abs_lim,
                    cmax=abs_lim,
                    colorscale=t.colorscales.divergence,
                    colorbar=themed_colorbar(theme, title="Δ_norm"),
                    line=dict(color=t.palette.bar_line, width=0.3),
                ),
                customdata=customdata,
                hovertemplate=hover,
                showlegend=False,
            )
        )
        fig.update_layout(
            scene=dict(
                xaxis=dict(title="Position x", title_font=t.section_title_font_dict(), tickfont=t.tick_font_dict()),
                yaxis=dict(title="Position y", title_font=t.section_title_font_dict(), tickfont=t.tick_font_dict()),
                zaxis=dict(title="Position z", title_font=t.section_title_font_dict(), tickfont=t.tick_font_dict()),
                aspectmode="data",
            ),
        )
    elif has_2d:
        fig.add_trace(
            go.Scatter(
                x=cx_,
                y=cy_,
                mode="markers",
                marker=dict(
                    size=size_scaled,
                    color=div,
                    cmin=-abs_lim,
                    cmax=abs_lim,
                    colorscale=t.colorscales.divergence,
                    colorbar=themed_colorbar(theme, title="Δ_norm"),
                    line=dict(color=t.palette.bar_line, width=0.4),
                ),
                customdata=customdata,
                hovertemplate=hover,
                showlegend=False,
            )
        )
        fig.update_xaxes(title_text="Position x", **themed_axis_style(theme))
        fig.update_yaxes(title_text="Position y", **themed_axis_style(theme))
    else:
        hint_ax = bundle.get("spatial_coord_hint")
        bx, xt = infer_divergence_abscissa(bundle, af.size, hint_axis=str(hint_ax) if hint_ax else None)
        fig.add_trace(
            go.Bar(
                x=bx,
                y=abs_div,
                marker=dict(color=div, cmin=-abs_lim, cmax=abs_lim, colorscale=t.colorscales.divergence, colorbar=themed_colorbar(theme, title="Δ_norm")),
                customdata=customdata,
                hovertemplate=hover,
                showlegend=False,
            )
        )
        fig.update_xaxes(title_text=xt, **themed_axis_style(theme))
        fig.update_yaxes(title_text="|Δ_norm|", **themed_axis_style(theme))
    return apply_theme(fig, theme, title=title or f"{pretty_residual_key(bn)} (top-{top_k} hotspots)", height=height or t.layout.card_height)


# ---------------------------------------------------------------------------
# Composite + dispatcher
# ---------------------------------------------------------------------------


_MODES = ("spatial", "scatter", "distribution", "hotspot")


def build_constitutive_divergence_card(
    bundle: Dict[str, Any],
    *,
    residual_basename: Optional[str] = None,
    mode: str = "spatial",
    theme: Any = MOJU_LIGHT,
    height: Optional[int] = None,
    title: Optional[str] = None,
    **mode_kwargs: Any,
) -> Any:
    """
    Single-mode constitutive divergence card.

    Parameters
    ----------
    bundle:
        Visualisation bundle (must contain ``closure_debug``).
    residual_basename:
        Specific constitutive ``residual_basename`` (e.g.
        ``"thermal_diffusivity/law_fourier_conduction"``).  When ``None``,
        auto-selects the basename with the worst final-step R_norm.
    mode:
        One of ``"spatial"``, ``"scatter"``, ``"distribution"``, ``"hotspot"``.
    """
    if mode not in _MODES:
        raise ValueError(f"Unknown divergence mode {mode!r}; expected one of {_MODES}")
    if mode == "spatial":
        return build_spatial_divergence_panel(
            bundle, residual_basename, theme=theme, height=height, title=title, **mode_kwargs
        )
    if mode == "scatter":
        return build_scatter_divergence_panel(
            bundle, residual_basename, theme=theme, height=height, title=title, **mode_kwargs
        )
    if mode == "distribution":
        return build_distribution_divergence_panel(
            bundle, residual_basename, theme=theme, height=height, title=title, **mode_kwargs
        )
    return build_hotspot_divergence_panel(
        bundle, residual_basename, theme=theme, height=height, title=title, **mode_kwargs
    )


def build_constitutive_divergence_dashboard(
    bundle: Dict[str, Any],
    *,
    residual_basename: Optional[str] = None,
    theme: Any = MOJU_LIGHT,
    height: Optional[int] = None,
    title: Optional[str] = None,
) -> Any:
    """
    Composite 2x2 constitutive divergence dashboard.

    Top row: spatial divergence | scatter (pred vs implied).
    Bottom row: distribution (histogram) | hotspot map.

    When ``residual_basename`` is ``None``, the worst-performing constitutive
    basename (largest final-step R_norm) is selected automatically so the
    operator sees the most informative divergence first.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    t = get_theme(theme)
    debug = _closure_debug(bundle)
    if not debug:
        return _empty_card(title or "Constitutive divergence", "No closure_debug sidecar available", theme)
    bn = residual_basename or _auto_select_basename(bundle, debug)
    if bn is None or bn not in debug:
        return _empty_card(title or "Constitutive divergence", "No constitutive basename to render", theme)

    sub_specs = [[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]]
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=sub_specs,
        horizontal_spacing=0.10,
        vertical_spacing=0.14,
        subplot_titles=("Spatial divergence", "Model vs Implied", "Divergence distribution", "Top hotspots"),
    )

    panels = [
        build_spatial_divergence_panel(bundle, bn, theme=theme),
        build_scatter_divergence_panel(bundle, bn, theme=theme),
        build_distribution_divergence_panel(bundle, bn, theme=theme),
        build_hotspot_divergence_panel(bundle, bn, theme=theme),
    ]
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for panel, (row, col) in zip(panels, positions):
        for trace in panel.data:
            # Subplots only support a subset of trace types; if hotspot uses
            # Scatter3d we keep the existing layout but skip nesting
            if trace.type in ("scatter3d", "volume", "surface"):
                continue
            # Avoid duplicating colorbars across the composite
            try:
                if hasattr(trace, "marker") and getattr(trace.marker, "colorbar", None) is not None:
                    trace.marker.colorbar = None  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
            try:
                if hasattr(trace, "colorbar") and getattr(trace, "colorbar", None) is not None:
                    trace.colorbar = None  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
            try:
                if hasattr(trace, "showscale"):
                    trace.showscale = False  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
            fig.add_trace(trace, row=row, col=col)

    for row in (1, 2):
        for col in (1, 2):
            fig.update_xaxes(row=row, col=col, **themed_axis_style(theme, show_grid=False, zero_line=False))
            fig.update_yaxes(row=row, col=col, **themed_axis_style(theme, show_grid=False, zero_line=False))
    fig.update_layout(showlegend=False)
    target_height = height or int(t.layout.card_height * 1.9)
    return apply_theme(fig, theme, title=title or f"Constitutive divergence — {pretty_residual_key(bn)}", height=target_height)


def list_constitutive_basenames(bundle: Dict[str, Any]) -> List[str]:
    """Return all available constitutive basenames in the bundle, sorted by worst R_norm first."""
    debug = _closure_debug(bundle)
    if not debug:
        return []
    finals = _final_rnorm_by_basename(bundle)
    keys = list(debug.keys())
    keys.sort(key=lambda k: -finals.get(k, 0.0))
    return keys


__all__ = [
    "USER_CONSTIT_MODEL",
    "USER_CONSTIT_IMPLIED",
    "build_constitutive_divergence_card",
    "build_constitutive_divergence_dashboard",
    "build_distribution_divergence_panel",
    "build_hotspot_divergence_panel",
    "build_scatter_divergence_panel",
    "build_spatial_divergence_panel",
    "divergence_y_quantity_label",
    "infer_divergence_abscissa",
    "list_constitutive_basenames",
    "prepare_monitor_closure_divergence_embed",
    "primary_closure_debug_field_length",
    "worst_div_mean_abs_row_index",
]
