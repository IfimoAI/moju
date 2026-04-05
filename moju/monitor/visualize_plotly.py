"""
Interactive Plotly dashboard for :func:`moju.monitor.auditor.visualize`.

Requires ``pip install plotly`` (optional extra ``moju[viz]``).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from moju.monitor.visualize_labels import (
    category_adm_bar_axis_range_percent_full,
    format_admissibility_pct,
    pretty_category_name,
    truncate_display_label,
)

R_NORM_LOG_EPS = 1e-12

# Fixed height for Moju Studio Dashboard Plotly cards (law/adm bars + spatial heatmaps).
MOJU_STUDIO_DASHBOARD_CARD_HEIGHT = 400

RESIDUAL_COLOR_LAWS = "#8B5CF6"  # royal purple
RESIDUAL_COLOR_CONSTITUTIVE = "#14B8A6"  # teal
RESIDUAL_COLOR_OTHER = "#6b7280"  # neutral fallback
ADMISSIBLE_COLOR = "#10B981"
DISSONANCE_COLOR = "#EF4444"


def _residual_color_from_key(flat_key: str) -> str:
    key = str(flat_key or "")
    if key.startswith("laws/"):
        return RESIDUAL_COLOR_LAWS
    if key.startswith("constitutive/"):
        return RESIDUAL_COLOR_CONSTITUTIVE
    return RESIDUAL_COLOR_OTHER


def _adm_bar_color_plotly(score: float) -> str:
    if not math.isfinite(score):
        return "#bdc3c7"
    if score >= 0.9:
        return "#27ae60"
    if score >= 0.7:
        return "#e67e22"
    return "#c0392b"


def _three_pillar_labels_values(metrics: List[Dict[str, Any]]) -> Tuple[List[str], List[float]]:
    last_cat = metrics[-1]["category_admissibility_score"]
    order = ("laws", "constitutive")
    labels = [pretty_category_name(c) for c in order]
    vals = [
        float(last_cat[c]) if c in last_cat and math.isfinite(float(last_cat[c])) else float("nan")
        for c in order
    ]
    return labels, vals


def _admissibility_status_hml_plotly(score: float) -> str:
    if not math.isfinite(score):
        return "N/A"
    if score >= 0.9:
        return "HIGH"
    if score >= 0.7:
        return "MODERATE"
    return "LOW"


def format_admissibility_status_label(score: float) -> str:
    """Public alias for HIGH / MODERATE / LOW bands (same as Plotly dashboards)."""
    return _admissibility_status_hml_plotly(score)


def _plotly_layout_axis_domain(fig: Any, axis_ref: Optional[str]) -> Optional[Tuple[float, float]]:
    """Return ``(lo, hi)`` paper-domain for a subplot axis (e.g. ``y3``, ``x``)."""
    ref = (axis_ref or "y").strip()
    try:
        if ref in ("x", "y"):
            ax_obj = getattr(fig.layout, f"{ref}axis", None)
        elif len(ref) >= 2 and ref[0] in "xy" and ref[1:].isdigit():
            ax_obj = getattr(fig.layout, f"{ref[0]}axis{ref[1:]}", None)
        else:
            ax_obj = getattr(fig.layout, f"{ref}axis", None)
        if ax_obj is None:
            return None
        dom = getattr(ax_obj, "domain", None)
        if dom is None or len(dom) < 2:
            return None
        return (float(dom[0]), float(dom[1]))
    except Exception:  # noqa: BLE001
        return None


def align_heatmap_colorbars_to_subplot_domains(fig: Any) -> None:
    """
    Shrink and anchor each Heatmap colorbar to its subplot's y-domain and place it beside the heatmap x-domain.
    Call after ``fig.update_layout`` so axis domains are finalized.
    """
    for tr in fig.data:
        if getattr(tr, "type", None) != "heatmap":
            continue
        cb = getattr(tr, "colorbar", None)
        if cb is None:
            continue
        yref = getattr(tr, "yaxis", None) or "y"
        xref = getattr(tr, "xaxis", None) or "x"
        ydom = _plotly_layout_axis_domain(fig, yref)
        xdom = _plotly_layout_axis_domain(fig, xref)
        if ydom is None or xdom is None:
            continue
        y0, y1 = ydom
        x0, x1 = xdom
        y_mid = 0.5 * (y0 + y1)
        y_len = max(0.06, (y1 - y0) * 0.88)
        x_cb = min(x1 + 0.012, 0.992)
        thick = getattr(cb, "thickness", None)
        if thick is None:
            thick = 12
        title = getattr(cb, "title", None)
        title_dict: Dict[str, Any] = {}
        if title is not None:
            if getattr(title, "text", None):
                title_dict["text"] = title.text
            if getattr(title, "side", None):
                title_dict["side"] = title.side
            if getattr(title, "font", None) is not None:
                title_dict["font"] = title.font
        cb_kwargs: Dict[str, Any] = dict(
            len=y_len,
            y=y_mid,
            yanchor="middle",
            x=x_cb,
            xanchor="left",
            thickness=thick,
            xpad=getattr(cb, "xpad", None) or 8,
        )
        if title_dict:
            cb_kwargs["title"] = title_dict
        tr.update(colorbar=cb_kwargs)


def _spatial_log_step_suffix(spatial: Dict[str, Any]) -> str:
    li = spatial.get("log_step_index")
    if li is None:
        return ""
    try:
        return f" — log step {int(li)}"
    except (TypeError, ValueError):
        return ""


def _heatmap_subplot_title_from_spatial(sp: Optional[Dict[str, Any]]) -> str:
    """Human-readable title from spatial ``row_labels`` (law / constitutive names)."""
    if not sp:
        return ""
    row_labels = list(sp.get("row_labels") or [])
    if not row_labels:
        return ""
    first = truncate_display_label(str(row_labels[0]), 40)
    if len(row_labels) == 1:
        return first
    return f"{first} (+{len(row_labels) - 1} more)"


def _spatial_heatmap_subplot_title(sp: Optional[Dict[str, Any]], fallback: str) -> str:
    h = _heatmap_subplot_title_from_spatial(sp)
    if h:
        return h + _spatial_log_step_suffix(sp or {})
    return fallback


def _convergence_subplot_title(cat: str, info: Dict[str, Any]) -> str:
    displays = list(info.get("displays") or [])
    keys = list(info.get("keys") or [])
    if len(keys) == 1 and displays:
        return f"{displays[0]} Convergence"
    return f"{pretty_category_name(cat)} Convergence"


def _plotly_spatial_panel_title_with_subtitle(main: str, rnorm_y_title: str) -> str:
    """Two-line subplot title: main + R_norm scale as smaller second line (HTML)."""
    esc = (
        str(rnorm_y_title)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return f"{main}<br><span style='font-size:10px'>{esc}</span>"


def _plotly_add_spatial_panel_to_subplot(
    fig: Any,
    *,
    row: int,
    col: int,
    spatial: Dict[str, Any],
    hm_cs: str,
    mid: float,
    colorbar_compact: bool,
    use_log_rnorm: bool = True,
    colorbar_scale_title: str = "log10(|residual| + ε)",
) -> None:
    """Add law or constitutive spatial trace (1D keys×x, 2D x×y, or 3D z-slice) to a subplot cell."""
    import numpy as np
    import plotly.graph_objects as go

    kind = spatial.get("kind", "1d")
    row_labels: List[str] = list(spatial["row_labels"])

    cb_kw: Dict[str, Any] = dict(
        title=dict(text=colorbar_scale_title, side="right"),
        len=0.38 if colorbar_compact else 0.36,
        xpad=8,
        thickness=14,
    )
    if not colorbar_compact:
        cb_kw["x"] = 1.01
        cb_kw["xpad"] = 10

    def _z_display(z: Any) -> Any:
        a = np.asarray(z, dtype=float)
        if use_log_rnorm:
            return np.log10(np.maximum(a, 0.0) + R_NORM_LOG_EPS)
        return a

    if kind == "1d":
        Z = _z_display(spatial["Z"])
        x_sp = spatial["x"]
        pos_ax = spatial.get("position_axis") or "x"
        li = spatial.get("log_step_index")
        ht_1d = (
            "x=%{x:.4g}<br>%{customdata}<br>log step "
            + str(int(li))
            + "<extra></extra>"
            if li is not None
            else "x=%{x:.4g}<br>%{customdata}<extra></extra>"
        )
        fig.add_trace(
            go.Heatmap(
                x=x_sp,
                y=list(range(len(row_labels))),
                z=Z,
                colorscale=hm_cs,
                colorbar=cb_kw,
                hovertemplate=ht_1d,
                customdata=np.broadcast_to(
                    np.asarray(row_labels, dtype=object)[:, np.newaxis],
                    (len(row_labels), len(x_sp)),
                ),
            ),
            row=row,
            col=col,
        )
        fig.update_yaxes(
            showticklabels=False,
            row=row,
            col=col,
            showline=True,
            linewidth=1.1,
            mirror=True,
            linecolor="black",
            tickcolor="black",
            automargin=True,
        )
        fig.update_xaxes(
            title_text=f"Position {pos_ax}",
            row=row,
            col=col,
            showline=True,
            linewidth=1.1,
            mirror=True,
            linecolor="black",
            tickcolor="black",
            automargin=True,
        )
        return

    if kind == "2d":
        Zs = np.asarray(spatial["Z"], dtype=float)
        x_sp = np.asarray(spatial["x"], dtype=float)
        y_sp = np.asarray(spatial["y"], dtype=float)
        z0 = _z_display(Zs[0])
        nk = int(Zs.shape[0])
        hl = row_labels[0] if row_labels else ""
        fig.add_trace(
            go.Heatmap(
                x=x_sp,
                y=y_sp,
                z=z0,
                colorscale=hm_cs,
                colorbar=cb_kw,
                hovertemplate="x=%{x:.4g}<br>y=%{y:.4g}<br>display=%{z:.4g}<extra></extra>",
                name=truncate_display_label(hl, 40) + (f" (+{nk - 1} more)" if nk > 1 else ""),
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="x", row=row, col=col, showline=True, linewidth=1.1, mirror=True, linecolor="black", tickcolor="black", automargin=True)
        fig.update_yaxes(title_text="y", row=row, col=col, showline=True, linewidth=1.1, mirror=True, linecolor="black", tickcolor="black", automargin=True)
        return

    if kind == "3d":
        V = np.asarray(spatial["V"], dtype=float)
        x_sp = np.asarray(spatial["x"], dtype=float)
        y_sp = np.asarray(spatial["y"], dtype=float)
        z_sp = np.asarray(spatial["z"], dtype=float)
        vol = np.asarray(_z_display(V[0]), dtype=float)
        nk = int(V.shape[0])
        nz = vol.shape[2]
        kz = max(0, nz // 2)
        sl = vol[:, :, kz].T
        zk = float(z_sp[kz])
        fig.add_trace(
            go.Heatmap(
                x=x_sp,
                y=y_sp,
                z=sl,
                colorscale=hm_cs,
                colorbar=cb_kw,
                hovertemplate="x=%{x:.4g}<br>y=%{y:.4g}<br>display=%{z:.4g}<extra></extra>",
                name=f"z-slice z={zk:.4g}" + (f" (+{nk - 1} keys)" if nk > 1 else ""),
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="x (z-slice)", row=row, col=col, showline=True, linewidth=1.1, mirror=True, linecolor="black", tickcolor="black", automargin=True)
        fig.update_yaxes(title_text="y (z-slice)", row=row, col=col, showline=True, linewidth=1.1, mirror=True, linecolor="black", tickcolor="black", automargin=True)
        return

    fig.add_trace(
        go.Scatter(
            x=[mid],
            y=[0.0],
            mode="text",
            text=["Unsupported spatial kind"],
            textposition="middle center",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )
    fig.update_xaxes(visible=False, row=row, col=col)
    fig.update_yaxes(visible=False, row=row, col=col)


def _build_plotly_monitor_figure_single(
    bundle: Dict[str, Any],
    *,
    figure_title: Optional[str] = None,
    step_label: str = "Step",
    r_norm_scale: str = "log",
    spatial_heatmap_colorscale: Optional[str] = None,
    theme: str = "light",
    baseline_score: Optional[float] = None,
    export_buttons: bool = True,
) -> Any:
    """Build a decision-oriented Plotly physics admissibility report."""
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if r_norm_scale not in ("log", "linear"):
        raise ValueError("r_norm_scale must be 'log' or 'linear'")
    if theme not in ("dark", "light"):
        raise ValueError("theme must be 'dark' or 'light'")

    use_log_rnorm = r_norm_scale == "log"
    rnorm_y_title = "log10(R_norm + ε)" if use_log_rnorm else "Normalized residual (R norm)"
    hm_cs = spatial_heatmap_colorscale or "Viridis"

    n = int(bundle["n"])
    indices = list(range(n))
    mode_eff = bundle["mode"]
    bar_keys: List[str] = bundle.get("bar_keys") or []
    bar_display: List[str] = bundle.get("bar_display") or []
    _bar_values_raw = bundle.get("bar_values")
    bar_values = np.asarray([] if _bar_values_raw is None else _bar_values_raw, dtype=float)
    _overall_raw = bundle.get("overall_adm")
    overall_adm = np.asarray([] if _overall_raw is None else _overall_raw, dtype=float)
    metrics = list(bundle.get("metrics") or [])
    category_training: Dict[str, Dict[str, Any]] = bundle.get("category_training") or {}
    spatial = bundle.get("spatial")
    spatial_rnorm = bundle.get("spatial_rnorm")
    log_entries = list(bundle.get("log") or [])
    first_rms_map = (log_entries[0].get("rms") if log_entries else {}) or {}
    last_scale_map = (log_entries[-1].get("scale") if log_entries else {}) or {}

    def _scale_for_key(flat_key: str) -> float:
        sv = last_scale_map.get(flat_key)
        if sv is None:
            sv = first_rms_map.get(flat_key)
        try:
            fv = float(sv)
        except (TypeError, ValueError):
            fv = 1.0
        if not math.isfinite(fv) or fv <= 0.0:
            fv = 1.0
        return fv

    dark = theme == "dark"
    paper_bg = "#0b1220" if dark else "#ffffff"
    plot_bg = "#111827" if dark else "#f7f8fa"
    font_color = "#e5e7eb" if dark else "#111827"
    muted = "#94a3b8" if dark else "#6b7280"
    warn_color = "#F59E0B"

    specs = [
        [{"type": "xy", "colspan": 8}, None, None, None, None, None, None, None],
        [{"type": "domain", "colspan": 2}, None, {"type": "domain", "colspan": 2}, None, {"type": "domain", "colspan": 2}, None, {"type": "domain", "colspan": 2}, None],
        [{"type": "xy", "colspan": 4}, None, None, None, {"type": "xy", "colspan": 4}, None, None, None],
        [{"type": "xy", "colspan": 4}, None, None, None, {"type": "xy", "colspan": 4}, None, None, None],
        [{"type": "xy", "colspan": 4}, None, None, None, {"type": "xy", "colspan": 4}, None, None, None],
    ]
    fig = make_subplots(
        rows=5,
        cols=8,
        specs=specs,
        row_heights=[0.12, 0.16, 0.22, 0.24, 0.26],
        vertical_spacing=0.07,
        horizontal_spacing=0.08,
        subplot_titles=(
            "",
            "Overall Admissibility",
            "Governing Score",
            "Constitutive Score",
            "Scaling Score",
            "Admissibility Trend",
            "Category Breakdown",
            "Governing Residuals",
            "Constitutive Residuals",
            "Governing Residual Field",
            "Constitutive Residual Field",
        ),
    )

    # Header band
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)
    last_ov = float(overall_adm[-1]) if len(overall_adm) else float("nan")
    first_ov = float(overall_adm[0]) if len(overall_adm) else float("nan")
    status = _admissibility_status_hml_plotly(last_ov)
    status_color = ADMISSIBLE_COLOR if status == "HIGH" else (warn_color if status == "MODERATE" else DISSONANCE_COLOR)
    trend_arrow = "↑" if (math.isfinite(last_ov) and math.isfinite(first_ov) and last_ov >= first_ov) else "↓"
    final_idx = indices[-1] if indices else 0
    fig.add_annotation(
        x=0.01,
        y=1.0,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        text=(
            f"<span style='font-size:28px;font-weight:700'>{format_admissibility_pct(last_ov) if math.isfinite(last_ov) else 'N/A'}</span>"
            f" <span style='font-size:14px;color:{status_color};font-weight:700'>[{status}]</span>"
            f"<br><span style='font-size:12px;color:{muted}'>Final Step: {final_idx} | Trend: {trend_arrow}</span>"
        ),
        font=dict(family="Inter, Arial, sans-serif", color=font_color),
    )

    # KPI cards as indicators
    last_cat = (metrics[-1].get("category_admissibility_score") if metrics else {}) or {}
    first_cat = (metrics[0].get("category_admissibility_score") if metrics else {}) or {}

    def _kpi_indicator(value: float, title: str, ref: Optional[float]) -> Any:
        v = float(value) if math.isfinite(float(value)) else 0.0
        card_color = ADMISSIBLE_COLOR if v >= 0.9 else (warn_color if v >= 0.7 else DISSONANCE_COLOR)
        ind_kwargs: Dict[str, Any] = dict(
            mode=("number+delta" if ref is not None and math.isfinite(float(ref)) else "number"),
            value=v,
            number={"valueformat": ".1%", "font": {"size": 30, "color": card_color}},
            title={"text": title, "font": {"size": 13, "color": muted}},
        )
        ind = go.Indicator(**ind_kwargs)
        if ref is not None and math.isfinite(float(ref)):
            ind.delta = {"reference": float(ref), "increasing": {"color": ADMISSIBLE_COLOR}, "decreasing": {"color": DISSONANCE_COLOR}, "valueformat": ".1%"}
        return ind

    overall_ref = float(baseline_score) if baseline_score is not None else (first_ov if math.isfinite(first_ov) else None)
    laws_last = float(last_cat.get("laws", float("nan")))
    laws_ref = float(first_cat.get("laws", float("nan"))) if "laws" in first_cat else None
    const_last = float(last_cat.get("constitutive", float("nan")))
    const_ref = float(first_cat.get("constitutive", float("nan"))) if "constitutive" in first_cat else None
    scaling_last = float(last_cat.get("data", float("nan")))
    scaling_ref = float(first_cat.get("data", float("nan"))) if "data" in first_cat else None

    # Bind indicators to row-2 domain subplot cells (cols 1,3,5,7) so Plotly does not assign cartesian xaxis/yaxis.
    fig.add_trace(_kpi_indicator(last_ov if math.isfinite(last_ov) else 0.0, "Overall Admissibility", overall_ref), row=2, col=1)
    fig.add_trace(_kpi_indicator(laws_last if math.isfinite(laws_last) else 0.0, "Governing Score", laws_ref), row=2, col=3)
    fig.add_trace(_kpi_indicator(const_last if math.isfinite(const_last) else 0.0, "Constitutive Score", const_ref), row=2, col=5)
    fig.add_trace(_kpi_indicator(scaling_last if math.isfinite(scaling_last) else 0.0, "Scaling Score", scaling_ref), row=2, col=7)

    # Trend analysis (test mode: no vs-step plot)
    if mode_eff == "test":
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[last_ov if math.isfinite(last_ov) else 0.0],
                mode="text",
                text=[
                    (
                        f"Final admissibility: {format_admissibility_pct(last_ov)}<br>"
                        f"Status: {status}"
                    )
                    if math.isfinite(last_ov)
                    else "Final admissibility unavailable"
                ],
                showlegend=False,
                hoverinfo="skip",
            ),
            row=3,
            col=1,
        )
        fig.update_xaxes(visible=False, row=3, col=1)
        fig.update_yaxes(visible=False, row=3, col=1)
    elif any(np.isfinite(overall_adm)):
        adm_hover = [format_admissibility_pct(float(y)) if np.isfinite(y) else "N/A" for y in overall_adm]
        fig.add_trace(
            go.Scatter(
                x=indices,
                y=overall_adm,
                mode="lines",
                name="Overall admissibility",
                    line=dict(color="black", width=2.8),
                text=adm_hover,
                hovertemplate="Overall<br>%{x}<br>%{text}<extra></extra>",
                showlegend=False,
            ),
            row=3,
            col=1,
        )
        if len(overall_adm) >= 4:
            kernel = np.ones(3, dtype=float) / 3.0
            smooth = np.convolve(overall_adm, kernel, mode="same")
            fig.add_trace(
                go.Scatter(
                    x=indices,
                    y=smooth,
                    mode="lines",
                    name="Smoothed",
                    line=dict(color="#93C5FD", width=2, dash="dot"),
                    hovertemplate="Smoothed<br>%{x}<br>%{y:.2%}<extra></extra>",
                    showlegend=False,
                ),
                row=3,
                col=1,
            )
        if math.isfinite(last_ov):
            fig.add_trace(
                go.Scatter(
                    x=[final_idx],
                    y=[last_ov],
                    mode="markers",
                    marker=dict(size=10, color=DISSONANCE_COLOR, line=dict(width=1.5, color="white")),
                    hovertemplate="Final<br>%{y:.2%}<extra></extra>",
                    showlegend=False,
                ),
                row=3,
                col=1,
            )
        finite = overall_adm[np.isfinite(overall_adm)]
        if finite.size:
            ymin = max(0.0, float(np.min(finite)) - 0.05)
            ymax = min(1.0, float(np.max(finite)) + 0.05)
            if ymax - ymin < 0.08:
                mid = 0.5 * (ymin + ymax)
                ymin = max(0.0, mid - 0.04)
                ymax = min(1.0, mid + 0.04)
            fig.update_yaxes(range=[ymin, ymax], row=3, col=1)
        if len(overall_adm) >= 5:
            delta = np.diff(overall_adm.astype(float))
            if np.nanstd(delta) > 0.03:
                # Do not use add_hrect(row=..., col=...): Plotly scans all fig.data for axis
                # refs and reads xaxis/yaxis on every trace; Indicator traces have no xaxis.
                trend_ref = next(
                    (
                        tr
                        for tr in fig.data
                        if getattr(tr, "type", None) == "scatter"
                        and getattr(tr, "name", "") == "Overall admissibility"
                    ),
                    None,
                )
                if trend_ref is not None:
                    xa = getattr(trend_ref, "xaxis", None) or "x"
                    ya = getattr(trend_ref, "yaxis", None) or "y"
                    fig.add_shape(
                        type="rect",
                        x0=0,
                        x1=1,
                        xref=f"{xa} domain",
                        y0=max(0.0, float(np.nanmin(overall_adm))),
                        y1=min(1.0, float(np.nanmax(overall_adm))),
                        yref=ya,
                        fillcolor=DISSONANCE_COLOR,
                        opacity=0.07,
                        line_width=0,
                        layer="below",
                    )

    if mode_eff != "test":
        fig.update_xaxes(
            title_text=step_label,
            row=3,
            col=1,
            showline=True,
            linewidth=1.2,
            mirror=True,
            linecolor="black",
            tickcolor="black",
            automargin=True,
        )
        fig.update_yaxes(
            title_text="Admissibility (%)",
            tickformat=".2f%",
            row=3,
            col=1,
            showline=True,
            linewidth=1.2,
            mirror=True,
            linecolor="black",
            tickcolor="black",
            automargin=True,
        )

    # Category breakdown (worst -> best)
    order_keys = []
    order_vals = []
    for k, v in last_cat.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        order_keys.append(k)
        order_vals.append(fv)
    if not order_keys:
        order_keys = ["laws", "constitutive"]
        order_vals = [laws_last, const_last]
    labels_vals = sorted(zip(order_keys, order_vals), key=lambda kv: kv[1] if math.isfinite(kv[1]) else 2.0)
    cat_labels_raw = [pretty_category_name(k) for k, _ in labels_vals]
    cat_labels = ["Constitutive<br>Relations" if v == "Constitutive Relations" else v for v in cat_labels_raw]
    cat_vals = [float(v) if math.isfinite(v) else 0.0 for _, v in labels_vals]
    cat_text = [format_admissibility_pct(v) if math.isfinite(v) else "N/A" for _, v in labels_vals]
    cat_colors = [ADMISSIBLE_COLOR if v >= 0.9 else (warn_color if v >= 0.7 else DISSONANCE_COLOR) for v in cat_vals]
    fig.add_trace(
        go.Bar(
            x=cat_vals,
            y=cat_labels,
            orientation="h",
            marker=dict(color=cat_colors, line=dict(color="#374151", width=1)),
            text=cat_text,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>admissibility=%{text}<extra></extra>",
            showlegend=False,
        ),
        row=3,
        col=5,
    )
    cat_axis = getattr(fig.data[-1], "xaxis", "x2")
    cat_yaxis = getattr(fig.data[-1], "yaxis", "y2")
    fig.add_shape(
        type="line",
        x0=0.9,
        x1=0.9,
        y0=0,
        y1=1,
        xref=cat_axis,
        yref=f"{cat_yaxis} domain",
        line=dict(color=warn_color, dash="dash"),
    )
    worst_label = cat_labels_raw[0] if cat_labels_raw else "N/A"
    fig.add_annotation(
        x=0.02,
        y=1.04,
        xref=f"{cat_axis} domain",
        yref=f"{cat_yaxis} domain",
        text=f"Primary Issue: {worst_label}",
        showarrow=False,
        align="left",
        font=dict(size=11, color=DISSONANCE_COLOR, family="Inter, Arial, sans-serif"),
    )
    fig.update_xaxes(
        title_text="Admissibility (%)",
        range=list(category_adm_bar_axis_range_percent_full()),
        tickformat=".2f%",
        row=3,
        col=5,
        showline=True,
        linewidth=1.2,
        linecolor="black",
        tickcolor="black",
        automargin=True,
    )
    fig.update_yaxes(row=3, col=5, showline=True, linewidth=1.2, linecolor="black", tickcolor="black", automargin=True, tickfont=dict(size=10))

    # Residual diagnostics
    def _plot_residual_panel(cat: str, row: int, col: int, title_prefix: str) -> None:
        info = category_training.get(cat, {"keys": [], "displays": [], "r_norm_mat": np.zeros((0, n))})
        ckeys: List[str] = list(info.get("keys") or [])
        displays: List[str] = list(info.get("displays") or [])
        mat = np.asarray(info.get("r_norm_mat") if info.get("r_norm_mat") is not None else np.zeros((0, n)), dtype=float)
        if mode_eff != "test" and mat.size and len(ckeys):
            terminal = mat[:, -1]
            worst_i = int(np.nanargmax(terminal)) if np.any(np.isfinite(terminal)) else 0
            unstable = np.nanstd(np.diff(mat, axis=1), axis=1) if mat.shape[1] > 1 else np.zeros(mat.shape[0])
            unstable_i = int(np.nanargmax(unstable)) if unstable.size else -1
            for i, key in enumerate(ckeys):
                ys = mat[i, :]
                if not np.all(np.isfinite(ys)):
                    continue
                y_plot = np.log10(np.maximum(ys, 0.0) + R_NORM_LOG_EPS) if use_log_rnorm else ys
                colr = RESIDUAL_COLOR_LAWS if cat == "laws" else RESIDUAL_COLOR_CONSTITUTIVE
                width = 3.4 if i == worst_i else 2.0
                dash = "dash" if i == unstable_i and i != worst_i else "solid"
                scale_k = _scale_for_key(key)
                customdata = np.full(len(indices), scale_k)
                name = displays[i] if i < len(displays) else key
                fig.add_trace(
                    go.Scatter(
                        x=indices,
                        y=y_plot,
                        mode="lines",
                        name=name,
                        line=dict(color=colr, width=width, dash=dash),
                        showlegend=False,
                        customdata=customdata,
                        hovertemplate=f"{name}<br>{step_label}=%{{x}}<br>{'log10(R_norm+ε)' if use_log_rnorm else 'R_norm'}=%{{y:.4g}}<br>scale_k=%{{customdata:.4g}}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )
            worst_name = displays[worst_i] if worst_i < len(displays) else (ckeys[worst_i] if ckeys else "N/A")
            if len(ckeys) > 1:
                fig.add_annotation(
                    x=0.01,
                    y=1.02,
                    xref=("x3 domain" if col == 1 else "x4 domain"),
                    yref=("y3 domain" if col == 1 else "y4 domain"),
                    text=f"Worst violation: {truncate_display_label(worst_name, 44)}",
                    showarrow=False,
                    align="left",
                    font=dict(size=10, color=DISSONANCE_COLOR, family="Inter, Arial, sans-serif"),
                    row=row,
                    col=col,
                )
        else:
            fk = [i for i, k in enumerate(bar_keys) if str(k).startswith(f"{cat}/")]
            if fk:
                xs = [truncate_display_label(bar_display[i], 40) for i in fk]
                vals = [float(bar_values[i]) if i < len(bar_values) and np.isfinite(bar_values[i]) else 0.0 for i in fk]
                ys = [float(np.log10(max(v, 0.0) + R_NORM_LOG_EPS)) if use_log_rnorm else v for v in vals]
                cds = [_scale_for_key(bar_keys[i]) for i in fk]
                fig.add_trace(
                    go.Bar(
                        x=xs,
                        y=ys,
                        marker_color=(RESIDUAL_COLOR_LAWS if cat == "laws" else RESIDUAL_COLOR_CONSTITUTIVE),
                        customdata=cds,
                        showlegend=False,
                        hovertemplate="%{x}<br>" + (("log10(R_norm+ε)=%{y:.4g}") if use_log_rnorm else ("R_norm=%{y:.4g}")) + "<br>scale_k=%{customdata:.4g}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )
            else:
                fig.add_trace(
                    go.Scatter(x=[indices[len(indices)//2] if indices else 0], y=[0.0], mode="text", text=["No residual keys in this category"], showlegend=False, hoverinfo="skip"),
                    row=row,
                    col=col,
                )
        fig.update_xaxes(
            title_text=("Residual key" if mode_eff == "test" else (step_label if mat.size else "Residual key")),
            row=row,
            col=col,
            showline=True,
            linewidth=1.2,
            mirror=True,
            linecolor="black",
            tickcolor="black",
            automargin=True,
        )
        fig.update_yaxes(
            title_text=rnorm_y_title if col == 1 else "",
            row=row,
            col=col,
            showline=True,
            linewidth=1.2,
            mirror=True,
            linecolor="black",
            tickcolor="black",
            automargin=True,
        )

    _plot_residual_panel("laws", 4, 1, "Governing")
    _plot_residual_panel("constitutive", 4, 5, "Constitutive")

    # Spatial maps
    mid = indices[len(indices) // 2] if indices else 0
    if spatial is not None:
        _plotly_add_spatial_panel_to_subplot(fig, row=5, col=1, spatial=spatial, hm_cs=hm_cs, mid=mid, colorbar_compact=False, use_log_rnorm=use_log_rnorm, colorbar_scale_title="log10 residual")
    else:
        fig.add_trace(go.Scatter(x=[mid], y=[0.0], mode="text", text=["No governing spatial field"], showlegend=False, hoverinfo="skip"), row=5, col=1)
        fig.update_xaxes(visible=False, row=5, col=1)
        fig.update_yaxes(visible=False, row=5, col=1)

    if spatial_rnorm is not None:
        _plotly_add_spatial_panel_to_subplot(fig, row=5, col=5, spatial=spatial_rnorm, hm_cs=hm_cs, mid=mid, colorbar_compact=False, use_log_rnorm=use_log_rnorm, colorbar_scale_title="log10 residual")
    else:
        fig.add_trace(go.Scatter(x=[mid], y=[0.0], mode="text", text=["No constitutive spatial field"], showlegend=False, hoverinfo="skip"), row=5, col=5)
        fig.update_xaxes(visible=False, row=5, col=5)
        fig.update_yaxes(visible=False, row=5, col=5)

    # Actionable summary
    summary_lines = []
    if math.isfinite(laws_last):
        summary_lines.append("Governing laws satisfied" if laws_last >= 0.9 else "Governing-law violations detected")
    if math.isfinite(const_last):
        summary_lines.append("Constitutive consistency acceptable" if const_last >= 0.9 else "Constitutive inconsistency detected")
    if math.isfinite(last_ov) and math.isfinite(first_ov):
        summary_lines.append("Training trend improving" if last_ov >= first_ov else "Training trend degrading")
    if status != "HIGH":
        summary_lines.append("Recommend: adjust optimizer, residual weighting, or data scaling")
    summary_text = "Summary:<br>- " + "<br>- ".join(summary_lines[:4]) if summary_lines else "Summary: insufficient diagnostics"

    # Intentionally remove the report-level layout title text (enterprise dashboard feel).
    title_text = ""
    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor="center", pad=dict(t=14, b=8), font=dict(size=21, family="Inter, Arial, sans-serif")),
        height=1680,
        showlegend=False,
        margin=dict(l=90, r=90, t=130, b=165),
        hovermode="closest",
        template=("plotly_dark" if dark else "plotly_white"),
        plot_bgcolor=plot_bg,
        paper_bgcolor=paper_bg,
        font=dict(size=12, family="Inter, Arial, sans-serif", color=font_color),
    )
    if export_buttons:
        fig.update_layout(modebar_add=["toImage"])
    fig.add_annotation(text="Ifimo Lab: Moju Forensic Suite", x=0.995, y=1.08, xref="paper", yref="paper", showarrow=False, xanchor="right", font=dict(size=10, color=muted, family="Inter, Arial, sans-serif"))
    fig.add_annotation(text=summary_text, x=0.01, y=-0.12, xref="paper", yref="paper", showarrow=False, align="left", font=dict(size=12, color=font_color, family="Inter, Arial, sans-serif"), bordercolor="#334155", borderwidth=1, borderpad=8, bgcolor=("rgba(30,41,59,0.35)" if dark else "rgba(241,245,249,0.85)"))

    align_heatmap_colorbars_to_subplot_domains(fig)
    return fig


def _pos_axis_title_card(sp: Optional[Dict[str, Any]]) -> str:
    if not sp:
        return "Position x"
    ax = sp.get("position_axis") or "x"
    return f"Position {ax}"



def _build_kpi_figure(
    overall_adm: List[float],
    *,
    baseline_score: Optional[float],
    theme: str,
) -> Any:
    import plotly.graph_objects as go
    value = float(overall_adm[-1]) if overall_adm else float("nan")
    dark = theme == "dark"
    paper = "#0f172a" if dark else "#ffffff"
    fontc = "#e5e7eb" if dark else "#111827"
    fig = go.Figure()
    ind_kwargs = dict(mode="gauge+number", value=max(0.0, min(1.0, value if math.isfinite(value) else 0.0)))
    if baseline_score is not None and math.isfinite(float(baseline_score)):
        ind_kwargs["mode"] = "gauge+number+delta"
        ind_kwargs["delta"] = {"reference": float(baseline_score), "increasing": {"color": ADMISSIBLE_COLOR}, "decreasing": {"color": DISSONANCE_COLOR}}
    ind_kwargs["gauge"] = {
        "axis": {"range": [0, 1], "tickformat": ".0%"},
        "bar": {"color": ADMISSIBLE_COLOR if (math.isfinite(value) and value >= 0.9) else DISSONANCE_COLOR},
        "steps": [{"range": [0, 0.9], "color": "#1f2937" if dark else "#fee2e2"}, {"range": [0.9, 1.0], "color": "#064e3b" if dark else "#d1fae5"}],
    }
    fig.add_trace(go.Indicator(**ind_kwargs, title={"text": "Overall Admissibility A"}))
    fig.add_annotation(text="A = 1 / (1 + R_norm)", x=0.5, y=-0.08, xref="paper", yref="paper", showarrow=False, font=dict(size=11, family="Inter, Arial, sans-serif", color=fontc))
    fig.update_layout(template="plotly_dark" if dark else "plotly_white", paper_bgcolor=paper, plot_bgcolor=paper, font=dict(family="Inter, Arial, sans-serif", color=fontc), margin=dict(l=40, r=40, t=80, b=50), height=360)
    return fig


def _build_forensic_heatmap_figure(bundle: Dict[str, Any], *, theme: str = "light") -> Any:
    import plotly.graph_objects as go
    import numpy as np

    log_entries = list(bundle.get("log") or [])
    n = len(log_entries)
    if n == 0:
        return go.Figure()
    # Use coord snapshot x if available; fallback to index-like x.
    snap = (log_entries[-1].get("coord_snapshot") if log_entries else {}) or {}
    x = np.asarray(snap.get("x") or list(range(max(2, n))), dtype=float)
    if x.size == 0:
        x = np.asarray(list(range(max(2, n))), dtype=float)

    # pick first law or constitutive key for forensic trace
    keys = []
    for ent in log_entries:
        keys.extend(list((ent.get("rms") or {}).keys()))
    target = next((k for k in keys if str(k).startswith("laws/")), None) or next((k for k in keys if str(k).startswith("constitutive/")), None) or keys[0]

    z_rows = []
    scales = []
    for ent in log_entries:
        rms = ent.get("rms") or {}
        scale = (ent.get("scale") or {}).get(target, 1.0)
        try:
            rv = float(rms.get(target, float("nan")))
        except Exception:
            rv = float("nan")
        try:
            sv = float(scale)
        except Exception:
            sv = 1.0
        if not math.isfinite(sv) or sv <= 0:
            sv = 1.0
        scales.append(sv)
        display = np.log10(abs(rv) + R_NORM_LOG_EPS) if math.isfinite(rv) else float("nan")
        z_rows.append(np.full((x.size,), display, dtype=float))
    Z = np.asarray(z_rows, dtype=float)

    fig = go.Figure(
        data=[
            go.Heatmap(
                x=x,
                y=list(range(n)),
                z=Z,
                colorscale="Viridis",
                colorbar=dict(title="log10(|residual| + ε)"),
                customdata=np.broadcast_to(np.asarray(scales, dtype=float)[:, None], Z.shape),
                hovertemplate="step=%{y}<br>x=%{x:.4g}<br>log10(|r|+ε)=%{z:.4g}<br>scale_k=%{customdata:.4g}<extra></extra>",
            )
        ]
    )
    dark = theme == "dark"
    fig.update_layout(
        title=f"Forensic Spatial Dissonance — {truncate_display_label(str(target), 64)}",
        xaxis_title="Spatial position x",
        yaxis_title="Logged Step / Epoch",
        template="plotly_dark" if dark else "plotly_white",
        font=dict(family="Inter, Arial, sans-serif"),
    )
    return fig


def build_plotly_monitor_dash_payload(
    bundle: Dict[str, Any],
    *,
    figure_title: Optional[str] = None,
    step_label: str = "Step",
    r_norm_scale: str = "log",
    spatial_heatmap_colorscale: Optional[str] = None,
    theme: str = "light",
    baseline_score: Optional[float] = None,
    export_buttons: bool = True,
) -> Dict[str, Any]:
    full = _build_plotly_monitor_figure_single(
        bundle,
        figure_title=figure_title,
        step_label=step_label,
        r_norm_scale=r_norm_scale,
        spatial_heatmap_colorscale=spatial_heatmap_colorscale,
        theme=theme,
        baseline_score=baseline_score,
        export_buttons=export_buttons,
    )
    kpi = _build_kpi_figure(bundle.get("overall_adm") or [], baseline_score=baseline_score, theme=theme)
    forensic = _build_forensic_heatmap_figure(bundle, theme=theme)
    return {
        "mode": "dash-tabs",
        "tabs": {
            "kpi": kpi,
            "admissibility": full,
            "forensic_heatmaps": forensic,
            "convergence": full,
        },
        "filter_contract": {"bar_customdata_field": "category", "threshold": 0.99},
        "toggles": {"mode": ["training", "test"]},
        "export": {"html": True, "pdf": True, "export_buttons": bool(export_buttons)},
    }


def build_plotly_monitor_figure(
    bundle: Dict[str, Any],
    *,
    figure_title: Optional[str] = None,
    step_label: str = "Step",
    r_norm_scale: str = "log",
    spatial_heatmap_colorscale: Optional[str] = None,
    dashboard_mode: str = "single-figure",
    theme: str = "light",
    baseline_score: Optional[float] = None,
    export_buttons: bool = True,
) -> Any:
    if dashboard_mode == "dash-tabs":
        return build_plotly_monitor_dash_payload(
            bundle,
            figure_title=figure_title,
            step_label=step_label,
            r_norm_scale=r_norm_scale,
            spatial_heatmap_colorscale=spatial_heatmap_colorscale,
            theme=theme,
            baseline_score=baseline_score,
            export_buttons=export_buttons,
        )
    return _build_plotly_monitor_figure_single(
        bundle,
        figure_title=figure_title,
        step_label=step_label,
        r_norm_scale=r_norm_scale,
        spatial_heatmap_colorscale=spatial_heatmap_colorscale,
        theme=theme,
        baseline_score=baseline_score,
        export_buttons=export_buttons,
    )

def build_plotly_law_rnorm_final_bar_figure(
    bundle: Dict[str, Any],
    *,
    card_height: int = MOJU_STUDIO_DASHBOARD_CARD_HEIGHT,
    r_norm_scale: str = "log",
) -> Any:
    """Vertical bar chart: log10(R_norm+ε) or raw R_norm at the final log step per law key (Studio card)."""
    import numpy as np
    import plotly.graph_objects as go

    metrics = bundle["metrics"]
    n = int(bundle["n"])
    indices = bundle["indices"]
    category_training = bundle.get("category_training") or {}
    info = category_training.get("laws", {"keys": [], "displays": [], "r_norm_mat": np.zeros((0, n))})
    lk = info["keys"]
    ld = info["displays"]
    mat_lb = np.asarray(info["r_norm_mat"], dtype=float)
    mid = int(indices[len(indices) // 2]) if len(indices) else 0
    use_log = r_norm_scale == "log"

    fig = go.Figure()
    if not lk or mat_lb.size == 0 or n < 1:
        fig.add_trace(
            go.Scatter(
                x=[mid],
                y=[0.0],
                mode="text",
                text=["No law keys in this run"],
                textposition="middle center",
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
    else:
        x_lbl = [truncate_display_label(d, 44) for d in ld]
        r_last = [float(mat_lb[i, -1]) if np.isfinite(mat_lb[i, -1]) else float("nan") for i in range(len(lk))]
        bar_colors = [_residual_color_from_key(k) for k in lk]
        y_bar = [
            (
                float(np.log10(max(rv, 0.0) + R_NORM_LOG_EPS))
                if np.isfinite(rv) and use_log
                else (float(rv) if np.isfinite(rv) else 0.0)
            )
            for rv in r_last
        ]
        fig.add_trace(
            go.Bar(
                x=x_lbl,
                y=y_bar,
                marker_color=bar_colors,
                showlegend=False,
                hovertemplate="%{x}<br>"
                + ("log10(R_norm+ε)=%{y:.4g}<extra></extra>" if use_log else "R norm=%{y:.4g}<extra></extra>"),
            )
        )
        fig.update_xaxes(
            title_text="Governing law (residual key)",
            showline=True,
            linewidth=1.2,
            automargin=True,
        )
        y_ax = "log10(R_norm + ε)" if use_log else "Normalized residual (R norm)"
        fig.update_yaxes(title_text=y_ax, showline=True, linewidth=1.2, automargin=True)

    fig.update_layout(
        title=dict(text="Law R_norm (final step)", font=dict(size=14, family="Arial, sans-serif")),
        height=card_height,
        margin=dict(l=12, r=12, t=48, b=48),
        template="plotly_white",
        font=dict(size=11, family="Arial, sans-serif"),
    )
    return fig


def build_plotly_category_admissibility_bar_figure(
    bundle: Dict[str, Any],
    *,
    card_height: int = MOJU_STUDIO_DASHBOARD_CARD_HEIGHT,
) -> Any:
    """Horizontal bar chart: laws / constitutive admissibility at final step (Studio card)."""
    import plotly.graph_objects as go

    metrics = bundle["metrics"]
    blabels, bvals = _three_pillar_labels_values(metrics)
    bcolors = [_adm_bar_color_plotly(v) for v in bvals]
    bx = [v if math.isfinite(v) else 0.0 for v in bvals]
    btext = [format_admissibility_pct(v) if math.isfinite(v) else "N/A" for v in bvals]
    adm_ht = [format_admissibility_pct(v) if math.isfinite(v) else "N/A" for v in bx]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=bx,
            y=blabels,
            orientation="h",
            marker=dict(color=bcolors, line=dict(color="#333333", width=1)),
            text=btext,
            textposition="outside",
            cliponaxis=False,
            showlegend=False,
            customdata=adm_ht,
            hovertemplate="%{y}<br>admissibility=%{customdata}<extra></extra>",
        )
    )
    _adm_full = category_adm_bar_axis_range_percent_full()
    fig.update_xaxes(
        title_text="Admissibility (%)",
        range=list(_adm_full),
        tickformat=".2f%",
        showline=True,
        linewidth=1.2,
        automargin=True,
    )
    fig.update_yaxes(showline=True, linewidth=1.2, automargin=True)
    fig.update_layout(
        title=dict(text="Category admissibility (final step)", font=dict(size=14, family="Arial, sans-serif")),
        height=card_height,
        margin=dict(l=12, r=80, t=48, b=48),
        template="plotly_white",
        font=dict(size=11, family="Arial, sans-serif"),
    )
    return fig


def build_plotly_spatial_rnorm_heatmap_card(
    spatial_parsed: Optional[Dict[str, Any]],
    *,
    colorscale: str = "Jet",
    card_title: str = "Spatial |residual|",
    colorbar_scale_title: Optional[str] = None,
    card_height: int = MOJU_STUDIO_DASHBOARD_CARD_HEIGHT,
) -> Any:
    """
    Single heatmap + slim colorbar for one spatial panel (law or constitutive).

    ``spatial_parsed`` is the output of ``_parse_spatial_law_panel`` or
    ``_parse_spatial_rnorm_panel`` (``kind`` ``1d`` / ``2d`` / ``3d``; see parser docstring).

    The figure title uses human names from ``row_labels`` when present; ``card_title`` is the
    fallback. ``colorbar_scale_title`` defaults to a log-scale residual label (override to match
    the plotted quantity).
    """
    import numpy as np
    import plotly.graph_objects as go

    fig = go.Figure()
    cb_default = colorbar_scale_title or "log10(|residual| + ε)"
    full_title = card_title
    if spatial_parsed is None:
        fig.add_annotation(
            text="No spatial slice (upload state with coords or adjust sidebar axis)",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=12),
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
    else:
        full_title = _spatial_heatmap_subplot_title(spatial_parsed, card_title)
        kind = spatial_parsed.get("kind", "1d")
        cb = dict(
            title=dict(text=cb_default, side="right", font=dict(size=10)),
            len=0.72,
            thickness=10,
            x=1.02,
            xpad=6,
        )
        if kind == "1d":
            Z = spatial_parsed["Z"]
            x_sp = spatial_parsed["x"]
            row_labels = spatial_parsed["row_labels"]
            fig.add_trace(
                go.Heatmap(
                    x=x_sp,
                    y=list(range(len(row_labels))),
                    z=Z,
                    colorscale=colorscale,
                    colorbar=cb,
                    hovertemplate="x=%{x:.4g}<br>%{customdata}<extra></extra>",
                    customdata=np.broadcast_to(
                        np.asarray(row_labels, dtype=object)[:, np.newaxis],
                        (len(row_labels), len(x_sp)),
                    ),
                )
            )
            fig.update_yaxes(
                showticklabels=False,
                automargin=True,
            )
            fig.update_xaxes(
                title_text=_pos_axis_title_card(spatial_parsed),
                showline=True,
                linewidth=1.2,
                automargin=True,
            )
        elif kind == "2d":
            Zs = spatial_parsed["Z"]
            x_sp = np.asarray(spatial_parsed["x"], dtype=float)
            y_sp = np.asarray(spatial_parsed["y"], dtype=float)
            row_labels = spatial_parsed["row_labels"]
            nk = int(Zs.shape[0])
            fig.add_trace(
                go.Heatmap(
                    x=x_sp,
                    y=y_sp,
                    z=Zs[0],
                    colorscale=colorscale,
                    colorbar=cb,
                    hovertemplate="x=%{x:.4g}<br>y=%{y:.4g}<br>display=%{z:.4g}<extra></extra>",
                )
            )
            fig.update_xaxes(title_text="x", automargin=True)
            fig.update_yaxes(title_text="y", automargin=True)
            if nk > 1:
                fig.add_annotation(
                    text=f"First of {nk} keys: {truncate_display_label(row_labels[0], 40)}",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=1.08,
                    showarrow=False,
                    yanchor="bottom",
                    font=dict(size=10),
                )
        else:
            V = np.asarray(spatial_parsed["V"][0], dtype=float)
            x_sp = np.asarray(spatial_parsed["x"], dtype=float)
            y_sp = np.asarray(spatial_parsed["y"], dtype=float)
            z_sp = np.asarray(spatial_parsed["z"], dtype=float)
            nk = int(spatial_parsed["V"].shape[0])
            fig.add_trace(
                go.Volume(
                    x=x_sp,
                    y=y_sp,
                    z=z_sp,
                    value=V,
                    colorscale=colorscale,
                    opacity=0.35,
                    surface_count=18,
                    caps=dict(x_show=False, y_show=False, z_show=False),
                    colorbar=cb,
                )
            )
            fig.update_layout(
                scene=dict(
                    xaxis_title="x",
                    yaxis_title="y",
                    zaxis_title="z",
                    aspectmode="data",
                ),
                margin=dict(l=8, r=8, t=48, b=8),
            )
            if nk > 1:
                fig.add_annotation(
                    text=f"First of {nk} keys: {truncate_display_label(spatial_parsed['row_labels'][0], 40)}",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=1.06,
                    showarrow=False,
                    yanchor="bottom",
                    font=dict(size=10),
                )

    fig.update_layout(
        title=dict(text=full_title, font=dict(size=14, family="Arial, sans-serif")),
        height=card_height,
        margin=dict(l=8, r=100, t=48, b=48),
        template="plotly_white",
        font=dict(size=11, family="Arial, sans-serif"),
    )
    return fig
