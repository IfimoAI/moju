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

RESIDUAL_COLOR_LAWS = "#7e57c2"  # purple
RESIDUAL_COLOR_CONSTITUTIVE = "#009688"  # teal
RESIDUAL_COLOR_OTHER = "#6b7280"  # neutral fallback


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
        x_cb = min(x1 + 0.02, 0.992)
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
        xpad=12,
        thickness=14,
    )
    if not colorbar_compact:
        cb_kw["x"] = 1.02
        cb_kw["xpad"] = 14

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
            automargin=True,
        )
        fig.update_xaxes(
            title_text=f"Position {pos_ax}",
            row=row,
            col=col,
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
        fig.update_xaxes(title_text="x", row=row, col=col, automargin=True)
        fig.update_yaxes(title_text="y", row=row, col=col, automargin=True)
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
        fig.update_xaxes(title_text="x (z-slice)", row=row, col=col, automargin=True)
        fig.update_yaxes(title_text="y (z-slice)", row=row, col=col, automargin=True)
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


def build_plotly_monitor_figure(
    bundle: Dict[str, Any],
    *,
    figure_title: Optional[str] = None,
    step_label: str = "Step",
    r_norm_scale: str = "log",
    spatial_heatmap_colorscale: Optional[str] = None,
) -> Any:
    """
    Build a ``plotly.graph_objects.Figure`` from a bundle produced by
    :func:`moju.monitor.auditor._build_visualize_bundle`.
    """
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if r_norm_scale not in ("log", "linear"):
        raise ValueError("r_norm_scale must be 'log' or 'linear'")
    use_log_rnorm = r_norm_scale == "log"
    rnorm_y_title = "log10(R_norm + ε)" if use_log_rnorm else "Normalized residual (R norm)"
    spatial_norm = bool(bundle.get("spatial_normalize", False))
    spatial_z_title = (
        rnorm_y_title
        if spatial_norm
        else ("log10(|residual| + ε)" if use_log_rnorm else "|residual|")
    )
    law_sp_full = (
        "Governing laws R_norm (spatial, last step)"
        if spatial_norm
        else "Governing laws |residual| (spatial, last step)"
    )
    const_sp_full = (
        "Constitutive R_norm (spatial, last step)"
        if spatial_norm
        else "Constitutive |residual| (spatial, last step)"
    )
    hm_cs = spatial_heatmap_colorscale or "Jet"

    n = bundle["n"]
    indices = list(range(n))
    category_training: Dict[str, Dict[str, Any]] = bundle.get("category_training") or {}
    use_bar_chart = bundle["use_bar_chart"]
    bar_keys: List[str] = bundle.get("bar_keys") or []
    bar_display: List[str] = bundle["bar_display"]
    bar_values = bundle["bar_values"]
    overall_adm = bundle["overall_adm"]
    metrics = bundle["metrics"]
    spatial = bundle["spatial"]
    spatial_rnorm = bundle.get("spatial_rnorm")
    mode_eff = bundle["mode"]

    has_spatial = spatial is not None
    has_spatial_rnorm = spatial_rnorm is not None
    training_lines = mode_eff == "training" and not use_bar_chart
    polar_full = mode_eff == "test" and not (has_spatial or has_spatial_rnorm)

    if training_lines:
        specs_row_overall = [{"type": "xy", "colspan": 6}, None, None, None, None, None]
        specs_row_bars = [
            {"type": "xy", "colspan": 3},
            None,
            None,
            {"type": "xy", "colspan": 3},
            None,
            None,
        ]
        specs_row_lines = [
            {"type": "xy", "colspan": 3},
            None,
            None,
            {"type": "xy", "colspan": 3},
            None,
            None,
        ]
        specs_row_spatial = [
            {"type": "xy", "colspan": 3},
            None,
            None,
            {"type": "xy", "colspan": 3},
            None,
            None,
        ]
        specs: List[List[Any]] = [specs_row_overall, specs_row_bars, specs_row_lines, specs_row_spatial]
        row_heights = [0.19, 0.19, 0.25, 0.30]

        info_laws_tr = category_training.get(
            "laws", {"keys": [], "displays": [], "r_norm_mat": np.zeros((0, n))}
        )
        info_const_tr = category_training.get(
            "constitutive", {"keys": [], "displays": [], "r_norm_mat": np.zeros((0, n))}
        )
        sub_titles = [
            "Overall admissibility",
            "Law R_norm (final step)",
            "Category admissibility (final step)",
            _convergence_subplot_title("laws", info_laws_tr),
            _convergence_subplot_title("constitutive", info_const_tr),
            _spatial_heatmap_subplot_title(spatial if has_spatial else None, law_sp_full),
            _spatial_heatmap_subplot_title(spatial_rnorm if has_spatial_rnorm else None, const_sp_full),
        ]

        fig = make_subplots(
            rows=len(specs),
            cols=6,
            specs=specs,
            vertical_spacing=0.152,
            horizontal_spacing=0.115,
            row_heights=row_heights,
            subplot_titles=tuple(sub_titles),
        )
        fig.update_annotations(font=dict(size=11, family="Arial, sans-serif", color="#1a1a1a"))

        last_ov = float(overall_adm[-1]) if len(overall_adm) else float("nan")
        adm_hover = []
        for y in overall_adm:
            try:
                fy = float(y)
            except (TypeError, ValueError):
                adm_hover.append("N/A")
            else:
                adm_hover.append(format_admissibility_pct(fy) if math.isfinite(fy) else "N/A")
        if any(np.isfinite(overall_adm)):
            fig.add_trace(
                go.Scatter(
                    x=indices,
                    y=overall_adm,
                    mode="lines",
                    name="Overall admissibility",
                    line=dict(color="#2c3e50", width=2.8),
                    showlegend=False,
                    text=adm_hover,
                    hovertemplate="Overall<br>%{x}<br>%{text}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            if math.isfinite(last_ov):
                lix = indices[-1]
                fig.add_trace(
                    go.Scatter(
                        x=[lix],
                        y=[last_ov],
                        mode="markers",
                        marker=dict(size=12, color="#c0392b", line=dict(width=2, color="white")),
                        showlegend=False,
                        text=[format_admissibility_pct(last_ov)],
                        hovertemplate="Final<br>%{text}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_annotation(
                    x=lix,
                    y=last_ov,
                    text=f" {format_admissibility_pct(last_ov)}",
                    showarrow=False,
                    xanchor="left",
                    yanchor="bottom",
                    font=dict(size=11, color="#2c3e50", family="Arial, sans-serif"),
                    row=1,
                    col=1,
                )
        fig.update_xaxes(
            title_text=step_label,
            row=1,
            col=1,
            showline=True,
            linewidth=1.2,
            mirror=True,
            automargin=True,
        )
        fig.update_yaxes(
            title_text="Admissibility (%)",
            tickformat=".2f%",
            autorange=True,
            row=1,
            col=1,
            showline=True,
            linewidth=1.2,
            mirror=True,
            automargin=True,
        )

        if not bar_display or len(bar_values) == 0 or n < 1:
            fig.add_trace(
                go.Scatter(
                    x=[indices[len(indices) // 2] if indices else 0],
                    y=[0.0],
                    mode="text",
                    text=["No residual keys in this run"],
                    textposition="middle center",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=2,
                col=1,
            )
        else:
            x_lbl = [truncate_display_label(d, 44) for d in bar_display]
            r_last = [float(v) if np.isfinite(v) else float("nan") for v in np.asarray(bar_values, dtype=float)]
            bar_colors = [
                _residual_color_from_key(bar_keys[i] if i < len(bar_keys) else "")
                for i in range(len(x_lbl))
            ]
            y_bar = [
                (
                    float(np.log10(max(rv, 0.0) + R_NORM_LOG_EPS))
                    if np.isfinite(rv) and use_log_rnorm
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
                    + ("log10(R_norm+ε)=%{y:.4g}<extra></extra>" if use_log_rnorm else "R norm=%{y:.4g}<extra></extra>"),
                ),
                row=2,
                col=1,
            )
        fig.update_xaxes(
            title_text="All Residuals",
            row=2,
            col=1,
            showline=True,
            linewidth=1.2,
            automargin=True,
        )
        fig.update_yaxes(
            title_text=rnorm_y_title,
            row=2,
            col=1,
            showline=True,
            linewidth=1.2,
            automargin=True,
        )

        blabels, bvals = _three_pillar_labels_values(metrics)
        bcolors = [_adm_bar_color_plotly(v) for v in bvals]
        bx = [v if math.isfinite(v) else 0.0 for v in bvals]
        btext = [format_admissibility_pct(v) if math.isfinite(v) else "N/A" for v in bvals]
        adm_ht = [format_admissibility_pct(v) if math.isfinite(v) else "N/A" for v in bx]
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
            ),
            row=2,
            col=4,
        )
        _adm_full = category_adm_bar_axis_range_percent_full()
        fig.update_xaxes(
            title_text="Admissibility (%)",
            range=list(_adm_full),
            tickformat=".2f%",
            row=2,
            col=4,
            showline=True,
            linewidth=1.2,
            automargin=True,
        )
        fig.update_yaxes(row=2, col=4, showline=True, linewidth=1.2, automargin=True)

        cat_order = ("laws", "constitutive")
        plot_cols = (1, 4)
        line_row = 3
        for ci, cat in enumerate(cat_order):
            col = plot_cols[ci]
            info = category_training.get(cat, {"keys": [], "displays": [], "r_norm_mat": np.zeros((0, n))})
            ckeys: List[str] = info["keys"]
            displays: List[str] = info["displays"]
            mat = info["r_norm_mat"]
            if not ckeys:
                fig.add_trace(
                    go.Scatter(
                        x=[indices[len(indices) // 2] if indices else 0],
                        y=[0.0],
                        mode="text",
                        text=["No keys in this category"],
                        textposition="middle center",
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=line_row,
                    col=col,
                )
            else:
                for i, _kk in enumerate(ckeys):
                    ys = mat[i, :]
                    if np.all(np.isfinite(ys)):
                        y_plot = (
                            np.log10(np.maximum(ys, 0.0) + R_NORM_LOG_EPS)
                            if use_log_rnorm
                            else ys
                        )
                        ht_y = "log10 R_norm" if use_log_rnorm else "R norm"
                        fig.add_trace(
                            go.Scatter(
                                x=indices,
                                y=y_plot,
                                mode="lines",
                                name=displays[i],
                                legendgroup=f"cat_{cat}",
                                line=dict(
                                    color=(
                                        RESIDUAL_COLOR_LAWS
                                        if cat == "laws"
                                        else (RESIDUAL_COLOR_CONSTITUTIVE if cat == "constitutive" else RESIDUAL_COLOR_OTHER)
                                    ),
                                    width=2.2,
                                ),
                                hovertemplate=f"{displays[i]}<br>{step_label}=%{{x}}<br>{ht_y}=%{{y:.4g}}<extra></extra>",
                            ),
                            row=line_row,
                            col=col,
                        )
            fig.update_xaxes(
                title_text=step_label,
                row=line_row,
                col=col,
                showline=True,
                linewidth=1.2,
                mirror=True,
                automargin=True,
            )
            fig.update_yaxes(
                title_text=rnorm_y_title if ci == 0 else "",
                row=line_row,
                col=col,
                showline=True,
                linewidth=1.2,
                mirror=True,
                automargin=True,
            )

        if n > 15:
            fig.update_xaxes(tickangle=-35, row=line_row, col=1, automargin=True)
            fig.update_xaxes(tickangle=-35, row=line_row, col=4, automargin=True)

        spatial_row = 4
        mid = indices[len(indices) // 2] if indices else 0

        if has_spatial:
            _plotly_add_spatial_panel_to_subplot(
                fig,
                row=spatial_row,
                col=1,
                spatial=spatial,
                hm_cs=hm_cs,
                mid=mid,
                colorbar_compact=False,
                use_log_rnorm=use_log_rnorm,
                colorbar_scale_title=spatial_z_title,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[mid],
                    y=[0.0],
                    mode="text",
                    text=["No law spatial slice<br>(pass spatial_law_panel or residuals + state_pred)"],
                    textposition="middle center",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=spatial_row,
                col=1,
            )
            fig.update_xaxes(visible=False, row=spatial_row, col=1)
            fig.update_yaxes(visible=False, row=spatial_row, col=1)

        if has_spatial_rnorm:
            _plotly_add_spatial_panel_to_subplot(
                fig,
                row=spatial_row,
                col=4,
                spatial=spatial_rnorm,
                hm_cs=hm_cs,
                mid=mid,
                colorbar_compact=False,
                use_log_rnorm=use_log_rnorm,
                colorbar_scale_title=spatial_z_title,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[mid],
                    y=[0.0],
                    mode="text",
                    text=["No constitutive spatial<br>|residual| (pass panel or residuals + state_pred)"],
                    textposition="middle center",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=spatial_row,
                col=4,
            )
            fig.update_xaxes(visible=False, row=spatial_row, col=4)
            fig.update_yaxes(visible=False, row=spatial_row, col=4)

        title_base = figure_title or "Monitor visualization"
        status_line = ""
        if math.isfinite(last_ov):
            status_line = (
                f"<br><span style='font-size:14px;font-weight:600'>"
                f"Overall admissibility (final): {format_admissibility_pct(last_ov)} — "
                f"{_admissibility_status_hml_plotly(last_ov)}"
                f"</span>"
            )
        title_text = f"<b>{title_base}</b>{status_line}"

        n_rows = len(specs)
        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.5,
                xanchor="center",
                pad=dict(t=18, b=24),
                font=dict(size=19, family="Arial, sans-serif"),
            ),
            height=330 + 325 * n_rows,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.06,
                x=0.5,
                xanchor="center",
                font=dict(size=9, family="Arial, sans-serif"),
                tracegroupgap=10,
                itemsizing="constant",
            ),
            margin=dict(l=100, r=100, t=205, b=150),
            hovermode="closest",
            template="plotly_white",
            plot_bgcolor="#f7f8fa",
            font=dict(size=12, family="Arial, sans-serif"),
        )
        align_heatmap_colorbars_to_subplot_domains(fig)
        return fig

    if polar_full:
        st_law = _plotly_spatial_panel_title_with_subtitle(law_sp_full, spatial_z_title)
        st_const = _plotly_spatial_panel_title_with_subtitle(const_sp_full, spatial_z_title)
        specs = [
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy", "colspan": 2}, None],
        ]
        row_heights = [0.28, 0.32, 0.40]
        fig = make_subplots(
            rows=3,
            cols=2,
            specs=specs,
            vertical_spacing=0.10,
            row_heights=row_heights,
            subplot_titles=(
                st_law,
                st_const,
                "Overall admissibility",
                "Category admissibility (final step)",
                "Normalized residuals (all keys)",
            ),
        )
        fig.update_annotations(font=dict(size=11, family="Arial, sans-serif", color="#1a1a1a"))

        mid_pf = indices[len(indices) // 2] if indices else 0
        fig.add_trace(
            go.Scatter(
                x=[mid_pf],
                y=[0.0],
                mode="text",
                text=["No law spatial slice<br>(pass panel or residuals + state_pred)"],
                textposition="middle center",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.update_xaxes(visible=False, row=1, col=1)
        fig.update_yaxes(visible=False, row=1, col=1)
        fig.add_trace(
            go.Scatter(
                x=[mid_pf],
                y=[0.0],
                mode="text",
                text=["No constitutive spatial<br>|residual| rows"],
                textposition="middle center",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(visible=False, row=1, col=2)
        fig.update_yaxes(visible=False, row=1, col=2)

        last_ov_pf = float(overall_adm[-1]) if len(overall_adm) else float("nan")
        adm_hover_pf = []
        for y in overall_adm:
            try:
                fy = float(y)
            except (TypeError, ValueError):
                adm_hover_pf.append("N/A")
            else:
                adm_hover_pf.append(format_admissibility_pct(fy) if math.isfinite(fy) else "N/A")
        if any(np.isfinite(overall_adm)):
            fig.add_trace(
                go.Scatter(
                    x=indices,
                    y=overall_adm,
                    mode="lines",
                    name="Overall admissibility",
                    line=dict(color="#2c3e50", width=2.8),
                    showlegend=False,
                    text=adm_hover_pf,
                    hovertemplate="Overall<br>%{x}<br>%{text}<extra></extra>",
                ),
                row=2,
                col=1,
            )
            if math.isfinite(last_ov_pf):
                lix_pf = indices[-1]
                fig.add_trace(
                    go.Scatter(
                        x=[lix_pf],
                        y=[last_ov_pf],
                        mode="markers",
                        marker=dict(size=12, color="#c0392b", line=dict(width=2, color="white")),
                        showlegend=False,
                        text=[format_admissibility_pct(last_ov_pf)],
                        hovertemplate="Final<br>%{text}<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )
                fig.add_annotation(
                    x=lix_pf,
                    y=last_ov_pf,
                    text=f" {format_admissibility_pct(last_ov_pf)}",
                    showarrow=False,
                    xanchor="left",
                    yanchor="bottom",
                    font=dict(size=11, color="#2c3e50", family="Arial, sans-serif"),
                    row=2,
                    col=1,
                )
        fig.update_xaxes(
            title_text=step_label,
            row=2,
            col=1,
            showline=True,
            linewidth=1.2,
            mirror=True,
            automargin=True,
        )
        fig.update_yaxes(
            title_text="Admissibility (%)",
            tickformat=".2f%",
            autorange=True,
            row=2,
            col=1,
            showline=True,
            linewidth=1.2,
            mirror=True,
            automargin=True,
        )

        blabels, bvals = _three_pillar_labels_values(metrics)
        bcolors = [_adm_bar_color_plotly(v) for v in bvals]
        bx = [v if math.isfinite(v) else 0.0 for v in bvals]
        btext = [format_admissibility_pct(v) if math.isfinite(v) else "N/A" for v in bvals]
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
                customdata=btext,
                hovertemplate="%{y}<br>admissibility=%{customdata}<extra></extra>",
            ),
            row=2,
            col=2,
        )
        _adm_pf = category_adm_bar_axis_range_percent_full()
        fig.update_xaxes(
            title_text="Admissibility (%)",
            range=list(_adm_pf),
            tickformat=".2f%",
            row=2,
            col=2,
            showline=True,
            linewidth=1.2,
            automargin=True,
        )
        fig.update_yaxes(row=2, col=2, automargin=True)

        row_bar = 3
        if mode_eff == "test" or use_bar_chart:
            x_labels_pf = [truncate_display_label(d, 44) for d in bar_display]
            bv = np.asarray(bar_values, dtype=float)
            bar_colors_pf = [
                _residual_color_from_key(bar_keys[i] if i < len(bar_keys) else "")
                for i in range(len(x_labels_pf))
            ]
            y_vals_pf = [
                (
                    float(np.log10(max(float(rv), 0.0) + R_NORM_LOG_EPS))
                    if np.isfinite(rv) and use_log_rnorm
                    else (float(rv) if np.isfinite(rv) else 0.0)
                )
                for rv in bv
            ]
            fig.add_trace(
                go.Bar(
                    x=x_labels_pf,
                    y=y_vals_pf,
                    marker_color=bar_colors_pf,
                    showlegend=False,
                    hovertemplate="%{x}<br>"
                    + (
                        "log10(R_norm+ε)=%{y:.4g}<extra></extra>"
                        if use_log_rnorm
                        else "R norm=%{y:.4g}<extra></extra>"
                    ),
                ),
                row=row_bar,
                col=1,
            )
            fig.update_xaxes(
                title_text="Residual key",
                row=row_bar,
                col=1,
                showline=True,
                linewidth=1.2,
                automargin=True,
            )
            fig.update_yaxes(
                title_text=rnorm_y_title,
                row=row_bar,
                col=1,
                automargin=True,
            )
    else:
        st_law_c = _spatial_heatmap_subplot_title(spatial if has_spatial else None, law_sp_full)
        st_const_c = _spatial_heatmap_subplot_title(spatial_rnorm if has_spatial_rnorm else None, const_sp_full)
        nr_compact = bundle.get("nr_title") or "Normalized Residuals"
        specs = [
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy", "colspan": 2}, None],
        ]
        row_heights = [0.32, 0.20, 0.48]
        fig = make_subplots(
            rows=3,
            cols=2,
            specs=specs,
            vertical_spacing=0.085,
            row_heights=row_heights,
            subplot_titles=(
                st_law_c,
                st_const_c,
                "Category admissibility (final step)",
                "",
                nr_compact,
                "",
            ),
        )
        fig.update_annotations(font=dict(size=11, family="Arial, sans-serif", color="#1a1a1a"))

        mid = indices[len(indices) // 2] if indices else 0
        if has_spatial:
            _plotly_add_spatial_panel_to_subplot(
                fig,
                row=1,
                col=1,
                spatial=spatial,
                hm_cs=hm_cs,
                mid=mid,
                colorbar_compact=True,
                use_log_rnorm=use_log_rnorm,
                colorbar_scale_title=spatial_z_title,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[mid],
                    y=[0.0],
                    mode="text",
                    text=["No law spatial slice<br>(need coord in state_pred)"],
                    textposition="middle center",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )
            fig.update_xaxes(visible=False, row=1, col=1)
            fig.update_yaxes(visible=False, row=1, col=1)

        if has_spatial_rnorm:
            _plotly_add_spatial_panel_to_subplot(
                fig,
                row=1,
                col=2,
                spatial=spatial_rnorm,
                hm_cs=hm_cs,
                mid=mid,
                colorbar_compact=True,
                use_log_rnorm=use_log_rnorm,
                colorbar_scale_title=spatial_z_title,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[mid],
                    y=[0.0],
                    mode="text",
                    text=["No constitutive spatial<br>|residual| rows"],
                    textposition="middle center",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=2,
            )
            fig.update_xaxes(visible=False, row=1, col=2)
            fig.update_yaxes(visible=False, row=1, col=2)

        blabels, bvals = _three_pillar_labels_values(metrics)
        bcolors = [_adm_bar_color_plotly(v) for v in bvals]
        bx = [v if math.isfinite(v) else 0.0 for v in bvals]
        btext = [format_admissibility_pct(v) if math.isfinite(v) else "N/A" for v in bvals]
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
                customdata=btext,
                hovertemplate="%{y}<br>admissibility=%{customdata}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        _adm3_full = category_adm_bar_axis_range_percent_full()
        fig.update_xaxes(
            title_text="Admissibility (%)",
            range=list(_adm3_full),
            tickformat=".2f%",
            row=2,
            col=1,
            showline=True,
            linewidth=1.2,
            automargin=True,
        )
        fig.update_yaxes(row=2, col=1, showline=True, linewidth=1.2, automargin=True)

        if mode_eff == "test" or use_bar_chart:
            x_labels_c = [truncate_display_label(d, 44) for d in bar_display]
            bv_c = np.asarray(bar_values, dtype=float)
            bar_colors_c = [
                _residual_color_from_key(bar_keys[i] if i < len(bar_keys) else "")
                for i in range(len(x_labels_c))
            ]
            y_vals_c = [
                (
                    float(np.log10(max(float(rv), 0.0) + R_NORM_LOG_EPS))
                    if np.isfinite(rv) and use_log_rnorm
                    else (float(rv) if np.isfinite(rv) else 0.0)
                )
                for rv in bv_c
            ]
            fig.add_trace(
                go.Bar(
                    x=x_labels_c,
                    y=y_vals_c,
                    marker_color=bar_colors_c,
                    showlegend=False,
                    hovertemplate="%{x}<br>"
                    + (
                        "log10(R_norm+ε)=%{y:.4g}<extra></extra>"
                        if use_log_rnorm
                        else "R norm=%{y:.4g}<extra></extra>"
                    ),
                ),
                row=3,
                col=1,
            )
            fig.update_xaxes(
                title_text="Residual key",
                row=3,
                col=1,
                showline=True,
                linewidth=1.2,
                automargin=True,
            )
            fig.update_yaxes(title_text=rnorm_y_title, row=3, col=1, automargin=True)

    last_ov_c = float(overall_adm[-1]) if len(overall_adm) else float("nan")
    title_base = figure_title or "Monitor visualization"
    status_c = ""
    if math.isfinite(last_ov_c):
        status_c = (
            f"<br><span style='font-size:14px;font-weight:600'>"
            f"Overall admissibility (final): {format_admissibility_pct(last_ov_c)} — "
            f"{_admissibility_status_hml_plotly(last_ov_c)}"
            f"</span>"
        )
    title_text = f"<b>{title_base}</b>{status_c}"

    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            xanchor="center",
            pad=dict(t=18, b=20),
            font=dict(size=19, family="Arial, sans-serif"),
        ),
        height=280 + 305 * len(specs),
        margin=dict(t=155, r=88, l=88, b=88),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            font=dict(size=9, family="Arial, sans-serif"),
        ),
        hovermode="closest",
        template="plotly_white",
        plot_bgcolor="#f7f8fa",
        font=dict(size=12, family="Arial, sans-serif"),
    )
    align_heatmap_colorbars_to_subplot_domains(fig)

    return fig


def _pos_axis_title_card(sp: Optional[Dict[str, Any]]) -> str:
    if not sp:
        return "Position x"
    ax = sp.get("position_axis") or "x"
    return f"Position {ax}"


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
