"""
Interactive Plotly dashboard for :func:`moju.monitor.auditor.visualize`.

Requires ``pip install plotly`` (optional extra ``moju[viz]``).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from moju.monitor.visualize_labels import (
    category_adm_bar_x_range,
    format_admissibility_pct,
    pretty_category_name,
    truncate_display_label,
)

R_NORM_LOG_EPS = 1e-12


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
    hm_cs = spatial_heatmap_colorscale or "Jet"

    def _pos_axis_title(sp: Optional[Dict[str, Any]]) -> str:
        if not sp:
            return "Position x"
        ax = sp.get("position_axis") or "x"
        return f"Position {ax}"

    n = bundle["n"]
    indices = list(range(n))
    category_training: Dict[str, Dict[str, Any]] = bundle.get("category_training") or {}
    category_titles: Dict[str, str] = bundle.get("category_titles") or {}
    use_bar_chart = bundle["use_bar_chart"]
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
    polar_full = mode_eff == "test" and not has_spatial

    palette = [
        "#4e79a7",
        "#f28e2b",
        "#59a14f",
        "#e15759",
        "#76b7b2",
        "#edc948",
        "#b07aa1",
        "#ff9da7",
        "#9c755f",
        "#bab0ac",
    ]

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
        specs_row_hm = [
            {"type": "xy", "colspan": 3},
            None,
            None,
            {"type": "xy", "colspan": 3},
            None,
            None,
        ]
        specs: List[List[Any]] = [specs_row_overall, specs_row_bars, specs_row_lines, specs_row_hm]
        row_heights = [0.19, 0.19, 0.25, 0.25]
        if has_spatial or has_spatial_rnorm:
            specs.append(
                [
                    {"type": "xy", "colspan": 3},
                    None,
                    None,
                    {"type": "xy", "colspan": 3},
                    None,
                    None,
                ]
            )
            row_heights.append(0.28)

        pos_ax_law = spatial.get("position_axis", "x") if spatial else "x"
        pos_ax_ic = (spatial_rnorm or {}).get("position_axis", "x")
        sub_titles = [
            "Overall admissibility",
            "Law R_norm (final step)",
            "Category admissibility (final step)",
            category_titles.get("laws", "Normalized Governing Laws Residuals"),
            category_titles.get("constitutive", "Normalized Constitutive Residuals"),
            "Governing laws R_norm (vs step)",
            "Constitutive R_norm (vs step)",
        ]
        if has_spatial or has_spatial_rnorm:
            sub_titles.append(f"Law R_norm vs {pos_ax_law}")
            sub_titles.append(f"Implied constitutive R_norm vs {pos_ax_ic}")

        fig = make_subplots(
            rows=len(specs),
            cols=6,
            specs=specs,
            vertical_spacing=0.145,
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

        info_laws_bar = category_training.get("laws", {"keys": [], "displays": [], "r_norm_mat": np.zeros((0, n))})
        lk = info_laws_bar["keys"]
        ld = info_laws_bar["displays"]
        mat_lb = np.asarray(info_laws_bar["r_norm_mat"], dtype=float)
        if not lk or mat_lb.size == 0 or n < 1:
            fig.add_trace(
                go.Scatter(
                    x=[indices[len(indices) // 2] if indices else 0],
                    y=[0.0],
                    mode="text",
                    text=["No law keys in this run"],
                    textposition="middle center",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=2,
                col=1,
            )
        else:
            y_lbl = [truncate_display_label(d, 44) for d in ld]
            x_fin = [float(mat_lb[i, -1]) if np.isfinite(mat_lb[i, -1]) else 0.0 for i in range(len(lk))]
            fig.add_trace(
                go.Bar(
                    x=x_fin,
                    y=y_lbl,
                    orientation="h",
                    marker_color="#6baed6",
                    showlegend=False,
                    hovertemplate="%{y}<br>R norm=%{x:.4g}<extra></extra>",
                ),
                row=2,
                col=1,
            )
        fig.update_xaxes(
            title_text="Normalized residual (R norm)",
            row=2,
            col=1,
            showline=True,
            linewidth=1.2,
            automargin=True,
        )
        fig.update_yaxes(row=2, col=1, showline=True, linewidth=1.2, automargin=True)

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
        _adm_x0, _adm_x1 = category_adm_bar_x_range(list(bvals))
        fig.update_xaxes(
            title_text="Admissibility (%)",
            range=[_adm_x0, _adm_x1],
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
                                line=dict(color=palette[i % len(palette)], width=2.2),
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

        hm_row = 4
        jet_timeline = "Jet"
        hm_cb = rnorm_y_title
        for ci, cat in enumerate(cat_order):
            col = plot_cols[ci]
            info = category_training.get(cat, {"keys": [], "displays": [], "r_norm_mat": np.zeros((0, n))})
            ckeys: List[str] = info["keys"]
            displays: List[str] = info["displays"]
            mat = np.asarray(info["r_norm_mat"], dtype=float)
            if not ckeys or mat.size == 0:
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
                    row=hm_row,
                    col=col,
                )
            else:
                z_hm = (
                    np.log10(np.maximum(mat, 0.0) + R_NORM_LOG_EPS)
                    if use_log_rnorm
                    else mat
                )
                disp_short = [truncate_display_label(d, 34) for d in displays]
                _cd_hm = np.broadcast_to(
                    np.asarray(displays, dtype=object)[:, np.newaxis],
                    (len(displays), n),
                )
                fig.add_trace(
                    go.Heatmap(
                        x=list(indices),
                        y=list(range(len(displays))),
                        z=z_hm,
                        colorscale=jet_timeline,
                        colorbar=dict(title=dict(text=hm_cb, side="right"), len=0.36, xpad=12, thickness=14),
                        customdata=_cd_hm,
                        hovertemplate=(
                            f"{step_label}=%{{x}}<br>%{{customdata}}<br>value=%{{z:.4g}}<extra></extra>"
                        ),
                    ),
                    row=hm_row,
                    col=col,
                )
                fig.update_yaxes(
                    tickmode="array",
                    tickvals=list(range(len(displays))),
                    ticktext=disp_short,
                    tickangle=-25,
                    row=hm_row,
                    col=col,
                    automargin=True,
                )
            fig.update_xaxes(
                title_text=step_label,
                row=hm_row,
                col=col,
                showline=True,
                linewidth=1.2,
                automargin=True,
            )

        if n > 12:
            fig.update_xaxes(tickangle=-38, row=hm_row, col=1, automargin=True)
            fig.update_xaxes(tickangle=-38, row=hm_row, col=4, automargin=True)

        spatial_row = 5 if (has_spatial or has_spatial_rnorm) else None
        if spatial_row is not None:
            mid = indices[len(indices) // 2] if indices else 0

            if has_spatial:
                Z = spatial["Z"]
                x_sp = spatial["x"]
                row_labels = spatial["row_labels"]
                rl_s = [truncate_display_label(lb, 34) for lb in row_labels]
                fig.add_trace(
                    go.Heatmap(
                        x=x_sp,
                        y=list(range(len(row_labels))),
                        z=Z,
                        colorscale=hm_cs,
                        colorbar=dict(
                            title=dict(text="R norm", side="right"),
                            len=0.36,
                            x=1.02,
                            xpad=14,
                            thickness=14,
                        ),
                        hovertemplate="x=%{x:.4g}<br>%{customdata}<extra></extra>",
                        customdata=np.broadcast_to(
                            np.asarray(row_labels, dtype=object)[:, np.newaxis],
                            (len(row_labels), len(x_sp)),
                        ),
                    ),
                    row=spatial_row,
                    col=1,
                )
                fig.update_yaxes(
                    tickmode="array",
                    tickvals=list(range(len(row_labels))),
                    ticktext=rl_s,
                    tickangle=-20,
                    row=spatial_row,
                    col=1,
                    automargin=True,
                )
                fig.update_xaxes(
                    title_text=_pos_axis_title(spatial),
                    row=spatial_row,
                    col=1,
                    automargin=True,
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
                    row=spatial_row,
                    col=1,
                )
                fig.update_xaxes(visible=False, row=spatial_row, col=1)
                fig.update_yaxes(visible=False, row=spatial_row, col=1)

            if has_spatial_rnorm:
                Zr = spatial_rnorm["Z"]
                xr = spatial_rnorm["x"]
                rlabels = spatial_rnorm["row_labels"]
                rl2 = [truncate_display_label(lb, 34) for lb in rlabels]
                fig.add_trace(
                    go.Heatmap(
                        x=xr,
                        y=list(range(len(rlabels))),
                        z=Zr,
                        colorscale=hm_cs,
                        colorbar=dict(
                            title=dict(text="R norm", side="right"),
                            len=0.36,
                            x=1.02,
                            xpad=14,
                            thickness=14,
                        ),
                        hovertemplate="x=%{x:.4g}<br>%{customdata}<extra></extra>",
                        customdata=np.broadcast_to(
                            np.asarray(rlabels, dtype=object)[:, np.newaxis],
                            (len(rlabels), len(xr)),
                        ),
                    ),
                    row=spatial_row,
                    col=4,
                )
                fig.update_yaxes(
                    tickmode="array",
                    tickvals=list(range(len(rlabels))),
                    ticktext=rl2,
                    tickangle=-20,
                    row=spatial_row,
                    col=4,
                    automargin=True,
                )
                fig.update_xaxes(
                    title_text=_pos_axis_title(spatial_rnorm),
                    row=spatial_row,
                    col=4,
                    automargin=True,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=[mid],
                        y=[0.0],
                        mode="text",
                        text=["No implied constitutive<br>spatial rows"],
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
                pad=dict(t=16, b=10),
                font=dict(size=19, family="Arial, sans-serif"),
            ),
            height=310 + 325 * n_rows,
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
            margin=dict(l=100, r=100, t=170, b=150),
            hovermode="closest",
            template="plotly_white",
            font=dict(size=12, family="Arial, sans-serif"),
        )
        return fig

    if polar_full:
        specs = [
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy", "colspan": 2}, None],
        ]
        row_heights = [0.45, 0.55]
    else:
        specs = [
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy"}, {"type": "xy"}],
        ]
        row_heights = [0.45, 0.55]

    fig = make_subplots(
        rows=len(specs),
        cols=2,
        specs=specs,
        vertical_spacing=0.135,
        row_heights=row_heights,
    )
    fig.update_annotations(font=dict(size=11, family="Arial, sans-serif", color="#1a1a1a"))

    row = 1
    if mode_eff == "test" or use_bar_chart:
        y_labels = [truncate_display_label(d, 44) for d in bar_display]
        fig.add_trace(
            go.Bar(
                x=np.where(np.isfinite(bar_values), bar_values, 0.0),
                y=y_labels,
                orientation="h",
                marker_color="#6baed6",
                showlegend=False,
                hovertemplate="%{y}<br>R norm=%{x:.4g}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        fig.update_xaxes(title_text="Normalized residual (R norm)", row=row, col=1, showline=True, linewidth=1.2)
        fig.update_yaxes(title_text="", row=row, col=1)
        row += 1

    polar_row = row
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
        row=polar_row,
        col=1,
    )
    _adm2_x0, _adm2_x1 = category_adm_bar_x_range(list(bvals))
    fig.update_xaxes(
        title_text="Admissibility (%)",
        range=[_adm2_x0, _adm2_x1],
        tickformat=".2f%",
        row=polar_row,
        col=1,
        showline=True,
        linewidth=1.2,
        automargin=True,
    )
    fig.update_yaxes(row=polar_row, col=1, automargin=True)

    if not polar_full:
        if spatial is not None:
            Z = spatial["Z"]
            x_sp = spatial["x"]
            row_labels = spatial["row_labels"]
            rl_p = [truncate_display_label(lb, 34) for lb in row_labels]
            fig.add_trace(
                go.Heatmap(
                    x=x_sp,
                    y=list(range(len(row_labels))),
                    z=Z,
                    colorscale=hm_cs,
                    colorbar=dict(
                        title=dict(text="R norm", side="right"),
                        len=0.38,
                        xpad=12,
                        thickness=14,
                    ),
                    hovertemplate="x=%{x:.4g}<br>%{customdata}<extra></extra>",
                    customdata=np.broadcast_to(
                        np.asarray(row_labels, dtype=object)[:, np.newaxis],
                        (len(row_labels), len(x_sp)),
                    ),
                ),
                row=polar_row,
                col=2,
            )
            fig.update_yaxes(
                tickmode="array",
                tickvals=list(range(len(row_labels))),
                ticktext=rl_p,
                tickangle=-18,
                automargin=True,
                row=polar_row,
                col=2,
            )
            fig.update_xaxes(
                title_text=_pos_axis_title(spatial),
                title_standoff=8,
                automargin=True,
                row=polar_row,
                col=2,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[0],
                    mode="text",
                    text=["Pass spatial_law_panel with<br>x and per-law values"],
                    textposition="middle center",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=polar_row,
                col=2,
            )
            fig.update_xaxes(visible=False, range=[-1, 1], row=polar_row, col=2)
            fig.update_yaxes(visible=False, range=[-1, 1], row=polar_row, col=2)

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
            pad=dict(t=16, b=10),
            font=dict(size=19, family="Arial, sans-serif"),
        ),
        height=260 + 305 * len(specs),
        margin=dict(t=130, r=88, l=88, b=88),
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
        font=dict(size=12, family="Arial, sans-serif"),
    )

    return fig


def _pos_axis_title_card(sp: Optional[Dict[str, Any]]) -> str:
    if not sp:
        return "Position x"
    ax = sp.get("position_axis") or "x"
    return f"Position {ax}"


def build_plotly_law_rnorm_final_bar_figure(bundle: Dict[str, Any]) -> Any:
    """Horizontal bar chart: law R_norm at the final log step (Studio card)."""
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
        y_lbl = [truncate_display_label(d, 44) for d in ld]
        x_fin = [float(mat_lb[i, -1]) if np.isfinite(mat_lb[i, -1]) else 0.0 for i in range(len(lk))]
        fig.add_trace(
            go.Bar(
                x=x_fin,
                y=y_lbl,
                orientation="h",
                marker_color="#6baed6",
                showlegend=False,
                hovertemplate="%{y}<br>R norm=%{x:.4g}<extra></extra>",
            )
        )
        fig.update_xaxes(title_text="Normalized residual (R norm)", showline=True, linewidth=1.2, automargin=True)
        fig.update_yaxes(showline=True, linewidth=1.2, automargin=True)

    fig.update_layout(
        title=dict(text="Law R_norm (final step)", font=dict(size=14, family="Arial, sans-serif")),
        height=max(280, 60 + 28 * max(1, len(lk) if lk else 1)),
        margin=dict(l=12, r=12, t=48, b=48),
        template="plotly_white",
        font=dict(size=11, family="Arial, sans-serif"),
    )
    return fig


def build_plotly_category_admissibility_bar_figure(bundle: Dict[str, Any]) -> Any:
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
    _adm_x0, _adm_x1 = category_adm_bar_x_range(list(bvals))
    fig.update_xaxes(
        title_text="Admissibility (%)",
        range=[_adm_x0, _adm_x1],
        tickformat=".2f%",
        showline=True,
        linewidth=1.2,
        automargin=True,
    )
    fig.update_yaxes(showline=True, linewidth=1.2, automargin=True)
    fig.update_layout(
        title=dict(text="Category admissibility (final step)", font=dict(size=14, family="Arial, sans-serif")),
        height=max(220, 60 + 36 * len(blabels)),
        margin=dict(l=12, r=80, t=48, b=48),
        template="plotly_white",
        font=dict(size=11, family="Arial, sans-serif"),
    )
    return fig


def build_plotly_spatial_rnorm_heatmap_card(
    spatial_parsed: Optional[Dict[str, Any]],
    *,
    colorscale: str = "Jet",
    card_title: str = "Spatial R_norm",
) -> Any:
    """
    Single heatmap + slim colorbar for one spatial panel (law or constitutive).

    ``spatial_parsed`` is the output of ``_parse_spatial_law_panel`` or
    ``_parse_spatial_rnorm_panel`` (keys ``x``, ``Z``, ``row_labels``, optional ``position_axis``).
    """
    import numpy as np
    import plotly.graph_objects as go

    n_rows = len(spatial_parsed["row_labels"]) if spatial_parsed else 1
    fig = go.Figure()
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
        Z = spatial_parsed["Z"]
        x_sp = spatial_parsed["x"]
        row_labels = spatial_parsed["row_labels"]
        rl_s = [truncate_display_label(lb, 34) for lb in row_labels]
        fig.add_trace(
            go.Heatmap(
                x=x_sp,
                y=list(range(len(row_labels))),
                z=Z,
                colorscale=colorscale,
                colorbar=dict(
                    title=dict(text="R_norm", side="right", font=dict(size=10)),
                    len=0.72,
                    thickness=10,
                    x=1.02,
                    xpad=6,
                ),
                hovertemplate="x=%{x:.4g}<br>%{customdata}<extra></extra>",
                customdata=np.broadcast_to(
                    np.asarray(row_labels, dtype=object)[:, np.newaxis],
                    (len(row_labels), len(x_sp)),
                ),
            )
        )
        fig.update_yaxes(
            tickmode="array",
            tickvals=list(range(len(row_labels))),
            ticktext=rl_s,
            tickangle=-20,
            automargin=True,
        )
        fig.update_xaxes(
            title_text=_pos_axis_title_card(spatial_parsed),
            showline=True,
            linewidth=1.2,
            automargin=True,
        )

    fig.update_layout(
        title=dict(text=card_title, font=dict(size=14, family="Arial, sans-serif")),
        height=max(320, 80 + 32 * n_rows),
        margin=dict(l=8, r=100, t=48, b=48),
        template="plotly_white",
        font=dict(size=11, family="Arial, sans-serif"),
    )
    return fig
