"""
Interactive Plotly dashboard for :func:`moju.monitor.auditor.visualize`.

Requires ``pip install plotly`` (optional extra ``moju[viz]``).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from moju.monitor.visualize_labels import pretty_category_name

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
        specs_row1 = [
            {"type": "xy", "colspan": 3},
            None,
            None,
            {"type": "xy", "colspan": 3},
            None,
            None,
        ]
        specs_row2 = [
            {"type": "xy", "colspan": 3},
            None,
            None,
            {"type": "xy", "colspan": 3},
            None,
            None,
        ]
        specs: List[List[Any]] = [specs_row1, specs_row2]
        row_heights = [0.34, 0.36]
        if has_spatial or has_spatial_rnorm:
            if has_spatial and has_spatial_rnorm:
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
            else:
                specs.append([{"type": "xy", "colspan": 6}, None, None, None, None, None])
            row_heights.append(0.30)

        sub_titles = [
            "Overall admissibility",
            "Category admissibility (final step)",
            category_titles.get("laws", "Normalized Governing Laws Residuals"),
            category_titles.get("constitutive", "Normalized Constitutive Residuals"),
        ]
        if has_spatial:
            sub_titles.append(
                f"Governing laws R_norm ({spatial.get('position_axis', 'x')})" if spatial else "Governing laws R_norm"
            )
        if has_spatial_rnorm:
            sr0 = spatial_rnorm or {}
            sub_titles.append(f"Constitutive R_norm ({sr0.get('position_axis', 'x')})")

        fig = make_subplots(
            rows=len(specs),
            cols=6,
            specs=specs,
            vertical_spacing=0.1,
            horizontal_spacing=0.06,
            row_heights=row_heights,
            subplot_titles=tuple(sub_titles),
        )

        last_ov = float(overall_adm[-1]) if len(overall_adm) else float("nan")
        if any(np.isfinite(overall_adm)):
            fig.add_trace(
                go.Scatter(
                    x=indices,
                    y=overall_adm,
                    mode="lines",
                    name="Overall admissibility",
                    line=dict(color="#2c3e50", width=2.8),
                    showlegend=False,
                    hovertemplate="Overall<br>%{x}<br>%{y:.4f}<extra></extra>",
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
                        hovertemplate="Final<br>%{y:.4f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_annotation(
                    x=lix,
                    y=last_ov,
                    text=f" {last_ov:.4f}",
                    showarrow=False,
                    xanchor="left",
                    yanchor="bottom",
                    font=dict(size=11, color="#2c3e50", family="Arial, sans-serif"),
                    row=1,
                    col=1,
                )
        fig.update_xaxes(title_text=step_label, row=1, col=1, showline=True, linewidth=1.2, mirror=True)
        fig.update_yaxes(
            title_text="Admissibility",
            autorange=True,
            row=1,
            col=1,
            showline=True,
            linewidth=1.2,
            mirror=True,
        )

        blabels, bvals = _three_pillar_labels_values(metrics)
        bcolors = [_adm_bar_color_plotly(v) for v in bvals]
        bx = [v if math.isfinite(v) else 0.0 for v in bvals]
        btext = [f"{v:.3f}" if math.isfinite(v) else "N/A" for v in bvals]
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
                hovertemplate="%{y}<br>admissibility=%{x:.4f}<extra></extra>",
            ),
            row=1,
            col=4,
        )
        fig.update_xaxes(title_text="Admissibility", range=[0, 1.12], row=1, col=4, showline=True, linewidth=1.2)
        fig.update_yaxes(row=1, col=4, showline=True, linewidth=1.2)

        cat_order = ("laws", "constitutive")
        plot_cols = (1, 4)
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
                    row=2,
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
                            row=2,
                            col=col,
                        )
            fig.update_xaxes(
                title_text=step_label,
                row=2,
                col=col,
                showline=True,
                linewidth=1.2,
                mirror=True,
            )
            fig.update_yaxes(
                title_text=rnorm_y_title if ci == 0 else "",
                row=2,
                col=col,
                showline=True,
                linewidth=1.2,
                mirror=True,
            )

        if has_spatial:
            Z = spatial["Z"]
            x_sp = spatial["x"]
            row_labels = spatial["row_labels"]
            fig.add_trace(
                go.Heatmap(
                    x=x_sp,
                    y=list(range(len(row_labels))),
                    z=Z,
                    colorscale=hm_cs,
                    colorbar=dict(title="R norm", len=0.5),
                    hovertemplate="x=%{x:.4g}<br>row=%{y}<extra></extra>",
                ),
                row=3,
                col=1,
            )
            fig.update_yaxes(
                tickmode="array",
                tickvals=list(range(len(row_labels))),
                ticktext=row_labels,
                row=3,
                col=1,
            )
            fig.update_xaxes(title_text=_pos_axis_title(spatial), row=3, col=1)
        if has_spatial_rnorm:
            Zr = spatial_rnorm["Z"]
            xr = spatial_rnorm["x"]
            rlabels = spatial_rnorm["row_labels"]
            rcol = 4 if has_spatial else 1
            fig.add_trace(
                go.Heatmap(
                    x=xr,
                    y=list(range(len(rlabels))),
                    z=Zr,
                    colorscale=hm_cs,
                    colorbar=dict(title="R norm", len=0.5),
                    hovertemplate="x=%{x:.4g}<br>row=%{y}<extra></extra>",
                ),
                row=3,
                col=rcol,
            )
            fig.update_yaxes(
                tickmode="array",
                tickvals=list(range(len(rlabels))),
                ticktext=rlabels,
                row=3,
                col=rcol,
            )
            fig.update_xaxes(title_text=_pos_axis_title(spatial_rnorm), row=3, col=rcol)

        title_base = figure_title or "Monitor visualization"
        status_line = ""
        if math.isfinite(last_ov):
            status_line = (
                f"<br><span style='font-size:14px;font-weight:600'>"
                f"Overall admissibility (final): {last_ov:.4f} — {_admissibility_status_hml_plotly(last_ov)}"
                f"</span>"
            )
        title_text = f"<b>{title_base}</b>{status_line}"

        n_rows = len(specs)
        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.5,
                xanchor="center",
                font=dict(size=20, family="Arial, sans-serif"),
            ),
            height=240 + 300 * n_rows,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.36,
                x=1.02,
                xanchor="left",
                font=dict(size=10),
                tracegroupgap=4,
            ),
            margin=dict(l=70, r=180, t=120, b=70),
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
        vertical_spacing=0.11,
        row_heights=row_heights,
    )

    row = 1
    if mode_eff == "test" or use_bar_chart:
        y_labels = [d if len(d) < 50 else d[:47] + "…" for d in bar_display]
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
    btext = [f"{v:.3f}" if math.isfinite(v) else "N/A" for v in bvals]
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
            hovertemplate="%{y}<br>admissibility=%{x:.4f}<extra></extra>",
        ),
        row=polar_row,
        col=1,
    )
    fig.update_xaxes(title_text="Admissibility", range=[0, 1.12], row=polar_row, col=1, showline=True, linewidth=1.2)
    fig.update_yaxes(row=polar_row, col=1)

    if not polar_full:
        if spatial is not None:
            Z = spatial["Z"]
            x_sp = spatial["x"]
            row_labels = spatial["row_labels"]
            fig.add_trace(
                go.Heatmap(
                    x=x_sp,
                    y=list(range(len(row_labels))),
                    z=Z,
                    colorscale=hm_cs,
                    colorbar=dict(title="R norm", len=0.45),
                    hovertemplate="x=%{x:.4g}<br>row=%{y}<extra></extra>",
                ),
                row=polar_row,
                col=2,
            )
            fig.update_yaxes(
                tickmode="array",
                tickvals=list(range(len(row_labels))),
                ticktext=row_labels,
                row=polar_row,
                col=2,
            )
            fig.update_xaxes(title_text=_pos_axis_title(spatial), row=polar_row, col=2)
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
            f"Overall admissibility (final): {last_ov_c:.4f} — {_admissibility_status_hml_plotly(last_ov_c)}"
            f"</span>"
        )
    title_text = f"<b>{title_base}</b>{status_c}"

    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            xanchor="center",
            font=dict(size=20, family="Arial, sans-serif"),
        ),
        height=220 + 280 * len(specs),
        margin=dict(t=110, r=48, l=64, b=56),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
        template="plotly_white",
        font=dict(size=12, family="Arial, sans-serif"),
    )

    return fig
