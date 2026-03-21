"""
Interactive Plotly dashboard mirroring :func:`moju.monitor.auditor.visualize` (matplotlib).

Requires ``pip install plotly`` (optional extra ``moju[viz]``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

_CAT_COLORS = {
    "laws": "#1f77b4",
    "constitutive": "#ff7f0e",
    "scaling": "#2ca02c",
    "data": "#9467bd",
}


def build_plotly_monitor_figure(bundle: Dict[str, Any]) -> Any:
    """
    Build a single interactive ``plotly.graph_objects.Figure`` from a bundle
    produced by :func:`moju.monitor.auditor._build_visualize_bundle`.
    """
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    log = bundle["log"]
    metrics = bundle["metrics"]
    n = bundle["n"]
    plot_keys: List[str] = bundle["plot_keys"]
    legend_keys: List[str] = bundle["legend_keys"]
    buckets = bundle["buckets"]
    ordered_keys: List[str] = bundle["ordered_keys"]
    mat = bundle["mat"]
    short_labels: List[str] = bundle["short_labels"]
    ranked: List[Tuple[str, float]] = bundle["ranked"]
    kinds_order: List[str] = bundle["kinds_order"]
    means_closure: List[float] = bundle["means_closure"]
    om: List[int] = bundle["om"]
    inf: List[int] = bundle["inf"]
    top5: List[str] = bundle["top5"]
    cats_fin: List[Tuple[str, float]] = bundle["cats_fin"]
    rms_title = bundle["rms_title"]

    x_idx = list(range(n))

    subplot_titles = (
        rms_title,
        "Admissibility per key (1/(1+R_norm))",
        "Overall and per-category geometric mean admissibility",
        "Admissibility heatmap (keys × step)",
        "Top residual keys by R_norm (last step)",
        "Constitutive/scaling by closure type (last step)",
        "Omitted / inferred log messages",
        "Top keys: RMS (left) vs R_norm (right)",
        "Category admissibility (last step)",
        "Worst key per category (min admissibility)",
    )

    fig = make_subplots(
        rows=8,
        cols=2,
        specs=[
            [{"colspan": 2, "type": "xy"}, None],
            [{"colspan": 2, "type": "xy"}, None],
            [{"colspan": 2, "type": "xy"}, None],
            [{"colspan": 2, "type": "xy"}, None],
            [{"colspan": 2, "type": "xy"}, None],
            [{"type": "xy"}, {"type": "xy"}],
            [{"colspan": 2, "type": "xy", "secondary_y": True}, None],
            [{"type": "polar"}, {"type": "xy"}],
        ],
        subplot_titles=subplot_titles,
        vertical_spacing=0.045,
        horizontal_spacing=0.08,
        row_heights=[0.11, 0.11, 0.1, 0.14, 0.1, 0.09, 0.12, 0.1],
    )

    # 1 — RMS
    for k in legend_keys:
        rms_vals = [e.get("rms", {}).get(k) for e in log]
        if all(v is not None for v in rms_vals):
            fig.add_trace(
                go.Scatter(
                    x=x_idx,
                    y=rms_vals,
                    mode="lines",
                    name=k,
                    legendgroup=f"rms_{k}",
                    hovertemplate=f"{k}<br>step=%{{x}}<br>RMS=%{{y:.4g}}<extra></extra>",
                ),
                row=1,
                col=1,
            )
    fig.update_yaxes(title_text="RMS", row=1, col=1)

    # 2 — Admissibility per key
    for k in legend_keys:
        adm_vals = [metrics[i]["admissibility_score"].get(k) for i in range(n)]
        if all(v is not None for v in adm_vals):
            fig.add_trace(
                go.Scatter(
                    x=x_idx,
                    y=adm_vals,
                    mode="lines",
                    name=f"{k} (adm)",
                    legendgroup=f"adm_{k}",
                    visible="legendonly",
                    hovertemplate=f"{k}<br>step=%{{x}}<br>admissibility=%{{y:.4f}}<extra></extra>",
                ),
                row=2,
                col=1,
            )
    fig.update_yaxes(title_text="Admissibility", range=[0, 1.05], row=2, col=1)

    # 3 — Overall + categories
    overall_s = [metrics[i]["overall_admissibility_score"] for i in range(n)]
    if any(np.isfinite(overall_s)):
        fig.add_trace(
            go.Scatter(
                x=x_idx,
                y=overall_s,
                mode="lines",
                name="Overall",
                line=dict(color="black", width=3),
                hovertemplate="Overall<br>step=%{x}<br>score=%{y:.4f}<extra></extra>",
            ),
            row=3,
            col=1,
        )
    for cat, col in _CAT_COLORS.items():
        ys = []
        for i in range(n):
            v = metrics[i]["category_admissibility_score"].get(cat)
            ys.append(float(v) if v is not None else float("nan"))
        if any(np.isfinite(ys)):
            fig.add_trace(
                go.Scatter(
                    x=x_idx,
                    y=ys,
                    mode="lines",
                    name=cat,
                    line=dict(color=col, dash="dash"),
                    hovertemplate=f"{cat}<br>step=%{{x}}<br>score=%{{y:.4f}}<extra></extra>",
                ),
                row=3,
                col=1,
            )
    fig.update_yaxes(title_text="Admissibility", range=[0, 1.05], row=3, col=1)

    # 4 — Heatmap
    z_plot = np.where(np.isfinite(mat), mat, np.nan)
    fig.add_trace(
        go.Heatmap(
            z=z_plot,
            x=x_idx,
            y=short_labels,
            colorscale="Viridis",
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="Adm.", len=0.35, y=0.52, thickness=12),
            hovertemplate="%{y}<br>step=%{x}<br>admissibility=%{z:.4f}<extra></extra>",
        ),
        row=4,
        col=1,
    )
    fig.update_xaxes(title_text="Log index", row=4, col=1)

    # 5 — Top offenders
    top_n = min(15, len(ranked))
    if top_n:
        rk, rv = zip(*ranked[:top_n])
        labels = [x if len(x) < 60 else x[:57] + "…" for x in rk]
        fig.add_trace(
            go.Bar(
                x=list(rv),
                y=labels,
                orientation="h",
                marker_color="#c44e52",
                name="R_norm",
                showlegend=False,
                hovertemplate="%{y}<br>R_norm=%{x:.4g}<extra></extra>",
            ),
            row=5,
            col=1,
        )
        fig.update_xaxes(title_text="R_norm (last step)", row=5, col=1)

    # 6a — Closure types
    if kinds_order:
        fig.add_trace(
            go.Bar(
                x=kinds_order,
                y=means_closure,
                marker_color="#4c72b0",
                showlegend=False,
                hovertemplate="%{x}<br>mean adm=%{y:.4f}<extra></extra>",
            ),
            row=6,
            col=1,
        )
        fig.update_yaxes(title_text="Mean admissibility", range=[0, 1.05], row=6, col=1)
    else:
        fig.add_annotation(
            text="No constitutive/scaling keys",
            x=0.5,
            y=0.5,
            xref="x domain",
            yref="y domain",
            showarrow=False,
            row=6,
            col=1,
        )

    # 6b — Omitted / inferred
    if max(om + inf, default=0) > 0:
        fig.add_trace(
            go.Bar(
                x=x_idx,
                y=om,
                name="omitted",
                marker_color="#8c564b",
                hovertemplate="step=%{x}<br>omitted=%{y}<extra></extra>",
            ),
            row=6,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                x=x_idx,
                y=inf,
                name="inferred",
                marker_color="#17becf",
                hovertemplate="step=%{x}<br>inferred=%{y}<extra></extra>",
            ),
            row=6,
            col=2,
        )
        fig.update_yaxes(title_text="Count", row=6, col=2)
    else:
        fig.add_annotation(
            text="No omitted/inferred entries",
            x=0.5,
            y=0.5,
            xref="x domain",
            yref="y domain",
            showarrow=False,
            row=6,
            col=2,
        )

    # 7 — Twin RMS / R_norm
    for k in top5:
        rms_s = [e.get("rms", {}).get(k) for e in log]
        rn_s = [metrics[i]["r_norm"].get(k) for i in range(n)]
        if all(v is not None for v in rms_s):
            fig.add_trace(
                go.Scatter(
                    x=x_idx,
                    y=rms_s,
                    mode="lines",
                    name=f"{k} RMS",
                    legendgroup=f"twin_{k}",
                    hovertemplate=f"{k} RMS<br>step=%{{x}}<br>RMS=%{{y:.4g}}<extra></extra>",
                ),
                row=7,
                col=1,
                secondary_y=False,
            )
        if all(v is not None for v in rn_s):
            fig.add_trace(
                go.Scatter(
                    x=x_idx,
                    y=rn_s,
                    mode="lines",
                    name=f"{k} R_norm",
                    line=dict(dash="dash"),
                    legendgroup=f"twin_{k}",
                    hovertemplate=f"{k} R_norm<br>step=%{{x}}<br>R_norm=%{{y:.4g}}<extra></extra>",
                ),
                row=7,
                col=1,
                secondary_y=True,
            )
    fig.update_yaxes(title_text="RMS", row=7, col=1, secondary_y=False)
    fig.update_yaxes(title_text="R_norm", row=7, col=1, secondary_y=True)

    # 8a — Radar
    if len(cats_fin) >= 2:
        labels = [c[0] for c in cats_fin]
        vals = [c[1] for c in cats_fin]
        theta = labels + [labels[0]]
        rvals = vals + [vals[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=rvals,
                theta=theta,
                fill="toself",
                name="categories",
                line_color="#bcbd22",
                hovertemplate="%{theta}<br>admissibility=%{r:.4f}<extra></extra>",
            ),
            row=8,
            col=1,
        )
        fig.update_polars(radialaxis_range=[0, 1], row=8, col=1)
    else:
        fig.add_annotation(
            text="Radar: need ≥2 categories",
            x=0.5,
            y=0.5,
            xref="x domain",
            yref="y domain",
            showarrow=False,
            row=8,
            col=1,
        )

    # 8b — Worst per category
    for cat, col in _CAT_COLORS.items():
        bk = buckets[cat]
        if not bk:
            continue
        ys = []
        for i in range(n):
            adm = metrics[i]["admissibility_score"]
            mnv = min((float(adm[kk]) for kk in bk if kk in adm), default=float("nan"))
            ys.append(mnv)
        if any(np.isfinite(ys)):
            fig.add_trace(
                go.Scatter(
                    x=x_idx,
                    y=ys,
                    mode="lines",
                    name=f"min {cat}",
                    line=dict(color=col),
                    hovertemplate=f"min in {cat}<br>step=%{{x}}<br>admissibility=%{{y:.4f}}<extra></extra>",
                ),
                row=8,
                col=2,
            )
    fig.update_yaxes(title_text="Admissibility", range=[0, 1.05], row=8, col=2)

    fig.update_layout(
        title=dict(text="Moju monitor visualization (interactive)", x=0.5, xanchor="center"),
        height=2300,
        showlegend=True,
        barmode="group",
        legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="left", x=1.01),
        hovermode="closest",
        margin=dict(r=120),
    )
    fig.update_xaxes(title_text="Log index / step", row=7, col=1)
    fig.update_xaxes(title_text="Log index / step", row=8, col=2)

    return fig
