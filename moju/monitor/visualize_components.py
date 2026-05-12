"""
Reusable Plotly card builders for the Moju enterprise visualisation suite.

Each ``build_*_card`` function returns a :class:`plotly.graph_objects.Figure`
themed via :func:`moju.monitor.visualize_theme.apply_theme`.  Cards are
deliberately small, composable units: the audit dashboard, the constitutive
divergence card, and Studio pages all assemble their layouts from these
primitives so the look-and-feel is consistent everywhere.

Bundle contract
---------------

Most builders accept a ``bundle`` dict produced by
:func:`moju.monitor.auditor._build_visualize_bundle` /
:func:`moju.monitor.auditor.build_monitor_visualize_bundle`.  Required keys
are documented per builder.

All builders are defensive: they return an empty themed figure with an
informative annotation when required data is missing rather than raising.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from moju.monitor.visualize_theme import (
    MOJU_LIGHT,
    MojuTheme,
    apply_theme,
    get_theme,
    themed_axis_style,
    themed_colorbar,
)
from moju.monitor.visualize_labels import (
    format_admissibility_pct,
    pretty_category_name,
    pretty_residual_key,
    truncate_display_label,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _go():
    """Lazy import so visualization is optional."""
    import plotly.graph_objects as go

    return go


def _empty_card(title: str, message: str, theme: Any = MOJU_LIGHT) -> Any:
    """Themed empty figure used when bundle data is missing."""
    go = _go()
    t = get_theme(theme)
    fig = go.Figure()
    fig.update_layout(
        annotations=[
            dict(
                text=message,
                showarrow=False,
                font=t.font_dict(size=12, color=t.palette.muted),
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
            )
        ],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return apply_theme(fig, theme, title=title)


def _classify_residual_category(flat_key: str) -> str:
    """Map a flat residual key to its category (laws / constitutive / scaling / data / other)."""
    if not isinstance(flat_key, str):
        return "other"
    head = flat_key.split("/", 1)[0]
    if head in ("laws", "constitutive", "scaling", "data"):
        return head
    return "other"


def _category_color(category: str, theme: Any = MOJU_LIGHT) -> str:
    t = get_theme(theme)
    return {
        "laws": t.palette.cat_laws,
        "constitutive": t.palette.cat_constitutive,
        "scaling": t.palette.cat_scaling,
        "data": t.palette.cat_data,
    }.get(category, t.palette.cat_other)


def _admissibility_color(score: Optional[float], theme: Any = MOJU_LIGHT) -> str:
    t = get_theme(theme)
    if score is None or not np.isfinite(score):
        return t.palette.muted
    if score >= 0.80:
        return t.palette.adm_high
    if score >= 0.60:
        return t.palette.adm_med
    if score >= 0.40:
        return t.palette.accent_warn
    return t.palette.adm_low


def _safe_get(bundle: Dict[str, Any], *keys: str) -> Any:
    """Navigate nested dicts safely; return ``None`` on any miss."""
    node: Any = bundle
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node


def _last_log_entry(bundle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    log = bundle.get("log") or []
    if not log:
        return None
    return log[-1]


def _hovertemplate_kpi() -> str:
    return "<b>%{label}</b><br>Score: %{value:.1%}<extra></extra>"


# ---------------------------------------------------------------------------
# Card builders
# ---------------------------------------------------------------------------


def build_overall_admissibility_kpi(
    bundle: Dict[str, Any],
    *,
    theme: Any = MOJU_LIGHT,
    height: Optional[int] = None,
    title: Optional[str] = "Overall Admissibility",
) -> Any:
    """Enterprise KPI scorecard for overall admissibility (0-100%).

    Uses ``go.Indicator`` with themed color thresholds.
    """
    go = _go()
    t = get_theme(theme)
    last = _last_log_entry(bundle)
    if last is None:
        return _empty_card(title or "Overall Admissibility", "No log entries available", theme)
    score = float(last.get("overall_admissibility_score") or 0.0)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score * 100.0,
            number=dict(suffix="%", font=t.font_dict(size=28, color=t.palette.title_color)),
            gauge=dict(
                axis=dict(range=[0, 100], tickfont=t.tick_font_dict(), tickcolor=t.palette.axis_line),
                bar=dict(color=_admissibility_color(score, theme)),
                bgcolor=t.palette.plot_bg,
                bordercolor=t.palette.axis_line,
                steps=[
                    dict(range=[0, 40], color="rgba(239,68,68,0.18)"),
                    dict(range=[40, 60], color="rgba(230,126,34,0.18)"),
                    dict(range=[60, 80], color="rgba(245,158,11,0.18)"),
                    dict(range=[80, 100], color="rgba(16,185,129,0.18)"),
                ],
                threshold=dict(
                    line=dict(color=t.palette.adm_high, width=3),
                    thickness=0.78,
                    value=80,
                ),
            ),
        )
    )
    return apply_theme(fig, theme, title=title, height=height or t.layout.card_height)


def build_admissibility_timeline_card(
    bundle: Dict[str, Any],
    *,
    theme: Any = MOJU_LIGHT,
    height: Optional[int] = None,
    title: str = "Admissibility vs Step",
    show_categories: bool = True,
) -> Any:
    """Overall + per-category admissibility line chart vs training step."""
    go = _go()
    t = get_theme(theme)
    indices: List[int] = bundle.get("indices") or []
    overall: List[float] = bundle.get("overall_adm") or []
    if not indices or not overall:
        return _empty_card(title, "No admissibility timeline data", theme)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=indices,
            y=[v * 100.0 if v is not None else None for v in overall],
            mode="lines+markers",
            name="Overall",
            line=dict(color=t.palette.accent_neutral, width=2.4),
            marker=dict(size=5, color=t.palette.accent_neutral),
            hovertemplate="Step %{x}<br>Overall: %{y:.2f}%<extra></extra>",
        )
    )

    if show_categories:
        cat_training: Dict[str, List[Optional[float]]] = bundle.get("category_training") or {}
        for cat_name, series in cat_training.items():
            if not series:
                continue
            color = _category_color(cat_name, theme)
            fig.add_trace(
                go.Scatter(
                    x=indices,
                    y=[v * 100.0 if v is not None else None for v in series],
                    mode="lines",
                    name=pretty_category_name(cat_name),
                    line=dict(color=color, width=1.8, dash="solid"),
                    hovertemplate=(
                        f"Step %{{x}}<br>{pretty_category_name(cat_name)}: %{{y:.2f}}%<extra></extra>"
                    ),
                )
            )

    fig.update_xaxes(title_text="Step", **themed_axis_style(theme, show_grid=False, zero_line=False))
    fig.update_yaxes(
        title_text="Admissibility (%)",
        range=[0, 100],
        ticksuffix="%",
        **themed_axis_style(theme, show_grid=True, zero_line=False),
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=t.layout.legend_y_top,
            xanchor="right",
            x=t.layout.legend_x_right,
            font=t.tick_font_dict(),
        ),
        hovermode="x unified",
    )
    return apply_theme(fig, theme, title=title, height=height or t.layout.card_height)


def build_category_admissibility_bar_card(
    bundle: Dict[str, Any],
    *,
    theme: Any = MOJU_LIGHT,
    height: Optional[int] = None,
    title: str = "Final Category Admissibility",
) -> Any:
    """Horizontal bar of final-step per-category admissibility."""
    go = _go()
    t = get_theme(theme)
    cats_fin: Dict[str, float] = bundle.get("cats_fin") or {}
    if not cats_fin:
        return _empty_card(title, "No category admissibility data", theme)

    order = ["laws", "constitutive", "scaling", "data"]
    items: List[Tuple[str, float]] = [
        (c, float(cats_fin[c])) for c in order if c in cats_fin and cats_fin[c] is not None
    ]
    for c, v in cats_fin.items():
        if c not in order and v is not None:
            items.append((c, float(v)))
    if not items:
        return _empty_card(title, "No finite category admissibilities", theme)

    labels = [pretty_category_name(c) for c, _ in items]
    values = [v * 100.0 for _, v in items]
    colors = [_category_color(c, theme) for c, _ in items]
    fig = go.Figure(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker=dict(color=colors, line=dict(color=t.palette.bar_line, width=0.5)),
            text=[format_admissibility_pct(v / 100.0) for v in values],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Admissibility: %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_xaxes(
        title_text="Admissibility (%)",
        range=[0, 110],
        ticksuffix="%",
        **themed_axis_style(theme, show_grid=True, zero_line=False),
    )
    fig.update_yaxes(autorange="reversed", **themed_axis_style(theme, show_grid=False, zero_line=False))
    return apply_theme(fig, theme, title=title, height=height or t.layout.card_height)


def build_rnorm_timeline_card(
    bundle: Dict[str, Any],
    *,
    theme: Any = MOJU_LIGHT,
    height: Optional[int] = None,
    title: str = "R_norm vs Step",
    keys: Optional[Sequence[str]] = None,
    log_y: bool = True,
    max_keys: int = 12,
) -> Any:
    """Per-key R_norm timeline.  Auto-picks worst ``max_keys`` if none provided."""
    go = _go()
    t = get_theme(theme)
    plot_keys: Optional[List[str]] = bundle.get("plot_keys")
    r_norm_mat = bundle.get("r_norm_mat")
    indices: List[int] = bundle.get("indices") or []
    if not plot_keys or r_norm_mat is None or not indices:
        return _empty_card(title, "No R_norm timeline data", theme)

    mat = np.asarray(r_norm_mat, dtype=float)  # shape: (steps, keys)
    if mat.ndim != 2 or mat.shape[0] != len(indices) or mat.shape[1] != len(plot_keys):
        return _empty_card(title, "Inconsistent R_norm matrix shape", theme)

    selected: List[str]
    if keys:
        selected = [k for k in keys if k in plot_keys][:max_keys]
    else:
        # Rank by final-step R_norm descending
        final_vals = mat[-1, :]
        sort_idx = np.argsort(-np.where(np.isfinite(final_vals), final_vals, -np.inf))
        selected = [plot_keys[i] for i in sort_idx[:max_keys]]
    if not selected:
        return _empty_card(title, "No keys to plot", theme)

    fig = go.Figure()
    for k in selected:
        col = plot_keys.index(k)
        ys = mat[:, col]
        cat = _classify_residual_category(k)
        color = _category_color(cat, theme)
        fig.add_trace(
            go.Scatter(
                x=indices,
                y=ys,
                mode="lines",
                name=truncate_display_label(pretty_residual_key(k), 42),
                line=dict(color=color, width=1.6),
                hovertemplate=f"<b>{pretty_residual_key(k)}</b><br>Step %{{x}}<br>R_norm: %{{y:.3g}}<extra></extra>",
            )
        )

    fig.update_xaxes(title_text="Step", **themed_axis_style(theme, show_grid=False, zero_line=False))
    fig.update_yaxes(
        title_text="R_norm",
        type="log" if log_y else "linear",
        **themed_axis_style(theme, show_grid=True, zero_line=not log_y),
    )
    fig.update_layout(
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            font=t.tick_font_dict(),
        ),
        hovermode="x unified",
    )
    return apply_theme(fig, theme, title=title, height=height or t.layout.card_height)


def build_law_rnorm_final_bar_card(
    bundle: Dict[str, Any],
    *,
    theme: Any = MOJU_LIGHT,
    height: Optional[int] = None,
    title: str = "Final-step R_norm by Law",
    log_y: bool = True,
) -> Any:
    """Vertical bar chart of final-step R_norm for keys in the ``laws`` category."""
    go = _go()
    t = get_theme(theme)
    plot_keys: Optional[List[str]] = bundle.get("plot_keys")
    r_norm_mat = bundle.get("r_norm_mat")
    if not plot_keys or r_norm_mat is None:
        return _empty_card(title, "No R_norm data", theme)

    mat = np.asarray(r_norm_mat, dtype=float)
    if mat.ndim != 2 or mat.shape[1] != len(plot_keys):
        return _empty_card(title, "Inconsistent R_norm matrix", theme)

    finals = mat[-1, :]
    laws_keys: List[Tuple[str, float]] = [
        (k, float(v))
        for k, v in zip(plot_keys, finals)
        if _classify_residual_category(k) == "laws" and np.isfinite(v)
    ]
    if not laws_keys:
        return _empty_card(title, "No law residual keys", theme)
    laws_keys.sort(key=lambda kv: -kv[1])
    labels = [truncate_display_label(pretty_residual_key(k), 38) for k, _ in laws_keys]
    values = [v for _, v in laws_keys]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(color=t.palette.cat_laws, line=dict(color=t.palette.bar_line, width=0.5)),
            hovertemplate="<b>%{x}</b><br>R_norm: %{y:.3g}<extra></extra>",
        )
    )
    fig.update_xaxes(
        title_text="Law residual",
        tickangle=-30 if len(labels) > 6 else 0,
        **themed_axis_style(theme, show_grid=False, zero_line=False),
    )
    fig.update_yaxes(
        title_text="R_norm (final step)",
        type="log" if log_y else "linear",
        **themed_axis_style(theme, show_grid=True, zero_line=not log_y),
    )
    return apply_theme(fig, theme, title=title, height=height or t.layout.card_height)


def build_spatial_residual_heatmap_card(
    bundle: Dict[str, Any],
    key: str,
    *,
    theme: Any = MOJU_LIGHT,
    height: Optional[int] = None,
    title: Optional[str] = None,
    log_z: bool = True,
    colorscale: Optional[str] = None,
) -> Any:
    """Spatial heatmap of |residual| for a single key (auto-selects 2D layout)."""
    go = _go()
    t = get_theme(theme)
    spatial: Dict[str, Any] = bundle.get("spatial") or {}
    residuals: Dict[str, Any] = bundle.get("residuals") or {}
    arr = residuals.get(key)
    if arr is None:
        return _empty_card(title or key, f"No residual array for {key!r}", theme)
    a = np.asarray(arr, dtype=float)
    if a.ndim == 0:
        return _empty_card(title or key, "Residual is a scalar; nothing to map", theme)
    coords = spatial.get("coords") or {}
    x = coords.get("x")
    y = coords.get("y")
    if a.ndim == 1:
        xs = np.asarray(x) if x is not None else np.arange(a.shape[0])
        if xs.shape[0] != a.shape[0]:
            xs = np.arange(a.shape[0])
        fig = go.Figure(
            go.Scatter(
                x=xs,
                y=np.abs(a),
                mode="lines",
                line=dict(color=_category_color(_classify_residual_category(key), theme), width=1.8),
                hovertemplate="x=%{x:.3g}<br>|r|=%{y:.3g}<extra></extra>",
            )
        )
        fig.update_xaxes(title_text="x", **themed_axis_style(theme))
        fig.update_yaxes(
            title_text="|residual|",
            type="log" if log_z else "linear",
            **themed_axis_style(theme),
        )
        return apply_theme(fig, theme, title=title or pretty_residual_key(key), height=height or t.layout.card_height)

    # 2-D: squeeze leading singletons
    a2 = a
    while a2.ndim > 2 and a2.shape[0] == 1:
        a2 = a2[0]
    if a2.ndim != 2:
        return _empty_card(title or key, f"Cannot render ndim={a2.ndim} residual as heatmap", theme)
    z = np.log10(np.abs(a2) + 1e-12) if log_z else np.abs(a2)
    cs = colorscale or t.colorscales.sequential
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=np.asarray(x) if x is not None and len(x) == a2.shape[1] else None,
            y=np.asarray(y) if y is not None and len(y) == a2.shape[0] else None,
            colorscale=cs,
            colorbar=themed_colorbar(theme, title="log10(|r| + ε)" if log_z else "|r|"),
            hovertemplate="x=%{x:.3g}<br>y=%{y:.3g}<br>z=%{z:.3g}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="x", **themed_axis_style(theme, show_grid=False, zero_line=False))
    fig.update_yaxes(title_text="y", **themed_axis_style(theme, show_grid=False, zero_line=False))
    return apply_theme(fig, theme, title=title or pretty_residual_key(key), height=height or t.layout.card_height)


def build_field_explorer_card(
    field: Any,
    *,
    title: str,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    z: Optional[np.ndarray] = None,
    t_coord: Optional[np.ndarray] = None,
    time_index: Optional[int] = None,
    time_axis: int = 0,
    spatial_view: str = "auto",
    colorscale: Optional[str] = None,
    theme: Any = MOJU_LIGHT,
    height: Optional[int] = None,
) -> Any:
    """
    Generic enterprise-themed field viewer.

    Behaves like the legacy ``plotly_residual_or_state`` but with the unified
    theme and perceptually uniform colorscale defaults.  Auto-selects:

    - 1-D array → line chart
    - 2-D array → heatmap
    - 3-D array → surface (``spatial_view="surface3d"``) or volume
      (``spatial_view="volume3d"``); ``"auto"`` chooses surface for 2-D fields,
      volume for 3-D.
    - higher-D → histogram of values
    """
    go = _go()
    t = get_theme(theme)
    cs = colorscale or t.colorscales.sequential
    a0 = np.asarray(field, dtype=float)
    if a0.size == 0:
        return _empty_card(title, "Empty array", theme)

    # Time slice if requested
    if time_index is not None and a0.ndim >= 1 and a0.shape[time_axis] > 0:
        idx = min(int(time_index), a0.shape[time_axis] - 1)
        a0 = np.take(a0, idx, axis=time_axis)
    # Strip leading singleton dims
    while a0.ndim > 2 and a0.shape[0] == 1:
        a0 = a0[0]

    if spatial_view == "surface3d" and a0.ndim == 2 and x is not None and y is not None:
        return _build_surface(a0, x, y, title, cs, theme, height)
    if spatial_view == "volume3d" and a0.ndim == 3 and x is not None and y is not None and z is not None:
        return _build_volume(a0, x, y, z, title, cs, theme, height)

    if a0.ndim == 1:
        xs = np.asarray(x) if x is not None and len(x) == a0.shape[0] else np.arange(a0.shape[0])
        fig = go.Figure(
            go.Scatter(
                x=xs,
                y=a0,
                mode="lines",
                line=dict(color=t.palette.line_primary, width=1.8),
                hovertemplate="x=%{x:.3g}<br>y=%{y:.3g}<extra></extra>",
            )
        )
        fig.update_xaxes(title_text="x", **themed_axis_style(theme))
        fig.update_yaxes(title_text="value", **themed_axis_style(theme))
        return apply_theme(fig, theme, title=title, height=height or t.layout.card_height)

    if a0.ndim == 2:
        xa = np.asarray(x) if x is not None and len(x) == a0.shape[1] else None
        ya = np.asarray(y) if y is not None and len(y) == a0.shape[0] else None
        fig = go.Figure(
            go.Heatmap(
                z=a0,
                x=xa,
                y=ya,
                colorscale=cs,
                colorbar=themed_colorbar(theme, title="value"),
                hovertemplate="x=%{x:.3g}<br>y=%{y:.3g}<br>z=%{z:.3g}<extra></extra>",
            )
        )
        fig.update_xaxes(title_text="x" if xa is not None else "j", **themed_axis_style(theme, show_grid=False, zero_line=False))
        fig.update_yaxes(title_text="y" if ya is not None else "i", **themed_axis_style(theme, show_grid=False, zero_line=False))
        return apply_theme(fig, theme, title=title, height=height or t.layout.card_height)

    if a0.ndim == 3 and x is not None and y is not None and z is not None and spatial_view in ("auto", "volume3d"):
        return _build_volume(a0, x, y, z, title, cs, theme, height)

    # Fallback: histogram for high-D / unmatched shapes
    fig = go.Figure(go.Histogram(x=a0.ravel(), nbinsx=80, marker=dict(color=t.palette.line_primary)))
    fig.update_xaxes(title_text="value", **themed_axis_style(theme))
    fig.update_yaxes(title_text="count", **themed_axis_style(theme))
    return apply_theme(fig, theme, title=f"{title} (histogram, ndim={a0.ndim})", height=height or t.layout.card_height)


def _build_surface(
    z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    colorscale: str,
    theme: Any,
    height: Optional[int],
) -> Any:
    go = _go()
    t = get_theme(theme)
    x1 = np.asarray(x, dtype=float).ravel()
    y1 = np.asarray(y, dtype=float).ravel()
    if z.shape == (y1.shape[0], x1.shape[0]):
        z_surf = z
    elif z.shape == (x1.shape[0], y1.shape[0]):
        z_surf = z.T
    else:
        return _empty_card(title, "Surface shape does not match x/y", theme)
    fig = go.Figure(
        go.Surface(
            x=x1,
            y=y1,
            z=z_surf,
            colorscale=colorscale,
            colorbar=themed_colorbar(theme, title="value"),
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="x", title_font=t.section_title_font_dict(), tickfont=t.tick_font_dict()),
            yaxis=dict(title="y", title_font=t.section_title_font_dict(), tickfont=t.tick_font_dict()),
            zaxis=dict(title="value", title_font=t.section_title_font_dict(), tickfont=t.tick_font_dict()),
            aspectmode="data",
        )
    )
    return apply_theme(fig, theme, title=title, height=height or t.layout.card_height)


def _build_volume(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str,
    colorscale: str,
    theme: Any,
    height: Optional[int],
    max_voxels: int = 125_000,
) -> Any:
    go = _go()
    t = get_theme(theme)
    x1 = np.asarray(x, dtype=float).ravel()
    y1 = np.asarray(y, dtype=float).ravel()
    z1 = np.asarray(z, dtype=float).ravel()
    if field.shape != (x1.shape[0], y1.shape[0], z1.shape[0]):
        return _empty_card(title, "Volume shape does not match x/y/z", theme)
    if field.size > max_voxels:
        return _empty_card(title, f"Field too large for volume ({field.size} > {max_voxels} voxels)", theme)
    X, Y, Z = np.meshgrid(x1, y1, z1, indexing="ij")
    flat = field.ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return _empty_card(title, "No finite values", theme)
    v0, v1 = float(np.nanpercentile(finite, 5)), float(np.nanpercentile(finite, 95))
    if not np.isfinite(v0) or not np.isfinite(v1) or v0 == v1:
        v0, v1 = float(np.nanmin(finite)), float(np.nanmax(finite))
    if v0 == v1:
        v1 = v0 + 1.0
    fig = go.Figure(
        go.Volume(
            x=X.ravel(),
            y=Y.ravel(),
            z=Z.ravel(),
            value=flat,
            isomin=v0,
            isomax=v1,
            opacity=0.12,
            surface_count=18,
            colorscale=colorscale,
            colorbar=themed_colorbar(theme, title="value"),
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="x", title_font=t.section_title_font_dict(), tickfont=t.tick_font_dict()),
            yaxis=dict(title="y", title_font=t.section_title_font_dict(), tickfont=t.tick_font_dict()),
            zaxis=dict(title="z", title_font=t.section_title_font_dict(), tickfont=t.tick_font_dict()),
            aspectmode="data",
        )
    )
    return apply_theme(fig, theme, title=title, height=height or t.layout.card_height)


def build_worst_keys_table_card(
    bundle: Dict[str, Any],
    *,
    theme: Any = MOJU_LIGHT,
    height: Optional[int] = None,
    title: str = "Worst Residual Keys",
    limit: int = 12,
) -> Any:
    """Themed ``go.Table`` listing the worst residual keys by R_norm."""
    go = _go()
    t = get_theme(theme)
    rows: List[Dict[str, Any]] = bundle.get("worst_keys_rows") or []
    if not rows:
        return _empty_card(title, "No worst-key rows available", theme)
    rows = rows[:limit]
    header_fill = "#1f2937" if t.name == "dark" else "#0f172a"
    header_font = dict(family=t.typography.font_family, color="#ffffff", size=12)
    cell_font = t.tick_font_dict()
    columns = ["Key", "Category", "R_norm", "Admissibility"]
    keys = [pretty_residual_key(r.get("key") or "") for r in rows]
    cats = [pretty_category_name(_classify_residual_category(r.get("key") or "")) for r in rows]
    rnorms = [f"{float(r.get('r_norm') or 0.0):.3g}" for r in rows]
    adms = [format_admissibility_pct(float(r.get("admissibility_score") or 0.0)) for r in rows]
    cell_colors = [
        ["#0b1220" if (i % 2 == 0 and t.name == "dark") else (t.palette.plot_bg if i % 2 == 0 else t.palette.summary_bg) for i in range(len(rows))]
    ] * len(columns)
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[f"<b>{c}</b>" for c in columns],
                    fill_color=header_fill,
                    font=header_font,
                    align="left",
                    height=32,
                ),
                cells=dict(
                    values=[keys, cats, rnorms, adms],
                    fill_color=cell_colors,
                    font=cell_font,
                    align="left",
                    height=26,
                ),
            )
        ]
    )
    return apply_theme(fig, theme, title=title, height=height or t.layout.card_height)


__all__ = [
    "build_overall_admissibility_kpi",
    "build_admissibility_timeline_card",
    "build_category_admissibility_bar_card",
    "build_rnorm_timeline_card",
    "build_law_rnorm_final_bar_card",
    "build_spatial_residual_heatmap_card",
    "build_field_explorer_card",
    "build_worst_keys_table_card",
]
