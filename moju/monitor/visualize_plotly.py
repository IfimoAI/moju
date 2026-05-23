"""
Interactive Plotly dashboard for :func:`moju.monitor.auditor.visualize`.

Admissibility colors and status bands follow :func:`moju.monitor.auditor.admissibility_level` /
:data:`moju.monitor.auditor.ADM_HIGH_THRESHOLD`.

Requires Plotly (included in core ``moju``).
"""

from __future__ import annotations

import math
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from moju.monitor.auditor import (
    ADM_HIGH_THRESHOLD,
    ADM_LOW_THRESHOLD,
    ADM_MODERATE_THRESHOLD,
    admissibility_level,
    is_high_admissibility,
)
from moju.monitor.constitutive_closure_summary import build_constitutive_closure_summary
from moju.monitor.visualize_labels import (
    category_adm_bar_axis_range_percent_full,
    format_admissibility_pct,
    pretty_category_name,
    truncate_display_label,
)
from moju.monitor.visualize_constitutive import (
    build_spatial_normalized_divergence_figure,
    constitutive_divergence_title_for_bundle,
    infer_divergence_abscissa,
    prepare_constitutive_model_implied_vs_x_embed,
    primary_closure_debug_field_length,
)
from moju.monitor.visualize_theme import MOJU_LIGHT, apply_theme, get_theme

R_NORM_LOG_EPS = 1e-12
# Spatial heatmap colorbar: matches plotted z (log10(|residual| + ε)) and hover `z` in display space.
SPATIAL_HEATMAP_COLORBAR_TITLE_LOG = "log10(|residual| + ε)"
# Perceptually uniform default (avoid Jet for scientific heatmaps).
DEFAULT_SPATIAL_HEATMAP_COLORSCALE = "Viridis"

# Fixed height for Moju Studio Dashboard Plotly cards (law/adm bars + spatial heatmaps).
MOJU_STUDIO_DASHBOARD_CARD_HEIGHT = 400

# Single-figure visualize: eval canvas height; training scales up so each chart row matches eval's
# pixel height (training has one extra chart row with the same relative weight as eval's lower chart row).
MONITOR_SINGLE_FIGURE_HEIGHT = 1040
_MONITOR_ROW_HEIGHT_SUM_EVAL = 0.002 + 0.074 + 0.262 + 0.222
_MONITOR_ROW_HEIGHT_SUM_TRAINING = 0.002 + 0.074 + 0.262 + 0.222 + 0.222
MONITOR_SINGLE_FIGURE_HEIGHT_TRAINING = int(
    math.ceil(
        MONITOR_SINGLE_FIGURE_HEIGHT * _MONITOR_ROW_HEIGHT_SUM_TRAINING / _MONITOR_ROW_HEIGHT_SUM_EVAL
    )
)
# Summary box below subplots (``yref="paper"``, ``yanchor="top"``): higher y = closer to charts; margin_b reserves pixels so the border is not clipped (training + eval).
MONITOR_SINGLE_FIGURE_MARGIN_BOTTOM = 268
MONITOR_SUMMARY_ANNOTATION_Y_PAPER = -0.098

MONITOR_CONSTITUTIVE_DIVERGENCE_EXTRA_PX = 280
MONITOR_DIV_ROW_WEIGHT_MULT = 3.0
MONITOR_STATE_OVERLAY_EXTRA_PX = 144

# Heatmap colorbars: paper coordinates; positioned via align_heatmap_colorbars_to_subplot_domains.
HEATMAP_COLORBAR_X_GAP_PAPER = 0.012
HEATMAP_COLORBAR_LEN_FRAC = 0.88
HEATMAP_COLORBAR_LEN_MIN = 0.06
HEATMAP_COLORBAR_X_RIGHT_CAP = 0.995
HEATMAP_COLORBAR_THICKNESS = 12
HEATMAP_COLORBAR_XPAD = 8
# Inset each heatmap subplot x-domain so the colorbar (paper-x anchored) sits beside the heatmap in-cell.
HEATMAP_COLORBAR_DOMAIN_RESERVE_PAPER = 0.024
HEATMAP_COLORBAR_X_DOMAIN_MIN_WIDTH = 0.12

RESIDUAL_COLOR_LAWS = "#8B5CF6"  # royal purple
RESIDUAL_COLOR_CONSTITUTIVE = "#14B8A6"  # teal
# Aligned with bundle cat_colors in auditor._build_visualize_bundle
RESIDUAL_COLOR_SCALING = "#59a14f"
RESIDUAL_COLOR_DATA = "#b07aa1"
RESIDUAL_COLOR_OTHER = "#6b7280"  # neutral fallback
ADMISSIBLE_COLOR = "#10B981"
DISSONANCE_COLOR = "#EF4444"
# Training Overall Admissibility vs-step line (single-figure): black on white plotly_white panel.
OVERALL_ADMISSIBILITY_TREND_LINE_COLOR = "#000000"
LOW_ADM_COLOR = "#e67e22"  # between moderate (amber) and non-admissible (red)
# Horizontal category-breakdown bars: fraction of category spacing (Plotly default ~0.8); higher = thicker bars, tighter vertical gaps.
CATEGORY_BREAKDOWN_BAR_WIDTH = 0.46
# Y (axis domain, 0=bottom 1=top): Primary Issue label above the plot so it does not cover bars.
PRIMARY_ISSUE_ANNOTATION_Y_DOMAIN = 1.08
# KPI ``go.Indicator`` domain x-span inside each subplot cell (symmetric → centered score stack).
KPI_INDICATOR_DOMAIN_X = (0.10, 0.90)


def _wrap_category_tick_label_html(label: str, *, max_line_chars: int = 14, max_lines: int = 4) -> str:
    """Multi-line category names on the y-axis using Plotly HTML (<br>); preserves existing breaks."""
    s = (label or "").strip()
    if not s:
        return s
    if "<br>" in s.lower():
        return s
    if len(s) <= max_line_chars:
        return s
    words = s.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        cand = f"{cur} {w}".strip() if cur else w
        if len(cand) <= max_line_chars:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            if len(w) <= max_line_chars:
                cur = w
            else:
                for i in range(0, len(w), max_line_chars):
                    lines.append(w[i : i + max_line_chars])
                cur = ""
            if len(lines) >= max_lines:
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    return "<br>".join(lines[:max_lines])


def _enterprise_theme_dict() -> Dict[str, str]:
    """Light theme tokens aligned with :data:`MOJU_LIGHT` (single source of truth)."""
    t = get_theme("light")
    p = t.palette
    return {
        "plot_bg": p.plot_bg,
        "paper_bg": p.paper_bg,
        "font_stack": t.typography.font_family,
        "font_color": p.font_color,
        "muted": p.muted,
        "tick_color": p.tick_color,
        "title_color": p.title_color,
        "axis_line": p.axis_line,
        "grid_color": p.grid_color,
        "zeroline_color": p.zeroline_color,
        "line_primary": p.line_primary,
        "bar_line": p.bar_line,
        "summary_border": p.summary_border,
        "summary_bg": p.summary_bg,
    }


_ENTERPRISE_THEME: Dict[str, str] = _enterprise_theme_dict()


def _require_light_theme(theme: str) -> None:
    if theme != "light":
        raise ValueError(
            "visualize Plotly styling supports theme='light' only; dark mode is no longer supported."
        )


def _apply_enterprise_axis_style(
    fig: Any,
    row: int,
    col: int,
    *,
    y_log: bool = False,
    x_grid: bool = False,
    y_grid: bool = True,
) -> None:
    """Major grids, softer axis frame, and typography for cartesian training/test panels."""
    T = _ENTERPRISE_THEME
    tick_font = dict(family=T["font_stack"], size=11, color=T["tick_color"])
    title_font = dict(family=T["font_stack"], size=13, color=T["title_color"])
    fig.update_xaxes(
        row=row,
        col=col,
        showline=True,
        linewidth=1,
        mirror=True,
        linecolor=T["axis_line"],
        tickcolor=T["axis_line"],
        showgrid=x_grid,
        gridcolor=T["grid_color"],
        gridwidth=1,
        minor_showgrid=False,
        zeroline=False,
        tickfont=tick_font,
        title_font=title_font,
        automargin=True,
    )
    fig.update_yaxes(
        row=row,
        col=col,
        showline=True,
        linewidth=1,
        mirror=True,
        linecolor=T["axis_line"],
        tickcolor=T["axis_line"],
        showgrid=y_grid,
        gridcolor=T["grid_color"],
        gridwidth=1,
        minor_showgrid=False,
        zeroline=not y_log,
        zerolinecolor=T["zeroline_color"],
        tickfont=tick_font,
        title_font=title_font,
        automargin=True,
    )


def _apply_enterprise_spatial_axes(fig: Any, row: int, col: int, *, hide_y_ticklabels: bool = False) -> None:
    """Axis frame for heatmap subplots (no grid)."""
    T = _ENTERPRISE_THEME
    tick_font = dict(family=T["font_stack"], size=11, color=T["tick_color"])
    title_font = dict(family=T["font_stack"], size=13, color=T["title_color"])
    fig.update_xaxes(
        row=row,
        col=col,
        showline=True,
        linewidth=1,
        mirror=True,
        linecolor=T["axis_line"],
        tickcolor=T["axis_line"],
        showgrid=False,
        minor_showgrid=False,
        zeroline=False,
        tickfont=tick_font,
        title_font=title_font,
        automargin=True,
    )
    y_kw: Dict[str, Any] = dict(
        row=row,
        col=col,
        showline=True,
        linewidth=1,
        mirror=True,
        linecolor=T["axis_line"],
        tickcolor=T["axis_line"],
        showgrid=False,
        minor_showgrid=False,
        zeroline=False,
        tickfont=tick_font,
        title_font=title_font,
        automargin=True,
    )
    if hide_y_ticklabels:
        y_kw["showticklabels"] = False
    fig.update_yaxes(**y_kw)


def _apply_enterprise_axis_style_xy(
    fig: Any,
    *,
    y_log: bool = False,
    x_grid: bool = True,
    y_grid: bool = True,
) -> None:
    """Major grids and axis frame on a single-panel :class:`plotly.graph_objects.Figure`."""
    T = _ENTERPRISE_THEME
    tick_font = dict(family=T["font_stack"], size=11, color=T["tick_color"])
    title_font = dict(family=T["font_stack"], size=13, color=T["title_color"])
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        mirror=True,
        linecolor=T["axis_line"],
        tickcolor=T["axis_line"],
        showgrid=x_grid,
        gridcolor=T["grid_color"],
        gridwidth=1,
        minor_showgrid=False,
        zeroline=False,
        tickfont=tick_font,
        title_font=title_font,
        automargin=True,
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        mirror=True,
        linecolor=T["axis_line"],
        tickcolor=T["axis_line"],
        showgrid=y_grid,
        gridcolor=T["grid_color"],
        gridwidth=1,
        minor_showgrid=False,
        zeroline=not y_log,
        zerolinecolor=T["zeroline_color"],
        tickfont=tick_font,
        title_font=title_font,
        automargin=True,
    )


def _enterprise_axis_frame_xy(fig: Any, *, grid: bool = False, hide_y_ticklabels: bool = False) -> None:
    """Softer axis frame for a single-panel figure (optional grid; heatmaps use ``grid=False``)."""
    T = _ENTERPRISE_THEME
    tick_font = dict(family=T["font_stack"], size=11, color=T["tick_color"])
    title_font = dict(family=T["font_stack"], size=13, color=T["title_color"])
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        mirror=True,
        linecolor=T["axis_line"],
        tickcolor=T["axis_line"],
        showgrid=grid,
        gridcolor=T["grid_color"],
        gridwidth=1,
        minor_showgrid=False,
        zeroline=False,
        tickfont=tick_font,
        title_font=title_font,
        automargin=True,
    )
    y_kw: Dict[str, Any] = dict(
        showline=True,
        linewidth=1,
        mirror=True,
        linecolor=T["axis_line"],
        tickcolor=T["axis_line"],
        showgrid=grid,
        gridcolor=T["grid_color"],
        gridwidth=1,
        minor_showgrid=False,
        zeroline=False,
        tickfont=tick_font,
        title_font=title_font,
        automargin=True,
    )
    if hide_y_ticklabels:
        y_kw["showticklabels"] = False
    fig.update_yaxes(**y_kw)


def _residual_color_from_key(flat_key: str) -> str:
    key = str(flat_key or "")
    if key.startswith("laws/"):
        return RESIDUAL_COLOR_LAWS
    if key.startswith("constitutive/"):
        return RESIDUAL_COLOR_CONSTITUTIVE
    if key.startswith("scaling/") or key.startswith("groups/"):
        return RESIDUAL_COLOR_SCALING
    if key.startswith("data/"):
        return RESIDUAL_COLOR_DATA
    return RESIDUAL_COLOR_OTHER


def _adm_bar_color_plotly(score: float) -> str:
    if not math.isfinite(score):
        return "#bdc3c7"
    if is_high_admissibility(score):
        return "#27ae60"
    if score >= ADM_MODERATE_THRESHOLD:
        return "#F59E0B"
    if score >= ADM_LOW_THRESHOLD:
        return LOW_ADM_COLOR
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


def _admissibility_status_bracket_plotly(score: float) -> str:
    """Short status tag for the Plotly header bracket (matches :func:`admissibility_level` bands)."""
    if not math.isfinite(score):
        return "N/A"
    if is_high_admissibility(score):
        return "HIGH"
    if score >= ADM_MODERATE_THRESHOLD:
        return "MODERATE"
    if score >= ADM_LOW_THRESHOLD:
        return "LOW"
    return "NON-ADM"


def format_admissibility_status_label(score: float) -> str:
    """
    Human-readable admissibility band for captions (e.g. Moju Studio).

    Uses the same thresholds as :func:`moju.monitor.auditor.admissibility_level`; returns ``N/A`` when
    the score is non-finite.
    """
    if not math.isfinite(score):
        return "N/A"
    return admissibility_level(score)


def _status_bracket_color(tag: str, *, warn_color: str) -> str:
    if tag == "HIGH":
        return ADMISSIBLE_COLOR
    if tag == "MODERATE":
        return warn_color
    if tag == "LOW":
        return LOW_ADM_COLOR
    return DISSONANCE_COLOR


def _go_category_kpi_indicator(
    value: float,
    title: str,
    ref: Optional[float],
    *,
    font_color: str,
    warn_color: str,
    title_px: int = 13,
    num_px: int = 22,
    delta_px: int = 9,
) -> Any:
    """Single ``go.Indicator`` for Governing / Constitutive / Scaling scorecards."""
    import plotly.graph_objects as go

    T = _ENTERPRISE_THEME
    v = float(value) if math.isfinite(float(value)) else 0.0
    card_color = _kpi_indicator_value_color(v, warn_color=warn_color)
    has_delta = ref is not None and math.isfinite(float(ref))
    num_size = num_px - 2 if has_delta else num_px
    domain_y = [0.07, 0.93] if has_delta else [0.10, 0.90]
    _x0, _x1 = KPI_INDICATOR_DOMAIN_X
    ind_kwargs: Dict[str, Any] = dict(
        align="center",
        mode=("number+delta" if has_delta else "number"),
        value=v,
        number={
            "valueformat": ".1%",
            "font": {"size": num_size, "color": card_color},
        },
        title={
            "text": title,
            "align": "center",
            "font": {"family": T["font_stack"], "size": title_px, "color": font_color},
        },
        domain=dict(x=[_x0, _x1], y=domain_y),
    )
    ind = go.Indicator(**ind_kwargs)
    if has_delta:
        ind.delta = {
            "reference": float(ref),
            "increasing": {"color": ADMISSIBLE_COLOR},
            "decreasing": {"color": DISSONANCE_COLOR},
            "valueformat": ".1%",
            "position": "bottom",
            "font": {"size": delta_px, "family": T["font_stack"]},
        }
    return ind


def _kpi_indicator_value_color(score: float, *, warn_color: str) -> str:
    if not math.isfinite(score):
        return DISSONANCE_COLOR
    if is_high_admissibility(score):
        return ADMISSIBLE_COLOR
    if score >= ADM_MODERATE_THRESHOLD:
        return warn_color
    if score >= ADM_LOW_THRESHOLD:
        return LOW_ADM_COLOR
    return DISSONANCE_COLOR


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


def _heatmap_trace_subplot_meta(tr: Any) -> Optional[Tuple[int, int]]:
    m = getattr(tr, "meta", None)
    if not isinstance(m, dict):
        return None
    r, c = m.get("subplot_row"), m.get("subplot_col")
    if r is None or c is None:
        return None
    try:
        return int(r), int(c)
    except (TypeError, ValueError):
        return None


def _inset_heatmap_x_domain_for_colorbar(fig: Any, *, x0: float, x1: float, row: Optional[int], col: Optional[int]) -> None:
    new_x1 = max(x0 + HEATMAP_COLORBAR_X_DOMAIN_MIN_WIDTH, x1 - HEATMAP_COLORBAR_DOMAIN_RESERVE_PAPER)
    if new_x1 >= x1 - 1e-9:
        return
    if row is not None and col is not None:
        try:
            fig.update_xaxes(domain=[x0, new_x1], row=row, col=col)
            return
        except Exception:
            pass
    # Never call ``fig.update_xaxes(domain=...)`` without row/col — that resets **all** subplots and
    # makes adjacent panels (e.g. law vs constitutive) overlap. Traces must set
    # ``meta=dict(subplot_row=..., subplot_col=...)`` (see spatial builders + monitor embed).


def align_heatmap_colorbars_to_subplot_domains(fig: Any) -> None:
    """
    Inset each heatmap subplot's x-domain (pass A) so the colorbar fits in the cell's horizontal
    footprint—same reserve + gap for left and right columns. Then anchor each colorbar to the
    inset x-domain right edge + gap (pass B). Call after ``fig.update_layout`` so domains are final.
    """
    heat_traces = [
        tr
        for tr in fig.data
        if getattr(tr, "type", None) == "heatmap" and getattr(tr, "colorbar", None) is not None
    ]
    for tr in heat_traces:
        xref = getattr(tr, "xaxis", None) or "x"
        raw_xdom = _plotly_layout_axis_domain(fig, xref)
        xdom = raw_xdom if raw_xdom is not None else (0.0, 1.0)
        x0, x1 = xdom
        meta = _heatmap_trace_subplot_meta(tr)
        if meta is not None:
            row, col = meta
            _inset_heatmap_x_domain_for_colorbar(fig, x0=x0, x1=x1, row=row, col=col)
        else:
            _inset_heatmap_x_domain_for_colorbar(fig, x0=x0, x1=x1, row=None, col=None)

    for tr in heat_traces:
        cb = getattr(tr, "colorbar", None)
        if cb is None:
            continue
        yref = getattr(tr, "yaxis", None) or "y"
        xref = getattr(tr, "xaxis", None) or "x"
        ydom = _plotly_layout_axis_domain(fig, yref)
        xdom = _plotly_layout_axis_domain(fig, xref)
        if ydom is None:
            ydom = (0.0, 1.0)
        if xdom is None:
            xdom = (0.0, 1.0)
        y0, y1 = ydom
        _x0, x1 = xdom
        y_mid = 0.5 * (y0 + y1)
        y_len = max(HEATMAP_COLORBAR_LEN_MIN, (y1 - y0) * HEATMAP_COLORBAR_LEN_FRAC)
        x_cb = min(x1 + HEATMAP_COLORBAR_X_GAP_PAPER, HEATMAP_COLORBAR_X_RIGHT_CAP)
        thick = getattr(cb, "thickness", None) or HEATMAP_COLORBAR_THICKNESS
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
            xpad=getattr(cb, "xpad", None) or HEATMAP_COLORBAR_XPAD,
        )
        if title_dict:
            cb_kwargs["title"] = title_dict
        prev_cb = getattr(tr, "colorbar", None)
        if prev_cb is not None and hasattr(prev_cb, "to_plotly_json"):
            merged = dict(prev_cb.to_plotly_json())
            merged.update(cb_kwargs)
            tr.update(colorbar=merged)
        else:
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


def _drop_paper_subplot_title_annots_with_text(fig: Any, texts: FrozenSet[str]) -> None:
    """Remove make_subplots-generated subplot titles (paper xref/yref) whose text matches."""
    kept: List[Any] = []
    for ann in list(fig.layout.annotations):
        if (
            str(getattr(ann, "xref", "") or "") == "paper"
            and str(getattr(ann, "yref", "") or "") == "paper"
            and str(getattr(ann, "text", "") or "").strip() in texts
        ):
            continue
        kept.append(ann)
    fig.layout.annotations = tuple(kept)


def _plotly_spatial_panel_title_with_subtitle(main: str, rnorm_y_title: str) -> str:
    """Two-line subplot title: main + R_norm scale as smaller second line (HTML)."""
    esc = (
        str(rnorm_y_title)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return f"{main}<br><span style='font-size:10px'>{esc}</span>"


def _finite_z_range_from_array(arr: Any) -> Optional[Tuple[float, float]]:
    """Return (zmin, zmax) for finite values, expanded slightly if degenerate; None if no finite data."""
    import numpy as np

    a = np.asarray(arr, dtype=float)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return None
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if lo == hi:
        eps = 1e-9 if abs(lo) < 1.0 else abs(lo) * 1e-9
        lo, hi = lo - eps, hi + eps
    return lo, hi


def _spatial_panel_plotted_display_z_flat(spatial: Dict[str, Any], *, use_log_rnorm: bool) -> Any:
    """1D float array of z values exactly as plotted by _plotly_add_spatial_panel_to_subplot (same slice/transform)."""
    import numpy as np

    def _z_display(z: Any) -> Any:
        a = np.asarray(z, dtype=float)
        if use_log_rnorm:
            return np.log10(np.maximum(a, 0.0) + R_NORM_LOG_EPS)
        return a

    kind = spatial.get("kind", "1d")
    if kind == "1d":
        return np.asarray(_z_display(spatial["Z"]), dtype=float).ravel()
    if kind == "2d":
        Zs = np.asarray(spatial["Z"], dtype=float)
        return np.asarray(_z_display(Zs[0]), dtype=float).ravel()
    if kind == "3d":
        V = np.asarray(spatial["V"], dtype=float)
        vol = np.asarray(_z_display(V[0]), dtype=float)
        nz = int(vol.shape[2])
        kz = max(0, nz // 2)
        sl = vol[:, :, kz].T
        return np.asarray(sl, dtype=float).ravel()
    return np.array([], dtype=float)


def _heatmap_zlim_kwargs(z_range: Optional[Tuple[float, float]]) -> Dict[str, Any]:
    if z_range is None:
        return {}
    lo, hi = z_range
    return {"zmin": lo, "zmax": hi}


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
    colorbar_scale_title: str = SPATIAL_HEATMAP_COLORBAR_TITLE_LOG,
    z_range: Optional[Tuple[float, float]] = None,
) -> None:
    """Add law or constitutive spatial trace (1D keys×x, 2D x×y, or 3D z-slice) to a subplot cell."""
    import numpy as np
    import plotly.graph_objects as go

    kind = spatial.get("kind", "1d")
    row_labels: List[str] = list(spatial["row_labels"])
    zlim = _heatmap_zlim_kwargs(z_range)

    cb_kw: Dict[str, Any] = dict(
        title=dict(text=colorbar_scale_title, side="right"),
        thickness=HEATMAP_COLORBAR_THICKNESS,
        xpad=HEATMAP_COLORBAR_XPAD,
    )

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
        _z1 = colorbar_scale_title
        # customdata = residual key name; z is in display space (log or linear).
        ht_1d = (
            f"x=%{{x:.4g}}<br>%{{customdata}} ({_z1})=%{{z:.4g}}<br>log step "
            + str(int(li))
            + "<extra></extra>"
            if li is not None
            else f"x=%{{x:.4g}}<br>%{{customdata}} ({_z1})=%{{z:.4g}}<extra></extra>"
        )
        fig.add_trace(
            go.Heatmap(
                x=x_sp,
                y=list(range(len(row_labels))),
                z=Z,
                colorscale=hm_cs,
                colorbar=cb_kw,
                hovertemplate=ht_1d,
                meta=dict(subplot_row=row, subplot_col=col),
                customdata=np.broadcast_to(
                    np.asarray(row_labels, dtype=object)[:, np.newaxis],
                    (len(row_labels), len(x_sp)),
                ),
                **zlim,
            ),
            row=row,
            col=col,
        )
        _apply_enterprise_spatial_axes(fig, row, col, hide_y_ticklabels=True)
        fig.update_xaxes(title_text=f"Position {pos_ax}", row=row, col=col)
        return

    if kind == "2d":
        Zs = np.asarray(spatial["Z"], dtype=float)
        x_sp = np.asarray(spatial["x"], dtype=float)
        y_sp = np.asarray(spatial["y"], dtype=float)
        z0 = _z_display(Zs[0])
        nk = int(Zs.shape[0])
        hl = row_labels[0] if row_labels else ""
        _hk = truncate_display_label(str(hl), 48) if hl else "residual"
        _z2 = colorbar_scale_title
        fig.add_trace(
            go.Heatmap(
                x=x_sp,
                y=y_sp,
                z=z0,
                colorscale=hm_cs,
                colorbar=cb_kw,
                hovertemplate=f"x=%{{x:.4g}}<br>y=%{{y:.4g}}<br>{_hk} ({_z2})=%{{z:.4g}}<extra></extra>",
                name=truncate_display_label(hl, 40) + (f" (+{nk - 1} more)" if nk > 1 else ""),
                meta=dict(subplot_row=row, subplot_col=col),
                **zlim,
            ),
            row=row,
            col=col,
        )
        _apply_enterprise_spatial_axes(fig, row, col)
        fig.update_xaxes(title_text="x", row=row, col=col)
        fig.update_yaxes(title_text="y", row=row, col=col)
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
        hl3 = row_labels[0] if row_labels else ""
        _hk3 = truncate_display_label(str(hl3), 48) if hl3 else "residual"
        _z3 = colorbar_scale_title
        fig.add_trace(
            go.Heatmap(
                x=x_sp,
                y=y_sp,
                z=sl,
                colorscale=hm_cs,
                colorbar=cb_kw,
                hovertemplate=f"x=%{{x:.4g}}<br>y=%{{y:.4g}}<br>{_hk3} ({_z3})=%{{z:.4g}}<extra></extra>",
                name=f"z-slice z={zk:.4g}" + (f" (+{nk - 1} keys)" if nk > 1 else ""),
                meta=dict(subplot_row=row, subplot_col=col),
                **zlim,
            ),
            row=row,
            col=col,
        )
        _apply_enterprise_spatial_axes(fig, row, col)
        fig.update_xaxes(title_text="x (z-slice)", row=row, col=col)
        fig.update_yaxes(title_text="y (z-slice)", row=row, col=col)
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


def _monitor_flat_subplot_titles(
    *,
    n_rows: int,
    is_eval: bool,
    nr_panel_title: str,
    trailing_rows: Tuple[str, ...] = (),
    constitutive_divergence_title: str = "Constitutive Divergence",
    constitutive_dissonance_title: str = "Constitutive Consistency",
) -> Tuple[str, ...]:
    """
    Row-major titles for ``make_subplots(rows=n_rows, cols=8)`` so paper titles align with merged panels.

    Anchor columns follow ``specs``: combined bars / trend panels start at col 1 or 5 (0-based col index 0 or 4).
    ``trailing_rows`` identifies optional rows appended after the base layout (eval: 4 / training: 5),
    in order — e.g. ``(\"state\", \"constitutive_divergence\")``.
    """
    st = [""] * (n_rows * 8)
    base_eval_rows = 4
    base_train_rows = 5
    base_rows = base_eval_rows if is_eval else base_train_rows
    if is_eval:
        st[16] = "Category Breakdown"
        st[20] = nr_panel_title
        st[24] = "Governing Residual"
        st[28] = "Constitutive Residual"
    else:
        st[16] = "Overall Admissibility"
        st[20] = "Category Breakdown"
        st[24] = "Governing Residuals"
        st[28] = "Constitutive Residuals"
        st[32] = "Governing Residual"
        st[36] = "Constitutive Residual"
    tag_title = {
        "state": "State snapshot (predicted)",
        "constitutive_divergence": constitutive_divergence_title,
    }
    for i, tag in enumerate(trailing_rows):
        row_1 = base_rows + 1 + i
        ix = (row_1 - 1) * 8
        if tag == "constitutive_divergence":
            st[ix] = tag_title["constitutive_divergence"]
            st[ix + 4] = constitutive_dissonance_title
        else:
            st[ix] = tag_title.get(tag, "")
    return tuple(st)


def _plotly_xy_axis_ref_strings_for_subplot(fig: Any, row: int, col: int) -> Tuple[str, str]:
    """Return ``(xref, yref)`` axis names (e.g. ``('x2','y2')``) for a ``Figure.get_subplot`` cell."""
    sp = fig.get_subplot(row, col)

    def _ref(axis_obj: Any, kind: str) -> str:
        root = getattr(fig.layout, f"{kind}axis", None)
        if axis_obj is root:
            return kind
        for i in range(2, 64):
            ax = getattr(fig.layout, f"{kind}axis{i}", None)
            if ax is axis_obj:
                return f"{kind}{i}"
        raise ValueError(f"Could not resolve {kind} axis ref for subplot ({row}, {col})")

    return _ref(sp.xaxis, "x"), _ref(sp.yaxis, "y")


def _legend_upper_right_paper_xy_for_subplot(fig: Any, row: int, col: int) -> Tuple[float, float]:
    """Paper-normalized (0--1) position for legend upper-right inside a subplot cell.

    Maps the subplot axis ``domain`` rectangles into figure ``paper`` coordinates for
    ``layout.legend`` (legacy Plotly does not support ``legend.xref`` / ``yref``).
    """
    xref, yref = _plotly_xy_axis_ref_strings_for_subplot(fig, row, col)
    xa = getattr(fig.layout, "xaxis" if xref == "x" else f"xaxis{xref[1:]}", None)
    ya = getattr(fig.layout, "yaxis" if yref == "y" else f"yaxis{yref[1:]}", None)
    if xa is None or ya is None:
        return 0.99, 0.99
    rdx = getattr(xa, "domain", None) or (0.0, 1.0)
    rdy = getattr(ya, "domain", None) or (0.0, 1.0)
    lx0, lx1 = float(rdx[0]), float(rdx[1])
    ly0, ly1 = float(rdy[0]), float(rdy[1])
    return lx0 + 0.98 * (lx1 - lx0), ly0 + 0.98 * (ly1 - ly0)


def _add_dissonance_inline_legend(
    fig: Any,
    traces: List[Any],
    *,
    row: int,
    col: int,
) -> None:
    """Draw a compact inline legend centered at the top of the dissonance subplot."""
    items = [tr for tr in traces if str(getattr(tr, "name", "") or "").strip()]
    if not items:
        return
    try:
        xref, yref = _plotly_xy_axis_ref_strings_for_subplot(fig, row, col)
    except Exception:
        return
    x_positions = [0.36, 0.58] if len(items) == 2 else [0.47]
    y = 0.955
    for tr, x_text in zip(items[:2], x_positions):
        line = getattr(tr, "line", None)
        color = getattr(line, "color", None) if line is not None else None
        dash = getattr(line, "dash", None) if line is not None else None
        width = getattr(line, "width", None) if line is not None else None
        x0 = x_text - 0.09
        x1 = x_text - 0.02
        fig.add_shape(
            type="line",
            x0=x0,
            x1=x1,
            y0=y,
            y1=y,
            xref=f"{xref} domain",
            yref=f"{yref} domain",
            line=dict(
                color=color or _ENTERPRISE_THEME["font_color"],
                width=width or 2,
                dash=dash or "solid",
            ),
        )
        fig.add_annotation(
            x=x_text,
            y=y,
            xref=f"{xref} domain",
            yref=f"{yref} domain",
            text=str(getattr(tr, "name", "") or ""),
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            align="left",
            font=dict(
                size=11,
                color=color or _ENTERPRISE_THEME["font_color"],
                family=_ENTERPRISE_THEME["font_stack"],
            ),
            bgcolor="rgba(255,255,255,0.72)",
            borderpad=1,
        )


def _add_dissonance_tier_annotations(
    fig: Any,
    traces: List[Any],
    *,
    row: int,
    col: int,
) -> None:
    """Add right-edge ±% Δ labels for tier boundary traces in the dissonance subplot.

    Looks for traces whose ``name`` contains ``% Δ`` (the tier boundary lines
    emitted by ``_build_dissonance_tier_lines``), reads the last y-value, and
    places a small annotation at the right edge of the subplot using domain
    coordinates so it does not affect the data axis range.
    """
    tier_traces = [
        tr for tr in traces
        if isinstance(getattr(tr, "meta", None), dict) and "tier_label" in tr.meta
    ]
    if not tier_traces:
        return
    try:
        xref, yref = _plotly_xy_axis_ref_strings_for_subplot(fig, row, col)
    except Exception:
        return
    muted_color = _ENTERPRISE_THEME.get("muted", "#64748b")
    for tr in tier_traces:
        y_data = getattr(tr, "y", None)
        if y_data is None or len(y_data) == 0:
            continue
        y_val = float(y_data[-1])
        label = str((getattr(tr, "meta", None) or {}).get("tier_label", "") or "")
        fig.add_annotation(
            x=1.01,
            y=y_val,
            xref=f"{xref} domain",
            yref=yref,
            text=label,
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(
                size=9,
                color=muted_color,
                family=_ENTERPRISE_THEME.get("font_stack", ""),
            ),
        )


def _add_monitor_panel_title_annotations(
    fig: Any,
    flat_titles: Tuple[str, ...],
    *,
    font_color: str,
) -> None:
    """Attach one title annotation per non-empty anchor cell.

    Plotly may omit ``subplot_titles`` for grids that mix ``colspan`` and ``domain`` cells; we still pass
    ``subplot_titles`` for API consistency, then place matching labels with axis **domain** coordinates
    (``xref``/``yref`` ending in `` domain``) so ``x=0.5``, ``y>1`` sit **above** the plot, not in data space.
    """
    T = _ENTERPRISE_THEME
    fs = T["font_stack"]
    for i, text in enumerate(flat_titles):
        t = str(text or "").strip()
        if not t:
            continue
        row = i // 8 + 1
        col = i % 8 + 1
        try:
            xa, ya = _plotly_xy_axis_ref_strings_for_subplot(fig, row, col)
        except Exception:
            continue
        fig.add_annotation(
            text=t,
            xref=f"{xa} domain",
            yref=f"{ya} domain",
            x=0.5,
            y=1.02,
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=13, color=font_color, family=fs),
        )


def build_worst_keys_table_figure(bundle: Dict[str, Any]) -> Any:
    """Standalone table: highest ``R_norm`` keys at the final log step (troubleshooting)."""
    from moju.monitor.visualize_components import build_worst_keys_table_card

    rows = list(bundle.get("worst_keys_rows") or [])
    h = min(120 + 28 * len(rows), 720) if rows else None
    return build_worst_keys_table_card(
        bundle,
        title="Worst residual keys (by R_norm, final log step)",
        height=h,
        limit=max(len(rows), 12) if rows else 12,
    )


def _build_eval_kpi_figure(bundle: Dict[str, Any]) -> Any:
    """Dash KPI tab for eval: category indicators plus overall rollup summary."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    T = _ENTERPRISE_THEME
    metrics = list(bundle.get("metrics") or [])
    n = int(bundle.get("n") or 0)
    if not metrics:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", paper_bgcolor=T["paper_bg"], height=320)
        fig.add_annotation(
            text="No log metrics",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(family=T["font_stack"], color=T["font_color"]),
        )
        return fig
    last_cat = metrics[-1].get("category_admissibility_score") or {}
    first_cat = metrics[0].get("category_admissibility_score") or {} if n > 1 else {}
    use_ref = n > 1
    font_color = T["font_color"]
    warn_color = "#F59E0B"

    def ref_for(key: str) -> Optional[float]:
        if not use_ref or key not in first_cat:
            return None
        try:
            v = float(first_cat[key])
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) else None

    laws_last = float(last_cat.get("laws", float("nan")))
    const_last = float(last_cat.get("constitutive", float("nan")))

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        horizontal_spacing=0.06,
    )
    fig.add_trace(
        _go_category_kpi_indicator(
            laws_last if math.isfinite(laws_last) else 0.0,
            "Governing Score",
            ref_for("laws"),
            font_color=font_color,
            warn_color=warn_color,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        _go_category_kpi_indicator(
            const_last if math.isfinite(const_last) else 0.0,
            "Constitutive Score",
            ref_for("constitutive"),
            font_color=font_color,
            warn_color=warn_color,
        ),
        row=1,
        col=2,
    )
    overall = float(metrics[-1].get("overall_admissibility_score", float("nan")))
    overall_pct = format_admissibility_pct(overall) if math.isfinite(overall) else "N/A"
    status = format_admissibility_status_label(overall)
    explain = (
        f"<b>Overall admissibility (rollup): {overall_pct}</b> [{status}] — "
        "computed from available category scores. Use the Admissibility tab for per-key detail."
    )
    fig.add_annotation(
        text=explain,
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.06,
        showarrow=False,
        font=dict(size=11, family=T["font_stack"], color=T["muted"]),
        align="center",
    )
    fig.update_layout(
        title=dict(text="Eval — category admissibility", x=0.5, xanchor="center"),
        template="plotly_white",
        paper_bgcolor=T["paper_bg"],
        plot_bgcolor=T["paper_bg"],
        font=dict(family=T["font_stack"], color=font_color),
        margin=dict(l=32, r=32, t=56, b=120),
        height=380,
    )
    return fig


def _build_plotly_monitor_figure_single(
    bundle: Dict[str, Any],
    *,
    figure_title: Optional[str] = None,
    step_label: str = "Step",
    r_norm_scale: str = "log",
    spatial_heatmap_colorscale: Optional[str] = None,
    theme: str = "light",
    baseline_score: Optional[float] = None,  # API parity with visualize(); unused here (KPI tab uses _build_kpi_figure).
    export_buttons: bool = True,
    show_branding: bool = False,
    density: str = "comfortable",
) -> Any:
    """Build a decision-oriented Plotly physics admissibility report."""
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if r_norm_scale not in ("log", "linear"):
        raise ValueError("r_norm_scale must be 'log' or 'linear'")
    if density not in ("comfortable", "compact"):
        raise ValueError("density must be 'comfortable' or 'compact'")
    _require_light_theme(theme)

    use_log_rnorm = r_norm_scale == "log"
    rnorm_y_title = "log10(R_norm + ε)" if use_log_rnorm else "Normalized residual (R norm)"
    reff_y_title = "log10(R_eff + ε)" if use_log_rnorm else "Effective residual (R_eff)"
    hm_cs = spatial_heatmap_colorscale or DEFAULT_SPATIAL_HEATMAP_COLORSCALE
    compact = density == "compact"
    _sub_px = 14 if compact else 15
    _main_title_px = 18 if compact else 20
    kpi_title_px = 12 if compact else 13
    kpi_num_px = 20 if compact else 22

    n = int(bundle["n"])
    indices = list(range(n))
    mode_eff = bundle["mode"]
    bar_keys: List[str] = bundle.get("bar_keys") or []
    bar_display: List[str] = bundle.get("bar_display") or []
    _bar_values_raw = bundle.get("bar_values")
    bar_values = np.asarray([] if _bar_values_raw is None else _bar_values_raw, dtype=float)
    _bar_values_eff_raw = bundle.get("bar_values_eff")
    bar_values_eff = np.asarray(
        [] if _bar_values_eff_raw is None else _bar_values_eff_raw, dtype=float
    )
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

    T = _ENTERPRISE_THEME
    paper_bg = T["paper_bg"]
    plot_bg = T["plot_bg"]
    font_color = T["font_color"]
    muted = T["muted"]
    warn_color = MOJU_LIGHT.palette.adm_med

    is_eval = mode_eff == "eval"
    spatial_row = 4 if is_eval else 5
    has_cd = bool(isinstance(bundle.get("closure_debug"), dict) and bundle.get("closure_debug"))
    _snapshots_for_row = bundle.get("state_field_snapshots")
    has_state_ov = isinstance(_snapshots_for_row, list) and len(_snapshots_for_row) > 0
    trailing: List[str] = []
    if has_state_ov:
        trailing.append("state")
    if has_cd:
        trailing.append("constitutive_divergence")

    base_n_rows = 4 if is_eval else 5
    n_rows = base_n_rows + len(trailing)
    state_row: Optional[int] = (base_n_rows + 1) if has_state_ov else None
    cd_row: Optional[int] = None
    if has_cd:
        cd_row = base_n_rows + 1 + trailing.index("constitutive_divergence")

    monitor_show_legend = False
    apply_cd_dissonance_legend = False

    div_legacy_eval = 0.18
    div_legacy_train = 0.14

    _div_row_spec: List[Any] = [
        {"type": "xy", "colspan": 8},
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]

    _split_xy_row: List[Any] = [
        {"type": "xy", "colspan": 4},
        None,
        None,
        None,
        {"type": "xy", "colspan": 4},
        None,
        None,
        None,
    ]
    kpi_row_training = [
        None,
        None,
        {"type": "domain", "colspan": 2},
        None,
        None,
        {"type": "domain", "colspan": 2},
        None,
        None,
    ]
    # Eval: two KPIs (Governing, Constitutive), same 8-slot pattern as training.
    kpi_row_eval = [
        None,
        None,
        {"type": "domain", "colspan": 2},
        None,
        None,
        {"type": "domain", "colspan": 2},
        None,
        None,
    ]
    specs: List[List[Any]] = [
        [{"type": "xy", "colspan": 8}, None, None, None, None, None, None, None],
        kpi_row_eval if is_eval else kpi_row_training,
    ]
    if is_eval:
        specs.append(_split_xy_row)
        specs.append(_split_xy_row)
        if has_cd or has_state_ov:
            row_heights = [0.002, 0.06, 0.22, 0.18]
            if has_state_ov:
                row_heights.append(div_legacy_eval)
            if has_cd:
                row_heights.append(div_legacy_eval)
                row_heights[-1] *= MONITOR_DIV_ROW_WEIGHT_MULT * 0.96
            s_tot = sum(row_heights)
            row_heights = [h / s_tot for h in row_heights]
        else:
            row_heights = [0.002, 0.074, 0.262, 0.222]
        for tag in trailing:
            specs.append(_split_xy_row if tag == "constitutive_divergence" else _div_row_spec)
    else:
        specs.append(_split_xy_row)
        specs.append(_split_xy_row)
        specs.append(_split_xy_row)
        if has_cd or has_state_ov:
            row_heights = [0.002, 0.065, 0.21, 0.185, 0.18]
            if has_state_ov:
                row_heights.append(div_legacy_train)
            if has_cd:
                row_heights.append(div_legacy_train)
                row_heights[-1] *= MONITOR_DIV_ROW_WEIGHT_MULT * 0.96
            s_tot = sum(row_heights)
            row_heights = [h / s_tot for h in row_heights]
        else:
            row_heights = [0.002, 0.074, 0.262, 0.222, 0.222]
        for tag in trailing:
            specs.append(_split_xy_row if tag == "constitutive_divergence" else _div_row_spec)
    prefer_last_t = bool(bundle.get("spatial_prefer_last_t", True))
    cd_dissonance_embed: Optional[Dict[str, Any]] = None
    cd_dissonance_title = "Constitutive Consistency"
    cd_divergence_title = "Constitutive Divergence"
    if has_cd:
        cd_dissonance_embed = prepare_constitutive_model_implied_vs_x_embed(
            bundle,
            prefer_last_t=prefer_last_t,
        )
        if cd_dissonance_embed:
            cd_dissonance_title = str(cd_dissonance_embed.get("title") or cd_dissonance_title)
        cd_divergence_title = constitutive_divergence_title_for_bundle(bundle)

    nr_panel_title = truncate_display_label(str(bundle.get("nr_title") or "Normalized Residuals"), 56)
    subplot_titles = _monitor_flat_subplot_titles(
        n_rows=n_rows,
        is_eval=is_eval,
        nr_panel_title=nr_panel_title,
        trailing_rows=tuple(trailing),
        constitutive_divergence_title=cd_divergence_title,
        constitutive_dissonance_title=cd_dissonance_title,
    )
    fig = make_subplots(
        rows=n_rows,
        cols=8,
        specs=specs,
        row_heights=row_heights,
        # Row gaps (fraction of figure height): test mode uses a larger gap so upper-row x labels
        # clear the subplot titles below (chart row_heights trimmed slightly to compensate).
        vertical_spacing=0.122 if not is_eval else 0.126,
        horizontal_spacing=0.095,
        subplot_titles=subplot_titles,
    )
    _drop_paper_subplot_title_annots_with_text(
        fig,
        frozenset(
            {
                "Governing Score",
                "Constitutive Score",
                "Scaling Score",
                "Data Score",
                "Duality Score",
            }
        ),
    )

    # Header band
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)
    last_ov = float(overall_adm[-1]) if len(overall_adm) else float("nan")
    first_ov = float(overall_adm[0]) if len(overall_adm) else float("nan")
    status = _admissibility_status_bracket_plotly(last_ov)
    status_color = _status_bracket_color(status, warn_color=warn_color)
    final_idx = indices[-1] if indices else 0
    pct_html = (
        format_admissibility_pct(last_ov) if math.isfinite(last_ov) else "N/A"
    )
    # Merged into layout title at update_layout (one step smaller than main title for hierarchy).
    overall_subtitle_html = (
        f"<span style='font-size:{_sub_px}px;font-weight:600;color:{font_color}'>Overall admissibility (final): </span>"
        f"<span style='font-size:{_sub_px}px;font-weight:700'>{pct_html}</span>"
        f" <span style='font-size:{_sub_px}px;color:{status_color};font-weight:700'>– [{status}]</span>"
    )

    # KPI cards as indicators
    last_cat = (metrics[-1].get("category_admissibility_score") if metrics else {}) or {}
    first_cat = (metrics[0].get("category_admissibility_score") if metrics else {}) or {}

    def _kpi_indicator(value: float, title: str, ref: Optional[float]) -> Any:
        return _go_category_kpi_indicator(
            value,
            title,
            ref,
            font_color=font_color,
            warn_color=warn_color,
            title_px=kpi_title_px,
            num_px=kpi_num_px,
            delta_px=8 if compact else 9,
        )

    laws_last = float(last_cat.get("laws", float("nan")))
    laws_ref = float(first_cat.get("laws", float("nan"))) if "laws" in first_cat else None
    const_last = float(last_cat.get("constitutive", float("nan")))
    const_ref = float(first_cat.get("constitutive", float("nan"))) if "constitutive" in first_cat else None
    # Governing + Constitutive (cols 3–4 and 6–7).
    fig.add_trace(
        _kpi_indicator(
            laws_last if math.isfinite(laws_last) else 0.0, "Governing Score", laws_ref
        ),
        row=2,
        col=3,
    )
    fig.add_trace(
        _kpi_indicator(
            const_last if math.isfinite(const_last) else 0.0,
            "Constitutive Score",
            const_ref,
        ),
        row=2,
        col=6,
    )

    trend_y_min: Optional[float] = None
    trend_y_top: Optional[float] = None

    # Training: vs-step Overall Admissibility (eval mode uses full-width category row only).
    if (not is_eval) and any(np.isfinite(overall_adm)):
        adm_hover = [format_admissibility_pct(float(y)) if np.isfinite(y) else "N/A" for y in overall_adm]
        overall_adm_pct = overall_adm * 100.0
        fig.add_trace(
            go.Scatter(
                x=indices,
                y=overall_adm_pct,
                mode="lines",
                name="Overall Admissibility",
                line=dict(color=OVERALL_ADMISSIBILITY_TREND_LINE_COLOR, width=2.8),
                text=adm_hover,
                hovertemplate="Overall Admissibility<br>%{x}<br>%{text}<extra></extra>",
                showlegend=False,
            ),
            row=3,
            col=1,
        )
        trend_line_tr = next(
            (
                tr
                for tr in fig.data
                if getattr(tr, "type", None) == "scatter"
                and getattr(tr, "name", "") == "Overall Admissibility"
                and getattr(tr, "mode", "") == "lines"
            ),
            None,
        )
        if trend_line_tr is not None:
            _xa = getattr(trend_line_tr, "xaxis", None) or "x"
            _ya = getattr(trend_line_tr, "yaxis", None) or "y"
            fig.add_shape(
                type="rect",
                x0=0,
                x1=1,
                xref=f"{_xa} domain",
                y0=0,
                y1=1,
                yref=f"{_ya} domain",
                fillcolor=plot_bg,
                line_width=0,
                layer="below",
            )
        if math.isfinite(last_ov):
            fig.add_trace(
                go.Scatter(
                    x=[final_idx],
                    y=[last_ov * 100.0],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=OVERALL_ADMISSIBILITY_TREND_LINE_COLOR,
                        line=dict(width=2, color="#ffffff"),
                    ),
                    text=[format_admissibility_pct(last_ov)] if math.isfinite(last_ov) else ["N/A"],
                    hovertemplate="Final<br>%{text}<extra></extra>",
                    showlegend=False,
                ),
                row=3,
                col=1,
            )
        finite = overall_adm[np.isfinite(overall_adm)]
        if finite.size:
            y_lo = float(np.min(finite))
            y_hi = float(np.max(finite))
            ymin = max(0.0, y_lo - 0.05) * 100.0
            y_top = min(1.02, y_hi + 0.08) * 100.0
            if y_top - ymin < 8.0:
                mid = 0.5 * (ymin + y_top)
                ymin = max(0.0, mid - 4.0)
                y_top = min(102.0, mid + 4.0 + 6.0)
            trend_y_min, trend_y_top = ymin, y_top

    if mode_eff != "eval":
        fig.update_xaxes(title_text=step_label, row=3, col=1, automargin=True)
        yaxis_kw: Dict[str, Any] = dict(title_text="Admissibility (%)", tickformat=".0f", row=3, col=1, automargin=True)
        if trend_y_min is not None and trend_y_top is not None:
            yaxis_kw["range"] = [trend_y_min, trend_y_top]
            yaxis_kw["tickvals"] = [0, 25, 50, 75, 100]
        fig.update_yaxes(**yaxis_kw)
        _apply_enterprise_axis_style(fig, 3, 1, y_log=False, x_grid=False, y_grid=True)

    cat_col = 1 if is_eval else 5

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
    cat_labels = [
        _wrap_category_tick_label_html(
            "Constitutive<br>Relations" if v == "Constitutive Relations" else v,
        )
        for v in cat_labels_raw
    ]
    cat_vals = [float(v) if math.isfinite(v) else 0.0 for _, v in labels_vals]
    cat_x_pct = [v * 100.0 for v in cat_vals]
    cat_text = [format_admissibility_pct(v) if math.isfinite(v) else "N/A" for _, v in labels_vals]
    cat_colors = [_kpi_indicator_value_color(v, warn_color=warn_color) for v in cat_vals]
    fig.add_trace(
        go.Bar(
            x=cat_x_pct,
            y=cat_labels,
            orientation="h",
            width=CATEGORY_BREAKDOWN_BAR_WIDTH,
            marker=dict(color=cat_colors, line=dict(color=T["bar_line"], width=0.5)),
            text=cat_text,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>admissibility=%{text}<extra></extra>",
            showlegend=False,
        ),
        row=3,
        col=cat_col,
    )
    cat_axis = getattr(fig.data[-1], "xaxis", "x2")
    cat_yaxis = getattr(fig.data[-1], "yaxis", "y2")
    fig.add_shape(
        type="line",
        x0=ADM_HIGH_THRESHOLD * 100.0,
        x1=ADM_HIGH_THRESHOLD * 100.0,
        y0=0,
        y1=1,
        xref=cat_axis,
        yref=f"{cat_yaxis} domain",
        line=dict(color=warn_color, dash="dash"),
    )
    worst_label = cat_labels_raw[0] if cat_labels_raw else "N/A"
    show_primary_issue = False
    for _, v in labels_vals:
        fv = float(v) if math.isfinite(float(v)) else float("nan")
        if not math.isfinite(fv) or not is_high_admissibility(fv):
            show_primary_issue = True
            break
    if show_primary_issue:
        fig.add_annotation(
            x=0.02,
            y=PRIMARY_ISSUE_ANNOTATION_Y_DOMAIN,
            xref=f"{cat_axis} domain",
            yref=f"{cat_yaxis} domain",
            text=f"Primary Issue: {worst_label}",
            showarrow=False,
            yanchor="bottom",
            align="left",
            font=dict(size=11, color=DISSONANCE_COLOR, family=T["font_stack"]),
        )
    fig.update_xaxes(
        title_text="Admissibility (%)",
        range=list(category_adm_bar_axis_range_percent_full()),
        tickformat=".0f",
        row=3,
        col=cat_col,
        automargin=True,
    )
    fig.update_yaxes(side="left", automargin=True, row=3, col=cat_col)
    _apply_enterprise_axis_style(fig, 3, cat_col, y_log=False, x_grid=True, y_grid=False)

    if is_eval:
        # Single bar chart: all bar_keys in user order, color per category prefix.
        _tc = 5
        if not bar_keys:
            fig.add_trace(
                go.Scatter(
                    x=[indices[len(indices) // 2] if indices else 0],
                    y=[0.0],
                    mode="text",
                    text=["No residual keys selected"],
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=3,
                col=_tc,
            )
        else:
            xs = [truncate_display_label(bar_display[i], 40) for i in range(len(bar_keys))]
            vals = [
                float(bar_values[i]) if i < len(bar_values) and np.isfinite(bar_values[i]) else 0.0
                for i in range(len(bar_keys))
            ]
            # Test combined panel: semilog Y on R_norm + ε (positive magnitudes); not log10(y) on a linear axis.
            ys = [float(max(v, 0.0) + R_NORM_LOG_EPS) if use_log_rnorm else float(v) for v in vals]
            cds = [_scale_for_key(bar_keys[i]) for i in range(len(bar_keys))]
            bar_cols = [_residual_color_from_key(bar_keys[i]) for i in range(len(bar_keys))]
            cd_stack = (
                np.column_stack([cds, [np.log10(y) for y in ys]]) if use_log_rnorm else np.asarray(cds).reshape(-1, 1)
            )
            hover_tmpl = (
                "%{x}<br>R_norm+ε=%{y:.4g}<br>log10=%{customdata[1]:.4g}<br>scale_k=%{customdata[0]:.4g}<extra></extra>"
                if use_log_rnorm
                else "%{x}<br>R_norm=%{y:.4g}<br>scale_k=%{customdata[0]:.4g}<extra></extra>"
            )
            fig.add_trace(
                go.Bar(
                    x=xs,
                    y=ys,
                    marker=dict(color=bar_cols, line=dict(color=T["bar_line"], width=0.5)),
                    customdata=cd_stack,
                    showlegend=False,
                    hovertemplate=hover_tmpl,
                ),
                row=3,
                col=_tc,
            )
        fig.update_xaxes(title_text="Residual key", row=3, col=_tc, automargin=True)
        if use_log_rnorm:
            fig.update_yaxes(
                title_text="R_norm + ε",
                type="log",
                exponentformat="power",
                showexponent="all",
                dtick=1,
                row=3,
                col=_tc,
                automargin=True,
            )
        else:
            fig.update_yaxes(title_text="Normalized residual (R norm)", row=3, col=_tc, automargin=True)
        _apply_enterprise_axis_style(fig, 3, _tc, y_log=False, x_grid=False, y_grid=True)
        if use_log_rnorm:
            fig.update_yaxes(zeroline=False, row=3, col=_tc)

    # Residual diagnostics (training only; eval uses combined bars on row 3).
    # Training panels plot R_eff (the raw effective residual) so the chart matches the
    # quantity the loss is minimising; eval bar chart remains on R_norm so different
    # keys can be compared at a glance after per-key scale normalisation.
    def _plot_residual_panel(cat: str, row: int, col: int, title_prefix: str) -> None:
        info = category_training.get(
            cat, {"keys": [], "displays": [], "r_norm_mat": np.zeros((0, n)), "r_eff_mat": np.zeros((0, n))}
        )
        ckeys: List[str] = list(info.get("keys") or [])
        displays: List[str] = list(info.get("displays") or [])
        reff_raw = info.get("r_eff_mat")
        if reff_raw is None:
            rn_mat = np.asarray(
                info.get("r_norm_mat") if info.get("r_norm_mat") is not None else np.zeros((0, n)),
                dtype=float,
            )
            if rn_mat.size and ckeys:
                scales = np.asarray([_scale_for_key(k) for k in ckeys], dtype=float).reshape(-1, 1)
                mat = rn_mat * scales
            else:
                mat = np.zeros((0, n), dtype=float)
        else:
            mat = np.asarray(reff_raw, dtype=float)
        if mode_eff != "eval" and mat.size and len(ckeys):
            terminal = mat[:, -1]
            worst_i = int(np.nanargmax(terminal)) if np.any(np.isfinite(terminal)) else 0
            unstable = np.nanstd(np.diff(mat, axis=1), axis=1) if mat.shape[1] > 1 else np.zeros(mat.shape[0])
            unstable_i = int(np.nanargmax(unstable)) if unstable.size else -1
            _first_rv_idx: Optional[int] = None
            for i, key in enumerate(ckeys):
                ys = mat[i, :]
                if not np.all(np.isfinite(ys)):
                    continue
                if _first_rv_idx is None:
                    _first_rv_idx = len(fig.data)
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
                        hovertemplate=f"{name}<br>{step_label}=%{{x}}<br>{'log10(R_eff+ε)' if use_log_rnorm else 'R_eff'}=%{{y:.4g}}<br>scale_k=%{{customdata:.4g}}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )
            worst_name = displays[worst_i] if worst_i < len(displays) else (ckeys[worst_i] if ckeys else "N/A")
            if len(ckeys) > 1 and _first_rv_idx is not None:
                _t0 = fig.data[_first_rv_idx]
                _xa = (getattr(_t0, "xaxis", None) or "x") if _t0 is not None else "x"
                _ya = (getattr(_t0, "yaxis", None) or "y") if _t0 is not None else "y"
                fig.add_annotation(
                    x=0.01,
                    y=1.02,
                    xref=f"{_xa} domain",
                    yref=f"{_ya} domain",
                    text=f"Worst violation: {truncate_display_label(worst_name, 44)}",
                    showarrow=False,
                    align="left",
                    font=dict(size=10, color=DISSONANCE_COLOR, family=_ENTERPRISE_THEME["font_stack"]),
                    row=row,
                    col=col,
                )
        else:
            fk = [i for i, k in enumerate(bar_keys) if str(k).startswith(f"{cat}/")]
            if fk:
                xs = [truncate_display_label(bar_display[i], 40) for i in fk]
                if bar_values_eff.size:
                    vals = [
                        float(bar_values_eff[i])
                        if i < len(bar_values_eff) and np.isfinite(bar_values_eff[i])
                        else 0.0
                        for i in fk
                    ]
                else:
                    vals = [
                        float(bar_values[i]) * _scale_for_key(bar_keys[i])
                        if i < len(bar_values) and np.isfinite(bar_values[i])
                        else 0.0
                        for i in fk
                    ]
                ys = [float(np.log10(max(v, 0.0) + R_NORM_LOG_EPS)) if use_log_rnorm else v for v in vals]
                cds = [_scale_for_key(bar_keys[i]) for i in fk]
                _bc = RESIDUAL_COLOR_LAWS if cat == "laws" else RESIDUAL_COLOR_CONSTITUTIVE
                fig.add_trace(
                    go.Bar(
                        x=xs,
                        y=ys,
                        marker=dict(color=_bc, line=dict(color=_ENTERPRISE_THEME["bar_line"], width=0.5)),
                        customdata=cds,
                        showlegend=False,
                        hovertemplate="%{x}<br>" + (("log10(R_eff+ε)=%{y:.4g}") if use_log_rnorm else ("R_eff=%{y:.4g}")) + "<br>scale_k=%{customdata:.4g}<extra></extra>",
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
            title_text=("Residual key" if mode_eff == "eval" else (step_label if mat.size else "Residual key")),
            row=row,
            col=col,
            automargin=True,
        )
        fig.update_yaxes(
            title_text=reff_y_title if col == 1 else "",
            row=row,
            col=col,
            automargin=True,
        )
        _apply_enterprise_axis_style(fig, row, col, y_log=use_log_rnorm, x_grid=False, y_grid=True)

    if not is_eval:
        _plot_residual_panel("laws", 4, 1, "Governing")
        _plot_residual_panel("constitutive", 4, 5, "Constitutive")

    # Spatial maps: each heatmap colorbar uses display-space zmin/zmax from that panel only (log or linear).
    mid = indices[len(indices) // 2] if indices else 0
    _spatial_cb_title = SPATIAL_HEATMAP_COLORBAR_TITLE_LOG if use_log_rnorm else "|residual|"
    if spatial is not None:
        _z_range_law = _finite_z_range_from_array(
            _spatial_panel_plotted_display_z_flat(spatial, use_log_rnorm=use_log_rnorm)
        )
        _plotly_add_spatial_panel_to_subplot(
            fig,
            row=spatial_row,
            col=1,
            spatial=spatial,
            hm_cs=hm_cs,
            mid=mid,
            colorbar_compact=False,
            use_log_rnorm=use_log_rnorm,
            colorbar_scale_title=_spatial_cb_title,
            z_range=_z_range_law,
        )
    else:
        fig.add_trace(
            go.Scatter(x=[mid], y=[0.0], mode="text", text=["No governing spatial field"], showlegend=False, hoverinfo="skip"),
            row=spatial_row,
            col=1,
        )
        fig.update_xaxes(visible=False, row=spatial_row, col=1)
        fig.update_yaxes(visible=False, row=spatial_row, col=1)

    if spatial_rnorm is not None:
        _z_range_const = _finite_z_range_from_array(
            _spatial_panel_plotted_display_z_flat(spatial_rnorm, use_log_rnorm=use_log_rnorm)
        )
        _plotly_add_spatial_panel_to_subplot(
            fig,
            row=spatial_row,
            col=5,
            spatial=spatial_rnorm,
            hm_cs=hm_cs,
            mid=mid,
            colorbar_compact=False,
            use_log_rnorm=use_log_rnorm,
            colorbar_scale_title=_spatial_cb_title,
            z_range=_z_range_const,
        )
    else:
        fig.add_trace(
            go.Scatter(x=[mid], y=[0.0], mode="text", text=["No constitutive spatial field"], showlegend=False, hoverinfo="skip"),
            row=spatial_row,
            col=5,
        )
        fig.update_xaxes(visible=False, row=spatial_row, col=5)
        fig.update_yaxes(visible=False, row=spatial_row, col=5)

    if has_state_ov and state_row is not None:
        ss_rows = bundle.get("state_field_snapshots") or []
        sp_for_axis = spatial if isinstance(spatial, dict) else None
        x_lab_card = _pos_axis_title_card(sp_for_axis)
        for jcol in range(3):
            col = jcol + 1
            if jcol < len(ss_rows):
                snap = ss_rows[jcol]
                if snap.get("kind") == "1d":
                    fig.add_trace(
                        go.Scatter(
                            x=np.asarray(snap["x"], dtype=float),
                            y=np.asarray(snap["z"], dtype=float),
                            mode="lines",
                            line=dict(width=2, color=MOJU_LIGHT.palette.line_primary),
                            name=str(snap.get("name", "")),
                            showlegend=False,
                        ),
                        row=state_row,
                        col=col,
                    )
                    fig.update_xaxes(title_text=x_lab_card, row=state_row, col=col, automargin=True)
                    fig.update_yaxes(title_text=str(snap.get("name", "value")), row=state_row, col=col, automargin=True)
                elif snap.get("kind") == "2d":
                    fig.add_trace(
                        go.Heatmap(
                            z=np.asarray(snap["z"], dtype=float),
                            x=np.asarray(snap["x"], dtype=float),
                            y=np.asarray(snap["y"], dtype=float),
                            colorscale=hm_cs,
                            name=str(snap.get("name", "")),
                            showscale=(col == 3),
                            meta=dict(subplot_row=state_row, subplot_col=col),
                        ),
                        row=state_row,
                        col=col,
                    )
                    fig.update_xaxes(title_text="Position x", row=state_row, col=col, automargin=True)
                    fig.update_yaxes(title_text="Position y", row=state_row, col=col, automargin=True)
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=[float(mid)],
                            y=[0.0],
                            mode="text",
                            text=["Unsupported snapshot"],
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=state_row,
                        col=col,
                    )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=[float(mid)],
                        y=[0.0],
                        mode="text",
                        text=["—"],
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=state_row,
                    col=col,
                )

    if has_cd and cd_row is not None:
        div_fig = build_spatial_normalized_divergence_figure(bundle, title=None)
        traces = list(div_fig.data)
        if len(traces) >= 1:
            fig.add_trace(traces[0], row=cd_row, col=1)
            lt = fig.data[-1]
            if getattr(lt, "type", None) == "heatmap" and getattr(lt, "colorbar", None) is not None:
                lt.update(meta=dict(subplot_row=cd_row, subplot_col=1))
        else:
            fig.add_trace(
                go.Scatter(
                    x=[0.0],
                    y=[0.0],
                    mode="text",
                    text=["No constitutive divergence panel"],
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=cd_row,
                col=1,
            )
        fig.update_xaxes(automargin=True, row=cd_row, col=1)
        fig.update_yaxes(automargin=True, row=cd_row, col=1)
        tl0 = traces[0] if traces else None
        tty = getattr(tl0, "type", None) if tl0 is not None else None
        if tty == "scatter":
            nlen_fb = primary_closure_debug_field_length(bundle)
            xv0 = getattr(tl0, "x", None)
            nx = len(xv0) if xv0 is not None else 0
            try:
                nlen_i = int(nlen_fb) if nlen_fb is not None and int(nlen_fb) > 0 else int(nx)
            except (TypeError, ValueError):
                nlen_i = int(nx)
            hin = bundle.get("spatial_coord_hint")
            _, xtit = infer_divergence_abscissa(
                bundle, nlen_i, hint_axis=str(hin) if hin else None
            )
            fig.update_xaxes(title_text=xtit, row=cd_row, col=1, automargin=True)
        elif tty == "heatmap":
            div_x_title = getattr(getattr(div_fig.layout, "xaxis", None), "title", None)
            div_y_title = getattr(getattr(div_fig.layout, "yaxis", None), "title", None)
            fig.update_xaxes(
                title_text=str(getattr(div_x_title, "text", None) or "Position x"),
                row=cd_row,
                col=1,
                automargin=True,
            )
            fig.update_yaxes(
                title_text=str(getattr(div_y_title, "text", None) or "Position y"),
                row=cd_row,
                col=1,
                automargin=True,
            )

        emb = cd_dissonance_embed
        if emb and emb.get("traces"):
            dissonance_traces = list(emb["traces"])
            for tr in dissonance_traces:
                tr.showlegend = False
                fig.add_trace(tr, row=cd_row, col=5)
            _add_dissonance_inline_legend(fig, dissonance_traces, row=cd_row, col=5)
            fig.update_xaxes(
                title_text=str(emb.get("x_title") or "Position x"),
                row=cd_row,
                col=5,
                automargin=True,
            )
            yr = emb.get("y_range")
            fig.update_yaxes(
                title_text=str(emb.get("y_title") or "Value"),
                range=yr if yr else None,
                autorange=(yr is None),
                row=cd_row,
                col=5,
                automargin=True,
            )
            # Tier boundary annotations: pin labels at the right edge of the subplot
            _add_dissonance_tier_annotations(fig, dissonance_traces, row=cd_row, col=5)
            # Max-delta summary: bottom-left, clear of inline legend (top-centre) and tier labels (right)
            max_lbl = str(emb.get("max_delta_label") or "")
            if max_lbl:
                try:
                    xref_d, yref_d = _plotly_xy_axis_ref_strings_for_subplot(fig, cd_row, 5)
                    fig.add_annotation(
                        x=0.02,
                        y=0.04,
                        xref=f"{xref_d} domain",
                        yref=f"{yref_d} domain",
                        text=max_lbl,
                        showarrow=False,
                        xanchor="left",
                        yanchor="bottom",
                        font=dict(
                            size=9,
                            color=_ENTERPRISE_THEME.get("muted", "#64748b"),
                            family=_ENTERPRISE_THEME.get("font_stack", ""),
                        ),
                        bgcolor="rgba(255,255,255,0.0)",
                    )
                except Exception:
                    pass
        else:
            fig.add_trace(
                go.Scatter(
                    x=[0.5],
                    y=[0.0],
                    mode="text",
                    text=["No constitutive dissonance slice (1-D / 2-D data required)"],
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=cd_row,
                col=5,
            )
            fig.update_xaxes(visible=False, row=cd_row, col=5)
            fig.update_yaxes(visible=False, row=cd_row, col=5)

    # Actionable summary
    summary_lines = []
    if math.isfinite(laws_last):
        summary_lines.append(
            "Governing laws satisfied" if is_high_admissibility(laws_last) else "Governing-law violations detected"
        )
    if math.isfinite(const_last):
        per_key_last = (metrics[-1].get("per_key_report") if metrics else {}) or {}
        closure_summary = build_constitutive_closure_summary(per_key_last)
        if closure_summary:
            summary_lines.append(closure_summary)
        else:
            summary_lines.append(
                "Constitutive consistency acceptable"
                if is_high_admissibility(const_last)
                else "Constitutive inconsistency detected"
            )
    if not is_eval and math.isfinite(last_ov) and math.isfinite(first_ov):
        summary_lines.append("Training trend improving" if last_ov >= first_ov else "Training trend degrading")
    if show_primary_issue:
        summary_lines.append(
            "Recommend (NN-based models): Tune the optimizer and schedule, rebalance residual weights, "
            "and adjust width and depth."
        )
    summary_text = "Summary:<br>- " + "<br>- ".join(summary_lines[:5]) if summary_lines else "Summary: insufficient diagnostics"

    title_text = (figure_title or "").strip()
    fs = T["font_stack"]
    if title_text:
        esc_ft = (
            title_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        if not overall_subtitle_html.strip():
            layout_title_html = (
                f"<span style=\"font-size:{_main_title_px}px;font-weight:600;color:{font_color};font-family:{fs}\">{esc_ft}</span>"
            )
            margin_top = 72
        else:
            # Line break plus top margin so the subtitle sits clearly below the main title (~2× prior gap).
            layout_title_html = (
                f"<span style=\"font-size:{_main_title_px}px;font-weight:600;color:{font_color};font-family:{fs}\">{esc_ft}</span>"
                f"<br><span style=\"display:block;margin:88px 0 0 0;line-height:1.3\">{overall_subtitle_html}</span>"
            )
            margin_top = 152
    else:
        layout_title_html = overall_subtitle_html
        margin_top = 54 if overall_subtitle_html.strip() else 72
    div_legacy_px = div_legacy_eval if is_eval else div_legacy_train
    layout_base_h = MONITOR_SINGLE_FIGURE_HEIGHT_TRAINING if not is_eval else MONITOR_SINGLE_FIGURE_HEIGHT
    extra_px = 0
    if has_cd:
        extra_px += int(
            round(
                MONITOR_CONSTITUTIVE_DIVERGENCE_EXTRA_PX
                * (1.0 + div_legacy_px * (MONITOR_DIV_ROW_WEIGHT_MULT - 1.0))
            )
        )
    if has_state_ov:
        extra_px += MONITOR_STATE_OVERLAY_EXTRA_PX
    layout_h = layout_base_h + extra_px
    fig.update_layout(
        title=dict(
            text=layout_title_html,
            x=0.5,
            xanchor="center",
            # Minimal pad below title block so KPI row + charts sit closer to the subtitle.
            pad=dict(t=2, b=0),
            font=dict(size=12, family=fs, color=font_color),
        ),
        height=layout_h,
        showlegend=monitor_show_legend,
        # Bottom margin: spatial x labels + full summary box (border) without clipping.
        margin=dict(l=90, r=90, t=margin_top, b=MONITOR_SINGLE_FIGURE_MARGIN_BOTTOM),
        hovermode="closest",
        template="plotly_white",
        plot_bgcolor=plot_bg,
        paper_bgcolor=paper_bg,
        font=dict(size=12, family=T["font_stack"], color=font_color),
    )
    if apply_cd_dissonance_legend and cd_row is not None:
        _lx, _ly = _legend_upper_right_paper_xy_for_subplot(fig, cd_row, 5)
        fig.update_layout(
            legend=dict(
                x=_lx,
                y=_ly,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.88)",
                bordercolor="rgba(0,0,0,0.18)",
                borderwidth=1,
                font=dict(size=11, family=T["font_stack"], color=font_color),
            )
        )
    if export_buttons:
        fig.update_layout(modebar_add=["toImage"])
    if show_branding:
        fig.add_annotation(
            text="Ifimo Lab: Moju Forensic Suite",
            x=0.995,
            y=1.08,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor="right",
            font=dict(size=10, color=muted, family=T["font_stack"]),
        )
    fig.add_annotation(
        text=summary_text,
        x=0.5,
        y=MONITOR_SUMMARY_ANNOTATION_Y_PAPER,
        xref="paper",
        yref="paper",
        xanchor="center",
        yanchor="top",
        showarrow=False,
        align="left",
        font=dict(size=12, color=font_color, family=T["font_stack"]),
        bordercolor=T["summary_border"],
        borderwidth=1,
        borderpad=8,
        bgcolor=T["summary_bg"],
    )

    # Right half (col 5): Y ticks on the outer edge for constitutive / eval residuals only.
    # Category breakdown (training row 3 col 5) stays left with wrapped tick labels.
    if not is_eval:
        fig.update_yaxes(side="right", automargin=True, row=4, col=5)
    else:
        fig.update_yaxes(side="right", automargin=True, row=3, col=5)

    # Extra vertical rhythm: wider row gaps (make_subplots) + x-axis padding so tick labels
    # and axis titles clear the subplot title band of the row below.
    # Smaller standoff keeps x-axis titles closer to the plot (still clears tick labels via automargin).
    fig.update_xaxes(automargin=True, title=dict(standoff=14))
    fig.update_yaxes(automargin=True, title=dict(standoff=14))

    align_heatmap_colorbars_to_subplot_domains(fig)
    _add_monitor_panel_title_annotations(fig, subplot_titles, font_color=font_color)
    apply_theme(
        fig,
        MOJU_LIGHT,
        margin=dict(l=90, r=90, t=margin_top, b=MONITOR_SINGLE_FIGURE_MARGIN_BOTTOM),
    )
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
) -> Any:
    import plotly.graph_objects as go

    T = _ENTERPRISE_THEME
    value = float(overall_adm[-1]) if overall_adm else float("nan")
    paper = T["paper_bg"]
    fontc = T["font_color"]
    fig = go.Figure()
    ind_kwargs = dict(mode="gauge+number", value=max(0.0, min(1.0, value if math.isfinite(value) else 0.0)))
    if baseline_score is not None and math.isfinite(float(baseline_score)):
        ind_kwargs["mode"] = "gauge+number+delta"
        ind_kwargs["delta"] = {"reference": float(baseline_score), "increasing": {"color": ADMISSIBLE_COLOR}, "decreasing": {"color": DISSONANCE_COLOR}}
    ind_kwargs["gauge"] = {
        "axis": {"range": [0, 1], "tickformat": ".0%"},
        "bar": {
            "color": ADMISSIBLE_COLOR if (math.isfinite(value) and is_high_admissibility(value)) else DISSONANCE_COLOR
        },
        "steps": [
            {"range": [0, ADM_HIGH_THRESHOLD], "color": "#fee2e2"},
            {"range": [ADM_HIGH_THRESHOLD, 1.0], "color": "#d1fae5"},
        ],
    }
    fig.add_trace(
        go.Indicator(
            **ind_kwargs,
            title={"text": "Overall Admissibility A", "align": "center"},
            domain=dict(x=[0.14, 0.86], y=[0.22, 0.88]),
        )
    )
    fig.add_annotation(
        text="A = 1 / (1 + R_norm)",
        x=0.5,
        y=-0.08,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=11, family=T["font_stack"], color=fontc),
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=paper,
        plot_bgcolor=paper,
        font=dict(family=T["font_stack"], color=fontc),
        margin=dict(l=40, r=40, t=80, b=50),
        height=360,
    )
    return fig


def _build_forensic_heatmap_figure(
    bundle: Dict[str, Any], *, spatial_heatmap_colorscale: Optional[str] = None
) -> Any:
    import plotly.graph_objects as go
    import numpy as np
    from plotly.subplots import make_subplots

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
    forensic_z_range = _finite_z_range_from_array(Z)
    _ft_key = truncate_display_label(str(target), 56)

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Heatmap(
            x=x,
            y=list(range(n)),
            z=Z,
            colorscale=(spatial_heatmap_colorscale or DEFAULT_SPATIAL_HEATMAP_COLORSCALE),
            colorbar=dict(title="log10(|residual| + ε)"),
            customdata=np.broadcast_to(np.asarray(scales, dtype=float)[:, None], Z.shape),
            hovertemplate=(
                f"x=%{{x:.4g}}<br>{_ft_key}=%{{z:.4g}}<br>step=%{{y}}<br>scale_k=%{{customdata:.4g}}<extra></extra>"
            ),
            meta=dict(subplot_row=1, subplot_col=1),
            **_heatmap_zlim_kwargs(forensic_z_range),
        ),
        row=1,
        col=1,
    )
    T = _ENTERPRISE_THEME
    fig.update_layout(
        title=f"Forensic Spatial Dissonance — {truncate_display_label(str(target), 64)}",
        template="plotly_white",
        paper_bgcolor=T["paper_bg"],
        plot_bgcolor=T["plot_bg"],
        font=dict(family=T["font_stack"], color=T["font_color"]),
    )
    fig.update_xaxes(title_text="Spatial position x", row=1, col=1)
    fig.update_yaxes(title_text="Logged Step / Epoch", row=1, col=1)
    _apply_enterprise_axis_style(fig, 1, 1, y_log=False, x_grid=False, y_grid=False)
    align_heatmap_colorbars_to_subplot_domains(fig)
    return fig


def _build_state_snapshot_tab_figure(bundle: Dict[str, Any]) -> Any:
    """One-row figure mirroring embedded state overlay columns (dash-tabs only)."""
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    T = _ENTERPRISE_THEME
    snaps = list(bundle.get("state_field_snapshots") or [])
    titles: List[str] = [str(s.get("name", "—")) for s in snaps[:3]]
    while len(titles) < 3:
        titles.append("—")
    fig = make_subplots(rows=1, cols=3, subplot_titles=(titles[0], titles[1], titles[2]))
    hm_cs = DEFAULT_SPATIAL_HEATMAP_COLORSCALE
    for jcol in range(3):
        col = jcol + 1
        if jcol >= len(snaps):
            fig.add_trace(
                go.Scatter(x=[0.0], y=[0.0], mode="text", text=["—"], showlegend=False, hoverinfo="skip"),
                row=1,
                col=col,
            )
            continue
        snap = snaps[jcol]
        if snap.get("kind") == "1d":
            fig.add_trace(
                go.Scatter(
                    x=np.asarray(snap["x"], dtype=float),
                    y=np.asarray(snap["z"], dtype=float),
                    mode="lines",
                    line=dict(width=2, color=MOJU_LIGHT.palette.line_primary),
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
        elif snap.get("kind") == "2d":
            fig.add_trace(
                go.Heatmap(
                    z=np.asarray(snap["z"], dtype=float),
                    x=np.asarray(snap["x"], dtype=float),
                    y=np.asarray(snap["y"], dtype=float),
                    colorscale=hm_cs,
                    showscale=(col == 3),
                ),
                row=1,
                col=col,
            )
        else:
            fig.add_trace(
                go.Scatter(x=[0.0], y=[0.0], mode="text", text=["Unsupported"], showlegend=False, hoverinfo="skip"),
                row=1,
                col=col,
            )
    fig.update_layout(
        title=dict(text="State snapshot (predicted)", x=0.5, xanchor="center"),
        template="plotly_white",
        paper_bgcolor=T["paper_bg"],
        plot_bgcolor=T["plot_bg"],
        font=dict(family=T["font_stack"], color=T["font_color"]),
        height=340,
        margin=dict(t=72, l=72, r=72, b=64),
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
    show_branding: bool = False,
    density: str = "comfortable",
) -> Dict[str, Any]:
    _require_light_theme(theme)
    full = _build_plotly_monitor_figure_single(
        bundle,
        figure_title=figure_title,
        step_label=step_label,
        r_norm_scale=r_norm_scale,
        spatial_heatmap_colorscale=spatial_heatmap_colorscale,
        theme=theme,
        baseline_score=baseline_score,
        export_buttons=export_buttons,
        show_branding=show_branding,
        density=density,
    )
    if (bundle.get("mode") or "") == "eval":
        kpi = _build_eval_kpi_figure(bundle)
    else:
        kpi = _build_kpi_figure(bundle.get("overall_adm") or [], baseline_score=baseline_score)
    forensic = _build_forensic_heatmap_figure(bundle, spatial_heatmap_colorscale=spatial_heatmap_colorscale)
    # Optional constitutive divergence tab — only present when the engine produced
    # a closure_debug sidecar for at least one constitutive audit row.
    tabs: Dict[str, Any] = {
        "kpi": kpi,
        "admissibility": full,
        "forensic_heatmaps": forensic,
        "convergence": full,
    }
    try:
        from moju.monitor.visualize_constitutive import (
            build_constitutive_divergence_dashboard,
            list_constitutive_basenames,
        )
        if list_constitutive_basenames(bundle):
            tabs["constitutive_divergence"] = build_constitutive_divergence_dashboard(bundle)
    except Exception:  # noqa: BLE001
        # Constitutive divergence is additive; never block the rest of the dashboard.
        pass
    if isinstance(bundle.get("state_field_snapshots"), list) and bundle.get("state_field_snapshots"):
        tabs["state_snapshot"] = _build_state_snapshot_tab_figure(bundle)
    return {
        "mode": "dash-tabs",
        "tabs": tabs,
        "filter_contract": {"bar_customdata_field": "category", "threshold": 0.99},
        "toggles": {"mode": ["training", "eval"]},
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
    show_branding: bool = False,
    density: str = "comfortable",
) -> Any:
    _require_light_theme(theme)
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
            show_branding=show_branding,
            density=density,
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
        show_branding=show_branding,
        density=density,
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
    Tc = _ENTERPRISE_THEME

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
                marker=dict(color=bar_colors, line=dict(color=Tc["bar_line"], width=0.5)),
                showlegend=False,
                hovertemplate="%{x}<br>"
                + ("log10(R_norm+ε)=%{y:.4g}<extra></extra>" if use_log else "R norm=%{y:.4g}<extra></extra>"),
            )
        )
        fig.update_xaxes(title_text="Governing law (residual key)", automargin=True)
        y_ax = "log10(R_norm + ε)" if use_log else "Normalized residual (R norm)"
        fig.update_yaxes(title_text=y_ax, automargin=True)
        _apply_enterprise_axis_style_xy(fig, y_log=use_log, x_grid=False, y_grid=True)

    fig.update_layout(
        title=dict(text="Law R_norm (final step)", font=dict(size=14, family=Tc["font_stack"])),
        height=card_height,
        margin=dict(l=12, r=12, t=48, b=48),
        template="plotly_white",
        paper_bgcolor=Tc["paper_bg"],
        plot_bgcolor=Tc["plot_bg"],
        font=dict(size=11, family=Tc["font_stack"], color=Tc["font_color"]),
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
    bx = [(float(v) * 100.0) if math.isfinite(v) else 0.0 for v in bvals]
    btext = [format_admissibility_pct(v) if math.isfinite(v) else "N/A" for v in bvals]
    adm_ht = [format_admissibility_pct(v) if math.isfinite(v) else "N/A" for v in bvals]

    Tc = _ENTERPRISE_THEME
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=bx,
            y=blabels,
            orientation="h",
            marker=dict(color=bcolors, line=dict(color=Tc["bar_line"], width=0.5)),
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
        tickformat=".0f",
        automargin=True,
    )
    fig.update_yaxes(automargin=True)
    _apply_enterprise_axis_style_xy(fig, y_log=False, x_grid=True, y_grid=False)
    fig.update_layout(
        title=dict(text="Category admissibility (final step)", font=dict(size=14, family=Tc["font_stack"])),
        height=card_height,
        margin=dict(l=12, r=80, t=48, b=48),
        template="plotly_white",
        paper_bgcolor=Tc["paper_bg"],
        plot_bgcolor=Tc["plot_bg"],
        font=dict(size=11, family=Tc["font_stack"], color=Tc["font_color"]),
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

    Tc = _ENTERPRISE_THEME
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
            font=dict(size=12, family=Tc["font_stack"], color=Tc["font_color"]),
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
    else:
        full_title = _spatial_heatmap_subplot_title(spatial_parsed, card_title)
        kind = spatial_parsed.get("kind", "1d")
        cb = dict(
            title=dict(text=cb_default, side="right", font=dict(size=10)),
            thickness=HEATMAP_COLORBAR_THICKNESS,
            xpad=HEATMAP_COLORBAR_XPAD,
        )
        if kind == "1d":
            Z = spatial_parsed["Z"]
            x_sp = spatial_parsed["x"]
            row_labels = spatial_parsed["row_labels"]
            card_z_range = _finite_z_range_from_array(Z)
            fig.add_trace(
                go.Heatmap(
                    x=x_sp,
                    y=list(range(len(row_labels))),
                    z=Z,
                    colorscale=colorscale,
                    colorbar=cb,
                    hovertemplate="x=%{x:.4g}<br>%{customdata}=%{z:.4g}<extra></extra>",
                    meta=dict(subplot_row=1, subplot_col=1),
                    customdata=np.broadcast_to(
                        np.asarray(row_labels, dtype=object)[:, np.newaxis],
                        (len(row_labels), len(x_sp)),
                    ),
                    **_heatmap_zlim_kwargs(card_z_range),
                )
            )
            _enterprise_axis_frame_xy(fig, grid=False, hide_y_ticklabels=True)
            fig.update_xaxes(title_text=_pos_axis_title_card(spatial_parsed))
        elif kind == "2d":
            Zs = spatial_parsed["Z"]
            x_sp = np.asarray(spatial_parsed["x"], dtype=float)
            y_sp = np.asarray(spatial_parsed["y"], dtype=float)
            row_labels = spatial_parsed["row_labels"]
            nk = int(Zs.shape[0])
            card_z_range = _finite_z_range_from_array(Zs[0])
            _ck = truncate_display_label(str(row_labels[0]), 48) if row_labels else "residual"
            fig.add_trace(
                go.Heatmap(
                    x=x_sp,
                    y=y_sp,
                    z=Zs[0],
                    colorscale=colorscale,
                    colorbar=cb,
                    hovertemplate=f"x=%{{x:.4g}}<br>y=%{{y:.4g}}<br>{_ck}=%{{z:.4g}}<extra></extra>",
                    meta=dict(subplot_row=1, subplot_col=1),
                    **_heatmap_zlim_kwargs(card_z_range),
                )
            )
            _enterprise_axis_frame_xy(fig, grid=False)
            fig.update_xaxes(title_text="x")
            fig.update_yaxes(title_text="y")
            if nk > 1:
                fig.add_annotation(
                    text=f"First of {nk} keys: {truncate_display_label(row_labels[0], 40)}",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=1.08,
                    showarrow=False,
                    yanchor="bottom",
                    font=dict(size=10, family=Tc["font_stack"], color=Tc["muted"]),
                )
        else:
            V = np.asarray(spatial_parsed["V"][0], dtype=float)
            x_sp = np.asarray(spatial_parsed["x"], dtype=float)
            y_sp = np.asarray(spatial_parsed["y"], dtype=float)
            z_sp = np.asarray(spatial_parsed["z"], dtype=float)
            nk = int(spatial_parsed["V"].shape[0])
            vol_c_range = _finite_z_range_from_array(V)
            vol_kw: Dict[str, Any] = {}
            if vol_c_range is not None:
                vol_kw["cmin"], vol_kw["cmax"] = vol_c_range
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
                    **vol_kw,
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
                    font=dict(size=10, family=Tc["font_stack"], color=Tc["muted"]),
                )

    fig.update_layout(
        title=dict(text=full_title, font=dict(size=14, family=Tc["font_stack"])),
        height=card_height,
        margin=dict(l=8, r=100, t=48, b=48),
        template="plotly_white",
        paper_bgcolor=Tc["paper_bg"],
        plot_bgcolor=Tc["plot_bg"],
        font=dict(size=11, family=Tc["font_stack"], color=Tc["font_color"]),
    )
    if spatial_parsed is not None and spatial_parsed.get("kind", "1d") in ("1d", "2d"):
        align_heatmap_colorbars_to_subplot_domains(fig)
    return fig
