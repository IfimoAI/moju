"""
Unified enterprise-grade design system for every Moju Plotly visualization.

A single source of truth shared by:
- :mod:`moju.monitor.visualize_plotly` (the audit dashboard)
- :mod:`moju.monitor.visualize_components` (the reusable card library)
- :mod:`moju.monitor.visualize_constitutive` (the constitutive-divergence card)
- :mod:`apps.moju_studio.studio_plots` (Studio exploration helpers)

Public surface
--------------

- :class:`MojuTheme` — frozen dataclass capturing palette, typography,
  colorscales, layout tokens, and admissibility/category color hexes.
- :data:`MOJU_LIGHT`, :data:`MOJU_DARK` — two canonical themes (light is the
  default).
- :func:`get_theme(name)` — resolve by name (``"light"`` / ``"dark"``).
- :func:`apply_theme(fig, theme=MOJU_LIGHT, *, margin=None)` — set template,
  fonts, and background on a :class:`plotly.graph_objects.Figure`.
- :data:`COLORSCALES` — registry of perceptually uniform colorscales
  (``sequential``, ``sequential_alt``, ``diverging``, ``residual``,
  ``divergence``).  ``Jet`` is intentionally excluded.

Customisation
-------------

>>> from dataclasses import replace
>>> from moju.monitor.visualize_theme import MOJU_LIGHT
>>> my_theme = replace(MOJU_LIGHT, name="brand", palette=replace(
...     MOJU_LIGHT.palette, line_primary="#0066ff"))
>>> apply_theme(fig, my_theme)
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Atomic components
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Palette:
    """Background, foreground, and accent colors (light or dark)."""

    plot_bg: str
    paper_bg: str
    font_color: str
    title_color: str
    muted: str
    tick_color: str
    axis_line: str
    grid_color: str
    zeroline_color: str
    line_primary: str
    bar_line: str
    summary_border: str
    summary_bg: str
    # Admissibility tokens (apply across modes)
    adm_high: str = "#10B981"  # emerald
    adm_med: str = "#F59E0B"   # amber
    adm_low: str = "#EF4444"   # red
    adm_fail: str = "#B91C1C"  # deep red
    # Per-category residual colors (aligned with auditor._build_visualize_bundle)
    cat_laws: str = "#8B5CF6"          # royal purple
    cat_constitutive: str = "#14B8A6"  # teal
    cat_scaling: str = "#59A14F"       # leaf green
    cat_data: str = "#B07AA1"          # mauve
    cat_other: str = "#6B7280"         # neutral slate

    # Misc accents
    accent_neutral: str = "#000000"
    accent_warn: str = "#E67E22"


@dataclass(frozen=True)
class Typography:
    """Font stack and sizing for body / titles / ticks."""

    font_family: str = "Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
    base_size: int = 11
    title_size: int = 15
    section_title_size: int = 13
    tick_size: int = 11


@dataclass(frozen=True)
class Colorscales:
    """Perceptually uniform colorscale defaults."""

    sequential: str = "Viridis"
    sequential_alt: str = "Cividis"
    diverging: str = "RdBu_r"
    residual: str = "Viridis"
    divergence: str = "RdBu_r"
    high_dim: str = "Turbo"  # only when an extreme dynamic range warrants it

    def as_dict(self) -> Dict[str, str]:
        return {
            "sequential": self.sequential,
            "sequential_alt": self.sequential_alt,
            "diverging": self.diverging,
            "residual": self.residual,
            "divergence": self.divergence,
            "high_dim": self.high_dim,
        }


@dataclass(frozen=True)
class LayoutTokens:
    """Spacing, sizing, and structural constants shared across cards."""

    card_height: int = 400
    card_gutter: int = 24
    label_pad: int = 8
    margin: Tuple[int, int, int, int] = (82, 98, 104, 86)  # l, r, t, b
    margin_pad: int = 6
    legend_y_top: float = 1.02
    legend_x_right: float = 1.0
    colorbar_len: float = 0.62
    colorbar_thickness: int = 12
    colorbar_xpad: int = 14

    def margin_dict(self) -> Dict[str, int]:
        l, r, t, b = self.margin
        return dict(l=l, r=r, t=t, b=b, pad=self.margin_pad)


# ---------------------------------------------------------------------------
# Top-level theme
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MojuTheme:
    """
    Full enterprise theme: palette + typography + colorscales + layout.

    Use :data:`MOJU_LIGHT` (default) or :data:`MOJU_DARK`, or build a custom
    theme with :func:`dataclasses.replace`.
    """

    name: str
    palette: Palette
    typography: Typography = field(default_factory=Typography)
    colorscales: Colorscales = field(default_factory=Colorscales)
    layout: LayoutTokens = field(default_factory=LayoutTokens)
    template: str = "plotly_white"  # set to "plotly_dark" for dark mode

    # ----- Convenience accessors -----

    def font_dict(self, *, size: Optional[int] = None, color: Optional[str] = None) -> Dict[str, Any]:
        return dict(
            family=self.typography.font_family,
            size=size if size is not None else self.typography.base_size,
            color=color if color is not None else self.palette.font_color,
        )

    def title_font_dict(self) -> Dict[str, Any]:
        return dict(
            family=self.typography.font_family,
            size=self.typography.title_size,
            color=self.palette.title_color,
        )

    def section_title_font_dict(self) -> Dict[str, Any]:
        return dict(
            family=self.typography.font_family,
            size=self.typography.section_title_size,
            color=self.palette.title_color,
        )

    def tick_font_dict(self) -> Dict[str, Any]:
        return dict(
            family=self.typography.font_family,
            size=self.typography.tick_size,
            color=self.palette.tick_color,
        )


# ---------------------------------------------------------------------------
# Canonical themes
# ---------------------------------------------------------------------------


_LIGHT_PALETTE = Palette(
    plot_bg="#ffffff",
    paper_bg="#ffffff",
    font_color="#0f172a",
    title_color="#0f172a",
    muted="#64748b",
    tick_color="#334155",
    axis_line="#64748b",
    grid_color="#e2e8f0",
    zeroline_color="#cbd5e1",
    line_primary="#1d4ed8",
    bar_line="#94a3b8",
    summary_border="#334155",
    summary_bg="rgba(241, 245, 249, 0.85)",
)


_DARK_PALETTE = Palette(
    plot_bg="#0f172a",
    paper_bg="#0b1220",
    font_color="#e2e8f0",
    title_color="#f8fafc",
    muted="#94a3b8",
    tick_color="#cbd5e1",
    axis_line="#94a3b8",
    grid_color="#334155",
    zeroline_color="#475569",
    line_primary="#60a5fa",
    bar_line="#64748b",
    summary_border="#94a3b8",
    summary_bg="rgba(15, 23, 42, 0.85)",
    # Admissibility tokens adjusted for legibility on dark
    adm_high="#34d399",
    adm_med="#fbbf24",
    adm_low="#f87171",
    adm_fail="#fca5a5",
    # Slightly brighter category hues for dark-mode contrast
    cat_laws="#a78bfa",
    cat_constitutive="#5eead4",
    cat_scaling="#86efac",
    cat_data="#f0abfc",
    cat_other="#9ca3af",
)


MOJU_LIGHT: MojuTheme = MojuTheme(name="light", palette=_LIGHT_PALETTE, template="plotly_white")
MOJU_DARK: MojuTheme = MojuTheme(name="dark", palette=_DARK_PALETTE, template="plotly_dark")


_THEME_REGISTRY: Dict[str, MojuTheme] = {
    "light": MOJU_LIGHT,
    "dark": MOJU_DARK,
}


def get_theme(name_or_theme: Any = "light") -> MojuTheme:
    """
    Resolve a theme by name (``"light"`` / ``"dark"``) or pass through a
    :class:`MojuTheme` unchanged.
    """
    if isinstance(name_or_theme, MojuTheme):
        return name_or_theme
    key = str(name_or_theme).lower()
    if key not in _THEME_REGISTRY:
        raise ValueError(
            f"Unknown theme {name_or_theme!r}; available: {sorted(_THEME_REGISTRY)}"
        )
    return _THEME_REGISTRY[key]


# Public colorscale registry — explicit hash for callers that want a string key.
COLORSCALES: Dict[str, str] = MOJU_LIGHT.colorscales.as_dict()


# ---------------------------------------------------------------------------
# Apply theme to a Plotly figure
# ---------------------------------------------------------------------------


def apply_theme(
    fig: Any,
    theme: Any = MOJU_LIGHT,
    *,
    margin: Optional[Dict[str, int]] = None,
    title: Optional[str] = None,
    height: Optional[int] = None,
) -> Any:
    """
    Apply an enterprise theme to a Plotly figure.

    Sets template, background, font, default margins, and (optionally) the
    figure title.  Returns the figure for chaining.

    Parameters
    ----------
    fig:
        :class:`plotly.graph_objects.Figure`.
    theme:
        A :class:`MojuTheme` or theme name string (``"light"``, ``"dark"``).
    margin:
        Override the theme's default margin dict.
    title:
        Optional figure title text (centered, themed font).
    height:
        Optional figure height in pixels.
    """
    t = get_theme(theme)
    layout_update: Dict[str, Any] = dict(
        template=t.template,
        plot_bgcolor=t.palette.plot_bg,
        paper_bgcolor=t.palette.paper_bg,
        font=t.font_dict(),
        margin=margin if margin is not None else t.layout.margin_dict(),
    )
    if title is not None:
        layout_update["title"] = dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=t.title_font_dict(),
            pad=dict(t=8, b=6),
        )
    if height is not None:
        layout_update["height"] = int(height)
    fig.update_layout(**layout_update)
    return fig


def themed_colorbar(
    theme: Any = MOJU_LIGHT,
    *,
    title: Optional[str] = None,
    side: str = "right",
) -> Dict[str, Any]:
    """Themed colorbar dict for Heatmap / Surface / Volume traces."""
    t = get_theme(theme)
    cb: Dict[str, Any] = dict(
        len=t.layout.colorbar_len,
        thickness=t.layout.colorbar_thickness,
        xpad=t.layout.colorbar_xpad,
        tickfont=t.tick_font_dict(),
    )
    if title is not None:
        cb["title"] = dict(text=title, side=side, font=t.section_title_font_dict())
    return cb


def themed_axis_style(
    theme: Any = MOJU_LIGHT,
    *,
    show_grid: bool = True,
    zero_line: bool = True,
) -> Dict[str, Any]:
    """Reusable axis-style dict for ``update_xaxes`` / ``update_yaxes``."""
    t = get_theme(theme)
    return dict(
        showline=True,
        linewidth=1,
        mirror=True,
        linecolor=t.palette.axis_line,
        tickcolor=t.palette.axis_line,
        showgrid=show_grid,
        gridcolor=t.palette.grid_color,
        gridwidth=1,
        minor_showgrid=False,
        zeroline=zero_line,
        zerolinecolor=t.palette.zeroline_color,
        tickfont=t.tick_font_dict(),
        title_font=t.section_title_font_dict(),
        automargin=True,
    )


__all__ = [
    "Palette",
    "Typography",
    "Colorscales",
    "LayoutTokens",
    "MojuTheme",
    "MOJU_LIGHT",
    "MOJU_DARK",
    "COLORSCALES",
    "get_theme",
    "apply_theme",
    "themed_colorbar",
    "themed_axis_style",
    "replace",
]
