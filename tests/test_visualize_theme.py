"""Unit tests for :mod:`moju.monitor.visualize_theme`."""

from __future__ import annotations

from dataclasses import replace

import pytest


def test_themes_resolve_by_name() -> None:
    from moju.monitor.visualize_theme import MOJU_DARK, MOJU_LIGHT, get_theme

    assert get_theme("light") is MOJU_LIGHT
    assert get_theme("dark") is MOJU_DARK
    assert get_theme(MOJU_LIGHT) is MOJU_LIGHT  # passthrough


def test_unknown_theme_raises() -> None:
    from moju.monitor.visualize_theme import get_theme

    with pytest.raises(ValueError, match="Unknown theme"):
        get_theme("rainbow")


def test_apply_theme_sets_background_and_font() -> None:
    pytest.importorskip("plotly")
    import plotly.graph_objects as go
    from moju.monitor.visualize_theme import MOJU_DARK, MOJU_LIGHT, apply_theme

    fig = go.Figure(data=[go.Scatter(x=[0, 1], y=[0, 1])])
    apply_theme(fig, MOJU_LIGHT, title="Hello", height=320)
    assert fig.layout.paper_bgcolor == "#ffffff"
    assert "Inter" in fig.layout.font.family
    assert fig.layout.height == 320
    assert fig.layout.title.text == "Hello"

    apply_theme(fig, MOJU_DARK)
    assert fig.layout.paper_bgcolor == "#0b1220"


def test_colorscales_no_jet_by_default() -> None:
    from moju.monitor.visualize_theme import COLORSCALES, MOJU_LIGHT

    assert MOJU_LIGHT.colorscales.sequential == "Viridis"
    assert "Jet" not in COLORSCALES.values()
    assert MOJU_LIGHT.colorscales.diverging == "RdBu_r"


def test_theme_customization_via_replace() -> None:
    from moju.monitor.visualize_theme import MOJU_LIGHT

    custom_palette = replace(MOJU_LIGHT.palette, line_primary="#ff0000")
    my = replace(MOJU_LIGHT, name="brand", palette=custom_palette)
    assert my.palette.line_primary == "#ff0000"
    # Original untouched
    assert MOJU_LIGHT.palette.line_primary == "#1d4ed8"


def test_themed_colorbar_has_title_and_font() -> None:
    from moju.monitor.visualize_theme import themed_colorbar

    cb = themed_colorbar("light", title="value")
    assert cb["title"]["text"] == "value"
    assert "Inter" in cb["tickfont"]["family"]
    assert isinstance(cb["len"], float)
