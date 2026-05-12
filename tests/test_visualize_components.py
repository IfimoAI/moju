"""Unit tests for the reusable component library."""

from __future__ import annotations

import numpy as np
import pytest


def _mock_bundle() -> dict:
    return {
        "log": [{"overall_admissibility_score": 0.84, "coord_snapshot": {"x": list(np.linspace(0, 1, 6))}}],
        "indices": [0, 1, 2, 3, 4],
        "overall_adm": [0.50, 0.60, 0.70, 0.78, 0.84],
        "category_training": {
            "laws": [0.40, 0.55, 0.65, 0.72, 0.80],
            "constitutive": [0.60, 0.70, 0.75, 0.80, 0.86],
        },
        "cats_fin": {"laws": 0.80, "constitutive": 0.86, "scaling": 0.78},
        "plot_keys": ["laws/foo", "laws/bar", "constitutive/baz/implied_delta"],
        "r_norm_mat": [
            [0.10, 0.20, 0.15],
            [0.08, 0.18, 0.12],
            [0.05, 0.15, 0.10],
            [0.04, 0.13, 0.09],
            [0.03, 0.12, 0.08],
        ],
        "spatial": {"coords": {"x": np.linspace(0, 1, 6), "y": np.linspace(0, 1, 5)}},
        "residuals": {
            "laws/foo": np.random.rand(5, 6),
        },
        "worst_keys_rows": [
            {"key": "laws/foo", "r_norm": 0.03, "admissibility_score": 0.92},
            {"key": "constitutive/baz/implied_delta", "r_norm": 0.08, "admissibility_score": 0.60},
        ],
    }


@pytest.fixture
def mock_bundle() -> dict:
    return _mock_bundle()


def test_kpi_card_returns_indicator(mock_bundle: dict) -> None:
    pytest.importorskip("plotly")
    from moju.monitor.visualize_components import build_overall_admissibility_kpi

    fig = build_overall_admissibility_kpi(mock_bundle)
    assert fig.data[0].type == "indicator"
    assert 0 <= fig.data[0].value <= 100


def test_admissibility_timeline_has_overall_and_categories(mock_bundle: dict) -> None:
    pytest.importorskip("plotly")
    from moju.monitor.visualize_components import build_admissibility_timeline_card

    fig = build_admissibility_timeline_card(mock_bundle)
    names = [tr.name for tr in fig.data]
    assert "Overall" in names
    assert any("Law" in n for n in names) or any("Constitutive" in n for n in names)


def test_category_bar_card(mock_bundle: dict) -> None:
    pytest.importorskip("plotly")
    from moju.monitor.visualize_components import build_category_admissibility_bar_card

    fig = build_category_admissibility_bar_card(mock_bundle)
    assert fig.data[0].type == "bar"
    assert fig.data[0].orientation == "h"


def test_rnorm_timeline_card(mock_bundle: dict) -> None:
    pytest.importorskip("plotly")
    from moju.monitor.visualize_components import build_rnorm_timeline_card

    fig = build_rnorm_timeline_card(mock_bundle, max_keys=3)
    assert len(fig.data) >= 1
    # log y-axis
    assert fig.layout.yaxis.type in ("log", None)


def test_law_rnorm_final_bar_card(mock_bundle: dict) -> None:
    pytest.importorskip("plotly")
    from moju.monitor.visualize_components import build_law_rnorm_final_bar_card

    fig = build_law_rnorm_final_bar_card(mock_bundle)
    assert fig.data[0].type == "bar"


def test_spatial_residual_heatmap_card(mock_bundle: dict) -> None:
    pytest.importorskip("plotly")
    from moju.monitor.visualize_components import build_spatial_residual_heatmap_card

    fig = build_spatial_residual_heatmap_card(mock_bundle, key="laws/foo")
    assert fig.data[0].type == "heatmap"
    # Sequential colorscale, no Jet
    cs = fig.data[0].colorscale
    if cs is None:
        cs_str = ""
    else:
        cs_str = str(cs)
    assert "jet" not in cs_str.lower()


def test_field_explorer_card_auto_modes() -> None:
    pytest.importorskip("plotly")
    from moju.monitor.visualize_components import build_field_explorer_card

    # 1D
    fig = build_field_explorer_card(np.sin(np.linspace(0, 6.28, 32)), title="t1")
    assert fig.data[0].type == "scatter"
    # 2D
    fig = build_field_explorer_card(np.random.rand(8, 10), title="t2")
    assert fig.data[0].type == "heatmap"
    # No Jet
    cs_str = str(fig.data[0].colorscale)
    assert "jet" not in cs_str.lower()


def test_worst_keys_table_card(mock_bundle: dict) -> None:
    pytest.importorskip("plotly")
    from moju.monitor.visualize_components import build_worst_keys_table_card

    fig = build_worst_keys_table_card(mock_bundle, limit=2)
    assert fig.data[0].type == "table"


def test_empty_bundle_gracefully_returns_message() -> None:
    pytest.importorskip("plotly")
    from moju.monitor.visualize_components import build_overall_admissibility_kpi

    fig = build_overall_admissibility_kpi({})
    # Empty case has an annotation explaining the missing data
    assert fig.layout.annotations
