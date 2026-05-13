"""
Unit tests for the Constitutive Divergence card (all four modes, balance +
subtract closures, composite dashboard).
"""

from __future__ import annotations

import numpy as np
import pytest


def _balance_bundle(nx: int = 8, ny: int = 6) -> dict:
    rng = np.random.default_rng(0)
    T_t = rng.standard_normal((ny, nx)) * 0.1
    T_lap = rng.standard_normal((ny, nx)) * 0.05
    alpha_pred = np.full((ny, nx), 1.5e-5)
    scale_a = T_t
    scale_b = alpha_pred * T_lap
    raw = scale_a - scale_b
    return {
        "log": [
            {
                "overall_admissibility_score": 0.66,
                "coord_snapshot": {
                    "x": list(np.linspace(0, 1, nx * ny)),
                    "y": list(np.linspace(0, 1, nx * ny)),
                },
            }
        ],
        "indices": [0],
        "plot_keys": ["constitutive/thermal_diffusivity/law_fourier_conduction/implied_delta"],
        "r_norm_mat": [[0.42]],
        "spatial": {"coords": {"x": np.linspace(0, 1, nx), "y": np.linspace(0, 1, ny)}},
        "closure_debug": {
            "thermal_diffusivity/law_fourier_conduction": {
                "pred": alpha_pred,
                "implied": None,
                "raw": raw,
                "scale_a": scale_a,
                "scale_b": scale_b,
                "ref": None,
                "mode": "balance",
                "output_key": "alpha",
                "law_name": "fourier_conduction",
                "model_name": "thermal_diffusivity",
            },
        },
    }


def _subtract_bundle(n: int = 64) -> dict:
    rng = np.random.default_rng(1)
    pred = np.linspace(1.0, 2.0, n)
    implied = pred + rng.standard_normal(n) * 0.04
    return {
        "log": [
            {
                "overall_admissibility_score": 0.91,
                "coord_snapshot": {"x": list(np.linspace(0, 1, n))},
            }
        ],
        "indices": [0],
        "plot_keys": ["constitutive/ideal_gas_rho/law_mass_compressible/implied_delta"],
        "r_norm_mat": [[0.04]],
        "spatial": {"coords": {"x": np.linspace(0, 1, n)}},
        "closure_debug": {
            "ideal_gas_rho/law_mass_compressible": {
                "pred": pred,
                "implied": implied,
                "raw": pred - implied,
                "scale_a": None,
                "scale_b": None,
                "ref": None,
                "mode": "subtract",
                "output_key": "rho",
                "law_name": "mass_compressible",
                "model_name": "ideal_gas_rho",
            }
        },
    }


@pytest.mark.parametrize("mode", ["spatial", "scatter", "distribution", "hotspot"])
def test_balance_mode_all_panels(mode: str) -> None:
    pytest.importorskip("plotly")
    from moju.monitor.visualize_constitutive import build_constitutive_divergence_card

    bundle = _balance_bundle()
    fig = build_constitutive_divergence_card(bundle, mode=mode)
    assert fig is not None
    assert len(fig.data) >= 1


@pytest.mark.parametrize("mode", ["spatial", "scatter", "distribution", "hotspot"])
def test_subtract_mode_all_panels(mode: str) -> None:
    pytest.importorskip("plotly")
    from moju.monitor.visualize_constitutive import build_constitutive_divergence_card

    bundle = _subtract_bundle()
    fig = build_constitutive_divergence_card(bundle, mode=mode)
    assert fig is not None
    assert len(fig.data) >= 1


def test_constitutive_dashboard_composes_four_panels() -> None:
    pytest.importorskip("plotly")
    from moju.monitor.visualize_constitutive import build_constitutive_divergence_dashboard

    bundle = _balance_bundle()
    fig = build_constitutive_divergence_dashboard(bundle)
    # 2x2 composite: spatial trio + scatter (1) + hist (1) + hotspot (1) = at least 6
    assert len(fig.data) >= 6


def test_list_basenames_sorted_by_worst_rnorm() -> None:
    from moju.monitor.visualize_constitutive import list_constitutive_basenames

    bundle = _balance_bundle()
    bundle["closure_debug"]["second/law_other"] = {
        "pred": np.array([1.0]),
        "implied": np.array([1.0]),
        "raw": np.array([0.0]),
        "scale_a": None,
        "scale_b": None,
        "ref": None,
        "mode": "subtract",
        "output_key": "x",
        "law_name": "other",
        "model_name": "second",
    }
    bundle["plot_keys"] = [
        "constitutive/thermal_diffusivity/law_fourier_conduction/implied_delta",
        "constitutive/second/law_other/implied_delta",
    ]
    bundle["r_norm_mat"] = [[0.42, 0.01]]
    names = list_constitutive_basenames(bundle)
    assert names[0] == "thermal_diffusivity/law_fourier_conduction"
    assert "second/law_other" in names


def test_empty_bundle_returns_themed_empty_card() -> None:
    pytest.importorskip("plotly")
    from moju.monitor.visualize_constitutive import build_constitutive_divergence_card

    fig = build_constitutive_divergence_card({}, mode="spatial")
    assert fig.layout.annotations  # has a "no data" message


def test_unknown_mode_raises() -> None:
    from moju.monitor.visualize_constitutive import build_constitutive_divergence_card

    with pytest.raises(ValueError, match="Unknown divergence mode"):
        build_constitutive_divergence_card(_subtract_bundle(), mode="histogram")


def test_spatial_normalized_only_single_trace_2d() -> None:
    pytest.importorskip("plotly")
    from moju.monitor.visualize_constitutive import build_spatial_normalized_divergence_figure

    fig = build_spatial_normalized_divergence_figure(_balance_bundle())
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"


def test_spatial_normalized_heatmap_uses_time_axis_label() -> None:
    pytest.importorskip("plotly")
    from moju.monitor.visualize_constitutive import build_spatial_normalized_divergence_figure

    nt, nx = 4, 5
    pred = np.ones((nt, nx))
    implied = pred * 1.1
    bundle = {
        "log": [
            {
                "coord_snapshot": {
                    "t": list(np.linspace(0.0, 1.0, nt)),
                    "x": list(np.linspace(0.0, 2.0, nx)),
                },
            }
        ],
        "plot_keys": ["constitutive/c/law_x/implied_delta"],
        "r_norm_mat": [[0.1]],
        "spatial_coord_hint": "x",
        "closure_debug": {
            "c/law_x": {
                "pred": pred,
                "implied": implied,
                "raw": pred - implied,
                "scale_a": None,
                "scale_b": None,
                "ref": None,
                "mode": "subtract",
                "output_key": "alpha",
            }
        },
    }
    fig = build_spatial_normalized_divergence_figure(bundle)
    assert fig.data[0].type == "heatmap"
    assert fig.layout.xaxis.title.text == "Position x"
    assert fig.layout.yaxis.title.text == "Time t"


def test_spatial_normalized_accepts_scalar_model_with_field_implied() -> None:
    pytest.importorskip("plotly")
    from moju.monitor.visualize_constitutive import (
        build_spatial_normalized_divergence_figure,
        prepare_constitutive_model_implied_vs_x_embed,
    )

    x = np.linspace(0.0, 1.0, 6)
    implied = np.full_like(x, 2.0)
    bundle = {
        "log": [{"coord_snapshot": {"x": list(x)}}],
        "plot_keys": ["constitutive/c/law_x/implied_delta"],
        "r_norm_mat": [[0.0]],
        "spatial_coord_hint": "x",
        "closure_debug": {
            "c/law_x": {
                "pred": np.array(2.0),
                "implied": implied,
                "raw": np.zeros_like(implied),
                "scale_a": None,
                "scale_b": None,
                "ref": None,
                "mode": "subtract",
                "output_key": "alpha",
            }
        },
    }
    fig = build_spatial_normalized_divergence_figure(bundle)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    emb = prepare_constitutive_model_implied_vs_x_embed(bundle)
    assert emb is not None
    # 5 band-fill traces + 2 named line traces (model + implied)
    assert len(emb["traces"]) == 7
    # y_range must be present and well-formed
    assert "y_range" in emb
    assert len(emb["y_range"]) == 2
    assert emb["y_range"][0] < emb["y_range"][1]


def test_spatial_normalized_only_single_trace_1d() -> None:
    pytest.importorskip("plotly")
    from moju.monitor.visualize_constitutive import build_spatial_normalized_divergence_figure

    fig = build_spatial_normalized_divergence_figure(_subtract_bundle())
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"


def test_spatial_normalized_heatmap_uses_user_term_title_and_colorbar() -> None:
    pytest.importorskip("plotly")
    from moju.monitor.visualize_constitutive import build_spatial_normalized_divergence_figure

    fig = build_spatial_normalized_divergence_figure(_balance_bundle())
    assert str(fig.layout.title.text) == "Constitutive Divergence (Thermal Diffusivity)"
    hm = fig.data[0]
    assert hm.type == "heatmap"
    cb = hm.to_plotly_json().get("colorbar") or {}
    cb_title = cb.get("title")
    if isinstance(cb_title, dict):
        cb_title = cb_title.get("text") or ""
    assert str(cb_title) == "Normalized delta"


def test_spatial_three_panel_card_unchanged_trace_count() -> None:
    pytest.importorskip("plotly")
    from moju.monitor.visualize_constitutive import build_constitutive_divergence_card

    fig = build_constitutive_divergence_card(_balance_bundle(), mode="spatial")
    assert len(fig.data) == 3


def test_prepare_mi_vs_x_embed_2d_balance() -> None:
    pytest.importorskip("plotly")
    from moju.monitor.visualize_constitutive import prepare_constitutive_model_implied_vs_x_embed

    emb = prepare_constitutive_model_implied_vs_x_embed(_balance_bundle())
    assert emb is not None
    # 5 band-fill traces + 2 named line traces (model + implied)
    assert len(emb["traces"]) == 7
    assert "y_range" in emb
    assert emb["y_range"][0] < emb["y_range"][1]
    # last two traces are the model/implied line traces
    t0, t1 = emb["traces"][-2], emb["traces"][-1]
    assert t0.type == "scatter" and t1.type == "scatter"
    from moju.monitor.visualize_theme import MOJU_LIGHT

    assert t0.line.dash == "dash"
    assert t0.line.color == MOJU_LIGHT.palette.line_primary
    assert t1.line.color == MOJU_LIGHT.palette.title_color
    assert emb["y_title"] == "Thermal Diffusivity"
    assert emb["term_label"] == "Thermal Diffusivity"
    assert emb["title"] == "Constitutive Dissonance (worst slice)"
    assert emb["x_title"] == "Position x"
    assert "(0 to L)" not in emb["x_title"]
    assert len(t0.x) == len(t0.y) == len(t1.y)
    xs = list(t0.x)
    assert xs == sorted(xs) or len(xs) <= 1


def test_prepare_mi_vs_x_embed_1d_subtract() -> None:
    pytest.importorskip("plotly")
    from moju.monitor.visualize_constitutive import prepare_constitutive_model_implied_vs_x_embed

    emb = prepare_constitutive_model_implied_vs_x_embed(_subtract_bundle())
    assert emb is not None
    # 5 band-fill traces + 2 named line traces (model + implied)
    assert len(emb["traces"]) == 7
    assert "y_range" in emb
    assert emb["y_range"][0] < emb["y_range"][1]
    from moju.monitor.visualize_theme import MOJU_LIGHT

    # last two traces are the model/implied line traces
    assert emb["traces"][-2].line.dash == "dash"
    assert emb["traces"][-2].line.color == MOJU_LIGHT.palette.line_primary
    assert emb["traces"][-1].line.color == MOJU_LIGHT.palette.title_color
    assert emb["y_title"] == "Ideal Gas Rho"
    assert emb["title"] == "Constitutive Dissonance"
    assert emb["x_title"] == "Position x"
    assert "(0 to L)" not in emb["x_title"]
    n = 64
    assert len(emb["traces"][-2].x) == n


def test_worst_div_mean_abs_row_index_deterministic() -> None:
    from moju.monitor.visualize_constitutive import worst_div_mean_abs_row_index

    div = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=float)
    assert worst_div_mean_abs_row_index(div) == 1


def test_prepare_mi_vs_x_prefers_last_time_slice() -> None:
    pytest.importorskip("plotly")
    from moju.monitor.visualize_constitutive import prepare_constitutive_model_implied_vs_x_embed

    nx, ny = 4, 3
    rng = np.random.default_rng(42)
    # Two time slices; slice 1 differs so last-t reduction is visible.
    pred = np.stack(
        [
            rng.standard_normal((ny, nx)) * 0.01,
            np.full((ny, nx), 999.0),
        ],
        axis=0,
    )
    implied = np.stack([np.full((ny, nx), 1.4), np.full((ny, nx), 1.4)], axis=0)
    bundle = {
        "log": [
            {
                "overall_admissibility_score": 0.5,
                "coord_snapshot": {
                    "t": [0.0, 1.0],
                    "x": list(np.linspace(0, 1, nx * ny * 2)),
                    "y": list(np.linspace(0, 1, nx * ny * 2)),
                },
            }
        ],
        "indices": [0],
        "plot_keys": ["constitutive/c/law_x/implied_delta"],
        "r_norm_mat": [[0.99]],
        "spatial": {
            "coords": {"x": np.linspace(0, 1, nx), "y": np.linspace(0, 1, ny)},
        },
        "closure_debug": {
            "c/law_x": {
                "pred": pred,
                "implied": implied,
                "raw": pred - implied,
                "scale_a": None,
                "scale_b": None,
                "ref": None,
                "mode": "subtract",
                "output_key": "x",
                "law_name": "x",
                "model_name": "c",
            }
        },
    }
    emb = prepare_constitutive_model_implied_vs_x_embed(bundle, prefer_last_t=True)
    assert emb is not None
    assert emb["title"] == "Constitutive Dissonance (max t, worst slice)"
    assert emb["subtitle"] == "max t, worst slice"
    # Last t slice has pred grid 999 → implied lines flat 1.4
    # traces[-1] is the implied line (band fills are prepended)
    y1 = np.asarray(emb["traces"][-1].y, dtype=float)
    assert np.allclose(y1, 1.4)

