"""Tests for monitor-only constitutive closure divergence embedding."""

from __future__ import annotations

import numpy as np
import pytest

from moju.monitor.visualize_constitutive import (
    prepare_monitor_closure_divergence_embed,
    worst_div_mean_abs_row_index,
)


def test_worst_div_mean_abs_row_index_picks_expected_row() -> None:
    div = np.array(
        [
            [0.0, 0.1],
            [5.0, 5.0],
            [1.0, 2.0],
        ],
        dtype=float,
    )
    assert worst_div_mean_abs_row_index(div) == 1


def test_prepare_embed_last_time_slice_1d() -> None:
    pytest.importorskip("plotly")
    nt, nx = 4, 3
    rng = np.random.default_rng(0)
    pred = rng.standard_normal((nt, nx))
    implied = pred + 0.1 * rng.standard_normal((nt, nx))
    bundle = {
        "closure_debug": {
            "test_model": {
                "mode": "subtract",
                "pred": pred,
                "implied": implied,
                "output_key": "alpha",
            }
        },
        "log": [
            {
                "coord_snapshot": {
                    "x": list(np.linspace(0.0, 1.0, nx)),
                    "t": list(np.linspace(0.0, 3.0, nt)),
                }
            }
        ],
        "plot_keys": ["constitutive/test_model/implied_delta"],
        "spatial_coord_hint": "x",
        "r_norm_mat": np.array([[1.0]]),
        "spatial_prefer_last_t": True,
    }
    emb = prepare_monitor_closure_divergence_embed(bundle, prefer_last_t=True)
    assert emb is not None
    assert emb["left_kind"] == "scatter"
    assert len(emb["left_traces"]) == 1
    assert len(emb["right_traces"]) == 2
    xv = emb["left_traces"][0].x
    assert len(xv) == nx


def test_prepare_embed_heatmap_worst_stripe() -> None:
    pytest.importorskip("plotly")
    ny, nx = 5, 4
    pred = np.zeros((ny, nx), dtype=float)
    pred[2, :] = 10.0
    implied = np.zeros_like(pred)
    bundle = {
        "closure_debug": {
            "tm": {"mode": "subtract", "pred": pred, "implied": implied, "output_key": "rho"},
        },
        "log": [],
        "plot_keys": ["constitutive/tm/implied_delta"],
        "spatial": {
            "kind": "2d",
            "coords": {"x": np.linspace(0, 2, nx), "y": np.linspace(0, 1, ny)},
        },
        "r_norm_mat": np.array([[0.5]]),
        "spatial_prefer_last_t": True,
        "spatial_coord_hint": "x",
    }
    emb = prepare_monitor_closure_divergence_embed(bundle, prefer_last_t=True)
    assert emb is not None
    assert emb["left_kind"] == "heatmap"
    assert int(emb["stripe_row_index"]) == 2
    xm = np.asarray(emb["right_traces"][0].y, dtype=float)
    np.testing.assert_array_almost_equal(xm, pred[2, :])


def test_prepare_embed_balance_1d() -> None:
    pytest.importorskip("plotly")
    n = 8
    a = np.linspace(1.0, 2.0, n)
    b = np.linspace(2.5, 3.5, n)
    bundle = {
        "closure_debug": {
            "bm": {"mode": "balance", "scale_a": a, "scale_b": b, "output_key": "T"},
        },
        "log": [
            {"coord_snapshot": {"x": list(np.linspace(0.0, 0.05, n))}},
        ],
        "plot_keys": ["constitutive/bm/implied_delta"],
        "spatial_coord_hint": "x",
        "r_norm_mat": np.array([[0.42]]),
    }
    emb = prepare_monitor_closure_divergence_embed(bundle)
    assert emb is not None
    assert emb["left_kind"] == "scatter"

