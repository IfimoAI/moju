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
