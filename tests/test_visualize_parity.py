"""
Parity tests asserting the public :func:`moju.monitor.auditor.visualize` API
shape is preserved through the visualisation overhaul.

These guard against accidental signature drift while we refactor internals
on top of the new component library.
"""

from __future__ import annotations

from typing import Any, List

import jax.numpy as jnp
import pytest


def _build_minimal_engine() -> Any:
    from moju.piratio.models import Models
    from moju.monitor.auditor import ResidualEngine

    engine = ResidualEngine(
        laws=[],
        constitutive_audit=[
            {
                "name": "ideal_gas_rho",
                "output_key": "rho",
                "state_map": {"P": "P", "R": "R", "T": "T"},
                "implied_value_key": "rho_implied",
            }
        ],
    )
    P, R, T = jnp.array(1.0e5), jnp.array(287.0), jnp.array(300.0)
    rho = Models.ideal_gas_rho(P, R, T)
    engine.compute_residuals({"P": P, "R": R, "T": T, "rho": rho, "rho_implied": rho})
    # Run a second step to make timelines plottable
    engine.compute_residuals({"P": P * 1.01, "R": R, "T": T, "rho": rho * 1.01, "rho_implied": rho})
    return engine


def test_visualize_single_figure_returns_figure() -> None:
    pytest.importorskip("plotly")
    from moju.monitor.auditor import visualize

    engine = _build_minimal_engine()
    fig = visualize(engine.log, engine=engine, dashboard_mode="single-figure")
    assert fig is not None
    # Returns a Plotly Figure
    assert hasattr(fig, "data")
    assert hasattr(fig, "layout")


def test_visualize_dash_tabs_returns_payload() -> None:
    pytest.importorskip("plotly")
    from moju.monitor.auditor import visualize

    engine = _build_minimal_engine()
    payload = visualize(engine.log, engine=engine, dashboard_mode="dash-tabs")
    assert isinstance(payload, dict)
    assert payload.get("mode") == "dash-tabs"
    assert "tabs" in payload
    assert isinstance(payload["tabs"], dict)
    # Required tabs always present
    for key in ("kpi", "admissibility", "forensic_heatmaps", "convergence"):
        assert key in payload["tabs"]
    # New tab appears whenever closure_debug is populated
    assert "constitutive_divergence" in payload["tabs"]


def test_visualize_single_figure_raises_height_when_closure_debug() -> None:
    """Single-figure monitor adds a constitutive-divergence row when closure_debug is present."""
    pytest.importorskip("plotly")
    from moju.monitor.auditor import visualize
    from moju.monitor.visualize_plotly import MONITOR_SINGLE_FIGURE_HEIGHT

    engine = _build_minimal_engine()
    fig = visualize(engine.log, engine=engine, dashboard_mode="single-figure", mode="eval")
    assert fig.layout.height is not None
    assert int(fig.layout.height) >= MONITOR_SINGLE_FIGURE_HEIGHT + 199


def test_visualize_split_returns_dict_with_worst_keys() -> None:
    pytest.importorskip("plotly")
    from moju.monitor.auditor import visualize

    engine = _build_minimal_engine()
    out = visualize(engine.log, engine=engine, visualize_layout="split")
    # Split layout returns a dict bundle with separate worst-keys table figure
    assert isinstance(out, dict)
    assert "worst_keys" in out
    # At least one Plotly Figure value must be present in the bundle
    assert any(hasattr(v, "data") for v in out.values())
