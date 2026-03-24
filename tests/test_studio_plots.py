"""Tests for Moju Studio Plotly spatial helpers (including 3D)."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.parametrize("spatial_view", ["auto", "surface3d", "volume3d"])
def test_plotly_residual_or_state_smoke(spatial_view: str) -> None:
    pytest.importorskip("plotly")
    from apps.moju_studio.studio_plots import plotly_residual_or_state

    x = np.linspace(0, 1, 4)
    y = np.linspace(0, 1, 5)
    zc = np.linspace(0, 1, 6)
    if spatial_view == "auto":
        a = np.random.default_rng(0).standard_normal((5, 4))
        fig = plotly_residual_or_state(
            a,
            title="t",
            x=x,
            y=y,
            spatial_view="auto",  # type: ignore[arg-type]
        )
    elif spatial_view == "surface3d":
        a = np.random.default_rng(1).standard_normal((5, 4))
        fig = plotly_residual_or_state(
            a,
            title="surf",
            x=x,
            y=y,
            spatial_view="surface3d",  # type: ignore[arg-type]
        )
    else:
        a = np.random.default_rng(2).standard_normal((4, 5, 6))
        fig = plotly_residual_or_state(
            a,
            title="vol",
            x=x,
            y=y,
            z_coord=zc,
            spatial_view="volume3d",  # type: ignore[arg-type]
        )
    assert fig is not None
    assert hasattr(fig, "data")
    assert len(fig.data) >= 1


def test_plotly_pred_minus_ref_surface3d() -> None:
    pytest.importorskip("plotly")
    from apps.moju_studio.studio_plots import plotly_pred_minus_ref

    x = np.linspace(0, 1, 3)
    y = np.linspace(0, 1, 4)
    p = np.ones((4, 3))
    r = np.zeros((4, 3))
    fig = plotly_pred_minus_ref(
        p,
        r,
        title="d",
        spatial_view="surface3d",  # type: ignore[arg-type]
        x=x,
        y=y,
    )
    assert fig is not None
    assert fig.data[0].type == "surface"


def test_surface_misaligned_returns_message_figure() -> None:
    pytest.importorskip("plotly")
    from apps.moju_studio.studio_plots import plotly_surface_3d

    fig = plotly_surface_3d(
        np.ones((3, 3)),
        x=np.arange(2),
        y=np.arange(2),
        title="bad",
    )
    assert fig.layout.annotations
    assert fig.layout.annotations[0].text is not None
