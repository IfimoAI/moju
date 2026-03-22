"""Plotly helpers for Moju Studio (spatial / time slices)."""

from __future__ import annotations

from typing import Any, Optional

import jax.numpy as jnp
import numpy as np


def plotly_residual_or_state(
    z: Any,
    *,
    title: str,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    time_index: Optional[int] = None,
    time_axis: int = 0,
) -> Any:
    """Build plotly figure: 1D line, 2D heatmap, or histogram for high-D slice."""
    import plotly.graph_objects as go

    a = np.asarray(jnp.asarray(z), dtype=float)
    if a.size == 0:
        fig = go.Figure()
        fig.update_layout(title=title, annotations=[dict(text="Empty array", showarrow=False)])
        return fig

    if time_index is not None and a.ndim >= 1 and a.shape[time_axis] > 0:
        idx = min(int(time_index), a.shape[time_axis] - 1)
        a = np.take(a, idx, axis=time_axis)

    # Squeeze leading singletons
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]

    if a.ndim == 1:
        xs = np.asarray(x) if x is not None else np.arange(a.shape[0])
        if xs.shape[0] != a.shape[0]:
            xs = np.arange(a.shape[0])
        fig = go.Figure(data=[go.Scatter(x=xs, y=a, mode="lines", name="value")])
        fig.update_layout(title=title, xaxis_title="x / index", yaxis_title="value")
        return fig

    if a.ndim == 2:
        fig = go.Figure(
            data=go.Heatmap(z=a, colorscale="RdBu_r", colorbar=dict(title="value")),
        )
        fig.update_layout(title=title, xaxis_title="index j", yaxis_title="index i")
        return fig

    # ndim >= 3: show histogram of values as fallback
    fig = go.Figure(data=[go.Histogram(x=a.ravel(), nbinsx=80)])
    fig.update_layout(title=f"{title} (histogram, ndim={a.ndim})", xaxis_title="value", yaxis_title="count")
    return fig


def plotly_pred_minus_ref(
    pred: Any,
    ref: Any,
    *,
    title: str,
    time_index: Optional[int] = None,
    time_axis: int = 0,
) -> Any:
    """Difference pred - ref with broadcasting where shapes match."""
    import plotly.graph_objects as go

    p = jnp.asarray(pred)
    r = jnp.asarray(ref)
    try:
        d = p - r
    except Exception:  # noqa: BLE001
        fig = go.Figure()
        fig.update_layout(
            title=title,
            annotations=[dict(text="Shape mismatch for pred - ref", showarrow=False)],
        )
        return fig
    return plotly_residual_or_state(d, title=f"{title} (pred − ref)", time_index=time_index, time_axis=time_axis)
