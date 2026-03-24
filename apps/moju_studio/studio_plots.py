"""Plotly helpers for Moju Studio (spatial / time slices)."""

from __future__ import annotations

from typing import Any, Literal, Optional, Tuple

import jax.numpy as jnp
import numpy as np

SpatialViewMode = Literal["auto", "surface3d", "volume3d"]


def _squeeze_leading_ones(a: np.ndarray) -> np.ndarray:
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    return a


def _apply_time_slice_arr(
    a: np.ndarray,
    *,
    time_index: Optional[int],
    time_axis: int,
) -> np.ndarray:
    if time_index is not None and a.ndim >= 1 and a.shape[time_axis] > 0:
        idx = min(int(time_index), a.shape[time_axis] - 1)
        a = np.take(a, idx, axis=time_axis)
    return _squeeze_leading_ones(a)


def _align_surface_z_xy(
    a: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return (z_for_surface, x1d, y1d) with z shape (len(y), len(x)) for Plotly Surface."""
    x1 = np.asarray(x, dtype=float).ravel()
    y1 = np.asarray(y, dtype=float).ravel()
    if a.ndim != 2:
        return None
    # Plotly: z has rows = len(y), cols = len(x)
    if a.shape[1] == x1.shape[0] and a.shape[0] == y1.shape[0]:
        return a, x1, y1
    if a.shape[0] == x1.shape[0] and a.shape[1] == y1.shape[0]:
        return a.T, x1, y1
    return None


def _align_volume_xyz(
    a: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return (X, Y, Z, values_flat) for go.Volume with indexing='ij' meshgrid."""
    x1 = np.asarray(x, dtype=float).ravel()
    y1 = np.asarray(y, dtype=float).ravel()
    z1 = np.asarray(z, dtype=float).ravel()
    if a.ndim != 3:
        return None
    if a.shape != (x1.shape[0], y1.shape[0], z1.shape[0]):
        return None
    X, Y, Z = np.meshgrid(x1, y1, z1, indexing="ij")
    return X, Y, Z, a


def _empty_fig(title: str, message: str) -> Any:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.update_layout(title=title, annotations=[dict(text=message, showarrow=False)])
    return fig


def plotly_surface_3d(
    z_field: Any,
    *,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    heatmap_colorscale: str = "Jet",
    time_index: Optional[int] = None,
    time_axis: int = 0,
) -> Any:
    """3D surface z = f(x, y) after optional time slice (2D field + 1D coords)."""
    import plotly.graph_objects as go

    a = np.asarray(jnp.asarray(z_field), dtype=float)
    a = _apply_time_slice_arr(a, time_index=time_index, time_axis=time_axis)
    aligned = _align_surface_z_xy(a, x, y)
    if aligned is None:
        return _empty_fig(
            title,
            "Need a 2D array and 1D x,y matching the two spatial dimensions "
            "(e.g. T(nx,ny) with x(nx), y(ny), or transposed).",
        )
    z_surf, x1, y1 = aligned
    fig = go.Figure(
        data=[
            go.Surface(
                x=x1,
                y=y1,
                z=z_surf,
                colorscale=heatmap_colorscale,
                colorbar=dict(title="value"),
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="value", aspectmode="data"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def plotly_volume_3d(
    field: Any,
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str,
    heatmap_colorscale: str = "Jet",
    time_index: Optional[int] = None,
    time_axis: int = 0,
    max_voxels: int = 125_000,
) -> Any:
    """Volumetric scalar field on a regular grid (3D array + 1D x,y,z)."""
    import plotly.graph_objects as go

    a = np.asarray(jnp.asarray(field), dtype=float)
    a = _apply_time_slice_arr(a, time_index=time_index, time_axis=time_axis)
    if a.ndim != 3:
        return _empty_fig(title, "Need a 3D array after the time slice, with shapes matching x, y, z lengths.")
    aligned = _align_volume_xyz(a, x, y, z)
    if aligned is None:
        return _empty_fig(
            title,
            "Field shape must be (len(x), len(y), len(z)) with 1D mesh coordinates (ij ordering).",
        )
    X, Y, Z, vals3 = aligned
    if vals3.size > max_voxels:
        return _empty_fig(
            title,
            f"Field too large for volume plot ({vals3.size} voxels > {max_voxels}). "
            "Downsample the array or raise max_voxels in code.",
        )
    flat = vals3.ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return _empty_fig(title, "No finite values in field.")
    v0, v1 = float(np.nanpercentile(finite, 5)), float(np.nanpercentile(finite, 95))
    if not np.isfinite(v0) or not np.isfinite(v1) or v0 == v1:
        v0, v1 = float(np.nanmin(finite)), float(np.nanmax(finite))
    if v0 == v1:
        v1 = v0 + 1.0
    fig = go.Figure(
        data=[
            go.Volume(
                x=X.ravel(),
                y=Y.ravel(),
                z=Z.ravel(),
                value=flat,
                isomin=v0,
                isomax=v1,
                opacity=0.12,
                surface_count=18,
                colorscale=heatmap_colorscale,
                colorbar=dict(title="value"),
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def plotly_residual_or_state(
    z: Any,
    *,
    title: str,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    z_coord: Optional[np.ndarray] = None,
    time_index: Optional[int] = None,
    time_axis: int = 0,
    heatmap_colorscale: str = "Jet",
    spatial_view: SpatialViewMode = "auto",
) -> Any:
    """Build plotly figure: 1D line, 2D heatmap, 3D surface/volume, or histogram for high-D."""
    import plotly.graph_objects as go

    a = np.asarray(jnp.asarray(z), dtype=float)
    if a.size == 0:
        return _empty_fig(title, "Empty array")

    if spatial_view == "surface3d":
        if x is None or y is None:
            return _empty_fig(title, "3D surface needs `x` and `y` coordinates in state_pred.")
        return plotly_surface_3d(
            z,
            x=x,
            y=y,
            title=title,
            heatmap_colorscale=heatmap_colorscale,
            time_index=time_index,
            time_axis=time_axis,
        )

    if spatial_view == "volume3d":
        if x is None or y is None or z_coord is None:
            return _empty_fig(title, "3D volume needs `x`, `y`, and `z` coordinates in state_pred.")
        return plotly_volume_3d(
            z,
            x=x,
            y=y,
            z=z_coord,
            title=title,
            heatmap_colorscale=heatmap_colorscale,
            time_index=time_index,
            time_axis=time_axis,
        )

    a = _apply_time_slice_arr(a, time_index=time_index, time_axis=time_axis)

    if a.ndim == 1:
        xs = np.asarray(x) if x is not None else np.arange(a.shape[0])
        if xs.shape[0] != a.shape[0]:
            xs = np.arange(a.shape[0])
        fig = go.Figure(data=[go.Scatter(x=xs, y=a, mode="lines", name="value")])
        fig.update_layout(title=title, xaxis_title="x / index", yaxis_title="value")
        return fig

    if a.ndim == 2:
        x_axis = np.asarray(x) if x is not None else None
        y_axis = np.asarray(y) if y is not None else None
        fig = go.Figure(
            data=go.Heatmap(
                z=a,
                x=x_axis if x_axis is not None and x_axis.shape[0] == a.shape[1] else None,
                y=y_axis if y_axis is not None and y_axis.shape[0] == a.shape[0] else None,
                colorscale=heatmap_colorscale,
                colorbar=dict(title="value"),
            ),
        )
        fig.update_layout(
            title=title,
            xaxis_title="x" if x_axis is not None and x_axis.shape[0] == a.shape[1] else "index j",
            yaxis_title="y" if y_axis is not None and y_axis.shape[0] == a.shape[0] else "index i",
        )
        return fig

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
    heatmap_colorscale: str = "Jet",
    spatial_view: SpatialViewMode = "auto",
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    z_coord: Optional[np.ndarray] = None,
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
    return plotly_residual_or_state(
        d,
        title=f"{title} (pred − ref)",
        x=x,
        y=y,
        z_coord=z_coord,
        time_index=time_index,
        time_axis=time_axis,
        heatmap_colorscale=heatmap_colorscale,
        spatial_view=spatial_view,
    )
