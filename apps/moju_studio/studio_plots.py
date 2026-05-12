"""
Plotly helpers for Moju Studio.

These functions are thin backward-compatible shims over the unified component
library (:mod:`moju.monitor.visualize_components`).  They preserve the legacy
public signature used by Studio pages and the test suite but use the
enterprise theme (Inter + Viridis / RdBu_r) under the hood — no more ``Jet``
defaults, no Arial.  Any new code should call the component library
directly.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import jax.numpy as jnp
import numpy as np

from moju.monitor.visualize_components import (
    build_field_explorer_card,
    _empty_card,  # type: ignore[attr-defined]
)
from moju.monitor.visualize_theme import MOJU_LIGHT, get_theme

SpatialViewMode = Literal["auto", "surface3d", "volume3d"]
ArrayPlotMode = Literal["auto", "line", "heatmap2d"]


def _apply_time_slice_arr(
    a: np.ndarray,
    *,
    time_index: Optional[int],
    time_axis: int,
) -> np.ndarray:
    if time_index is not None and a.ndim >= 1 and a.shape[time_axis] > 0:
        idx = min(int(time_index), a.shape[time_axis] - 1)
        a = np.take(a, idx, axis=time_axis)
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    return a


def _empty_fig(title: str, message: str) -> Any:
    """Themed empty figure (kept for backward compatibility)."""
    return _empty_card(title, message, MOJU_LIGHT)


def plotly_surface_3d(
    z_field: Any,
    *,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    heatmap_colorscale: Optional[str] = None,
    time_index: Optional[int] = None,
    time_axis: int = 0,
) -> Any:
    """3D surface z = f(x, y) after optional time slice (2D field + 1D coords)."""
    t = get_theme(MOJU_LIGHT)
    cs = heatmap_colorscale or t.colorscales.sequential
    a = np.asarray(jnp.asarray(z_field), dtype=float)
    a = _apply_time_slice_arr(a, time_index=time_index, time_axis=time_axis)
    return build_field_explorer_card(
        a,
        title=title,
        x=x,
        y=y,
        spatial_view="surface3d",
        colorscale=cs,
        theme=MOJU_LIGHT,
    )


def plotly_volume_3d(
    field: Any,
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str,
    heatmap_colorscale: Optional[str] = None,
    time_index: Optional[int] = None,
    time_axis: int = 0,
    max_voxels: int = 125_000,
) -> Any:
    """Volumetric scalar field on a regular grid (3D array + 1D x,y,z)."""
    t = get_theme(MOJU_LIGHT)
    cs = heatmap_colorscale or t.colorscales.sequential
    a = np.asarray(jnp.asarray(field), dtype=float)
    a = _apply_time_slice_arr(a, time_index=time_index, time_axis=time_axis)
    return build_field_explorer_card(
        a,
        title=title,
        x=x,
        y=y,
        z=z,
        spatial_view="volume3d",
        colorscale=cs,
        theme=MOJU_LIGHT,
    )


def plotly_residual_or_state(
    z: Any,
    *,
    title: str,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    z_coord: Optional[np.ndarray] = None,
    time_index: Optional[int] = None,
    time_axis: int = 0,
    t_coord: Optional[np.ndarray] = None,
    heatmap_colorscale: Optional[str] = None,
    spatial_view: SpatialViewMode = "auto",
    array_plot: ArrayPlotMode = "auto",
) -> Any:
    """
    Build a themed Plotly figure for a residual / state field.

    Auto-routes between 1-D line, 2-D heatmap, surface3d, volume3d, and the
    high-dimensional fallback (histogram).  Backward-compatible signature.
    """
    import plotly.graph_objects as go

    t = get_theme(MOJU_LIGHT)
    cs = heatmap_colorscale or t.colorscales.sequential

    a0 = np.asarray(jnp.asarray(z), dtype=float)
    if a0.size == 0:
        return _empty_fig(title, "Empty array")

    if spatial_view == "surface3d":
        if x is None or y is None:
            return _empty_fig(title, "3D surface needs `x` and `y` coordinates in state_pred.")
        return plotly_surface_3d(
            z,
            x=x,
            y=y,
            title=title,
            heatmap_colorscale=cs,
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
            heatmap_colorscale=cs,
            time_index=time_index,
            time_axis=time_axis,
        )

    # Apply time slice for line/heatmap2d branches
    slice_idx = time_index if array_plot in ("auto", "line") else None
    a = _apply_time_slice_arr(a0, time_index=slice_idx, time_axis=time_axis)

    if array_plot == "heatmap2d":
        # Preserve legacy "force heatmap2d" semantics: prefer (t, x) layout when t_coord supplied
        if a.ndim < 2:
            return _empty_fig(title, "Heatmap (2D) needs a 2D array after squeezing singleton dimensions.")
        if a.ndim > 2:
            # Squeeze trailing/leading singletons to land on 2D
            b = a
            while b.ndim > 2 and b.shape[0] == 1:
                b = b[0]
            while b.ndim > 2 and b.shape[-1] == 1:
                b = b[..., 0]
            if b.ndim != 2:
                return _empty_fig(title, "Heatmap (2D) needs a 2D array after squeezing singleton dimensions.")
            a = b
        # Use t_coord on y when shape matches and y not supplied
        y_axis = y if y is not None else (t_coord if t_coord is not None and len(np.asarray(t_coord).ravel()) == a.shape[0] else None)
        return build_field_explorer_card(
            a,
            title=title,
            x=x,
            y=y_axis,
            spatial_view="auto",
            colorscale=cs,
            theme=MOJU_LIGHT,
        )

    return build_field_explorer_card(
        a,
        title=title,
        x=x,
        y=y,
        z=z_coord,
        time_index=None,  # already sliced above
        time_axis=time_axis,
        spatial_view=spatial_view,
        colorscale=cs,
        theme=MOJU_LIGHT,
    )


def plotly_pred_minus_ref(
    pred: Any,
    ref: Any,
    *,
    title: str,
    time_index: Optional[int] = None,
    time_axis: int = 0,
    heatmap_colorscale: Optional[str] = None,
    spatial_view: SpatialViewMode = "auto",
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    z_coord: Optional[np.ndarray] = None,
    t_coord: Optional[np.ndarray] = None,
    array_plot: ArrayPlotMode = "auto",
) -> Any:
    """Difference pred − ref with broadcasting where shapes match."""
    p = jnp.asarray(pred)
    r = jnp.asarray(ref)
    try:
        d = p - r
    except Exception:  # noqa: BLE001
        return _empty_fig(title, "Shape mismatch for pred − ref")
    return plotly_residual_or_state(
        d,
        title=f"{title} (pred − ref)",
        x=x,
        y=y,
        z_coord=z_coord,
        time_index=time_index,
        time_axis=time_axis,
        t_coord=t_coord,
        heatmap_colorscale=heatmap_colorscale,
        spatial_view=spatial_view,
        array_plot=array_plot,
    )


__all__ = [
    "SpatialViewMode",
    "ArrayPlotMode",
    "plotly_surface_3d",
    "plotly_volume_3d",
    "plotly_residual_or_state",
    "plotly_pred_minus_ref",
]
