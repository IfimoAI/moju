"""
Optional evaluation-point masking for residual statistics.

When a mask is supplied to :meth:`ResidualEngine.compute_residuals`, points with
``False`` are excluded from RMS / max used for logging and scoring for that call.
Without a mask, behaviour matches prior releases. Mask metadata is recorded in
provenance / the report and never changes tier *thresholds*.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import jax.numpy as jnp


def boundary_distance_mask(
    coords: Sequence[Any],
    *,
    distance: float,
    lengths: Optional[Sequence[Optional[float]]] = None,
) -> jnp.ndarray:
    """
    Boolean mask (True = keep) excluding points within ``distance`` of each axis boundary.

    Each entry of ``coords`` is a 1D coordinate vector (separable) or a full mesh array
    with the same shape for every axis. ``lengths`` optional period/domain lengths; when
    omitted, uses ``min``/``max`` of each coordinate.
    """
    if distance < 0:
        raise ValueError("distance must be non-negative")
    arrs = [jnp.asarray(c) for c in coords]
    if not arrs:
        raise ValueError("coords must be non-empty")
    if all(a.ndim == 1 for a in arrs):
        meshes = jnp.meshgrid(*arrs, indexing="ij")
    else:
        meshes = tuple(arrs)
        shapes = {m.shape for m in meshes}
        if len(shapes) != 1:
            raise ValueError("all coordinate arrays must share the same shape")
    keep = jnp.ones(meshes[0].shape, dtype=bool)
    for i, m in enumerate(meshes):
        if lengths is not None and i < len(lengths) and lengths[i] is not None:
            lo = 0.0
            hi = float(lengths[i])
        else:
            lo = float(jnp.min(m))
            hi = float(jnp.max(m))
        keep = keep & (m >= lo + distance) & (m <= hi - distance)
    return keep


def math_prod(shape: Tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def apply_mask_to_residual(arr: Any, mask: Any) -> Any:
    """
    Restrict a residual array to masked-in points for RMS/max.

    ``mask`` is True = keep. For arrays with a trailing component axis, ``mask`` applies
    to the leading spatial axes.
    """
    if arr is None or mask is None:
        return arr
    a = jnp.asarray(arr)
    m = jnp.asarray(mask).astype(bool)
    if a.shape == m.shape:
        return a[m]
    if a.ndim == m.ndim + 1 and a.shape[:-1] == m.shape:
        return a.reshape(math_prod(m.shape), a.shape[-1])[m.reshape(-1)]
    if a.ndim > m.ndim and a.shape[: m.ndim] == m.shape:
        trail = a.shape[m.ndim :]
        return a.reshape(math_prod(m.shape), *trail)[m.reshape(-1)]
    raise ValueError(f"cannot apply mask shape {m.shape} to residual shape {a.shape}")


def mask_flat_residuals(flat: Mapping[str, Any], mask: Any) -> Dict[str, Any]:
    """Apply the same mask to every residual array in a flat key→array map."""
    if mask is None:
        return dict(flat)
    out: Dict[str, Any] = {}
    for k, v in flat.items():
        try:
            out[k] = apply_mask_to_residual(v, mask)
        except ValueError:
            out[k] = v
    return out


def mask_provenance(mask: Any, *, kind: str = "boolean") -> Dict[str, Any]:
    """Compact provenance record for the report (not the full mask array)."""
    if mask is None:
        return {}
    m = jnp.asarray(mask).astype(bool)
    n = int(m.size)
    n_keep = int(jnp.sum(m))
    return {
        "kind": kind,
        "shape": list(m.shape),
        "n_points": n,
        "n_kept": n_keep,
        "n_excluded": n - n_keep,
        "fraction_kept": float(n_keep / n) if n else float("nan"),
    }
