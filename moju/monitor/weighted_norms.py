"""
Area/volume-weighted residual statistics (opt-in; diagnostics and optional scoring).

On structured 1D/2D/3D grids, trapezoid (composite) quadrature weights make RMS
independent of point clustering. Users may also pass an explicit weight array for
scattered collocation points.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import jax.numpy as jnp

from moju.monitor.path_b_derivatives import _uniform_1d_spacing


def trapezoid_weights_1d(coord: Any) -> jnp.ndarray:
    """Composite trapezoid weights for a (possibly non-uniform) 1D coordinate."""
    c = jnp.reshape(jnp.asarray(coord, dtype=float), (-1,))
    n = int(c.shape[0])
    if n < 2:
        return jnp.ones((n,), dtype=c.dtype)
    dc = jnp.diff(c)
    w = jnp.zeros((n,), dtype=c.dtype)
    w = w.at[0].set(0.5 * dc[0])
    w = w.at[-1].set(0.5 * dc[-1])
    if n > 2:
        interior = 0.5 * (dc[:-1] + dc[1:])
        w = w.at[1:-1].set(interior)
    # Absolute weights (orientation-independent)
    return jnp.abs(w)


def structured_point_weights(
    coords: Sequence[Any],
    *,
    lengths: Optional[Sequence[Optional[float]]] = None,
) -> jnp.ndarray:
    """
    Outer-product trapezoid weights on a separable or meshgrid-aligned structured grid.

    ``coords`` are 1D axis vectors (length n_i). Returns weights with shape
    ``(n0,)``, ``(n0, n1)``, or ``(n0, n1, n2)``.
    """
    axes = [trapezoid_weights_1d(c) for c in coords]
    if not axes:
        raise ValueError("coords must be non-empty")
    if len(axes) == 1:
        return axes[0]
    if len(axes) == 2:
        return axes[0][:, None] * axes[1][None, :]
    if len(axes) == 3:
        return axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    raise ValueError("structured_point_weights supports 1–3 axes")


def weights_from_state(
    state: Mapping[str, Any],
    *,
    key_x: str = "x",
    key_y: str = "y",
    key_z: str = "z",
    layout: str = "meshgrid",
) -> Optional[jnp.ndarray]:
    """
    Infer trapezoid weights from coordinate keys in ``state``.

    Returns ``None`` if coordinates are missing or incompatible.
    """
    x = state.get(key_x)
    if x is None:
        return None
    xa = jnp.asarray(x)
    y = state.get(key_y)
    z = state.get(key_z)
    try:
        if y is None:
            if xa.ndim == 1:
                return trapezoid_weights_1d(xa)
            # mesh (N,) flattened or (N,1)
            return trapezoid_weights_1d(jnp.reshape(xa, (-1,)))
        ya = jnp.asarray(y)
        if z is None:
            if xa.ndim == 1 and ya.ndim == 1:
                return structured_point_weights([xa, ya])
            # full mesh: recover 1D axes if rectilinear
            if xa.ndim == 2 and ya.ndim == 2:
                x1 = xa[:, 0] if bool(jnp.allclose(xa[:, 0:1], xa)) else None
                y1 = ya[0, :] if bool(jnp.allclose(ya[0:1, :], ya)) else None
                if x1 is not None and y1 is not None:
                    return structured_point_weights([x1, y1])
            return None
        za = jnp.asarray(z)
        if xa.ndim == 1 and ya.ndim == 1 and za.ndim == 1:
            return structured_point_weights([xa, ya, za])
    except Exception:  # noqa: BLE001
        return None
    return None


def _align_weights_to_residual(
    arr: Any, weights: Any
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Align quadrature weights to a residual array for weighted statistics.

    Handles common 2.0 default-path edge cases that must not raise:
    - scalar residuals (``()``): ignore spatial weights (use weight 1);
    - trailing singleton columns ``(n, 1)`` with 1D weights ``(n,)``;
    - overweight grids (e.g. inferred ``(n, n)`` weights vs 1D ``(n,)`` / ``(n, 1)``
      residual): collapse extra weight axes by summing.
    """
    a = jnp.asarray(arr)
    w_in = jnp.asarray(weights)
    w = jnp.asarray(w_in, dtype=a.dtype)

    if a.ndim == 0:
        return a, jnp.ones((), dtype=a.dtype)

    # Drop trailing singleton axes unless weights are also trailing-singleton with
    # the same leading shape (true 2D with one column). This turns 1D column
    # fields ``(n, 1)`` into ``(n,)`` even when overweight 2D weights ``(n, n)``
    # were inferred.
    while a.ndim >= 2 and int(a.shape[-1]) == 1:
        if (
            w.ndim == a.ndim
            and int(w.shape[-1]) == 1
            and tuple(a.shape[:-1]) == tuple(w.shape[:-1])
        ):
            break
        if tuple(w.shape) == tuple(a.shape):
            break
        a = a[..., 0]

    # Vector residual: Euclidean magnitude over the last axis when weights match
    # the leading spatial shape.
    if w.ndim == a.ndim - 1 and a.shape[: w.ndim] == w.shape:
        a = jnp.sqrt(jnp.sum(jnp.square(a), axis=-1) + 1e-30)

    # Weights have extra trailing spatial dims (mis-inferred mesh): marginalize.
    if w.ndim > a.ndim:
        if a.ndim >= 1 and int(w.shape[0]) == int(a.shape[0]):
            # If leading dims already match residual, sum the surplus axes.
            if tuple(a.shape) == tuple(w.shape[: a.ndim]):
                w = jnp.sum(w, axis=tuple(range(a.ndim, w.ndim)))
            else:
                w = jnp.sum(w, axis=tuple(range(1, w.ndim)))
        else:
            raise ValueError(
                f"weights shape {tuple(w_in.shape)} incompatible with residual {tuple(a.shape)}"
            )

    # Expand weights into trailing component dims when needed.
    while w.ndim < a.ndim:
        if a.shape[: w.ndim] == w.shape and a.ndim == w.ndim + 1:
            break
        w = w[..., None]

    try:
        w = jnp.broadcast_to(w, a.shape)
    except Exception as err:  # noqa: BLE001
        raise ValueError(
            f"weights shape {tuple(w_in.shape)} incompatible with residual {tuple(a.shape)}"
        ) from err
    return a, w


def weighted_rms(arr: Any, weights: Any) -> jnp.ndarray:
    """sqrt(sum(w r^2) / sum(w)) with broadcasting; NaN if total weight is 0."""
    a, w = _align_weights_to_residual(arr, weights)
    # Flatten
    af = a.ravel()
    wf = w.ravel()
    # Mask non-finite residual points
    finite = jnp.isfinite(af) & jnp.isfinite(wf) & (wf > 0)
    w_use = jnp.where(finite, wf, 0.0)
    a_use = jnp.where(finite, af, 0.0)
    denom = jnp.sum(w_use)
    num = jnp.sum(w_use * jnp.square(a_use))
    return jnp.where(denom > 0, jnp.sqrt(num / denom), jnp.asarray(float("nan")))


def weighted_max_abs(arr: Any, weights: Any) -> jnp.ndarray:
    """max |r| over points with positive weight (unweighted among kept points)."""
    try:
        a, w = _align_weights_to_residual(arr, weights)
    except ValueError:
        return jnp.max(jnp.abs(jnp.asarray(arr)))
    keep = w > 0
    return jnp.max(jnp.where(keep, jnp.abs(a), 0.0))


def normalize_weights(weights: Any) -> jnp.ndarray:
    w = jnp.asarray(weights, dtype=float)
    s = jnp.sum(w)
    return jnp.where(s > 0, w / s, w)


def weighted_r_eff_scalar(
    x: Any,
    weights: Any,
    *,
    jitter_sq: float = 0.0,
    q_power: float = 0.0,
    scale_eps: float = 1e-12,
) -> jnp.ndarray:
    """
    Weighted analogue of auditor ``_r_eff_scalar``: sqrt(sum(w r^2)/sum(w) + jitter).

    When ``q_power != 0``, applies the same Q^p factor using weighted means.
    """
    a, w = _align_weights_to_residual(x, weights)
    af = a.ravel()
    wf = w.ravel()
    finite = jnp.isfinite(af) & jnp.isfinite(wf) & (wf > 0)
    w_use = jnp.where(finite, wf, 0.0)
    a_use = jnp.where(finite, af, 0.0)
    denom = jnp.sum(w_use)
    mean_sq = jnp.where(denom > 0, jnp.sum(w_use * jnp.square(a_use)) / denom, jnp.nan)
    rms_r = jnp.sqrt(mean_sq + jnp.asarray(jitter_sq, dtype=a.dtype))
    p = float(q_power)
    if p == 0.0 or int(jnp.sum(finite)) <= 1:
        return rms_r
    eps = jnp.asarray(scale_eps, dtype=a.dtype)
    m = jnp.sqrt(jnp.square(a_use) + eps * eps)
    rms_m = jnp.sqrt(jnp.where(denom > 0, jnp.sum(w_use * jnp.square(m)) / denom, jnp.nan))
    mean_m = jnp.where(denom > 0, jnp.sum(w_use * m) / denom, jnp.nan)
    q = rms_m / mean_m
    return rms_r * jnp.power(q, p)
