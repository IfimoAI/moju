"""
Central finite-difference stencils of order 2, 4, and 6 with matching one-sided boundaries.

Uniform spacing only. Non-uniform grids should use :func:`jnp.gradient` with coordinate
arrays (order 2) or supply derivatives from the solver; see ``docs/getting_derivatives_right.md``.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Tuple, Union

import jax.numpy as jnp

FdOrder = Literal[2, 4, 6]
VALID_FD_ORDERS: Tuple[int, ...] = (2, 4, 6)


def validate_fd_order(order: int) -> int:
    o = int(order)
    if o not in VALID_FD_ORDERS:
        raise ValueError(f"fd_order must be one of {VALID_FD_ORDERS}, got {order}")
    return o


def _require_len(n: int, order: int, *, kind: str) -> None:
    need = {2: 3, 4: 5, 6: 7}[order]
    if n < need:
        raise ValueError(
            f"{kind}: need at least {need} points for fd_order={order}, got {n}"
        )


def diff1_uniform(values: jnp.ndarray, h: float, *, order: int = 2, axis: int = -1) -> jnp.ndarray:
    """
    First derivative along ``axis`` on a uniform grid with spacing ``h``.

    Interior: central stencil of order 2/4/6. Boundaries: consistent one-sided stencils
    of the same formal order.
    """
    order = validate_fd_order(order)
    a = jnp.moveaxis(jnp.asarray(values), axis, -1)
    n = int(a.shape[-1])
    _require_len(n, order, kind="diff1_uniform")
    h = float(h)
    if abs(h) < 1e-30:
        raise ValueError("diff1_uniform: spacing h must be non-zero")

    if order == 2:
        # Interior: (-f[-1] + f[+1]) / (2h)
        # Left: (-3f0 + 4f1 - f2) / (2h)
        # Right: (f[-3] - 4f[-2] + 3f[-1]) / (2h)
        left = (-3.0 * a[..., 0] + 4.0 * a[..., 1] - a[..., 2]) / (2.0 * h)
        right = (a[..., -3] - 4.0 * a[..., -2] + 3.0 * a[..., -1]) / (2.0 * h)
        if n == 3:
            mid = (a[..., 2] - a[..., 0]) / (2.0 * h)
            out = jnp.stack([left, mid, right], axis=-1)
        else:
            mid = (a[..., 2:] - a[..., :-2]) / (2.0 * h)
            out = jnp.concatenate(
                [left[..., None], mid, right[..., None]], axis=-1
            )
    elif order == 4:
        # Interior: (f[-2] - 8f[-1] + 8f[+1] - f[+2]) / (12h)
        # Left two points: forward O(h^4); right two: backward O(h^4)
        c0 = (
            -25.0 * a[..., 0]
            + 48.0 * a[..., 1]
            - 36.0 * a[..., 2]
            + 16.0 * a[..., 3]
            - 3.0 * a[..., 4]
        ) / (12.0 * h)
        c1 = (
            -3.0 * a[..., 0]
            - 10.0 * a[..., 1]
            + 18.0 * a[..., 2]
            - 6.0 * a[..., 3]
            + a[..., 4]
        ) / (12.0 * h)
        mid = (
            a[..., :-4]
            - 8.0 * a[..., 1:-3]
            + 8.0 * a[..., 3:-1]
            - a[..., 4:]
        ) / (12.0 * h)
        r1 = (
            -a[..., -5]
            + 6.0 * a[..., -4]
            - 18.0 * a[..., -3]
            + 10.0 * a[..., -2]
            + 3.0 * a[..., -1]
        ) / (12.0 * h)
        r0 = (
            3.0 * a[..., -5]
            - 16.0 * a[..., -4]
            + 36.0 * a[..., -3]
            - 48.0 * a[..., -2]
            + 25.0 * a[..., -1]
        ) / (12.0 * h)
        out = jnp.concatenate(
            [c0[..., None], c1[..., None], mid, r1[..., None], r0[..., None]], axis=-1
        )
    else:  # order == 6
        # Interior 6th-order central; 3-point one-sided of matching order at each end.
        c0 = (
            -147.0 * a[..., 0]
            + 360.0 * a[..., 1]
            - 450.0 * a[..., 2]
            + 400.0 * a[..., 3]
            - 225.0 * a[..., 4]
            + 72.0 * a[..., 5]
            - 10.0 * a[..., 6]
        ) / (60.0 * h)
        c1 = (
            -10.0 * a[..., 0]
            - 77.0 * a[..., 1]
            + 150.0 * a[..., 2]
            - 100.0 * a[..., 3]
            + 50.0 * a[..., 4]
            - 15.0 * a[..., 5]
            + 2.0 * a[..., 6]
        ) / (60.0 * h)
        c2 = (
            2.0 * a[..., 0]
            - 24.0 * a[..., 1]
            - 35.0 * a[..., 2]
            + 80.0 * a[..., 3]
            - 30.0 * a[..., 4]
            + 8.0 * a[..., 5]
            - a[..., 6]
        ) / (60.0 * h)
        mid = (
            -a[..., :-6]
            + 9.0 * a[..., 1:-5]
            - 45.0 * a[..., 2:-4]
            + 45.0 * a[..., 4:-2]
            - 9.0 * a[..., 5:-1]
            + a[..., 6:]
        ) / (60.0 * h)
        r2 = (
            a[..., -7]
            - 8.0 * a[..., -6]
            + 30.0 * a[..., -5]
            - 80.0 * a[..., -4]
            + 35.0 * a[..., -3]
            + 24.0 * a[..., -2]
            - 2.0 * a[..., -1]
        ) / (60.0 * h)
        r1 = (
            -2.0 * a[..., -7]
            + 15.0 * a[..., -6]
            - 50.0 * a[..., -5]
            + 100.0 * a[..., -4]
            - 150.0 * a[..., -3]
            + 77.0 * a[..., -2]
            + 10.0 * a[..., -1]
        ) / (60.0 * h)
        r0 = (
            10.0 * a[..., -7]
            - 72.0 * a[..., -6]
            + 225.0 * a[..., -5]
            - 400.0 * a[..., -4]
            + 450.0 * a[..., -3]
            - 360.0 * a[..., -2]
            + 147.0 * a[..., -1]
        ) / (60.0 * h)
        out = jnp.concatenate(
            [
                c0[..., None],
                c1[..., None],
                c2[..., None],
                mid,
                r2[..., None],
                r1[..., None],
                r0[..., None],
            ],
            axis=-1,
        )
    return jnp.moveaxis(out, -1, axis)


def diff2_uniform(values: jnp.ndarray, h: float, *, order: int = 2, axis: int = -1) -> jnp.ndarray:
    """
    Second derivative along ``axis`` on a uniform grid with spacing ``h``.

    For ``order > 2``, composed as ``diff1(diff1(...))`` with the same order (keeps
    boundary treatment consistent). For ``order == 2``, uses the classic 3-point
    stencil on the interior for better constant-factor accuracy.
    """
    order = validate_fd_order(order)
    a = jnp.moveaxis(jnp.asarray(values), axis, -1)
    n = int(a.shape[-1])
    _require_len(n, order, kind="diff2_uniform")
    h = float(h)
    if abs(h) < 1e-30:
        raise ValueError("diff2_uniform: spacing h must be non-zero")

    if order == 2:
        mid = (a[..., :-2] - 2.0 * a[..., 1:-1] + a[..., 2:]) / (h * h)
        if n >= 4:
            left = (2.0 * a[..., 0] - 5.0 * a[..., 1] + 4.0 * a[..., 2] - a[..., 3]) / (h * h)
            right = (2.0 * a[..., -1] - 5.0 * a[..., -2] + 4.0 * a[..., -3] - a[..., -4]) / (h * h)
        else:
            left = mid[..., 0]
            right = mid[..., -1]
        out = jnp.concatenate([left[..., None], mid, right[..., None]], axis=-1)
        return jnp.moveaxis(out, -1, axis)

    # Higher order: apply first-derivative stencil twice.
    first = diff1_uniform(jnp.moveaxis(a, -1, axis), h, order=order, axis=axis)
    return diff1_uniform(first, h, order=order, axis=axis)


def gradient_uniform_nd(
    field: jnp.ndarray,
    spacings: Sequence[float],
    *,
    order: int = 2,
    axes: Optional[Sequence[int]] = None,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
    """
    Gradient of a scalar field on a uniform rectilinear grid.

    Returns a single array when ``len(spacings)==1``, else a tuple of partials
    matching ``jnp.gradient`` convention.
    """
    order = validate_fd_order(order)
    f = jnp.asarray(field)
    if axes is None:
        axes = tuple(range(f.ndim))
    if len(spacings) != len(axes):
        raise ValueError("spacings and axes must have the same length")
    parts = [
        diff1_uniform(f, float(h), order=order, axis=int(ax))
        for h, ax in zip(spacings, axes)
    ]
    if len(parts) == 1:
        return parts[0]
    return tuple(parts)


def laplacian_uniform_nd(
    field: jnp.ndarray,
    spacings: Sequence[float],
    *,
    order: int = 2,
    axes: Optional[Sequence[int]] = None,
) -> jnp.ndarray:
    """Scalar Laplacian on a uniform rectilinear grid (sum of second derivatives)."""
    order = validate_fd_order(order)
    f = jnp.asarray(field)
    if axes is None:
        axes = tuple(range(f.ndim))
    if len(spacings) != len(axes):
        raise ValueError("spacings and axes must have the same length")
    acc = jnp.zeros_like(f)
    for h, ax in zip(spacings, axes):
        acc = acc + diff2_uniform(f, float(h), order=order, axis=int(ax))
    return acc
