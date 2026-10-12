"""
Central finite-difference stencils of order 2, 4, and 6 with matching one-sided boundaries.

Uniform spacing only. Non-uniform grids should use :func:`jnp.gradient` with coordinate
arrays (order 2) or supply derivatives from the solver; see ``docs/getting_derivatives_right.md``.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Sequence, Tuple, Union

import jax.numpy as jnp

FdOrder = Literal[2, 4, 6]
VALID_FD_ORDERS: Tuple[int, ...] = (2, 4, 6)
DEFAULT_FD_ORDER: int = 4


def validate_fd_order(order: int) -> int:
    o = int(order)
    if o not in VALID_FD_ORDERS:
        raise ValueError(f"fd_order must be one of {VALID_FD_ORDERS}, got {order}")
    return o


def effective_fd_order(order: Any) -> Optional[int]:
    """
    Normalize ``fd_order`` / ``fd_order_time`` to ``None`` (legacy ``jnp.gradient``) or 2/4/6.

    Accepts ``None``, ``\"legacy\"``, or 2/4/6. Moju 2.0 defaults use 4 via
    :class:`~moju.monitor.path_b_derivatives.PathBGridConfig`.
    """
    if order is None:
        return None
    if isinstance(order, str):
        if order.strip().lower() == "legacy":
            return None
        raise ValueError(
            f"fd_order string must be 'legacy', got {order!r}; "
            f"numeric orders: {VALID_FD_ORDERS}"
        )
    return validate_fd_order(order)


def _require_len(n: int, order: int, *, kind: str) -> None:
    need = {2: 3, 4: 5, 6: 7}[order]
    if n < need:
        raise ValueError(
            f"{kind}: need at least {need} points for fd_order={order}, got {n}"
        )


def diff1_uniform(
    values: jnp.ndarray,
    h: float,
    *,
    order: int = 2,
    axis: int = -1,
    boundary_closure: Any = "matching",
    boundary_values: Optional[Tuple[Optional[float], Optional[float]]] = None,
    use_field_boundary_values: bool = False,
    periodic: bool = False,
) -> jnp.ndarray:
    """
    First derivative along ``axis`` on a uniform grid with spacing ``h``.

    Interior: central stencil of order 2/4/6. Boundaries default to Fornberg
    one-sided of matching width (``boundary_closure=\"matching\"``). Opt-in
    closures: ``order_reduced``, ``sbp``, ``bc_aware`` — see
    :mod:`moju.monitor.boundary_closures`.

    ``periodic=True`` uses wrap-around central stencils on every row (overrides
    boundary closures on this axis). Never inferred — declare explicitly.
    """
    order = validate_fd_order(order)
    a = jnp.asarray(values)
    n = int(jnp.moveaxis(a, axis, -1).shape[-1])
    _require_len(n, order, kind="diff1_uniform")
    from moju.monitor.boundary_closures import apply_diff_uniform_1d

    return apply_diff_uniform_1d(
        a,
        float(h),
        deriv_order=1,
        order=order,
        axis=axis,
        boundary_closure=boundary_closure,
        boundary_values=boundary_values,
        use_field_boundary_values=use_field_boundary_values,
        periodic=periodic,
    )


def _edge_weights_uniform(n_stencil: int, deriv_order: int, h: float, *, left: bool) -> Any:
    """Fornberg one-sided weights on a uniform unit-index stencil, scaled by h."""
    import numpy as np

    idx = np.arange(n_stencil, dtype=float)
    z = 0.0 if left else float(n_stencil - 1)
    c = fornberg_weights(z, idx, deriv_order)
    # d^k/dx^k = h^{-k} * d^k/d(index)^k
    return c[deriv_order, :] / (h ** deriv_order)


def diff2_uniform(
    values: jnp.ndarray,
    h: float,
    *,
    order: int = 2,
    axis: int = -1,
    boundary_closure: Any = "matching",
    boundary_values: Optional[Tuple[Optional[float], Optional[float]]] = None,
    use_field_boundary_values: bool = False,
    periodic: bool = False,
) -> jnp.ndarray:
    """
    Second derivative along ``axis`` on a uniform grid with spacing ``h``.

    Uses **dedicated** second-derivative stencils on the interior for
    ``boundary_closure=\"matching\"`` / ``order_reduced``. ``sbp`` uses reduced-order
    Fornberg ends. ``bc_aware`` uses high-order ghosts when Dirichlet values are supplied.
    ``periodic=True`` uses wrap-around central stencils (overrides boundary closures).
    """
    order = validate_fd_order(order)
    a = jnp.asarray(values)
    n = int(jnp.moveaxis(a, axis, -1).shape[-1])
    _require_len(n, order, kind="diff2_uniform")
    from moju.monitor.boundary_closures import apply_diff_uniform_1d

    return apply_diff_uniform_1d(
        a,
        float(h),
        deriv_order=2,
        order=order,
        axis=axis,
        boundary_closure=boundary_closure,
        boundary_values=boundary_values,
        use_field_boundary_values=use_field_boundary_values,
        periodic=periodic,
    )


def mixed_diff_uniform(
    values: jnp.ndarray,
    hx: float,
    hy: float,
    *,
    order: int = 2,
    axis_x: int = 0,
    axis_y: int = 1,
    boundary_closure: Any = "matching",
    periodic_x: bool = False,
    periodic_y: bool = False,
) -> jnp.ndarray:
    """Mixed second derivative ∂²/∂x∂y via successive first-derivative stencils."""
    order = validate_fd_order(order)
    ddx = diff1_uniform(
        values,
        float(hx),
        order=order,
        axis=axis_x,
        boundary_closure=boundary_closure,
        periodic=periodic_x,
    )
    return diff1_uniform(
        ddx,
        float(hy),
        order=order,
        axis=axis_y,
        boundary_closure=boundary_closure,
        periodic=periodic_y,
    )


def _normalize_periodic_flags(
    periodic: Any, n_axes: int
) -> Tuple[bool, ...]:
    """Normalize ``bool | Sequence[bool]`` to a length-``n_axes`` tuple."""
    if isinstance(periodic, bool) or periodic is None:
        return tuple(bool(periodic) for _ in range(n_axes))
    flags = tuple(bool(p) for p in periodic)
    if len(flags) != n_axes:
        raise ValueError(
            f"periodic sequence length {len(flags)} must match n_axes={n_axes}"
        )
    return flags


def gradient_uniform_nd(
    field: jnp.ndarray,
    spacings: Sequence[float],
    *,
    order: int = 2,
    axes: Optional[Sequence[int]] = None,
    boundary_closure: Any = "matching",
    use_field_boundary_values: bool = False,
    periodic: Any = False,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
    """
    Gradient of a scalar field on a uniform rectilinear grid.

    Returns a single array when ``len(spacings)==1``, else a tuple of partials
    matching ``jnp.gradient`` convention. ``periodic`` may be a bool or a
    per-axis sequence.
    """
    order = validate_fd_order(order)
    f = jnp.asarray(field)
    if axes is None:
        axes = tuple(range(f.ndim))
    if len(spacings) != len(axes):
        raise ValueError("spacings and axes must have the same length")
    flags = _normalize_periodic_flags(periodic, len(axes))
    parts = [
        diff1_uniform(
            f,
            float(h),
            order=order,
            axis=int(ax),
            boundary_closure=boundary_closure,
            use_field_boundary_values=use_field_boundary_values,
            periodic=flags[i],
        )
        for i, (h, ax) in enumerate(zip(spacings, axes))
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
    boundary_closure: Any = "matching",
    use_field_boundary_values: bool = False,
    periodic: Any = False,
) -> jnp.ndarray:
    """Scalar Laplacian on a uniform rectilinear grid (sum of second derivatives)."""
    order = validate_fd_order(order)
    f = jnp.asarray(field)
    if axes is None:
        axes = tuple(range(f.ndim))
    if len(spacings) != len(axes):
        raise ValueError("spacings and axes must have the same length")
    flags = _normalize_periodic_flags(periodic, len(axes))
    acc = jnp.zeros_like(f)
    for i, (h, ax) in enumerate(zip(spacings, axes)):
        acc = acc + diff2_uniform(
            f,
            float(h),
            order=order,
            axis=int(ax),
            boundary_closure=boundary_closure,
            use_field_boundary_values=use_field_boundary_values,
            periodic=flags[i],
        )
    return acc


def fornberg_weights(z: float, x: Sequence[float], m: int) -> Any:
    """
    Fornberg (1988) finite-difference weights for derivatives ``0..m`` at ``z``.

    ``x`` is the stencil abscissae. Returns array ``c`` with shape ``(m + 1, len(x))``
    where ``c[k, j]`` multiplies ``f(x[j])`` for the ``k``-th derivative.
    """
    import numpy as np

    x_arr = np.asarray(x, dtype=float)
    n = int(x_arr.size) - 1
    if n < 0:
        raise ValueError("fornberg_weights: empty stencil")
    if m < 0:
        raise ValueError("fornberg_weights: m must be non-negative")
    c = np.zeros((m + 1, n + 1), dtype=float)
    c1 = 1.0
    c4 = x_arr[0] - float(z)
    c[0, 0] = 1.0
    for i in range(1, n + 1):
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = x_arr[i] - float(z)
        for j in range(i):
            c3 = x_arr[i] - x_arr[j]
            if abs(c3) < 1e-30:
                raise ValueError("fornberg_weights: duplicate stencil abscissae")
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[k, i] = c1 * (k * c[k - 1, i - 1] - c5 * c[k, i - 1]) / c2
                c[0, i] = -c1 * c5 * c[0, i - 1] / c2
            for k in range(mn, 0, -1):
                c[k, j] = (c4 * c[k, j] - k * c[k - 1, j]) / c3
            c[0, j] = c4 * c[0, j] / c3
        c1 = c2
    return c


def differentiation_matrix_1d(
    coord: Any,
    *,
    deriv_order: int = 1,
    fd_order: int = 2,
) -> Any:
    """
    Dense 1D differentiation matrix using Fornberg weights on a (possibly non-uniform) grid.

    Stencil width is ``fd_order + 1`` (3/5/7 for orders 2/4/6). Near boundaries the stencil
    is shifted to stay inside the domain (one-sided of matching width).
    """
    import numpy as np

    order = validate_fd_order(fd_order)
    if deriv_order not in (1, 2):
        raise ValueError(f"deriv_order must be 1 or 2, got {deriv_order}")
    x = np.asarray(jnp.asarray(coord).reshape(-1), dtype=float)
    n = int(x.size)
    width = order + 1
    if n < width:
        raise ValueError(
            f"need at least {width} points for fd_order={order}, got {n}"
        )
    D = np.zeros((n, n), dtype=float)
    half = width // 2
    for i in range(n):
        left = i - half
        right = left + width
        if left < 0:
            left, right = 0, width
        elif right > n:
            left, right = n - width, n
        idx = np.arange(left, right)
        w = fornberg_weights(float(x[i]), x[idx], deriv_order)
        D[i, idx] = w[deriv_order, :]
    return jnp.asarray(D)


def diff1_nonuniform(
    values: jnp.ndarray,
    coord: jnp.ndarray,
    *,
    order: int = 2,
    axis: int = -1,
) -> jnp.ndarray:
    """First derivative along ``axis`` on a (possibly non-uniform) 1D coordinate."""
    order = validate_fd_order(order)
    a = jnp.moveaxis(jnp.asarray(values), axis, -1)
    c = jnp.reshape(jnp.asarray(coord), (-1,))
    if int(c.shape[0]) != int(a.shape[-1]):
        raise ValueError(
            f"diff1_nonuniform: coord length {int(c.shape[0])} != axis length {int(a.shape[-1])}"
        )
    D = differentiation_matrix_1d(c, deriv_order=1, fd_order=order)
    flat = a.reshape(-1, a.shape[-1])
    out = flat @ D.T
    return jnp.moveaxis(out.reshape(a.shape), -1, axis)


def diff2_nonuniform(
    values: jnp.ndarray,
    coord: jnp.ndarray,
    *,
    order: int = 2,
    axis: int = -1,
) -> jnp.ndarray:
    """Second derivative along ``axis`` on a (possibly non-uniform) 1D coordinate."""
    order = validate_fd_order(order)
    a = jnp.moveaxis(jnp.asarray(values), axis, -1)
    c = jnp.reshape(jnp.asarray(coord), (-1,))
    if int(c.shape[0]) != int(a.shape[-1]):
        raise ValueError(
            f"diff2_nonuniform: coord length {int(c.shape[0])} != axis length {int(a.shape[-1])}"
        )
    D = differentiation_matrix_1d(c, deriv_order=2, fd_order=order)
    flat = a.reshape(-1, a.shape[-1])
    out = flat @ D.T
    return jnp.moveaxis(out.reshape(a.shape), -1, axis)


def subsample_every_other(values: jnp.ndarray, coord: jnp.ndarray, *, axis: int = -1):
    """
    Every-other-point coarse grid along ``axis`` (even indices).

    Returns ``(values_coarse, coord_coarse, fine_index)`` where ``fine_index`` maps
    coarse samples back onto the fine axis for error comparison.
    """
    a = jnp.moveaxis(jnp.asarray(values), axis, -1)
    c = jnp.reshape(jnp.asarray(coord), (-1,))
    n = int(c.shape[0])
    if n < 4:
        raise ValueError("subsample_every_other: need at least 4 points")
    idx = jnp.arange(0, n, 2)
    if int(idx.shape[0]) < 3:
        raise ValueError("subsample_every_other: coarse grid too short")
    ac = a[..., idx]
    cc = c[idx]
    return jnp.moveaxis(ac, -1, axis), cc, idx
