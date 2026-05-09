"""
Opt-in finite-difference fill for Path B: **law** inputs (gradients, Laplacians, etc.).

When ``fill_law_recipes`` is True, fills registered ``Laws.*`` arguments via
``law_fd_recipes``. Does not overwrite existing non-None entries in ``state_pred``.
Use with structured grids; see ``PathBGridConfig`` for layout conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

MIN_POINTS_FD_ORDER_4 = 5
"""Minimum points per axis for 4th-order stencils; smaller axes fall back to 2nd order."""


@dataclass(frozen=True)
class PathBGridConfig:
    """
    Coordinate layout for ``fill_path_b_derivatives``.

    - **meshgrid**: each present spatial coord array has the **same shape** as scalar fields.
      For 2D/3D, **rectilinear** grids (e.g. ``jnp.meshgrid(xs, ys, indexing='ij')``) are detected
      and differenced with 1D axes; general curvilinear mesh coordinates are not supported for FD.
    - **separable**: ``x`` length ``nx``, ``y`` length ``ny``, ``z`` length ``nz``; field shape
      ``(nx,)``, ``(nx, ny)``, or ``(nx, ny, nz)``. Passed as 1D spacing vectors to ``jnp.gradient``.
    - **fd_order**: ``4`` (default) uses explicit 4th-order stencils on **uniform** structured grids
      (interior and boundary bands). Non-uniform spacing or non-separable meshgrid paths fall back
      to 2nd-order ``jnp.gradient`` with an optional warning. ``2`` preserves legacy behavior.
    """

    layout: Literal["meshgrid", "separable"] = "meshgrid"
    spatial_dimension: Union[Literal[1, 2, 3], Literal["auto"]] = "auto"
    steady: bool = True
    fd_order: Literal[2, 4] = 4
    key_x: str = "x"
    key_y: str = "y"
    key_z: str = "z"
    key_t: str = "t"


def _merged(state: Dict[str, Any], constants: Dict[str, Any]) -> Dict[str, Any]:
    """Constants first, then ``state`` — prediction / NPZ fields must not be masked by constants."""
    return {**constants, **state}


def _get_coord(
    m: Dict[str, Any], cfg: PathBGridConfig, axis: str
) -> Optional[jnp.ndarray]:
    k = {"x": cfg.key_x, "y": cfg.key_y, "z": cfg.key_z, "t": cfg.key_t}[axis]
    v = m.get(k)
    if v is None:
        return None
    return jnp.asarray(v)


def _spatial_ndim_from_field(K: jnp.ndarray, steady: bool) -> int:
    if steady:
        return int(K.ndim)
    if K.ndim < 2:
        return K.ndim
    return int(K.ndim - 1)


def _infer_spatial_dim(K: jnp.ndarray, steady: bool, declared: Union[int, str]) -> int:
    if declared != "auto":
        return int(declared)
    # Column-shaped uploads (N, 1) are common; treat as 1D spatial for FD (uses coord ``x``).
    if steady and K.ndim == 2 and int(K.shape[1]) == 1:
        return 1
    # Row-shaped (1, N) is common in tabular / exported NPZ; same 1D spatial interpretation.
    if steady and K.ndim == 2 and int(K.shape[0]) == 1:
        return 1
    return _spatial_ndim_from_field(K, steady)


def _grad_along_axis_uniform(K: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Central differences; uniform spacing (JAX default)."""
    return jnp.gradient(K, axis=axis)


def _uniform_1d_spacing(
    c: jnp.ndarray, *, rtol: float = 1e-4, atol: float = 1e-6
) -> Optional[float]:
    """
    Return spacing ``h`` if ``c`` is effectively uniform (``diff(c)`` constant).

    JAX ``jnp.gradient(values, coord)`` with a coordinate array can fail on uniform
    ``linspace`` grids in float32 (tiny ``diff`` jitter) or raise
    "Non-constant spacing not implemented" across versions. Using scalar ``h`` avoids that.
    """
    c = jnp.reshape(jnp.asarray(c), (-1,))
    n = int(c.shape[0])
    if n < 2:
        return None
    dc = jnp.diff(c)
    if not bool(jnp.allclose(dc, dc[0], rtol=rtol, atol=atol)):
        return None
    h = dc[0]
    if abs(float(h)) < 1e-15:
        return None
    return float(h)


def _jnp_gradient_multi(
    K: jnp.ndarray, coord_list: Sequence[jnp.ndarray]
) -> Tuple[jnp.ndarray, ...]:
    """
    ``jnp.gradient(K, *coords)`` but use scalar spacings when each 1D coord is uniform.

    Keeps Path B / law-FD behavior stable across JAX versions (esp. float32 + Python 3.9 CI).
    """
    coords = [jnp.asarray(x) for x in coord_list]
    hs = [_uniform_1d_spacing(c) for c in coords]
    if len(hs) == len(coords) and all(h is not None for h in hs):
        return jnp.gradient(K, *tuple(hs))
    return jnp.gradient(K, *coords)


def _grad_1d_nonuniform(values: jnp.ndarray, coord: jnp.ndarray) -> jnp.ndarray:
    h = _uniform_1d_spacing(coord)
    if h is not None:
        return jnp.asarray(jnp.gradient(values, h))
    return jnp.asarray(jnp.gradient(values, coord))


def _diff1_uniform_4th_1d(u: jnp.ndarray, h: Any) -> jnp.ndarray:
    """First derivative on a uniform 1D grid; 4th-order interior and boundary bands."""
    u = jnp.asarray(u)
    h = jnp.asarray(h, dtype=u.dtype)
    n = int(u.shape[0])
    if n < MIN_POINTS_FD_ORDER_4:
        return jnp.asarray(jnp.gradient(u, h))
    inv12h = 1.0 / (12.0 * h)
    out = jnp.empty_like(u)
    out = out.at[0].set(
        (-25.0 * u[0] + 48.0 * u[1] - 36.0 * u[2] + 16.0 * u[3] - 3.0 * u[4]) * inv12h
    )
    out = out.at[1].set(
        (-3.0 * u[0] - 10.0 * u[1] + 18.0 * u[2] - 6.0 * u[3] + u[4]) * inv12h
    )
    out = out.at[n - 2].set(
        (
            -u[n - 5]
            + 6.0 * u[n - 4]
            - 18.0 * u[n - 3]
            + 10.0 * u[n - 2]
            + 3.0 * u[n - 1]
        )
        * inv12h
    )
    out = out.at[n - 1].set(
        (
            25.0 * u[n - 1]
            - 48.0 * u[n - 2]
            + 36.0 * u[n - 3]
            - 16.0 * u[n - 4]
            + 3.0 * u[n - 5]
        )
        * inv12h
    )
    im2 = u[0:-4]
    im1 = u[1:-3]
    ip1 = u[3:-1]
    ip2 = u[4:]
    interior = (im2 - 8.0 * im1 + 8.0 * ip1 - ip2) * inv12h
    out = out.at[2 : n - 2].set(interior)
    return out


def _diff2_uniform_4th_1d(u: jnp.ndarray, h: Any) -> jnp.ndarray:
    """Second derivative on a uniform 1D grid; 4th-order interior and boundary bands."""
    u = jnp.asarray(u)
    h = jnp.asarray(h, dtype=u.dtype)
    n = int(u.shape[0])
    if n < MIN_POINTS_FD_ORDER_4:
        g = jnp.asarray(jnp.gradient(u, h))
        return jnp.asarray(jnp.gradient(g, h))
    inv12h2 = 1.0 / (12.0 * h * h)
    out = jnp.empty_like(u)
    out = out.at[0].set(
        (35.0 * u[0] - 104.0 * u[1] + 114.0 * u[2] - 56.0 * u[3] + 11.0 * u[4]) * inv12h2
    )
    out = out.at[1].set(
        (11.0 * u[0] - 20.0 * u[1] + 6.0 * u[2] + 4.0 * u[3] - u[4]) * inv12h2
    )
    out = out.at[n - 2].set(
        (
            -u[n - 5]
            + 4.0 * u[n - 4]
            + 6.0 * u[n - 3]
            - 20.0 * u[n - 2]
            + 11.0 * u[n - 1]
        )
        * inv12h2
    )
    out = out.at[n - 1].set(
        (
            11.0 * u[n - 5]
            - 56.0 * u[n - 4]
            + 114.0 * u[n - 3]
            - 104.0 * u[n - 2]
            + 35.0 * u[n - 1]
        )
        * inv12h2
    )
    im2 = u[0:-4]
    im1 = u[1:-3]
    ui = u[2:-2]
    ip1 = u[3:-1]
    ip2 = u[4:]
    interior = (-im2 + 16.0 * im1 - 30.0 * ui + 16.0 * ip1 - ip2) * inv12h2
    out = out.at[2 : n - 2].set(interior)
    return out


def _diff1_along_axis_nd(K: jnp.ndarray, axis: int, h: float) -> jnp.ndarray:
    """First partial derivative along ``axis`` with uniform spacing ``h``."""
    K = jnp.asarray(K)
    ax = int(axis) % int(K.ndim)
    moved = jnp.moveaxis(K, ax, -1)
    n = int(moved.shape[-1])
    flat = jnp.reshape(moved, (-1, n))

    def row_d1(row: jnp.ndarray) -> jnp.ndarray:
        return _diff1_uniform_4th_1d(row, h)

    out_flat = jax.vmap(row_d1)(flat)
    out = jnp.reshape(out_flat, moved.shape)
    return jnp.moveaxis(out, -1, ax)


def _diff2_along_axis_nd(K: jnp.ndarray, axis: int, h: float) -> jnp.ndarray:
    """Second partial derivative along ``axis`` with uniform spacing ``h``."""
    K = jnp.asarray(K)
    ax = int(axis) % int(K.ndim)
    moved = jnp.moveaxis(K, ax, -1)
    n = int(moved.shape[-1])
    flat = jnp.reshape(moved, (-1, n))

    def row_d2(row: jnp.ndarray) -> jnp.ndarray:
        return _diff2_uniform_4th_1d(row, h)

    out_flat = jax.vmap(row_d2)(flat)
    out = jnp.reshape(out_flat, moved.shape)
    return jnp.moveaxis(out, -1, ax)


def _uniform_spacing_per_axis(
    coord_list: Sequence[jnp.ndarray],
    spatial_shape: Tuple[int, ...],
    warnings: List[str],
    *,
    context: str,
) -> Optional[List[float]]:
    """Return ``h`` per axis if all coords are uniform and lengths match ``spatial_shape``."""
    if len(coord_list) != len(spatial_shape):
        return None
    hs: List[float] = []
    for i, c in enumerate(coord_list):
        c1 = jnp.reshape(jnp.asarray(c), (-1,))
        if int(c1.shape[0]) != int(spatial_shape[i]):
            return None
        h = _uniform_1d_spacing(c1)
        if h is None:
            warnings.append(
                f"{context}: fd_order=4 requires uniform spacing on axis {i}; "
                "falling back to 2nd-order jnp.gradient for this operation"
            )
            return None
        hs.append(h)
    return hs


def _spatial_gradients_uniform_4th(
    K: jnp.ndarray, hs: Sequence[float]
) -> Tuple[jnp.ndarray, ...]:
    return tuple(_diff1_along_axis_nd(K, ax, float(hs[ax])) for ax in range(len(hs)))


def _laplacian_uniform_4th_nd(K: jnp.ndarray, hs: Sequence[float]) -> jnp.ndarray:
    acc = jnp.zeros_like(K)
    for ax, h in enumerate(hs):
        acc = acc + _diff2_along_axis_nd(K, ax, float(h))
    return acc


def _separable_1d_coords(
    spatial_shape: Tuple[int, ...],
    x: Optional[jnp.ndarray],
    y: Optional[jnp.ndarray],
    z: Optional[jnp.ndarray],
    dim: int,
) -> List[jnp.ndarray]:
    """1D coordinate vectors per axis for ``jnp.gradient`` on separable grids."""
    if dim == 1:
        if x is None or x.shape[0] != spatial_shape[0]:
            raise ValueError("separable 1D: need x with length nx")
        return [x]
    if dim == 2:
        if x is None or y is None:
            raise ValueError("separable 2D: need x and y")
        nx, ny = spatial_shape
        if x.shape[0] != nx or y.shape[0] != ny:
            raise ValueError("separable 2D: coord lengths must match field shape")
        return [x, y]
    if dim == 3:
        if x is None or y is None or z is None:
            raise ValueError("separable 3D: need x, y, z")
        nx, ny, nz = spatial_shape
        if x.shape[0] != nx or y.shape[0] != ny or z.shape[0] != nz:
            raise ValueError("separable 3D: coord lengths must match field shape")
        return [x, y, z]
    raise ValueError(f"unsupported spatial dim {dim}")


def _spatial_part(K: jnp.ndarray, steady: bool) -> jnp.ndarray:
    if steady:
        return K
    if K.ndim == 1:
        return K
    return K[0]


def _spatial_shape(K: jnp.ndarray, steady: bool) -> Tuple[int, ...]:
    sp = _spatial_part(K, steady)
    return tuple(int(s) for s in sp.shape)


def _coerce_1d_axis_vector(c: Optional[jnp.ndarray], n: int) -> Optional[jnp.ndarray]:
    """``(n,)`` from any 1D-compatible array; ``None`` if length does not match ``n``."""
    if c is None:
        return None
    a = jnp.reshape(jnp.asarray(c), (-1,))
    if int(a.shape[0]) != int(n):
        return None
    return a


def _meshgrid_separable_axis_coords(
    K: jnp.ndarray,
    x: Optional[jnp.ndarray],
    y: Optional[jnp.ndarray],
    z: Optional[jnp.ndarray],
    dim: int,
) -> Optional[List[jnp.ndarray]]:
    """
    Many NPZs store a tensor-product grid as ``T (nx, ny[, nz])`` with **1D** ``x (nx,)``,
    ``y (ny,)``, ``z (nz,)`` while ``PathBGridConfig.layout`` defaults to ``meshgrid``.
    Return spacing vectors for ``jnp.gradient`` when that layout applies; else ``None``.
    """
    K = jnp.asarray(K)
    if dim == 2:
        if K.ndim != 2:
            return None
        nx, ny = int(K.shape[0]), int(K.shape[1])
        xc = _coerce_1d_axis_vector(x, nx)
        yc = _coerce_1d_axis_vector(y, ny)
        if xc is None or yc is None:
            return None
        return [xc, yc]
    if dim == 3:
        if K.ndim != 3:
            return None
        nx, ny, nz = int(K.shape[0]), int(K.shape[1]), int(K.shape[2])
        xc = _coerce_1d_axis_vector(x, nx)
        yc = _coerce_1d_axis_vector(y, ny)
        zc = _coerce_1d_axis_vector(z, nz)
        if xc is None or yc is None or zc is None:
            return None
        return [xc, yc]
    return None


def _rectilinear_meshgrid_1d_axes(
    K: jnp.ndarray,
    x: Optional[jnp.ndarray],
    y: Optional[jnp.ndarray],
    z: Optional[jnp.ndarray],
    dim: int,
) -> Optional[List[jnp.ndarray]]:
    """
    If ``x,y,z`` are tensor-product coordinates (each depends on a single index),
    return 1D spacing vectors for ``jnp.gradient``. JAX rejects full multi-D coord arrays.
    """
    if dim == 2:
        if x is None or y is None or x.shape != K.shape or y.shape != K.shape:
            return None
        if bool(jnp.allclose(x, x[:, :1])) and bool(jnp.allclose(y, y[:1, :])):
            return [x[:, 0], y[0, :]]
        return None
    if dim == 3:
        if (
            x is None
            or y is None
            or z is None
            or x.shape != K.shape
            or y.shape != K.shape
            or z.shape != K.shape
        ):
            return None
        if (
            bool(jnp.allclose(x, x[:, :1, :1]))
            and bool(jnp.allclose(y, y[:1, :, :1]))
            and bool(jnp.allclose(z, z[:1, :1, :]))
        ):
            return [x[:, 0, 0], y[0, :, 0], z[0, 0, :]]
        return None
    return None


def _is_steady_leading_time_stack(
    K: jnp.ndarray,
    cfg: PathBGridConfig,
    m: Dict[str, Any],
) -> bool:
    """
    ``steady=True`` (Studio default) but the leading axis is often **time / snapshots**, not space.

    - If ``t(nt,)`` exists and matches ``K.shape[0]``, treat as a time stack (including ``(nt, nx)``
      when ``nt == nx`` square 2D arrays).
    - If **no** ``t`` (common in NPZ stacks), use a conservative rule for ``(n0, nx, ny)`` with 1D
      ``x(nx)``, ``y(ny)``: when ``n0 > max(nx, ny)`` and ``z`` does not look like the coordinate
      for the leading **spatial** axis, assume ``n0`` is snapshot index and ``vmap`` spatial FD.
    """
    if not cfg.steady or K.ndim < 2:
        return False
    K = jnp.asarray(K)
    n0 = int(K.shape[0])
    t = _get_coord(m, cfg, "t")
    if t is not None:
        ta = jnp.asarray(t)
        if ta.ndim == 1 and int(ta.shape[0]) == n0:
            if K.ndim >= 3:
                return True
            # 2D (nt, nx): treat as time stack when t matches leading dim and x matches second dim,
            # including square (n, n) where nt == nx (previously excluded and mis-handled as 2D spatial).
            if K.ndim == 2:
                x = _get_coord(m, cfg, "x")
                return _coerce_1d_axis_vector(x, int(K.shape[1])) is not None
        return False

    # No ``t`` (or wrong length): snapshot stack heuristic for 3D-shaped arrays
    if K.ndim == 3:
        z = _get_coord(m, cfg, "z")
        if z is not None:
            zc_n0 = _coerce_1d_axis_vector(z, n0)
            if zc_n0 is not None:
                # z aligns with leading dim → first axis is a spatial direction, not snapshots
                return False
        nx, ny = int(K.shape[1]), int(K.shape[2])
        x = _get_coord(m, cfg, "x")
        y = _get_coord(m, cfg, "y")
        if _coerce_1d_axis_vector(x, nx) is None or _coerce_1d_axis_vector(y, ny) is None:
            return False
        mxy = max(nx, ny)
        mnx = min(nx, ny)
        if mxy == 0:
            return False
        # Many snapshots (nt >> nx) OR short stacks (nt < min in-plane) vs a full spatial brick
        return n0 > mxy or n0 < mnx

    if K.ndim == 2 and int(K.shape[0]) != int(K.shape[1]):
        nx = int(K.shape[1])
        x = _get_coord(m, cfg, "x")
        if _coerce_1d_axis_vector(x, nx) is None:
            return False
        return n0 > nx

    return False


def _fill_spatial_derivative(
    K: jnp.ndarray,
    deriv_axis: str,
    cfg: PathBGridConfig,
    m: Dict[str, Any],
    warnings: List[str],
) -> Optional[jnp.ndarray]:
    """deriv_axis in x,y,z."""
    x, y, z = _get_coord(m, cfg, "x"), _get_coord(m, cfg, "y"), _get_coord(m, cfg, "z")
    steady = cfg.steady
    dim = _infer_spatial_dim(K, steady, cfg.spatial_dimension)

    if steady and _is_steady_leading_time_stack(K, cfg, m):
        dim_s = _infer_spatial_dim(jnp.asarray(K[0]), True, cfg.spatial_dimension)

        def _slice_fill(Ks: jnp.ndarray) -> jnp.ndarray:
            return _fill_spatial_derivative_steady(
                Ks, deriv_axis, cfg, x, y, z, dim_s, []
            )

        first = _slice_fill(K[0])
        if first is None:
            return None
        return jax.vmap(_slice_fill)(K)

    if not steady:
        if K.ndim < 2:
            warnings.append("unsteady field expected ndim>=2 with leading time axis")
            return None
        t = _get_coord(m, cfg, "t")
        if t is None or t.shape[0] != K.shape[0]:
            warnings.append("unsteady: need t(nt,) matching K leading dim for spatial FD along slices")
            return None

        def _slice_fill(Ks: jnp.ndarray) -> jnp.ndarray:
            # Do not mutate shared warnings inside vmap.
            return _fill_spatial_derivative_steady(
                Ks, deriv_axis, cfg, x, y, z, dim, []
            )

        first = _slice_fill(K[0])
        if first is None:
            return None
        return jax.vmap(_slice_fill)(K)

    return _fill_spatial_derivative_steady(K, deriv_axis, cfg, x, y, z, dim, warnings)


def _align_1d_field_and_coord(
    K: jnp.ndarray,
    c: Optional[jnp.ndarray],
    warnings: List[str],
    context: str,
) -> Optional[Tuple[jnp.ndarray, jnp.ndarray, Tuple[int, ...]]]:
    """
    For 1D meshgrid FD, allow ``field`` and ``x`` to differ in shape if sizes match
    (e.g. ``(N,)`` vs ``(N, 1)``), then ravel for ``jnp.gradient``-style ops.
    Returns ``(K1, c1, original_K_shape)``.
    """
    if c is None:
        warnings.append(f"{context}: missing x coordinate")
        return None
    K_a, c_a = jnp.asarray(K), jnp.asarray(c)
    if K_a.size == 0:
        warnings.append(f"{context}: empty field")
        return None
    orig_shape = tuple(int(s) for s in K_a.shape)
    if K_a.shape == c_a.shape:
        if K_a.ndim == 1:
            return K_a, c_a, orig_shape
        # (N, 1), (1, N), etc.: same shape as coord but need 1D vectors for jnp.gradient.
        if min(int(s) for s in K_a.shape) == 1:
            return jnp.reshape(K_a, (-1,)), jnp.reshape(c_a, (-1,)), orig_shape
        return K_a, c_a, orig_shape
    if K_a.size != c_a.size:
        warnings.append(
            f"{context}: x and field must match shape or total size "
            f"(got field {K_a.shape}, x {c_a.shape})"
        )
        return None
    return jnp.reshape(K_a, (-1,)), jnp.reshape(c_a, (-1,)), orig_shape


def _fill_spatial_derivative_steady(
    K: jnp.ndarray,
    deriv_axis: str,
    cfg: PathBGridConfig,
    x: Optional[jnp.ndarray],
    y: Optional[jnp.ndarray],
    z: Optional[jnp.ndarray],
    dim: int,
    warnings: List[str],
) -> Optional[jnp.ndarray]:
    axis_index = {"x": 0, "y": 1, "z": 2}[deriv_axis]
    if axis_index >= dim:
        warnings.append(f"skip d/d{deriv_axis}: spatial_dimension {dim} < axis index")
        return None

    if cfg.layout == "separable":
        try:
            coords = _separable_1d_coords(K.shape, x, y, z, dim)
        except ValueError as e:
            warnings.append(str(e))
            return None
        if cfg.fd_order == 4:
            hs = _uniform_spacing_per_axis(
                coords, tuple(int(s) for s in K.shape), warnings, context="separable"
            )
            if hs is not None and all(
                int(K.shape[i]) >= MIN_POINTS_FD_ORDER_4 for i in range(dim)
            ):
                grads = _spatial_gradients_uniform_4th(K, hs)
                return grads[axis_index]
        try:
            grads = _jnp_gradient_multi(K, coords)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"jnp.gradient failed: {e}")
            return None
        return grads[axis_index]

    # meshgrid: coordinate arrays same shape as K (or same size — ravel)
    if dim == 1:
        aligned = _align_1d_field_and_coord(K, x, warnings, "meshgrid 1D")
        if aligned is None:
            return None
        K1, c1, orig_shape = aligned
        if cfg.fd_order == 4:
            h = _uniform_1d_spacing(c1)
            if h is not None and int(K1.shape[0]) >= MIN_POINTS_FD_ORDER_4:
                out = _diff1_uniform_4th_1d(K1, h)
                return jnp.reshape(out, orig_shape)
            if h is None:
                warnings.append(
                    "meshgrid 1D: fd_order=4 requires uniform x spacing; using 2nd-order gradient"
                )
        out = _grad_1d_nonuniform(K1, c1)
        return jnp.reshape(out, orig_shape)
    rect1d = _rectilinear_meshgrid_1d_axes(K, x, y, z, dim)
    if rect1d is not None:
        if cfg.fd_order == 4:
            hs = _uniform_spacing_per_axis(
                rect1d, tuple(int(s) for s in K.shape), warnings, context="rectilinear meshgrid"
            )
            if hs is not None and all(
                int(K.shape[i]) >= MIN_POINTS_FD_ORDER_4 for i in range(dim)
            ):
                grads = _spatial_gradients_uniform_4th(K, hs)
                return grads[axis_index]
        try:
            grads = _jnp_gradient_multi(K, rect1d)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"jnp.gradient failed: {e}")
            return None
        return grads[axis_index]

    sep_axes = _meshgrid_separable_axis_coords(K, x, y, z, dim)
    if sep_axes is not None:
        if cfg.fd_order == 4:
            hs = _uniform_spacing_per_axis(
                sep_axes, tuple(int(s) for s in K.shape), warnings, context="meshgrid separable axes"
            )
            if hs is not None and all(
                int(K.shape[i]) >= MIN_POINTS_FD_ORDER_4 for i in range(dim)
            ):
                grads = _spatial_gradients_uniform_4th(K, hs)
                return grads[axis_index]
        try:
            grads = _jnp_gradient_multi(K, sep_axes)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"jnp.gradient failed: {e}")
            return None
        return grads[axis_index]

    coords_m = []
    for ax, c in [("x", x), ("y", y), ("z", z)]:
        if ax in ("x", "y", "z") and {"x": 0, "y": 1, "z": 2}[ax] < dim:
            if c is None or c.shape != K.shape:
                warnings.append(f"meshgrid: need {ax} same shape as field")
                return None
            coords_m.append(c)
    if cfg.fd_order == 4:
        warnings.append(
            "fd_order=4: full curvilinear meshgrid coordinates; using 2nd-order jnp.gradient"
        )
    try:
        grads = jnp.gradient(K, *coords_m)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"jnp.gradient failed: {e}")
        return None
    return grads[axis_index]


def _fill_temporal_derivative(
    K: jnp.ndarray,
    cfg: PathBGridConfig,
    m: Dict[str, Any],
    warnings: List[str],
) -> Optional[jnp.ndarray]:
    t = _get_coord(m, cfg, "t")
    if t is None:
        warnings.append("missing t for d/dt")
        return None
    if K.ndim == 0:
        warnings.append("scalar field has no time derivative")
        return None
    if t.shape[0] != K.shape[0]:
        warnings.append("t length must match K leading dimension")
        return None
    nt = K.shape[0]
    tail = int(jnp.prod(jnp.array(K.shape[1:]))) if K.ndim > 1 else 1
    K2 = jnp.reshape(K, (nt, tail))

    ht = _uniform_1d_spacing(t)
    if cfg.fd_order == 4 and ht is None:
        warnings.append(
            "fd_order=4: non-uniform t spacing; using 2nd-order jnp.gradient for d/dt"
        )

    def col_grad(col: jnp.ndarray) -> jnp.ndarray:
        if cfg.fd_order == 4 and ht is not None and nt >= MIN_POINTS_FD_ORDER_4:
            return _diff1_uniform_4th_1d(col, ht)
        if ht is not None:
            return jnp.asarray(jnp.gradient(col, ht))
        return jnp.asarray(jnp.gradient(col, t))

    d2 = jax.vmap(col_grad, in_axes=1, out_axes=1)(K2)
    return jnp.reshape(d2, K.shape)


def fill_path_b_derivatives(
    state_pred: Dict[str, Any],
    *,
    constitutive_audit: Sequence[Dict[str, Any]] = (),
    laws_spec: Sequence[Dict[str, Any]] = (),
    constants: Optional[Dict[str, Any]] = None,
    grid: Optional[PathBGridConfig] = None,
    copy: bool = True,
    fill_law_recipes: bool = False,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    When ``fill_law_recipes`` is True and ``laws_spec`` is non-empty, fills **registered**
    ``Laws.*`` inputs (gradients, Laplacians, time derivatives) from primitive fields on the
    same grid; see ``moju.monitor.law_fd_recipes``.

    ``constitutive_audit`` is accepted for API compatibility but is not used for finite-difference fill.

    Returns ``(new_state, warnings)``.
    """
    cfg = grid or PathBGridConfig()
    c = dict(constants or {})
    state: Dict[str, Any] = dict(state_pred) if copy else state_pred
    warnings: List[str] = []

    if fill_law_recipes and laws_spec:
        from moju.monitor.law_fd_recipes import fill_law_fd_from_primitives

        state, law_warn = fill_law_fd_from_primitives(
            state,
            list(laws_spec),
            constants=c,
            grid=cfg,
            copy=False,
        )
        warnings.extend(law_warn)

    return state, warnings


__all__ = [
    "MIN_POINTS_FD_ORDER_4",
    "PathBGridConfig",
    "fill_path_b_derivatives",
]
