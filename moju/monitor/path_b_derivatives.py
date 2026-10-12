"""
Opt-in Path B fill for **law** inputs (gradients, Laplacians, etc.).

When ``fill_law_recipes`` is True, fills registered ``Laws.*`` arguments via
``law_fd_recipes``. Spatial derivatives use finite differences by default, or
periodic Fourier (FFT) differentiation when ``PathBGridConfig.diff_method=\"spectral\"``
and ``periodic=True``. Temporal ``dt`` / ``dtt`` always use FD.

Does not overwrite existing non-None entries in ``state_pred``.
Use with structured grids; see ``PathBGridConfig`` for layout conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from moju.monitor.fd_stencils import effective_fd_order


@dataclass(frozen=True)
class PathBGridConfig:
    """
    Coordinate layout for ``fill_path_b_derivatives``.

    - **meshgrid**: each present spatial coord array has the **same shape** as scalar fields.
      For 2D/3D, **rectilinear** grids (e.g. ``jnp.meshgrid(xs, ys, indexing='ij')``) are detected
      and differenced with 1D axes. For **curvilinear** structured meshes, differentiate in
      computational coordinates and transform with metrics via
      :mod:`moju.monitor.curvilinear` (not auto-selected by this config yet — call explicitly).
    - **separable**: ``x`` length ``nx``, ``y`` length ``ny``, ``z`` length ``nz``; field shape
      ``(nx,)``, ``(nx, ny)``, or ``(nx, ny, nz)``. Passed as 1D spacing vectors to ``jnp.gradient``.

    ``diff_method``:
      - ``\"fd\"`` (default): central finite differences (``jnp.gradient``).
      - ``\"spectral\"``: periodic Fourier spatial derivatives (requires ``periodic=True`` and
        uniform spacing). Temporal fill stays FD. Never enabled by bare
        ``auto_path_b_derivatives=True`` — pass an explicit ``PathBGridConfig``.

    ``fd_order`` (Moju 2.0 default **4**):
      - ``4`` (default), ``2``, or ``6``: explicit stencils. Uniform grids use centred /
        Fornberg one-sided boundaries; **non-uniform** grids use Fornberg weights
        (:mod:`moju.monitor.fd_stencils`).
      - ``None`` or ``\"legacy\"``: 1.x ``jnp.gradient`` path (bit-exact with moju<2).

    ``fd_order_time``: same convention for temporal ``dt`` / ``dtt`` (default **4**).

    ``estimate_derivative_error``: when True, also build a coarser (every-other-point) fill
    for diagnostic error estimates; never changes residual values. Pair with
    ``derivative_error_method`` on :meth:`ResidualEngine.compute_residuals`.

    ``boundary_closure`` (Phase A4, opt-in; default ``\"matching\"``):
      - ``\"matching\"``: Fornberg one-sided of the same formal order as the interior.
      - ``\"order_reduced\"``: boundary rows one order lower (4→2, 6→4).
      - ``\"sbp\"``: summation-by-parts style boundary blocks.
      - ``\"bc_aware\"``: ghost-point centred closures using field boundary samples as
        Dirichlet data when ``use_field_boundary_values`` is True (never assumed
        otherwise — pass explicit values via stencil kwargs).

    ``periodic`` (never inferred):
      - ``False`` (default): every spatial axis uses boundary closures.
      - ``True``: every spatial axis uses wrap-around central FD (or FFT when spectral).
      - per-axis sequence, e.g. ``(False, True)`` for ``x`` bounded / ``y`` periodic.
        Misdeclaring a periodic physical axis as bounded inflates boundary residuals.

    Spectral ``diff_method`` requires **all** spatial axes declared periodic
    (``periodic=True`` or a sequence of all ``True``) and uniform spacing, else
    :class:`ValueError`.
    """

    layout: Literal["meshgrid", "separable"] = "meshgrid"
    spatial_dimension: Union[Literal[1, 2, 3], Literal["auto"]] = "auto"
    steady: bool = True
    key_x: str = "x"
    key_y: str = "y"
    key_z: str = "z"
    key_t: str = "t"
    diff_method: Literal["fd", "spectral"] = "fd"
    periodic: Any = False
    fd_order: Any = 4
    fd_order_time: Any = 4
    estimate_derivative_error: bool = False
    boundary_closure: str = "matching"
    use_field_boundary_values: bool = False

    def spatial_periodic_flags(self, n_axes: int) -> Tuple[bool, ...]:
        """Normalize ``periodic`` to a length-``n_axes`` bool tuple."""
        from moju.monitor.fd_stencils import _normalize_periodic_flags

        return _normalize_periodic_flags(self.periodic, int(n_axes))

    def all_spatial_axes_periodic(self, n_axes: Optional[int] = None) -> bool:
        """True when every declared spatial axis is periodic."""
        if isinstance(self.periodic, bool) or self.periodic is None:
            return bool(self.periodic)
        flags = tuple(bool(p) for p in self.periodic)
        if n_axes is not None and len(flags) != int(n_axes):
            raise ValueError(
                f"periodic sequence length {len(flags)} must match n_axes={n_axes}"
            )
        return bool(flags) and all(flags)


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
    K: jnp.ndarray,
    coord_list: Sequence[jnp.ndarray],
    *,
    fd_order: Optional[int] = None,
    warnings: Optional[List[str]] = None,
    boundary_closure: str = "matching",
    use_field_boundary_values: bool = False,
    periodic: Any = False,
) -> Tuple[jnp.ndarray, ...]:
    """
    ``jnp.gradient(K, *coords)`` but use scalar spacings when each 1D coord is uniform.

    When ``fd_order`` is 2/4/6 and every axis is uniform, uses :mod:`moju.monitor.fd_stencils`.
    Keeps Path B / law-FD behavior stable across JAX versions when ``fd_order`` is ``None``.
    ``periodic`` may be a bool or per-axis sequence.
    """
    from moju.monitor.fd_stencils import _normalize_periodic_flags

    coords = [jnp.asarray(x) for x in coord_list]
    hs = [_uniform_1d_spacing(c) for c in coords]
    flags = _normalize_periodic_flags(periodic, len(coords))
    order = effective_fd_order(fd_order)
    if order is not None:
        from moju.monitor.fd_stencils import diff1_nonuniform, diff1_uniform

        if len(hs) == len(coords) and all(h is not None for h in hs):
            parts = [
                diff1_uniform(
                    K,
                    float(h),
                    order=order,
                    axis=i,
                    boundary_closure=boundary_closure,
                    use_field_boundary_values=use_field_boundary_values,
                    periodic=flags[i],
                )
                for i, h in enumerate(hs)
            ]
            return tuple(parts)
        # Non-uniform: Fornberg weights per axis (closure modes / periodic are uniform-only)
        parts = []
        for i, c in enumerate(coords):
            if flags[i]:
                raise ValueError(
                    "periodic=True on a non-uniform axis is not supported; "
                    "use uniform spacing for wrap-around FD"
                )
            parts.append(diff1_nonuniform(K, c, order=order, axis=i))
        return tuple(parts)
    if len(hs) == len(coords) and all(h is not None for h in hs):
        return jnp.gradient(K, *tuple(hs))
    return jnp.gradient(K, *coords)


def _grad_1d_nonuniform(
    values: jnp.ndarray,
    coord: jnp.ndarray,
    *,
    fd_order: Any = None,
    warnings: Optional[List[str]] = None,
    boundary_closure: str = "matching",
    use_field_boundary_values: bool = False,
    periodic: bool = False,
) -> jnp.ndarray:
    h = _uniform_1d_spacing(coord)
    order = effective_fd_order(fd_order)
    if order is not None and h is not None:
        from moju.monitor.fd_stencils import diff1_uniform

        return diff1_uniform(
            values,
            float(h),
            order=order,
            axis=0,
            boundary_closure=boundary_closure,
            use_field_boundary_values=use_field_boundary_values,
            periodic=periodic,
        )
    if order is not None and h is None:
        from moju.monitor.fd_stencils import diff1_nonuniform

        return diff1_nonuniform(values, coord, order=order, axis=0)
    if h is not None:
        return jnp.asarray(jnp.gradient(values, h))
    return jnp.asarray(jnp.gradient(values, coord))


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

    if cfg.diff_method == "spectral":
        from moju.monitor.path_b_spectral import spectral_grad_along_named_axis

        return spectral_grad_along_named_axis(
            K, deriv_axis, cfg, x, y, z, dim, warnings
        )

    pflags = cfg.spatial_periodic_flags(dim)

    if cfg.layout == "separable":
        try:
            coords = _separable_1d_coords(K.shape, x, y, z, dim)
        except ValueError as e:
            warnings.append(str(e))
            return None
        try:
            grads = _jnp_gradient_multi(
                K,
                coords,
                fd_order=cfg.fd_order,
                warnings=warnings,
                boundary_closure=getattr(cfg, "boundary_closure", "matching"),
                use_field_boundary_values=bool(
                    getattr(cfg, "use_field_boundary_values", False)
                ),
                periodic=pflags,
            )
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
        out = _grad_1d_nonuniform(
            K1,
            c1,
            fd_order=cfg.fd_order,
            warnings=warnings,
            boundary_closure=getattr(cfg, "boundary_closure", "matching"),
            use_field_boundary_values=bool(
                getattr(cfg, "use_field_boundary_values", False)
            ),
            periodic=pflags[0],
        )
        return jnp.reshape(out, orig_shape)
    rect1d = _rectilinear_meshgrid_1d_axes(K, x, y, z, dim)
    if rect1d is not None:
        try:
            grads = _jnp_gradient_multi(
                K,
                rect1d,
                fd_order=cfg.fd_order,
                warnings=warnings,
                boundary_closure=getattr(cfg, "boundary_closure", "matching"),
                use_field_boundary_values=bool(
                    getattr(cfg, "use_field_boundary_values", False)
                ),
                periodic=pflags,
            )
        except Exception as e:  # noqa: BLE001
            warnings.append(f"jnp.gradient failed: {e}")
            return None
        return grads[axis_index]

    sep_axes = _meshgrid_separable_axis_coords(K, x, y, z, dim)
    if sep_axes is not None:
        try:
            grads = _jnp_gradient_multi(
                K,
                sep_axes,
                fd_order=cfg.fd_order,
                warnings=warnings,
                boundary_closure=getattr(cfg, "boundary_closure", "matching"),
                use_field_boundary_values=bool(
                    getattr(cfg, "use_field_boundary_values", False)
                ),
                periodic=pflags,
            )
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
    order_t = effective_fd_order(cfg.fd_order_time)

    def col_grad(col: jnp.ndarray) -> jnp.ndarray:
        if order_t is not None and ht is not None:
            from moju.monitor.fd_stencils import diff1_uniform

            return diff1_uniform(col, float(ht), order=order_t, axis=0)
        if order_t is not None and ht is None:
            from moju.monitor.fd_stencils import diff1_nonuniform

            return diff1_nonuniform(col, t, order=order_t, axis=0)
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
    return_provenance: bool = False,
) -> Any:
    """
    When ``fill_law_recipes`` is True and ``laws_spec`` is non-empty, fills **registered**
    ``Laws.*`` inputs (gradients, Laplacians, time derivatives) from primitive fields on the
    same grid; see ``moju.monitor.law_fd_recipes``.

    Spatial fill uses ``grid.diff_method`` (``\"fd\"`` default, or ``\"spectral\"`` with
    ``periodic=True``). Temporal ``dt`` / ``dtt`` always use FD.

    ``constitutive_audit`` is accepted for API compatibility but is not used for derivative fill.

    Returns ``(new_state, warnings)``, or ``(new_state, warnings, provenance)`` with
    ``return_provenance=True`` (filled key -> ``"finite_difference"`` | ``"spectral"``).
    """
    cfg = grid or PathBGridConfig()
    warnings: List[str] = []
    if cfg.diff_method == "spectral":
        from moju.monitor.path_b_spectral import validate_spectral_grid_config

        # Hard error (not a soft warning) when spectral requirements are not met.
        validate_spectral_grid_config(cfg)
    if effective_fd_order(cfg.fd_order) is not None and cfg.diff_method == "spectral":
        warnings.append(
            "fd_order is ignored when diff_method='spectral' (spatial derivatives use FFT)"
        )
    c = dict(constants or {})
    state: Dict[str, Any] = dict(state_pred) if copy else state_pred
    provenance: Dict[str, str] = {}

    if fill_law_recipes and laws_spec:
        from moju.monitor.law_fd_recipes import fill_law_fd_from_primitives

        state, law_warn, provenance = fill_law_fd_from_primitives(
            state,
            list(laws_spec),
            constants=c,
            grid=cfg,
            copy=False,
            return_provenance=True,
        )
        warnings.extend(law_warn)

    if return_provenance:
        return state, warnings, provenance
    return state, warnings


def fill_path_b_spectral(
    state_pred: Dict[str, Any],
    *,
    constitutive_audit: Sequence[Dict[str, Any]] = (),
    laws_spec: Sequence[Dict[str, Any]] = (),
    constants: Optional[Dict[str, Any]] = None,
    grid: Optional[PathBGridConfig] = None,
    copy: bool = True,
    return_provenance: bool = False,
) -> Any:
    """
    Path B law-input fill with **periodic Fourier** spatial derivatives.

    Sets ``diff_method=\"spectral\"`` and ``periodic=True`` on ``grid`` (or a default
    ``PathBGridConfig``) and always enables law recipes. Temporal derivatives remain FD.
    ``return_provenance`` behaves as in :func:`fill_path_b_derivatives`.
    """
    base = grid or PathBGridConfig()
    cfg = replace(base, diff_method="spectral", periodic=True)
    return fill_path_b_derivatives(
        state_pred,
        constitutive_audit=constitutive_audit,
        laws_spec=laws_spec,
        constants=constants,
        grid=cfg,
        copy=copy,
        fill_law_recipes=True,
        return_provenance=return_provenance,
    )


__all__ = [
    "PathBGridConfig",
    "fill_path_b_derivatives",
    "fill_path_b_spectral",
]
