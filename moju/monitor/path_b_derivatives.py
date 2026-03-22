"""
Opt-in finite-difference fill for Path B monitor derivative keys (d_<field>_dx, _dy, _dz, _dt).

Does not overwrite existing non-None entries in ``state_pred``. Use with structured grids;
see ``PathBGridConfig`` for layout and dimension conventions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from moju.monitor.derivative_keys import collect_audit_derivative_keys, derivative_state_key

_DERIV_KEY_RE = re.compile(r"^d_(.+)_((?:dx|dy|dz)|dt)$")
_SUFFIX_TO_DERIV = {"dx": "x", "dy": "y", "dz": "z", "dt": "t"}


@dataclass(frozen=True)
class PathBGridConfig:
    """
    Coordinate layout for ``fill_path_b_derivatives``.

    - **meshgrid**: each present spatial coord array has the **same shape** as scalar fields.
      For 2D/3D, **rectilinear** grids (e.g. ``jnp.meshgrid(xs, ys, indexing='ij')``) are detected
      and differenced with 1D axes; general curvilinear mesh coordinates are not supported for FD.
    - **separable**: ``x`` length ``nx``, ``y`` length ``ny``, ``z`` length ``nz``; field shape
      ``(nx,)``, ``(nx, ny)``, or ``(nx, ny, nz)``. Passed as 1D spacing vectors to ``jnp.gradient``.
    """

    layout: Literal["meshgrid", "separable"] = "meshgrid"
    spatial_dimension: Union[Literal[1, 2, 3], Literal["auto"]] = "auto"
    steady: bool = True
    key_x: str = "x"
    key_y: str = "y"
    key_z: str = "z"
    key_t: str = "t"


def _merged(state: Dict[str, Any], constants: Dict[str, Any]) -> Dict[str, Any]:
    return {**state, **constants}


def _parse_deriv_key(key: str) -> Optional[Tuple[str, str]]:
    m = _DERIV_KEY_RE.match(key)
    if not m:
        return None
    field, suf = m.group(1), m.group(2)
    deriv = _SUFFIX_TO_DERIV.get(suf)
    if deriv is None:
        return None
    return field, deriv


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
        try:
            grads = _jnp_gradient_multi(K, coords)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"jnp.gradient failed: {e}")
            return None
        return grads[axis_index]

    # meshgrid: coordinate arrays same shape as K
    if dim == 1:
        c = x
        if c is None or c.shape != K.shape:
            warnings.append("meshgrid 1D: need x same shape as field")
            return None
        return _grad_1d_nonuniform(K, c)
    rect1d = _rectilinear_meshgrid_1d_axes(K, x, y, z, dim)
    if rect1d is not None:
        try:
            grads = _jnp_gradient_multi(K, rect1d)
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

    def col_grad(col: jnp.ndarray) -> jnp.ndarray:
        if ht is not None:
            return jnp.asarray(jnp.gradient(col, ht))
        return jnp.asarray(jnp.gradient(col, t))

    d2 = jax.vmap(col_grad, in_axes=1, out_axes=1)(K2)
    return jnp.reshape(d2, K.shape)


def fill_path_b_derivatives(
    state_pred: Dict[str, Any],
    *,
    constitutive_audit: Sequence[Dict[str, Any]] = (),
    scaling_audit: Sequence[Dict[str, Any]] = (),
    laws_spec: Sequence[Dict[str, Any]] = (),
    constants: Optional[Dict[str, Any]] = None,
    grid: Optional[PathBGridConfig] = None,
    copy: bool = True,
    fill_law_recipes: bool = False,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Fill missing monitor derivative keys using finite differences.

    When ``fill_law_recipes`` is True and ``laws_spec`` is non-empty, also fills **registered**
    ``Laws.*`` inputs (gradients, Laplacians, time derivatives) from primitive fields on the
    same grid; see ``moju.monitor.law_fd_recipes``.

    Returns ``(new_state, warnings)``. Skips any key already present with a **non-None** value.
    """
    cfg = grid or PathBGridConfig()
    c = dict(constants or {})
    state: Dict[str, Any] = dict(state_pred) if copy else state_pred
    m = _merged(state, c)

    spatial_needed, temporal_needed = collect_audit_derivative_keys(
        list(constitutive_audit), list(scaling_audit)
    )
    warnings: List[str] = []

    for key in sorted(spatial_needed | temporal_needed):
        if state.get(key) is not None:
            continue
        parsed = _parse_deriv_key(key)
        if parsed is None:
            continue
        field, deriv = parsed
        Kraw = m.get(field)
        if Kraw is None:
            warnings.append(f"skip {key}: field {field!r} missing")
            continue
        K = jnp.asarray(Kraw)
        try:
            if deriv == "t":
                arr = _fill_temporal_derivative(K, cfg, m, warnings)
            else:
                arr = _fill_spatial_derivative(K, deriv, cfg, m, warnings)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"{key}: {e}")
            arr = None
        if arr is not None:
            state[key] = arr
            m[key] = arr

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
    "PathBGridConfig",
    "fill_path_b_derivatives",
]
