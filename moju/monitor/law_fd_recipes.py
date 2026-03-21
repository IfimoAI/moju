"""
Optional finite-difference fill for ``Laws.*`` inputs (gradients, Laplacians, time derivatives).

Only **registered** law arguments are filled when primitives exist on a structured grid and the
target key is missing (``None``). Does not overwrite non-``None`` values.

Naming convention when ``source_arg`` is omitted: the **state key** for the target (value in
``state_map``) must use a predictable suffix so the primitive field key can be inferred, e.g.
``phi_laplacian`` → primitive ``phi``, ``T_t`` → ``T``, ``rho_grad`` → ``rho``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from moju.monitor.path_b_derivatives import (
    PathBGridConfig,
    _fill_spatial_derivative,
    _fill_spatial_derivative_steady,
    _fill_temporal_derivative,
    _get_coord,
    _infer_spatial_dim,
    _merged,
    _rectilinear_meshgrid_1d_axes,
    _separable_1d_coords,
)


@dataclass(frozen=True)
class LawFDArgRecipe:
    """
    How to FD-fill one ``Laws.*`` argument.

    - ``kind``: ``laplacian`` | ``vector_laplacian`` | ``grad_scalar`` | ``jacobian`` | ``dt`` | ``dtt``
    - ``source_arg``: function argument name whose **state_map value** is the primitive field key
      (e.g. ``u`` for ``u_grad``). If ``None``, infer primitive key from the **target** state key
      using ``_infer_primitive_key``.
    """

    kind: str
    source_arg: Optional[str] = None


def _infer_primitive_key(target_state_key: str, law_arg_name: str) -> Optional[str]:
    """Derive primitive state key from the configured target key string."""
    if target_state_key.endswith("_laplacian"):
        return target_state_key[: -len("_laplacian")]
    if target_state_key.endswith("_grad"):
        return target_state_key[: -len("_grad")]
    if target_state_key.endswith("_tt"):
        return target_state_key[: -len("_tt")]
    if target_state_key.endswith("_t") and not target_state_key.endswith("_tt"):
        # rho_t, T_t, phi_t, u_t, B_t — not *_tt (handled above)
        return target_state_key[: -len("_t")]
    if law_arg_name == "u_grad":
        return "u"
    if law_arg_name == "rho_t":
        return "rho"
    return None


# law_name -> law_fn_arg_name -> recipe
LAW_FD_RECIPES: Dict[str, Dict[str, LawFDArgRecipe]] = {
    "laplace_equation": {"phi_laplacian": LawFDArgRecipe("laplacian")},
    "poisson_equation": {"phi_laplacian": LawFDArgRecipe("laplacian")},
    "helmholtz_equation": {"phi_laplacian": LawFDArgRecipe("laplacian", source_arg="phi")},
    "schrodinger_steady": {"psi_laplacian": LawFDArgRecipe("laplacian")},
    "laplace_beltrami": {},  # metric-specific; no generic FD
    "fourier_conduction": {
        "T_laplacian": LawFDArgRecipe("laplacian"),
        "T_t": LawFDArgRecipe("dt", source_arg="T"),
    },
    "fick_diffusion": {
        "phi_laplacian": LawFDArgRecipe("laplacian"),
        "phi_t": LawFDArgRecipe("dt", source_arg="phi"),
    },
    "advection_diffusion": {
        "phi_laplacian": LawFDArgRecipe("laplacian"),
        "phi_grad": LawFDArgRecipe("grad_scalar", source_arg="phi"),
        "phi_t": LawFDArgRecipe("dt", source_arg="phi"),
    },
    "wave_equation": {
        "phi_laplacian": LawFDArgRecipe("laplacian"),
        "phi_tt": LawFDArgRecipe("dtt", source_arg="phi"),
    },
    "mass_incompressible": {"u_grad": LawFDArgRecipe("jacobian", source_arg="u")},
    "mass_compressible": {
        "rho_t": LawFDArgRecipe("dt", source_arg="rho"),
        "rho_grad": LawFDArgRecipe("grad_scalar", source_arg="rho"),
        "u_grad": LawFDArgRecipe("jacobian", source_arg="u"),
    },
    "momentum_navier_stokes": {
        "u_t": LawFDArgRecipe("dt", source_arg="u"),
        "u_grad": LawFDArgRecipe("jacobian", source_arg="u"),
        "u_laplacian": LawFDArgRecipe("vector_laplacian", source_arg="u"),
        "p_grad": LawFDArgRecipe("grad_scalar", source_arg="p"),
    },
    "stokes_flow": {
        "u_laplacian": LawFDArgRecipe("vector_laplacian", source_arg="u"),
        "p_grad": LawFDArgRecipe("grad_scalar", source_arg="p"),
    },
    "euler_momentum": {
        "u_t": LawFDArgRecipe("dt", source_arg="u"),
        "u_grad": LawFDArgRecipe("jacobian", source_arg="u"),
        "p_grad": LawFDArgRecipe("grad_scalar", source_arg="p"),
    },
    "burgers_equation": {
        "u_t": LawFDArgRecipe("dt", source_arg="u"),
        "u_grad": LawFDArgRecipe("jacobian", source_arg="u"),
        "u_laplacian": LawFDArgRecipe("vector_laplacian", source_arg="u"),
    },
    "darcy_flow": {"p_grad": LawFDArgRecipe("grad_scalar", source_arg="p")},
    "brinkman_extension": {
        "u_laplacian": LawFDArgRecipe("vector_laplacian", source_arg="u"),
        "p_grad": LawFDArgRecipe("grad_scalar", source_arg="p"),
    },
    "viscous_dissipation": {"u_grad": LawFDArgRecipe("jacobian", source_arg="u")},
    # faraday_law: curl E — not supported
    "hookes_law_residual": {},
}


def _scalar_laplacian_steady(
    K: jnp.ndarray,
    cfg: PathBGridConfig,
    x: Optional[jnp.ndarray],
    y: Optional[jnp.ndarray],
    z: Optional[jnp.ndarray],
    dim: int,
    warnings: List[str],
) -> Optional[jnp.ndarray]:
    if cfg.layout == "separable":
        try:
            coords = _separable_1d_coords(K.shape, x, y, z, dim)
        except ValueError as e:
            warnings.append(str(e))
            return None
        try:
            grads = jnp.gradient(K, *coords)
            acc = jnp.zeros_like(K)
            for i, gi in enumerate(grads):
                parts = jnp.gradient(gi, *coords)
                acc = acc + parts[i]
            return acc
        except Exception as e:  # noqa: BLE001
            warnings.append(f"laplacian jnp.gradient failed: {e}")
            return None
    # meshgrid
    if dim == 1:
        c = x
        if c is None or c.shape != K.shape:
            warnings.append("meshgrid 1D laplacian: need x same shape as field")
            return None
        g = _fill_spatial_derivative_steady(K, "x", cfg, x, y, z, dim, [])
        if g is None:
            return None
        return _fill_spatial_derivative_steady(g, "x", cfg, x, y, z, dim, [])
    rect1d = _rectilinear_meshgrid_1d_axes(K, x, y, z, dim)
    if rect1d is not None:
        try:
            grads = jnp.gradient(K, *rect1d)
            acc = jnp.zeros_like(K)
            for i, gi in enumerate(grads):
                parts = jnp.gradient(gi, *rect1d)
                acc = acc + parts[i]
            return acc
        except Exception as e:  # noqa: BLE001
            warnings.append(f"laplacian jnp.gradient failed: {e}")
            return None
    coords_m = []
    for ax, c in [("x", x), ("y", y), ("z", z)]:
        if {"x": 0, "y": 1, "z": 2}[ax] < dim:
            if c is None or c.shape != K.shape:
                warnings.append(f"meshgrid laplacian: need {ax} same shape as field")
                return None
            coords_m.append(c)
    try:
        grads = jnp.gradient(K, *coords_m)
        acc = jnp.zeros_like(K)
        for i, gi in enumerate(grads):
            parts = jnp.gradient(gi, *coords_m)
            acc = acc + parts[i]
        return acc
    except Exception as e:  # noqa: BLE001
        warnings.append(f"laplacian jnp.gradient failed: {e}")
        return None


def _scalar_laplacian(
    K: jnp.ndarray,
    cfg: PathBGridConfig,
    m: Dict[str, Any],
    warnings: List[str],
) -> Optional[jnp.ndarray]:
    x, y, z = _get_coord(m, cfg, "x"), _get_coord(m, cfg, "y"), _get_coord(m, cfg, "z")
    steady = cfg.steady
    dim = _infer_spatial_dim(K, steady, cfg.spatial_dimension)
    if not steady:
        if K.ndim < 2:
            warnings.append("unsteady laplacian: need leading time axis")
            return None
        t = _get_coord(m, cfg, "t")
        if t is None or t.shape[0] != K.shape[0]:
            warnings.append("unsteady laplacian: t must match leading dim")
            return None

        first = _scalar_laplacian_steady(K[0], cfg, x, y, z, dim, warnings)
        if first is None:
            return None
        return jax.vmap(lambda ks: _scalar_laplacian_steady(ks, cfg, x, y, z, dim, []))(K)
    return _scalar_laplacian_steady(K, cfg, x, y, z, dim, warnings)


def _scalar_gradient(
    K: jnp.ndarray,
    cfg: PathBGridConfig,
    m: Dict[str, Any],
    warnings: List[str],
) -> Optional[jnp.ndarray]:
    """Return (..., spatial_dim) with axes ordered x, y, z (up to ``dim``)."""
    dim = _infer_spatial_dim(K, cfg.steady, cfg.spatial_dimension)
    comps = []
    for ax_name, _ in [("x", 0), ("y", 1), ("z", 2)][:dim]:
        gi = _fill_spatial_derivative(K, ax_name, cfg, m, warnings)
        if gi is None:
            return None
        comps.append(gi)
    return jnp.stack(comps, axis=-1)


def _vector_laplacian(
    u: jnp.ndarray,
    cfg: PathBGridConfig,
    m: Dict[str, Any],
    warnings: List[str],
) -> Optional[jnp.ndarray]:
    """u shape (..., d); Laplacian per component, same shape."""
    if u.shape[-1] < 1:
        return None
    parts = []
    for i in range(int(u.shape[-1])):
        Ki = u[..., i]
        li = _scalar_laplacian(jnp.asarray(Ki), cfg, m, warnings)
        if li is None:
            return None
        parts.append(li)
    return jnp.stack(parts, axis=-1)


def _velocity_jacobian(
    u: jnp.ndarray,
    cfg: PathBGridConfig,
    m: Dict[str, Any],
    warnings: List[str],
) -> Optional[jnp.ndarray]:
    """
    u (..., d). Return (..., d, spatial_dim) with [..., i, j] = d u_i / d x_j.
    """
    spatial_dim = _infer_spatial_dim(
        jnp.asarray(u[..., 0]), cfg.steady, cfg.spatial_dimension
    )
    d_vel = int(u.shape[-1])
    rows = []
    for i in range(d_vel):
        gi = _scalar_gradient(u[..., i], cfg, m, warnings)
        if gi is None:
            return None
        if gi.shape[-1] != spatial_dim:
            warnings.append("jacobian: spatial_dim mismatch")
            return None
        rows.append(gi)
    return jnp.stack(rows, axis=-2)


def _second_time_derivative(
    K: jnp.ndarray,
    cfg: PathBGridConfig,
    m: Dict[str, Any],
    warnings: List[str],
) -> Optional[jnp.ndarray]:
    dt1 = _fill_temporal_derivative(K, cfg, m, warnings)
    if dt1 is None:
        return None
    return _fill_temporal_derivative(dt1, cfg, m, warnings)


def _law_name_from_spec(spec: Dict[str, Any]) -> Optional[str]:
    n = spec.get("name")
    if n:
        return str(n)
    fn = spec.get("fn")
    if fn is not None:
        return getattr(fn, "__name__", None)
    return None


def _resolve_source_state_key(
    recipe: LawFDArgRecipe,
    law_arg_name: str,
    target_state_key: str,
    state_map: Dict[str, str],
) -> Optional[str]:
    if recipe.source_arg is not None:
        sk = state_map.get(recipe.source_arg)
        if sk is not None:
            return sk
        # Common Path B convention: primitive uses the same name as the law argument (e.g. ``u``).
        return recipe.source_arg
    return _infer_primitive_key(target_state_key, law_arg_name)


def fill_law_fd_from_primitives(
    state_pred: Dict[str, Any],
    laws_spec: Sequence[Dict[str, Any]],
    *,
    constants: Optional[Dict[str, Any]] = None,
    grid: Optional[PathBGridConfig] = None,
    copy: bool = True,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Fill missing **registered** ``Laws.*`` argument keys using FD on structured grids.

    Returns ``(new_state, warnings)``. Skips targets that are already non-``None``.
    Unknown law names or unregistered arguments are ignored (no error).
    """
    cfg = grid or PathBGridConfig()
    c = dict(constants or {})
    state: Dict[str, Any] = dict(state_pred) if copy else state_pred
    m = _merged(state, c)
    warnings: List[str] = []

    def try_fill_one(
        law_name: str,
        arg_name: str,
        target_sk: str,
        sm: Dict[str, str],
    ) -> bool:
        if state.get(target_sk) is not None:
            return False
        recipes = LAW_FD_RECIPES.get(law_name) or {}
        recipe = recipes.get(arg_name)
        if recipe is None:
            return False
        src_sk = _resolve_source_state_key(recipe, arg_name, target_sk, sm)
        if not src_sk:
            warnings.append(
                f"law_fd {law_name}.{arg_name}: could not resolve primitive key for {target_sk!r}"
            )
            return False
        raw = m.get(src_sk)
        if raw is None:
            warnings.append(
                f"law_fd {law_name}.{arg_name}: skip {target_sk!r}: primitive {src_sk!r} missing"
            )
            return False
        K = jnp.asarray(raw)
        arr: Optional[jnp.ndarray] = None
        try:
            if recipe.kind == "laplacian":
                arr = _scalar_laplacian(K, cfg, m, warnings)
            elif recipe.kind == "vector_laplacian":
                arr = _vector_laplacian(K, cfg, m, warnings)
            elif recipe.kind == "grad_scalar":
                arr = _scalar_gradient(K, cfg, m, warnings)
            elif recipe.kind == "jacobian":
                arr = _velocity_jacobian(K, cfg, m, warnings)
            elif recipe.kind == "dt":
                arr = _fill_temporal_derivative(K, cfg, m, warnings)
            elif recipe.kind == "dtt":
                arr = _second_time_derivative(K, cfg, m, warnings)
            else:
                warnings.append(f"law_fd: unknown recipe kind {recipe.kind!r}")
                return False
        except Exception as e:  # noqa: BLE001
            warnings.append(f"law_fd {law_name}.{arg_name}: {e}")
            arr = None
        if arr is not None:
            state[target_sk] = arr
            m[target_sk] = arr
            return True
        return False

    max_passes = max(3, len(laws_spec) * 4)
    for _ in range(max_passes):
        progressed = False
        for spec in laws_spec:
            law_name = _law_name_from_spec(spec)
            if not law_name:
                continue
            sm = spec.get("state_map") or {}
            if not isinstance(sm, dict):
                continue
            for arg_name, target_sk in sm.items():
                if try_fill_one(law_name, str(arg_name), str(target_sk), sm):
                    progressed = True
        if not progressed:
            break

    return state, warnings


def list_law_fd_supported_laws() -> List[str]:
    """Law names that have at least one registered FD recipe (may be empty dict = none)."""
    return sorted(
        n for n, r in LAW_FD_RECIPES.items() if r
    )


__all__ = [
    "LAW_FD_RECIPES",
    "LawFDArgRecipe",
    "fill_law_fd_from_primitives",
    "list_law_fd_supported_laws",
]