"""
Term-balance auto ``scale_k`` recipes for governing **laws/** residuals.

Each recipe estimates a characteristic magnitude from merged state (not from the residual,
to avoid circular scaling). Fallback chain ends at :data:`DEFAULT_NONDIM_R_NORM_SCALE_K`.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import jax.numpy as jnp

from moju.monitor.law_implied_diagnostics import _val, all_law_names
from moju.piratio.nondim import NondimScales

# Match :data:`moju.monitor.auditor.DEFAULT_NONDIM_R_NORM_SCALE_K` (avoid circular import).
_DEFAULT_LAW_SCALE_K = 1e-2

_SCALE_EPS = 1e-12

ScaleSource = str  # "auto" | "user_recipe" | "auto_fallback" | "fixed"


def _as_host_float(value: Any) -> Any:
    """Python float outside a trace; the traced value itself under ``jax.jit``."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _rms_mag_array(arr: Any) -> jnp.ndarray:
    """
    Traceable RMS used by built-in recipes.

    A floating array whose last axis is longer than 1 uses ``sqrt(mean(sum of squares))``.
    Every other array uses ``sqrt(mean(square))``. That split is what published scores use.
    """
    if arr is None:
        return jnp.asarray(jnp.nan)
    a = jnp.asarray(arr)
    if a.size == 0:
        return jnp.asarray(jnp.nan)
    if a.ndim >= 1 and a.shape[-1] > 1 and jnp.issubdtype(a.dtype, jnp.floating):
        sq = jnp.sum(a**2, axis=-1)
        return jnp.sqrt(jnp.mean(sq) + _SCALE_EPS)
    return jnp.sqrt(jnp.mean(a**2) + _SCALE_EPS)


def _rms_mag(arr: Any) -> float:
    """Scalar RMS; vector last-axis uses sqrt(mean(sum sq))."""
    return _as_host_float(_rms_mag_array(arr))


def _term_max_rms(*terms: Any) -> float:
    """Largest finite positive RMS. ``None`` terms are skipped. NaN if none qualify."""
    vals = [_rms_mag_array(t) for t in terms if t is not None]
    if not vals:
        return _as_host_float(jnp.nan)
    stacked = jnp.stack([jnp.asarray(v) for v in vals])
    positive = jnp.where(jnp.isfinite(stacked) & (stacked > 0), stacked, -jnp.inf)
    best = jnp.max(positive)
    return _as_host_float(jnp.where(jnp.isfinite(best), best, jnp.nan))


def _floor_scale(scale: float) -> float:
    if not math.isfinite(scale) or scale <= 0:
        return float(_DEFAULT_LAW_SCALE_K)
    return max(float(scale), float(_DEFAULT_LAW_SCALE_K))


def _advection_mag(u: Any, u_grad: Any) -> Any:
    u_a = jnp.asarray(u)
    g = jnp.asarray(u_grad)
    return jnp.einsum("...ij,...j->...i", g, u_a)


def _safe_inv_re(re: Any) -> Any:
    r = jnp.asarray(re)
    return 1.0 / (r + 1e-30)


def _recipe_laplace(merged, constants, law_spec, nondim_scales):
    sm = law_spec.get("state_map") or {}
    lap = _val(merged, constants, sm, "phi_laplacian")
    return _term_max_rms(lap)


def _recipe_laplace_beltrami(merged, constants, law_spec, nondim_scales):
    sm = law_spec.get("state_map") or {}
    lap = _val(merged, constants, sm, "phi_laplacian_g")
    return _term_max_rms(lap)


def _recipe_mass_incompressible(merged, constants, law_spec, nondim_scales):
    sm = law_spec.get("state_map") or {}
    ug = _val(merged, constants, sm, "u_grad")
    return _term_max_rms(
        jnp.trace(jnp.asarray(ug), axis1=-2, axis2=-1) if ug is not None else None
    )


def _recipe_mass_compressible(merged, constants, law_spec, nondim_scales):
    sm = law_spec.get("state_map") or {}
    rho = _val(merged, constants, sm, "rho")
    rho_t = _val(merged, constants, sm, "rho_t")
    u = _val(merged, constants, sm, "u")
    rho_grad = _val(merged, constants, sm, "rho_grad")
    u_grad = _val(merged, constants, sm, "u_grad")
    adv = None
    if u is not None and rho_grad is not None:
        adv = jnp.sum(jnp.asarray(u) * jnp.asarray(rho_grad), axis=-1)
    div = None
    if rho is not None and u_grad is not None:
        div = jnp.asarray(rho) * jnp.trace(jnp.asarray(u_grad), axis1=-2, axis2=-1)
    return _term_max_rms(rho_t, adv, div)


def _recipe_momentum_ns(merged, constants, law_spec, nondim_scales):
    sm = law_spec.get("state_map") or {}
    u_t = _val(merged, constants, sm, "u_t")
    u = _val(merged, constants, sm, "u")
    u_grad = _val(merged, constants, sm, "u_grad")
    p_grad = _val(merged, constants, sm, "p_grad")
    u_lap = _val(merged, constants, sm, "u_laplacian")
    re = _val(merged, constants, sm, "re")
    adv = _advection_mag(u, u_grad) if u is not None and u_grad is not None else None
    visc = None
    if u_lap is not None and re is not None:
        visc = _safe_inv_re(re) * jnp.asarray(u_lap)
    return _term_max_rms(u_t, adv, p_grad, visc)


def _recipe_momentum_newtonian(merged, constants, law_spec, nondim_scales):
    sm = law_spec.get("state_map") or {}
    u_t = _val(merged, constants, sm, "u_t")
    u = _val(merged, constants, sm, "u")
    u_grad = _val(merged, constants, sm, "u_grad")
    p_grad = _val(merged, constants, sm, "p_grad")
    u_lap = _val(merged, constants, sm, "u_laplacian")
    nu = _val(merged, constants, sm, "nu_eff")
    adv = _advection_mag(u, u_grad) if u is not None and u_grad is not None else None
    visc = None
    if u_lap is not None and nu is not None:
        visc = jnp.asarray(nu) * jnp.asarray(u_lap)
    return _term_max_rms(u_t, adv, p_grad, visc)


def _recipe_momentum_comp_newtonian(merged, constants, law_spec, nondim_scales):
    sm = law_spec.get("state_map") or {}
    rho = _val(merged, constants, sm, "rho")
    u_t = _val(merged, constants, sm, "u_t")
    u = _val(merged, constants, sm, "u")
    u_grad = _val(merged, constants, sm, "u_grad")
    p_grad = _val(merged, constants, sm, "p_grad")
    u_lap = _val(merged, constants, sm, "u_laplacian")
    nu = _val(merged, constants, sm, "nu_eff")
    adv = _advection_mag(u, u_grad) if u is not None and u_grad is not None else None
    mom = None
    if rho is not None and adv is not None:
        mom = (
            jnp.asarray(rho) * (jnp.asarray(u_t) + adv)
            if u_t is not None
            else jnp.asarray(rho) * adv
        )
    visc = None
    if rho is not None and u_lap is not None and nu is not None:
        visc = jnp.asarray(rho) * jnp.asarray(nu) * jnp.asarray(u_lap)
    return _term_max_rms(mom, p_grad, visc)


def _recipe_stokes(merged, constants, law_spec, nondim_scales):
    sm = law_spec.get("state_map") or {}
    p_grad = _val(merged, constants, sm, "p_grad")
    u_lap = _val(merged, constants, sm, "u_laplacian")
    re = _val(merged, constants, sm, "re")
    visc = None
    if u_lap is not None and re is not None:
        visc = _safe_inv_re(re) * jnp.asarray(u_lap)
    return _term_max_rms(p_grad, visc)


def _recipe_euler(merged, constants, law_spec, nondim_scales):
    sm = law_spec.get("state_map") or {}
    u_t = _val(merged, constants, sm, "u_t")
    u = _val(merged, constants, sm, "u")
    u_grad = _val(merged, constants, sm, "u_grad")
    p_grad = _val(merged, constants, sm, "p_grad")
    adv = _advection_mag(u, u_grad) if u is not None and u_grad is not None else None
    return _term_max_rms(u_t, adv, p_grad)


def _recipe_fourier(merged, constants, law_spec, nondim_scales):
    sm = law_spec.get("state_map") or {}
    T_t = _val(merged, constants, sm, "T_t")
    T_lap = _val(merged, constants, sm, "T_laplacian")
    fo = _val(merged, constants, sm, "fo")
    t = _val(merged, constants, sm, "t")
    L = _val(merged, constants, sm, "L")
    alpha_term = None
    if fo is not None and t is not None and L is not None and T_lap is not None:
        alpha = jnp.asarray(fo) * (jnp.asarray(L) ** 2) / jnp.asarray(t)
        alpha_term = alpha * jnp.asarray(T_lap)
    return _term_max_rms(T_t, alpha_term)


def _recipe_fick(merged, constants, law_spec, nondim_scales):
    sm = law_spec.get("state_map") or {}
    phi_t = _val(merged, constants, sm, "phi_t")
    phi_lap = _val(merged, constants, sm, "phi_laplacian")
    fo_mass = _val(merged, constants, sm, "fo_mass")
    t = _val(merged, constants, sm, "t")
    L = _val(merged, constants, sm, "L")
    diff = None
    if fo_mass is not None and t is not None and L is not None and phi_lap is not None:
        D = jnp.asarray(fo_mass) * (jnp.asarray(L) ** 2) / jnp.asarray(t)
        diff = D * jnp.asarray(phi_lap)
    return _term_max_rms(phi_t, diff)


def _recipe_adv_diff(merged, constants, law_spec, nondim_scales):
    sm = law_spec.get("state_map") or {}
    phi_t = _val(merged, constants, sm, "phi_t")
    u = _val(merged, constants, sm, "u")
    phi_grad = _val(merged, constants, sm, "phi_grad")
    phi_lap = _val(merged, constants, sm, "phi_laplacian")
    pe = _val(merged, constants, sm, "pe")
    adv = None
    if u is not None and phi_grad is not None:
        adv = jnp.sum(jnp.asarray(phi_grad) * jnp.asarray(u), axis=-1)
    diff = None
    if phi_lap is not None and pe is not None:
        diff = _safe_inv_re(pe) * jnp.asarray(phi_lap)
    return _term_max_rms(phi_t, adv, diff)


def _recipe_burgers(merged, constants, law_spec, nondim_scales):
    sm = law_spec.get("state_map") or {}
    u_t = _val(merged, constants, sm, "u_t")
    u = _val(merged, constants, sm, "u")
    u_grad = _val(merged, constants, sm, "u_grad")
    u_lap = _val(merged, constants, sm, "u_laplacian")
    re = _val(merged, constants, sm, "re")
    adv = _advection_mag(u, u_grad) if u is not None and u_grad is not None else None
    visc = None
    if u_lap is not None and re is not None:
        visc = _safe_inv_re(re) * jnp.asarray(u_lap)
    return _term_max_rms(u_t, adv, visc)


def _recipe_poisson(merged, constants, law_spec, nondim_scales):
    sm = law_spec.get("state_map") or {}
    lap = _val(merged, constants, sm, "phi_laplacian")
    source = _val(merged, constants, sm, "source")
    eps = _val(merged, constants, sm, "epsilon")
    src = None
    if source is not None and eps is not None:
        src = jnp.asarray(source) / jnp.asarray(eps)
    return _term_max_rms(lap, src)


def _recipe_wave(merged, constants, law_spec, nondim_scales):
    sm = law_spec.get("state_map") or {}
    phi_tt = _val(merged, constants, sm, "phi_tt")
    phi_lap = _val(merged, constants, sm, "phi_laplacian")
    st_wave = _val(merged, constants, sm, "st_wave")
    omega = _val(merged, constants, sm, "omega")
    L = _val(merged, constants, sm, "L")
    lap_term = None
    if (
        phi_lap is not None
        and st_wave is not None
        and omega is not None
        and L is not None
    ):
        coeff = (jnp.asarray(st_wave) * jnp.asarray(omega) / jnp.asarray(L)) ** 2
        lap_term = coeff * jnp.asarray(phi_lap)
    return _term_max_rms(phi_tt, lap_term)


def _recipe_helmholtz(merged, constants, law_spec, nondim_scales):
    sm = law_spec.get("state_map") or {}
    phi = _val(merged, constants, sm, "phi")
    phi_lap = _val(merged, constants, sm, "phi_laplacian")
    kL = _val(merged, constants, sm, "kL")
    L = _val(merged, constants, sm, "L")
    mass = None
    if phi is not None and kL is not None and L is not None:
        mass = (jnp.asarray(kL) / jnp.asarray(L)) ** 2 * jnp.asarray(phi)
    return _term_max_rms(phi, phi_lap, mass)


def _recipe_darcy(merged, constants, law_spec, nondim_scales):
    sm = law_spec.get("state_map") or {}
    u = _val(merged, constants, sm, "u")
    p_grad = _val(merged, constants, sm, "p_grad")
    return _term_max_rms(u, p_grad)


def _recipe_brinkman(merged, constants, law_spec, nondim_scales):
    sm = law_spec.get("state_map") or {}
    u = _val(merged, constants, sm, "u")
    u_lap = _val(merged, constants, sm, "u_laplacian")
    p_grad = _val(merged, constants, sm, "p_grad")
    re = _val(merged, constants, sm, "re")
    visc = None
    if u_lap is not None and re is not None:
        visc = _safe_inv_re(re) * jnp.asarray(u_lap)
    return _term_max_rms(u, p_grad, visc)


def _recipe_hookes(merged, constants, law_spec, nondim_scales):
    sm = law_spec.get("state_map") or {}
    stress = _val(merged, constants, sm, "stress")
    strain = _val(merged, constants, sm, "strain")
    C = _val(merged, constants, sm, "stiffness_tensor")
    Ce = None
    if C is not None and strain is not None:
        Ce = jnp.einsum("...ij,...j->...i", jnp.asarray(C), jnp.asarray(strain))
    return _term_max_rms(stress, Ce, strain)


def _recipe_viscous_dissipation(merged, constants, law_spec, nondim_scales):
    sm = law_spec.get("state_map") or {}
    u_grad = _val(merged, constants, sm, "u_grad")
    re = _val(merged, constants, sm, "re")
    ec = _val(merged, constants, sm, "ec")
    U = _val(merged, constants, sm, "U")
    L = _val(merged, constants, sm, "L")
    src = None
    if (
        u_grad is not None
        and re is not None
        and ec is not None
        and U is not None
        and L is not None
    ):
        strain = 0.5 * (jnp.asarray(u_grad) + jnp.swapaxes(jnp.asarray(u_grad), -2, -1))
        strain_star = strain * (jnp.asarray(L) / jnp.asarray(U))
        src = (
            (jnp.asarray(ec) / jnp.asarray(re))
            * 2.0
            * jnp.sum(strain_star**2, axis=(-2, -1))
        )
    return _term_max_rms(u_grad, src)


def _recipe_faraday(merged, constants, law_spec, nondim_scales):
    sm = law_spec.get("state_map") or {}
    E_curl = _val(merged, constants, sm, "E_curl")
    B_t = _val(merged, constants, sm, "B_t")
    return _term_max_rms(E_curl, B_t)


def _recipe_schrodinger(merged, constants, law_spec, nondim_scales):
    sm = law_spec.get("state_map") or {}
    psi_lap = _val(merged, constants, sm, "psi_laplacian")
    V = _val(merged, constants, sm, "V")
    E = _val(merged, constants, sm, "E")
    psi = _val(merged, constants, sm, "psi")
    sch = _val(merged, constants, sm, "sch_kin_l2")
    pot = None
    if V is not None and E is not None and psi is not None and sch is not None:
        pot = jnp.asarray(sch) * (jnp.asarray(V) - jnp.asarray(E)) * jnp.asarray(psi)
    return _term_max_rms(psi_lap, pot)


LAW_SCALE_RECIPES: Dict[str, Callable[..., float]] = {
    "mass_incompressible": _recipe_mass_incompressible,
    "mass_compressible": _recipe_mass_compressible,
    "momentum_navier_stokes": _recipe_momentum_ns,
    "momentum_incompressible_newtonian_laplacian": _recipe_momentum_newtonian,
    "momentum_compressible_newtonian_laplacian": _recipe_momentum_comp_newtonian,
    "stokes_flow": _recipe_stokes,
    "euler_momentum": _recipe_euler,
    "fourier_conduction": _recipe_fourier,
    "fick_diffusion": _recipe_fick,
    "advection_diffusion": _recipe_adv_diff,
    "burgers_equation": _recipe_burgers,
    "laplace_equation": _recipe_laplace,
    "laplace_beltrami": _recipe_laplace_beltrami,
    "poisson_equation": _recipe_poisson,
    "wave_equation": _recipe_wave,
    "helmholtz_equation": _recipe_helmholtz,
    "darcy_flow": _recipe_darcy,
    "brinkman_extension": _recipe_brinkman,
    "hookes_law_residual": _recipe_hookes,
    "viscous_dissipation": _recipe_viscous_dissipation,
    "faraday_law": _recipe_faraday,
    "schrodinger_steady": _recipe_schrodinger,
}


def _generic_state_map_rms(
    merged: Mapping[str, Any],
    constants: Mapping[str, Any],
    law_spec: Mapping[str, Any],
) -> float:
    sm = law_spec.get("state_map") or {}
    parts = []
    for sk in sm.values():
        v = merged.get(sk)
        if v is None:
            v = constants.get(sk)
        if v is not None:
            parts.append(_rms_mag(v))
    finite = [p for p in parts if math.isfinite(p) and p > 0]
    if not finite:
        return float("nan")
    return max(finite)


LawScaleRecipe = Callable[
    [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Optional[NondimScales]],
    float,
]

_USER_LAW_SCALE_RECIPES: Dict[str, LawScaleRecipe] = {}


def register_law_scale_recipe(
    law_name: str, fn: LawScaleRecipe, *, overwrite: bool = False
) -> None:
    """
    Register a term-balance recipe ``fn(merged, constants, law_spec, nondim_scales) -> float`` for
    ``law_name`` (custom laws match on the spec ``name``). User recipes take precedence over built-in
    recipes; a law spec's own ``"scale_recipe"`` key takes precedence over both.

    Build recipes from :func:`law_arg_value` and :func:`term_max_rms`, e.g.
    ``term_max_rms(m * x_tt, c * x_t, k * x)`` for ``m x_tt + c x_t + k x = 0``.
    """
    if not callable(fn):
        raise TypeError("scale recipe must be callable")
    if law_name in _USER_LAW_SCALE_RECIPES and not overwrite:
        raise ValueError(
            f"scale recipe for {law_name!r} already registered; pass overwrite=True"
        )
    _USER_LAW_SCALE_RECIPES[str(law_name)] = fn


def unregister_law_scale_recipe(law_name: str) -> None:
    _USER_LAW_SCALE_RECIPES.pop(str(law_name), None)


def list_user_law_scale_recipes() -> Tuple[str, ...]:
    return tuple(sorted(_USER_LAW_SCALE_RECIPES))


def _term_rms_array(arr: Any, *, vector: bool = False) -> jnp.ndarray:
    """Traceable RMS. NaN if ``arr`` is missing or empty."""
    if arr is None:
        return jnp.asarray(jnp.nan)
    a = jnp.asarray(arr)
    if a.size == 0:
        return jnp.asarray(jnp.nan)
    if vector and a.ndim >= 1:
        return jnp.sqrt(jnp.mean(jnp.sum(a**2, axis=-1)) + _SCALE_EPS)
    return jnp.sqrt(jnp.mean(a**2) + _SCALE_EPS)


def term_rms(arr: Any, *, vector: bool = False) -> float:
    """
    RMS magnitude of one term; NaN if ``arr`` is ``None`` or empty.

    With ``vector=True`` the last axis holds vector components and the result is
    ``sqrt(mean(sum of squares over the last axis))``.
    """
    return _as_host_float(_term_rms_array(arr, vector=vector))


def term_max_rms(*terms: Any, vector: bool = False) -> float:
    """Largest finite positive :func:`term_rms` among ``terms`` (``None`` skipped); NaN if none."""
    vals = [_term_rms_array(t, vector=vector) for t in terms if t is not None]
    if not vals:
        return _as_host_float(jnp.nan)
    stacked = jnp.stack([jnp.asarray(v) for v in vals])
    positive = jnp.where(jnp.isfinite(stacked) & (stacked > 0), stacked, -jnp.inf)
    best = jnp.max(positive)
    return _as_host_float(jnp.where(jnp.isfinite(best), best, jnp.nan))


def law_arg_value(
    merged: Mapping[str, Any],
    constants: Mapping[str, Any],
    law_spec: Mapping[str, Any],
    arg: str,
) -> Any:
    """Value bound to law argument ``arg`` through the spec ``state_map`` (state, then constants)."""
    return _val(merged, constants, law_spec.get("state_map") or {}, arg)


def _resolve_scale_recipe(
    law_name: str, law_spec: Mapping[str, Any]
) -> Tuple[Optional[LawScaleRecipe], bool]:
    spec_recipe = law_spec.get("scale_recipe")
    if spec_recipe is not None:
        return spec_recipe, True
    from moju.registry import law_record_or_none

    record = law_record_or_none(law_name)
    if record is not None and record.scale_recipe is not None:
        return record.scale_recipe, True
    if law_name in _USER_LAW_SCALE_RECIPES:
        return _USER_LAW_SCALE_RECIPES[law_name], True
    return LAW_SCALE_RECIPES.get(law_name), False


def characteristic_law_scale_k(
    law_name: str,
    *,
    merged: Mapping[str, Any],
    constants: Mapping[str, Any],
    law_spec: Mapping[str, Any],
    nondim_scales: Optional[NondimScales] = None,
) -> Tuple[float, ScaleSource]:
    """
    Return ``(scale_k, scale_source)`` for a governing law.

    Fallback: spec ``scale_recipe`` → law-registry recipe → registered user recipe → built-in recipe → generic
    state_map RMS → ``DEFAULT_NONDIM_R_NORM_SCALE_K``.
    """
    recipe, is_user = _resolve_scale_recipe(law_name, law_spec)
    scale = float("nan")
    recipe_ok = False
    if recipe is not None:
        try:
            scale = float(recipe(merged, constants, law_spec, nondim_scales))
        except Exception as err:  # noqa: BLE001
            if is_user:
                import warnings

                warnings.warn(
                    f"scale recipe for law {law_name!r} raised {type(err).__name__}: {err}; "
                    "falling back to generic state_map RMS",
                    UserWarning,
                    stacklevel=2,
                )
            scale = float("nan")
        recipe_ok = math.isfinite(scale) and scale > 0
    if not recipe_ok:
        scale = _generic_state_map_rms(merged, constants, law_spec)
    if math.isfinite(scale) and scale > 0:
        floored = _floor_scale(scale)
        src: ScaleSource = "user_recipe" if recipe_ok and is_user else "auto"
        return floored, src
    return float(_DEFAULT_LAW_SCALE_K), "auto_fallback"


def list_laws_with_scale_recipes() -> Tuple[str, ...]:
    """Law names with explicit term-balance recipes."""
    return tuple(sorted(LAW_SCALE_RECIPES.keys()))


def law_scale_coverage_report() -> Dict[str, str]:
    """Map every public law (and user-registered recipe) to ``user_recipe``, ``recipe``, or ``generic_only``."""
    out: Dict[str, str] = {}
    for name in all_law_names():
        out[name] = "recipe" if name in LAW_SCALE_RECIPES else "generic_only"
    for name in _USER_LAW_SCALE_RECIPES:
        out[name] = "user_recipe"
    return out
