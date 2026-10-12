"""
Per-term residual breakdown for governing laws (diagnostics only; does not affect tiers).

For each configured law, reports RMS and max of each PDE term, of the residual, the
ratio of residual RMS to the largest term RMS, and the normalizing ``scale_k`` from
the law's scale recipe.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import jax.numpy as jnp

from moju.monitor.law_implied_diagnostics import _val
from moju.monitor.law_scale_recipes import (
    _advection_mag,
    _rms_mag,
    _safe_inv_re,
    characteristic_law_scale_k,
)

_EPS = 1e-30


def _max_mag(arr: Any) -> float:
    if arr is None:
        return float("nan")
    a = jnp.asarray(arr)
    if a.size == 0:
        return float("nan")
    if a.ndim >= 1 and a.shape[-1] > 1 and jnp.issubdtype(a.dtype, jnp.floating):
        mag = jnp.sqrt(jnp.sum(a**2, axis=-1) + _EPS)
        return float(jnp.max(mag))
    return float(jnp.max(jnp.abs(a)))


def _stats(arr: Any) -> Dict[str, float]:
    return {"rms": float(_rms_mag(arr)), "max": _max_mag(arr)}


def _apply_mask(arr: Any, mask: Optional[Any]) -> Any:
    if arr is None or mask is None:
        return arr
    from moju.monitor.eval_mask import apply_mask_to_residual

    try:
        return apply_mask_to_residual(arr, mask)
    except ValueError:
        return arr


TermExtractor = Callable[
    [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], Dict[str, Any]
]


def _terms_momentum_ns(merged, constants, sm) -> Dict[str, Any]:
    u_t = _val(merged, constants, sm, "u_t")
    u = _val(merged, constants, sm, "u")
    u_grad = _val(merged, constants, sm, "u_grad")
    p_grad = _val(merged, constants, sm, "p_grad")
    u_lap = _val(merged, constants, sm, "u_laplacian")
    re = _val(merged, constants, sm, "re")
    adv = _advection_mag(u, u_grad) if u is not None and u_grad is not None else None
    # Match Laws.momentum_navier_stokes: advection block is u_t + (u·∇)u
    time_adv = None
    if u_t is not None and adv is not None:
        time_adv = jnp.asarray(u_t) + jnp.asarray(adv)
    elif u_t is not None:
        time_adv = u_t
    elif adv is not None:
        time_adv = adv
    visc = None
    if u_lap is not None and re is not None:
        visc = _safe_inv_re(re) * jnp.asarray(u_lap)
    return {
        "time_advection": time_adv,
        "pressure_gradient": p_grad,
        "viscous": visc,
    }


def _terms_momentum_newtonian(merged, constants, sm) -> Dict[str, Any]:
    u_t = _val(merged, constants, sm, "u_t")
    u = _val(merged, constants, sm, "u")
    u_grad = _val(merged, constants, sm, "u_grad")
    p_grad = _val(merged, constants, sm, "p_grad")
    u_lap = _val(merged, constants, sm, "u_laplacian")
    nu = _val(merged, constants, sm, "nu_eff")
    adv = _advection_mag(u, u_grad) if u is not None and u_grad is not None else None
    time_adv = None
    if u_t is not None and adv is not None:
        time_adv = jnp.asarray(u_t) + jnp.asarray(adv)
    elif u_t is not None:
        time_adv = u_t
    elif adv is not None:
        time_adv = adv
    visc = None
    if u_lap is not None and nu is not None:
        visc = jnp.asarray(nu) * jnp.asarray(u_lap)
    return {
        "time_advection": time_adv,
        "pressure_gradient": p_grad,
        "viscous": visc,
    }


def _terms_stokes(merged, constants, sm) -> Dict[str, Any]:
    p_grad = _val(merged, constants, sm, "p_grad")
    u_lap = _val(merged, constants, sm, "u_laplacian")
    re = _val(merged, constants, sm, "re")
    visc = None
    if u_lap is not None and re is not None:
        visc = _safe_inv_re(re) * jnp.asarray(u_lap)
    return {"pressure_gradient": p_grad, "viscous": visc}


def _terms_euler(merged, constants, sm) -> Dict[str, Any]:
    u_t = _val(merged, constants, sm, "u_t")
    u = _val(merged, constants, sm, "u")
    u_grad = _val(merged, constants, sm, "u_grad")
    p_grad = _val(merged, constants, sm, "p_grad")
    adv = _advection_mag(u, u_grad) if u is not None and u_grad is not None else None
    time_adv = None
    if u_t is not None and adv is not None:
        time_adv = jnp.asarray(u_t) + jnp.asarray(adv)
    elif u_t is not None:
        time_adv = u_t
    elif adv is not None:
        time_adv = adv
    return {"time_advection": time_adv, "pressure_gradient": p_grad}


def _terms_fourier(merged, constants, sm) -> Dict[str, Any]:
    T_t = _val(merged, constants, sm, "T_t")
    T_lap = _val(merged, constants, sm, "T_laplacian")
    fo = _val(merged, constants, sm, "fo")
    t = _val(merged, constants, sm, "t")
    L = _val(merged, constants, sm, "L")
    alpha_term = None
    if fo is not None and t is not None and L is not None and T_lap is not None:
        alpha = jnp.asarray(fo) * (jnp.asarray(L) ** 2) / jnp.asarray(t)
        alpha_term = alpha * jnp.asarray(T_lap)
    return {"time_derivative": T_t, "diffusion": alpha_term}


def _terms_fick(merged, constants, sm) -> Dict[str, Any]:
    phi_t = _val(merged, constants, sm, "phi_t")
    phi_lap = _val(merged, constants, sm, "phi_laplacian")
    fo_mass = _val(merged, constants, sm, "fo_mass")
    t = _val(merged, constants, sm, "t")
    L = _val(merged, constants, sm, "L")
    diff = None
    if fo_mass is not None and t is not None and L is not None and phi_lap is not None:
        D = jnp.asarray(fo_mass) * (jnp.asarray(L) ** 2) / jnp.asarray(t)
        diff = D * jnp.asarray(phi_lap)
    return {"time_derivative": phi_t, "diffusion": diff}


def _terms_burgers(merged, constants, sm) -> Dict[str, Any]:
    u_t = _val(merged, constants, sm, "u_t")
    u = _val(merged, constants, sm, "u")
    u_grad = _val(merged, constants, sm, "u_grad")
    u_lap = _val(merged, constants, sm, "u_laplacian")
    re = _val(merged, constants, sm, "re")
    U = _val(merged, constants, sm, "U")
    L = _val(merged, constants, sm, "L")
    adv = _advection_mag(u, u_grad) if u is not None and u_grad is not None else None
    time_adv = None
    if u_t is not None and adv is not None:
        time_adv = jnp.asarray(u_t) + jnp.asarray(adv)
    elif u_t is not None:
        time_adv = u_t
    elif adv is not None:
        time_adv = adv
    visc = None
    if u_lap is not None and re is not None and U is not None and L is not None:
        nu = (jnp.asarray(U) * jnp.asarray(L)) / (jnp.asarray(re) + 1e-30)
        visc = nu * jnp.asarray(u_lap)
    return {"time_advection": time_adv, "viscous": visc}


def _terms_mass_incompressible(merged, constants, sm) -> Dict[str, Any]:
    ug = _val(merged, constants, sm, "u_grad")
    div = None
    if ug is not None:
        div = jnp.trace(jnp.asarray(ug), axis1=-2, axis2=-1)
    return {"divergence": div}


def _terms_mass_compressible(merged, constants, sm) -> Dict[str, Any]:
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
    return {"time_derivative": rho_t, "advection": adv, "divergence": div}


def _terms_laplace(merged, constants, sm) -> Dict[str, Any]:
    return {"laplacian": _val(merged, constants, sm, "phi_laplacian")}


def _terms_poisson(merged, constants, sm) -> Dict[str, Any]:
    return {
        "laplacian": _val(merged, constants, sm, "phi_laplacian"),
        "source": _val(merged, constants, sm, "f"),
    }


def _terms_wave(merged, constants, sm) -> Dict[str, Any]:
    phi_tt = _val(merged, constants, sm, "phi_tt")
    phi_lap = _val(merged, constants, sm, "phi_laplacian")
    c = _val(merged, constants, sm, "c")
    wave = None
    if c is not None and phi_lap is not None:
        wave = (jnp.asarray(c) ** 2) * jnp.asarray(phi_lap)
    return {"second_time_derivative": phi_tt, "wave": wave}


_TERM_EXTRACTORS: Dict[str, TermExtractor] = {
    "momentum_navier_stokes": _terms_momentum_ns,
    "momentum_incompressible_newtonian_laplacian": _terms_momentum_newtonian,
    "momentum_compressible_newtonian_laplacian": _terms_momentum_newtonian,
    "stokes_flow": _terms_stokes,
    "euler_momentum": _terms_euler,
    "fourier_conduction": _terms_fourier,
    "fick_diffusion": _terms_fick,
    "burgers_equation": _terms_burgers,
    "mass_incompressible": _terms_mass_incompressible,
    "mass_compressible": _terms_mass_compressible,
    "laplace_equation": _terms_laplace,
    "poisson_equation": _terms_poisson,
    "wave_equation": _terms_wave,
}


def law_term_arrays(
    law_name: str,
    merged: Mapping[str, Any],
    constants: Mapping[str, Any],
    law_spec: Mapping[str, Any],
) -> Dict[str, Any]:
    """Named term arrays for a law (may be empty if no extractor is registered)."""
    sm = law_spec.get("state_map") or {}
    fn = _TERM_EXTRACTORS.get(str(law_name))
    if fn is None:
        return {}
    return {k: v for k, v in fn(merged, constants, sm).items() if v is not None}


def breakdown_for_law(
    law_name: str,
    residual: Any,
    merged: Mapping[str, Any],
    constants: Mapping[str, Any],
    law_spec: Mapping[str, Any],
    *,
    scale_k: Optional[float] = None,
    scale_source: Optional[str] = None,
    mask: Optional[Any] = None,
    nondim_scales: Any = None,
) -> Dict[str, Any]:
    """
    Build a JSON-serialisable term breakdown for one law.

    Does not modify residual scoring. ``mask`` (boolean, True = keep) only affects
    the statistics reported here when provided.
    """
    terms_raw = law_term_arrays(law_name, merged, constants, law_spec)
    if mask is not None:
        terms = {k: _apply_mask(v, mask) for k, v in terms_raw.items()}
        resid = _apply_mask(residual, mask)
    else:
        terms = terms_raw
        resid = residual

    term_stats = {name: _stats(arr) for name, arr in terms.items()}
    resid_stats = _stats(resid)
    term_rms = [s["rms"] for s in term_stats.values() if math.isfinite(s["rms"]) and s["rms"] > 0]
    largest = max(term_rms) if term_rms else float("nan")
    rel = (
        float(resid_stats["rms"] / largest)
        if math.isfinite(largest) and largest > 0 and math.isfinite(resid_stats["rms"])
        else float("nan")
    )

    if scale_k is None:
        sk, src = characteristic_law_scale_k(
            str(law_name),
            dict(merged),
            dict(constants),
            dict(law_spec),
            nondim_scales=nondim_scales,
        )
        scale_k = float(sk)
        scale_source = src
    else:
        scale_k = float(scale_k)
        scale_source = scale_source or "provided"

    return {
        "terms": term_stats,
        "residual": resid_stats,
        "residual_over_largest_term": rel,
        "scale_k": scale_k,
        "scale_source": scale_source,
    }


def build_term_breakdown(
    laws_spec: Sequence[Mapping[str, Any]],
    residuals_laws: Mapping[str, Any],
    merged: Mapping[str, Any],
    constants: Mapping[str, Any],
    *,
    scale_per_key: Optional[Mapping[str, float]] = None,
    scale_source_per_key: Optional[Mapping[str, str]] = None,
    mask: Optional[Any] = None,
    nondim_scales: Any = None,
) -> Dict[str, Any]:
    """Map ``law_name → breakdown`` for every law that produced a residual."""
    out: Dict[str, Any] = {}
    scale_per_key = scale_per_key or {}
    scale_source_per_key = scale_source_per_key or {}
    for spec in laws_spec:
        name = str(spec.get("name") or "")
        if name not in residuals_laws:
            continue
        flat_key = f"laws/{name}"
        out[name] = breakdown_for_law(
            name,
            residuals_laws[name],
            merged,
            constants,
            spec,
            scale_k=scale_per_key.get(flat_key),
            scale_source=scale_source_per_key.get(flat_key),
            mask=mask,
            nondim_scales=nondim_scales,
        )
    return out
