"""
Law-linked implied constitutive / scaling audits.

For selected :class:`Laws.*` entries, Moju can auto-append ``constitutive_audit`` /
``scaling_audit`` rows whose ``implied_fn`` recomputes a quantity by rearranging the
law using ``state_pred`` (and the law's ``state_map``). Residuals use the existing
``implied_delta`` closure: ``F(catalog args) - implied``.

We avoid pairing **both** a group and a constitutive closure that are algebraically
equivalent given fixed scales (e.g. Fourier: only ``thermal_diffusivity``, not a
separate ``fo`` implied row).

See README "Law-linked implied audits" and :func:`merge_law_implied_audit_specs`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _val(
    state: Dict[str, Any],
    constants: Dict[str, Any],
    law_sm: Dict[str, str],
    arg: str,
) -> Any:
    key = law_sm.get(arg)
    if key is None:
        return None
    v = state.get(key)
    if v is None:
        v = constants.get(key)
    return v


def _state_or_const(state: Dict[str, Any], constants: Dict[str, Any], key: str) -> Any:
    v = state.get(key)
    if v is None:
        v = constants.get(key)
    return v


def _safe_ratio(num: Any, den: Any, *, eps: float = 1e-20) -> jnp.ndarray:
    num = jnp.asarray(num, dtype=jnp.result_type(jnp.asarray(num), jnp.float32))
    den = jnp.asarray(den, dtype=jnp.result_type(jnp.asarray(den), jnp.float32))
    return jnp.where(jnp.abs(den) > eps, num / den, jnp.nan)


def _vec_norm_last(x: Any) -> jnp.ndarray:
    a = jnp.asarray(x)
    if a.ndim == 0:
        return jnp.abs(a)
    return jnp.sqrt(jnp.sum(a**2, axis=-1))


# ---------------------------------------------------------------------------
# Implied quantity makers: (law_state_map) -> implied_fn(state, constants)
# ---------------------------------------------------------------------------


def implied_alpha_fourier_conduction(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """α_implied = T_t / T_laplacian from :func:`Laws.fourier_conduction` rearrangement."""

    def implied_fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[jnp.ndarray]:
        T_t = _val(state, constants, law_sm, "T_t")
        T_lap = _val(state, constants, law_sm, "T_laplacian")
        if T_t is None or T_lap is None:
            return None
        return _safe_ratio(T_t, T_lap)

    return implied_fn


def implied_fo_mass_fick_diffusion(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """Fo_m implied = φ_t * t / (φ_laplacian * L²) from Fick's law."""

    def implied_fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[jnp.ndarray]:
        pt = _val(state, constants, law_sm, "phi_t")
        pl = _val(state, constants, law_sm, "phi_laplacian")
        t = _val(state, constants, law_sm, "t")
        L = _val(state, constants, law_sm, "L")
        if pt is None or pl is None or t is None or L is None:
            return None
        return _safe_ratio(pt * t, pl * (L**2))

    return implied_fn


def implied_D_fick_diffusion(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """D_implied = phi_t / phi_laplacian from :func:`Laws.fick_diffusion` rearrangement."""

    def implied_fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[jnp.ndarray]:
        pt = _val(state, constants, law_sm, "phi_t")
        pl = _val(state, constants, law_sm, "phi_laplacian")
        if pt is None or pl is None:
            return None
        return _safe_ratio(pt, pl)

    return implied_fn


def implied_st_wave_wave_equation(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """St_wave_implied = ω L / c with c² = φ_tt / φ_laplacian (wave equation rearrangement)."""

    def implied_fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[jnp.ndarray]:
        ptt = _val(state, constants, law_sm, "phi_tt")
        pl = _val(state, constants, law_sm, "phi_laplacian")
        omega = _val(state, constants, law_sm, "omega")
        L = _val(state, constants, law_sm, "L")
        if ptt is None or pl is None or omega is None or L is None:
            return None
        c_sq = _safe_ratio(ptt, pl)
        c = jnp.sqrt(jnp.maximum(c_sq, 0.0))
        return jnp.asarray(omega) * jnp.asarray(L) / (c + 1e-30)

    return implied_fn


def implied_c_wave_equation(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """c_implied = sqrt(phi_tt / phi_laplacian) from :func:`Laws.wave_equation`."""

    def implied_fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[jnp.ndarray]:
        ptt = _val(state, constants, law_sm, "phi_tt")
        pl = _val(state, constants, law_sm, "phi_laplacian")
        if ptt is None or pl is None:
            return None
        c_sq = _safe_ratio(ptt, pl)
        return jnp.sqrt(jnp.maximum(c_sq, 0.0))

    return implied_fn


def implied_pe_advection_diffusion(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """Pe_implied ≈ |∇²φ| / |φ_t + u·∇φ| from scalar transport rearrangement."""

    def implied_fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[jnp.ndarray]:
        phi_t = _val(state, constants, law_sm, "phi_t")
        u = _val(state, constants, law_sm, "u")
        g = _val(state, constants, law_sm, "phi_grad")
        lap = _val(state, constants, law_sm, "phi_laplacian")
        if phi_t is None or u is None or g is None or lap is None:
            return None
        u = jnp.asarray(u)
        g = jnp.asarray(g)
        if u.shape[-1] != g.shape[-1]:
            return None
        adv = jnp.sum(g * u, axis=-1)
        rhs = jnp.asarray(phi_t) + adv
        return _vec_norm_last(lap) / (_vec_norm_last(rhs) + 1e-30)

    return implied_fn


def implied_kappa_advection_diffusion(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """kappa_implied = U*L*(phi_t + u·grad(phi))/phi_laplacian (effective diffusivity)."""

    def implied_fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[jnp.ndarray]:
        phi_t = _val(state, constants, law_sm, "phi_t")
        u = _val(state, constants, law_sm, "u")
        g = _val(state, constants, law_sm, "phi_grad")
        lap = _val(state, constants, law_sm, "phi_laplacian")
        L = _state_or_const(state, constants, "L")
        if phi_t is None or u is None or g is None or lap is None or L is None:
            return None
        u = jnp.asarray(u)
        g = jnp.asarray(g)
        if u.shape[-1] != g.shape[-1]:
            return None
        adv = jnp.sum(g * u, axis=-1)
        rhs = jnp.asarray(phi_t) + adv
        U = _vec_norm_last(u)
        return jnp.asarray(U) * jnp.asarray(L) * _safe_ratio(rhs, lap)

    return implied_fn


def implied_re_momentum_navier_stokes(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """Re_implied ≈ ‖∇²u‖ / ‖u_t + u·∇u + ∇p‖ (diagnostic when the NS residual is small)."""

    def implied_fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[jnp.ndarray]:
        u_t = _val(state, constants, law_sm, "u_t")
        u = _val(state, constants, law_sm, "u")
        u_grad = _val(state, constants, law_sm, "u_grad")
        p_grad = _val(state, constants, law_sm, "p_grad")
        u_lap = _val(state, constants, law_sm, "u_laplacian")
        if u_t is None or u is None or u_grad is None or p_grad is None or u_lap is None:
            return None
        u = jnp.asarray(u)
        adv = jnp.einsum("...ij,...j->...i", jnp.asarray(u_grad), u)
        rhs = jnp.asarray(u_t) + adv + jnp.asarray(p_grad)
        return _vec_norm_last(u_lap) / (_vec_norm_last(rhs) + 1e-30)

    return implied_fn


def implied_mu_from_re_balance(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """
    μ_implied = ρ u_char L / Re_implied with u_char = ‖u‖, using the same Re balance as
    :func:`implied_re_momentum_navier_stokes`. Compare to ``mu`` in state (e.g. from Sutherland).
    """

    re_fn = implied_re_momentum_navier_stokes(law_sm)

    def implied_fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[jnp.ndarray]:
        re_i = re_fn(state, constants)
        if re_i is None:
            return None
        u = _val(state, constants, law_sm, "u")
        rho = state.get("rho")
        if rho is None:
            rho = constants.get("rho")
        L = state.get("L")
        if L is None:
            L = constants.get("L")
        if u is None or rho is None or L is None:
            return None
        u = jnp.asarray(u)
        u_char = _vec_norm_last(u)
        rho = jnp.asarray(rho)
        L = jnp.asarray(L)
        return rho * u_char * L / (jnp.asarray(re_i) + 1e-30)

    return implied_fn


def implied_mu_from_re_fn(
    law_sm: Dict[str, str],
    re_fn: Callable[[Dict[str, Any], Dict[str, Any]], Optional[jnp.ndarray]],
) -> Callable[..., Any]:
    """mu_implied = rho * |u| * L / Re_implied from a supplied implied-Re function."""

    def implied_fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[jnp.ndarray]:
        re_i = re_fn(state, constants)
        if re_i is None:
            return None
        u = _val(state, constants, law_sm, "u")
        if u is None:
            u = _state_or_const(state, constants, "u")
        rho = _state_or_const(state, constants, "rho")
        L = _state_or_const(state, constants, "L")
        if u is None or rho is None or L is None:
            return None
        u_char = _vec_norm_last(u)
        return jnp.asarray(rho) * jnp.asarray(u_char) * jnp.asarray(L) / (jnp.asarray(re_i) + 1e-30)

    return implied_fn


def implied_mu_momentum_navier_stokes(law_sm: Dict[str, str]) -> Callable[..., Any]:
    return implied_mu_from_re_fn(law_sm, implied_re_momentum_navier_stokes(law_sm))


def implied_re_stokes_flow(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """Re_implied ≈ ‖∇²u‖ / ‖∇p‖ for Stokes."""

    def implied_fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[jnp.ndarray]:
        p_grad = _val(state, constants, law_sm, "p_grad")
        u_lap = _val(state, constants, law_sm, "u_laplacian")
        if p_grad is None or u_lap is None:
            return None
        return _vec_norm_last(u_lap) / (_vec_norm_last(p_grad) + 1e-30)

    return implied_fn


def implied_mu_stokes_flow(law_sm: Dict[str, str]) -> Callable[..., Any]:
    return implied_mu_from_re_fn(law_sm, implied_re_stokes_flow(law_sm))


def implied_re_burgers_equation(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """Re_implied ≈ (U L) ‖∇²u‖ / ‖u_t + (u·∇)u‖."""

    def implied_fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[jnp.ndarray]:
        u_t = _val(state, constants, law_sm, "u_t")
        u = _val(state, constants, law_sm, "u")
        u_grad = _val(state, constants, law_sm, "u_grad")
        u_lap = _val(state, constants, law_sm, "u_laplacian")
        U = _val(state, constants, law_sm, "U")
        L = _val(state, constants, law_sm, "L")
        if u_t is None or u is None or u_grad is None or u_lap is None or U is None or L is None:
            return None
        u = jnp.asarray(u)
        adv = jnp.einsum("...ij,...j->...i", jnp.asarray(u_grad), u)
        rhs = jnp.asarray(u_t) + adv
        return (jnp.asarray(U) * jnp.asarray(L)) * (
            _vec_norm_last(u_lap) / (_vec_norm_last(rhs) + 1e-30)
        )

    return implied_fn


def implied_mu_burgers_equation(law_sm: Dict[str, str]) -> Callable[..., Any]:
    return implied_mu_from_re_fn(law_sm, implied_re_burgers_equation(law_sm))


def implied_viscous_acceleration_incompressible(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """Implied Newtonian viscous acceleration u_t + (u·∇)u + ∇p from the momentum balance."""

    def implied_fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[jnp.ndarray]:
        u_t = _val(state, constants, law_sm, "u_t")
        u = _val(state, constants, law_sm, "u")
        u_grad = _val(state, constants, law_sm, "u_grad")
        p_grad = _val(state, constants, law_sm, "p_grad")
        if u_t is None or u is None or u_grad is None or p_grad is None:
            return None
        adv = jnp.einsum("...ij,...j->...i", jnp.asarray(u_grad), jnp.asarray(u))
        return jnp.asarray(u_t) + adv + jnp.asarray(p_grad)

    return implied_fn


def implied_viscous_acceleration_compressible(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """Implied ρ(u_t + u·∇u) + ∇p for the simplified compressible momentum form."""

    def implied_fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[jnp.ndarray]:
        rho = _val(state, constants, law_sm, "rho")
        u_t = _val(state, constants, law_sm, "u_t")
        u = _val(state, constants, law_sm, "u")
        u_grad = _val(state, constants, law_sm, "u_grad")
        p_grad = _val(state, constants, law_sm, "p_grad")
        if rho is None or u_t is None or u is None or u_grad is None or p_grad is None:
            return None
        adv = jnp.einsum("...ij,...j->...i", jnp.asarray(u_grad), jnp.asarray(u))
        rho_b = jnp.asarray(rho)[..., jnp.newaxis]
        return rho_b * (jnp.asarray(u_t) + adv) + jnp.asarray(p_grad)

    return implied_fn



# ---------------------------------------------------------------------------
# Registry: law_name -> list of audit spec dict fragments (merged into engine lists)
# ---------------------------------------------------------------------------

# ``residual_basename`` becomes ``{basename}/implied_delta`` under constitutive/ or scaling/.
# ``include_ref_delta``: when False, skip F(pred)-F(ref) even if state_ref is set.

_LAW_IMPLIED_ROWS: Dict[str, List[Dict[str, Any]]] = {
    "fourier_conduction": [
        {
            "category": "constitutive",
            "name": "thermal_diffusivity",
            "output_key": "alpha",
            "state_map": {"k": "k", "rho": "rho", "cp": "cp"},
            "implied_maker": implied_alpha_fourier_conduction,
            "residual_basename": "thermal_diffusivity/law_fourier_conduction",
            "include_ref_delta": True,
        },
    ],
    "fick_diffusion": [
        {
            "category": "constitutive",
            "name": "mass_diffusivity",
            "output_key": "D",
            "state_map": {"fo_mass": "fo_mass", "t": "t", "L": "L"},
            "implied_maker": implied_D_fick_diffusion,
            "residual_basename": "mass_diffusivity/law_fick_diffusion",
            "include_ref_delta": True,
        },
    ],
    "wave_equation": [
        {
            "category": "constitutive",
            "name": "wave_speed_from_st",
            "output_key": "c",
            "state_map": {"omega": "omega", "L": "L", "st_wave": "st_wave"},
            "implied_maker": implied_c_wave_equation,
            "residual_basename": "wave_speed_from_st/law_wave_equation",
            "include_ref_delta": True,
        },
    ],
    "advection_diffusion": [
        {
            "category": "constitutive",
            "name": "scalar_diffusivity_from_pe",
            "output_key": "kappa",
            "state_map": {"u": "u", "L": "L", "pe": "pe"},
            "implied_maker": implied_kappa_advection_diffusion,
            "residual_basename": "scalar_diffusivity_from_pe/law_advection_diffusion",
            "include_ref_delta": True,
        },
    ],
    "momentum_navier_stokes": [
        {
            "category": "constitutive",
            "name": "dynamic_viscosity_from_re",
            "output_key": "mu",
            "state_map": {"rho": "rho", "u": "u", "L": "L", "re": "re"},
            "implied_maker": implied_mu_momentum_navier_stokes,
            "residual_basename": "dynamic_viscosity_from_re/law_momentum_navier_stokes",
            "include_ref_delta": True,
        },
    ],
    "stokes_flow": [
        {
            "category": "constitutive",
            "name": "dynamic_viscosity_from_re",
            "output_key": "mu",
            "state_map": {"rho": "rho", "u": "u", "L": "L", "re": "re"},
            "implied_maker": implied_mu_stokes_flow,
            "residual_basename": "dynamic_viscosity_from_re/law_stokes_flow",
            "include_ref_delta": True,
        },
    ],
    "burgers_equation": [
        {
            "category": "constitutive",
            "name": "dynamic_viscosity_from_re",
            "output_key": "mu",
            "state_map": {"rho": "rho", "u": "u", "L": "L", "re": "re"},
            "implied_maker": implied_mu_burgers_equation,
            "residual_basename": "dynamic_viscosity_from_re/law_burgers_equation",
            "include_ref_delta": True,
        },
    ],
    "momentum_incompressible_newtonian_laplacian": [
        {
            "category": "constitutive",
            "name": "turbulent_viscous_acceleration_k_omega",
            "output_key": "viscous_accel",
            "state_map": {
                "u_laplacian": "u_laplacian",
                "nu_molecular": "nu_molecular",
                "k": "k",
                "omega": "omega",
                "omega0": "omega0",
            },
            "implied_maker": implied_viscous_acceleration_incompressible,
            "residual_basename": "turbulent_viscous_acceleration_k_omega/law_momentum_incompressible_newtonian_laplacian",
            "include_ref_delta": True,
        },
        {
            "category": "constitutive",
            "name": "turbulent_viscous_acceleration_k_epsilon",
            "output_key": "viscous_accel",
            "state_map": {
                "u_laplacian": "u_laplacian",
                "nu_molecular": "nu_molecular",
                "C_mu": "C_mu",
                "k": "k",
                "epsilon": "epsilon",
                "eps0": "eps0",
            },
            "implied_maker": implied_viscous_acceleration_incompressible,
            "residual_basename": "turbulent_viscous_acceleration_k_epsilon/law_momentum_incompressible_newtonian_laplacian",
            "include_ref_delta": True,
        },
        {
            "category": "constitutive",
            "name": "turbulent_viscous_acceleration_smagorinsky",
            "output_key": "viscous_accel",
            "state_map": {
                "u_laplacian": "u_laplacian",
                "nu_molecular": "nu_molecular",
                "Cs": "Cs",
                "Delta": "Delta",
                "strain_rate_magnitude": "strain_rate_magnitude",
            },
            "implied_maker": implied_viscous_acceleration_incompressible,
            "residual_basename": "turbulent_viscous_acceleration_smagorinsky/law_momentum_incompressible_newtonian_laplacian",
            "include_ref_delta": True,
        },
    ],
    "momentum_compressible_newtonian_laplacian": [
        {
            "category": "constitutive",
            "name": "turbulent_viscous_acceleration_compressible_k_omega",
            "output_key": "viscous_accel",
            "state_map": {
                "rho": "rho",
                "u_laplacian": "u_laplacian",
                "nu_molecular": "nu_molecular",
                "k": "k",
                "omega": "omega",
                "omega0": "omega0",
            },
            "implied_maker": implied_viscous_acceleration_compressible,
            "residual_basename": "turbulent_viscous_acceleration_compressible_k_omega/law_momentum_compressible_newtonian_laplacian",
            "include_ref_delta": True,
        },
        {
            "category": "constitutive",
            "name": "turbulent_viscous_acceleration_compressible_k_epsilon",
            "output_key": "viscous_accel",
            "state_map": {
                "rho": "rho",
                "u_laplacian": "u_laplacian",
                "nu_molecular": "nu_molecular",
                "C_mu": "C_mu",
                "k": "k",
                "epsilon": "epsilon",
                "eps0": "eps0",
            },
            "implied_maker": implied_viscous_acceleration_compressible,
            "residual_basename": "turbulent_viscous_acceleration_compressible_k_epsilon/law_momentum_compressible_newtonian_laplacian",
            "include_ref_delta": True,
        },
        {
            "category": "constitutive",
            "name": "turbulent_viscous_acceleration_compressible_smagorinsky",
            "output_key": "viscous_accel",
            "state_map": {
                "rho": "rho",
                "u_laplacian": "u_laplacian",
                "nu_molecular": "nu_molecular",
                "Cs": "Cs",
                "Delta": "Delta",
                "strain_rate_magnitude": "strain_rate_magnitude",
            },
            "implied_maker": implied_viscous_acceleration_compressible,
            "residual_basename": "turbulent_viscous_acceleration_compressible_smagorinsky/law_momentum_compressible_newtonian_laplacian",
            "include_ref_delta": True,
        },
    ],
}


# Best-effort policy: these laws currently have no law-linked implied row because a
# stable constitutive/scaling rearrangement is not encoded in the monitor registry.
_LAW_IMPLIED_UNSUPPORTED_REASONS: Dict[str, str] = {
    "laplace_equation": "no explicit constitutive/scaling term to rearrange",
    "poisson_equation": "requires source-term closure choice not represented by one catalog model",
    "helmholtz_equation": "depends on forcing/wavenumber framing rather than one implied closure target",
    "schrodinger_steady": "complex-valued amplitude/phase closure not encoded as one implied scalar",
    "laplace_beltrami": "metric/geometry operators require chart-specific closure treatment",
    "mass_incompressible": "constraint law (divergence-free) has no direct constitutive target",
    "mass_compressible": "continuity balance has no single implied constitutive quantity",
    "euler_momentum": "inviscid law has no viscosity constitutive closure target",
    "darcy_flow": "permeability/drag closure requires porous-medium model context",
    "brinkman_extension": "mixed porous/viscous closure needs medium-specific constitutive choices",
    "viscous_dissipation": "dissipation source does not map to one implied constitutive/scaling scalar",
    "hookes_law_residual": "elastic constitutive inversion is material-model specific",
}


def list_laws_with_implied_diagnostics() -> Tuple[str, ...]:
    """Law registry names that contribute auto implied audits."""
    return tuple(sorted(_LAW_IMPLIED_ROWS.keys()))

def law_implied_unsupported_reasons() -> Dict[str, str]:
    """
    Best-effort coverage map for laws without auto implied rows.

    Returns law-name -> short reason. This documents intentional gaps in the
    law-linked implied registry rather than leaving them implicit.
    """
    return dict(_LAW_IMPLIED_UNSUPPORTED_REASONS)


def effective_audit_specs_for_fragment(d: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Return ``(constitutive_audit, scaling_audit)`` as the engine sees them: law-linked rows
    (when ``law_implied_audits`` is true) merged with fragment lists (see
    :func:`merge_fragment_law_implied_audit_specs`). Used by Studio dependency planning
    and JSON fragments without ``implied_fn``.
    """
    lic, lis = merge_law_implied_audit_specs(
        d.get("laws") or [],
        enabled=bool(d.get("law_implied_audits", True)),
    )
    frag_c = list(d.get("constitutive_audit") or [])
    frag_s = list(d.get("scaling_audit") or [])
    mc, rc = merge_fragment_law_implied_audit_specs(lic, frag_c)
    ms, rs = merge_fragment_law_implied_audit_specs(lis, frag_s)
    return mc + rc, ms + rs


def merge_law_implied_audit_specs(
    laws_spec: Sequence[Dict[str, Any]],
    *,
    enabled: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build extra ``constitutive_audit`` and ``scaling_audit`` dict rows for
    :class:`ResidualEngine` from the selected laws.

    Rows include ``implied_fn`` (not JSON-serializable) and ``residual_basename`` for
    unique log keys. When ``enabled`` is False, returns ``([], [])``.
    """
    if not enabled:
        return [], []
    constitutive: List[Dict[str, Any]] = []
    scaling: List[Dict[str, Any]] = []
    seen_basenames: set[str] = set()

    for law in laws_spec:
        law_name = str(law.get("name") or "")
        rows = _LAW_IMPLIED_ROWS.get(law_name)
        if not rows:
            continue
        law_sm = dict(law.get("state_map") or {})
        for row in rows:
            basename = str(row["residual_basename"])
            if basename in seen_basenames:
                continue
            seen_basenames.add(basename)
            implied_maker = row["implied_maker"]
            d = {
                "name": row["name"],
                "output_key": row["output_key"],
                "state_map": dict(row["state_map"]),
                "implied_fn": implied_maker(law_sm),
                "residual_basename": basename,
                "include_ref_delta": bool(row.get("include_ref_delta", True)),
            }
            if row["category"] == "constitutive":
                constitutive.append(d)
            else:
                scaling.append(d)

    return constitutive, scaling


def merge_fragment_law_implied_audit_specs(
    prepended: List[Dict[str, Any]],
    from_config: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Return ``(prepended, extras)``. Rows in ``from_config`` that declare the same
    ``residual_basename`` as a prepended law-linked row are dropped (legacy duplicate
    suppression); all other config rows are kept in ``extras``.
    """
    prep_bn = {str(r["residual_basename"]) for r in prepended if r.get("residual_basename")}
    extras: List[Dict[str, Any]] = []
    for d in from_config:
        bn = d.get("residual_basename")
        if bn and str(bn) in prep_bn:
            continue
        extras.append(d)
    return list(prepended), extras
