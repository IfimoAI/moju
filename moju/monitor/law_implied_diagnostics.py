"""
Law-linked implied constitutive audits.

For selected :class:`Laws.*` entries, Moju can auto-append ``constitutive_audit`` rows whose
``implied_fn`` recovers a constitutive quantity from the governing-law fields.  The residual
fed to ``R_eff`` is the model-normalised fractional form

    delta = (Models.F(pred) - implied) / (|Models.F(pred)| + eps)

(see :func:`moju.monitor.closure_registry.compute_implied_delta_with_debug`).  This is the
same array shown in the constitutive divergence and consistency plots, so what is plotted
is what is scored.

``implied_fn`` comes in two flavours:

1. **Scalar-coefficient via safe division.**  The law rearranges to solve for a single
   material property (e.g. Fourier: ``alpha = T_t / T_laplacian``).  The helper uses
   :func:`_safe_ratio` to NaN-mask ill-conditioned regions and returns a scalar field
   shaped like the source fields.  Covers Fourier → α, Fick → D, wave → c,
   adv-diff → κ.

2. **Approximate / direct-field reconstruction.**  The law's constitutive term is vector
   or tensor valued and cannot be inverted to a scalar coefficient by safe division.  The
   helper returns the law-implied vector/tensor field directly (e.g.
   ``viscous_acceleration = u_t + (u·grad) u + grad p`` from the momentum balance) so the
   engine can compute element-wise ``Models.F(pred) - implied``.  μ rows (NS / Stokes /
   Burgers) use :func:`_project_scalar_coefficient` (LSQ projection onto ``u_laplacian``)
   to recover a per-point scalar from a vector balance.

Both flavours feed into the same fractional residual via
:func:`moju.monitor.closure_registry.compute_implied_delta_with_debug`, producing an array
of the same shape as ``pred`` that flows uniformly into ``_r_eff_scalar``.

We avoid pairing **both** a group and a constitutive closure that are algebraically
equivalent given fixed scales (e.g. Fourier: only ``thermal_diffusivity``, not a separate
``fo`` implied row).

See README "Law-linked implied audits" and :func:`merge_law_implied_audit_specs`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import jax.numpy as jnp

from moju.piratio.laws import Laws

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


def _vec_norm_last(x: Any) -> jnp.ndarray:
    a = jnp.asarray(x)
    if a.ndim == 0:
        return jnp.abs(a)
    return jnp.sqrt(jnp.sum(a**2, axis=-1))


def _re_from_mu(
    u: Any,
    rho: Any,
    L: Any,
    mu_pred: Any,
    *,
    eps: float = 1e-30,
) -> jnp.ndarray:
    u_char = _vec_norm_last(jnp.asarray(u))
    return (jnp.asarray(rho) * u_char * jnp.asarray(L)) / (jnp.asarray(mu_pred) + eps)


# ---------------------------------------------------------------------------
# Direct implied constitutive terms
# ---------------------------------------------------------------------------


def _safe_ratio(
    numerator: Any,
    denominator: Any,
    *,
    rel_floor: float = 1e-9,
    abs_floor: float = 1e-30,
) -> jnp.ndarray:
    """Return ``numerator / denominator`` and mark ill-conditioned divisions as NaN."""
    num = jnp.asarray(numerator)
    den = jnp.asarray(denominator)
    abs_den = jnp.abs(den)
    finite_abs_den = jnp.where(jnp.isfinite(abs_den), abs_den, 0.0)
    scale = jnp.max(finite_abs_den)
    floor = jnp.maximum(jnp.asarray(abs_floor, dtype=abs_den.dtype), jnp.asarray(rel_floor, dtype=abs_den.dtype) * scale)
    invalid = (~jnp.isfinite(num)) | (~jnp.isfinite(den)) | (abs_den <= floor)
    return jnp.where(invalid, jnp.nan, num / den)


def _project_scalar_coefficient(rhs: Any, basis: Any) -> jnp.ndarray:
    """Least-squares coefficient ``coef`` for ``rhs ~= coef * basis`` over the last axis."""
    rhs_a = jnp.asarray(rhs)
    basis_a = jnp.asarray(basis)
    if rhs_a.shape[-1] != basis_a.shape[-1]:
        raise ValueError("projection rhs and basis must share the same last-axis dimension")
    numerator = jnp.sum(rhs_a * basis_a, axis=-1)
    denominator = jnp.sum(basis_a * basis_a, axis=-1)
    return _safe_ratio(numerator, denominator)


def implied_alpha_fourier_conduction(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """Recover ``alpha`` from ``T_t = alpha * T_laplacian``."""

    def implied_fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[jnp.ndarray]:
        T_t = _val(state, constants, law_sm, "T_t")
        T_lap = _val(state, constants, law_sm, "T_laplacian")
        if T_t is None or T_lap is None:
            return None
        return _safe_ratio(T_t, T_lap)

    return implied_fn


def implied_D_fick_diffusion(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """Recover mass diffusivity ``D`` from ``phi_t = D * phi_laplacian``."""

    def implied_fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[jnp.ndarray]:
        pt = _val(state, constants, law_sm, "phi_t")
        pl = _val(state, constants, law_sm, "phi_laplacian")
        if pt is None or pl is None:
            return None
        return _safe_ratio(pt, pl)

    return implied_fn


def implied_wave_speed(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """Recover wave speed ``c`` from ``phi_tt = c**2 * phi_laplacian``."""

    def implied_fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[jnp.ndarray]:
        ptt = _val(state, constants, law_sm, "phi_tt")
        pl = _val(state, constants, law_sm, "phi_laplacian")
        if ptt is None or pl is None:
            return None
        c2 = _safe_ratio(ptt, pl)
        return jnp.where(jnp.isfinite(c2) & (c2 >= 0.0), jnp.sqrt(c2), jnp.nan)

    return implied_fn


def implied_kappa_advection_diffusion(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """Recover ``kappa`` from advection-diffusion fields."""

    def implied_fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[jnp.ndarray]:
        phi_t = _val(state, constants, law_sm, "phi_t")
        u = _val(state, constants, law_sm, "u")
        g = _val(state, constants, law_sm, "phi_grad")
        lap = _val(state, constants, law_sm, "phi_laplacian")
        L = _state_or_const(state, constants, "L")
        if phi_t is None or u is None or g is None or lap is None or L is None:
            return None
        u_a = jnp.asarray(u)
        g_a = jnp.asarray(g)
        if u_a.shape[-1] != g_a.shape[-1]:
            return None
        rhs = jnp.asarray(phi_t) + jnp.sum(g_a * u_a, axis=-1)
        numerator = rhs * _vec_norm_last(u_a) * jnp.asarray(L)
        return _safe_ratio(numerator, lap)

    return implied_fn


def implied_mu_momentum_navier_stokes(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """Recover dynamic viscosity by projecting inertial terms onto ``u_laplacian``."""

    def implied_fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[jnp.ndarray]:
        u_t = _val(state, constants, law_sm, "u_t")
        u = _val(state, constants, law_sm, "u")
        u_grad = _val(state, constants, law_sm, "u_grad")
        p_grad = _val(state, constants, law_sm, "p_grad")
        u_lap = _val(state, constants, law_sm, "u_laplacian")
        rho = _state_or_const(state, constants, "rho")
        L = _state_or_const(state, constants, "L")
        if u_t is None or u is None or u_grad is None or p_grad is None or u_lap is None:
            return None
        if rho is None or L is None:
            return None
        u_a = jnp.asarray(u)
        adv = jnp.einsum("...ij,...j->...i", jnp.asarray(u_grad), u_a)
        inertial = jnp.asarray(u_t) + adv + jnp.asarray(p_grad)
        beta = _project_scalar_coefficient(inertial, u_lap)
        return beta * jnp.asarray(rho) * _vec_norm_last(u_a) * jnp.asarray(L)

    return implied_fn


def implied_mu_stokes_flow(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """Recover dynamic viscosity by projecting pressure gradient onto ``u_laplacian``."""

    def implied_fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[jnp.ndarray]:
        p_grad = _val(state, constants, law_sm, "p_grad")
        u_lap = _val(state, constants, law_sm, "u_laplacian")
        u = _val(state, constants, law_sm, "u")
        if u is None:
            u = _state_or_const(state, constants, "u")
        rho = _val(state, constants, law_sm, "rho")
        if rho is None:
            rho = _state_or_const(state, constants, "rho")
        L = _val(state, constants, law_sm, "L")
        if L is None:
            L = _state_or_const(state, constants, "L")
        if p_grad is None or u_lap is None or u is None or rho is None or L is None:
            return None
        u_a = jnp.asarray(u)
        beta = _project_scalar_coefficient(p_grad, u_lap)
        return beta * jnp.asarray(rho) * _vec_norm_last(u_a) * jnp.asarray(L)

    return implied_fn


def implied_mu_burgers_equation(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """Recover dynamic viscosity from Burgers' implied kinematic viscosity projection."""

    def implied_fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[jnp.ndarray]:
        u_t = _val(state, constants, law_sm, "u_t")
        u = _val(state, constants, law_sm, "u")
        u_grad = _val(state, constants, law_sm, "u_grad")
        u_lap = _val(state, constants, law_sm, "u_laplacian")
        U = _val(state, constants, law_sm, "U")
        Llaw = _val(state, constants, law_sm, "L")
        rho = _state_or_const(state, constants, "rho")
        L = _state_or_const(state, constants, "L")
        if u_t is None or u is None or u_grad is None or u_lap is None:
            return None
        if U is None or Llaw is None or rho is None or L is None:
            return None
        u_a = jnp.asarray(u)
        adv = jnp.einsum("...ij,...j->...i", jnp.asarray(u_grad), u_a)
        inertial = jnp.asarray(u_t) + adv
        nu = _project_scalar_coefficient(inertial, u_lap)
        numerator = nu * jnp.asarray(rho) * _vec_norm_last(u_a) * jnp.asarray(L)
        return _safe_ratio(numerator, jnp.asarray(U) * jnp.asarray(Llaw))

    return implied_fn


def implied_stress_hookes(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """
    Subtract-mode implied for :func:`Laws.hookes_law_residual`.

    Returns the PINN-predicted stress vector so the engine can compute
    ``raw = Models.isotropic_linear_stress(E, nu, strain) − stress_pinn``.
    """

    def implied_fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[jnp.ndarray]:
        stress = _val(state, constants, law_sm, "stress")
        if stress is None:
            return None
        return jnp.asarray(stress)

    return implied_fn


def implied_rho_mass_compressible(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """
    Subtract-mode implied for :func:`Laws.mass_compressible`.

    Returns the PINN-predicted density so the engine can compute
    ``raw = Models.ideal_gas_rho(P, R, T) − rho_pinn``
    (or the Boussinesq variant).
    """

    def implied_fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[jnp.ndarray]:
        rho = _val(state, constants, law_sm, "rho")
        if rho is None:
            return None
        return jnp.asarray(rho)

    return implied_fn


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
            "implied_maker": implied_wave_speed,
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
    "hookes_law_residual": [
        {
            "category": "constitutive",
            "name": "isotropic_linear_stress",
            "output_key": "stress_model",
            "state_map": {"E": "E", "nu": "nu", "strain": "strain"},
            "implied_maker": implied_stress_hookes,
            "residual_basename": "isotropic_linear_stress/law_hookes_law_residual",
            "include_ref_delta": True,
        },
    ],
    "mass_compressible": [
        {
            "category": "constitutive",
            "name": "ideal_gas_rho",
            "output_key": "rho_model",
            "state_map": {"P": "P", "R": "R", "T": "T"},
            "implied_maker": implied_rho_mass_compressible,
            "residual_basename": "ideal_gas_rho/law_mass_compressible",
            "include_ref_delta": True,
        },
        {
            "category": "constitutive",
            "name": "boussinesq_rho",
            "output_key": "rho_model",
            "state_map": {"rho0": "rho0", "beta": "beta", "dT": "dT"},
            "implied_maker": implied_rho_mass_compressible,
            "residual_basename": "boussinesq_rho/law_mass_compressible",
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
    "poisson_equation": (
        "source is a direct law argument, not a separate material property; "
        "f_implied = -eps*phi_laplacian is just the law rearranged"
    ),
    "helmholtz_equation": (
        "kL is a domain parameter, not derived from a catalog closure; "
        "ratio -phi_laplacian/phi is undefined at wave nodes (phi=0)"
    ),
    "schrodinger_steady": (
        "division by psi is undefined at wavefunction nodes; "
        "no general catalog model for spatially-varying potential V"
    ),
    "laplace_beltrami": "metric/geometry operators require chart-specific closure treatment",
    "mass_incompressible": "constraint law (divergence-free) has no direct constitutive target",
    "euler_momentum": (
        "law is rho-normalized (p* = p/(rho_ref U^2)); rho is not an argument; "
        "inversion yields rho*=1 everywhere by construction; "
        "EOS density check belongs to mass_compressible when solving compressible Euler"
    ),
    "darcy_flow": (
        "division by |grad_p|^2 is unstable where pressure gradient vanishes; "
        "da*L^2/mu is already fully determined from law inputs"
    ),
    "brinkman_extension": (
        "mu_eff definition is debated in literature (mu, mu/sqrt(eps), etc.); "
        "recovery requires per-component division of a vector equation"
    ),
    "viscous_dissipation": (
        "ratio Phi/Phi_calc violates no-division policy; "
        "law IS the source term and has no separate conservation residual form"
    ),
    "faraday_law": (
        "conductivity belongs to Ohm's law (J=sigma*E), not Faraday's law; "
        "sigma is not recoverable from E_curl and B_t alone"
    ),
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


def all_law_names() -> Tuple[str, ...]:
    """All public ``Laws.*`` registry names."""
    names = [
        n
        for n in dir(Laws)
        if not n.startswith("_") and callable(getattr(Laws, n))
    ]
    return tuple(sorted(names))


def classify_laws_for_implied_diagnostics() -> Dict[str, str]:
    """
    Law coverage map for auto implied constitutive diagnostics.

    Values are one of:
    - ``"supported"``: law has entries in ``_LAW_IMPLIED_ROWS``
    - ``"user_specified_only"``: law is intentionally unsupported in
      ``_LAW_IMPLIED_UNSUPPORTED_REASONS`` (user can add manual audits)
    - ``"unclassified"``: law is in ``Laws.*`` but absent from both maps
    """
    supported: Set[str] = set(_LAW_IMPLIED_ROWS.keys())
    unsupported: Set[str] = set(_LAW_IMPLIED_UNSUPPORTED_REASONS.keys())
    out: Dict[str, str] = {}
    for n in all_law_names():
        if n in supported:
            out[n] = "supported"
        elif n in unsupported:
            out[n] = "user_specified_only"
        else:
            out[n] = "unclassified"
    return out


def list_unclassified_laws_for_implied_diagnostics() -> Tuple[str, ...]:
    """``Laws.*`` names missing from both supported and unsupported implied maps."""
    cls = classify_laws_for_implied_diagnostics()
    return tuple(sorted(n for n, c in cls.items() if c == "unclassified"))


def supported_auto_implied_laws_for(
    laws_spec: Sequence[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """
    For selected laws, return ``(supported, user_specified_only)`` names.

    ``supported`` are those that auto-prepend implied constitutive rows.
    ``user_specified_only`` are intentionally unsupported laws where users should
    provide explicit constitutive audit specs.
    """
    selected = {
        str(law.get("name") or "")
        for law in laws_spec
        if isinstance(law, dict) and str(law.get("name") or "")
    }
    cls = classify_laws_for_implied_diagnostics()
    supported = sorted(n for n in selected if cls.get(n) == "supported")
    manual = sorted(n for n in selected if cls.get(n) == "user_specified_only")
    return supported, manual


def effective_audit_specs_for_fragment(d: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Return ``(constitutive_audit, [])``: law-linked rows (when ``law_implied_audits`` is true)
    merged with the fragment constitutive list. Second element is always empty (legacy
    ``scaling_audit`` removed). Used by Studio dependency planning.
    """
    lic, _lis = merge_law_implied_audit_specs(
        d.get("laws") or [],
        enabled=bool(d.get("law_implied_audits", True)),
    )
    frag_c = list(d.get("constitutive_audit") or [])
    mc, rc = merge_fragment_law_implied_audit_specs(lic, frag_c)
    return mc + rc, []


def merge_law_implied_audit_specs(
    laws_spec: Sequence[Dict[str, Any]],
    *,
    enabled: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build extra ``constitutive_audit`` dict rows for :class:`ResidualEngine` from the selected laws.

    Rows include ``implied_fn`` (not JSON-serializable) and ``residual_basename`` for unique
    log keys. When ``enabled`` is False, returns ``([], [])``
    (second list is always empty; kept for call-site compatibility).
    """
    if not enabled:
        return [], []
    constitutive: List[Dict[str, Any]] = []
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
            implied_maker = row.get("implied_maker")
            if implied_maker is None:
                raise ValueError(
                    f"law_implied row for {law_name!r} missing implied_maker"
                )
            d: Dict[str, Any] = {
                "name": row["name"],
                "output_key": row["output_key"],
                "state_map": dict(row["state_map"]),
                "residual_basename": basename,
                "include_ref_delta": bool(row.get("include_ref_delta", True)),
            }
            d["implied_fn"] = implied_maker(law_sm)
            if row["category"] != "constitutive":
                raise ValueError(
                    f"law_implied row for {law_name!r} has category {row['category']!r}; "
                    "only constitutive is supported."
                )
            constitutive.append(d)

    return constitutive, []


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
