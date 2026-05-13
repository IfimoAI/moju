"""
Torch-native implied-subtract functions for law-linked constitutive audits.

Parallel to ``moju.monitor.law_implied_diagnostics``.  Each factory
function returns a callable with the same signature as the JAX version but
using ``torch`` ops so that constitutive-audit residuals remain on the
autograd tape.

``_LAW_IMPLIED_ROWS_TORCH`` mirrors ``_LAW_IMPLIED_ROWS`` except that
``implied_maker_torch`` replaces the JAX-based ``implied_maker``. The engine
calls :func:`merge_law_implied_audit_specs_torch` to build runnable spec dicts.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch


# ---------------------------------------------------------------------------
# Helpers (torch equivalents of law_implied_diagnostics helpers)
# ---------------------------------------------------------------------------


def _val_t(
    state: Dict[str, Any],
    constants: Dict[str, Any],
    law_sm: Dict[str, str],
    arg: str,
) -> Optional[Any]:
    key = law_sm.get(arg)
    if key is None:
        return None
    v = state.get(key)
    if v is None:
        v = constants.get(key)
    return v


def _sc_t(state: Dict[str, Any], constants: Dict[str, Any], key: str) -> Optional[Any]:
    v = state.get(key)
    if v is None:
        v = constants.get(key)
    return v


def _as_t(v: Any) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        return v.float()
    return torch.as_tensor(v, dtype=torch.float32)


def _vec_norm_last_t(x: Any) -> torch.Tensor:
    a = _as_t(x)
    if a.ndim == 0:
        return torch.abs(a)
    return torch.sqrt(torch.sum(a ** 2, dim=-1))


def _re_from_mu_t(
    u: Any,
    rho: Any,
    L: Any,
    mu_pred: Any,
    eps: float = 1e-30,
) -> torch.Tensor:
    u_char = _vec_norm_last_t(_as_t(u))
    return (_as_t(rho) * u_char * _as_t(L)) / (_as_t(mu_pred) + eps)


# ---------------------------------------------------------------------------
# Direct implied constitutive terms (torch)
# ---------------------------------------------------------------------------


def _safe_ratio_t(
    numerator: Any,
    denominator: Any,
    *,
    rel_floor: float = 1e-9,
    abs_floor: float = 1e-30,
) -> torch.Tensor:
    """Return ``numerator / denominator`` and mark ill-conditioned divisions as NaN."""
    num = _as_t(numerator)
    den = _as_t(denominator)
    abs_den = torch.abs(den)
    finite_abs_den = torch.where(torch.isfinite(abs_den), abs_den, torch.zeros_like(abs_den))
    scale = torch.max(finite_abs_den)
    floor = torch.maximum(
        torch.as_tensor(abs_floor, dtype=abs_den.dtype, device=abs_den.device),
        torch.as_tensor(rel_floor, dtype=abs_den.dtype, device=abs_den.device) * scale,
    )
    invalid = (~torch.isfinite(num)) | (~torch.isfinite(den)) | (abs_den <= floor)
    return torch.where(invalid, torch.full_like(num / (den + 1.0), float("nan")), num / den)


def _project_scalar_coefficient_t(rhs: Any, basis: Any) -> torch.Tensor:
    """Least-squares coefficient ``coef`` for ``rhs ~= coef * basis`` over the last axis."""
    rhs_t = _as_t(rhs)
    basis_t = _as_t(basis)
    if rhs_t.shape[-1] != basis_t.shape[-1]:
        raise ValueError("projection rhs and basis must share the same last-axis dimension")
    numerator = torch.sum(rhs_t * basis_t, dim=-1)
    denominator = torch.sum(basis_t * basis_t, dim=-1)
    return _safe_ratio_t(numerator, denominator)


def implied_alpha_fourier_conduction_torch(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """Recover ``alpha`` from ``T_t = alpha * T_laplacian``."""

    def fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[torch.Tensor]:
        T_t = _val_t(state, constants, law_sm, "T_t")
        T_lap = _val_t(state, constants, law_sm, "T_laplacian")
        if T_t is None or T_lap is None:
            return None
        return _safe_ratio_t(T_t, T_lap)

    return fn


def implied_D_fick_diffusion_torch(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """Recover mass diffusivity ``D`` from ``phi_t = D * phi_laplacian``."""

    def fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[torch.Tensor]:
        pt = _val_t(state, constants, law_sm, "phi_t")
        pl = _val_t(state, constants, law_sm, "phi_laplacian")
        if pt is None or pl is None:
            return None
        return _safe_ratio_t(pt, pl)

    return fn


def implied_wave_speed_torch(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """Recover wave speed ``c`` from ``phi_tt = c**2 * phi_laplacian``."""

    def fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[torch.Tensor]:
        ptt = _val_t(state, constants, law_sm, "phi_tt")
        pl = _val_t(state, constants, law_sm, "phi_laplacian")
        if ptt is None or pl is None:
            return None
        c2 = _safe_ratio_t(ptt, pl)
        return torch.where(torch.isfinite(c2) & (c2 >= 0.0), torch.sqrt(c2), torch.full_like(c2, float("nan")))

    return fn


def implied_kappa_advection_diffusion_torch(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """Recover ``kappa`` from advection-diffusion fields."""

    def fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[torch.Tensor]:
        phi_t = _val_t(state, constants, law_sm, "phi_t")
        u = _val_t(state, constants, law_sm, "u")
        g = _val_t(state, constants, law_sm, "phi_grad")
        lap = _val_t(state, constants, law_sm, "phi_laplacian")
        L = _sc_t(state, constants, "L")
        if phi_t is None or u is None or g is None or lap is None or L is None:
            return None
        u_t = _as_t(u)
        g_t = _as_t(g)
        if u_t.shape[-1] != g_t.shape[-1]:
            return None
        rhs = _as_t(phi_t) + torch.sum(g_t * u_t, dim=-1)
        numerator = rhs * _vec_norm_last_t(u_t) * _as_t(L)
        return _safe_ratio_t(numerator, lap)

    return fn


def implied_mu_momentum_navier_stokes_torch(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """Recover dynamic viscosity by projecting inertial terms onto ``u_laplacian``."""

    def fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[torch.Tensor]:
        u_t = _val_t(state, constants, law_sm, "u_t")
        u = _val_t(state, constants, law_sm, "u")
        u_grad = _val_t(state, constants, law_sm, "u_grad")
        p_grad = _val_t(state, constants, law_sm, "p_grad")
        u_lap = _val_t(state, constants, law_sm, "u_laplacian")
        rho = _sc_t(state, constants, "rho")
        L = _sc_t(state, constants, "L")
        if any(v is None for v in [u_t, u, u_grad, p_grad, u_lap, rho, L]):
            return None
        u_t2 = _as_t(u)
        adv = torch.einsum("...ij,...j->...i", _as_t(u_grad), u_t2)
        inertial = _as_t(u_t) + adv + _as_t(p_grad)
        beta = _project_scalar_coefficient_t(inertial, u_lap)
        return beta * _as_t(rho) * _vec_norm_last_t(u_t2) * _as_t(L)

    return fn


def implied_mu_stokes_flow_torch(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """Recover dynamic viscosity by projecting pressure gradient onto ``u_laplacian``."""

    def fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[torch.Tensor]:
        p_grad = _val_t(state, constants, law_sm, "p_grad")
        u_lap = _val_t(state, constants, law_sm, "u_laplacian")
        u = _val_t(state, constants, law_sm, "u")
        if u is None:
            u = _sc_t(state, constants, "u")
        rho = _val_t(state, constants, law_sm, "rho")
        if rho is None:
            rho = _sc_t(state, constants, "rho")
        L = _val_t(state, constants, law_sm, "L")
        if L is None:
            L = _sc_t(state, constants, "L")
        if any(v is None for v in [p_grad, u_lap, u, rho, L]):
            return None
        u_t = _as_t(u)
        beta = _project_scalar_coefficient_t(p_grad, u_lap)
        return beta * _as_t(rho) * _vec_norm_last_t(u_t) * _as_t(L)

    return fn


def implied_mu_burgers_equation_torch(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """Recover dynamic viscosity from Burgers' implied kinematic viscosity projection."""

    def fn(state: Dict[str, Any], constants: Dict[str, Any]) -> Optional[torch.Tensor]:
        u_t = _val_t(state, constants, law_sm, "u_t")
        u = _val_t(state, constants, law_sm, "u")
        u_grad = _val_t(state, constants, law_sm, "u_grad")
        u_lap = _val_t(state, constants, law_sm, "u_laplacian")
        U_ref = _val_t(state, constants, law_sm, "U")
        Llaw = _val_t(state, constants, law_sm, "L")
        rho = _sc_t(state, constants, "rho")
        L = _sc_t(state, constants, "L")
        if any(v is None for v in [u_t, u, u_grad, u_lap, U_ref, Llaw, rho, L]):
            return None
        u_t2 = _as_t(u)
        adv = torch.einsum("...ij,...j->...i", _as_t(u_grad), u_t2)
        inertial = _as_t(u_t) + adv
        nu = _project_scalar_coefficient_t(inertial, u_lap)
        numerator = nu * _as_t(rho) * _vec_norm_last_t(u_t2) * _as_t(L)
        return _safe_ratio_t(numerator, _as_t(U_ref) * _as_t(Llaw))

    return fn


def implied_stress_hookes_torch(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """Pass-through PINN stress for Hooke's law subtract-mode audit — torch."""

    def fn(
        state: Dict[str, Any],
        constants: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        stress = _val_t(state, constants, law_sm, "stress")
        if stress is None:
            return None
        return _as_t(stress)

    return fn


def implied_rho_mass_compressible_torch(law_sm: Dict[str, str]) -> Callable[..., Any]:
    """Pass-through PINN density for mass-compressible subtract-mode audit — torch."""

    def fn(
        state: Dict[str, Any],
        constants: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        rho = _val_t(state, constants, law_sm, "rho")
        if rho is None:
            return None
        return _as_t(rho)

    return fn


def implied_viscous_acceleration_incompressible_torch(
    law_sm: Dict[str, str],
) -> Callable[..., Any]:
    """Implied viscous acceleration (incompressible) — torch."""

    def fn(
        state: Dict[str, Any],
        constants: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        u_t = _val_t(state, constants, law_sm, "u_t")
        u = _val_t(state, constants, law_sm, "u")
        u_grad = _val_t(state, constants, law_sm, "u_grad")
        p_grad = _val_t(state, constants, law_sm, "p_grad")
        if any(v is None for v in [u_t, u, u_grad, p_grad]):
            return None
        adv = torch.einsum("...ij,...j->...i", _as_t(u_grad), _as_t(u))
        return _as_t(u_t) + adv + _as_t(p_grad)

    return fn


def implied_viscous_acceleration_compressible_torch(
    law_sm: Dict[str, str],
) -> Callable[..., Any]:
    """Implied ρ(u_t + u·∇u) + ∇p (compressible) — torch."""

    def fn(
        state: Dict[str, Any],
        constants: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        rho = _val_t(state, constants, law_sm, "rho")
        u_t = _val_t(state, constants, law_sm, "u_t")
        u = _val_t(state, constants, law_sm, "u")
        u_grad = _val_t(state, constants, law_sm, "u_grad")
        p_grad = _val_t(state, constants, law_sm, "p_grad")
        if any(v is None for v in [rho, u_t, u, u_grad, p_grad]):
            return None
        adv = torch.einsum("...ij,...j->...i", _as_t(u_grad), _as_t(u))
        rho_b = _as_t(rho).unsqueeze(-1)
        return rho_b * (_as_t(u_t) + adv) + _as_t(p_grad)

    return fn


# ---------------------------------------------------------------------------
# Registry: mirrors _LAW_IMPLIED_ROWS with torch variants
# ---------------------------------------------------------------------------

_LAW_IMPLIED_ROWS_TORCH: Dict[str, List[Dict[str, Any]]] = {
    "fourier_conduction": [
        {
            "category": "constitutive",
            "name": "thermal_diffusivity",
            "output_key": "alpha",
            "state_map": {"k": "k", "rho": "rho", "cp": "cp"},
            "implied_maker_torch": implied_alpha_fourier_conduction_torch,
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
            "implied_maker_torch": implied_D_fick_diffusion_torch,
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
            "implied_maker_torch": implied_wave_speed_torch,
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
            "implied_maker_torch": implied_kappa_advection_diffusion_torch,
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
            "implied_maker_torch": implied_mu_momentum_navier_stokes_torch,
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
            "implied_maker_torch": implied_mu_stokes_flow_torch,
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
            "implied_maker_torch": implied_mu_burgers_equation_torch,
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
            "implied_maker_torch": implied_viscous_acceleration_incompressible_torch,
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
            "implied_maker_torch": implied_viscous_acceleration_incompressible_torch,
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
            "implied_maker_torch": implied_viscous_acceleration_incompressible_torch,
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
            "implied_maker_torch": implied_stress_hookes_torch,
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
            "implied_maker_torch": implied_rho_mass_compressible_torch,
            "residual_basename": "ideal_gas_rho/law_mass_compressible",
            "include_ref_delta": True,
        },
        {
            "category": "constitutive",
            "name": "boussinesq_rho",
            "output_key": "rho_model",
            "state_map": {"rho0": "rho0", "beta": "beta", "dT": "dT"},
            "implied_maker_torch": implied_rho_mass_compressible_torch,
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
            "implied_maker_torch": implied_viscous_acceleration_compressible_torch,
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
            "implied_maker_torch": implied_viscous_acceleration_compressible_torch,
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
            "implied_maker_torch": implied_viscous_acceleration_compressible_torch,
            "residual_basename": "turbulent_viscous_acceleration_compressible_smagorinsky/law_momentum_compressible_newtonian_laplacian",
            "include_ref_delta": True,
        },
    ],
}


def merge_law_implied_audit_specs_torch(
    laws_spec: Sequence[Dict[str, Any]],
    *,
    enabled: bool = True,
) -> List[Dict[str, Any]]:
    """
    Build torch-native constitutive audit spec dicts from selected laws.

    Mirrors :func:`moju.monitor.law_implied_diagnostics.merge_law_implied_audit_specs`
    but returns specs with ``implied_fn_torch`` callables instead of JAX versions.
    """
    if not enabled:
        return []
    constitutive: List[Dict[str, Any]] = []
    seen_basenames: Set[str] = set()

    for law in laws_spec:
        law_name = str(law.get("name") or "")
        rows = _LAW_IMPLIED_ROWS_TORCH.get(law_name)
        if not rows:
            continue
        law_sm = dict(law.get("state_map") or {})
        for row in rows:
            basename = str(row["residual_basename"])
            if basename in seen_basenames:
                continue
            seen_basenames.add(basename)
            sub_maker = row.get("implied_maker_torch")
            if sub_maker is None:
                continue
            d: Dict[str, Any] = {
                "name": row["name"],
                "output_key": row["output_key"],
                "state_map": dict(row["state_map"]),
                "residual_basename": basename,
                "include_ref_delta": bool(row.get("include_ref_delta", True)),
            }
            d["implied_fn_torch"] = sub_maker(law_sm)
            constitutive.append(d)

    return constitutive
