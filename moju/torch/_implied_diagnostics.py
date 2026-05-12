"""
Torch-native implied-balance / implied-subtract functions for law-linked
constitutive audits.

Parallel to ``moju.monitor.law_implied_diagnostics``.  Each factory
function returns a callable with the same signature as the JAX version but
using ``torch`` ops so that constitutive-audit residuals remain on the
autograd tape.

``_LAW_IMPLIED_ROWS_TORCH`` mirrors ``_LAW_IMPLIED_ROWS`` exactly except that
``implied_balance_maker_torch`` / ``implied_maker_torch`` replace the JAX-based
``implied_balance_maker`` / ``implied_maker``.  The engine calls
:func:`merge_law_implied_audit_specs_torch` to build runnable spec dicts.
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
# Balance functions (torch)
# ---------------------------------------------------------------------------


def balance_implied_fourier_conduction_torch(
    law_sm: Dict[str, str],
) -> Callable[..., Any]:
    """``T_t − pred · T_laplacian`` — torch version."""

    def fn(
        state: Dict[str, Any],
        constants: Dict[str, Any],
        pred: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        T_t = _val_t(state, constants, law_sm, "T_t")
        T_lap = _val_t(state, constants, law_sm, "T_laplacian")
        if T_t is None or T_lap is None:
            return None
        tt = _as_t(T_t)
        lap = _as_t(T_lap)
        diffusive = pred * lap
        return tt - diffusive, tt, diffusive

    return fn


def balance_implied_fick_diffusion_torch(
    law_sm: Dict[str, str],
) -> Callable[..., Any]:
    """``phi_t − pred · phi_laplacian`` — torch version."""

    def fn(
        state: Dict[str, Any],
        constants: Dict[str, Any],
        pred: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        pt = _val_t(state, constants, law_sm, "phi_t")
        pl = _val_t(state, constants, law_sm, "phi_laplacian")
        if pt is None or pl is None:
            return None
        pt_t = _as_t(pt)
        pl_t = _as_t(pl)
        diffusive = pred * pl_t
        return pt_t - diffusive, pt_t, diffusive

    return fn


def balance_implied_wave_equation_torch(
    law_sm: Dict[str, str],
) -> Callable[..., Any]:
    """``phi_tt − pred² · phi_laplacian`` — torch version."""

    def fn(
        state: Dict[str, Any],
        constants: Dict[str, Any],
        pred: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        ptt = _val_t(state, constants, law_sm, "phi_tt")
        pl = _val_t(state, constants, law_sm, "phi_laplacian")
        if ptt is None or pl is None:
            return None
        ptt_t = _as_t(ptt)
        pl_t = _as_t(pl)
        diffusive = (pred ** 2) * pl_t
        return ptt_t - diffusive, ptt_t, diffusive

    return fn


def balance_implied_kappa_advection_diffusion_torch(
    law_sm: Dict[str, str],
) -> Callable[..., Any]:
    """``phi_t + u·∇phi − (pred/(|u|L)) · phi_lap`` — torch version."""

    def fn(
        state: Dict[str, Any],
        constants: Dict[str, Any],
        pred: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
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
        adv = torch.sum(g_t * u_t, dim=-1)
        rhs = _as_t(phi_t) + adv
        U = _vec_norm_last_t(u_t)
        denom = U * _as_t(L)
        diffusive = (pred / (denom + 1e-30)) * _as_t(lap)
        return rhs - diffusive, rhs, diffusive

    return fn


def balance_implied_mu_momentum_navier_stokes_torch(
    law_sm: Dict[str, str],
) -> Callable[..., Any]:
    """NS momentum balance with Re from model μ — torch version."""

    def fn(
        state: Dict[str, Any],
        constants: Dict[str, Any],
        pred: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        u_t = _val_t(state, constants, law_sm, "u_t")
        u = _val_t(state, constants, law_sm, "u")
        u_grad = _val_t(state, constants, law_sm, "u_grad")
        p_grad = _val_t(state, constants, law_sm, "p_grad")
        u_lap = _val_t(state, constants, law_sm, "u_laplacian")
        rho = _sc_t(state, constants, "rho")
        L = _sc_t(state, constants, "L")
        if any(v is None for v in [u_t, u, u_grad, p_grad, u_lap, rho, L]):
            return None
        re_m = _re_from_mu_t(u, rho, L, pred)
        ut = _as_t(u_t)
        u_t2 = _as_t(u)
        ug = _as_t(u_grad)
        pg = _as_t(p_grad)
        ul = _as_t(u_lap)
        adv = torch.einsum("...ij,...j->...i", ug, u_t2)
        inertial = ut + adv + pg
        inv_re = (1.0 / re_m).unsqueeze(-1)
        viscous = inv_re * ul
        raw = inertial - viscous
        return raw, inertial, viscous

    return fn


def balance_implied_mu_stokes_flow_torch(
    law_sm: Dict[str, str],
) -> Callable[..., Any]:
    """Stokes balance with Re from model μ — torch version."""

    def fn(
        state: Dict[str, Any],
        constants: Dict[str, Any],
        pred: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        p_grad = _val_t(state, constants, law_sm, "p_grad")
        u_lap = _val_t(state, constants, law_sm, "u_laplacian")
        u = _val_t(state, constants, law_sm, "u") or _sc_t(state, constants, "u")
        rho = _val_t(state, constants, law_sm, "rho") or _sc_t(state, constants, "rho")
        L = _val_t(state, constants, law_sm, "L") or _sc_t(state, constants, "L")
        if any(v is None for v in [p_grad, u_lap, u, rho, L]):
            return None
        re_m = _re_from_mu_t(u, rho, L, pred)
        pg = _as_t(p_grad)
        ul = _as_t(u_lap)
        inv_re = (1.0 / re_m).unsqueeze(-1)
        viscous = inv_re * ul
        raw = pg - viscous
        return raw, pg, viscous

    return fn


def balance_implied_mu_burgers_equation_torch(
    law_sm: Dict[str, str],
) -> Callable[..., Any]:
    """Burgers balance with Re from model μ — torch version."""

    def fn(
        state: Dict[str, Any],
        constants: Dict[str, Any],
        pred: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
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
        re_m = _re_from_mu_t(u, rho, L, pred)
        ut = _as_t(u_t)
        u2 = _as_t(u)
        ug = _as_t(u_grad)
        ul = _as_t(u_lap)
        adv = torch.einsum("...ij,...j->...i", ug, u2)
        inertial = ut + adv
        nu = (_as_t(U_ref) * _as_t(Llaw)) / re_m
        viscous = nu.unsqueeze(-1) * ul
        raw = inertial - viscous
        return raw, inertial, viscous

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
            "implied_balance_maker_torch": balance_implied_fourier_conduction_torch,
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
            "implied_balance_maker_torch": balance_implied_fick_diffusion_torch,
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
            "implied_balance_maker_torch": balance_implied_wave_equation_torch,
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
            "implied_balance_maker_torch": balance_implied_kappa_advection_diffusion_torch,
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
            "implied_balance_maker_torch": balance_implied_mu_momentum_navier_stokes_torch,
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
            "implied_balance_maker_torch": balance_implied_mu_stokes_flow_torch,
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
            "implied_balance_maker_torch": balance_implied_mu_burgers_equation_torch,
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
    but returns specs with ``implied_balance_fn_torch`` / ``implied_fn_torch``
    callables instead of JAX versions.
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
            balance_maker = row.get("implied_balance_maker_torch")
            sub_maker = row.get("implied_maker_torch")
            if balance_maker is None and sub_maker is None:
                continue
            d: Dict[str, Any] = {
                "name": row["name"],
                "output_key": row["output_key"],
                "state_map": dict(row["state_map"]),
                "residual_basename": basename,
                "include_ref_delta": bool(row.get("include_ref_delta", True)),
            }
            if balance_maker is not None:
                d["implied_balance_fn_torch"] = balance_maker(law_sm)
            else:
                d["implied_fn_torch"] = sub_maker(law_sm)
            constitutive.append(d)

    return constitutive
