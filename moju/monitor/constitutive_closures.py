"""
Predefined JAX-differentiable constitutive closure residuals.

Each closure returns pred_field - Models.*(...) (or balance form). Return None if required
keys are missing from state/constants (closure skipped).

State keys (examples):
  sutherland_mu: mu, T + constants mu0, T0, S
  thermal_diffusivity: alpha, k, rho, cp
  ideal_gas_rho: rho, P, R, T
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp

from moju.piratio.models import Models


def _val(state: Dict[str, Any], constants: Dict[str, Any], key: str) -> Any:
    v = state.get(key)
    if v is None:
        v = constants.get(key)
    return v


def _all_present(state: Dict[str, Any], constants: Dict[str, Any], keys: Tuple[str, ...]) -> bool:
    for k in keys:
        if _val(state, constants, k) is None:
            return False
    return True


def _closure_sutherland_direct(state: Dict, constants: Dict) -> Optional[Any]:
    if not _all_present(state, constants, ("mu", "T", "mu0", "T0", "S")):
        return None
    mu = _val(state, constants, "mu")
    T = _val(state, constants, "T")
    mu0 = _val(state, constants, "mu0")
    T0 = _val(state, constants, "T0")
    S = _val(state, constants, "S")
    return mu - Models.sutherland_mu(T, mu0, T0, S)


def _closure_vft_direct(state: Dict, constants: Dict) -> Optional[Any]:
    if not _all_present(state, constants, ("mu", "T", "A", "B", "T0_v")):
        return None
    return _val(state, constants, "mu") - Models.vft_mu(
        _val(state, constants, "T"),
        _val(state, constants, "A"),
        _val(state, constants, "B"),
        _val(state, constants, "T0_v"),
    )


def _closure_ideal_gas_rho(state: Dict, constants: Dict) -> Optional[Any]:
    if not _all_present(state, constants, ("rho", "P", "R", "T")):
        return None
    return _val(state, constants, "rho") - Models.ideal_gas_rho(
        _val(state, constants, "P"),
        _val(state, constants, "R"),
        _val(state, constants, "T"),
    )


def _closure_boussinesq_rho(state: Dict, constants: Dict) -> Optional[Any]:
    if not _all_present(state, constants, ("rho", "rho0", "beta", "dT")):
        return None
    return _val(state, constants, "rho") - Models.boussinesq_rho(
        _val(state, constants, "rho0"),
        _val(state, constants, "beta"),
        _val(state, constants, "dT"),
    )


def _closure_thermal_diffusivity(state: Dict, constants: Dict) -> Optional[Any]:
    if not _all_present(state, constants, ("alpha", "k", "rho", "cp")):
        return None
    return _val(state, constants, "alpha") - Models.thermal_diffusivity(
        _val(state, constants, "k"),
        _val(state, constants, "rho"),
        _val(state, constants, "cp"),
    )


def _closure_kinematic_viscosity(state: Dict, constants: Dict) -> Optional[Any]:
    if not _all_present(state, constants, ("nu", "mu", "rho")):
        return None
    return _val(state, constants, "nu") - Models.kinematic_viscosity(
        _val(state, constants, "mu"),
        _val(state, constants, "rho"),
    )


def _closure_power_law_mu(state: Dict, constants: Dict) -> Optional[Any]:
    """mu_pred vs K * gamma_dot^(n-1). State: mu_pred (key mu_pl), gamma_dot, K, n."""
    if not _all_present(state, constants, ("mu_pl", "gamma_dot", "K", "n")):
        return None
    return _val(state, constants, "mu_pl") - Models.power_law_mu(
        _val(state, constants, "gamma_dot"),
        _val(state, constants, "K"),
        _val(state, constants, "n"),
    )


def _closure_arrhenius(state: Dict, constants: Dict) -> Optional[Any]:
    Rgas = _val(state, constants, "R_gas")
    if Rgas is None:
        Rgas = 8.314
    if not _all_present(state, constants, ("k_rate", "A", "Ea", "T")):
        return None
    return _val(state, constants, "k_rate") - Models.arrhenius_rate(
        _val(state, constants, "A"),
        _val(state, constants, "Ea"),
        _val(state, constants, "T"),
        R=Rgas,
    )


def _closure_stefan_boltzmann(state: Dict, constants: Dict) -> Optional[Any]:
    if not _all_present(state, constants, ("q_rad", "epsilon", "T")):
        return None
    return _val(state, constants, "q_rad") - Models.stefan_boltzmann_flux(
        _val(state, constants, "epsilon"),
        _val(state, constants, "T"),
    )


def _closure_heat_flux_conduction(state: Dict, constants: Dict) -> Optional[Any]:
    if not _all_present(state, constants, ("q_flux", "k", "dT", "dx")):
        return None
    return _val(state, constants, "q_flux") - Models.heat_flux_conduction(
        _val(state, constants, "k"),
        _val(state, constants, "dT"),
        _val(state, constants, "dx"),
    )


def _closure_speed_of_sound(state: Dict, constants: Dict) -> Optional[Any]:
    if not _all_present(state, constants, ("a", "gamma", "R", "T")):
        return None
    return _val(state, constants, "a") - Models.speed_of_sound(
        _val(state, constants, "gamma"),
        _val(state, constants, "R"),
        _val(state, constants, "T"),
    )


def _closure_specific_heat_nasa(state: Dict, constants: Dict) -> Optional[Any]:
    if _val(state, constants, "cp") is None or _val(state, constants, "T") is None:
        return None
    coeffs = _val(state, constants, "nasa_cp_coeffs")
    if coeffs is None:
        return None
    return _val(state, constants, "cp") - Models.specific_heat_nasa(
        _val(state, constants, "T"), coeffs
    )


# model_name -> list of (closure_id, fn)
CONSTITUTIVE_REGISTRY: Dict[str, List[Tuple[str, Callable[[Dict, Dict], Any]]]] = {
    "sutherland_mu": [("direct_mu", _closure_sutherland_direct)],
    "vft_mu": [("direct_mu", _closure_vft_direct)],
    "ideal_gas_rho": [("direct_rho", _closure_ideal_gas_rho)],
    "boussinesq_rho": [("direct_rho", _closure_boussinesq_rho)],
    "thermal_diffusivity": [("direct_alpha", _closure_thermal_diffusivity)],
    "kinematic_viscosity": [("direct_nu", _closure_kinematic_viscosity)],
    "power_law_mu": [("direct_mu_pl", _closure_power_law_mu)],
    "arrhenius_rate": [("direct_k_rate", _closure_arrhenius)],
    "stefan_boltzmann_flux": [("direct_q_rad", _closure_stefan_boltzmann)],
    "heat_flux_conduction": [("direct_q_flux", _closure_heat_flux_conduction)],
    "speed_of_sound": [("direct_a", _closure_speed_of_sound)],
    "specific_heat_nasa": [("direct_cp", _closure_specific_heat_nasa)],
}


def list_constitutive_models() -> List[str]:
    """Names that can be passed to ResidualEngine(constitutive_audit=[...])."""
    return sorted(CONSTITUTIVE_REGISTRY.keys())


def run_constitutive_closures(
    model_names: List[str],
    state: Dict[str, Any],
    constants: Dict[str, Any],
    custom: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Returns nested dict flat under one level: {"model/closure_id": array, ...}.
    Skips closures that return None.
    """
    out: Dict[str, Any] = {}
    for name in model_names:
        for closure_id, fn in CONSTITUTIVE_REGISTRY.get(name, []):
            arr = fn(state, constants)
            if arr is not None:
                out[f"{name}/{closure_id}"] = jnp.asarray(arr)
    if custom:
        for spec in custom:
            cid = spec["name"]
            arr = spec["fn"](state, constants)
            if arr is not None:
                out[f"custom/{cid}"] = jnp.asarray(arr)
    return out
