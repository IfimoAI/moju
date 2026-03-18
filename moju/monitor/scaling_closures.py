"""
Predefined scaling / similarity closure residuals (dimensionless identities).

Each closure returns residual = declared_value - Groups.*(...) or compound identity.
Return None if required keys missing.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp

from moju.piratio.groups import Groups


def _val(state: Dict[str, Any], constants: Dict[str, Any], key: str) -> Any:
    v = state.get(key)
    if v is None:
        v = constants.get(key)
    return v


def _all_present(state: Dict, constants: Dict, keys: Tuple[str, ...]) -> bool:
    for k in keys:
        if _val(state, constants, k) is None:
            return False
    return True


def _pe_identity(state: Dict, constants: Dict) -> Optional[Any]:
    if not _all_present(state, constants, ("Pe", "Re", "Pr")):
        return None
    return _val(state, constants, "Pe") - Groups.pe(
        _val(state, constants, "Re"), _val(state, constants, "Pr")
    )


def _le_identity(state: Dict, constants: Dict) -> Optional[Any]:
    if not _all_present(state, constants, ("Le", "Sc", "Pr")):
        return None
    return _val(state, constants, "Le") - Groups.le(
        _val(state, constants, "Sc"), _val(state, constants, "Pr")
    )


def _fo_definition(state: Dict, constants: Dict) -> Optional[Any]:
    if not _all_present(state, constants, ("Fo", "alpha", "t", "L")):
        return None
    return _val(state, constants, "Fo") - Groups.fo(
        _val(state, constants, "alpha"),
        _val(state, constants, "t"),
        _val(state, constants, "L"),
    )


def _bi_definition(state: Dict, constants: Dict) -> Optional[Any]:
    if not _all_present(state, constants, ("Bi", "h", "L", "k_solid")):
        return None
    return _val(state, constants, "Bi") - Groups.bi(
        _val(state, constants, "h"),
        _val(state, constants, "L"),
        _val(state, constants, "k_solid"),
    )


def _re_definition(state: Dict, constants: Dict) -> Optional[Any]:
    if not _all_present(state, constants, ("Re", "u", "L", "rho", "mu")):
        return None
    return _val(state, constants, "Re") - Groups.re(
        _val(state, constants, "u"),
        _val(state, constants, "L"),
        _val(state, constants, "rho"),
        _val(state, constants, "mu"),
    )


def _pr_definition(state: Dict, constants: Dict) -> Optional[Any]:
    if not _all_present(state, constants, ("Pr", "mu", "cp", "k")):
        return None
    return _val(state, constants, "Pr") - Groups.pr(
        _val(state, constants, "mu"),
        _val(state, constants, "cp"),
        _val(state, constants, "k"),
    )


def _ma_definition(state: Dict, constants: Dict) -> Optional[Any]:
    if not _all_present(state, constants, ("Ma", "u", "a")):
        return None
    return _val(state, constants, "Ma") - Groups.ma(
        _val(state, constants, "u"), _val(state, constants, "a")
    )


def _pe_mass_identity(state: Dict, constants: Dict) -> Optional[Any]:
    if not _all_present(state, constants, ("Pe_m", "Re", "Sc")):
        return None
    return _val(state, constants, "Pe_m") - Groups.pe_mass(
        _val(state, constants, "Re"), _val(state, constants, "Sc")
    )


def _nu_definition(state: Dict, constants: Dict) -> Optional[Any]:
    if not _all_present(state, constants, ("Nu", "h", "L", "k")):
        return None
    return _val(state, constants, "Nu") - Groups.nu(
        _val(state, constants, "h"),
        _val(state, constants, "L"),
        _val(state, constants, "k"),
    )


def _eu_definition(state: Dict, constants: Dict) -> Optional[Any]:
    if not _all_present(state, constants, ("Eu", "dp", "rho", "u")):
        return None
    return _val(state, constants, "Eu") - Groups.eu(
        _val(state, constants, "dp"),
        _val(state, constants, "rho"),
        _val(state, constants, "u"),
    )


SCALING_REGISTRY: Dict[str, Callable[[Dict, Dict], Any]] = {
    "pe_identity": _pe_identity,
    "le_identity": _le_identity,
    "fo_definition": _fo_definition,
    "bi_definition": _bi_definition,
    "re_definition": _re_definition,
    "pr_definition": _pr_definition,
    "ma_definition": _ma_definition,
    "pe_mass_identity": _pe_mass_identity,
    "nu_definition": _nu_definition,
    "eu_definition": _eu_definition,
}


def list_scaling_closures() -> List[str]:
    return sorted(SCALING_REGISTRY.keys())


def run_scaling_closures(
    closure_ids: List[str],
    state: Dict[str, Any],
    constants: Dict[str, Any],
    custom: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cid in closure_ids:
        fn = SCALING_REGISTRY.get(cid)
        if fn is None:
            continue
        arr = fn(state, constants)
        if arr is not None:
            out[cid] = jnp.asarray(arr)
    if custom:
        for spec in custom:
            name = spec["name"]
            arr = spec["fn"](state, constants)
            if arr is not None:
                out[f"custom/{name}"] = jnp.asarray(arr)
    return out
