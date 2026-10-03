"""
Unit and derivative declarations for state keys (``state_units="dimensional"``).

A :class:`~moju.monitor.types.KeyDeclaration` says a key is the ``(time_order, space_order)``
derivative of a base quantity. It compiles into an ``extra_rules`` entry for
:func:`moju.piratio.nondim.dimensional_to_nd` with factor
``t_ref**time_order * L_ref**space_order / ref(base)``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set

import jax.numpy as jnp

from moju.monitor.types import KeyDeclaration
from moju.piratio.nondim import _FIELD_SCALE_RULES, _PASSTHROUGH_KEYS, NondimScales

_BASE_REF: Dict[str, Callable[[NondimScales], float]] = {
    "length": lambda s: s.L_ref,
    "time": lambda s: s.t_ref,
    "velocity": lambda s: s.U_ref,
    "temperature": lambda s: s.dT_ref,
    "temperature_offset": lambda s: s.dT_ref,
    "pressure": lambda s: s._p_ref,
    "density": lambda s: s.rho_ref,
    "concentration": lambda s: s.phi_ref,
    "field": lambda s: s.phi_ref,
    "energy": lambda s: s.E_ref,
    "dimensionless": lambda s: 1.0,
}

UNDECLARED_KEY_POLICIES = ("warn", "error")

# Engine defaults in dimensional mode: the characteristic length/velocity that laws pair with
# nondimensional fields (e.g. alpha* = fo * L*^2 / t*, nu* = U* L* / re) must be nondimensional too.
ENGINE_DEFAULT_DECLARATIONS: Dict[str, KeyDeclaration] = {
    "L": KeyDeclaration("length"),
    "U": KeyDeclaration("velocity"),
}


def declaration_factor(decl: KeyDeclaration, scales: NondimScales) -> float:
    """Multiplicative SI -> nondimensional factor for a declared key."""
    ref = float(_BASE_REF[decl.base](scales))
    return float(scales.t_ref) ** int(decl.time_order) * float(scales.L_ref) ** int(decl.space_order) / ref


def _rule_for(decl: KeyDeclaration) -> Callable[[Any, NondimScales], Any]:
    if decl.base == "temperature" and decl.derivative_order == 0:
        return lambda v, s: (jnp.asarray(v) - s.T0) / s.dT_ref
    return lambda v, s, d=decl: jnp.asarray(v) * declaration_factor(d, s)


def normalize_state_declarations(declarations: Optional[Mapping[str, Any]]) -> Dict[str, KeyDeclaration]:
    return {str(k): KeyDeclaration.coerce(v) for k, v in (declarations or {}).items()}


def compile_state_declarations(declarations: Mapping[str, KeyDeclaration]) -> Dict[str, Callable[..., Any]]:
    """``{key: KeyDeclaration}`` -> ``extra_rules`` for :func:`dimensional_to_nd`."""
    return {k: _rule_for(d) for k, d in declarations.items()}


def undeclared_state_keys(
    keys: Iterable[str],
    declarations: Mapping[str, KeyDeclaration],
    *,
    extra_passthrough: Iterable[str] = (),
) -> List[str]:
    """Keys with no built-in scaling rule, no pass-through entry, and no declaration."""
    known: Set[str] = set(_FIELD_SCALE_RULES) | set(_PASSTHROUGH_KEYS) | set(declarations) | set(extra_passthrough)
    return sorted(k for k in keys if k not in known)


def validate_undeclared_keys(policy: str) -> str:
    if policy not in UNDECLARED_KEY_POLICIES:
        raise ValueError(f"undeclared_keys must be one of {UNDECLARED_KEY_POLICIES}, got {policy!r}")
    return policy
