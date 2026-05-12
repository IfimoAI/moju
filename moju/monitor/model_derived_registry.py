"""
Catalog ``Models.*`` → ``derived_state_chain`` JSON expressions.

When a constitutive audit row matches a registered model name and the audit ``output_key`` is
required as a **group input** (e.g. ``alpha`` for ``Groups.fo``), Moju can append a safe DSL step
so ``apply_derived_state_chain`` materializes that key before groups / laws run.

Used by :class:`ResidualEngine` and Moju Studio (``enrich_fragment_from_model_audits``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Set, Tuple

from moju.monitor.derived_state_chain import (
    collect_expr_ref_keys,
    keys_produced_by_chain,
)


@dataclass(frozen=True)
class ModelDerivedBridge:
    """Maps a constitutive model to a JSON DSL expr using audit ``state_map`` keys."""

    required_args: Tuple[str, ...]
    build_expr: Callable[[Dict[str, str]], Dict[str, Any]]
    """``arg_name -> state_key`` for required model arguments only."""


def _expr_thermal_diffusivity(sm: Dict[str, str]) -> Dict[str, Any]:
    """``alpha = k / (rho * cp)`` — matches ``Models.thermal_diffusivity``."""
    k, rho, cp = sm["k"], sm["rho"], sm["cp"]
    return {
        "op": "div",
        "a": {"op": "ref", "key": k},
        "b": {"op": "mul", "a": {"op": "ref", "key": rho}, "b": {"op": "ref", "key": cp}},
    }


def _expr_mass_diffusivity(sm: Dict[str, str]) -> Dict[str, Any]:
    """``D = fo_mass * L**2 / t`` — matches ``Models.mass_diffusivity``."""
    fo, t, L = sm["fo_mass"], sm["t"], sm["L"]
    return {
        "op": "div",
        "a": {
            "op": "mul",
            "a": {"op": "ref", "key": fo},
            "b": {"op": "square", "x": {"op": "ref", "key": L}},
        },
        "b": {"op": "ref", "key": t},
    }


def _expr_wave_speed_from_st(sm: Dict[str, str]) -> Dict[str, Any]:
    """``c = omega * L / st_wave`` — matches ``Models.wave_speed_from_st``."""
    omega, L, st = sm["omega"], sm["L"], sm["st_wave"]
    return {
        "op": "div",
        "a": {"op": "mul", "a": {"op": "ref", "key": omega}, "b": {"op": "ref", "key": L}},
        "b": {"op": "ref", "key": st},
    }


# ``name`` on constitutive audit rows must match ``Models.*`` registry name.
MODEL_DERIVED_REGISTRY: Dict[str, ModelDerivedBridge] = {
    "thermal_diffusivity": ModelDerivedBridge(
        required_args=("k", "rho", "cp"),
        build_expr=_expr_thermal_diffusivity,
    ),
    "mass_diffusivity": ModelDerivedBridge(
        required_args=("fo_mass", "t", "L"),
        build_expr=_expr_mass_diffusivity,
    ),
    "wave_speed_from_st": ModelDerivedBridge(
        required_args=("omega", "L", "st_wave"),
        build_expr=_expr_wave_speed_from_st,
    ),
}


def collect_group_input_state_keys(groups: Sequence[Dict[str, Any]]) -> Set[str]:
    """All state keys referenced as inputs in ``groups`` specs (``state_map`` values)."""
    keys: Set[str] = set()
    for spec in groups:
        sm = spec.get("state_map") or {}
        if not isinstance(sm, dict):
            continue
        for v in sm.values():
            if isinstance(v, str) and v:
                keys.add(v)
    return keys


def _expr_self_references_output(output_key: str, expr: Dict[str, Any]) -> bool:
    refs: Set[str] = set()
    collect_expr_ref_keys(expr, refs)
    return output_key in refs


def enrich_derived_state_from_constitutive_audits(
    constitutive_audit: Sequence[Dict[str, Any]],
    groups: Sequence[Dict[str, Any]],
    derived_state_chain: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Return a new chain list: ``list(derived_state_chain)`` plus appended steps when:

    - A constitutive audit ``name`` is listed in ``MODEL_DERIVED_REGISTRY``.
    - The audit ``output_key`` appears in some group ``state_map`` value.
    - The audit ``state_map`` defines every ``required_args`` for that bridge.
    - The ``output_key`` is not already produced by an earlier step in the merged chain.
    - The expression does not reference ``output_key`` (no self-cycle in the DSL tree).

    Steps are appended in constitutive_audit iteration order.
    """
    chain: List[Dict[str, Any]] = list(derived_state_chain or [])
    produced = keys_produced_by_chain(chain)
    needed = collect_group_input_state_keys(groups)

    for audit in constitutive_audit or []:
        if not isinstance(audit, dict):
            continue
        name = audit.get("name")
        if not isinstance(name, str) or name not in MODEL_DERIVED_REGISTRY:
            continue
        bridge = MODEL_DERIVED_REGISTRY[name]
        ok_out = audit.get("output_key")
        if not isinstance(ok_out, str) or not ok_out:
            continue
        if ok_out not in needed:
            continue
        if ok_out in produced:
            continue
        sm = audit.get("state_map") or {}
        if not isinstance(sm, dict):
            continue
        arg_to_key: Dict[str, str] = {}
        for arg in bridge.required_args:
            sk = sm.get(arg)
            if not isinstance(sk, str) or not sk:
                break
            arg_to_key[arg] = sk
        else:
            expr = bridge.build_expr(arg_to_key)
            if _expr_self_references_output(ok_out, expr):
                continue
            chain.append({"output_key": ok_out, "expr": expr})
            produced.add(ok_out)

    return chain


__all__ = [
    "MODEL_DERIVED_REGISTRY",
    "ModelDerivedBridge",
    "collect_group_input_state_keys",
    "enrich_derived_state_from_constitutive_audits",
]
