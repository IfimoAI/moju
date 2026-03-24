"""
Curated bridges from constitutive audit Models.* to group input primitives.

When a ``Groups.*`` spec needs a state key (e.g. ``alpha`` for ``Groups.fo``) and the user
selected a matching constitutive audit whose model implements the same closed form, Studio
can append a ``derived_state_chain`` step so NPZ need not duplicate that field.

Templates are hand-maintained (no JAX introspection). Extend ``MODEL_DERIVED_REGISTRY`` for
new algebraic models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Set, Tuple

from moju.monitor.derived_state_chain import keys_produced_by_chain


@dataclass(frozen=True)
class ModelDerivedBridge:
    """Maps a constitutive model to a JSON DSL expr using audit ``state_map`` keys."""

    required_args: Tuple[str, ...]
    build_expr: Callable[[Dict[str, str]], Dict[str, Any]]
    """``arg_name -> state_key`` for required model arguments only."""


def _expr_thermal_diffusivity(sm: Dict[str, str]) -> Dict[str, Any]:
    """alpha = k / (rho * cp) — matches Models.thermal_diffusivity."""
    k, rho, cp = sm["k"], sm["rho"], sm["cp"]
    return {
        "op": "div",
        "a": {"op": "ref", "key": k},
        "b": {"op": "mul", "a": {"op": "ref", "key": rho}, "b": {"op": "ref", "key": cp}},
    }


# model audit ``name`` -> bridge (must match ``Models.*`` registry name)
MODEL_DERIVED_REGISTRY: Dict[str, ModelDerivedBridge] = {
    "thermal_diffusivity": ModelDerivedBridge(
        required_args=("k", "rho", "cp"),
        build_expr=_expr_thermal_diffusivity,
    ),
}


def collect_group_input_state_keys(groups: List[Dict[str, Any]]) -> Set[str]:
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


def enrich_fragment_from_model_audits(frag: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of ``frag`` with extra ``derived_state_chain`` steps when:

    - A constitutive audit uses a model listed in ``MODEL_DERIVED_REGISTRY``.
    - The audit's ``output_key`` is a **group input** (appears in some ``groups`` ``state_map``).
    - The audit's ``state_map`` defines all ``required_args`` for that model.
    - The ``output_key`` is not already produced by an existing chain.

    Steps are appended in audit iteration order (registry currently has a single independent row).
    """
    out = dict(frag)
    chain: List[Dict[str, Any]] = list(frag.get("derived_state_chain") or [])
    produced = keys_produced_by_chain(chain)
    groups = list(frag.get("groups") or [])
    needed = collect_group_input_state_keys(groups)

    for audit in frag.get("constitutive_audit") or []:
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
            chain.append({"output_key": ok_out, "expr": expr})
            produced.add(ok_out)

    out["derived_state_chain"] = chain
    return out
