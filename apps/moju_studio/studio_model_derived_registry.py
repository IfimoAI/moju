"""
Curated bridges from constitutive audit Models.* to group input primitives.

When a ``Groups.*`` spec needs a state key (e.g. ``alpha`` for ``Groups.fo``) and the user
selected a matching constitutive audit whose model implements the same closed form, Studio
can append a ``derived_state_chain`` step so NPZ need not duplicate that field.

Implementation lives in ``moju.monitor.model_derived_registry``; this module re-exports for Studio.
"""

from __future__ import annotations

from typing import Any, Dict

from moju.monitor.model_derived_registry import (
    MODEL_DERIVED_REGISTRY,
    ModelDerivedBridge,
    collect_group_input_state_keys,
    enrich_derived_state_from_constitutive_audits,
)


def enrich_fragment_from_model_audits(frag: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of ``frag`` with extra ``derived_state_chain`` steps when registry rules apply.
    """
    out = dict(frag)
    chain = enrich_derived_state_from_constitutive_audits(
        list(frag.get("constitutive_audit") or []),
        list(frag.get("groups") or []),
        list(frag.get("derived_state_chain") or []),
    )
    out["derived_state_chain"] = chain
    return out


__all__ = [
    "MODEL_DERIVED_REGISTRY",
    "ModelDerivedBridge",
    "collect_group_input_state_keys",
    "enrich_fragment_from_model_audits",
]
