"""
Derivative provenance: which law inputs were supplied by the caller and which Moju filled.

Labels are ``"supplied"`` (present in the state passed in, including Path A autodiff outputs from
``state_builder``), ``"finite_difference"``, or ``"spectral"`` (filled by Path B recipes).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from moju.monitor.law_fd_recipes import LAW_FD_RECIPES
from moju.monitor.types import KeyDeclaration

DERIVATIVE_SUFFIXES = ("_t", "_tt", "_grad", "_laplacian")
DERIVATIVE_MODES = ("auto", "supplied_only")


class MissingSuppliedDerivativeError(KeyError):
    """A law needs a derivative key that was not supplied while ``derivatives="supplied_only"``."""

    def __init__(self, law: str, arg: str, key: str):
        self.law, self.arg, self.key = law, arg, key
        super().__init__(
            f"derivatives='supplied_only': law {law!r} argument {arg!r} needs derivative key {key!r}, "
            "which was not supplied. Provide it from the model (e.g. autodiff) or use derivatives='auto' "
            "with auto_path_b_derivatives / fill_law_fd to let Moju compute it."
        )

    def __str__(self) -> str:
        return str(self.args[0])


def is_derivative_key(
    law_name: str,
    arg: str,
    key: str,
    declarations: Optional[Mapping[str, KeyDeclaration]] = None,
) -> bool:
    """True if a law input is a derivative (FD recipe argument, derivative suffix, or declared order > 0)."""
    decl = (declarations or {}).get(key)
    if decl is not None:
        return decl.derivative_order > 0
    if arg in (LAW_FD_RECIPES.get(law_name) or {}):
        return True
    return any(key.endswith(s) for s in DERIVATIVE_SUFFIXES)


def law_derivative_inputs(
    laws_spec: Sequence[Mapping[str, Any]],
    declarations: Optional[Mapping[str, KeyDeclaration]] = None,
) -> List[Tuple[str, str, str]]:
    """``(law_name, arg, state_key)`` for every derivative input consumed by the configured laws."""
    out: List[Tuple[str, str, str]] = []
    for spec in laws_spec:
        name = str(spec.get("name") or "")
        for arg, key in (spec.get("state_map") or {}).items():
            if is_derivative_key(name, str(arg), str(key), declarations):
                out.append((name, str(arg), str(key)))
    return out


def label_provenance(
    inputs: Iterable[Tuple[str, str, str]],
    available: Mapping[str, Any],
    filled: Mapping[str, str],
) -> Dict[str, str]:
    """Map each derivative key present in ``available`` to its origin label."""
    out: Dict[str, str] = {}
    for _law, _arg, key in inputs:
        if key in filled:
            out[key] = filled[key]
        elif available.get(key) is not None:
            out[key] = "supplied"
    return out
