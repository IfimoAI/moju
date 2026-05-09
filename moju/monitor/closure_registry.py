"""
Model/Group closure registry for moju.monitor.

This module standardizes constitutive + scaling/similarity audits around:
  1) ref_delta: F(state_pred) - F(state_ref) (requires state_ref)
  2) implied_delta: F(state_pred) - implied (implied_value_key in state/constants, or implied_fn),
     or a law-style **balance** residual via implied_balance_fn(state_pred, constants, pred)
     with symmetric normalization scales (see :func:`compute_implied_delta`).

  **Nondimensional implied/ref discrepancies:** ``implied_delta`` and ``ref_delta`` are always
  dimensionless: default ``(pred - other) / (ε + |pred| + |other|)``. If a reference tensor is
  resolved (see ``implied_delta_ref_key`` / ``ref_delta_ref_key`` or ``{output_key}_ref`` in merged
  state/constants), use ``(pred - other) / (ε + |ref|)`` instead.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp

from moju.piratio.groups import Groups
from moju.piratio.models import Models


def apply_closure_discrepancy_normalize(
    diff: Any,
    pred: Any,
    other: Any,
    *,
    ref: Any = None,
    eps: float = 1e-30,
) -> jnp.ndarray:
    """
    Nondimensional closure discrepancy for constitutive implied/ref audits.

    - If ``ref`` is not ``None``: ``diff / (ε + |ref|)``
    - Else: ``diff / (ε + |pred| + |other|)`` (symmetric scale)
    """
    pred_a = jnp.asarray(pred)
    other_a = jnp.asarray(other)
    diff_a = jnp.asarray(diff)
    if ref is not None:
        ref_a = jnp.asarray(ref)
        denom = eps + jnp.abs(ref_a)
        return diff_a / denom
    denom = eps + jnp.abs(pred_a) + jnp.abs(other_a)
    return diff_a / denom


def _val(state: Dict[str, Any], constants: Dict[str, Any], key: str) -> Any:
    v = state.get(key)
    if v is None:
        v = constants.get(key)
    return v


def _fn_and_args(fn: Callable[..., Any]) -> Tuple[Callable[..., Any], List[str]]:
    sig = inspect.signature(fn)
    arg_names: List[str] = []
    for p in sig.parameters.values():
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            arg_names.append(p.name)
        elif p.kind == p.KEYWORD_ONLY:
            arg_names.append(p.name)
        else:
            raise TypeError("Variadic signatures are not supported for monitor closures")
    return fn, arg_names


def compute_ref_delta(
    *,
    fn: Callable[..., Any],
    arg_names: List[str],
    output_key: str,
    state_map: Dict[str, str],
    state_pred: Dict[str, Any],
    state_ref: Dict[str, Any],
    constants: Dict[str, Any],
    ref_delta_ref_key: Optional[str] = None,
) -> Optional[jnp.ndarray]:
    # Require ability to evaluate F on both pred and ref.
    pred_args = []
    ref_args = []
    for an in arg_names:
        sk = state_map.get(an)
        if sk is None:
            return None
        pv = _val(state_pred, constants, sk)
        rv = _val(state_ref, constants, sk)
        if pv is None or rv is None:
            return None
        pred_args.append(pv)
        ref_args.append(rv)
    pred = fn(*pred_args)
    refv = fn(*ref_args)
    # Some users may also provide output_key in state_ref; we don't require it.
    raw = jnp.asarray(pred - refv)
    ref_tensor = None
    if ref_delta_ref_key:
        ref_tensor = _val(state_pred, constants, ref_delta_ref_key)
    if ref_tensor is None:
        ref_tensor = _val(state_pred, constants, f"{output_key}_ref")
    try:
        return apply_closure_discrepancy_normalize(raw, pred, refv, ref=ref_tensor)
    except (TypeError, ValueError):
        return None


def compute_implied_delta(
    *,
    fn: Callable[..., Any],
    arg_names: List[str],
    state_map: Dict[str, str],
    state_pred: Dict[str, Any],
    constants: Dict[str, Any],
    implied_value_key: Optional[str] = None,
    implied_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Any]] = None,
    implied_balance_fn: Optional[
        Callable[[Dict[str, Any], Dict[str, Any], Any], Optional[Tuple[Any, Any, Any]]]
    ] = None,
    output_key: Optional[str] = None,
    implied_delta_ref_key: Optional[str] = None,
) -> Optional[jnp.ndarray]:
    """
    Residual comparing catalog ``F(pred args)`` to ``implied`` (state key or ``implied_fn``),
    or evaluating a governing-equation **balance** via ``implied_balance_fn``.

    **Subtract mode** (``implied_value_key`` or ``implied_fn``): nondimensional
    ``(F - implied) / (ε + |F| + |implied|)``, or ``/ (ε + |ref|)`` when a ref tensor resolves.

    **Balance mode** (``implied_balance_fn``): ``implied_balance_fn(state_pred, constants, pred)``
    must return ``(raw, scale_a, scale_b)`` or ``None``. Nondimensional residual is
    ``raw / (ε + |scale_a| + |scale_b|)``, or ``/ (ε + |ref|)`` when ref resolves.

    Provide **at most one** of ``implied_value_key``, ``implied_fn``, and ``implied_balance_fn``.

    Returns None if implied is not configured, if any model arg is missing, if implied/balance is
    missing, or if the result is not broadcastable.
    """
    modes = (
        (1 if implied_value_key is not None else 0)
        + (1 if implied_fn is not None else 0)
        + (1 if implied_balance_fn is not None else 0)
    )
    if modes == 0:
        return None
    if modes > 1:
        raise ValueError(
            "Provide at most one of implied_value_key, implied_fn, and implied_balance_fn"
        )

    pred_args: List[jnp.ndarray] = []
    for an in arg_names:
        sk = state_map.get(an)
        if sk is None:
            return None
        pv = _val(state_pred, constants, sk)
        if pv is None:
            return None
        pred_args.append(jnp.asarray(pv))
    pred = fn(*pred_args)

    ref_tensor = None
    if implied_delta_ref_key:
        ref_tensor = _val(state_pred, constants, implied_delta_ref_key)
    if ref_tensor is None and output_key is not None:
        ref_tensor = _val(state_pred, constants, f"{output_key}_ref")

    if implied_balance_fn is not None:
        triple = implied_balance_fn(state_pred, constants, pred)
        if triple is None:
            return None
        raw, scale_a, scale_b = triple
        try:
            return apply_closure_discrepancy_normalize(raw, scale_a, scale_b, ref=ref_tensor)
        except (TypeError, ValueError):
            return None

    if implied_value_key is not None:
        implied = _val(state_pred, constants, implied_value_key)
    else:
        implied = implied_fn(state_pred, constants)  # type: ignore[misc]
    if implied is None:
        return None
    implied = jnp.asarray(implied)
    try:
        raw = jnp.asarray(pred - implied)
    except (TypeError, ValueError):
        return None
    try:
        return apply_closure_discrepancy_normalize(raw, pred, implied, ref=ref_tensor)
    except (TypeError, ValueError):
        return None


# Registry: name -> (callable, arg_names)
MODEL_FNS: Dict[str, Tuple[Callable[..., Any], List[str]]] = {
    name: _fn_and_args(getattr(Models, name))
    for name in dir(Models)
    if not name.startswith("_") and callable(getattr(Models, name))
}

GROUP_FNS: Dict[str, Tuple[Callable[..., Any], List[str]]] = {
    name: _fn_and_args(getattr(Groups, name))
    for name in dir(Groups)
    if not name.startswith("_") and callable(getattr(Groups, name))
}


def has_model(name: str) -> bool:
    return name in MODEL_FNS


def has_group(name: str) -> bool:
    return name in GROUP_FNS


def list_models() -> List[str]:
    return sorted(MODEL_FNS.keys())


def list_groups() -> List[str]:
    return sorted(GROUP_FNS.keys())

