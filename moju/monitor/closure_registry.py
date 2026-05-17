"""
Model/Group closure registry for moju.monitor.

This module standardizes constitutive + scaling/similarity audits around:
  1) ref_delta: ``F(state_pred) - F(state_ref)`` (requires ``state_ref``)
  2) implied_delta: ``F(state_pred) - implied`` (``implied_value_key`` in state/constants, or
     ``implied_fn``).

The constitutive ``implied_delta`` residual fed to ``R_eff`` is the **model-normalized
fractional residual**:

    delta = (F(pred) - implied) / (|F(pred)| + eps)

where ``eps = _NORMALIZE_EPS = 1e-30`` (kept in sync with
``visualize_constitutive._DIVERGENCE_EPS``).  This is the same array shown in the
constitutive divergence and consistency plots, so what is plotted is what is scored.
The formula is element-wise, so it works for scalar, vector and tensor ``pred``.

``ref_delta`` retains a symmetric / reference normalisation for the
``F(pred) - F(ref)`` comparison.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp

from moju.piratio.groups import Groups
from moju.piratio.models import Models

_NORMALIZE_EPS: float = 1e-30


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
    raw = jnp.asarray(pred - refv)
    ref_tensor = None
    if ref_delta_ref_key:
        ref_tensor = _val(state_pred, constants, ref_delta_ref_key)
    if ref_tensor is None:
        ref_tensor = _val(state_pred, constants, f"{output_key}_ref")
    try:
        pred_a = jnp.asarray(pred)
        ref_a = jnp.asarray(refv)
        if ref_tensor is not None:
            denom = _NORMALIZE_EPS + jnp.abs(jnp.asarray(ref_tensor))
        else:
            denom = _NORMALIZE_EPS + jnp.abs(pred_a) + jnp.abs(ref_a)
        return raw / denom
    except (TypeError, ValueError):
        return None


def compute_implied_delta_with_debug(
    *,
    fn: Callable[..., Any],
    arg_names: List[str],
    state_map: Dict[str, str],
    state_pred: Dict[str, Any],
    constants: Dict[str, Any],
    implied_value_key: Optional[str] = None,
    implied_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Any]] = None,
    output_key: Optional[str] = None,
) -> Tuple[Optional[jnp.ndarray], Optional[Dict[str, Any]]]:
    """
    Compute the model-normalised fractional ``implied_delta`` residual plus a
    debug sidecar so visualisations can inspect the dimensional terms.

    The residual fed to ``R_eff`` is

        delta = (F(pred) - implied) / (|F(pred)| + _NORMALIZE_EPS)

    Element-wise, so scalar/vector/tensor ``pred`` are all supported.  The
    sidecar keeps the dimensional difference ``raw = F(pred) - implied`` for
    diagnostics, even though ``raw`` no longer feeds ``R_eff``.

    Returns ``(delta, debug)``.  ``debug`` is ``None`` when the row was skipped;
    otherwise has the shape::

        {
            "pred":       F(pred_args),
            "implied":    implied,
            "raw":        pred - implied,           # dimensional difference
            "delta":      raw / (|pred| + eps),     # array sent to R_eff
            "mode":       "subtract",
            "output_key": output_key,
        }
    """
    modes = (1 if implied_value_key is not None else 0) + (
        1 if implied_fn is not None else 0
    )
    if modes == 0:
        return None, None
    if modes > 1:
        raise ValueError("Provide at most one of implied_value_key and implied_fn")

    pred_args: List[jnp.ndarray] = []
    for an in arg_names:
        sk = state_map.get(an)
        if sk is None:
            return None, None
        pv = _val(state_pred, constants, sk)
        if pv is None:
            return None, None
        pred_args.append(jnp.asarray(pv))
    pred = fn(*pred_args)

    if implied_value_key is not None:
        implied = _val(state_pred, constants, implied_value_key)
    else:
        implied = implied_fn(state_pred, constants)  # type: ignore[misc]
    if implied is None:
        return None, None
    implied = jnp.asarray(implied)
    pred_a = jnp.asarray(pred)
    try:
        raw = jnp.asarray(pred_a - implied)
    except (TypeError, ValueError):
        return None, None
    try:
        pred_debug, implied_debug = jnp.broadcast_arrays(pred_a, implied)
    except (TypeError, ValueError):
        pred_debug, implied_debug = pred_a, implied
    try:
        delta = raw / (jnp.abs(pred_a) + _NORMALIZE_EPS)
    except (TypeError, ValueError):
        return None, None
    debug = {
        "pred": pred_debug,
        "implied": implied_debug,
        "raw": raw,
        "delta": jnp.asarray(delta),
        "mode": "subtract",
        "output_key": output_key,
    }
    return delta, debug


def compute_implied_delta(
    *,
    fn: Callable[..., Any],
    arg_names: List[str],
    state_map: Dict[str, str],
    state_pred: Dict[str, Any],
    constants: Dict[str, Any],
    implied_value_key: Optional[str] = None,
    implied_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Any]] = None,
    output_key: Optional[str] = None,
) -> Optional[jnp.ndarray]:
    """
    Model-normalised fractional ``implied_delta`` residual:

        delta = (F(pred) - implied) / (|F(pred)| + eps)

    where ``implied`` is either ``implied_value_key`` (state/constants lookup) or
    ``implied_fn(state_pred, constants)``.  Provide **at most one** of the two.

    Returns ``None`` when the configuration is incomplete, when any argument is
    missing, or when the result is not broadcastable.  Element-wise on ``pred``,
    so scalar / vector / tensor outputs all yield a residual of the same shape.

    Note: this is a thin wrapper around
    :func:`compute_implied_delta_with_debug` that discards the debug sidecar.
    """
    delta, _debug = compute_implied_delta_with_debug(
        fn=fn,
        arg_names=arg_names,
        state_map=state_map,
        state_pred=state_pred,
        constants=constants,
        implied_value_key=implied_value_key,
        implied_fn=implied_fn,
        output_key=output_key,
    )
    return delta


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
