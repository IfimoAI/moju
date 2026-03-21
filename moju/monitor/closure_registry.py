"""
Model/Group closure registry for moju.monitor.

This module standardizes constitutive + scaling/similarity audits around 4 closures:
  1) ref_delta: F(state_pred) - F(state_ref) (requires state_ref)
  2) implied_delta: F(state_pred) - implied (implied_value_key in state/constants, or implied_fn)
  3) chain_dx:  d/dx[F(phi)] - sum_i (dF/dphi_i) * dphi_i/dx   (requires spatially varying inputs)
  4) chain_dt:  d/dt[F(phi)] - sum_i (dF/dphi_i) * dphi_i/dt   (requires temporally varying inputs)

Notes:
  - Path A vs Path B: monitor may build derivatives (Path A) or accept them (Path B).
    In both cases, chain closures consume derivative keys in state_pred: d_<state_key>_dx, d_<state_key>_dt.
  - Chain closures are only computed when the corresponding predicted_* list is non-empty.
  - If required keys are missing, closures return None (auditor records as unknown or omits per policy).
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from moju.piratio.groups import Groups
from moju.piratio.models import Models


def _val(state: Dict[str, Any], constants: Dict[str, Any], key: str) -> Any:
    v = state.get(key)
    if v is None:
        v = constants.get(key)
    return v


def _deriv_key(state_key: str, deriv: str) -> str:
    if deriv == "x":
        return f"d_{state_key}_dx"
    if deriv == "t":
        return f"d_{state_key}_dt"
    raise ValueError(f"Unknown deriv {deriv!r}")


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


def _grad_wrt_args(fn: Callable[..., Any], args: List[jnp.ndarray]) -> List[jnp.ndarray]:
    # fn may return array; convert to scalar for grad via sum.
    def scalar_fn(*xs):
        return jnp.sum(fn(*xs))

    grads: List[jnp.ndarray] = []
    for i in range(len(args)):
        gi = jax.grad(scalar_fn, argnums=i)(*args)
        grads.append(jnp.asarray(gi))
    return grads


def compute_ref_delta(
    *,
    fn: Callable[..., Any],
    arg_names: List[str],
    output_key: str,
    state_map: Dict[str, str],
    state_pred: Dict[str, Any],
    state_ref: Dict[str, Any],
    constants: Dict[str, Any],
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
    ref = fn(*ref_args)
    # Some users may also provide output_key in state_ref; we don't require it.
    return jnp.asarray(pred - ref)


def compute_implied_delta(
    *,
    fn: Callable[..., Any],
    arg_names: List[str],
    state_map: Dict[str, str],
    state_pred: Dict[str, Any],
    constants: Dict[str, Any],
    implied_value_key: Optional[str] = None,
    implied_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Any]] = None,
) -> Optional[jnp.ndarray]:
    """
    Residual F(pred args) - implied, where implied is either a state key or implied_fn(merged, constants).

    Returns None if implied is not configured, if any model arg is missing, if implied is missing,
    or if pred - implied is not broadcastable.
    """
    if implied_value_key is None and implied_fn is None:
        return None
    if implied_value_key is not None and implied_fn is not None:
        raise ValueError("Provide at most one of implied_value_key and implied_fn")

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

    if implied_value_key is not None:
        implied = _val(state_pred, constants, implied_value_key)
    else:
        implied = implied_fn(state_pred, constants)  # type: ignore[misc]
    if implied is None:
        return None
    implied = jnp.asarray(implied)
    try:
        return jnp.asarray(pred - implied)
    except (TypeError, ValueError):
        return None


def compute_chain(
    *,
    fn: Callable[..., Any],
    arg_names: List[str],
    output_key: str,
    state_map: Dict[str, str],
    state_pred: Dict[str, Any],
    constants: Dict[str, Any],
    predicted_varying: List[str],
    deriv: str,  # "x" or "t"
) -> Optional[jnp.ndarray]:
    # Only compute when at least one input is varying (per plan).
    if not predicted_varying:
        return None

    d_out = state_pred.get(_deriv_key(output_key, deriv))
    if d_out is None:
        return None

    args: List[jnp.ndarray] = []
    d_args: List[jnp.ndarray] = []
    for an in arg_names:
        sk = state_map.get(an)
        if sk is None:
            return None
        v = _val(state_pred, constants, sk)
        if v is None:
            return None
        args.append(jnp.asarray(v))
        if sk in predicted_varying:
            dv = state_pred.get(_deriv_key(sk, deriv))
            if dv is None:
                return None
            d_args.append(jnp.asarray(dv))
        else:
            d_args.append(jnp.asarray(0.0))

    grads = _grad_wrt_args(fn, args)
    rhs = jnp.asarray(0.0)
    for g, dv in zip(grads, d_args):
        rhs = rhs + g * dv
    return jnp.asarray(d_out) - rhs


def _broadcast_weights_for_residual(
    r: jnp.ndarray, *, w: jnp.ndarray, deriv: str
) -> Optional[jnp.ndarray]:
    """
    Best-effort broadcasting for 1D/2D structured arrays.

    Conventions:
    - deriv='x': integrate along last axis (.., Nx)
    - deriv='t': integrate along first axis (Nt, ..)
    """
    if w.ndim == 0:
        return jnp.asarray(w)
    if r.ndim == 0:
        # scalar residual: any weight is equivalent
        return jnp.asarray(1.0)

    if deriv == "x":
        axis = -1
        n = r.shape[axis]
        if w.ndim == 1 and w.shape[0] == n:
            shape = (1,) * (r.ndim - 1) + (n,)
            return jnp.reshape(w, shape)
        if w.shape == r.shape:
            return w
        return None

    if deriv == "t":
        axis = 0
        n = r.shape[axis]
        if w.ndim == 1 and w.shape[0] == n:
            shape = (n,) + (1,) * (r.ndim - 1)
            return jnp.reshape(w, shape)
        if w.shape == r.shape:
            return w
        return None

    raise ValueError(f"Unknown deriv {deriv!r}")


def compute_chain_weak(
    *,
    fn: Callable[..., Any],
    arg_names: List[str],
    output_key: str,
    state_map: Dict[str, str],
    state_pred: Dict[str, Any],
    constants: Dict[str, Any],
    predicted_varying: List[str],
    deriv: str,  # "x" or "t"
    weight_key: Optional[str] = None,
) -> Optional[jnp.ndarray]:
    """
    Weak-form / integrated variant of chain closure.

    Returns a weighted RMS scalar (or reduced array if residual is higher-rank and weights
    are scalar) to improve robustness to noisy derivatives.
    """
    r = compute_chain(
        fn=fn,
        arg_names=arg_names,
        output_key=output_key,
        state_map=state_map,
        state_pred=state_pred,
        constants=constants,
        predicted_varying=predicted_varying,
        deriv=deriv,
    )
    if r is None:
        return None

    rr = jnp.asarray(r) ** 2
    if weight_key:
        wv = _val(state_pred, constants, weight_key)
    else:
        wv = None
    if wv is None:
        # uniform weights
        return jnp.sqrt(jnp.mean(rr))

    w = jnp.asarray(wv)
    wb = _broadcast_weights_for_residual(jnp.asarray(r), w=w, deriv=deriv)
    if wb is None:
        # If weights shape doesn't match our minimal structured conventions, fall back to uniform.
        return jnp.sqrt(jnp.mean(rr))

    num = jnp.sum(wb * rr)
    den = jnp.sum(wb)
    den = jnp.where(den > 0, den, 1.0)
    return jnp.sqrt(num / den)


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

