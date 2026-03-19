"""
Model/Group closure registry for moju.monitor.

This module standardizes constitutive + scaling/similarity audits around 3 closures:
  1) ref_delta: F(state_pred) - F(state_ref) (requires state_ref)
  2) chain_dx:  d/dx[F(phi)] - sum_i (dF/dphi_i) * dphi_i/dx   (requires spatially varying inputs)
  3) chain_dt:  d/dt[F(phi)] - sum_i (dF/dphi_i) * dphi_i/dt   (requires temporally varying inputs)

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

