"""
Torch-native closure normalisation and delta computation.

Mirrors :mod:`moju.monitor.closure_registry`:

- :func:`compute_implied_delta_torch`   ↔  ``compute_implied_delta``
- :func:`compute_ref_delta_torch`       ↔  ``compute_ref_delta``

``implied_delta`` is the model-normalised fractional residual

    delta = (F(pred) - implied) / (|F(pred)| + eps)

with ``eps = _NORMALIZE_EPS = 1e-30`` (kept in sync with the JAX side).  The
formula is element-wise, so scalar / vector / tensor predictions all work.

All operations use ``torch`` so residuals remain on the autograd tape.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

_NORMALIZE_EPS: float = 1e-30


def _val(state: Dict[str, Any], constants: Dict[str, Any], key: str) -> Optional[Any]:
    v = state.get(key)
    if v is None:
        v = constants.get(key)
    return v


def _to_tensor(v: Any) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        return v.float()
    return torch.as_tensor(v, dtype=torch.float32)


def compute_implied_delta_torch_with_debug(
    *,
    fn_wrapped: Callable[..., torch.Tensor],
    arg_names: List[str],
    state_map: Dict[str, str],
    state_pred: Dict[str, Any],
    constants: Dict[str, Any],
    implied_fn_torch: Optional[
        Callable[[Dict[str, Any], Dict[str, Any]], Optional[torch.Tensor]]
    ] = None,
    output_key: Optional[str] = None,
) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, Any]]]:
    """
    Torch version of
    :func:`moju.monitor.closure_registry.compute_implied_delta_with_debug`.

    Returns ``(delta, debug_dict)``.  ``debug_dict`` is ``None`` when the audit
    row was skipped, otherwise has keys ``pred``, ``implied``, ``raw``
    (dimensional difference), ``delta`` (array sent to ``R_eff``), ``mode``,
    ``output_key``.
    """
    if implied_fn_torch is None:
        return None, None

    pred_args: List[torch.Tensor] = []
    for an in arg_names:
        sk = state_map.get(an)
        if sk is None:
            return None, None
        pv = _val(state_pred, constants, sk)
        if pv is None:
            return None, None
        pred_args.append(_to_tensor(pv))

    try:
        pred = fn_wrapped(*pred_args)
    except Exception:  # noqa: BLE001
        return None, None

    try:
        implied = implied_fn_torch(state_pred, constants)
    except Exception:  # noqa: BLE001
        return None, None
    if implied is None:
        return None, None
    implied_t = _to_tensor(implied)
    try:
        raw = pred - implied_t
        pred_debug, implied_debug = torch.broadcast_tensors(pred, implied_t)
        delta = raw / (torch.abs(pred) + _NORMALIZE_EPS)
    except Exception:  # noqa: BLE001
        return None, None
    debug = {
        "pred": pred_debug,
        "implied": implied_debug,
        "raw": raw,
        "delta": delta,
        "mode": "subtract",
        "output_key": output_key,
    }
    return delta, debug


def compute_implied_delta_torch(
    *,
    fn_wrapped: Callable[..., torch.Tensor],
    arg_names: List[str],
    state_map: Dict[str, str],
    state_pred: Dict[str, Any],
    constants: Dict[str, Any],
    implied_fn_torch: Optional[
        Callable[[Dict[str, Any], Dict[str, Any]], Optional[torch.Tensor]]
    ] = None,
    output_key: Optional[str] = None,
) -> Optional[torch.Tensor]:
    """
    Torch version of :func:`moju.monitor.closure_registry.compute_implied_delta`.

    Evaluates ``fn_wrapped(*model_args)`` → ``pred`` and returns the
    model-normalised fractional residual ``(pred − implied) / (|pred| + eps)``.
    Returns ``None`` if any required key is missing or the result is
    numerically degenerate.
    """
    delta, _debug = compute_implied_delta_torch_with_debug(
        fn_wrapped=fn_wrapped,
        arg_names=arg_names,
        state_map=state_map,
        state_pred=state_pred,
        constants=constants,
        implied_fn_torch=implied_fn_torch,
        output_key=output_key,
    )
    return delta


def compute_ref_delta_torch(
    *,
    fn_wrapped: Callable[..., torch.Tensor],
    arg_names: List[str],
    output_key: str,
    state_map: Dict[str, str],
    state_pred: Dict[str, Any],
    state_ref: Dict[str, Any],
    constants: Dict[str, Any],
    ref_delta_ref_key: Optional[str] = None,
) -> Optional[torch.Tensor]:
    """
    Torch version of :func:`moju.monitor.closure_registry.compute_ref_delta`.

    Evaluates ``fn_wrapped`` on both pred and ref states and returns the
    normalised difference ``(F(pred) − F(ref)) / (ε + |F(pred)| + |F(ref)|)``,
    or ``/ (ε + |ref|)`` when a reference tensor is supplied.
    """
    pred_args: List[torch.Tensor] = []
    ref_args: List[torch.Tensor] = []
    for an in arg_names:
        sk = state_map.get(an)
        if sk is None:
            return None
        pv = _val(state_pred, constants, sk)
        rv = _val(state_ref, constants, sk)
        if pv is None or rv is None:
            return None
        pred_args.append(_to_tensor(pv))
        ref_args.append(_to_tensor(rv))

    try:
        pred = fn_wrapped(*pred_args)
        refv = fn_wrapped(*ref_args)
    except Exception:  # noqa: BLE001
        return None

    raw = pred - refv
    ref_tensor: Optional[torch.Tensor] = None
    if ref_delta_ref_key:
        rv = _val(state_pred, constants, ref_delta_ref_key)
        if rv is not None:
            ref_tensor = _to_tensor(rv)
    if ref_tensor is None:
        rv = _val(state_pred, constants, f"{output_key}_ref")
        if rv is not None:
            ref_tensor = _to_tensor(rv)

    try:
        if ref_tensor is not None:
            return raw / (_NORMALIZE_EPS + torch.abs(ref_tensor))
        return raw / (_NORMALIZE_EPS + torch.abs(pred) + torch.abs(refv))
    except Exception:  # noqa: BLE001
        return None
