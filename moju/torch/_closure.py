"""
Torch-native closure normalization and delta computation.

Mirrors ``moju.monitor.closure_registry`` functions:
- :func:`normalize_discrepancy_torch`  ↔  ``apply_closure_discrepancy_normalize``
- :func:`compute_implied_delta_torch`  ↔  ``compute_implied_delta``
- :func:`compute_ref_delta_torch`      ↔  ``compute_ref_delta``

All operations use ``torch`` so residuals remain on the autograd tape.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


def _val(state: Dict[str, Any], constants: Dict[str, Any], key: str) -> Optional[Any]:
    v = state.get(key)
    if v is None:
        v = constants.get(key)
    return v


def _to_tensor(v: Any) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        return v.float()
    return torch.as_tensor(v, dtype=torch.float32)


def normalize_discrepancy_torch(
    diff: Any,
    pred: Any,
    other: Any,
    *,
    ref: Any = None,
    eps: float = 1e-30,
) -> torch.Tensor:
    """
    Nondimensional closure discrepancy — torch version.

    - If ``ref`` is not ``None``: ``diff / (ε + |ref|)``
    - Else: ``diff / (ε + |pred| + |other|)``

    Matches :func:`moju.monitor.closure_registry.apply_closure_discrepancy_normalize`.
    """
    diff_t = _to_tensor(diff)
    if ref is not None:
        ref_t = _to_tensor(ref)
        return diff_t / (eps + torch.abs(ref_t))
    pred_t = _to_tensor(pred)
    other_t = _to_tensor(other)
    return diff_t / (eps + torch.abs(pred_t) + torch.abs(other_t))


def compute_implied_delta_torch_with_debug(
    *,
    fn_wrapped: Callable[..., torch.Tensor],
    arg_names: List[str],
    state_map: Dict[str, str],
    state_pred: Dict[str, Any],
    constants: Dict[str, Any],
    implied_balance_fn_torch: Optional[
        Callable[[Dict[str, Any], Dict[str, Any], torch.Tensor],
                 Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]
    ] = None,
    implied_fn_torch: Optional[
        Callable[[Dict[str, Any], Dict[str, Any]], Optional[torch.Tensor]]
    ] = None,
    output_key: Optional[str] = None,
    implied_delta_ref_key: Optional[str] = None,
) -> Tuple[Optional[torch.Tensor], Optional[Dict[str, Any]]]:
    """
    Torch version of :func:`moju.monitor.closure_registry.compute_implied_delta_with_debug`.

    Returns ``(delta, debug_dict)`` so visualisations can access raw
    ``pred`` / ``implied`` / balance terms.  ``debug_dict`` is ``None`` when
    the audit row was skipped.
    """
    modes = (
        (1 if implied_balance_fn_torch is not None else 0)
        + (1 if implied_fn_torch is not None else 0)
    )
    if modes == 0:
        return None, None
    if modes > 1:
        raise ValueError(
            "Provide at most one of implied_balance_fn_torch and implied_fn_torch"
        )

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

    ref_tensor: Optional[torch.Tensor] = None
    if implied_delta_ref_key:
        rv = _val(state_pred, constants, implied_delta_ref_key)
        if rv is not None:
            ref_tensor = _to_tensor(rv)
    if ref_tensor is None and output_key is not None:
        rv = _val(state_pred, constants, f"{output_key}_ref")
        if rv is not None:
            ref_tensor = _to_tensor(rv)

    if implied_balance_fn_torch is not None:
        try:
            triple = implied_balance_fn_torch(state_pred, constants, pred)
        except Exception:  # noqa: BLE001
            return None, None
        if triple is None:
            return None, None
        raw, scale_a, scale_b = triple
        try:
            delta = normalize_discrepancy_torch(raw, scale_a, scale_b, ref=ref_tensor)
        except Exception:  # noqa: BLE001
            return None, None
        debug = {
            "pred": pred,
            "implied": None,
            "raw": raw,
            "scale_a": scale_a,
            "scale_b": scale_b,
            "ref": ref_tensor,
            "mode": "balance",
            "output_key": output_key,
        }
        return delta, debug

    try:
        implied = implied_fn_torch(state_pred, constants)  # type: ignore[misc]
    except Exception:  # noqa: BLE001
        return None, None
    if implied is None:
        return None, None
    implied_t = _to_tensor(implied)
    try:
        raw = pred - implied_t
        delta = normalize_discrepancy_torch(raw, pred, implied_t, ref=ref_tensor)
    except Exception:  # noqa: BLE001
        return None, None
    debug = {
        "pred": pred,
        "implied": implied_t,
        "raw": raw,
        "scale_a": None,
        "scale_b": None,
        "ref": ref_tensor,
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
    implied_balance_fn_torch: Optional[
        Callable[[Dict[str, Any], Dict[str, Any], torch.Tensor],
                 Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]
    ] = None,
    implied_fn_torch: Optional[
        Callable[[Dict[str, Any], Dict[str, Any]], Optional[torch.Tensor]]
    ] = None,
    output_key: Optional[str] = None,
    implied_delta_ref_key: Optional[str] = None,
) -> Optional[torch.Tensor]:
    """
    Torch version of :func:`moju.monitor.closure_registry.compute_implied_delta`.

    Evaluates ``fn_wrapped(*model_args)`` → ``pred`` (torch.Tensor), then
    computes the normalized discrepancy in one of two modes:

    - **Balance mode** (``implied_balance_fn_torch``): calls
      ``fn(state_pred, constants, pred)`` → ``(raw, scale_a, scale_b)`` and
      normalizes symmetrically.
    - **Subtract mode** (``implied_fn_torch``): computes
      ``pred − implied`` normalized by ``|pred| + |implied|``.

    Returns ``None`` if any required key is missing or the result is
    numerically degenerate.

    Note: thin wrapper around
    :func:`compute_implied_delta_torch_with_debug` that discards the debug
    sidecar.  Use the ``_with_debug`` variant for visualisations.
    """
    delta, _debug = compute_implied_delta_torch_with_debug(
        fn_wrapped=fn_wrapped,
        arg_names=arg_names,
        state_map=state_map,
        state_pred=state_pred,
        constants=constants,
        implied_balance_fn_torch=implied_balance_fn_torch,
        implied_fn_torch=implied_fn_torch,
        output_key=output_key,
        implied_delta_ref_key=implied_delta_ref_key,
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

    Evaluates ``fn_wrapped`` on both pred and ref states, returns
    normalised difference ``(F(pred) − F(ref)) / (ε + |F(pred)| + |F(ref)|)``.
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
        return normalize_discrepancy_torch(raw, pred, refv, ref=ref_tensor)
    except Exception:  # noqa: BLE001
        return None
