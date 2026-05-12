"""
Torch-native port of ``moju.monitor.derived_state_chain``.

Evaluates the JSON DSL expression tree using ``torch`` operations so that
the entire derived-state computation stays on the autograd tape.

Supported ops: ``ref``, ``const``, ``add``, ``sub``, ``mul``, ``div``,
``neg``, ``abs``, ``sqrt``, ``square``, ``maximum``, ``minimum``,
``exp``, ``pow`` — identical whitelist to the JAX version.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch


def _as_tensor(v: Any) -> torch.Tensor:
    if v is None:
        raise ValueError("missing value for ref (None)")
    if isinstance(v, torch.Tensor):
        return v
    return torch.as_tensor(v, dtype=torch.float32)


def eval_derived_expr_torch(
    expr: Any,
    env: Dict[str, Any],
    *,
    max_depth: int = 64,
    _depth: int = 0,
) -> torch.Tensor:
    """
    Evaluate a JSON expression tree using ``torch`` operations.

    Mirrors :func:`moju.monitor.derived_state_chain.eval_derived_expr` exactly
    but returns a ``torch.Tensor`` instead of a JAX array.
    """
    if _depth > max_depth:
        raise ValueError(f"expression exceeded max depth {max_depth}")
    if not isinstance(expr, dict):
        raise TypeError(f"expression must be a dict, got {type(expr).__name__}")

    op = str(expr.get("op", ""))

    if op == "ref":
        key = expr.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError("ref node requires string 'key'")
        if key not in env:
            raise KeyError(f"ref {key!r} not in merged state/constants")
        return _as_tensor(env[key])

    if op == "const":
        if "value" not in expr:
            raise ValueError("const node requires 'value'")
        v = expr["value"]
        if isinstance(v, bool):
            return torch.tensor(float(v))
        if isinstance(v, (int, float)):
            return torch.tensor(float(v))
        raise TypeError(f"const value must be number, got {type(v).__name__}")

    def _child(key: str) -> torch.Tensor:
        return eval_derived_expr_torch(expr[key], env, max_depth=max_depth, _depth=_depth + 1)

    def _a() -> torch.Tensor:
        return _child("a") if "a" in expr else _child("left")

    def _b() -> torch.Tensor:
        return _child("b") if "b" in expr else _child("right")

    if op == "add":
        return _a() + _b()
    if op == "sub":
        return _a() - _b()
    if op == "mul":
        return _a() * _b()
    if op == "div":
        return _a() / _b()
    if op == "neg":
        return -_child("x")
    if op == "abs":
        return torch.abs(_child("x"))
    if op == "sqrt":
        return torch.sqrt(_child("x"))
    if op == "square":
        x = _child("x")
        return x * x
    if op == "maximum":
        return torch.maximum(_a(), _b())
    if op == "minimum":
        return torch.minimum(_a(), _b())
    if op == "exp":
        return torch.exp(_child("x"))
    if op == "pow":
        return torch.pow(_a(), _b())

    raise ValueError(f"unsupported DSL op: {op!r}")


def apply_derived_state_chain_torch(
    state: Dict[str, Any],
    constants: Dict[str, Any],
    steps: Sequence[Dict[str, Any]],
    *,
    max_depth: int = 64,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Apply ordered derived-state steps using ``torch`` ops.

    Mirrors :func:`moju.monitor.derived_state_chain.apply_derived_state_chain`
    but returns ``torch.Tensor`` values so gradients flow through derived keys.

    Parameters
    ----------
    state:
        Current state dict (torch.Tensor values).
    constants:
        Engine constants (Python scalars or torch tensors).
    steps:
        List of ``{"output_key": str, "expr": dict}`` steps.

    Returns
    -------
    (new_state, warnings)
    """
    out_state: Dict[str, Any] = dict(state)
    warnings_list: List[str] = []

    def merged_env() -> Dict[str, Any]:
        return {**constants, **out_state}

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            warnings_list.append(f"derived_state step {i}: step must be an object")
            continue
        ok = step.get("output_key")
        expr = step.get("expr")
        if not isinstance(ok, str) or not ok:
            warnings_list.append(f"derived_state step {i}: missing output_key")
            continue
        if expr is None:
            warnings_list.append(f"derived_state step {i}: missing expr")
            continue
        env = merged_env()
        try:
            result = eval_derived_expr_torch(expr, env, max_depth=max_depth)
        except Exception as exc:  # noqa: BLE001
            warnings_list.append(f"derived_state step {i} ({ok!r}): {exc}")
            continue
        out_state[ok] = result

    return out_state, warnings_list
