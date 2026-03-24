"""
Ordered **derived state** steps: compute extra ``state_pred`` keys from a safe JSON expression
DSL before ``groups`` / finite-difference fill / laws.

Each step writes ``output_key`` into state so later steps and ``Groups.*`` / ``Laws.*`` can use it
(e.g. ``kappa = 0.001 * T`` then ``alpha = kappa / (rho * cp)``, or ``k = k0 * exp(...)`` via ``exp`` / ``pow``).

Expression nodes are dicts with ``"op"`` (whitelist only). No ``eval`` of user strings.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import jax.numpy as jnp

_JSONScalar = Union[int, float, bool]


def collect_expr_ref_keys(expr: Any, out: Optional[Set[str]] = None) -> Set[str]:
    """All ``ref`` keys referenced in an expression tree."""
    if out is None:
        out = set()
    if not isinstance(expr, dict):
        return out
    op = str(expr.get("op", ""))
    if op == "ref":
        k = expr.get("key")
        if isinstance(k, str) and k:
            out.add(k)
        return out
    for child_key in ("a", "b", "x", "y", "left", "right"):
        if child_key in expr:
            collect_expr_ref_keys(expr[child_key], out)
    return out


def _as_array(v: Any) -> jnp.ndarray:
    if v is None:
        raise ValueError("missing value for ref (None)")
    return jnp.asarray(v)


def eval_derived_expr(
    expr: Any,
    env: Dict[str, Any],
    *,
    max_depth: int = 64,
    _depth: int = 0,
) -> jnp.ndarray:
    """
    Evaluate a JSON expression tree using only ``jnp`` ops.

    Supported ops
    -------------
    - ``{"op": "ref", "key": "<state_or_constant_key>"}``
    - ``{"op": "const", "value": <number>}``
    - ``{"op": "add"|"sub"|"mul"|"div", "a": expr, "b": expr}``
      (``left``/``right`` accepted as aliases for ``a``/``b``)
    - ``{"op": "neg"|"abs"|"sqrt", "x": expr}``
    - ``{"op": "square", "x": expr}``  (``x * x``)
    - ``{"op": "maximum"|"minimum", "a": expr, "b": expr}``
    - ``{"op": "exp", "x": expr}``  (``jnp.exp``)
    - ``{"op": "pow", "a": expr, "b": expr}``  (``jnp.power``; ``left``/``right`` aliases)
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
        return _as_array(env[key])
    if op == "const":
        if "value" not in expr:
            raise ValueError("const node requires 'value'")
        v = expr["value"]
        if isinstance(v, bool):
            return jnp.asarray(v, dtype=jnp.float32)
        if isinstance(v, (int, float)):
            return jnp.asarray(v, dtype=jnp.result_type(float))
        raise TypeError(f"const value must be number, got {type(v).__name__}")

    def sub(node: Any) -> jnp.ndarray:
        return eval_derived_expr(node, env, max_depth=max_depth, _depth=_depth + 1)

    def bin_args() -> Tuple[jnp.ndarray, jnp.ndarray]:
        a = expr.get("a", expr.get("left"))
        b = expr.get("b", expr.get("right"))
        if a is None or b is None:
            raise ValueError(f"{op} requires 'a' and 'b' (or 'left' and 'right')")
        return sub(a), sub(b)

    if op == "add":
        xa, xb = bin_args()
        return xa + xb
    if op == "sub":
        xa, xb = bin_args()
        return xa - xb
    if op == "mul":
        xa, xb = bin_args()
        return xa * xb
    if op == "div":
        xa, xb = bin_args()
        return xa / xb
    if op == "neg":
        if "x" not in expr:
            raise ValueError("neg requires 'x'")
        return -sub(expr["x"])
    if op == "abs":
        if "x" not in expr:
            raise ValueError("abs requires 'x'")
        return jnp.abs(sub(expr["x"]))
    if op == "sqrt":
        if "x" not in expr:
            raise ValueError("sqrt requires 'x'")
        return jnp.sqrt(sub(expr["x"]))
    if op == "square":
        if "x" not in expr:
            raise ValueError("square requires 'x'")
        t = sub(expr["x"])
        return t * t
    if op == "maximum":
        xa, xb = bin_args()
        return jnp.maximum(xa, xb)
    if op == "minimum":
        xa, xb = bin_args()
        return jnp.minimum(xa, xb)
    if op == "exp":
        if "x" not in expr:
            raise ValueError("exp requires 'x'")
        return jnp.exp(sub(expr["x"]))
    if op == "pow":
        xa, xb = bin_args()
        return jnp.power(xa, xb)

    raise ValueError(f"unsupported op {op!r}")


def apply_derived_state_chain(
    state: Dict[str, Any],
    constants: Dict[str, Any],
    steps: Sequence[Dict[str, Any]],
    *,
    max_depth: int = 64,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Apply ordered steps in ``steps``; each item is ``{"output_key": str, "expr": dict}``.

    Lookup order matches Path B merge: ``constants`` then ``state`` (state overwrites constants).
    Each step sees the updated state from previous steps.
    """
    out_state: Dict[str, Any] = dict(state)
    warnings: List[str] = []

    def merged_env() -> Dict[str, Any]:
        return {**constants, **out_state}

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            warnings.append(f"derived_state step {i}: step must be an object")
            continue
        ok = step.get("output_key")
        expr = step.get("expr")
        if not isinstance(ok, str) or not ok:
            warnings.append(f"derived_state step {i}: missing output_key")
            continue
        if expr is None:
            warnings.append(f"derived_state step {i}: missing expr")
            continue
        env = merged_env()
        try:
            arr = eval_derived_expr(expr, env, max_depth=max_depth)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"derived_state step {i} ({ok!r}): {e}")
            continue
        out_state[ok] = arr

    return out_state, warnings


def keys_produced_by_chain(steps: Sequence[Dict[str, Any]]) -> Set[str]:
    out: Set[str] = set()
    for step in steps:
        if not isinstance(step, dict):
            continue
        k = step.get("output_key")
        if isinstance(k, str) and k:
            out.add(k)
    return out


def all_ref_keys_from_chain(steps: Sequence[Dict[str, Any]]) -> Set[str]:
    keys: Set[str] = set()
    for step in steps:
        if not isinstance(step, dict):
            continue
        collect_expr_ref_keys(step.get("expr"), keys)
    return keys


__all__ = [
    "apply_derived_state_chain",
    "all_ref_keys_from_chain",
    "collect_expr_ref_keys",
    "eval_derived_expr",
    "keys_produced_by_chain",
]
