"""
Utilities to build `state_pred` for moju.monitor (Path A).

These helpers are intentionally small: they wrap moju.piratio.Operators so users can
generate fields and their spatial/temporal derivatives on collocation points.

Typical usage:
  - define scalar field T(params, t, x) -> scalar (or batched)
  - call build_scalar_state(...) to get T, dT_dt, dT_dx, d2T_dx2, ...
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp

from moju.piratio import Operators


ScalarFieldTX = Callable[[Any, jnp.ndarray, jnp.ndarray], jnp.ndarray]
ScalarFieldX = Callable[[Any, jnp.ndarray], jnp.ndarray]


def build_scalar_state_tx(
    *,
    field: ScalarFieldTX,
    params: Any,
    t: jnp.ndarray,
    x: jnp.ndarray,
    key: str,
    include: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    """
    Build state entries for a scalar field f(params, t, x) on batched collocation (t, x).

    include flags (all default False unless set True):
      - dt: d_<key>_dt
      - dx: d_<key>_dx (1D: returns (N,) or (N,1) depending on x shape)
      - dxx: d2_<key>_dxx (1D laplacian)
    """
    include = include or {}
    out: Dict[str, Any] = {key: field(params, t, x)}

    if include.get("dt"):
        out[f"d_{key}_dt"] = Operators.time_derivative(field, params, t, x)

    if include.get("dx"):
        # Operators.gradient expects f(params, x) -> scalar; close over t
        def fx(p, x_in):
            return field(p, t[0] if t.ndim == 0 else t, x_in)

        grad = Operators.gradient(fx, params, x)
        # For 1D x with shape (N,1), grad is (N,1). Store as same shape.
        out[f"d_{key}_dx"] = grad[..., 0] if grad.ndim > 1 and grad.shape[-1] == 1 else grad

    if include.get("dxx"):
        # Laplacian in x at each time sample; for batched t, vmap over points
        def body(ti, xi):
            return Operators.laplacian(lambda p, x_in: field(p, ti, x_in), params, xi)

        out[f"d2_{key}_dxx"] = jax.vmap(body)(t, x)

    return out


def build_scalar_state_x(
    *,
    field: ScalarFieldX,
    params: Any,
    x: jnp.ndarray,
    key: str,
    include_dx: bool = True,
    include_dxx: bool = False,
) -> Dict[str, Any]:
    """Build state for scalar field f(params, x) (no time)."""
    out: Dict[str, Any] = {key: field(params, x)}
    if include_dx:
        grad = Operators.gradient(field, params, x)
        out[f"d_{key}_dx"] = grad[..., 0] if grad.ndim > 1 and grad.shape[-1] == 1 else grad
    if include_dxx:
        out[f"d2_{key}_dxx"] = Operators.laplacian(field, params, x)
    return out

