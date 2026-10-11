"""
Exact incompressible Navier–Stokes solutions for derivative / residual diagnostics.

Generic analytic fields only (Taylor–Green vortex, Kovasznay flow). No application-domain
or solver-specific naming.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp


def taylor_green_2d(
    x: Any,
    y: Any,
    t: Any,
    *,
    re: float = 100.0,
    u0: float = 1.0,
) -> Dict[str, Any]:
    """
    2D Taylor–Green vortex (periodic, decaying).

    Velocity and pressure satisfy incompressible NS with kinematic viscosity ``nu = 1/re``
    in a nondimensional frame with characteristic scales U=1, L=1.

    Returns fields and exact derivatives on the supplied mesh (``x``, ``y`` broadcastable;
    ``t`` scalar or array).
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    t = jnp.asarray(t)
    nu = 1.0 / float(re)
    F = jnp.exp(-2.0 * nu * t)
    u = u0 * jnp.sin(x) * jnp.cos(y) * F
    v = -u0 * jnp.cos(x) * jnp.sin(y) * F
    p = 0.25 * (u0**2) * (jnp.cos(2.0 * x) + jnp.cos(2.0 * y)) * (F**2)

    # Exact spatial derivatives
    u_x = u0 * jnp.cos(x) * jnp.cos(y) * F
    u_y = -u0 * jnp.sin(x) * jnp.sin(y) * F
    v_x = u0 * jnp.sin(x) * jnp.sin(y) * F
    v_y = -u0 * jnp.cos(x) * jnp.cos(y) * F
    p_x = -0.5 * (u0**2) * jnp.sin(2.0 * x) * (F**2)
    p_y = -0.5 * (u0**2) * jnp.sin(2.0 * y) * (F**2)

    u_xx = -u0 * jnp.sin(x) * jnp.cos(y) * F
    u_yy = -u0 * jnp.sin(x) * jnp.cos(y) * F
    v_xx = u0 * jnp.cos(x) * jnp.sin(y) * F
    v_yy = u0 * jnp.cos(x) * jnp.sin(y) * F

    # Time derivatives
    dF = -2.0 * nu * F
    u_t = u0 * jnp.sin(x) * jnp.cos(y) * dF
    v_t = -u0 * jnp.cos(x) * jnp.sin(y) * dF

    vel = jnp.stack([u, v], axis=-1)
    vel_t = jnp.stack([u_t, v_t], axis=-1)
    # Jacobian [..., i, j] = d u_i / d x_j
    u_grad = jnp.stack(
        [
            jnp.stack([u_x, u_y], axis=-1),
            jnp.stack([v_x, v_y], axis=-1),
        ],
        axis=-2,
    )
    u_lap = jnp.stack([u_xx + u_yy, v_xx + v_yy], axis=-1)
    p_grad = jnp.stack([p_x, p_y], axis=-1)

    return {
        "u": vel,
        "u_t": vel_t,
        "u_grad": u_grad,
        "u_laplacian": u_lap,
        "p": p,
        "p_grad": p_grad,
        "re": float(re),
        "F": F,
    }


def kovasznay_2d(
    x: Any,
    y: Any,
    *,
    re: float = 40.0,
) -> Dict[str, Any]:
    """
    Steady 2D Kovasznay flow (exact NS solution).

    Domain typically ``x ∈ [0, 1]``, ``y ∈ [0, 2π]`` (or any y-period).
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    re_f = float(re)
    lam = 0.5 * re_f - jnp.sqrt(0.25 * re_f**2 + 4.0 * jnp.pi**2)
    u = 1.0 - jnp.exp(lam * x) * jnp.cos(2.0 * jnp.pi * y)
    v = (lam / (2.0 * jnp.pi)) * jnp.exp(lam * x) * jnp.sin(2.0 * jnp.pi * y)
    p = 0.5 * (1.0 - jnp.exp(2.0 * lam * x))

    u_x = -lam * jnp.exp(lam * x) * jnp.cos(2.0 * jnp.pi * y)
    u_y = 2.0 * jnp.pi * jnp.exp(lam * x) * jnp.sin(2.0 * jnp.pi * y)
    v_x = (lam**2 / (2.0 * jnp.pi)) * jnp.exp(lam * x) * jnp.sin(2.0 * jnp.pi * y)
    v_y = lam * jnp.exp(lam * x) * jnp.cos(2.0 * jnp.pi * y)
    p_x = -lam * jnp.exp(2.0 * lam * x)
    p_y = jnp.zeros_like(p)

    u_xx = -(lam**2) * jnp.exp(lam * x) * jnp.cos(2.0 * jnp.pi * y)
    u_yy = (2.0 * jnp.pi) ** 2 * jnp.exp(lam * x) * jnp.cos(2.0 * jnp.pi * y)
    v_xx = (lam**3 / (2.0 * jnp.pi)) * jnp.exp(lam * x) * jnp.sin(2.0 * jnp.pi * y)
    v_yy = -2.0 * jnp.pi * lam * jnp.exp(lam * x) * jnp.sin(2.0 * jnp.pi * y)

    vel = jnp.stack([u, v], axis=-1)
    u_grad = jnp.stack(
        [
            jnp.stack([u_x, u_y], axis=-1),
            jnp.stack([v_x, v_y], axis=-1),
        ],
        axis=-2,
    )
    u_lap = jnp.stack([u_xx + u_yy, v_xx + v_yy], axis=-1)
    p_grad = jnp.stack([p_x, p_y], axis=-1)
    # Steady: zero time derivative
    u_t = jnp.zeros_like(vel)

    return {
        "u": vel,
        "u_t": u_t,
        "u_grad": u_grad,
        "u_laplacian": u_lap,
        "p": p,
        "p_grad": p_grad,
        "re": re_f,
        "lam": lam,
    }


def ns_state_from_analytic(
    fields: Dict[str, Any],
    x: Any,
    y: Any,
    *,
    t: Optional[Any] = None,
) -> Dict[str, Any]:
    """Pack analytic fields into a Path B ``state_pred`` for ``momentum_navier_stokes``."""
    state = {
        "x": jnp.asarray(x),
        "y": jnp.asarray(y),
        "u": fields["u"],
        "u_t": fields["u_t"],
        "u_grad": fields["u_grad"],
        "u_laplacian": fields["u_laplacian"],
        "p": fields["p"],
        "p_grad": fields["p_grad"],
        "re": fields["re"],
    }
    if t is not None:
        state["t"] = jnp.asarray(t)
    return state
