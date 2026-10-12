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


def taylor_couette_2d(
    r: Any,
    theta: Any,
    *,
    r_inner: float = 1.0,
    r_outer: float = 2.0,
    omega_inner: float = 1.0,
    omega_outer: float = 0.0,
    re: float = 100.0,
) -> Dict[str, Any]:
    """
    Steady laminar circular Couette flow between concentric cylinders (exact NS).

    Returns Cartesian velocity/pressure and exact Cartesian derivatives on the
    polar product grid ``(r, θ)`` with shapes ``(nr, nt)``.
    """
    r1 = jnp.reshape(jnp.asarray(r, dtype=float), (-1,))
    t1 = jnp.reshape(jnp.asarray(theta, dtype=float), (-1,))
    rr, tt = jnp.meshgrid(r1, t1, indexing="ij")
    ri, ro = float(r_inner), float(r_outer)
    wi, wo = float(omega_inner), float(omega_outer)
    # Classical Couette: u_θ = A r + B/r
    A = (wo * ro**2 - wi * ri**2) / (ro**2 - ri**2)
    B = (ri**2 * ro**2 * (wi - wo)) / (ro**2 - ri**2)
    u_th = A * rr + B / rr
    u_r = jnp.zeros_like(rr)
    # Cartesian velocity
    u = -u_th * jnp.sin(tt)  # u_x
    v = u_th * jnp.cos(tt)  # u_y
    # Pressure from centrifugal balance: dp/dr = ρ u_θ^2 / r  (ρ=1)
    # p(r) = ∫_{ri}^r (A s + B/s)^2 / s ds
    # Analytic: ∫ (A^2 s + 2 A B / s + B^2 / s^3) ds = A^2 s^2/2 + 2AB ln s - B^2/(2 s^2)
    def _p_of_r(rad):
        return (
            0.5 * A**2 * rad**2
            + 2.0 * A * B * jnp.log(rad)
            - 0.5 * B**2 / (rad**2)
        )

    p = _p_of_r(rr) - _p_of_r(jnp.asarray(ri))

    # Exact Cartesian derivatives via polar chain rule
    # u_θ'(r) = A - B/r^2
    du_th_dr = A - B / (rr**2)
    # u = -u_θ sinθ, v = u_θ cosθ
    # ∂u/∂r = -du_th_dr sinθ, ∂u/∂θ = -u_θ cosθ
    # ∂v/∂r = du_th_dr cosθ, ∂v/∂θ = -u_θ sinθ
    # Cartesian: ∂/∂x = cosθ ∂/∂r - (sinθ/r) ∂/∂θ, ∂/∂y = sinθ ∂/∂r + (cosθ/r) ∂/∂θ
    c, s = jnp.cos(tt), jnp.sin(tt)
    u_r_comp = -du_th_dr * s
    u_th_comp = -u_th * c
    v_r_comp = du_th_dr * c
    v_th_comp = -u_th * s
    u_x = c * u_r_comp - (s / rr) * u_th_comp
    u_y = s * u_r_comp + (c / rr) * u_th_comp
    v_x = c * v_r_comp - (s / rr) * v_th_comp
    v_y = s * v_r_comp + (c / rr) * v_th_comp
    # Pressure gradient: dp/dr = u_θ^2 / r, dp/dθ = 0
    dp_dr = (u_th**2) / rr
    p_x = c * dp_dr
    p_y = s * dp_dr

    # Circular Couette: polar operator on u_θ vanishes ⇒ Cartesian Laplacian is 0.
    u_lap = jnp.zeros(rr.shape + (2,), dtype=rr.dtype)

    vel = jnp.stack([u, v], axis=-1)
    u_grad = jnp.stack(
        [
            jnp.stack([u_x, u_y], axis=-1),
            jnp.stack([v_x, v_y], axis=-1),
        ],
        axis=-2,
    )
    p_grad = jnp.stack([p_x, p_y], axis=-1)
    x = rr * c
    y = rr * s
    return {
        "u": vel,
        "u_t": jnp.zeros_like(vel),
        "u_grad": u_grad,
        "u_laplacian": u_lap,
        "p": p,
        "p_grad": p_grad,
        "re": float(re),
        "x": x,
        "y": y,
        "r": rr,
        "theta": tt,
        "A": A,
        "B": B,
    }
