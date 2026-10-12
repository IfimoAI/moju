"""
Curvilinear structured-grid differentiation via computational coordinates.

Differentiate fields on a uniform (or Fornberg) ``(ξ, η[, ζ])`` grid, then transform
physical gradients with the metric of the mapping ``x(ξ, η[, ζ])``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp

from moju.monitor.fd_stencils import (
    diff1_nonuniform,
    diff1_uniform,
    diff2_nonuniform,
    diff2_uniform,
    mixed_diff_uniform,
    validate_fd_order,
)
from moju.monitor.path_b_derivatives import _uniform_1d_spacing


def mapping_metrics_2d(
    x: Any,
    y: Any,
    *,
    d_xi: float = 1.0,
    d_eta: float = 1.0,
    order: int = 4,
) -> Dict[str, jnp.ndarray]:
    """
    First-fundamental metric terms for a 2D structured mapping ``x(ξ,η), y(ξ,η)``.

    ``x`` and ``y`` have shape ``(n_xi, n_eta)``. Computational spacings default to 1
    (index space); pass physical ``d_xi`` / ``d_eta`` when ξ, η are non-unit.
    """
    order = validate_fd_order(order)
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    x_xi = diff1_uniform(x, float(d_xi), order=order, axis=0)
    x_eta = diff1_uniform(x, float(d_eta), order=order, axis=1)
    y_xi = diff1_uniform(y, float(d_xi), order=order, axis=0)
    y_eta = diff1_uniform(y, float(d_eta), order=order, axis=1)
    jac = x_xi * y_eta - x_eta * y_xi
    # Inverse metrics (physical derivatives of computational coords)
    xi_x = y_eta / jac
    xi_y = -x_eta / jac
    eta_x = -y_xi / jac
    eta_y = x_xi / jac
    return {
        "x_xi": x_xi,
        "x_eta": x_eta,
        "y_xi": y_xi,
        "y_eta": y_eta,
        "jac": jac,
        "xi_x": xi_x,
        "xi_y": xi_y,
        "eta_x": eta_x,
        "eta_y": eta_y,
    }


def gradient_curvilinear_2d(
    field: Any,
    x: Any,
    y: Any,
    *,
    d_xi: float = 1.0,
    d_eta: float = 1.0,
    order: int = 4,
    metrics: Optional[Dict[str, jnp.ndarray]] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Physical ``(∂f/∂x, ∂f/∂y)`` from computational differences + metrics."""
    order = validate_fd_order(order)
    f = jnp.asarray(field)
    m = metrics if metrics is not None else mapping_metrics_2d(
        x, y, d_xi=d_xi, d_eta=d_eta, order=order
    )
    f_xi = diff1_uniform(f, float(d_xi), order=order, axis=0)
    f_eta = diff1_uniform(f, float(d_eta), order=order, axis=1)
    f_x = f_xi * m["xi_x"] + f_eta * m["eta_x"]
    f_y = f_xi * m["xi_y"] + f_eta * m["eta_y"]
    return f_x, f_y


def laplacian_curvilinear_2d(
    field: Any,
    x: Any,
    y: Any,
    *,
    d_xi: float = 1.0,
    d_eta: float = 1.0,
    order: int = 4,
    metrics: Optional[Dict[str, jnp.ndarray]] = None,
) -> jnp.ndarray:
    """
    Physical Laplacian via chain rule on computational second derivatives.

    Uses dedicated second-derivative and mixed stencils in ``(ξ, η)``.
    """
    order = validate_fd_order(order)
    f = jnp.asarray(field)
    m = metrics if metrics is not None else mapping_metrics_2d(
        x, y, d_xi=d_xi, d_eta=d_eta, order=order
    )
    f_xi = diff1_uniform(f, float(d_xi), order=order, axis=0)
    f_eta = diff1_uniform(f, float(d_eta), order=order, axis=1)
    f_xixi = diff2_uniform(f, float(d_xi), order=order, axis=0)
    f_etaeta = diff2_uniform(f, float(d_eta), order=order, axis=1)
    f_xieta = mixed_diff_uniform(
        f, float(d_xi), float(d_eta), order=order, axis_x=0, axis_y=1
    )
    # ∂/∂x (f_x) with f_x = f_ξ ξ_x + f_η η_x; metrics treated as spatially varying
    # Full expansion: f_xx + f_yy =
    #   f_ξξ (ξ_x²+ξ_y²) + f_ηη (η_x²+η_y²) + 2 f_ξη (ξ_x η_x + ξ_y η_y)
    #   + f_ξ (ξ_xx+ξ_yy) + f_η (η_xx+η_yy)
    xi_x, xi_y = m["xi_x"], m["xi_y"]
    eta_x, eta_y = m["eta_x"], m["eta_y"]
    # Second derivatives of inverse metrics from differentiating the first metrics
    # ξ_xx + ξ_yy = ∂(ξ_x)/∂x + ∂(ξ_y)/∂y
    xi_x_x, xi_x_y = gradient_curvilinear_2d(
        xi_x, x, y, d_xi=d_xi, d_eta=d_eta, order=order, metrics=m
    )
    xi_y_x, xi_y_y = gradient_curvilinear_2d(
        xi_y, x, y, d_xi=d_xi, d_eta=d_eta, order=order, metrics=m
    )
    eta_x_x, eta_x_y = gradient_curvilinear_2d(
        eta_x, x, y, d_xi=d_xi, d_eta=d_eta, order=order, metrics=m
    )
    eta_y_x, eta_y_y = gradient_curvilinear_2d(
        eta_y, x, y, d_xi=d_xi, d_eta=d_eta, order=order, metrics=m
    )
    lap_xi = xi_x_x + xi_y_y
    lap_eta = eta_x_x + eta_y_y
    return (
        f_xixi * (xi_x**2 + xi_y**2)
        + f_etaeta * (eta_x**2 + eta_y**2)
        + 2.0 * f_xieta * (xi_x * eta_x + xi_y * eta_y)
        + f_xi * lap_xi
        + f_eta * lap_eta
    )


def polar_mesh(
    r: Any,
    theta: Any,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Build Cartesian ``(x, y)`` on a polar ``(r, θ)`` product grid.

    ``r`` length ``nr``, ``theta`` length ``nt``; returns ``x, y`` shaped ``(nr, nt)``
    and the 1D ``r``, ``theta`` arrays.
    """
    r1 = jnp.reshape(jnp.asarray(r, dtype=float), (-1,))
    t1 = jnp.reshape(jnp.asarray(theta, dtype=float), (-1,))
    rr, tt = jnp.meshgrid(r1, t1, indexing="ij")
    x = rr * jnp.cos(tt)
    y = rr * jnp.sin(tt)
    return x, y, r1, t1


def gradient_on_polar(
    field: Any,
    r: Any,
    theta: Any,
    *,
    order: int = 4,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Physical Cartesian gradient of a scalar on a polar product grid.

    Computational coords are ``(r, θ)`` with physical spacings from the 1D axes.
    """
    r1 = jnp.reshape(jnp.asarray(r, dtype=float), (-1,))
    t1 = jnp.reshape(jnp.asarray(theta, dtype=float), (-1,))
    x, y, _, _ = polar_mesh(r1, t1)
    hr = _uniform_1d_spacing(r1)
    ht = _uniform_1d_spacing(t1)
    if hr is None or ht is None:
        # Fall back to index-space diffs + metric from physical x,y via nonuniform r
        # Use physical mapping with unit computational steps on index grid
        return gradient_curvilinear_2d(field, x, y, d_xi=1.0, d_eta=1.0, order=order)
    return gradient_curvilinear_2d(field, x, y, d_xi=float(hr), d_eta=float(ht), order=order)


def gradient_polar_physical(
    field: Any,
    r: Any,
    theta: Any,
    *,
    order: int = 4,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Cartesian gradient on a polar product grid using physical ``(r, θ)`` differences.

    ``r`` may be non-uniform (Fornberg); ``theta`` should be uniform (periodic).
    Field shape ``(nr, nt)``.
    """
    order = validate_fd_order(order)
    f = jnp.asarray(field)
    r1 = jnp.reshape(jnp.asarray(r, dtype=float), (-1,))
    t1 = jnp.reshape(jnp.asarray(theta, dtype=float), (-1,))
    rr, tt = jnp.meshgrid(r1, t1, indexing="ij")
    f_r = diff1_nonuniform(f, r1, order=order, axis=0)
    ht = _uniform_1d_spacing(t1)
    if ht is None:
        f_th = diff1_nonuniform(f, t1, order=order, axis=1)
    else:
        f_th = diff1_uniform(f, float(ht), order=order, axis=1)
    c, s = jnp.cos(tt), jnp.sin(tt)
    f_x = c * f_r - (s / rr) * f_th
    f_y = s * f_r + (c / rr) * f_th
    return f_x, f_y


def laplacian_polar_physical(
    field: Any,
    r: Any,
    theta: Any,
    *,
    order: int = 4,
) -> jnp.ndarray:
    """
    Scalar Laplacian in polar coordinates on a product grid, returned as a scalar field.

    Uses ``∇²f = f_rr + (1/r) f_r + (1/r²) f_θθ`` with dedicated/non-uniform stencils.
    """
    order = validate_fd_order(order)
    f = jnp.asarray(field)
    r1 = jnp.reshape(jnp.asarray(r, dtype=float), (-1,))
    t1 = jnp.reshape(jnp.asarray(theta, dtype=float), (-1,))
    rr, _ = jnp.meshgrid(r1, t1, indexing="ij")
    f_r = diff1_nonuniform(f, r1, order=order, axis=0)
    f_rr = diff2_nonuniform(f, r1, order=order, axis=0)
    ht = _uniform_1d_spacing(t1)
    if ht is None:
        f_thth = diff2_nonuniform(f, t1, order=order, axis=1)
    else:
        f_thth = diff2_uniform(f, float(ht), order=order, axis=1)
    return f_rr + f_r / rr + f_thth / (rr**2)


def fill_vector_grad_lap_polar(
    u: Any,
    r: Any,
    theta: Any,
    *,
    order: int = 4,
) -> Dict[str, jnp.ndarray]:
    """Cartesian ``u_grad`` / ``u_laplacian`` for a 2-component velocity on a polar grid."""
    order = validate_fd_order(order)
    vel = jnp.asarray(u)
    comps = []
    laps = []
    for c in range(int(vel.shape[-1])):
        fx, fy = gradient_polar_physical(vel[..., c], r, theta, order=order)
        comps.append(jnp.stack([fx, fy], axis=-1))
        laps.append(laplacian_polar_physical(vel[..., c], r, theta, order=order))
    return {
        "u_grad": jnp.stack(comps, axis=-2),
        "u_laplacian": jnp.stack(laps, axis=-1),
    }


def fill_vector_grad_lap_curvilinear_2d(
    u: Any,
    x: Any,
    y: Any,
    *,
    d_xi: float = 1.0,
    d_eta: float = 1.0,
    order: int = 4,
) -> Dict[str, jnp.ndarray]:
    """
    Build ``u_grad`` ``[..., 2, 2]`` and ``u_laplacian`` ``[..., 2]`` for a 2-component
    velocity on a curvilinear structured mesh.
    """
    order = validate_fd_order(order)
    vel = jnp.asarray(u)
    m = mapping_metrics_2d(x, y, d_xi=d_xi, d_eta=d_eta, order=order)
    comps = []
    laps = []
    for c in range(int(vel.shape[-1])):
        fx, fy = gradient_curvilinear_2d(
            vel[..., c], x, y, d_xi=d_xi, d_eta=d_eta, order=order, metrics=m
        )
        comps.append(jnp.stack([fx, fy], axis=-1))
        laps.append(
            laplacian_curvilinear_2d(
                vel[..., c], x, y, d_xi=d_xi, d_eta=d_eta, order=order, metrics=m
            )
        )
    u_grad = jnp.stack(comps, axis=-2)
    u_lap = jnp.stack(laps, axis=-1)
    return {"u_grad": u_grad, "u_laplacian": u_lap, "metrics": m}
