"""
Periodic Fourier (FFT) spatial differentiation for Path B law-input fill.

Opt-in via ``PathBGridConfig(diff_method=\"spectral\", periodic=True)`` or
``fill_path_b_spectral``. Temporal ``dt`` / ``dtt`` remain finite-difference
elsewhere. Canonical user guide: ``docs/path_b_derivatives.md``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import jax.numpy as jnp

from moju.monitor.path_b_derivatives import (
    PathBGridConfig,
    _align_1d_field_and_coord,
    _coerce_1d_axis_vector,
    _meshgrid_separable_axis_coords,
    _rectilinear_meshgrid_1d_axes,
    _separable_1d_coords,
    _uniform_1d_spacing,
)


def validate_spectral_grid_config(cfg: PathBGridConfig) -> None:
    """Raise if spectral settings are incomplete or inconsistent."""
    if cfg.diff_method != "spectral":
        return
    if cfg.diff_method not in ("fd", "spectral"):
        raise ValueError(
            f"PathBGridConfig.diff_method must be 'fd' or 'spectral', got {cfg.diff_method!r}"
        )
    if not cfg.periodic:
        raise ValueError(
            "diff_method='spectral' requires PathBGridConfig(periodic=True) "
            "(periodic Fourier differentiation on a structured grid)"
        )


def period_length_1d(coord: jnp.ndarray, n: int) -> float:
    """
    Periodic domain length ``L = n * dx`` for a uniform 1D coordinate of length ``n``.

    Raises ``ValueError`` if spacing is non-uniform or length mismatches.
    """
    c = jnp.reshape(jnp.asarray(coord), (-1,))
    if int(c.shape[0]) != int(n):
        raise ValueError(
            f"spectral: coordinate length {int(c.shape[0])} != field axis length {n}"
        )
    h = _uniform_1d_spacing(c)
    if h is None:
        raise ValueError(
            "spectral Path B requires uniform spatial spacing; "
            "non-uniform coordinates are not supported (use diff_method='fd' or resample)"
        )
    return float(n) * float(h)


def spectral_diff_along_axis(
    field: jnp.ndarray,
    axis: int,
    L: float,
    *,
    order: int = 1,
) -> jnp.ndarray:
    """
    Differentiate ``field`` along ``axis`` with periodic Fourier multipliers ``(ik)^order``.

    ``L`` is the period length along that axis. Returns a real array (imag discarded).
    """
    if order not in (1, 2):
        raise ValueError(f"spectral order must be 1 or 2, got {order}")
    field = jnp.asarray(field)
    if axis < 0:
        axis = field.ndim + axis
    n = int(field.shape[axis])
    if n < 2:
        raise ValueError("spectral: need at least 2 points along differentiation axis")
    if not (L > 0.0):
        raise ValueError(f"spectral: period length L must be positive, got {L}")

    # fftfreq(n, d=L/n) → cycles per unit length; multiply by 2π for angular wavenumber
    k = (2.0 * jnp.pi) * jnp.fft.fftfreq(n, d=L / float(n))
    factor = (1j * k) ** int(order)

    # Move target axis to last, FFT along last axis, multiply, iFFT, restore.
    # (float32 FFT is sufficient for Path B fill; float64 needs jax_enable_x64.)
    moved = jnp.moveaxis(field, axis, -1)
    flat_leading = int(jnp.prod(jnp.array(moved.shape[:-1]))) if moved.ndim > 1 else 1
    reshaped = jnp.reshape(moved, (flat_leading, n))
    u_hat = jnp.fft.fft(reshaped, axis=-1)
    du_hat = u_hat * factor[None, :]
    du = jnp.fft.ifft(du_hat, axis=-1).real
    du = jnp.reshape(du, moved.shape)
    return jnp.moveaxis(du, -1, axis)


def _resolve_spatial_1d_axes(
    K: jnp.ndarray,
    cfg: PathBGridConfig,
    x: Optional[jnp.ndarray],
    y: Optional[jnp.ndarray],
    z: Optional[jnp.ndarray],
    dim: int,
    warnings: List[str],
) -> Optional[List[jnp.ndarray]]:
    """Return length-``dim`` list of 1D axis vectors matching ``K``'s spatial shape."""
    if cfg.layout == "separable":
        try:
            return _separable_1d_coords(K.shape, x, y, z, dim)
        except ValueError as e:
            warnings.append(str(e))
            return None

    if dim == 1:
        aligned = _align_1d_field_and_coord(K, x, warnings, "spectral meshgrid 1D")
        if aligned is None:
            return None
        _K1, c1, _orig = aligned
        return [c1]

    rect1d = _rectilinear_meshgrid_1d_axes(K, x, y, z, dim)
    if rect1d is not None:
        return rect1d

    sep_axes = _meshgrid_separable_axis_coords(K, x, y, z, dim)
    if sep_axes is not None:
        return sep_axes

    warnings.append(
        "spectral: need separable/rectilinear coordinates "
        "(1D axis vectors or tensor-product meshgrid); curvilinear grids are unsupported"
    )
    return None


def spectral_grad_along_named_axis(
    K: jnp.ndarray,
    deriv_axis: str,
    cfg: PathBGridConfig,
    x: Optional[jnp.ndarray],
    y: Optional[jnp.ndarray],
    z: Optional[jnp.ndarray],
    dim: int,
    warnings: List[str],
) -> Optional[jnp.ndarray]:
    """First spatial derivative along ``x``/``y``/``z`` for a scalar field slice."""
    validate_spectral_grid_config(cfg)
    axis_index = {"x": 0, "y": 1, "z": 2}[deriv_axis]
    if axis_index >= dim:
        warnings.append(f"skip d/d{deriv_axis}: spatial_dimension {dim} < axis index")
        return None

    K_work = K
    orig_shape: Optional[Tuple[int, ...]] = None
    if dim == 1 and cfg.layout != "separable":
        aligned = _align_1d_field_and_coord(K, x, warnings, "spectral 1D")
        if aligned is None:
            return None
        K_work, c1, orig_shape = aligned
        axes = [c1]
        # After ravel, field is 1D; differentiate axis 0
        try:
            L = period_length_1d(axes[0], int(K_work.shape[0]))
            out = spectral_diff_along_axis(K_work, 0, L, order=1)
        except ValueError as e:
            warnings.append(str(e))
            return None
        return jnp.reshape(out, orig_shape)

    axes = _resolve_spatial_1d_axes(K, cfg, x, y, z, dim, warnings)
    if axes is None:
        return None
    try:
        L = period_length_1d(axes[axis_index], int(K.shape[axis_index]))
        return spectral_diff_along_axis(K, axis_index, L, order=1)
    except ValueError as e:
        warnings.append(str(e))
        return None


def spectral_scalar_laplacian_steady(
    K: jnp.ndarray,
    cfg: PathBGridConfig,
    x: Optional[jnp.ndarray],
    y: Optional[jnp.ndarray],
    z: Optional[jnp.ndarray],
    dim: int,
    warnings: List[str],
) -> Optional[jnp.ndarray]:
    """Sum of second periodic Fourier derivatives along each spatial axis."""
    validate_spectral_grid_config(cfg)

    if dim == 1 and cfg.layout != "separable":
        aligned = _align_1d_field_and_coord(K, x, warnings, "spectral 1D laplacian")
        if aligned is None:
            return None
        K1, c1, orig_shape = aligned
        try:
            L = period_length_1d(c1, int(K1.shape[0]))
            out = spectral_diff_along_axis(K1, 0, L, order=2)
        except ValueError as e:
            warnings.append(str(e))
            return None
        return jnp.reshape(out, orig_shape)

    axes = _resolve_spatial_1d_axes(K, cfg, x, y, z, dim, warnings)
    if axes is None:
        return None
    try:
        acc = jnp.zeros_like(K)
        for i in range(dim):
            L = period_length_1d(axes[i], int(K.shape[i]))
            acc = acc + spectral_diff_along_axis(K, i, L, order=2)
        return acc
    except ValueError as e:
        warnings.append(str(e))
        return None


__all__ = [
    "period_length_1d",
    "spectral_diff_along_axis",
    "spectral_grad_along_named_axis",
    "spectral_scalar_laplacian_steady",
    "validate_spectral_grid_config",
]
