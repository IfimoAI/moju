"""Torch FD stencils mirroring ``moju.monitor.fd_stencils`` (orders 2/4/6 + Fornberg)."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import torch

from moju.monitor.fd_stencils import (
    DEFAULT_FD_ORDER,
    VALID_FD_ORDERS,
    _normalize_periodic_flags,
    effective_fd_order,
    fornberg_weights,
    validate_fd_order,
)

__all__ = [
    "DEFAULT_FD_ORDER",
    "effective_fd_order",
    "validate_fd_order",
    "diff1_uniform_torch",
    "diff2_uniform_torch",
    "diff1_nonuniform_torch",
    "diff2_nonuniform_torch",
    "laplacian_uniform_nd_torch",
]


def _diff_matrix_uniform(
    n: int, h: float, *, order: int, deriv: int, periodic: bool = False
) -> torch.Tensor:
    order = validate_fd_order(order)
    if periodic:
        from moju.monitor.boundary_closures import _periodic_central_matrix

        D = np.asarray(
            _periodic_central_matrix(n, float(h), deriv_order=deriv, order=order),
            dtype=np.float64,
        )
        return torch.as_tensor(D, dtype=torch.float64)

    # Interior half-width from centred stencil; boundary Fornberg width =
    # order + deriv so formal order of the m-th derivative matches ``order``.
    int_half = (order + 1) // 2
    b_width = int(order + deriv)
    D = np.zeros((n, n), dtype=np.float64)
    if deriv == 1:
        if order == 2:
            for i in range(1, n - 1):
                D[i, i - 1] = -0.5 / h
                D[i, i + 1] = 0.5 / h
        elif order == 4:
            c = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / (12.0 * h)
            for i in range(2, n - 2):
                D[i, i - 2 : i + 3] = c
        else:
            c = np.array([-1.0, 9.0, -45.0, 0.0, 45.0, -9.0, 1.0]) / (60.0 * h)
            for i in range(3, n - 3):
                D[i, i - 3 : i + 4] = c
    else:
        if order == 2:
            for i in range(1, n - 1):
                D[i, i - 1] = 1.0 / (h * h)
                D[i, i] = -2.0 / (h * h)
                D[i, i + 1] = 1.0 / (h * h)
        elif order == 4:
            c = np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / (12.0 * h * h)
            for i in range(2, n - 2):
                D[i, i - 2 : i + 3] = c
        else:
            c = np.array([2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0]) / (180.0 * h * h)
            for i in range(3, n - 3):
                D[i, i - 3 : i + 4] = c
    for i in range(int_half):
        idx = np.arange(0, b_width)
        D[i, idx] = fornberg_weights(float(i), idx.astype(float), deriv)[deriv, :] / (
            h**deriv
        )
    for i in range(n - int_half, n):
        idx = np.arange(n - b_width, n)
        D[i, idx] = (
            fornberg_weights(
                float(i - (n - b_width)), (idx - (n - b_width)).astype(float), deriv
            )[deriv, :]
            / (h**deriv)
        )
    return torch.as_tensor(D, dtype=torch.float64)


def _apply_D(values: torch.Tensor, D: torch.Tensor, axis: int) -> torch.Tensor:
    a = torch.moveaxis(values, axis, -1)
    n = int(a.shape[-1])
    flat = a.reshape(-1, n).to(dtype=D.dtype)
    out = flat @ D.T
    return torch.moveaxis(out.reshape(a.shape).to(dtype=values.dtype), -1, axis)


def diff1_uniform_torch(
    values: torch.Tensor,
    h: float,
    *,
    order: int = 4,
    axis: int = -1,
    periodic: bool = False,
) -> torch.Tensor:
    a = torch.as_tensor(values)
    n = int(torch.moveaxis(a, axis, -1).shape[-1])
    D = _diff_matrix_uniform(
        n, float(h), order=order, deriv=1, periodic=periodic
    ).to(device=a.device)
    return _apply_D(a, D, axis)


def diff2_uniform_torch(
    values: torch.Tensor,
    h: float,
    *,
    order: int = 4,
    axis: int = -1,
    periodic: bool = False,
) -> torch.Tensor:
    a = torch.as_tensor(values)
    n = int(torch.moveaxis(a, axis, -1).shape[-1])
    D = _diff_matrix_uniform(
        n, float(h), order=order, deriv=2, periodic=periodic
    ).to(device=a.device)
    return _apply_D(a, D, axis)


def _diff_matrix_nonuniform(coord: torch.Tensor, *, order: int, deriv: int) -> torch.Tensor:
    from moju.monitor.fd_stencils import differentiation_matrix_1d

    # Reuse NumPy/JAX Fornberg builder (host), then to torch
    D_jax = differentiation_matrix_1d(coord.detach().cpu().numpy(), deriv_order=deriv, fd_order=order)
    return torch.as_tensor(np.asarray(D_jax), dtype=torch.float64)


def diff1_nonuniform_torch(
    values: torch.Tensor, coord: torch.Tensor, *, order: int = 4, axis: int = -1
) -> torch.Tensor:
    a = torch.as_tensor(values)
    c = torch.as_tensor(coord).reshape(-1)
    D = _diff_matrix_nonuniform(c, order=order, deriv=1).to(device=a.device)
    return _apply_D(a, D, axis)


def diff2_nonuniform_torch(
    values: torch.Tensor, coord: torch.Tensor, *, order: int = 4, axis: int = -1
) -> torch.Tensor:
    a = torch.as_tensor(values)
    c = torch.as_tensor(coord).reshape(-1)
    D = _diff_matrix_nonuniform(c, order=order, deriv=2).to(device=a.device)
    return _apply_D(a, D, axis)


def laplacian_uniform_nd_torch(
    field: torch.Tensor,
    spacings: Sequence[float],
    *,
    order: int = 4,
    periodic: Any = False,
) -> torch.Tensor:
    f = torch.as_tensor(field)
    flags = _normalize_periodic_flags(periodic, len(spacings))
    acc = torch.zeros_like(f)
    for ax, h in enumerate(spacings):
        acc = acc + diff2_uniform_torch(
            f, float(h), order=order, axis=ax, periodic=flags[ax]
        )
    return acc
