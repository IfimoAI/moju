"""
Torch-native Path-B derivative fill (FD or periodic spectral).

Uses ``torch.gradient`` (PyTorch ≥ 1.11) for finite differences, or ``torch.fft``
for periodic Fourier spatial derivatives when ``diff_method=\"spectral\"``.

Mirrors the intent of ``moju.monitor.path_b_derivatives.fill_path_b_derivatives``.

Key notes:
- Gradients flow through ``torch.gradient`` / FFT outputs (autograd-compatible for FD;
  FFT path uses real iFFT).
- Grid spacing is inferred from coordinate tensors in ``state`` (``x``, ``y``, ``z``).
- Temporal ``*_t`` fill always uses FD along the leading time axis.
- Only the most common derivative fills are implemented: ``*_grad``,
  ``*_laplacian``, ``*_t``.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import torch

# Fields for which we auto-fill spatial grad / laplacian.
# Each entry: (primitive_key, grad_key, laplacian_key)
_SPATIAL_FILL_RECIPES: List[Tuple[str, str, str]] = [
    ("u", "u_grad", "u_laplacian"),
    ("T", "T_grad", "T_laplacian"),
    ("phi", "phi_grad", "phi_laplacian"),
    ("rho", "rho_grad", None),
    ("p", None, None),  # p_grad filled if "p_grad" missing
]

# Time-derivative fill recipes: (primitive_key, time_deriv_key)
_TIME_FILL_RECIPES: List[Tuple[str, str]] = [
    ("u", "u_t"),
    ("T", "T_t"),
    ("phi", "phi_t"),
    ("rho", "rho_t"),
]


def _infer_spacing(coord: torch.Tensor) -> float:
    """Infer grid spacing from a 1-D coordinate tensor (mean of diffs)."""
    if coord.ndim != 1 or coord.numel() < 2:
        return 1.0
    diffs = torch.diff(coord.float())
    spacing = float(diffs.mean())
    if diffs.std() / (abs(spacing) + 1e-30) > 0.05:
        warnings.warn(
            "fill_path_b_derivatives_torch: non-uniform spacing detected; "
            "torch.gradient assumes uniform spacing — results may be inaccurate.",
            UserWarning,
            stacklevel=4,
        )
    return spacing if abs(spacing) > 1e-30 else 1.0


def _uniform_spacing_or_raise(coord: torch.Tensor, *, rtol: float = 1e-4, atol: float = 1e-6) -> float:
    """Return uniform spacing or raise (for spectral)."""
    c = coord.reshape(-1).float()
    if c.numel() < 2:
        raise ValueError("spectral: need at least 2 coordinate points")
    diffs = torch.diff(c)
    if not bool(torch.allclose(diffs, diffs[0], rtol=rtol, atol=atol)):
        raise ValueError(
            "spectral Path B requires uniform spatial spacing; "
            "non-uniform coordinates are not supported (use diff_method='fd' or resample)"
        )
    h = float(diffs[0])
    if abs(h) < 1e-15:
        raise ValueError("spectral: zero grid spacing")
    return h


def _period_length(coord: torch.Tensor, n: int) -> float:
    h = _uniform_spacing_or_raise(coord)
    if int(coord.reshape(-1).numel()) != int(n):
        raise ValueError(
            f"spectral: coordinate length {int(coord.reshape(-1).numel())} != axis length {n}"
        )
    return float(n) * h


def _spectral_diff_along_axis(
    field: torch.Tensor,
    axis: int,
    L: float,
    *,
    order: int = 1,
) -> torch.Tensor:
    if order not in (1, 2):
        raise ValueError(f"spectral order must be 1 or 2, got {order}")
    if axis < 0:
        axis = field.ndim + axis
    n = int(field.shape[axis])
    if n < 2:
        raise ValueError("spectral: need at least 2 points along differentiation axis")
    if not (L > 0.0):
        raise ValueError(f"spectral: period length L must be positive, got {L}")

    k = (2.0 * torch.pi) * torch.fft.fftfreq(n, d=L / float(n), device=field.device, dtype=torch.float64)
    factor = (1j * k.to(torch.complex128)) ** int(order)

    moved = torch.moveaxis(field, axis, -1)
    leading = int(moved[..., 0].numel()) if moved.ndim > 1 else 1
    reshaped = moved.reshape(leading, n).to(torch.complex128)
    u_hat = torch.fft.fft(reshaped, dim=-1)
    du_hat = u_hat * factor.unsqueeze(0)
    du = torch.fft.ifft(du_hat, dim=-1).real.to(dtype=field.dtype)
    du = du.reshape(moved.shape)
    return torch.moveaxis(du, -1, axis)


def _laplacian_torch_fd(f: torch.Tensor, sp: List[float]) -> torch.Tensor:
    """Sum of second spatial derivatives (FD)."""
    lap = torch.zeros_like(f)
    for dim_idx, s in enumerate(sp):
        g = torch.gradient(f, spacing=[s], dim=[dim_idx])[0]
        g2 = torch.gradient(g, spacing=[s], dim=[dim_idx])[0]
        lap = lap + g2
    return lap


def _laplacian_torch_spectral(f: torch.Tensor, coords: List[torch.Tensor]) -> torch.Tensor:
    lap = torch.zeros_like(f)
    for dim_idx, c in enumerate(coords):
        L = _period_length(c.reshape(-1), int(f.shape[dim_idx]))
        lap = lap + _spectral_diff_along_axis(f, dim_idx, L, order=2)
    return lap


def fill_path_b_derivatives_torch(
    state: Dict[str, Any],
    *,
    laws_spec: Sequence[Dict[str, Any]] = (),
    constants: Optional[Dict[str, Any]] = None,
    coord_keys: Tuple[str, ...] = ("x", "y", "z"),
    copy: bool = True,
    diff_method: Literal["fd", "spectral"] = "fd",
    periodic: bool = False,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Fill missing derivative keys in *state* using FD or periodic spectral spatial ops.

    Only keys already absent from *state* are filled; existing keys are never
    overwritten. Temporal ``*_t`` always uses FD.

    Parameters
    ----------
    state:
        State dict containing primitive fields and (optionally) coordinate
        tensors.  Values may be ``torch.Tensor`` or array-like.
    laws_spec:
        Law specs (used for future recipe expansion; currently ignored).
    constants:
        Engine constants (not used for derivative fill).
    coord_keys:
        Names of spatial coordinate tensors in *state* (default ``x``, ``y``, ``z``).
    copy:
        If ``True`` (default), return a new dict; otherwise mutate in-place.
    diff_method:
        ``\"fd\"`` (default) or ``\"spectral\"`` (requires ``periodic=True``).
    periodic:
        Required ``True`` when ``diff_method=\"spectral\"``.

    Returns
    -------
    (new_state, warnings)
    """
    if diff_method == "spectral" and not periodic:
        raise ValueError(
            "diff_method='spectral' requires periodic=True "
            "(periodic Fourier differentiation on a structured grid)"
        )
    if diff_method not in ("fd", "spectral"):
        raise ValueError(f"diff_method must be 'fd' or 'spectral', got {diff_method!r}")

    out: Dict[str, Any] = dict(state) if copy else state
    warn_list: List[str] = []
    _ = laws_spec, constants  # API parity with JAX fill

    # Collect coordinate tensors and their spacings
    coords: List[torch.Tensor] = []
    spacings: List[float] = []
    active_coord_keys: List[str] = []
    for ck in coord_keys:
        if ck in out:
            c = torch.as_tensor(out[ck], dtype=torch.float32)
            if c.ndim >= 1:
                coords.append(c)
                if diff_method == "fd":
                    spacings.append(_infer_spacing(c.reshape(-1)))
                else:
                    try:
                        spacings.append(_uniform_spacing_or_raise(c.reshape(-1)))
                    except ValueError as exc:
                        warn_list.append(f"fill_path_b_derivatives_torch: {ck}: {exc}")
                        return out, warn_list
                active_coord_keys.append(ck)

    n_spatial = len(coords)

    def _grad_and_laplacian(
        field: torch.Tensor,
        grad_key: Optional[str],
        lap_key: Optional[str],
        field_name: str,
    ) -> None:
        """Compute and store spatial gradient / laplacian for *field*."""
        if n_spatial == 0:
            warn_list.append(
                f"fill_path_b_derivatives_torch: no coordinate tensors found; "
                f"cannot fill {field_name} derivatives"
            )
            return

        f = torch.as_tensor(field, dtype=torch.float32)

        # For vector fields (u), handle each component separately.
        # f shape: (..., d) for vector, (...) for scalar.
        is_vector = f.ndim > 0 and (f.shape[-1] == n_spatial or n_spatial == 1)

        if grad_key is not None and grad_key not in out:
            try:
                if is_vector and f.ndim > 1:
                    grad_components = []
                    for comp_idx in range(f.shape[-1]):
                        comp = f[..., comp_idx]
                        if diff_method == "spectral":
                            g_parts = []
                            for dim_idx, c in enumerate(coords):
                                L = _period_length(c.reshape(-1), int(comp.shape[dim_idx]))
                                g_parts.append(_spectral_diff_along_axis(comp, dim_idx, L, order=1))
                            grad_components.append(torch.stack(g_parts, dim=-1))
                        else:
                            g_parts = torch.gradient(
                                comp, spacing=spacings, dim=list(range(n_spatial))
                            )
                            grad_components.append(torch.stack(g_parts, dim=-1))
                    out[grad_key] = torch.stack(grad_components, dim=-2)
                else:
                    if diff_method == "spectral":
                        g_parts = []
                        for dim_idx, c in enumerate(coords):
                            L = _period_length(c.reshape(-1), int(f.shape[dim_idx]))
                            g_parts.append(_spectral_diff_along_axis(f, dim_idx, L, order=1))
                        out[grad_key] = torch.stack(g_parts, dim=-1)
                    else:
                        g_parts = torch.gradient(f, spacing=spacings, dim=list(range(n_spatial)))
                        out[grad_key] = torch.stack(g_parts, dim=-1)
            except Exception as exc:  # noqa: BLE001
                warn_list.append(f"fill_path_b_derivatives_torch: {grad_key}: {exc}")

        if lap_key is not None and lap_key not in out:
            try:
                if is_vector and f.ndim > 1:
                    lap_components = []
                    for comp_idx in range(f.shape[-1]):
                        comp = f[..., comp_idx]
                        if diff_method == "spectral":
                            lap = _laplacian_torch_spectral(comp, coords)
                        else:
                            lap = _laplacian_torch_fd(comp, spacings)
                        lap_components.append(lap)
                    out[lap_key] = torch.stack(lap_components, dim=-1)
                else:
                    if diff_method == "spectral":
                        out[lap_key] = _laplacian_torch_spectral(f, coords)
                    else:
                        out[lap_key] = _laplacian_torch_fd(f, spacings)
            except Exception as exc:  # noqa: BLE001
                warn_list.append(f"fill_path_b_derivatives_torch: {lap_key}: {exc}")

    # Fill pressure gradient (scalar)
    if "p" in out and "p_grad" not in out:
        _grad_and_laplacian(out["p"], "p_grad", None, "p")

    # Fill spatial derivatives for registered fields
    for prim_key, grad_key, lap_key in _SPATIAL_FILL_RECIPES:
        if prim_key in out:
            _grad_and_laplacian(out[prim_key], grad_key, lap_key, prim_key)

    # Fill time derivatives using coordinate 't' (always FD)
    if "t" in out:
        t_tensor = torch.as_tensor(out["t"], dtype=torch.float32)
        t_spacing = _infer_spacing(t_tensor.reshape(-1))
        for prim_key, t_key in _TIME_FILL_RECIPES:
            if prim_key in out and t_key not in out:
                f = torch.as_tensor(out[prim_key], dtype=torch.float32)
                try:
                    dt_parts = torch.gradient(f, spacing=[t_spacing], dim=[0])
                    out[t_key] = dt_parts[0]
                except Exception as exc:  # noqa: BLE001
                    warn_list.append(
                        f"fill_path_b_derivatives_torch: {t_key}: {exc}"
                    )

    return out, warn_list
