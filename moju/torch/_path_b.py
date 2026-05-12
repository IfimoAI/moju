"""
Torch-native Path-B finite-difference derivative fill.

Uses ``torch.gradient`` (PyTorch ≥ 1.11) to compute spatial gradients and
Laplacians from primitive field tensors on a grid, mirroring the intent of
``moju.monitor.path_b_derivatives.fill_path_b_derivatives``.

Key differences vs the JAX version:
- Gradients flow through ``torch.gradient`` outputs (autograd-compatible).
- Grid spacing is inferred from coordinate tensors in ``state`` (``x``, ``y``,
  ``z``).  Uniform spacing is assumed; a warning is emitted otherwise.
- Only the most common derivative fills are implemented: ``*_grad``,
  ``*_laplacian``, ``*_t`` (time derivative via temporal coordinate ``t``).
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def fill_path_b_derivatives_torch(
    state: Dict[str, Any],
    *,
    laws_spec: Sequence[Dict[str, Any]] = (),
    constants: Optional[Dict[str, Any]] = None,
    coord_keys: Tuple[str, ...] = ("x", "y", "z"),
    copy: bool = True,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Fill missing derivative keys in *state* using ``torch.gradient``.

    Only keys already absent from *state* are filled; existing keys are never
    overwritten.

    Parameters
    ----------
    state:
        State dict containing primitive fields and (optionally) coordinate
        tensors.  Values may be ``torch.Tensor`` or array-like.
    laws_spec:
        Law specs (used for future recipe expansion; currently ignored).
    constants:
        Engine constants (not used for FD fill).
    coord_keys:
        Names of spatial coordinate tensors in *state* (default ``x``, ``y``, ``z``).
    copy:
        If ``True`` (default), return a new dict; otherwise mutate in-place.

    Returns
    -------
    (new_state, warnings)
    """
    out: Dict[str, Any] = dict(state) if copy else state
    warn_list: List[str] = []

    # Collect coordinate tensors and their spacings
    coords: List[torch.Tensor] = []
    spacings: List[float] = []
    active_coord_keys: List[str] = []
    for ck in coord_keys:
        if ck in out:
            c = torch.as_tensor(out[ck], dtype=torch.float32)
            if c.ndim >= 1:
                coords.append(c)
                spacings.append(_infer_spacing(c.reshape(-1)))
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
                    # Jacobian: shape (..., d, d_space)
                    grad_components = []
                    for comp_idx in range(f.shape[-1]):
                        comp = f[..., comp_idx]
                        g_parts = torch.gradient(comp, spacing=spacings, dim=list(range(n_spatial)))
                        grad_components.append(torch.stack(g_parts, dim=-1))
                    out[grad_key] = torch.stack(grad_components, dim=-2)
                else:
                    # Scalar: shape (..., d_space)
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
                        lap = _laplacian_torch(comp, spacings)
                        lap_components.append(lap)
                    out[lap_key] = torch.stack(lap_components, dim=-1)
                else:
                    out[lap_key] = _laplacian_torch(f, spacings)
            except Exception as exc:  # noqa: BLE001
                warn_list.append(f"fill_path_b_derivatives_torch: {lap_key}: {exc}")

    def _laplacian_torch(f: torch.Tensor, sp: List[float]) -> torch.Tensor:
        """Sum of second spatial derivatives."""
        lap = torch.zeros_like(f)
        for dim_idx, s in enumerate(sp):
            g = torch.gradient(f, spacing=[s], dim=[dim_idx])[0]
            g2 = torch.gradient(g, spacing=[s], dim=[dim_idx])[0]
            lap = lap + g2
        return lap

    # Fill pressure gradient (scalar)
    if "p" in out and "p_grad" not in out:
        _grad_and_laplacian(out["p"], "p_grad", None, "p")

    # Fill spatial derivatives for registered fields
    for prim_key, grad_key, lap_key in _SPATIAL_FILL_RECIPES:
        if prim_key in out:
            _grad_and_laplacian(out[prim_key], grad_key, lap_key, prim_key)

    # Fill time derivatives using coordinate 't'
    if "t" in out:
        t_tensor = torch.as_tensor(out["t"], dtype=torch.float32)
        t_spacing = _infer_spacing(t_tensor.reshape(-1))
        for prim_key, t_key in _TIME_FILL_RECIPES:
            if prim_key in out and t_key not in out:
                f = torch.as_tensor(out[prim_key], dtype=torch.float32)
                try:
                    # Assume time is along the first dimension
                    dt_parts = torch.gradient(f, spacing=[t_spacing], dim=[0])
                    out[t_key] = dt_parts[0]
                except Exception as exc:  # noqa: BLE001
                    warn_list.append(
                        f"fill_path_b_derivatives_torch: {t_key}: {exc}"
                    )

    return out, warn_list
