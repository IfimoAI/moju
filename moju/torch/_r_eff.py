"""
Torch-native R_eff scalar and build_loss.

Exact port of ``_r_eff_scalar`` and ``build_loss`` from
``moju.monitor.auditor`` so that the training loss computed by
``TorchResidualEngine`` is numerically identical to what the JAX audit reports
under the ``rms`` field.

``R_eff = RMS_δ(r)`` when **p** = 0 (default constant :data:`R_EFF_Q_POWER`), else ``R_eff = RMS_δ(r) · Q^p``.

::

    RMS_δ(r) = sqrt(mean(r²) + δ²)         # δ² = R_EFF_RMS_JITTER_SQ
    m_i      = sqrt(r_i² + ε²)             # ε  = _SCALE_EPS (only if p ≠ 0)
    Q        = RMS(m) / mean(m)             # ≥ 1 when |r| is uneven
    p        = R_EFF_Q_POWER               # set globally via configure_r_eff
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

# Mirror moju.monitor.auditor constants exactly (configure_r_eff updates both).
R_EFF_Q_POWER: float = 0.0
R_EFF_RMS_JITTER_SQ: float = 1e-20
_SCALE_EPS: float = 1e-30


def r_eff_scalar_torch(r: torch.Tensor) -> torch.Tensor:
    """
    Effective residual scalar for a batch of collocation-point residuals.

    Matches :func:`moju.monitor.auditor._r_eff_scalar` exactly.

    Parameters
    ----------
    r:
        Residual tensor (any shape; flattened internally).

    Returns
    -------
    torch.Tensor
        Scalar R_eff (0-d tensor, differentiable).
    """
    a = r.reshape(-1).float()
    if a.numel() == 0:
        return torch.tensor(float("nan"))

    jitter = torch.tensor(R_EFF_RMS_JITTER_SQ, dtype=a.dtype, device=a.device)
    rms_r = torch.sqrt(torch.nanmean(a.square()) + jitter)

    p = float(R_EFF_Q_POWER)
    if p == 0.0 or a.numel() == 1:
        return rms_r

    eps = torch.tensor(_SCALE_EPS, dtype=a.dtype, device=a.device)
    m = torch.sqrt(a.square() + eps * eps)
    rms_m = torch.sqrt(torch.nanmean(m.square()))
    mean_m = torch.nanmean(m)
    q = rms_m / mean_m
    return rms_r * torch.pow(q, p)


def build_loss_torch(
    residual_dict: Dict[str, Any],
    *,
    law_weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """
    Weighted sum of per-law R_eff scalars.

    Matches :func:`moju.monitor.auditor.build_loss` exactly.

    Parameters
    ----------
    residual_dict:
        Output of :meth:`TorchResidualEngine.compute_residuals_torch`.
        Uses the ``"laws"`` sub-dict.
    law_weights:
        Optional per-law weight overrides.  Defaults to ``1/n`` for each
        of the *n* laws.

    Returns
    -------
    torch.Tensor
        Scalar differentiable loss.
    """
    laws = residual_dict.get("laws", {})
    if not laws:
        return torch.tensor(0.0)

    names = list(laws.keys())
    n = len(names)
    weights = law_weights or {}

    total = torch.tensor(0.0)
    for name in names:
        w = weights.get(name, 1.0 / n)
        r_eff = r_eff_scalar_torch(torch.as_tensor(laws[name], dtype=torch.float32))
        total = total + w * r_eff
    return total
