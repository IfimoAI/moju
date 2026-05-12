"""
Torch-native dimensional ↔ nondimensional state conversion.

Mirrors ``moju.piratio.nondim`` exactly but applies all scale factors using
native PyTorch arithmetic so that the entire nondimensionalisation step stays
in the autograd tape and gradients flow back through the model.

Reuses ``NondimScales`` and ``_PASSTHROUGH_KEYS`` from ``moju.piratio.nondim``.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Optional, Tuple

from moju.piratio.nondim import NondimScales, _PASSTHROUGH_KEYS

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# ---------------------------------------------------------------------------
# Scale factor table  (mirrors moju.piratio.nondim._FIELD_SCALE_RULES)
# Each entry: (fwd_fn, inv_fn) where fn(v: Tensor, s: NondimScales) -> Tensor.
# ---------------------------------------------------------------------------

_TorchRuleFn = Callable[[Any, NondimScales], Any]
_TorchRule = Tuple[_TorchRuleFn, _TorchRuleFn]


def _mul(factor_fn: Callable[[NondimScales], float]) -> _TorchRule:
    def fwd(v: Any, s: NondimScales) -> Any:
        return v * factor_fn(s)

    def inv(v: Any, s: NondimScales) -> Any:
        return v / factor_fn(s)

    return fwd, inv


def _affine(fwd_fn: _TorchRuleFn, inv_fn: _TorchRuleFn) -> _TorchRule:
    return fwd_fn, inv_fn


_FIELD_SCALE_RULES_TORCH: Dict[str, _TorchRule] = {
    # Spatial coordinates
    "x": _mul(lambda s: 1.0 / s.L_ref),
    "y": _mul(lambda s: 1.0 / s.L_ref),
    "z": _mul(lambda s: 1.0 / s.L_ref),
    # Time
    "t": _mul(lambda s: 1.0 / s.t_ref),
    # Velocity
    "u": _mul(lambda s: 1.0 / s.U_ref),
    "v": _mul(lambda s: 1.0 / s.U_ref),
    "w": _mul(lambda s: 1.0 / s.U_ref),
    "u_t": _mul(lambda s: s.t_ref / s.U_ref),
    "u_grad": _mul(lambda s: s.L_ref / s.U_ref),
    "u_laplacian": _mul(lambda s: s.L_ref ** 2 / s.U_ref),
    # Pressure
    "p": _mul(lambda s: 1.0 / s._p_ref),
    "p_grad": _mul(lambda s: s.L_ref / s._p_ref),
    # Density
    "rho": _mul(lambda s: 1.0 / s.rho_ref),
    "rho_t": _mul(lambda s: s.t_ref / s.rho_ref),
    "rho_grad": _mul(lambda s: s.L_ref / s.rho_ref),
    # Temperature (affine)
    "T": _affine(
        fwd_fn=lambda v, s: (v - s.T0) / s.dT_ref,
        inv_fn=lambda v, s: v * s.dT_ref + s.T0,
    ),
    "T_t": _mul(lambda s: s.t_ref / s.dT_ref),
    "T_grad": _mul(lambda s: s.L_ref / s.dT_ref),
    "T_laplacian": _mul(lambda s: s.L_ref ** 2 / s.dT_ref),
    # Generic scalar
    "phi": _mul(lambda s: 1.0 / s.phi_ref),
    "phi_t": _mul(lambda s: s.t_ref / s.phi_ref),
    "phi_grad": _mul(lambda s: s.L_ref / s.phi_ref),
    "phi_laplacian": _mul(lambda s: s.L_ref ** 2 / s.phi_ref),
    "phi_tt": _mul(lambda s: s.t_ref ** 2 / s.phi_ref),
    # Schrödinger
    "psi_laplacian": _mul(lambda s: s.L_ref ** 2),
    # Solid mechanics
    "stress": _mul(lambda s: 1.0 / s.E_ref),
    "stiffness_tensor": _mul(lambda s: 1.0 / s.E_ref),
    "strain": _mul(lambda s: 1.0),
    # Turbulence
    "nu_eff": _mul(lambda s: 1.0 / (s.U_ref * s.L_ref)),
    "nu_molecular": _mul(lambda s: 1.0 / (s.U_ref * s.L_ref)),
    "strain_rate_magnitude": _mul(lambda s: s.L_ref / s.U_ref),
    "Delta": _mul(lambda s: 1.0 / s.L_ref),
}


def _build_effective_rules_torch(
    extra_rules: Optional[Dict[str, Any]],
) -> Dict[str, _TorchRule]:
    effective: Dict[str, _TorchRule] = dict(_FIELD_SCALE_RULES_TORCH)
    if not extra_rules:
        return effective
    for key, rule in extra_rules.items():
        if callable(rule):
            effective[key] = (rule, None)  # type: ignore[assignment]
        else:
            factor = float(rule)
            effective[key] = _mul(lambda s, f=factor: f)
    return effective


def dimensional_to_nd_torch(
    state: Dict[str, Any],
    scales: NondimScales,
    *,
    extra_rules: Optional[Dict[str, Any]] = None,
    warn_unknown: bool = True,
) -> Dict[str, Any]:
    """
    Torch-native dimensional → nondimensional state conversion.

    Identical semantics to :func:`moju.piratio.dimensional_to_nd` but operates
    on ``torch.Tensor`` values using pure Python-float arithmetic so all
    operations remain on the autograd tape.

    Parameters
    ----------
    state:
        Dict of field names → ``torch.Tensor`` (or Python scalar).
    scales:
        Reference scales (reuses :class:`moju.piratio.NondimScales`).
    extra_rules:
        Per-key overrides: float → multiplicative scale; callable
        ``fn(value, scales) → tensor`` → arbitrary forward transform.
    warn_unknown:
        Warn on unrecognised keys (default ``True``).

    Returns
    -------
    Dict[str, Any]
        New dict with nondimensional values.  Gradients flow through all
        multiplicative and affine transformations.
    """
    effective_rules = _build_effective_rules_torch(extra_rules)
    out: Dict[str, Any] = {}
    for key, value in state.items():
        if key in _PASSTHROUGH_KEYS:
            out[key] = value
        elif key in effective_rules:
            fwd_fn, _ = effective_rules[key]
            out[key] = fwd_fn(value, scales)
        else:
            if warn_unknown:
                warnings.warn(
                    f"dimensional_to_nd_torch: unrecognised key {key!r} — copied "
                    f"unchanged. Supply extra_rules={{'{key}': scale}} to scale it, "
                    f"or pass warn_unknown=False to suppress this warning.",
                    UserWarning,
                    stacklevel=2,
                )
            out[key] = value
    return out


def nd_to_dimensional_torch(
    state_nd: Dict[str, Any],
    scales: NondimScales,
    *,
    extra_rules: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Exact inverse of :func:`dimensional_to_nd_torch`.

    Callable extra-rules are not invertible and are copied unchanged.
    """
    effective_rules = _build_effective_rules_torch(extra_rules)
    out: Dict[str, Any] = {}
    for key, value in state_nd.items():
        if key in _PASSTHROUGH_KEYS:
            out[key] = value
        elif key in effective_rules:
            _, inv_fn = effective_rules[key]
            if inv_fn is None:
                out[key] = value
            else:
                out[key] = inv_fn(value, scales)
        else:
            out[key] = value
    return out
