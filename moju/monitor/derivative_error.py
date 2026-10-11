"""
Opt-in derivative-error estimation (diagnostics only; does not change tiers).

Estimate the local truncation / differencing error by comparing two approximations
(two spacings or two stencil orders). Flag a law when the estimated derivative
contribution to the residual is comparable to the residual itself.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

import jax.numpy as jnp

from moju.monitor.law_scale_recipes import _rms_mag

# Default: flag when estimated derivative error RMS >= this fraction of residual RMS.
DEFAULT_DERIVATIVE_ERROR_RATIO = 0.5


def estimate_from_pair(
    fine: Any,
    coarse: Any,
    *,
    order: Optional[int] = None,
) -> Dict[str, float]:
    """
    Estimate absolute error from a fine/coarse pair.

    Without ``order``, uses ``|fine - coarse|`` (Richardson step when spacings
    differ by 2× and the method order is unknown). With ``order``, applies the
    classical Richardson factor ``|f - c| / (2^p - 1)``.
    """
    f = jnp.asarray(fine)
    c = jnp.asarray(coarse)
    if f.shape != c.shape:
        raise ValueError(
            f"estimate_from_pair: shape mismatch fine={f.shape} coarse={c.shape}"
        )
    diff = jnp.abs(f - c)
    if order is not None and int(order) > 0:
        denom = float(2 ** int(order) - 1)
        if denom <= 0:
            denom = 1.0
        err = diff / denom
    else:
        err = diff
    return {
        "error_rms": float(_rms_mag(err)),
        "error_max": float(jnp.max(err)) if err.size else float("nan"),
    }


def estimate_derivative_errors(
    primary: Mapping[str, Any],
    secondary: Mapping[str, Any],
    *,
    keys: Optional[Sequence[str]] = None,
    order: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Per-key error estimates for keys present in both ``primary`` and ``secondary``.

    ``secondary`` is typically a coarser-spacing or lower-order recomputation, or a
    user-supplied alternate derivative set (``state_pred_alt``).
    """
    if keys is None:
        keys = sorted(set(primary) & set(secondary))
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        if k not in primary or k not in secondary:
            continue
        if primary[k] is None or secondary[k] is None:
            continue
        try:
            out[str(k)] = estimate_from_pair(primary[k], secondary[k], order=order)
        except ValueError:
            continue
    return out


def law_derivative_error_contribution(
    law_name: str,
    residual: Any,
    error_by_key: Mapping[str, Mapping[str, float]],
    state_map: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Combine per-key derivative errors that feed a law into a single contribution estimate.

    Uses the max error_rms among derivative keys appearing in the law's ``state_map``.
    This is a conservative diagnostic, not a rigorous residual uncertainty.
    """
    from moju.monitor.derivative_provenance import is_derivative_key

    contrib_keys = []
    err_vals = []
    for arg, key in state_map.items():
        k = str(key)
        if k not in error_by_key:
            continue
        if not is_derivative_key(str(law_name), str(arg), k):
            continue
        e = error_by_key[k].get("error_rms")
        if e is not None and math.isfinite(float(e)):
            contrib_keys.append(k)
            err_vals.append(float(e))
    resid_rms = float(_rms_mag(residual))
    est = max(err_vals) if err_vals else float("nan")
    ratio = (
        float(est / resid_rms)
        if math.isfinite(est) and math.isfinite(resid_rms) and resid_rms > 0
        else float("nan")
    )
    return {
        "estimated_derivative_error_rms": est,
        "residual_rms": resid_rms,
        "error_over_residual": ratio,
        "keys": contrib_keys,
    }


def flag_derivative_dominated(
    contribution: Mapping[str, Any],
    *,
    threshold: float = DEFAULT_DERIVATIVE_ERROR_RATIO,
) -> bool:
    """True when estimated derivative error is comparable to the residual (ratio >= threshold)."""
    ratio = contribution.get("error_over_residual")
    if ratio is None or not math.isfinite(float(ratio)):
        return False
    return float(ratio) >= float(threshold)


def build_derivative_error_report(
    laws_spec: Sequence[Mapping[str, Any]],
    residuals_laws: Mapping[str, Any],
    error_by_key: Mapping[str, Mapping[str, float]],
    *,
    threshold: float = DEFAULT_DERIVATIVE_ERROR_RATIO,
) -> Dict[str, Any]:
    """
    Per-law derivative-error diagnostics and flags.

    Return shape::

        {
          "per_key": {...},
          "per_law": {
             law: {..., "residual_may_be_dominated_by_derivative_error": bool}
          },
          "threshold": threshold,
        }
    """
    per_law: Dict[str, Any] = {}
    for spec in laws_spec:
        name = str(spec.get("name") or "")
        if name not in residuals_laws:
            continue
        sm = spec.get("state_map") or {}
        contrib = law_derivative_error_contribution(
            name, residuals_laws[name], error_by_key, sm
        )
        flagged = flag_derivative_dominated(contrib, threshold=threshold)
        per_law[name] = {
            **contrib,
            "residual_may_be_dominated_by_derivative_error": flagged,
        }
    return {
        "per_key": dict(error_by_key),
        "per_law": per_law,
        "threshold": float(threshold),
    }
