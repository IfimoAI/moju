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


def estimate_dual_spacing_on_axis(
    fine_deriv: Any,
    coarse_deriv: Any,
    fine_index: Any,
    *,
    order: Optional[int] = None,
    axis: int = -1,
) -> Dict[str, float]:
    """
    Compare a fine-grid derivative to a coarse (every-other-point) derivative.

    ``coarse_deriv`` lives on the subsampled axis; ``fine_index`` maps those samples
    onto the fine axis. Comparison uses the fine values at ``fine_index``.
    """
    f = jnp.moveaxis(jnp.asarray(fine_deriv), axis, -1)
    c = jnp.moveaxis(jnp.asarray(coarse_deriv), axis, -1)
    idx = jnp.asarray(fine_index)
    f_at = f[..., idx]
    if f_at.shape != c.shape:
        raise ValueError(
            f"dual-spacing shape mismatch fine_at={f_at.shape} coarse={c.shape}"
        )
    return estimate_from_pair(f_at, c, order=order)


def _subsample_like_mesh(arr: Any, ndim: int) -> Any:
    a = jnp.asarray(arr)
    slices = tuple(slice(None, None, 2) for _ in range(ndim))
    if a.ndim > ndim:
        return a[slices]
    return a[slices]


def estimate_errors_dual_spacing(
    fine_state: Mapping[str, Any],
    coarse_state: Mapping[str, Any],
    *,
    keys: Optional[Sequence[str]] = None,
    spatial_ndim: int = 2,
    order: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Per-key dual-spacing errors: compare fine[::2, ::2, ...] to coarse derivatives.

    ``spatial_ndim`` is the number of leading axes subsampled (meshgrid convention).
    """
    deriv_suffixes = ("_t", "_tt", "_grad", "_laplacian")
    if keys is None:
        keys = sorted(
            k
            for k in set(fine_state) & set(coarse_state)
            if any(str(k).endswith(s) for s in deriv_suffixes)
        )
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        if fine_state.get(k) is None or coarse_state.get(k) is None:
            continue
        try:
            fine_sub = _subsample_like_mesh(fine_state[k], spatial_ndim)
            coarse = jnp.asarray(coarse_state[k])
            if fine_sub.shape != coarse.shape:
                # Try 1D subsample on axis 0
                fine_sub = jnp.asarray(fine_state[k])[::2]
            if fine_sub.shape != coarse.shape:
                continue
            out[str(k)] = estimate_from_pair(fine_sub, coarse, order=order)
        except Exception:  # noqa: BLE001
            continue
    return out


def fill_dual_spacing_alternate(
    state_pred: Mapping[str, Any],
    *,
    laws_spec: Sequence[Mapping[str, Any]],
    constants: Optional[Mapping[str, Any]] = None,
    grid: Any = None,
    fill_law_recipes: bool = True,
) -> Dict[str, Any]:
    """
    Path B fill on an every-other-point spatial subsample of ``state_pred``.

    Used only for diagnostics; does not replace the primary fill.
    """
    from dataclasses import replace

    from moju.monitor.path_b_derivatives import PathBGridConfig, fill_path_b_derivatives

    cfg: PathBGridConfig = grid or PathBGridConfig()
    coarse_state: Dict[str, Any] = {}
    xk = cfg.key_x
    xref = jnp.asarray(state_pred[xk]) if xk in state_pred and state_pred[xk] is not None else None
    spatial_ndim = int(xref.ndim) if xref is not None else 1
    if xref is not None and xref.ndim == 1:
        spatial_ndim = 1
    elif xref is not None:
        spatial_ndim = int(xref.ndim)

    deriv_suffixes = ("_t", "_tt", "_grad", "_laplacian")
    for k, v in state_pred.items():
        if v is None or any(str(k).endswith(s) for s in deriv_suffixes):
            continue
        arr = jnp.asarray(v)
        if xref is not None and arr.shape[: xref.ndim] == xref.shape:
            coarse_state[k] = _subsample_like_mesh(arr, xref.ndim)
        elif arr.ndim == 1 and int(arr.shape[0]) >= 4:
            coarse_state[k] = arr[::2]
        else:
            coarse_state[k] = v

    filled, _, _ = fill_path_b_derivatives(
        coarse_state,
        laws_spec=list(laws_spec),
        constants=dict(constants or {}),
        grid=replace(cfg, estimate_derivative_error=False),
        copy=True,
        fill_law_recipes=fill_law_recipes,
        return_provenance=True,
    )
    out = dict(filled)
    out["_dual_spacing_meta"] = {"spatial_ndim": spatial_ndim}
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
