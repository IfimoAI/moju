"""
Opt-in robust scoring for law-linked ``implied_delta`` rows (Moju 2.2).

Defaults match Moju 2.1: ``implied_metric="worst_point"`` and
``implied_noise_normalization=False``. Governing-law scoring is unaffected.

Noise normalization (when enabled)
----------------------------------
For a property that enters the local balance linearly,
``lhs ≈ F · basis`` (ratio recovery) or ``rhs ≈ β(F) · basis`` (LSQ viscosity):

1. Balance residual with the **catalog / model** property ``F``::

       r_balance = lhs - F · basis     # ratio (e.g. T_t − α_F ∇²T)
       r_balance = rhs - β_F · basis   # LSQ (vector; take last-axis norm)

2. Property-unit noise level::

       ε(x) = |r_balance(x)| / (|basis(x)| + ε_m)

   For ratio recovery this equals ``|F − implied|`` when
   ``implied = lhs / basis``.

3. Noise-normalized fractional residual (element-wise)::

       δ_nn(x) = (F − implied) / (|F| + ε(x) + ε_norm)

   with the same ``ε_norm = 1e-30`` as ordinary ``implied_delta``.
   When ``ε → 0``, ``δ_nn`` reduces to the 2.1 fractional ``δ``.
   When ``|F − implied| ≈ ε`` (deviation explained by residual noise),
   ``|δ_nn|`` is softened relative to raw ``δ``.

Aggregation (``implied_metric``)
--------------------------------
Applied to the scoring field (``δ`` or ``δ_nn``), finite points only:

- ``"worst_point"`` (default): ``nanmax |·|``
- ``"percentile"``: ``nanpercentile(|·|, implied_percentile)`` (default **95**)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

import jax.numpy as jnp

from moju.monitor.closure_registry import _NORMALIZE_EPS

DEFAULT_IMPLIED_METRIC = "worst_point"
DEFAULT_IMPLIED_PERCENTILE = 95.0
_VALID_METRICS = frozenset({"worst_point", "percentile"})


def parse_implied_scoring_options(constants: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Read opt-in implied scoring knobs from engine ``constants`` (defaults = 2.1)."""
    c = dict(constants or {})
    metric = str(c.get("implied_metric", DEFAULT_IMPLIED_METRIC)).strip().lower()
    if metric not in _VALID_METRICS:
        raise ValueError(
            f"implied_metric must be one of {sorted(_VALID_METRICS)}, got {metric!r}"
        )
    pct = float(c.get("implied_percentile", DEFAULT_IMPLIED_PERCENTILE))
    if not (0.0 < pct <= 100.0):
        raise ValueError(f"implied_percentile must be in (0, 100], got {pct!r}")
    noise = bool(c.get("implied_noise_normalization", False))
    return {
        "implied_metric": metric,
        "implied_percentile": pct,
        "implied_noise_normalization": noise,
    }


def _vec_norm_last(x: Any) -> jnp.ndarray:
    a = jnp.asarray(x)
    if a.ndim >= 1 and a.shape[-1] in (2, 3) and a.ndim >= 2:
        return jnp.sqrt(jnp.sum(jnp.square(a), axis=-1) + 1e-30)
    return jnp.abs(a)


def property_noise_level(
    *,
    pred: Any,
    implied: Any,
    raw: Any,
    debug: Optional[Mapping[str, Any]] = None,
) -> jnp.ndarray:
    """
    Pointwise property-unit noise ``ε(x) = |r_balance| / (|basis| + ε_m)``.

    Prefer LSQ sidecars ``noise_rhs`` / ``noise_basis`` (and optional ``noise_scale``)
    from viscosity recovery; otherwise use the ratio identity ``ε = |raw|``.
    """
    dbg = dict(debug or {})
    basis = dbg.get("noise_basis")
    rhs = dbg.get("noise_rhs")
    if basis is not None and rhs is not None:
        basis_a = jnp.asarray(basis)
        rhs_a = jnp.asarray(rhs)
        pred_a = jnp.asarray(pred)
        scale = dbg.get("noise_scale")
        # pred is catalog property in model units (μ or ν); convert to β on the basis.
        if scale is not None:
            beta_f = pred_a / (jnp.asarray(scale) + 1e-30)
        else:
            beta_f = pred_a
        if beta_f.ndim == basis_a.ndim - 1:
            model_term = beta_f[..., None] * basis_a
        else:
            model_term = beta_f * basis_a
        r_bal = _vec_norm_last(rhs_a - model_term)
        bmag = _vec_norm_last(basis_a)
        eps = r_bal / (bmag + 1e-30)
        if scale is not None:
            eps = eps * jnp.abs(jnp.asarray(scale))
        return eps
    return jnp.abs(jnp.asarray(raw))


def noise_normalized_delta(
    *,
    pred: Any,
    implied: Any,
    raw: Any,
    debug: Optional[Mapping[str, Any]] = None,
) -> jnp.ndarray:
    """Element-wise ``δ_nn = raw / (|F| + ε + ε_norm)``."""
    pred_a = jnp.asarray(pred)
    raw_a = jnp.asarray(raw)
    eps = property_noise_level(pred=pred_a, implied=implied, raw=raw_a, debug=debug)
    try:
        pred_b, eps_b = jnp.broadcast_arrays(pred_a, eps)
        raw_b, _ = jnp.broadcast_arrays(raw_a, eps_b)
    except (TypeError, ValueError):
        pred_b, eps_b, raw_b = pred_a, eps, raw_a
    return raw_b / (jnp.abs(pred_b) + jnp.abs(eps_b) + _NORMALIZE_EPS)


def reduce_implied_magnitude(
    arr: Any,
    *,
    metric: str = DEFAULT_IMPLIED_METRIC,
    percentile: float = DEFAULT_IMPLIED_PERCENTILE,
) -> float:
    """Aggregate a residual field to a scalar magnitude for admissibility."""
    a = jnp.asarray(arr)
    if a.ndim >= 1 and a.shape[-1] in (2, 3) and a.ndim >= 2:
        mag = jnp.sqrt(jnp.sum(jnp.square(a), axis=-1) + 1e-30)
    else:
        mag = jnp.abs(a)
    flat = jnp.reshape(mag, (-1,))
    finite = flat[jnp.isfinite(flat)]
    if finite.size == 0:
        return float("nan")
    if metric == "percentile":
        return float(jnp.percentile(finite, float(percentile)))
    return float(jnp.max(finite))


def score_implied_delta_array(
    delta: Any,
    debug: Optional[Mapping[str, Any]],
    options: Mapping[str, Any],
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Return ``(v_eff, r_max_raw, meta)`` for one implied_delta residual array.

    ``r_max_raw`` is always the unnormalized worst-point ``max |δ|``.
    ``v_eff`` uses the configured metric on ``δ`` or ``δ_nn``.
    """
    d = jnp.asarray(delta)
    r_max_raw = reduce_implied_magnitude(d, metric="worst_point")
    use_noise = bool(options.get("implied_noise_normalization", False))
    metric = str(options.get("implied_metric", DEFAULT_IMPLIED_METRIC))
    pct = float(options.get("implied_percentile", DEFAULT_IMPLIED_PERCENTILE))
    scored = d
    if use_noise and isinstance(debug, Mapping):
        pred = debug.get("pred")
        implied = debug.get("implied")
        raw = debug.get("raw")
        if pred is not None and raw is not None:
            scored = noise_normalized_delta(
                pred=pred, implied=implied, raw=raw, debug=debug
            )
    v_eff = reduce_implied_magnitude(scored, metric=metric, percentile=pct)
    label = "max" if metric == "worst_point" else f"p{pct:g}"
    if use_noise:
        label = f"noise_norm+{label}"
    meta = {
        "implied_metric": metric,
        "implied_percentile": pct if metric == "percentile" else None,
        "implied_noise_normalization": use_noise,
        "admissibility_metric": label,
        "r_max_raw": r_max_raw,
        "v_eff": v_eff,
    }
    return v_eff, r_max_raw, meta


def is_default_implied_scoring(options: Mapping[str, Any]) -> bool:
    """True when options match Moju 2.1 behaviour."""
    return (
        str(options.get("implied_metric", DEFAULT_IMPLIED_METRIC)) == DEFAULT_IMPLIED_METRIC
        and not bool(options.get("implied_noise_normalization", False))
    )


__all__ = [
    "DEFAULT_IMPLIED_METRIC",
    "DEFAULT_IMPLIED_PERCENTILE",
    "parse_implied_scoring_options",
    "property_noise_level",
    "noise_normalized_delta",
    "reduce_implied_magnitude",
    "score_implied_delta_array",
    "is_default_implied_scoring",
]
