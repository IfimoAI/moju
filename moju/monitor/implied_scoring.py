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

   Interpretation: the deviation is scored relative to ``|F|`` **plus** a
   per-point noise estimate ``ε`` derived from the local balance residual.
   When ``|F − implied| ≈ ε`` (deviation explained by residual noise),
   ``|δ_nn|`` is softened relative to raw ``δ``. When ``ε → 0``, ``δ_nn``
   reduces to the 2.1 fractional ``δ``.

Aggregation (``implied_metric``)
--------------------------------
Applied to the scoring field (``δ`` or ``δ_nn``), finite points only:

- ``"worst_point"`` (default): ``nanmax |·|``
- ``"percentile"``: ``nanpercentile(|·|, implied_percentile)`` (default **95**)

Configuration precedence
------------------------
Explicit keyword argument > ``constants`` > module default.
If an explicit kwarg and ``constants`` both set the same option and disagree,
the kwarg wins and a :class:`UserWarning` is raised. The ``constants`` route
remains supported (no deprecation in 2.2).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Mapping, Optional, Tuple

import jax.numpy as jnp

from moju.monitor.closure_registry import _NORMALIZE_EPS

DEFAULT_IMPLIED_METRIC = "worst_point"
DEFAULT_IMPLIED_PERCENTILE = 95.0
_VALID_METRICS = frozenset({"worst_point", "percentile"})
_OPTION_KEYS = (
    "implied_metric",
    "implied_percentile",
    "implied_noise_normalization",
)


def _validate_metric(metric: Any) -> str:
    if not isinstance(metric, str):
        raise TypeError(
            f"implied_metric must be a str, got {type(metric).__name__}"
        )
    m = metric.strip().lower()
    if m not in _VALID_METRICS:
        raise ValueError(
            f"implied_metric must be one of {sorted(_VALID_METRICS)}, got {metric!r}"
        )
    return m


def _validate_percentile(pct: Any) -> float:
    try:
        p = float(pct)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"implied_percentile must be a real number, got {pct!r}"
        ) from exc
    if not (0.0 < p <= 100.0):
        raise ValueError(f"implied_percentile must be in (0, 100], got {pct!r}")
    return p


def _validate_noise_flag(noise: Any) -> bool:
    if not isinstance(noise, bool):
        raise TypeError(
            f"implied_noise_normalization must be a bool, got {type(noise).__name__}"
        )
    return bool(noise)


def parse_implied_scoring_options(constants: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Read opt-in implied scoring knobs from engine ``constants`` (defaults = 2.1).

    Missing keys fall back to module defaults. Present keys are validated.
    """
    c = dict(constants or {})
    metric = (
        _validate_metric(c["implied_metric"])
        if "implied_metric" in c
        else DEFAULT_IMPLIED_METRIC
    )
    pct = (
        _validate_percentile(c["implied_percentile"])
        if "implied_percentile" in c
        else DEFAULT_IMPLIED_PERCENTILE
    )
    noise = (
        _validate_noise_flag(c["implied_noise_normalization"])
        if "implied_noise_normalization" in c
        else False
    )
    return {
        "implied_metric": metric,
        "implied_percentile": pct,
        "implied_noise_normalization": noise,
    }


def resolve_implied_scoring_options(
    *,
    constants: Optional[Mapping[str, Any]] = None,
    engine_defaults: Optional[Mapping[str, Any]] = None,
    implied_metric: Optional[str] = None,
    implied_percentile: Optional[float] = None,
    implied_noise_normalization: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Resolve implied scoring options with precedence::

        explicit kwarg > constants > engine constructor defaults > module default

    When an explicit kwarg and ``constants`` both provide the same option with
    different values, the kwarg wins and a :class:`UserWarning` is emitted.
    """
    const_opts = parse_implied_scoring_options(constants)
    eng = dict(engine_defaults or {})
    # Start from module defaults, then engine constructor (validated), then constants.
    base = {
        "implied_metric": DEFAULT_IMPLIED_METRIC,
        "implied_percentile": DEFAULT_IMPLIED_PERCENTILE,
        "implied_noise_normalization": False,
    }
    if "implied_metric" in eng and eng["implied_metric"] is not None:
        base["implied_metric"] = _validate_metric(eng["implied_metric"])
    if "implied_percentile" in eng and eng["implied_percentile"] is not None:
        base["implied_percentile"] = _validate_percentile(eng["implied_percentile"])
    if "implied_noise_normalization" in eng and eng["implied_noise_normalization"] is not None:
        base["implied_noise_normalization"] = _validate_noise_flag(
            eng["implied_noise_normalization"]
        )
    # constants override engine defaults (existing 2.2 constants route).
    c = dict(constants or {})
    if "implied_metric" in c:
        base["implied_metric"] = const_opts["implied_metric"]
    if "implied_percentile" in c:
        base["implied_percentile"] = const_opts["implied_percentile"]
    if "implied_noise_normalization" in c:
        base["implied_noise_normalization"] = const_opts["implied_noise_normalization"]

    explicit = {
        "implied_metric": implied_metric,
        "implied_percentile": implied_percentile,
        "implied_noise_normalization": implied_noise_normalization,
    }
    for key, val in explicit.items():
        if val is None:
            continue
        if key == "implied_metric":
            resolved = _validate_metric(val)
        elif key == "implied_percentile":
            resolved = _validate_percentile(val)
        else:
            resolved = _validate_noise_flag(val)
        if key in c:
            const_val = const_opts[key]
            if const_val != resolved:
                warnings.warn(
                    f"{key} explicit kwarg {resolved!r} overrides constants value "
                    f"{const_val!r} (kwarg wins).",
                    UserWarning,
                    stacklevel=2,
                )
        base[key] = resolved
    return base


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


def score_from_r_max(r_max: float, *, scale_k: float = 1.0e-2) -> float:
    """Admissibility score ``1 / (1 + r_max / scale_k)`` (nondim implied gauge)."""
    r = float(r_max)
    s = float(scale_k)
    if not (s > 0.0) or r != r:  # NaN check
        return float("nan")
    return 1.0 / (1.0 + r / s)


__all__ = [
    "DEFAULT_IMPLIED_METRIC",
    "DEFAULT_IMPLIED_PERCENTILE",
    "parse_implied_scoring_options",
    "resolve_implied_scoring_options",
    "property_noise_level",
    "noise_normalized_delta",
    "reduce_implied_magnitude",
    "score_implied_delta_array",
    "is_default_implied_scoring",
    "score_from_r_max",
]
