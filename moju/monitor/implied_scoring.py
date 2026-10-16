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
Per-call keyword argument > constructor keyword argument > ``constants`` >
module default. Any conflict between two sources that set the same option to
different values raises a :class:`UserWarning` naming both sources and the
value used. The ``constants`` route remains supported (no deprecation in 2.2).
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


def _warn_implied_conflict(
    key: str,
    *,
    winner_source: str,
    winner_value: Any,
    loser_source: str,
    loser_value: Any,
) -> None:
    warnings.warn(
        f"{key}: {winner_source} value {winner_value!r} overrides {loser_source} value "
        f"{loser_value!r} (using {winner_value!r}).",
        UserWarning,
        stacklevel=3,
    )


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

        per-call kwarg > constructor kwarg > constants > module default

    A :class:`UserWarning` is emitted for every conflict between two sources that
    both set the same option to different values. The higher-precedence value is
    used; the warning names both sources and the value chosen.
    """
    c = dict(constants or {})
    eng = dict(engine_defaults or {})
    const_present = {k: k in c for k in _OPTION_KEYS}
    eng_present = {
        k: (k in eng and eng[k] is not None) for k in _OPTION_KEYS
    }
    explicit = {
        "implied_metric": implied_metric,
        "implied_percentile": implied_percentile,
        "implied_noise_normalization": implied_noise_normalization,
    }
    call_present = {k: explicit[k] is not None for k in _OPTION_KEYS}

    const_vals = {
        "implied_metric": (
            _validate_metric(c["implied_metric"]) if const_present["implied_metric"] else None
        ),
        "implied_percentile": (
            _validate_percentile(c["implied_percentile"])
            if const_present["implied_percentile"]
            else None
        ),
        "implied_noise_normalization": (
            _validate_noise_flag(c["implied_noise_normalization"])
            if const_present["implied_noise_normalization"]
            else None
        ),
    }
    eng_vals = {
        "implied_metric": (
            _validate_metric(eng["implied_metric"]) if eng_present["implied_metric"] else None
        ),
        "implied_percentile": (
            _validate_percentile(eng["implied_percentile"])
            if eng_present["implied_percentile"]
            else None
        ),
        "implied_noise_normalization": (
            _validate_noise_flag(eng["implied_noise_normalization"])
            if eng_present["implied_noise_normalization"]
            else None
        ),
    }
    call_vals = {
        "implied_metric": (
            _validate_metric(explicit["implied_metric"])
            if call_present["implied_metric"]
            else None
        ),
        "implied_percentile": (
            _validate_percentile(explicit["implied_percentile"])
            if call_present["implied_percentile"]
            else None
        ),
        "implied_noise_normalization": (
            _validate_noise_flag(explicit["implied_noise_normalization"])
            if call_present["implied_noise_normalization"]
            else None
        ),
    }
    module_defaults = {
        "implied_metric": DEFAULT_IMPLIED_METRIC,
        "implied_percentile": DEFAULT_IMPLIED_PERCENTILE,
        "implied_noise_normalization": False,
    }

    resolved: Dict[str, Any] = {}
    for key in _OPTION_KEYS:
        # Precedence: per-call > constructor > constants > module default.
        if call_present[key]:
            winner_src, winner = "per-call kwarg", call_vals[key]
        elif eng_present[key]:
            winner_src, winner = "constructor kwarg", eng_vals[key]
        elif const_present[key]:
            winner_src, winner = "constants", const_vals[key]
        else:
            winner_src, winner = "module default", module_defaults[key]

        if call_present[key] and eng_present[key] and call_vals[key] != eng_vals[key]:
            _warn_implied_conflict(
                key,
                winner_source="per-call kwarg",
                winner_value=call_vals[key],
                loser_source="constructor kwarg",
                loser_value=eng_vals[key],
            )
        if call_present[key] and const_present[key] and call_vals[key] != const_vals[key]:
            _warn_implied_conflict(
                key,
                winner_source="per-call kwarg",
                winner_value=call_vals[key],
                loser_source="constants",
                loser_value=const_vals[key],
            )
        if (
            not call_present[key]
            and eng_present[key]
            and const_present[key]
            and eng_vals[key] != const_vals[key]
        ):
            _warn_implied_conflict(
                key,
                winner_source="constructor kwarg",
                winner_value=eng_vals[key],
                loser_source="constants",
                loser_value=const_vals[key],
            )
        # silence unused when winner is module default
        _ = winner_src
        resolved[key] = winner
    return resolved


def implied_scoring_settings_payload(
    options: Mapping[str, Any],
    *,
    moju_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Compact provenance block for audit / evaluate / export reports."""
    from moju import __version__ as _ver

    metric = str(options.get("implied_metric", DEFAULT_IMPLIED_METRIC))
    return {
        "implied_metric": metric,
        "implied_percentile": (
            float(options.get("implied_percentile", DEFAULT_IMPLIED_PERCENTILE))
            if metric == "percentile"
            else None
        ),
        "implied_noise_normalization": bool(
            options.get("implied_noise_normalization", False)
        ),
        "moju_version": moju_version if moju_version is not None else _ver,
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


def reduce_implied_magnitude_arrays(
    arr: Any,
    *,
    metric: str = DEFAULT_IMPLIED_METRIC,
    percentile: float = DEFAULT_IMPLIED_PERCENTILE,
) -> jnp.ndarray:
    """
    Jit-safe aggregate of a residual field to a scalar magnitude.

    Uses NaN-masking (non-finite → NaN) so ``nanpercentile`` / ``nanmax`` match the
    host ``reduce_implied_magnitude`` path under jit/vmap.
    """
    a = jnp.asarray(arr)
    if a.ndim >= 1 and a.shape[-1] in (2, 3) and a.ndim >= 2:
        mag = jnp.sqrt(jnp.sum(jnp.square(a), axis=-1) + 1e-30)
    else:
        mag = jnp.abs(a)
    flat = jnp.reshape(mag, (-1,))
    scored = jnp.where(jnp.isfinite(flat), flat, jnp.nan)
    has = jnp.any(jnp.isfinite(scored))
    if metric == "percentile":
        v = jnp.nanpercentile(scored, float(percentile))
    else:
        v = jnp.nanmax(scored)
    return jnp.where(has, v, jnp.asarray(jnp.nan, dtype=flat.dtype))


def reduce_implied_magnitude(
    arr: Any,
    *,
    metric: str = DEFAULT_IMPLIED_METRIC,
    percentile: float = DEFAULT_IMPLIED_PERCENTILE,
) -> float:
    """Aggregate a residual field to a scalar magnitude for admissibility."""
    return float(
        reduce_implied_magnitude_arrays(arr, metric=metric, percentile=percentile)
    )


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
    "reduce_implied_magnitude_arrays",

    "DEFAULT_IMPLIED_METRIC",
    "DEFAULT_IMPLIED_PERCENTILE",
    "parse_implied_scoring_options",
    "resolve_implied_scoring_options",
    "implied_scoring_settings_payload",
    "property_noise_level",
    "noise_normalized_delta",
    "reduce_implied_magnitude",
    "score_implied_delta_array",
    "is_default_implied_scoring",
    "score_from_r_max",
]
