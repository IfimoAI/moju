"""
ResidualEngine: residuals for governing laws, optional groups (dimensionless numbers), constitutive audits, and data comparison in eval.

- compute_residuals(..., state_ref=None, run_mode=\"training\"|\"eval\", ...)
- build_loss: cascaded **R_eff** over laws only (training); **R_eff** matches logged ``rms`` (see below).
- audit / visualize: same metrics (**R_eff**, R_norm, admissibility) for all residual keys;
  visualize builds Plotly training/eval dashboards (optional spatial law panel for x slices).

Constitutive audits are tied to Models.* functions via
**ref_delta** and **implied_delta**.
**implied_delta** and **ref_delta** are always **nondimensional** (see
``moju.monitor.closure_registry.apply_closure_discrepancy_normalize``). Logged ``rms`` per key is
**R_eff** uses **RMS_δ(r)** = √(mean(r²)+δ²) with tiny **δ²** = :data:`R_EFF_RMS_JITTER_SQ` (AD-smooth at **r=0**), times **Q^0.5**; **Q** = RMS(m)/mean(m), **m_i** = sqrt(r_i²+ε²); **Q=1** when magnitudes are
uniform across collocation points (single-point tensors use **Q=1**). **R_norm** = **R_eff**/scale_k.
For **R_norm**, default **scale_k** is **2×10⁻²** (plus ``_SCALE_EPS``) for **laws/** and for nondimensional closure keys;
other **constitutive/** residuals and **data/** use state- or reference-derived scales. Optional ``audit(..., r_ref=...)`` overrides
**scale_k** per key. Per-key **admissibility_score** is ``1 / (1 + R_norm)`` when finite.
:func:`admissibility_level` maps scores in ``[0, 1]`` to four bands (see its docstring). Metrics are
consistency indicators, not certification.
"""

from __future__ import annotations

import datetime
import math
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Union
import inspect

import jax
import jax.numpy as jnp

from moju.piratio.groups import Groups
from moju.piratio.laws import Laws
from moju.monitor.closure_registry import (
    MODEL_FNS,
    compute_implied_delta,
    compute_ref_delta,
)
from moju.monitor.derived_state_chain import all_ref_keys_from_chain, keys_produced_by_chain
from moju.monitor.law_implied_diagnostics import (
    merge_fragment_law_implied_audit_specs,
    merge_law_implied_audit_specs,
    supported_auto_implied_laws_for,
)
from moju.monitor.law_group_inference import (
    build_law_spec_identity,
    implied_group_specs_for_laws,
    merge_implied_groups_first,
)
from moju.monitor.spatial_rnorm_panels import build_spatial_rnorm_panels_from_residuals
from moju.monitor.visualize_labels import pretty_category_name, pretty_residual_key

DEFAULT_VISUALIZE_TITLE_TRAINING = "Physics Admissibility Audit (model training)"
DEFAULT_VISUALIZE_TITLE_EVAL = "Physics Admissibility Audit (model evaluation)"
DEFAULT_VISUALIZE_TITLE_TEST = DEFAULT_VISUALIZE_TITLE_EVAL

# Admissibility score in [0, 1]; "High" is strictly above this threshold.
ADM_HIGH_THRESHOLD = 0.95
# Imbalance factor in **R_eff** = RMS_δ(r)·Q**p** (see :func:`_r_eff_scalar`).
R_EFF_Q_POWER = 0.5
# Jitter inside **RMS_δ(r)** = sqrt(mean(r^2) + δ²) so **R_eff** is smooth in autodiff at r=0.
R_EFF_RMS_JITTER_SQ = 1e-20
# Logged ``scale_k`` for **laws/** and nondimensional **implied_delta** / **ref_delta** (R_norm denominator).
DEFAULT_NONDIM_R_NORM_SCALE_K = 2e-2


def _normalize_visualize_mode(mode: str) -> str:
    """Map legacy ``mode='test'`` to ``'eval'`` (silent); otherwise return ``mode`` unchanged."""
    if mode == "test":
        return "eval"
    return mode


def _visualize_capitalize_first_word(title: str) -> str:
    """Ensure the first word starts with an uppercase letter (rest unchanged)."""
    t = title.strip()
    if not t:
        return t
    parts = t.split(None, 1)
    first = parts[0]
    if not first:
        return t
    head = first[0].upper() + first[1:] if len(first) > 1 else first.upper()
    return head + (" " + parts[1] if len(parts) > 1 else "")


def _resolve_visualize_figure_title(mode: str, figure_title: Optional[str]) -> str:
    """Figure title for :func:`visualize`; non-empty ``figure_title`` overrides mode default."""
    if figure_title is not None and str(figure_title).strip():
        return _visualize_capitalize_first_word(str(figure_title).strip())
    if mode == "training":
        return _visualize_capitalize_first_word(DEFAULT_VISUALIZE_TITLE_TRAINING)
    return _visualize_capitalize_first_word(DEFAULT_VISUALIZE_TITLE_EVAL)


def _state_key_suggests_law_fd_fill(key: str) -> bool:
    """True if missing key may be fillable via fill_law_fd (derived law input)."""
    if key.endswith("_laplacian") or key.endswith("_grad") or key.endswith("_tt"):
        return True
    if key.endswith("_t") and not key.endswith("_tt"):
        return True
    return key == "u_grad"


def _string_placeholder_like(v: Any, *, key: str, arg_name: str) -> bool:
    """Best-effort check for UI placeholders accidentally saved as strings."""
    if not isinstance(v, str):
        return False
    s = v.strip()
    if not s:
        return True
    low = s.lower()
    if low in {"none", "null", "nan", "na"}:
        return True
    return s == key or s == arg_name


def _kwargs_from_state(
    state: Dict[str, Any],
    constants: Dict[str, Any],
    state_map: Dict[str, str],
    *,
    law_context: Optional[str] = None,
) -> Dict[str, Any]:
    """Build kwargs for a law/group from state_map (arg_name -> state_key)."""
    out = {}
    for arg_name, key in state_map.items():
        val = state.get(key)
        if _string_placeholder_like(val, key=key, arg_name=str(arg_name)):
            val = None
        if val is None:
            val = constants.get(key)
            if _string_placeholder_like(val, key=key, arg_name=str(arg_name)):
                val = None
        if val is None:
            msg = f"Key {key!r} not found in state or constants (arg {arg_name})"
            if law_context is not None:
                msg += f" (law {law_context!r})"
                if _state_key_suggests_law_fd_fill(key):
                    msg += (
                        " — on Path B, enable auto_path_b_derivatives=True and fill_law_fd=True, "
                        "provide the primitive field (e.g. T for T_laplacian) and mesh coordinates "
                        "(x, and y/z/t as needed); see moju.monitor.law_fd_recipes.LAW_FD_RECIPES. "
                        "The engine does not rename NPZ keys: use key `T` (or change law state_map), "
                        "not only Studio alias hints."
                    )
            raise KeyError(msg)
        if isinstance(val, str):
            msg = (
                f"Key {key!r} resolved to string value {val!r} (arg {arg_name})"
                " but laws expect numeric array-like inputs."
            )
            if law_context is not None:
                msg += f" (law {law_context!r})"
            raise TypeError(msg)
        out[arg_name] = val
    return out


def _get_fn(spec: Dict[str, Any], builtin_class: Any) -> Any:
    if "fn" in spec:
        return spec["fn"]
    return getattr(builtin_class, spec["name"])


def _build_state(
    state_pred: Dict[str, Any],
    constants: Dict[str, Any],
    groups_spec: List[Dict],
) -> Dict[str, Any]:
    """Run group specs in order; write output_key into state."""
    state = dict(state_pred)
    merged = {**constants, **state}
    for spec in groups_spec:
        state_map = spec["state_map"]
        output_key = spec["output_key"]
        kwargs = _kwargs_from_state(merged, constants, state_map)
        fn = _get_fn(spec, Groups)
        state[output_key] = fn(**kwargs)
        merged[output_key] = state[output_key]
    return state


def _build_state_best_effort(
    state_pred: Dict[str, Any],
    constants: Dict[str, Any],
    groups_spec: List[Dict],
    *,
    log_skip: Callable[[str], None],
) -> Dict[str, Any]:
    """Best-effort group execution: skip groups whose inputs are unavailable."""
    state = dict(state_pred)
    merged = {**constants, **state}
    for spec in groups_spec:
        state_map = spec["state_map"]
        output_key = spec["output_key"]
        try:
            kwargs = _kwargs_from_state(merged, constants, state_map)
        except KeyError as err:
            log_skip(f"group:{spec.get('name')} skipped in best_effort_partial mode: {err}")
            continue
        fn = _get_fn(spec, Groups)
        state[output_key] = fn(**kwargs)
        merged[output_key] = state[output_key]
    return state


def _rms_scalar(x: jnp.ndarray) -> jnp.ndarray:
    a = jnp.asarray(x)
    if a.size == 0:
        return jnp.asarray(float("nan"))
    return jnp.sqrt(jnp.nanmean(jnp.square(a)))


def _r_eff_scalar(x: jnp.ndarray) -> jnp.ndarray:
    """
    Effective residual scalar **R_eff** = RMS_δ(r)·Q^p for collocation-point residuals, ``p`` = :data:`R_EFF_Q_POWER`.

    **RMS_δ(r)** = ``sqrt(mean(r^2) + δ^2)`` with ``δ^2`` = :data:`R_EFF_RMS_JITTER_SQ` (negligible vs typical residuals;
    avoids a singular gradient of plain ``sqrt(mean(r^2))`` at **r = 0** in autodiff).

    Let ``m_i = sqrt(r_i^2 + ε^2)`` with ε = :data:`_SCALE_EPS`. Define
    ``Q = RMS(m) / mean(m)`` (NaN-mean reductions). Then ``R_eff = RMS_δ(r) * (Q ** p)`` with ``p`` a positive float.
    For nonnegative ``m``, ``Q >= 1`` with equality iff all ``m_i`` are equal (uniform magnitude).
    For a single value (0-d or length-1), ``Q = 1`` so ``R_eff`` matches ``RMS_δ(r)``.
    """
    a = jnp.asarray(x).ravel()
    if a.size == 0:
        return jnp.asarray(float("nan"))
    jitter = jnp.asarray(R_EFF_RMS_JITTER_SQ, dtype=a.dtype)
    rms_r = jnp.sqrt(jnp.nanmean(jnp.square(a)) + jitter)
    if a.size == 1:
        return rms_r
    eps = jnp.asarray(_SCALE_EPS, dtype=a.dtype)
    m = jnp.sqrt(jnp.square(a) + eps * eps)
    rms_m = jnp.sqrt(jnp.nanmean(jnp.square(m)))
    mean_m = jnp.nanmean(m)
    q = rms_m / mean_m
    return rms_r * jnp.power(q, float(R_EFF_Q_POWER))


def admissibility_level(score: float) -> str:
    """
    Map admissibility score in ``[0, 1]`` to a qualitative label (``Unknown`` if non-finite).

    Bands: below 50% Non-Admissible; 50–75% Low; 75–95% Moderate; above 95% High.
    Boundaries: ``0.5`` and ``0.75`` start Low and Moderate; ``0.95`` is still Moderate; High requires
    ``score > 0.95``.
    """
    if not math.isfinite(score):
        return "Unknown"
    if score > ADM_HIGH_THRESHOLD:
        return "High Admissibility"
    if score >= 0.75:
        return "Moderate Admissibility"
    if score >= 0.5:
        return "Low Admissibility"
    return "Non-Admissible"


def is_high_admissibility(score: float) -> bool:
    """True if ``score`` is finite and above :data:`ADM_HIGH_THRESHOLD` (strictly greater)."""
    return math.isfinite(score) and score > ADM_HIGH_THRESHOLD


def _geom_mean_admissibility(values: Sequence[float]) -> float:
    """
    Geometric mean over finite scores only. If any finite value is <= 0, returns 0.0.
    If no finite values, returns NaN.
    """
    finite: List[float] = []
    for x in values:
        try:
            xf = float(x)
        except (TypeError, ValueError):
            continue
        if math.isfinite(xf):
            finite.append(xf)
    if not finite:
        return float("nan")
    if min(finite) <= 0.0:
        return 0.0
    return float(math.exp(sum(math.log(x) for x in finite) / len(finite)))


def _rms_per_key(
    residuals_flat: Dict[str, jnp.ndarray],
    *,
    to_python: bool = True,
) -> Dict[str, Any]:
    """Per-key **R_eff** (``rms`` in the log) via :func:`_r_eff_scalar`."""
    out: Dict[str, Any] = {}
    for key, arr in residuals_flat.items():
        r = _r_eff_scalar(arr)
        if to_python:
            out[key] = float(jax.device_get(r))
        else:
            out[key] = r
    return out


def _flatten_residual_dict(residuals: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
    flat: Dict[str, jnp.ndarray] = {}
    for category, content in residuals.items():
        if not isinstance(content, dict):
            if hasattr(content, "shape"):
                flat[category] = jnp.asarray(content)
            else:
                flat[category] = jnp.asarray(content)
            continue
        for name, arr in content.items():
            flat[f"{category}/{name}"] = jnp.asarray(arr)
    return flat


_SCALE_EPS = 1e-12


def _default_unit_scale_k(*, to_python: bool) -> float:
    """scale_k = :data:`DEFAULT_NONDIM_R_NORM_SCALE_K` + eps for nondimensional law / implied / ref residuals."""
    s = _SCALE_EPS + DEFAULT_NONDIM_R_NORM_SCALE_K
    return float(jax.device_get(jnp.asarray(s))) if to_python else float(s)


def _suffix_is_nd_closure(rest: str) -> bool:
    """True if flat tail is implied_delta or ref_delta (nondimensional closure residual)."""
    parts = rest.split("/")
    return bool(parts) and parts[-1] in ("implied_delta", "ref_delta")


def _state_derived_scale_per_key(
    flat_keys: Iterable[str],
    merged: Dict[str, Any],
    _laws_spec: List[Dict[str, Any]],
    constitutive_audit: List[Dict[str, Any]],
    state_ref_built: Optional[Dict[str, Any]] = None,
    *,
    to_python: bool = True,
) -> Dict[str, float]:
    """
    Per-key scale for ``R_norm = R_eff / scale_k`` (``R_eff`` in ``entry["rms"]``) stored on each log entry (``entry["scale"]``).

    Default **scale_k** is **≈ 2×10⁻²** (plus ε) for governing **laws/** and for nondimensional
    **implied_delta** / **ref_delta** under **constitutive/**. Other audit keys and
    **data/** use RMS of relevant state (or reference) fields. Optional ``r_ref`` in
    :func:`audit` overrides per key after logging.
    """
    out: Dict[str, float] = {}
    ref = state_ref_built if state_ref_built is not None else merged

    for k in flat_keys:
        if "/" not in k:
            out[k] = _default_unit_scale_k(to_python=to_python)
            continue
        prefix, rest = k.split("/", 1)
        if prefix == "laws":
            out[k] = _default_unit_scale_k(to_python=to_python)
            continue
        if prefix == "constitutive":
            if _suffix_is_nd_closure(rest):
                out[k] = _default_unit_scale_k(to_python=to_python)
                continue
            name = rest.split("/")[0]
            spec = next(
                (
                    s
                    for s in constitutive_audit
                    if s.get("name") == name
                    or (s.get("residual_basename") or "").split("/")[0] == name
                ),
                None,
            )
            if spec is None:
                scale = _SCALE_EPS + _rms_scalar(jnp.asarray(1.0))
            else:
                state_map = spec.get("state_map") or {}
                output_key = spec.get("output_key")
                parts = []
                for sk in state_map.values():
                    if sk in merged:
                        parts.append(jnp.ravel(jnp.asarray(merged[sk])))
                if output_key and output_key in merged:
                    parts.append(jnp.ravel(jnp.asarray(merged[output_key])))
                ivk = spec.get("implied_value_key")
                if ivk and ivk in merged:
                    parts.append(jnp.ravel(jnp.asarray(merged[ivk])))
                if parts:
                    big = jnp.concatenate(parts)
                    scale = _SCALE_EPS + _rms_scalar(big)
                else:
                    scale = _SCALE_EPS + _rms_scalar(jnp.asarray(1.0))
            out[k] = float(jax.device_get(scale)) if to_python else float(scale)
            continue
        if prefix == "scaling":
            # Legacy logs only (scaling audit removed from engine). Same default as unknown keys.
            out[k] = _default_unit_scale_k(to_python=to_python)
            continue
        if prefix == "data":
            state_key = rest
            if state_key in ref:
                v = jnp.asarray(ref[state_key])
                scale = _SCALE_EPS + _rms_scalar(jnp.ravel(v))
            else:
                scale = _SCALE_EPS + _rms_scalar(jnp.asarray(1.0))
            out[k] = float(jax.device_get(scale)) if to_python else float(scale)
            continue
        out[k] = _default_unit_scale_k(to_python=to_python)
    return out


def build_loss(
    residual_dict: Dict[str, Any],
    option: str = "cascaded",
    law_weights: Optional[Dict[str, float]] = None,
) -> jnp.ndarray:
    """
    Weighted sum of per-law **R_eff** scalars (same reduction as ``compute_residuals`` logs under ``rms``).
    """
    if option != "cascaded":
        raise ValueError(f"Only option='cascaded' is implemented, got {option!r}")
    laws = residual_dict.get("laws", {})
    if not laws:
        return jnp.array(0.0)
    names = list(laws.keys())
    n = len(names)
    weights = law_weights or {}
    w = jnp.array([weights.get(name, 1.0 / n) for name in names])
    rms_vals = jnp.array([_r_eff_scalar(jnp.asarray(laws[name])) for name in names])
    return jnp.sum(w * rms_vals)


def _compute_log_step_metrics(
    log: List[Dict[str, Any]],
    r_ref: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Per-log-entry admissibility metrics (same rules as ``audit``), without mutating ``log``.

    Returns one dict per entry with keys: ``r_norm``, ``admissibility_score``,
    ``category_admissibility_score``, ``overall_admissibility_score``, and
    ``per_key_report`` (flat key -> {rms, r_norm, admissibility_score, admissibility_level}).
    Field ``rms`` is **R_eff** from :func:`compute_residuals` (see :func:`_r_eff_scalar`).
    Scale precedence per key: **r_ref** (if given) > entry ``scale`` > first-step RMS > 1.
    Category scores are **0** if any per-key admissibility in that category is non-finite.

    **Overall admissibility:** for ``run_mode == \"training\"``, geometric mean of **laws** and
    **constitutive** only. For ``run_mode == \"eval\"``, geometric mean of finite, present
    category scores (missing categories are excluded). For legacy entries without ``run_mode``,
    geometric mean of **all** present categories.
    """
    if not log:
        return []
    first_rms = log[0].get("rms", {})
    out: List[Dict[str, Any]] = []
    category_buckets = ("laws", "constitutive", "scaling", "data")
    for entry in log:
        rms = entry.get("rms", {})
        entry_scale = entry.get("scale") or {}
        r_norm: Dict[str, float] = {}
        admissibility: Dict[str, float] = {}
        per_key_report: Dict[str, Any] = {}
        for k, v in rms.items():
            if r_ref is not None and k in r_ref and r_ref[k] is not None and r_ref[k] > 0:
                scale_k = r_ref[k]
            elif k in entry_scale and entry_scale[k] is not None and entry_scale[k] > 0:
                scale_k = entry_scale[k]
            elif k in first_rms and first_rms[k] is not None and first_rms[k] > 0:
                scale_k = first_rms[k]
            else:
                scale_k = 1.0
            if scale_k <= 0 or not math.isfinite(float(scale_k)):
                scale_k = 1.0
            try:
                v_f = float(v)
            except (TypeError, ValueError):
                v_f = float("nan")
            if not math.isfinite(v_f):
                r_norm[k] = float("nan")
                admissibility[k] = float("nan")
            else:
                r_norm[k] = v_f / scale_k
                if not math.isfinite(r_norm[k]):
                    admissibility[k] = float("nan")
                else:
                    admissibility[k] = 1.0 / (1.0 + r_norm[k])
            per_key_report[k] = {
                "rms": v,
                "r_norm": r_norm[k],
                "admissibility_score": admissibility[k],
                "admissibility_level": admissibility_level(admissibility[k]),
            }
        category_keys: Dict[str, List[str]] = {c: [] for c in category_buckets}
        for k in admissibility.keys():
            if "/" in k:
                prefix = k.split("/", 1)[0]
                if prefix in category_keys:
                    category_keys[prefix].append(k)
        category_scores: Dict[str, float] = {}
        for cat, keys in category_keys.items():
            if not keys:
                continue
            vals: List[float] = []
            category_failed = False
            for kk in keys:
                try:
                    vf = float(admissibility[kk])
                except (TypeError, ValueError):
                    category_failed = True
                    break
                if not math.isfinite(vf):
                    category_failed = True
                    break
                vals.append(vf)
            if category_failed:
                category_scores[cat] = 0.0
            else:
                category_scores[cat] = _geom_mean_admissibility(vals)
        cats_present = list(category_scores.keys())
        rm = entry.get("run_mode")
        if rm == "training":
            train_cats = [c for c in ("laws", "constitutive") if c in category_scores]
            if not train_cats:
                overall = float("nan")
            else:
                overall = _geom_mean_admissibility(
                    [float(category_scores[c]) for c in train_cats]
                )
        elif rm == "eval":
            eval_vals = [
                float(vv)
                for vv in category_scores.values()
                if math.isfinite(float(vv))
            ]
            overall = _geom_mean_admissibility(eval_vals) if eval_vals else float("nan")
        else:
            # Legacy logs without ``run_mode``: all present categories.
            if not cats_present:
                overall = float("nan")
            else:
                overall = _geom_mean_admissibility(
                    [float(category_scores[c]) for c in cats_present]
                )
        out.append(
            {
                "r_norm": r_norm,
                "admissibility_score": admissibility,
                "category_admissibility_score": category_scores,
                "overall_admissibility_score": overall,
                "per_key_report": per_key_report,
            }
        )
    return out


def audit(
    log: List[Dict[str, Any]],
    r_ref: Optional[Dict[str, float]] = None,
    weights: Optional[Dict[str, float]] = None,
    *,
    export_dir: Optional[str] = None,
    save_residuals: bool = False,
    last_residual_dict: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Physics admissibility from logged **R_eff** (field ``rms``) and scales.

    **R_norm** uses ``R_eff / scale_k`` where each entry's ``rms`` is **R_eff** = RMS_δ(r)·Q^0.5 from
    :func:`compute_residuals` (see :func:`_r_eff_scalar`). ``scale_k`` comes from each entry's ``scale`` when
    positive, else the first step's RMS fallback, else 1. **ResidualEngine** logs default
    ``scale_k ≈ 2×10⁻²`` for **laws/** and nondimensional **implied_delta** / **ref_delta** keys;
    other residuals and **data/** use state-derived scales. Optional **r_ref** (flat key → positive
    float) overrides ``scale_k`` for those keys. Per-key **admissibility_score** is
    ``1 / (1 + R_norm)`` when finite.

    Reporting uses three levels (no extra aggregation API): (1) per residual key in
    ``per_key``; (2) geometric mean within each category — ``per_category`` keys
    ``laws``, ``constitutive``, ``scaling``, ``data`` — only when **every** per-key
    admissibility in that category is finite; if any key is non-finite (NaN/Inf) or
    non-numeric, the category score is **0** (inadmissible for that bucket); empty
    categories omitted; (3) **overall** score — for ``run_mode == \"training\"``, geometric mean of
    **laws** and **constitutive** only; for ``run_mode == \"eval\"``, geometric mean of finite,
    present category scores (missing categories excluded); for legacy entries without ``run_mode``,
    geometric mean of all present categories.

    The returned dict includes ``monitor_run_mode`` from the last log entry when present.
    """
    if not log:
        return {"per_key": {}, "overall_admissibility_score": 0.0, "overall_admissibility_level": "Non-Admissible"}
    step_metrics = _compute_log_step_metrics(log, r_ref)
    last_report_per_key: Dict[str, Any] = {}
    for entry, m in zip(log, step_metrics):
        entry["r_norm"] = m["r_norm"]
        entry["admissibility_score"] = m["admissibility_score"]
        entry["category_admissibility_score"] = m["category_admissibility_score"]
        entry["overall_admissibility_score"] = m["overall_admissibility_score"]
        last_report_per_key = dict(m["per_key_report"])
    overall = log[-1].get("overall_admissibility_score", 0.0) if log else 0.0
    report = {
        "per_key": last_report_per_key,
        "per_category": log[-1].get("category_admissibility_score", {}) if log else {},
        "overall_admissibility_score": overall,
        "overall_admissibility_level": admissibility_level(overall),
        "monitor_run_mode": (log[-1].get("run_mode") if log else None),
    }

    if export_dir:
        import zipfile
        from pathlib import Path
        session_name = datetime.datetime.now().strftime("audit_%Y%m%d_%H%M")
        session_dir = Path(export_dir) / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        try:
            from moju.monitor.report import write_audit_pdf, write_residuals_json
            pdf_path = session_dir / "report.pdf"
            write_audit_pdf(report, str(pdf_path), model_name=model_name, model_id=model_id)
            if save_residuals and last_residual_dict is not None:
                json_path = session_dir / "residuals.json"
                write_residuals_json(last_residual_dict, str(json_path))
            zip_path = Path(export_dir) / f"{session_name}.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in session_dir.iterdir():
                    zf.write(f, f"{session_name}/{f.name}")
        except ImportError as e:
            raise ImportError(
                "PDF export requires reportlab. Install with: pip install moju[report] or pip install reportlab"
            ) from e

    return report


def _keys_by_category(plot_keys: Sequence[str]) -> Dict[str, List[str]]:
    order = ("laws", "constitutive", "scaling", "data")
    buckets: Dict[str, List[str]] = {c: [] for c in order}
    for k in plot_keys:
        if "/" not in k:
            continue
        p = k.split("/", 1)[0]
        if p in buckets:
            buckets[p].append(k)
    return buckets


def _parse_spatial_values_panel(panel: Mapping[str, Any], *, law_style: bool) -> Optional[Dict[str, Any]]:
    """
    Parse spatial law or constitutive panel with 1D, 2D, or 3D ``values`` arrays.

    **1D:** ``x`` (length ``nx``) and each ``values[k]`` shape ``(nx,)``. Output includes
    ``kind="1d"``, ``Z`` with shape ``(n_keys, nx)``.

    **2D:** ``x`` (``nx``), ``y`` (``ny``), each ``values[k]`` shape ``(ny, nx)`` with
    entry ``[j, i]`` at ``y[j], x[i]``. Output: ``kind="2d"``, ``Z`` ``(n_keys, ny, nx)``.

    **3D:** ``x``, ``y``, ``z`` 1D and each ``values[k]`` shape ``(nx, ny, nz)`` with
    ``[i, j, k]`` at ``x[i], y[j], z[k]``. Output: ``kind="3d"``, ``V`` ``(n_keys, nx, ny, nz)``.

    Optional ``log_step_index`` (int) is copied through for captions.
    Optional ``position_axis`` is kept for 1D (horizontal axis label hint).
    """
    values = panel.get("values")
    if values is None or not isinstance(values, Mapping):
        return None
    import numpy as np

    keys_sorted = sorted(values.keys())
    if not keys_sorted:
        return None
    arrs = [np.asarray(values[k], dtype=float) for k in keys_sorted]
    if not arrs:
        return None
    s0 = arrs[0].shape
    if not all(a.shape == s0 and a.ndim == arrs[0].ndim for a in arrs):
        return None
    nd = arrs[0].ndim

    log_step_index = panel.get("log_step_index")
    if log_step_index is not None and not isinstance(log_step_index, int):
        log_step_index = None

    def _labels() -> List[str]:
        lab: List[str] = []
        for k in keys_sorted:
            lk = str(k)
            if law_style:
                flat = lk if lk.startswith("laws/") else f"laws/{lk}"
                lab.append(pretty_residual_key(flat))
            else:
                lab.append(pretty_residual_key(lk))
        return lab

    labels = _labels()
    extra: Dict[str, Any] = {"row_labels": labels}
    if log_step_index is not None:
        extra["log_step_index"] = int(log_step_index)

    if nd == 1:
        if panel.get("y") is not None or panel.get("z") is not None:
            return None
        x = panel.get("x")
        if x is None:
            return None
        x_arr = np.asarray(x, dtype=float).ravel()
        if x_arr.size != s0[0]:
            return None
        out: Dict[str, Any] = {
            "kind": "1d",
            "x": x_arr,
            "Z": np.stack(arrs, axis=0),
            **extra,
        }
        pos_ax = panel.get("position_axis")
        if pos_ax is not None and str(pos_ax).strip():
            out["position_axis"] = str(pos_ax).strip()
        return out

    if nd == 2:
        if panel.get("z") is not None:
            return None
        x = panel.get("x")
        y = panel.get("y")
        if x is None or y is None:
            return None
        x_arr = np.asarray(x, dtype=float).ravel()
        y_arr = np.asarray(y, dtype=float).ravel()
        ny, nx = s0
        if ny != y_arr.size or nx != x_arr.size:
            return None
        return {
            "kind": "2d",
            "x": x_arr,
            "y": y_arr,
            "Z": np.stack(arrs, axis=0),
            **extra,
        }

    if nd == 3:
        x = panel.get("x")
        y = panel.get("y")
        z = panel.get("z")
        if x is None or y is None or z is None:
            return None
        x_arr = np.asarray(x, dtype=float).ravel()
        y_arr = np.asarray(y, dtype=float).ravel()
        z_arr = np.asarray(z, dtype=float).ravel()
        nx, ny, nz = s0
        if nx != x_arr.size or ny != y_arr.size or nz != z_arr.size:
            return None
        return {
            "kind": "3d",
            "x": x_arr,
            "y": y_arr,
            "z": z_arr,
            "V": np.stack(arrs, axis=0),
            **extra,
        }

    return None


def _parse_spatial_law_panel(spatial_law_panel: Optional[Any]) -> Optional[Dict[str, Any]]:
    """See :func:`_parse_spatial_values_panel` (``law_style=True``)."""
    if spatial_law_panel is None:
        return None
    if not isinstance(spatial_law_panel, Mapping):
        return None
    return _parse_spatial_values_panel(spatial_law_panel, law_style=True)


def _parse_spatial_rnorm_panel(spatial_rnorm_panel: Optional[Any]) -> Optional[Dict[str, Any]]:
    """See :func:`_parse_spatial_values_panel` (``law_style=False``)."""
    if spatial_rnorm_panel is None:
        return None
    if not isinstance(spatial_rnorm_panel, Mapping):
        return None
    return _parse_spatial_values_panel(spatial_rnorm_panel, law_style=False)


def _worst_keys_table_rows(
    log: List[Dict[str, Any]],
    plot_keys: List[str],
    metrics: List[Dict[str, Any]],
    *,
    top_n: int = 12,
) -> List[Dict[str, Any]]:
    """Final-step per-key rows sorted by R_norm descending (for troubleshooting table)."""
    if not log or not metrics:
        return []
    from moju.monitor.visualize_labels import pretty_residual_key

    last_entry = log[-1]
    scale_map = last_entry.get("scale") or {}
    per = metrics[-1].get("per_key_report") or {}
    rows: List[Dict[str, Any]] = []
    for k in plot_keys:
        rep = per.get(k)
        if not isinstance(rep, Mapping):
            continue
        try:
            rn_f = float(rep.get("r_norm"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(rn_f):
            continue
        try:
            r_eff = float(rep.get("rms"))
        except (TypeError, ValueError):
            r_eff = float("nan")
        sk_raw = scale_map.get(k)
        try:
            sk_f = float(sk_raw) if sk_raw is not None else float("nan")
        except (TypeError, ValueError):
            sk_f = float("nan")
        if not math.isfinite(sk_f) or sk_f <= 0:
            if math.isfinite(r_eff) and rn_f != 0.0:
                sk_f = abs(r_eff / rn_f)
            else:
                sk_f = float("nan")
        try:
            adm_f = float(rep.get("admissibility_score"))
        except (TypeError, ValueError):
            adm_f = float("nan")
        prefix = k.split("/", 1)[0] if "/" in str(k) else "other"
        rows.append(
            {
                "key": k,
                "display": pretty_residual_key(str(k)),
                "r_eff": r_eff,
                "scale_k": sk_f,
                "r_norm": rn_f,
                "admissibility": adm_f,
                "category": prefix,
            }
        )
    rows.sort(key=lambda r: r["r_norm"], reverse=True)
    cap = max(1, int(top_n))
    return rows[:cap]


def _build_visualize_bundle(
    log: List[Dict[str, Any]],
    keys: Optional[List[str]],
    r_ref: Optional[Dict[str, float]],
    max_legend_keys: int,
    *,
    spatial_parsed: Optional[Dict[str, Any]],
    spatial_rnorm_parsed: Optional[Dict[str, Any]] = None,
    mode: str,
    spatial_normalize: bool = False,
    worst_keys_top_n: int = 12,
) -> Optional[Dict[str, Any]]:
    """
    Shared arrays and metadata for :func:`visualize` (Plotly).

    Does not mutate ``log``. Requires ``numpy`` (already a moju dependency).
    """
    import numpy as np

    if not log:
        return None
    first_rms = log[0].get("rms", {})
    plot_keys = list(keys) if keys is not None else list(first_rms.keys())
    if not plot_keys:
        return None
    metrics = _compute_log_step_metrics(log, r_ref)
    n = len(log)
    indices = np.arange(n, dtype=float)
    cap = max(1, int(max_legend_keys))
    use_bar_chart = mode == "eval" or (mode == "training" and n == 1)
    bar_keys = plot_keys[: min(48, len(plot_keys))]

    buckets = _keys_by_category(plot_keys)
    category_training: Dict[str, Dict[str, Any]] = {}
    cat_order = ("laws", "constitutive")
    for cat in cat_order:
        cat_keys = sorted(buckets.get(cat, []))[:cap]
        nk = len(cat_keys)
        mat = np.zeros((nk, n)) if nk else np.zeros((0, n))
        for j in range(n):
            for i, kk in enumerate(cat_keys):
                v = metrics[j]["r_norm"].get(kk)
                mat[i, j] = float(v) if v is not None else float("nan")
        category_training[cat] = {
            "keys": cat_keys,
            "displays": [pretty_residual_key(k) for k in cat_keys],
            "r_norm_mat": mat,
        }

    legend_keys = plot_keys[: min(cap, len(plot_keys))]
    r_norm_mat = np.zeros((len(legend_keys), n))
    for j in range(n):
        for i, kk in enumerate(legend_keys):
            v = metrics[j]["r_norm"].get(kk)
            r_norm_mat[i, j] = float(v) if v is not None else float("nan")

    legend_display = [pretty_residual_key(k) for k in legend_keys]
    bar_display = [pretty_residual_key(k) for k in bar_keys]
    bar_values = []
    last_rn = metrics[-1]["r_norm"]
    for kk in bar_keys:
        v = last_rn.get(kk)
        bar_values.append(float(v) if v is not None else float("nan"))
    bar_values_arr = np.asarray(bar_values, dtype=float)

    cat_colors = {
        "laws": "#4e79a7",
        "constitutive": "#f28e2b",
        "scaling": "#59a14f",
        "data": "#b07aa1",
    }
    last_cat = metrics[-1]["category_admissibility_score"]
    cats_fin = [
        (pretty_category_name(c), float(last_cat[c]))
        for c in cat_colors
        if c in last_cat and np.isfinite(last_cat[c])
    ]
    nr_title = "Normalized Residuals"
    if not use_bar_chart and len(plot_keys) > len(legend_keys):
        nr_title = f"Normalized Residuals (showing {len(legend_keys)} of {len(plot_keys)} keys)"

    category_titles = {
        "laws": "Normalized Governing Laws Residuals",
        "constitutive": "Normalized Constitutive Residuals",
        "scaling": "Normalized Scaling Residuals",
    }

    return {
        "log": log,
        "metrics": metrics,
        "n": n,
        "indices": indices,
        "plot_keys": plot_keys,
        "legend_keys": legend_keys,
        "legend_display": legend_display,
        "r_norm_mat": r_norm_mat,
        "category_training": category_training,
        "category_titles": category_titles,
        "use_bar_chart": use_bar_chart,
        "bar_keys": bar_keys,
        "bar_display": bar_display,
        "bar_values": bar_values_arr,
        "overall_adm": [float(metrics[i]["overall_admissibility_score"]) for i in range(n)],
        "cats_fin": cats_fin,
        "cat_colors": cat_colors,
        "spatial": spatial_parsed,
        "spatial_rnorm": spatial_rnorm_parsed,
        "mode": mode,
        "nr_title": nr_title,
        "np": np,
        "spatial_normalize": bool(spatial_normalize),
        "worst_keys_rows": _worst_keys_table_rows(
            log, plot_keys, metrics, top_n=worst_keys_top_n
        ),
        "monitor_run_mode": log[-1].get("run_mode"),
    }


def _maybe_build_spatial_panels(
    log_full: List[Dict[str, Any]],
    spatial_law_panel: Optional[Dict[str, Any]],
    spatial_rnorm_panel: Optional[Dict[str, Any]],
    residuals: Optional[Dict[str, Any]],
    state_pred: Optional[Mapping[str, Any]],
    r_ref: Optional[Dict[str, float]],
    spatial_coord_key: str,
    spatial_prefer_last_t: bool,
    *,
    spatial_normalize: bool = False,
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if residuals is None:
        return spatial_law_panel, spatial_rnorm_panel

    pred: Dict[str, Any] = {}
    if state_pred is not None:
        pred.update(dict(state_pred))
    if log_full:
        snap = log_full[-1].get("coord_snapshot")
        if isinstance(snap, Mapping):
            for k in ("x", "y", "z", "t"):
                if k in snap and snap[k] is not None and k not in pred:
                    pred[k] = snap[k]

    auto_law: Optional[Dict[str, Any]] = None
    auto_rnorm: Optional[Dict[str, Any]] = None
    log_entry = log_full[-1] if log_full else None
    first_rms = (log_full[0].get("rms") or {}) if log_full else {}
    log_step_index = len(log_full) - 1 if log_full else None
    rr = dict(r_ref) if r_ref else {}
    auto_law, auto_rnorm = build_spatial_rnorm_panels_from_residuals(
        residuals,
        pred,
        coord_key=str(spatial_coord_key),
        prefer_last_t=bool(spatial_prefer_last_t),
        log_entry=log_entry,
        first_rms=first_rms,
        r_ref=rr,
        log_step_index=log_step_index,
        normalize_spatial=bool(spatial_normalize),
    )

    law_out = spatial_law_panel if spatial_law_panel is not None else auto_law
    rnorm_out = spatial_rnorm_panel if spatial_rnorm_panel is not None else auto_rnorm
    return law_out, rnorm_out


def build_monitor_visualize_bundle(
    log: List[Dict[str, Any]],
    keys: Optional[List[str]] = None,
    r_ref: Optional[Dict[str, float]] = None,
    max_legend_keys: int = 16,
    *,
    spatial_law_panel: Optional[Dict[str, Any]] = None,
    spatial_rnorm_panel: Optional[Dict[str, Any]] = None,
    mode: str = "training",
    residuals: Optional[Dict[str, Any]] = None,
    state_pred: Optional[Mapping[str, Any]] = None,
    spatial_coord_key: str = "x",
    spatial_prefer_last_t: bool = True,
    spatial_normalize: bool = False,
    engine: Optional[Any] = None,
    worst_keys_top_n: int = 12,
) -> Optional[Dict[str, Any]]:
    """
    Build the internal visualization bundle (same as :func:`visualize` uses for Plotly).

    Intended for Studio and other callers that want several small Plotly figures instead of
    one combined subplot grid.

    ``spatial_normalize``: if True, auto-built spatial panels use R_norm-style :math:`|r|/s_k`;
    default False uses absolute :math:`|r|` vs position (labels match).

    ``engine``: when ``residuals`` is omitted, use :attr:`ResidualEngine.last_residuals` if given.

    ``mode``: same as :func:`visualize` — ``training`` or ``eval``; ``test`` is a silent alias for ``eval``.
    """
    eff_residuals = residuals
    if eff_residuals is None and engine is not None:
        eff_residuals = getattr(engine, "last_residuals", None)
    law_panel, rnorm_panel = _maybe_build_spatial_panels(
        log,
        spatial_law_panel,
        spatial_rnorm_panel,
        eff_residuals,
        state_pred,
        r_ref,
        spatial_coord_key,
        spatial_prefer_last_t,
        spatial_normalize=spatial_normalize,
    )
    mode = _normalize_visualize_mode(mode)
    work_log = list(log)
    if mode == "eval" and len(work_log) > 1:
        work_log = work_log[-1:]
    spatial_parsed = _parse_spatial_law_panel(law_panel)
    spatial_rnorm_parsed = _parse_spatial_rnorm_panel(rnorm_panel)
    return _build_visualize_bundle(
        work_log,
        keys,
        r_ref,
        max_legend_keys,
        spatial_parsed=spatial_parsed,
        spatial_rnorm_parsed=spatial_rnorm_parsed,
        mode=mode,
        spatial_normalize=spatial_normalize,
        worst_keys_top_n=worst_keys_top_n,
    )


def visualize(
    log: List[Dict[str, Any]],
    keys: Optional[List[str]] = None,
    backend: str = "plotly",
    *,
    r_ref: Optional[Dict[str, float]] = None,
    max_legend_keys: int = 16,
    mode: str = "training",
    spatial_law_panel: Optional[Dict[str, Any]] = None,
    spatial_rnorm_panel: Optional[Dict[str, Any]] = None,
    residuals: Optional[Dict[str, Any]] = None,
    state_pred: Optional[Mapping[str, Any]] = None,
    spatial_coord_key: str = "x",
    spatial_prefer_last_t: bool = True,
    spatial_normalize: bool = False,
    engine: Optional[Any] = None,
    figure_title: Optional[str] = None,
    step_label: str = "Step",
    r_norm_scale: str = "log",
    spatial_heatmap_colorscale: Optional[str] = None,
    dashboard_mode: str = "single-figure",
    theme: str = "light",
    baseline_score: Optional[float] = None,
    export_buttons: bool = True,
    show_branding: bool = False,
    visualize_layout: str = "single",
    worst_keys_top_n: int = 12,
    density: str = "comfortable",
) -> Any:
    """
    Monitor dashboard from ``ResidualEngine`` log entries (``rms``, ``scale``).

    Uses the same R_norm / admissibility rules as :func:`audit` via
    :func:`_compute_log_step_metrics` and **does not mutate** ``log``.

    **Modes**

    - ``training`` (multi-step) — **Top row:** overall admissibility vs step (with final
      value marker) and **horizontal category admissibility bars** (laws / constitutive,
      final step). **Second row:** **two** KPI indicators (Governing, Constitutive) only.
      **Third row:** two panels — :math:`R_{\\mathrm{norm}}` vs
      step for **governing laws** and **constitutive** (``data/`` and ``scaling/`` omitted);
      **y-axis** is ``log10(R_{\\mathrm{norm}} + \\varepsilon)`` by default, or linear if
      ``r_norm_scale="linear"``. **Fourth row:** **vs spatial coordinate** (last logged step)
      for laws and constitutive: by default :math:`|r|` (**absolute residual**); set
      ``spatial_normalize=True`` for :math:`R_{\\mathrm{norm}}`-style :math:`|r|/s_k`.
      Placeholders if no spatial data. Pass ``spatial_*_panel``, or ``residuals`` with coordinates from
      ``state_pred``, or ``residuals`` with the **last** log entry's ``coord_snapshot``
      (written by :meth:`ResidualEngine.compute_residuals` when the built state has ``x`` /
      ``y`` / ``z`` / ``t``).
    - ``training`` (single log entry) — horizontal bars for normalized residuals, category
      admissibility bars, spatial row (|residual| vs position by default).
    - ``eval`` — Uses the **last** log entry for scalar metrics; **no vs-step admissibility
      panel** (category breakdown is **full width** on its row). **Second row:** **three**
      KPI indicators (Governing, Constitutive, Scaling; **no** Data KPI card—``data/`` remains
      in breakdown and per-key views). **Spatial row** shows |residual| vs position by default
      (same log/linear display scale as training for heatmaps), plus horizontal bar chart of
      normalized residuals and category admissibility. The legacy value ``mode="test"`` is
      accepted and treated like ``eval``.

    **Spatial panel**

    The log stores **scalar** RMS per key per step only. Pass panels built from the
    **last** ``compute_residuals`` / final log step (callers may set optional
    ``log_step_index`` on the dict for captions).

    **1D:** ``x`` (length ``nx``) and ``values`` with 1D arrays ``(nx,)``. Law keys may be
    bare names or ``laws/<name>``. Optional ``position_axis`` sets the horizontal axis
    label (default ``x``).

    **2D:** ``x``, ``y`` 1D and each ``values[k]`` with shape ``(len(y), len(x))``.

    **3D:** ``x``, ``y``, ``z`` 1D and each ``values[k]`` with shape
    ``(len(x), len(y), len(z))``.

    For constitutive-only slices use ``spatial_rnorm_panel`` (flat keys). **Plotly**
    renders 1D/2D/3D spatial panels.

    **Backends**

    - ``plotly`` (default) — interactive figure (requires ``pip install plotly`` or ``moju[viz]``).
    - ``none`` — returns ``None``.

    ``backend="matplotlib"`` is **not supported** (raises ``ValueError``); use ``plotly``.

    Parameters
    ----------
    log
        Entries from ``ResidualEngine.log`` (after ``compute_residuals``).
    keys
        Subset of flat residual keys to plot; default = all keys in the first entry.
    backend
        ``plotly`` (default) or ``none``.
    r_ref
        Optional per-key reference scale overrides (same as :func:`audit`).
    max_legend_keys
        Cap legend entries on per-key line plots for readability (training mode, multi-step).
    mode
        ``training`` or ``eval``. Eval mode slices to the last entry when ``len(log) > 1``.
        ``test`` is accepted as a silent alias for ``eval``.
    spatial_law_panel
        Optional ``dict`` with ``x`` and ``values`` (see above).
    spatial_rnorm_panel
        Optional ``dict`` with ``x`` and ``values`` for per-key R_norm spatial slices.
    residuals
        Optional nested residual dict (same shape as ``compute_residuals`` output). When
        both ``spatial_law_panel`` and ``spatial_rnorm_panel`` are omitted, panels are built
        for the **last** log step from ``state_pred`` coordinates and/or the last entry's
        ``coord_snapshot`` (lists of floats), merged so snapshot fills only missing axes.
        If coordinates are still missing but law/constitutive arrays share a consistent 1D
        length, a neutral ``linspace(0, 1, n)`` axis is inferred so ``residuals=`` alone can
        suffice (e.g. ``visualize(log, mode=\"eval\", residuals=...)``).
    state_pred
        Optional state dict with coordinate arrays (``x``, optional ``y`` / ``z``, ``t``) for
        auto spatial panels. May be omitted or partial when ``log[-1]["coord_snapshot"]``
        supplies the missing axes.
    spatial_coord_key
        Primary 1D axis name when falling back from 3D/2D to a line slice (default ``"x"``).
    spatial_prefer_last_t
        If True and ``t`` is present, use the last time slice of residual fields.
    spatial_normalize
        If True, auto-built spatial panels use :math:`|r|/s_k` (R_norm-style). Default False
        uses absolute residual :math:`|r|` vs position.
    engine
        Optional :class:`ResidualEngine` whose :attr:`~ResidualEngine.last_residuals` is used
        when ``residuals`` is omitted, so ``visualize(engine.log, engine=engine)`` can build
        spatial heatmaps using log ``coord_snapshot`` (same convenience as Moju Studio).
    figure_title
        Optional override for the Plotly figure layout title. If omitted or blank, a mode-specific
        default is used (training vs eval). The single-figure dashboard shows this as the report
        header (training default: :data:`DEFAULT_VISUALIZE_TITLE_TRAINING`;
        eval default: :data:`DEFAULT_VISUALIZE_TITLE_EVAL`).
    step_label
        X-axis label for training step axis (e.g. ``Iteration`` or ``Epoch``).
    r_norm_scale
        ``log`` (default) plots ``log10(R_norm + ε)`` on the three category residual
        panels; ``linear`` plots raw ``R_norm``. Does not affect the overall admissibility
        axis.
    spatial_heatmap_colorscale
        Plotly colorscale name for optional spatial heatmaps (e.g. ``\"Viridis\"``, ``\"Cividis\"``).
        Default ``None`` uses ``Viridis`` in the Plotly backend.
    show_branding
        If True, show the Moju forensic-suite watermark on the main dashboard. Default False.
    visualize_layout
        ``\"single\"`` (default) returns one Plotly figure (or dash-tabs payload). ``\"split\"``
        also includes a separate ``worst_keys`` table figure: for ``single-figure`` the return value
        is ``{\"monitor\": fig, \"worst_keys\": table_fig}``; for ``dash-tabs`` the payload dict
        gains a ``\"worst_keys\"`` entry.
    worst_keys_top_n
        Number of rows in the worst-keys table (final step, sorted by ``R_norm``).
    density
        ``\"comfortable\"`` (default) or ``\"compact\"`` — slightly tighter typography on the main figure.
    theme
        Must be ``\"light\"`` (default). Plotly monitor figures use a single light enterprise style;
        ``\"dark\"`` is not supported.
    """
    if backend == "none":
        return None
    if backend == "matplotlib":
        raise ValueError(
            'visualize(..., backend="matplotlib") is no longer supported; '
            'use backend="plotly" (default) and pip install plotly or moju[viz].'
        )
    if backend != "plotly":
        raise ValueError(f"Unknown visualize backend {backend!r}; use 'plotly' or 'none'.")
    mode = _normalize_visualize_mode(mode)
    if mode not in ("training", "eval"):
        raise ValueError("mode must be 'training' or 'eval' (or 'test' as an alias for 'eval')")
    if r_norm_scale not in ("log", "linear"):
        raise ValueError("r_norm_scale must be 'log' or 'linear'")
    if dashboard_mode not in ("single-figure", "dash-tabs"):
        raise ValueError("dashboard_mode must be 'single-figure' or 'dash-tabs'")
    if visualize_layout not in ("single", "split"):
        raise ValueError("visualize_layout must be 'single' or 'split'")
    if density not in ("comfortable", "compact"):
        raise ValueError("density must be 'comfortable' or 'compact'")
    if theme != "light":
        raise ValueError("visualize Plotly styling supports theme='light' only; dark mode is no longer supported.")

    eff_residuals = residuals
    if eff_residuals is None and engine is not None:
        eff_residuals = getattr(engine, "last_residuals", None)

    law_panel, rnorm_panel = _maybe_build_spatial_panels(
        log,
        spatial_law_panel,
        spatial_rnorm_panel,
        eff_residuals,
        state_pred,
        r_ref,
        spatial_coord_key,
        spatial_prefer_last_t,
        spatial_normalize=spatial_normalize,
    )

    work_log = list(log)
    if mode == "eval" and len(work_log) > 1:
        work_log = work_log[-1:]

    spatial_parsed = _parse_spatial_law_panel(law_panel)
    spatial_rnorm_parsed = _parse_spatial_rnorm_panel(rnorm_panel)
    bundle = _build_visualize_bundle(
        work_log,
        keys,
        r_ref,
        max_legend_keys,
        spatial_parsed=spatial_parsed,
        spatial_rnorm_parsed=spatial_rnorm_parsed,
        mode=mode,
        spatial_normalize=spatial_normalize,
        worst_keys_top_n=worst_keys_top_n,
    )
    if bundle is None:
        return None

    resolved_title = _resolve_visualize_figure_title(mode, figure_title)

    try:
        from moju.monitor.visualize_plotly import (
            build_plotly_monitor_figure,
            build_worst_keys_table_figure,
        )

        out = build_plotly_monitor_figure(
            bundle,
            figure_title=resolved_title,
            step_label=step_label,
            r_norm_scale=r_norm_scale,
            spatial_heatmap_colorscale=spatial_heatmap_colorscale,
            dashboard_mode=dashboard_mode,
            theme=theme,
            baseline_score=baseline_score,
            export_buttons=export_buttons,
            show_branding=show_branding,
            density=density,
        )
        if visualize_layout == "split":
            wk = build_worst_keys_table_figure(bundle)
            if isinstance(out, dict):
                out = dict(out)
                out["worst_keys"] = wk
                return out
            return {"monitor": out, "worst_keys": wk}
        return out
    except ImportError:
        return None


def _coord_snapshot_from_merged(merged: Mapping[str, Any]) -> Dict[str, List[float]]:
    """
    Extract 1D ``x`` / ``y`` / ``z`` / ``t`` vectors from built state for JSON-safe log storage.

    Used for :attr:`ResidualEngine.log` ``coord_snapshot`` so :func:`visualize` can build
    position heatmaps when ``state_pred`` is omitted but the last log step recorded coords.
    """
    import numpy as np

    out: Dict[str, List[float]] = {}
    for key in ("x", "y", "z", "t"):
        if key not in merged:
            continue
        raw = merged.get(key)
        if raw is None:
            continue
        try:
            arr = np.asarray(jax.device_get(jnp.asarray(raw)), dtype=float).ravel()
        except (TypeError, ValueError):
            continue
        if arr.size == 0:
            continue
        out[key] = [float(v) for v in arr.tolist()]
    return out


def _default_minimal_user_fns() -> Dict[str, Callable[..., Any]]:
    """
    Default helper closures for the minimal-input path.
    These are only used when the caller does not provide the same key in ``user_fns``.
    """
    return {
        "alpha": lambda k, rho, cp: jnp.asarray(k) / (jnp.asarray(rho) * jnp.asarray(cp)),
        "D": lambda fo_mass, t, L: jnp.asarray(fo_mass) * (jnp.asarray(L) ** 2) / jnp.asarray(t),
        "nu": lambda mu, rho: jnp.asarray(mu) / jnp.asarray(rho),
        "mu": lambda rho, nu: jnp.asarray(rho) * jnp.asarray(nu),
        "kappa": lambda u, L, pe: jnp.asarray(u) * jnp.asarray(L) / jnp.asarray(pe),
        "c": lambda omega, L, st_wave: jnp.asarray(omega) * jnp.asarray(L) / jnp.asarray(st_wave),
    }


def build_minimal_residual_engine(
    *,
    law_names: Sequence[str],
    constants: Optional[Dict[str, Any]] = None,
    groups: Optional[List[Dict[str, Any]]] = None,
    constitutive_audit: Optional[List[Dict[str, Any]]] = None,
    user_fns: Optional[Dict[str, Callable[..., Any]]] = None,
    derived_state_chain: Optional[List[Dict[str, Any]]] = None,
    state_builder: Optional[
        Callable[[Any, Any, Dict[str, Any], Dict[str, Any]], Dict[str, Any]]
    ] = None,
    primary_fields: Optional[List[str]] = None,
    law_implied_audits: bool = True,
    enable_omit_messages: bool = True,
    best_effort_partial: bool = True,
    coord_dimension: int = 1,
) -> "ResidualEngine":
    """
    Build a law-first :class:`ResidualEngine` for minimal-input workflows.

    The engine auto-builds identity law specs from ``law_names`` and prepends implied
    groups for those laws. User-provided group specs override inferred groups by
    ``output_key``.
    """
    if coord_dimension not in (1, 2, 3):
        raise ValueError("coord_dimension must be one of {1, 2, 3}")
    law_names = [str(n) for n in (law_names or []) if str(n).strip()]
    if not law_names:
        raise ValueError("build_minimal_residual_engine requires at least one law name")
    laws = [build_law_spec_identity(n) for n in law_names]
    implied_groups = implied_group_specs_for_laws(law_names)
    merged_groups = merge_implied_groups_first(implied_groups, list(groups or []))
    eff_user_fns = dict(_default_minimal_user_fns())
    eff_user_fns.update(dict(user_fns or {}))
    return ResidualEngine(
        constants=constants,
        laws=laws,
        groups=merged_groups,
        constitutive_audit=constitutive_audit,
        derived_state_chain=derived_state_chain,
        state_builder=state_builder,
        user_fns=eff_user_fns,
        enable_omit_messages=enable_omit_messages,
        primary_fields=primary_fields,
        law_implied_audits=law_implied_audits,
        best_effort_partial=best_effort_partial,
        default_coord_dimension=coord_dimension,
    )


class ResidualEngine:
    """
    Governing laws (Laws.*), optional group specs to enrich state, and model/group closures.

    Entry points:
      - Path A (recommended): provide (model, params, collocation) and a state_builder
        so moju can build state_pred (and derivatives) internally.
      - Path B (advanced): provide state_pred directly.

    Closure policy:
      - **ref_delta** runs when ``state_ref`` is provided **and** :meth:`compute_residuals` is called
        with ``run_mode=\"eval\"`` (default ``run_mode=\"training\"`` ignores ``state_ref`` for
        ref_delta and for the ``data/`` pred−ref block). Unless the spec sets ``include_ref_delta=False``.
      - **implied_delta** runs for **constitutive** specs when ``implied_value_key`` or ``implied_fn``
        is set; omitted if implied is missing.
      - **ref_delta** / **implied_delta** residuals are nondimensional (see ``closure_registry``).
      - A spec with no applicable closure does nothing (optional omit log).
      - Law-linked implied rows (see ``moju.monitor.law_implied_diagnostics``) are prepended when
        ``law_implied_audits`` is true (``MonitorConfig`` default). Use optional ``residual_basename``
        for unique flat keys under each category.

    Audit spec shape (constitutive_audit items):
      {
        "name": "sutherland_mu",               # Models.<name>
        "output_key": "mu",                    # state key for F output (ref_delta / implied_delta)
        "state_map": {"T": "T", "mu0": "mu0", "T0": "T0", "S": "S"},  # fn arg -> state key
      }

    Each :meth:`compute_residuals` log entry may include ``coord_snapshot``: a dict with optional
    keys ``x``, ``y``, ``z``, ``t`` (1D float lists) copied from built state when present, for
    spatial dashboards.

    :attr:`last_residuals` references the nested dict from the latest successful
    :meth:`compute_residuals` (``None`` after :meth:`clear_log`) so :func:`visualize` can run as
    ``visualize(engine.log, engine=engine)`` without a separate ``residuals=`` argument.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        constants: Optional[Dict[str, Any]] = None,
        laws: Optional[List[Dict[str, Any]]] = None,
        groups: Optional[List[Dict[str, Any]]] = None,
        *,
        constitutive_audit: Optional[List[Dict[str, Any]]] = None,
        constitutive_custom: Optional[List[Dict[str, Any]]] = None,
        derived_state_chain: Optional[List[Dict[str, Any]]] = None,
        state_builder: Optional[
            Callable[[Any, Any, Dict[str, Any], Dict[str, Any]], Dict[str, Any]]
        ] = None,
        user_fns: Optional[Dict[str, Callable[..., Any]]] = None,
        enable_omit_messages: bool = True,
        primary_fields: Optional[List[str]] = None,
        law_implied_audits: bool = True,
        best_effort_partial: bool = False,
        default_coord_dimension: int = 1,
    ):
        law_implied_enabled = bool(law_implied_audits)
        # MonitorConfig convenience
        if config is not None:
            from moju.monitor.config import MonitorConfig, audit_spec_to_engine_dict

            if isinstance(config, MonitorConfig):
                constants = config.constants
                laws = config.laws
                groups = config.groups
                constitutive_audit = [audit_spec_to_engine_dict(s) for s in config.constitutive_audit]
                constitutive_custom = config.constitutive_custom
                derived_state_chain = list(config.derived_state_chain or [])
                primary_fields = list(config.primary_fields)
                law_implied_enabled = bool(config.law_implied_audits)
                if config.state_builder is not None and state_builder is None:
                    state_builder = config.state_builder
            else:
                raise TypeError("config must be a MonitorConfig")

        self.constants = dict(constants or {})
        self.laws_spec = list(laws or [])
        self.groups_spec = list(groups or [])
        self.constitutive_audit = list(constitutive_audit or [])
        li_c, _li_s = merge_law_implied_audit_specs(self.laws_spec, enabled=law_implied_enabled)
        mc, rc = merge_fragment_law_implied_audit_specs(li_c, self.constitutive_audit)
        self.constitutive_audit = mc + rc
        self.constitutive_custom = list(constitutive_custom or [])
        self.derived_state_chain = list(derived_state_chain or [])
        self.state_builder = state_builder
        self.user_fns = dict(user_fns or {})
        self.enable_omit_messages = bool(enable_omit_messages)
        self.primary_fields = list(primary_fields or ["T", "u", "v", "w", "p", "rho"])
        self.best_effort_partial = bool(best_effort_partial)
        if default_coord_dimension not in (1, 2, 3):
            raise ValueError("default_coord_dimension must be one of {1, 2, 3}")
        self.default_coord_dimension = int(default_coord_dimension)

        # Config-time validation (low effort)
        def _validate_specs(
            specs: Sequence[Dict[str, Any]],
            registry: Dict[str, Any],
            category: str,
        ) -> None:
            for spec in specs:
                if "name" not in spec:
                    raise ValueError(f"{category} spec missing 'name'")
                name = spec["name"]
                pred_fn_key = spec.get("pred_fn_key")
                pred_state_map = spec.get("pred_state_map")
                reg = registry.get(name)
                if reg is None and pred_fn_key is None:
                    raise ValueError(f"{category} spec name {name!r} is not registered")
                if "output_key" not in spec:
                    raise ValueError(f"{category}:{name} missing 'output_key'")
                if "state_map" not in spec or not isinstance(spec["state_map"], dict):
                    raise ValueError(f"{category}:{name} missing 'state_map' dict")
                if pred_fn_key is not None:
                    if category != "constitutive":
                        raise ValueError(
                            f"{category}:{name} pred_fn_key is only supported for constitutive specs"
                        )
                    if not isinstance(pred_fn_key, str) or not pred_fn_key.strip():
                        raise ValueError(f"{category}:{name} pred_fn_key must be a non-empty string")
                    if pred_state_map is None or not isinstance(pred_state_map, dict):
                        raise ValueError(f"{category}:{name} pred_state_map missing dict")
                    if not pred_state_map:
                        raise ValueError(f"{category}:{name} pred_state_map must be non-empty")
                else:
                    _, arg_names = reg
                    missing_args = [an for an in arg_names if an not in spec["state_map"]]
                    if missing_args:
                        raise ValueError(f"{category}:{name} state_map missing args: {missing_args}")
                ivk = spec.get("implied_value_key")
                ifn = spec.get("implied_fn")
                if ivk and ifn is not None:
                    raise ValueError(
                        f"{category}:{name} use only one of implied_value_key and implied_fn, not both"
                    )
        _validate_specs(self.constitutive_audit, MODEL_FNS, "constitutive")

        self._log: List[Dict[str, Any]] = []
        self._index = 0
        self._last_residuals: Optional[Dict[str, Any]] = None

    @property
    def log(self) -> List[Dict[str, Any]]:
        return self._log

    @property
    def last_residuals(self) -> Optional[Dict[str, Any]]:
        """Nested residual dict from the latest successful :meth:`compute_residuals` (same object)."""
        return self._last_residuals

    def clear_log(self) -> None:
        """Remove all logged steps and reset the step counter.

        Safe to call between training runs; does not alter engine configuration.
        """
        self._log.clear()
        self._index = 0
        self._last_residuals = None

    def _state_builder(
        self,
        state_pred: Dict[str, Any],
        constants_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        c = self.constants if constants_override is None else constants_override
        return _build_state(state_pred, c, self.groups_spec)

    def compute_residuals(
        self,
        state_pred: Optional[Dict[str, Any]] = None,
        state_ref: Optional[Dict[str, Any]] = None,
        *,
        model: Any = None,
        params: Any = None,
        collocation: Optional[Dict[str, Any]] = None,
        log_to_python: bool = True,
        auto_path_b_derivatives: Any = False,
        fill_law_fd: bool = False,
        run_mode: str = "training",
    ) -> Dict[str, Any]:
        """
        Compute residuals.

        Path A: pass (model, params, collocation) and configure engine.state_builder.
        Path B: pass state_pred directly.

        ``run_mode``:
          - ``\"training\"`` (default): **ref_delta** and ``data/`` pred−ref are skipped even if
            ``state_ref`` is passed (use for the optimization loop).
          - ``\"eval\"``: reference **ref_delta** and ``data/`` run when configured.

        If ``auto_path_b_derivatives`` is True, uses default ``PathBGridConfig``; if a
        ``PathBGridConfig`` instance, uses that layout. When ``fill_law_fd`` is also True, missing
        **registered** ``Laws.*`` inputs (e.g. ``phi_laplacian``, ``u_grad``) are filled from
        primitives on the same grid via finite differences (see ``law_fd_recipes``).
        Warnings are appended to the log ``inferred`` list when enabled.

        If ``fill_law_fd`` is True, ``auto_path_b_derivatives`` must also be enabled.
        """
        if run_mode not in ("training", "eval"):
            raise ValueError("run_mode must be 'training' or 'eval'")
        residuals: Dict[str, Any] = {"laws": {}}
        pb_warn: List[str] = []

        if state_pred is None:
            if self.state_builder is None:
                raise ValueError("Path A requires ResidualEngine(state_builder=...)")
            if model is None or params is None or collocation is None:
                raise ValueError("Path A requires model, params, and collocation")
            state_pred = self.state_builder(model, params, collocation, self.constants)

        omitted_msgs: List[str] = []
        inferred_msgs: List[str] = []

        def _maybe_log_omit(msg: str) -> None:
            if self.enable_omit_messages:
                omitted_msgs.append(msg)

        def _maybe_log_infer(msg: str) -> None:
            if self.enable_omit_messages:
                inferred_msgs.append(msg)

        _auto_implied_supported, _auto_implied_manual = supported_auto_implied_laws_for(
            self.laws_spec
        )
        if _auto_implied_manual:
            _maybe_log_infer(
                "law_implied_audits: user-specified constitutive audits required for laws without "
                f"auto implied mapping: {', '.join(_auto_implied_manual)}"
            )

        ref_for_audits = state_ref if run_mode == "eval" else None
        if run_mode == "training" and state_ref is not None:
            _maybe_log_omit(
                "state_ref ignored until run_mode='eval' (ref_delta and data/ are eval-only)"
            )

        state_for_groups = dict(state_pred)
        if self.derived_state_chain:
            from moju.monitor.derived_state_chain import apply_derived_state_chain

            state_for_groups, _dwarn = apply_derived_state_chain(
                state_for_groups,
                self.constants,
                self.derived_state_chain,
            )
            for w in _dwarn:
                _maybe_log_infer(f"derived_state: {w}")

        def _materialize_user_keys(
            state: Dict[str, Any],
            *,
            needed_keys: Set[str],
            stage: str,
        ) -> None:
            """
            Best-effort: for each missing required key in ``needed_keys``, if a callable is present in
            ``self.user_fns`` under that key, compute and add it to ``state`` (and log inferred).
            Keys are treated as **output state keys**: e.g. key ``'k'`` uses ``user_fns['k']``.
            """
            if not self.user_fns:
                return
            # Multi-pass so dependencies like alpha(k,rho,cp) can resolve after k and rho are built.
            max_passes = max(2, len(needed_keys) * 2)
            missing_inputs_last: Dict[str, List[str]] = {}
            for _ in range(max_passes):
                progressed = False
                for k in sorted(needed_keys):
                    if k in state or k in self.constants:
                        continue
                    fn = self.user_fns.get(k)
                    if fn is None:
                        continue
                    try:
                        sig = inspect.signature(fn)
                    except (TypeError, ValueError):
                        sig = None
                    if sig is None:
                        raise TypeError(
                            f"user_fns[{k!r}] must be a Python callable with an inspectable signature"
                        )
                    kwargs: Dict[str, Any] = {}
                    missing: List[str] = []
                    for p in sig.parameters.values():
                        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                            raise TypeError(
                                f"user_fns[{k!r}] must not use *args/**kwargs; declare explicit inputs"
                            )
                        pn = str(p.name)
                        v = state.get(pn)
                        if v is None:
                            v = self.constants.get(pn)
                        if v is None:
                            missing.append(pn)
                        else:
                            kwargs[pn] = v
                    if missing:
                        # Not computable yet; try again after other keys materialize.
                        missing_inputs_last[str(k)] = list(missing)
                        continue
                    out_v = fn(**kwargs)
                    if out_v is None:
                        continue
                    if isinstance(out_v, str):
                        raise TypeError(
                            f"user_fns[{k!r}] returned a string; expected numeric array-like"
                        )
                    state[k] = out_v
                    progressed = True
                    _maybe_log_infer(f"user_fns({stage}): materialized {k!r} from callable inputs")
                    if str(k) in missing_inputs_last:
                        missing_inputs_last.pop(str(k), None)
                if not progressed:
                    break

            # If a required key has a callable but still can't be computed, raise a targeted error.
            blocked = [
                k
                for k in sorted(needed_keys)
                if k not in state
                and k not in self.constants
                and k in self.user_fns
                and k in missing_inputs_last
            ]
            if blocked and not self.best_effort_partial:
                avail = sorted(set(state.keys()) | set(self.constants.keys()))
                lines = [
                    f"user_fns could not materialize required keys at stage={stage!r}:"
                ]
                for k in blocked[:32]:
                    miss = missing_inputs_last.get(k) or []
                    lines.append(f"- {k!r}: missing inputs {miss}")
                lines.append("")
                lines.append(f"Available keys (state ∪ constants): {avail[:64]}")
                if len(avail) > 64:
                    lines.append(f"... (+{len(avail) - 64} more)")
                raise KeyError("\n".join(lines))
            if blocked and self.best_effort_partial:
                _maybe_log_omit(
                    f"user_fns({stage}) unresolved outputs in best_effort_partial mode: {blocked}"
                )

        # Materialize keys needed to run groups (inputs only).
        group_needed: Set[str] = set()
        for spec in self.groups_spec:
            sm = spec.get("state_map") or {}
            if isinstance(sm, dict):
                group_needed |= set(str(v) for v in sm.values())
        # Include transitive deps: a needed key (e.g. 'alpha') may require other user_fns outputs
        # (e.g. 'k', 'rho') to be materialized first.
        group_needed |= set(self.user_fns.keys())
        _materialize_user_keys(state_for_groups, needed_keys=group_needed, stage="pre_groups")

        if self.best_effort_partial:
            state_pred_built = _build_state_best_effort(
                state_for_groups,
                self.constants,
                self.groups_spec,
                log_skip=_maybe_log_omit,
            )
        else:
            state_pred_built = self._state_builder(state_for_groups)
        merged = {**self.constants, **state_pred_built}

        def _state_ref_raw_after_derived(sr: Dict[str, Any]) -> Dict[str, Any]:
            s0 = dict(sr)
            if self.derived_state_chain:
                from moju.monitor.derived_state_chain import apply_derived_state_chain

                s0, wr = apply_derived_state_chain(s0, self.constants, self.derived_state_chain)
                for x in wr:
                    _maybe_log_infer(f"derived_state(ref): {x}")
            return s0

        def _merge_state_ref(sr: Dict[str, Any]) -> Dict[str, Any]:
            return {**self._state_builder(_state_ref_raw_after_derived(sr)), **self.constants}

        if fill_law_fd and not auto_path_b_derivatives:
            raise ValueError(
                "fill_law_fd=True requires auto_path_b_derivatives=True or a PathBGridConfig"
            )

        if auto_path_b_derivatives:
            from moju.monitor.path_b_derivatives import PathBGridConfig, fill_path_b_derivatives

            if auto_path_b_derivatives is True:
                grid = PathBGridConfig(spatial_dimension=self.default_coord_dimension)
                _maybe_log_infer(
                    f"path_b_derivatives: using engine default coord dimension "
                    f"{self.default_coord_dimension}D"
                )
            elif isinstance(auto_path_b_derivatives, PathBGridConfig):
                grid = auto_path_b_derivatives
            else:
                raise TypeError(
                    "auto_path_b_derivatives must be False, True, or a PathBGridConfig instance"
                )
            state_pred_built, pb_warn = fill_path_b_derivatives(
                state_pred_built,
                constitutive_audit=self.constitutive_audit,
                laws_spec=self.laws_spec,
                constants=self.constants,
                grid=grid,
                copy=False,
                fill_law_recipes=bool(fill_law_fd),
            )
            merged = {**self.constants, **state_pred_built}
            for w in pb_warn:
                _maybe_log_infer(f"path_b_derivatives: {w}")
                if (
                    self.best_effort_partial
                    and self.default_coord_dimension > 1
                    and ("coord" in w.lower() or "meshgrid" in w.lower() or "separable" in w.lower())
                ):
                    _maybe_log_omit(
                        f"path_b_derivatives hint: configured coord_dimension="
                        f"{self.default_coord_dimension} requires compatible coordinates"
                    )

        # Materialize any remaining required keys for laws/groups/audits (after groups + FD).
        needed_all = self.required_state_keys()
        _materialize_user_keys(state_pred_built, needed_keys=set(needed_all), stage="post_fd")
        merged = {**self.constants, **state_pred_built}

        unresolved_dependencies: List[Dict[str, Any]] = []

        for spec in self.laws_spec:
            name = spec["name"]
            state_map = spec["state_map"]
            try:
                kwargs = _kwargs_from_state(
                    merged, self.constants, state_map, law_context=str(name)
                )
            except KeyError as err:
                if self.best_effort_partial:
                    missing = sorted(
                        {
                            str(v)
                            for v in state_map.values()
                            if v not in merged and v not in self.constants
                        }
                    )
                    unresolved_dependencies.append(
                        {
                            "stage": "law",
                            "name": str(name),
                            "missing_keys": missing,
                        }
                    )
                    _maybe_log_omit(
                        f"law:{name} skipped in best_effort_partial mode: missing {missing}"
                    )
                    continue
                if pb_warn:
                    tail = "\n".join(f"  - {w}" for w in pb_warn[:32])
                    raise KeyError(
                        f"{err.args[0]}\n\nFinite-difference / law-FD fill reported:\n{tail}"
                    ) from err
                raise
            fn = _get_fn(spec, Laws)
            residuals["laws"][name] = fn(**kwargs)

        def _run_specs(
            specs: Sequence[Dict[str, Any]],
            *,
            registry: Dict[str, Any],
            category: str,
        ) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for spec in specs:
                name = spec["name"]
                output_key = spec.get("output_key")
                state_map = spec.get("state_map") or {}
                has_implied = bool(spec.get("implied_value_key")) or spec.get("implied_fn") is not None
                missing = sorted(
                    {
                        str(v)
                        for v in state_map.values()
                        if v not in merged and v not in self.constants
                    }
                )
                if missing and self.best_effort_partial:
                    unresolved_dependencies.append(
                        {
                            "stage": category,
                            "name": str(name),
                            "missing_keys": missing,
                        }
                    )
                    _maybe_log_omit(
                        f"{category}:{name} skipped in best_effort_partial mode: missing {missing}"
                    )
                    continue
                if ref_for_audits is None and not has_implied:
                    _maybe_log_omit(
                        f"{category}:{name} omitted: no ref_delta or implied_delta applicable"
                    )
                    continue

                fn = None
                arg_names: Sequence[str] = ()
                # Optional: callable-backed constitutive audit for implied/ref deltas.
                pred_fn_key = spec.get("pred_fn_key")
                if pred_fn_key is not None:
                    if category != "constitutive":
                        continue
                    fn = self.user_fns.get(str(pred_fn_key))
                    if fn is None:
                        raise KeyError(
                            f"constitutive:{name} pred_fn_key {pred_fn_key!r} not found in user_fns; "
                            f"available: {sorted(self.user_fns.keys())}"
                        )
                    pred_state_map = spec.get("pred_state_map") or {}
                    if not isinstance(pred_state_map, dict) or not pred_state_map:
                        raise ValueError(f"constitutive:{name} pred_state_map must be a non-empty dict")
                    state_map = pred_state_map
                    arg_names = list(str(a) for a in pred_state_map.keys())
                else:
                    reg = registry.get(name)
                    if reg is None:
                        # unknown function name -> omit silently (config validation should catch)
                        continue
                    fn, arg_names = reg
                base = spec.get("residual_basename") or name

                if (
                    ref_for_audits is not None
                    and output_key is not None
                    and spec.get("include_ref_delta", True)
                ):
                    arr = compute_ref_delta(
                        fn=fn,
                        arg_names=arg_names,
                        output_key=output_key,
                        state_map=state_map,
                        state_pred=merged,
                        state_ref=_merge_state_ref(ref_for_audits),
                        constants=self.constants,
                        ref_delta_ref_key=spec.get("ref_delta_ref_key"),
                    )
                    if arr is not None:
                        out[f"{base}/ref_delta"] = jnp.asarray(arr)

                if has_implied:
                    arr = compute_implied_delta(
                        fn=fn,
                        arg_names=arg_names,
                        state_map=state_map,
                        state_pred=merged,
                        constants=self.constants,
                        implied_value_key=spec.get("implied_value_key"),
                        implied_fn=spec.get("implied_fn"),
                        output_key=output_key,
                        implied_delta_ref_key=spec.get("implied_delta_ref_key"),
                    )
                    if arr is not None:
                        out[f"{base}/implied_delta"] = jnp.asarray(arr)

            return out

        if self.constitutive_audit or self.constitutive_custom:
            c = _run_specs(self.constitutive_audit, registry=MODEL_FNS, category="constitutive")
            if self.constitutive_custom:
                for spec in self.constitutive_custom:
                    cname = spec["name"]
                    arr = spec["fn"](merged, self.constants)
                    if arr is not None:
                        c[f"custom/{cname}"] = jnp.asarray(arr)
            if c:
                residuals["constitutive"] = c

        if ref_for_audits is not None:
            state_ref_built = self._state_builder(_state_ref_raw_after_derived(ref_for_audits))
            common = set(state_pred_built.keys()) & set(state_ref_built.keys())
            residuals["data"] = {
                k: jnp.asarray(state_ref_built[k]) - jnp.asarray(state_pred_built[k])
                for k in common
            }

        flat = _flatten_residual_dict(residuals)
        rms_per_key = _rms_per_key(flat, to_python=log_to_python)
        state_ref_built_for_scale = None
        if ref_for_audits is not None:
            state_ref_built_for_scale = self._state_builder(
                _state_ref_raw_after_derived(ref_for_audits)
            )
        scale_per_key = _state_derived_scale_per_key(
            flat.keys(),
            merged,
            self.laws_spec,
            self.constitutive_audit,
            state_ref_built_for_scale,
            to_python=log_to_python,
        )
        entry: Dict[str, Any] = {
            "index": self._index,
            "rms": rms_per_key,
            "scale": scale_per_key,
            "run_mode": run_mode,
        }
        if omitted_msgs:
            entry["omitted"] = omitted_msgs
        if inferred_msgs:
            entry["inferred"] = inferred_msgs
        if unresolved_dependencies:
            entry["unresolved_dependencies"] = unresolved_dependencies
        if "coord_snapshot" not in entry:
            cs = _coord_snapshot_from_merged(merged)
            if cs:
                entry["coord_snapshot"] = cs
        self._log.append(entry)
        self._index += 1
        self._last_residuals = residuals
        return residuals

    def required_state_keys(
        self,
        *,
        include_groups: bool = True,
        include_laws: bool = True,
        include_audits: bool = True,
    ) -> Set[str]:
        keys: Set[str] = set()
        if include_laws:
            for spec in self.laws_spec:
                keys |= set((spec.get("state_map") or {}).values())
        if include_groups:
            for spec in self.groups_spec:
                keys |= set((spec.get("state_map") or {}).values())
                ok = spec.get("output_key")
                if ok:
                    keys.add(ok)
        if include_audits:
            for spec in self.constitutive_audit:
                keys |= set((spec.get("state_map") or {}).values())
                ok = spec.get("output_key")
                if ok:
                    keys.add(ok)
                ivk = spec.get("implied_value_key")
                if ivk:
                    keys.add(ivk)
        keys |= all_ref_keys_from_chain(self.derived_state_chain)
        keys -= keys_produced_by_chain(self.derived_state_chain)
        return keys



def list_constitutive_models():
    from moju.monitor.closure_registry import list_models
    return list_models()


def list_scaling_closure_ids():
    """Registered **Groups.*** names (same as :func:`moju.monitor.closure_registry.list_groups`)."""
    from moju.monitor.closure_registry import list_groups

    return list_groups()
