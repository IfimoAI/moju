"""One compiled ``jax.jit`` / ``jax.vmap`` call for same-shaped ``evaluate`` candidates."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from moju.monitor.derivative_provenance import (
    MissingSuppliedDerivativeError,
    label_provenance,
    law_derivative_inputs,
)
from moju.monitor.derived_state_chain import keys_produced_by_chain
from moju.monitor.law_scale_recipes import _resolve_scale_recipe, _term_rms_array


def evaluate_batched(
    engine: Any,
    candidates: Sequence[Mapping[str, Any]],
    refs: Sequence[Optional[Mapping[str, Any]]],
    *,
    run_mode: str,
    r_ref: Optional[Dict[str, float]],
    return_residuals: bool,
) -> Optional[List[Dict[str, Any]]]:
    """
    Score every candidate in one compiled call.

    Returns ``None`` when the batch cannot be traced, so the caller keeps the Python loop.
    ``derivatives="supplied_only"`` still raises :class:`MissingSuppliedDerivativeError` before
    that call when a required derivative is absent.
    """
    if not candidates:
        return []
    if engine.best_effort_partial or engine.user_fns:
        return None
    if not _uniform(candidates) or not _uniform_refs(refs):
        return None
    _require_supplied_derivatives(engine, candidates)

    from moju.monitor.auditor import (
        DEFAULT_NONDIM_R_NORM_SCALE_K,
        _SCALE_EPS,
        _default_unit_scale_k,
        _get_fn,
        _key_uses_worst_point_admissibility,
        _r_eff_scalar,
        _r_max_scalar,
        _rms_scalar,
        audit,
    )
    from moju.monitor.closure_registry import MODEL_FNS, compute_implied_delta, compute_ref_delta
    from moju.piratio.groups import Groups
    from moju.piratio.laws import Laws

    use_ref = run_mode == "eval" and any(r is not None for r in refs)
    plan = _Plan.build(
        engine,
        candidates[0],
        refs[0] if use_ref else None,
        run_mode=run_mode,
        use_ref=use_ref,
        get_fn=_get_fn,
        laws_cls=Laws,
        groups_cls=Groups,
        model_fns=MODEL_FNS,
        default_scale=_default_unit_scale_k(to_python=True),
        floor=float(DEFAULT_NONDIM_R_NORM_SCALE_K),
        scale_eps=float(_SCALE_EPS),
    )
    stacked_state = _stack(candidates)
    stacked_ref = _stack([r or {} for r in refs]) if use_ref else None

    def one(state: Dict[str, Any], ref: Dict[str, Any]) -> Dict[str, Any]:
        return _score_one(
            state,
            ref if use_ref else None,
            plan,
            r_eff=_r_eff_scalar,
            r_max=_r_max_scalar,
            rms_scalar=_rms_scalar,
            implied=compute_implied_delta,
            ref_delta=compute_ref_delta,
        )

    try:
        if use_ref:
            compiled = jax.jit(jax.vmap(one))
            out = compiled(stacked_state, stacked_ref)
        else:
            compiled = jax.jit(jax.vmap(lambda state: one(state, {})))
            out = compiled(stacked_state)
    except MissingSuppliedDerivativeError:
        raise
    except Exception:
        return None

    flags = out["recipe_ok"]
    if flags:
        ok = jnp.stack([jnp.asarray(v) for v in flags.values()])
        if not bool(jnp.all(ok)):
            return None

    provenance = _provenance(engine, candidates[0])
    reports: List[Dict[str, Any]] = []
    for i, _state in enumerate(candidates):
        residuals = _unstack(out["residuals"], i)
        entry = _entry(
            i,
            out,
            residuals,
            plan,
            provenance,
            run_mode=run_mode,
            worst=_key_uses_worst_point_admissibility,
        )
        report = audit([entry], r_ref=r_ref)
        if return_residuals:
            report["residuals"] = residuals
        reports.append(report)
    return reports


class _Plan:
    """Callables and static scoring resolved once, outside the compiled function."""

    def __init__(self) -> None:
        self.constants: Dict[str, Any] = {}
        self.derived: List[Dict[str, Any]] = []
        self.dimensional: bool = False
        self.laws_spec: List[Dict[str, Any]] = []
        self.groups: List[Tuple[Any, Dict[str, str], str]] = []
        self.laws: List[Tuple[str, Any, Dict[str, str], Dict[str, Any]]] = []
        self.audits: List[Dict[str, Any]] = []
        self.customs: List[Tuple[str, Any]] = []
        self.bounds: List[Any] = []
        self.auto_scale: List[Tuple[str, Any, Dict[str, Any], bool]] = []
        self.static_scale: Dict[str, float] = {}
        self.static_source: Dict[str, str] = {}
        self.scoring: Dict[str, Any] = {}
        self.data_keys: Tuple[str, ...] = ()
        self.floor: float = 1e-2
        self.scale_eps: float = 1e-12
        self.law_scale_mode: str = "auto"
        self.state_units: str = "nondimensional"
        self.nondim_scales: Any = None

    @classmethod
    def build(cls, engine: Any, sample: Mapping[str, Any], ref: Optional[Mapping[str, Any]], **kw: Any) -> "_Plan":
        plan = cls()
        plan.constants = dict(engine.constants)
        plan.derived = list(engine.derived_state_chain or [])
        plan.dimensional = engine.state_units == "dimensional"
        plan.laws_spec = list(engine.laws_spec)
        plan.floor = float(kw["floor"])
        plan.scale_eps = float(kw["scale_eps"])
        plan.law_scale_mode = engine.law_scale_mode
        plan.state_units = engine.state_units
        plan.nondim_scales = engine.nondim_scales
        get_fn = kw["get_fn"]
        for spec in engine.groups_spec:
            plan.groups.append((get_fn(spec, kw["groups_cls"]), dict(spec.get("state_map") or {}), str(spec.get("output_key") or spec["name"])))
        for spec in engine.laws_spec:
            name = str(spec["name"])
            plan.laws.append((name, get_fn(spec, kw["laws_cls"]), dict(spec.get("state_map") or {}), spec))
        for spec in engine.constitutive_audit:
            reg = kw["model_fns"].get(spec["name"])
            if reg is None and spec.get("pred_fn_key") is None:
                continue
            plan.audits.append({"spec": spec, "reg": reg})
        for spec in engine.constitutive_custom:
            plan.customs.append((str(spec["name"]), spec["fn"]))
        plan.bounds = list(engine.bound_checks)
        if ref is not None:
            plan.data_keys = tuple(sorted(set(sample) & set(ref)))
        scoring = engine._scoring_by_key
        default_scale = float(kw["default_scale"])
        for spec in engine.laws_spec:
            name = str(spec["name"])
            key = f"laws/{name}"
            decl = scoring.get(key)
            if decl is not None:
                plan.static_scale[key] = float(decl.scale) if decl.scale is not None else default_scale
                plan.static_source[key] = "declared"
                plan.scoring[key] = decl
            elif engine.law_scale_mode == "fixed":
                plan.static_scale[key] = default_scale
                plan.static_source[key] = "fixed"
            else:
                recipe, is_user = _resolve_scale_recipe(name, spec)
                plan.auto_scale.append((key, recipe, spec, is_user))
                plan.static_source[key] = "user_recipe" if is_user else "auto"
        for spec in engine.constitutive_custom:
            key = f"constitutive/custom/{spec['name']}"
            decl = scoring.get(key)
            if decl is not None:
                plan.static_scale[key] = float(decl.scale) if decl.scale is not None else default_scale
                plan.static_source[key] = "declared"
                plan.scoring[key] = decl
            else:
                plan.static_scale[key] = default_scale
                plan.static_source[key] = "state_derived"
        for check in engine.bound_checks:
            key = f"constitutive/bound/{check.name}/violation"
            decl = scoring.get(key)
            plan.static_scale[key] = float(decl.scale) if decl is not None and decl.scale is not None else default_scale
            plan.static_source[key] = "declared"
            if decl is not None:
                plan.scoring[key] = decl
        for key in plan.data_keys:
            plan.static_source[f"data/{key}"] = "state_derived"
        return plan


def _score_one(state, ref, plan: _Plan, *, r_eff, r_max, rms_scalar, implied, ref_delta) -> Dict[str, Any]:
    merged: Dict[str, Any] = {**plan.constants, **dict(state)}
    if plan.derived:
        from moju.monitor.derived_state_chain import apply_derived_state_chain

        merged, _warns = apply_derived_state_chain(merged, plan.constants, plan.derived)
    if plan.dimensional:
        from moju.monitor.nondim_inference import infer_nondim_scales
        from moju.piratio.nondim import dimensional_to_nd

        names = [str(s["name"]) for s in plan.laws_spec]
        scales, _src = infer_nondim_scales(names, merged, plan.constants, law_specs=plan.laws_spec)
        if plan.nondim_scales is not None:
            scales = plan.nondim_scales
        merged = dimensional_to_nd(merged, scales, warn_unknown=False)
    for fn, state_map, output_key in plan.groups:
        merged[output_key] = fn(**_kwargs(merged, plan.constants, state_map))
    laws: Dict[str, Any] = {}
    for name, fn, state_map, _spec in plan.laws:
        laws[name] = fn(**_kwargs(merged, plan.constants, state_map))
    constitutive: Dict[str, Any] = {}
    for audit_spec in plan.audits:
        spec = audit_spec["spec"]
        reg = audit_spec["reg"]
        if reg is None:
            continue
        fn, arg_names = reg
        state_map = dict(spec.get("state_map") or {})
        base = spec.get("residual_basename") or spec["name"]
        output_key = spec.get("output_key")
        has_implied = bool(spec.get("implied_value_key")) or spec.get("implied_fn") is not None
        if ref is not None and output_key is not None and spec.get("include_ref_delta", True):
            arr = ref_delta(
                fn=fn,
                arg_names=arg_names,
                output_key=output_key,
                state_map=state_map,
                state_pred=merged,
                state_ref={**plan.constants, **dict(ref)},
                constants=plan.constants,
                ref_delta_ref_key=spec.get("ref_delta_ref_key"),
            )
            if arr is not None:
                constitutive[f"{base}/ref_delta"] = jnp.asarray(arr)
        if has_implied:
            arr = implied(
                fn=fn,
                arg_names=list(arg_names),
                state_map=state_map,
                state_pred=merged,
                constants=plan.constants,
                implied_value_key=spec.get("implied_value_key"),
                implied_fn=spec.get("implied_fn"),
                output_key=output_key,
            )
            if arr is not None:
                constitutive[f"{base}/implied_delta"] = jnp.asarray(arr)
    for name, fn in plan.customs:
        constitutive[f"custom/{name}"] = jnp.asarray(fn(merged, plan.constants))
    for check in plan.bounds:
        if check.value_fn is not None:
            value = check.value_fn(merged, plan.constants)
        else:
            value = merged.get(check.value_key)
            if value is None:
                value = plan.constants.get(check.value_key)
        value = jnp.asarray(value)
        lower = _bound_array(check.lower, merged, plan.constants)
        upper = _bound_array(check.upper, merged, plan.constants)
        viol = jnp.zeros_like(value)
        if lower is not None:
            viol = viol + jax.nn.relu(jnp.asarray(lower) - value)
        if upper is not None:
            viol = viol + jax.nn.relu(value - jnp.asarray(upper))
        constitutive[f"bound/{check.name}/violation"] = viol / float(check.scale)
    data: Dict[str, Any] = {}
    if ref is not None:
        for key in plan.data_keys:
            data[key] = jnp.asarray(ref[key]) - jnp.asarray(state[key])
    residuals: Dict[str, Any] = {}
    if laws:
        residuals["laws"] = laws
    if constitutive:
        residuals["constitutive"] = constitutive
    if data:
        residuals["data"] = data
    flat = _flat(residuals)
    rms = {k: r_eff(v) for k, v in flat.items()}
    rmax = {k: r_max(v) for k, v in flat.items()}
    dynamic_scale: Dict[str, Any] = {}
    recipe_ok: Dict[str, Any] = {}
    for key, recipe, spec, _is_user in plan.auto_scale:
        if recipe is None:
            raw = _generic_scale(merged, plan.constants, spec)
        else:
            raw = jnp.asarray(recipe(merged, plan.constants, spec, plan.nondim_scales))
        ok = jnp.isfinite(raw) & (raw > 0)
        dynamic_scale[key] = jnp.where(ok, jnp.maximum(raw, plan.floor), jnp.asarray(plan.floor))
        recipe_ok[key] = ok
    if ref is not None:
        for key in plan.data_keys:
            dynamic_scale[f"data/{key}"] = plan.scale_eps + rms_scalar(jnp.ravel(jnp.asarray(ref[key])))
    for key, value in plan.static_scale.items():
        if key not in dynamic_scale:
            dynamic_scale[key] = jnp.asarray(value)
    return {"residuals": residuals, "rms": rms, "r_max": rmax, "scale": dynamic_scale, "recipe_ok": recipe_ok}


def _generic_scale(merged: Mapping[str, Any], constants: Mapping[str, Any], spec: Mapping[str, Any]) -> jnp.ndarray:
    parts = []
    for key in (spec.get("state_map") or {}).values():
        value = merged.get(key)
        if value is None:
            value = constants.get(key)
        if value is not None:
            parts.append(_term_rms_array(value))
    if not parts:
        return jnp.asarray(jnp.nan)
    return jnp.max(jnp.stack(parts))


def _kwargs(merged: Mapping[str, Any], constants: Mapping[str, Any], state_map: Mapping[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for arg, key in state_map.items():
        value = merged.get(key)
        if value is None:
            value = constants.get(key)
        if value is None:
            raise KeyError(key)
        out[str(arg)] = value
    return out


def _bound_array(bound: Any, merged: Mapping[str, Any], constants: Mapping[str, Any]) -> Any:
    if bound is None:
        return None
    if callable(bound):
        return bound(dict(merged), dict(constants))
    return bound


def _flat(residuals: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for category, content in residuals.items():
        if not isinstance(content, dict):
            continue
        for name, value in content.items():
            out[f"{category}/{name}"] = value
    return out


def _entry(index, out, residuals, plan: _Plan, provenance: Dict[str, str], *, run_mode: str, worst) -> Dict[str, Any]:
    flat_keys = list(_flat(residuals))
    rms = {k: float(out["rms"][k][index]) for k in flat_keys}
    r_max = {
        k: float(out["r_max"][k][index])
        for k in flat_keys
        if worst(k, plan.scoring)
    }
    scale = dict(plan.static_scale)
    source = dict(plan.static_source)
    for key in out["scale"]:
        if key in plan.static_scale:
            continue
        scale[key] = float(out["scale"][key][index])
    for key, _recipe, _spec, is_user in plan.auto_scale:
        ok = bool(out["recipe_ok"][key][index])
        if ok:
            source[key] = "user_recipe" if is_user else "auto"
        else:
            source[key] = "auto_fallback"
            scale[key] = plan.floor
    entry: Dict[str, Any] = {
        "index": index,
        "rms": rms,
        "r_max": r_max,
        "scale": {k: scale[k] for k in flat_keys if k in scale},
        "scale_source": {k: source[k] for k in flat_keys if k in source},
        "run_mode": run_mode,
        "monitor_settings": {"law_scale_mode": plan.law_scale_mode, "state_units": plan.state_units},
    }
    if provenance:
        entry["derivative_provenance"] = dict(provenance)
    from moju.registry import law_sources_for_specs

    law_sources = law_sources_for_specs(plan.laws_spec)
    if law_sources:
        entry["law_sources"] = law_sources
    scoring = {k: plan.scoring[k].to_dict() for k in plan.scoring if k in flat_keys}
    if scoring:
        entry["scoring"] = scoring
    return entry


def _provenance(engine: Any, sample: Mapping[str, Any]) -> Dict[str, str]:
    available = {**engine.constants, **dict(sample)}
    return label_provenance(law_derivative_inputs(engine.laws_spec, engine.state_declarations), available, {})


def _require_supplied_derivatives(engine: Any, candidates: Sequence[Mapping[str, Any]]) -> None:
    if engine.derivatives != "supplied_only":
        return
    produced = keys_produced_by_chain(engine.derived_state_chain)
    inputs = law_derivative_inputs(engine.laws_spec, engine.state_declarations)
    for state in candidates:
        for law, arg, key in inputs:
            if key in produced or key in engine.constants:
                continue
            value = state.get(key)
            if value is None:
                raise MissingSuppliedDerivativeError(law, arg, key)


def _uniform(states: Sequence[Mapping[str, Any]]) -> bool:
    shapes = [_shape_map(s) for s in states]
    return all(s is not None and s == shapes[0] for s in shapes)


def _uniform_refs(refs: Sequence[Optional[Mapping[str, Any]]]) -> bool:
    if all(r is None for r in refs):
        return True
    if any(r is None for r in refs):
        return False
    return _uniform([r for r in refs if r is not None])


def _shape_map(state: Mapping[str, Any]) -> Optional[Dict[str, Tuple[Tuple[int, ...], str]]]:
    out: Dict[str, Tuple[Tuple[int, ...], str]] = {}
    for key, value in state.items():
        if isinstance(value, str):
            return None
        try:
            arr = jnp.asarray(value)
        except (TypeError, ValueError):
            return None
        out[str(key)] = (tuple(arr.shape), str(arr.dtype))
    return out


def _stack(states: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    keys = list(states[0])
    return {k: jnp.stack([jnp.asarray(s[k]) for s in states]) for k in keys}


def _unstack(tree: Any, index: int) -> Any:
    if isinstance(tree, dict):
        return {k: _unstack(v, index) for k, v in tree.items()}
    return tree[index]
