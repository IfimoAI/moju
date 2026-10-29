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

# Incremented once per successful compiled batch invocation (tests assert a single call).
_BATCH_TRACE_COUNT: int = 0


def reset_batch_trace_count() -> None:
    global _BATCH_TRACE_COUNT
    _BATCH_TRACE_COUNT = 0


def get_batch_trace_count() -> int:
    return int(_BATCH_TRACE_COUNT)


def evaluate_batched(
    engine: Any,
    candidates: Sequence[Mapping[str, Any]],
    refs: Sequence[Optional[Mapping[str, Any]]],
    *,
    run_mode: str,
    r_ref: Optional[Dict[str, float]],
    return_residuals: bool,
    regions: Any = None,
    region_names: Optional[Mapping[Any, str]] = None,
    region_min_points: Optional[int] = None,
    implied_metric: Optional[str] = None,
    implied_percentile: Optional[float] = None,
    implied_noise_normalization: Optional[bool] = None,
    require_batch: bool = False,
    nonfinite_policy: Optional[str] = None,
    eval_mask_label: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """
    Score every candidate in one compiled call.

    When ``regions`` is set, region masks are static across the batch and enter the
    jitted function as stacked 0/1 weights ``(n_regions, n_points)``. Numeric
    per-region outputs are computed inside the transform; status strings are
    derived on the host afterward.

    Returns ``None`` when the batch cannot be traced **and** ``require_batch`` is
    False (caller may loop). When ``require_batch`` is True (regions path),
    failures raise instead of returning ``None``.
    ``derivatives="supplied_only"`` still raises :class:`MissingSuppliedDerivativeError`
    before that call when a required derivative is absent.
    """
    global _BATCH_TRACE_COUNT

    def _fail(exc: Optional[BaseException] = None) -> Optional[List[Dict[str, Any]]]:
        if require_batch:
            msg = (
                "evaluate requires the batched jit/vmap path for regions and/or "
                "opt-in implied scoring; batching failed or is unavailable. "
                "Pass evaluate_debug_loop=True only for debugging."
            )
            if exc is not None:
                raise RuntimeError(msg) from exc
            raise RuntimeError(msg)
        return None

    if not candidates:
        return []
    if engine.best_effort_partial or engine.user_fns:
        return _fail()
    if not _uniform(candidates) or not _uniform_refs(refs):
        return _fail()
    _require_supplied_derivatives(engine, candidates)

    from moju import __version__ as _moju_version
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
    from moju.monitor.closure_registry import (
        MODEL_FNS,
        compute_implied_delta_with_debug,
        compute_ref_delta,
    )
    from moju.monitor.implied_scoring import (
        DEFAULT_IMPLIED_METRIC,
        DEFAULT_IMPLIED_PERCENTILE,
        implied_scoring_settings_payload,
        is_default_implied_scoring,
        resolve_implied_scoring_options,
        score_from_r_max,
    )
    from moju.monitor.law_implied_diagnostics import (
        _gate_params,
        materialize_identifiability_meta,
    )
    from moju.monitor.regions import (
        DEFAULT_REGION_MIN_POINTS,
        normalize_regions,
        regional_viscosity_fit_arrays,
        regions_provenance,
        status_from_region_numerics,
        validate_region_min_points,
        weighted_reduce_magnitude_arrays,
    )
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
    scoring_opts = resolve_implied_scoring_options(
        constants=engine.constants,
        engine_defaults={
            "implied_metric": engine.implied_metric,
            "implied_percentile": engine.implied_percentile,
            "implied_noise_normalization": engine.implied_noise_normalization,
        },
        implied_metric=implied_metric,
        implied_percentile=implied_percentile,
        implied_noise_normalization=implied_noise_normalization,
    )
    metric = str(scoring_opts.get("implied_metric", DEFAULT_IMPLIED_METRIC))
    pct = float(scoring_opts.get("implied_percentile", DEFAULT_IMPLIED_PERCENTILE))
    use_noise = bool(scoring_opts.get("implied_noise_normalization", False))
    apply_optin = not is_default_implied_scoring(scoring_opts)
    term_thr, fit_thr, min_frac, min_count = _gate_params(engine.constants)

    region_pack: Optional[Dict[str, Any]] = None
    region_weights = None
    eff_rmin = DEFAULT_REGION_MIN_POINTS
    if regions is not None:
        # Infer reference shape from a residual-sized state field.
        ref_shape = None
        sample = candidates[0]
        for key in ("u", "T", "u_laplacian", "T_laplacian"):
            if key in sample:
                a = jnp.asarray(sample[key])
                if a.ndim >= 2 and a.shape[-1] in (2, 3):
                    ref_shape = a.shape[:-1]
                else:
                    ref_shape = a.shape
                break
        normalized = normalize_regions(
            regions, reference_shape=ref_shape, region_names=region_names
        )
        names = list(normalized["names"])
        weights_list = [jnp.asarray(normalized["weights"][n], dtype=jnp.float32).reshape(-1) for n in names]
        region_weights = jnp.stack(weights_list, axis=0)
        eff_rmin = (
            validate_region_min_points(region_min_points)
            if region_min_points is not None
            else int(
                getattr(engine, "region_min_points", None) or DEFAULT_REGION_MIN_POINTS
            )
        )
        region_pack = {
            "names": names,
            "normalized": normalized,
            "n_points": [int(normalized["n_points"][n]) for n in names],
            "region_min_points": eff_rmin,
            "moju_version": _moju_version,
        }

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
            implied_with_debug=compute_implied_delta_with_debug,
            ref_delta=compute_ref_delta,
            region_weights=region_weights,
            region_min_points=eff_rmin,
            region_metric=metric,
            region_percentile=pct,
            term_threshold=term_thr,
            fit_threshold=fit_thr,
            min_frac=min_frac,
            min_count=min_count,
            apply_optin=apply_optin,
            use_noise=use_noise,
        )

    try:
        if use_ref:
            compiled = jax.jit(jax.vmap(one))
            out = compiled(stacked_state, stacked_ref)
        else:
            compiled = jax.jit(jax.vmap(lambda state: one(state, {})))
            out = compiled(stacked_state)
        _BATCH_TRACE_COUNT += 1
    except MissingSuppliedDerivativeError:
        raise
    except Exception as exc:
        return _fail(exc)

    flags = out["recipe_ok"]
    if flags:
        ok = jnp.stack([jnp.asarray(v) for v in flags.values()])
        if not bool(jnp.all(ok)):
            return _fail()

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
        settings = implied_scoring_settings_payload(
            scoring_opts, moju_version=_moju_version
        )
        entry["implied_scoring_settings"] = settings
        ms = dict(entry.get("monitor_settings") or {})
        ms["implied_scoring"] = settings
        if apply_optin and "implied_optin" in out:
            r_max = dict(entry.get("r_max") or {})
            implied_scoring: Dict[str, Any] = {}
            label = "max" if metric == "worst_point" else f"p{pct:g}"
            if use_noise:
                label = f"noise_norm+{label}"
            for k, nums in out["implied_optin"].items():
                v_eff = float(jnp.asarray(nums["v_eff"][i]).item())
                r_raw = float(jnp.asarray(nums["r_max_raw"][i]).item())
                r_max[k] = r_raw
                implied_scoring[k] = {
                    "implied_metric": metric,
                    "implied_percentile": pct if metric == "percentile" else None,
                    "implied_noise_normalization": use_noise,
                    "admissibility_metric": label,
                    "r_max_raw": r_raw,
                    "v_eff": v_eff,
                }
            entry["r_max"] = r_max
            entry["implied_scoring"] = implied_scoring
        if region_pack is not None and "region_numeric" in out:
            entry["region_results"] = _assemble_region_results(
                out["region_numeric"],
                index=i,
                region_pack=region_pack,
                flat_keys=list(_flat(residuals)),
                scale=entry.get("scale") or {},
                metric=metric,
                percentile=pct,
                score_from_r_max=score_from_r_max,
            )
            ms["regions"] = entry["region_results"].get("provenance")
        # Host-side: materialise not-identifiable flags so audit rollup matches
        # the loop / ResidualEngine path (NaN implied scores must not zero the tier).
        if "implied_id" in out:
            id_map: Dict[str, Any] = {}
            for bn, nums in out["implied_id"].items():
                compact: Dict[str, Any] = {}
                for sk, arr in nums.items():
                    compact[sk] = jnp.asarray(arr)[i]
                id_map[str(bn)] = materialize_identifiability_meta(compact)
            entry["implied_identifiability"] = id_map
        if "coverage_nums" in out:
            from moju.monitor.evaluation_coverage import materialize_coverage_counts

            cov_map: Dict[str, Any] = {}
            for k, nums in out["coverage_nums"].items():
                host_nums = {sk: jnp.asarray(arr)[i] for sk, arr in nums.items()}
                cov_map[str(k)] = materialize_coverage_counts(
                    host_nums, eval_mask_label=eval_mask_label
                )
            entry["evaluation_coverage"] = cov_map
        entry["monitor_settings"] = ms
        report = audit(
            [entry],
            r_ref=r_ref,
            last_residual_dict=residuals,
            nonfinite_policy=nonfinite_policy,
            eval_mask_label=eval_mask_label,
        )
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


def _score_one(
    state,
    ref,
    plan: _Plan,
    *,
    r_eff,
    r_max,
    rms_scalar,
    implied_with_debug,
    ref_delta,
    region_weights=None,
    region_min_points: int = 16,
    region_metric: str = "worst_point",
    region_percentile: float = 95.0,
    term_threshold: float = 0.1,
    fit_threshold: float = 0.1,
    min_frac: float = 0.05,
    min_count: int = 8,
    apply_optin: bool = False,
    use_noise: bool = False,
) -> Dict[str, Any]:
    from moju.monitor.implied_scoring import (
        noise_normalized_delta,
        reduce_implied_magnitude_arrays,
    )
    from moju.monitor.regions import (
        regional_viscosity_fit_arrays,
        weighted_reduce_magnitude_arrays,
    )

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
    # Numeric sidecars for regional viscosity + opt-in global scoring (arrays only).
    visc_rhs = None
    visc_basis = None
    visc_tf = None
    visc_fq = None
    visc_scale = None
    implied_debug: Dict[str, Dict[str, Any]] = {}
    # Tracer-safe identifiability scalars (status strings materialised on the host).
    implied_id: Dict[str, Dict[str, Any]] = {}
    # Pointwise NI masks / row-NI flags for coverage (keyed by residual basename).
    batch_id_masks: Dict[str, Any] = {}
    batch_id_row_ni: Dict[str, Any] = {}
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
            arr, dbg = implied_with_debug(
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
            if isinstance(dbg, dict):
                key = f"constitutive/{base}/implied_delta"
                side: Dict[str, Any] = {}
                for sk in ("pred", "implied", "raw", "noise_rhs", "noise_basis", "noise_scale"):
                    if dbg.get(sk) is not None:
                        side[sk] = jnp.asarray(dbg[sk])
                if side:
                    implied_debug[key] = side
                if "identifiable" in dbg or dbg.get("status") == "not_identifiable":
                    pack: Dict[str, Any] = {}
                    if "identifiable" in dbg:
                        pack["identifiable"] = jnp.asarray(dbg["identifiable"])
                    elif dbg.get("status") == "not_identifiable":
                        pack["identifiable"] = jnp.asarray(False)
                    for sk in (
                        "excluded_fraction",
                        "n_identifiable",
                        "n_total",
                        "conditioning_threshold",
                        "fit_quality_threshold",
                    ):
                        if dbg.get(sk) is not None:
                            pack[sk] = jnp.asarray(dbg[sk])
                    if pack:
                        implied_id[str(base)] = pack
                    if dbg.get("identifiable_mask") is not None:
                        batch_id_masks[str(base)] = jnp.asarray(dbg["identifiable_mask"])
                    # Row-level wipe (tracer-safe 0-d bool): entire δ is NaN for scoring.
                    if "identifiable" in dbg:
                        batch_id_row_ni[str(base)] = jnp.logical_not(
                            jnp.asarray(dbg["identifiable"])
                        )
                    elif dbg.get("status") == "not_identifiable":
                        batch_id_row_ni[str(base)] = jnp.asarray(True)
                if (
                    dbg.get("noise_rhs") is not None
                    and dbg.get("noise_basis") is not None
                    and dbg.get("term_fraction") is not None
                    and dbg.get("fit_quality") is not None
                    and visc_rhs is None
                ):
                    visc_rhs = jnp.asarray(dbg["noise_rhs"])
                    visc_basis = jnp.asarray(dbg["noise_basis"])
                    visc_tf = jnp.asarray(dbg["term_fraction"])
                    visc_fq = jnp.asarray(dbg["fit_quality"])
                    if dbg.get("noise_scale") is not None:
                        visc_scale = jnp.asarray(dbg["noise_scale"])
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
    from moju.monitor.evaluation_coverage import coverage_counts_arrays

    coverage_nums: Dict[str, Dict[str, Any]] = {}
    for k, v in flat.items():
        id_mask = None
        row_ni: Any = False
        if str(k).startswith("constitutive/") and str(k).endswith("/implied_delta"):
            bn = str(k)[len("constitutive/") : -len("/implied_delta")]
            id_mask = batch_id_masks.get(bn)
            if bn in batch_id_row_ni:
                row_ni = batch_id_row_ni[bn]
        coverage_nums[k] = coverage_counts_arrays(
            v, identifiable_mask=id_mask, row_not_identifiable=row_ni
        )
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

    out: Dict[str, Any] = {
        "residuals": residuals,
        "rms": rms,
        "r_max": rmax,
        "scale": dynamic_scale,
        "recipe_ok": recipe_ok,
    }
    if coverage_nums:
        out["coverage_nums"] = coverage_nums
    if implied_id:
        out["implied_id"] = implied_id

    if apply_optin:
        # Global opt-in: same arithmetic as score_implied_delta_array, jit-safe.
        optin: Dict[str, Dict[str, Any]] = {}
        for k, arr in flat.items():
            if not (str(k).startswith("constitutive/") and str(k).endswith("/implied_delta")):
                continue
            dbg = implied_debug.get(k) or {}
            r_max_raw = reduce_implied_magnitude_arrays(arr, metric="worst_point")
            scored = arr
            if use_noise and dbg.get("pred") is not None and dbg.get("raw") is not None:
                scored = noise_normalized_delta(
                    pred=dbg["pred"],
                    implied=dbg.get("implied"),
                    raw=dbg["raw"],
                    debug=dbg,
                )
            v_eff = reduce_implied_magnitude_arrays(
                scored, metric=region_metric, percentile=region_percentile
            )
            optin[k] = {"v_eff": v_eff, "r_max_raw": r_max_raw}
        if optin:
            out["implied_optin"] = optin

    if region_weights is not None:
        # region_weights: (R, P). Static Python loop over R unrolls under jit.
        n_regions = int(region_weights.shape[0])
        # Per-key stacks: (R,)
        v_eff_by_key: Dict[str, Any] = {}
        worst_by_key: Dict[str, Any] = {}
        for k, arr in flat.items():
            veffs = []
            worsts = []
            for r in range(n_regions):
                v, wi = weighted_reduce_magnitude_arrays(
                    arr,
                    region_weights[r],
                    metric=region_metric,
                    percentile=region_percentile,
                )
                veffs.append(v)
                worsts.append(wi)
            v_eff_by_key[k] = jnp.stack(veffs)
            worst_by_key[k] = jnp.stack(worsts)

        # Viscosity numerics per region (or NaNs if no sidecar).
        nu_list, mu_list, fq_list, n_ok_list, ident_list, insuf_list, n_pts_list = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for r in range(n_regions):
            if visc_rhs is None:
                nan = jnp.asarray(jnp.nan)
                nu_list.append(nan)
                mu_list.append(nan)
                fq_list.append(nan)
                n_ok_list.append(jnp.asarray(0, dtype=jnp.int32))
                ident_list.append(jnp.asarray(False))
                insuf_list.append(jnp.asarray(1, dtype=jnp.int32))
                n_pts_list.append(jnp.sum((region_weights[r] > 0.5).astype(jnp.int32)))
            else:
                fit = regional_viscosity_fit_arrays(
                    rhs=visc_rhs,
                    basis=visc_basis,
                    region_weights=region_weights[r],
                    term_fraction=visc_tf,
                    fit_quality=visc_fq,
                    term_threshold=term_threshold,
                    fit_threshold=fit_threshold,
                    min_frac=min_frac,
                    min_count=min_count,
                    region_min_points=region_min_points,
                    scale=visc_scale,
                )
                nu_list.append(fit["nu_tilde"])
                mu_list.append(fit["mu_tilde"])
                fq_list.append(fit["fit_quality_median"])
                n_ok_list.append(fit["n_identifiable"])
                ident_list.append(fit["identifiable"])
                insuf_list.append(fit["insufficient"])
                n_pts_list.append(fit["n_points"])
        out["region_numeric"] = {
            "v_eff": v_eff_by_key,
            "worst_idx": worst_by_key,
            "nu_tilde": jnp.stack(nu_list),
            "mu_tilde": jnp.stack(mu_list),
            "fit_quality_median": jnp.stack(fq_list),
            "n_identifiable": jnp.stack(n_ok_list),
            "identifiable": jnp.stack(ident_list),
            "insufficient": jnp.stack(insuf_list),
            "n_points": jnp.stack(n_pts_list),
        }
    return out


def _assemble_region_results(
    region_numeric: Mapping[str, Any],
    *,
    index: int,
    region_pack: Mapping[str, Any],
    flat_keys: Sequence[str],
    scale: Mapping[str, float],
    metric: str,
    percentile: float,
    score_from_r_max,
) -> Dict[str, Any]:
    """Host-side: turn batched numeric region outputs into the report schema."""
    from moju.monitor.auditor import admissibility_level
    from moju.monitor.regions import regions_provenance, status_from_region_numerics

    names = list(region_pack["names"])
    normalized = region_pack["normalized"]
    rmin = int(region_pack["region_min_points"])
    prov = regions_provenance(
        normalized,
        region_min_points=rmin,
        moju_version=str(region_pack["moju_version"]),
    )
    out_regions: Dict[str, Any] = {}
    for r, name in enumerate(names):
        n_pts = int(jnp.asarray(region_numeric["n_points"][index, r]).item())
        ident = bool(jnp.asarray(region_numeric["identifiable"][index, r]).item())
        n_ok = int(jnp.asarray(region_numeric["n_identifiable"][index, r]).item())
        insuf = bool(jnp.asarray(region_numeric["insufficient"][index, r]).item())
        flags = status_from_region_numerics(
            n_points=n_pts,
            region_min_points=rmin,
            identifiable=ident and not insuf,
            n_identifiable=n_ok,
            metric=metric,
            percentile=percentile,
        )
        nu = float(jnp.asarray(region_numeric["nu_tilde"][index, r]).item())
        mu = float(jnp.asarray(region_numeric["mu_tilde"][index, r]).item())
        fq = float(jnp.asarray(region_numeric["fit_quality_median"][index, r]).item())
        vis_status = flags.get("viscosity_status", "ok")
        viscosity_meta: Optional[Dict[str, Any]] = {
            "status": vis_status,
            "n_points": n_pts,
            "n_identifiable": n_ok,
            "nu_tilde": nu if vis_status == "ok" else float("nan"),
            "mu_tilde": mu if vis_status == "ok" else float("nan"),
            "fit_quality_median": fq if vis_status != "insufficient_points" else float("nan"),
            "reason": flags.get("viscosity_reason")
            or (flags.get("reason") if vis_status == "insufficient_points" else None),
        }
        per_key: Dict[str, Any] = {}
        for k in flat_keys:
            if k not in region_numeric["v_eff"]:
                continue
            v_eff = float(jnp.asarray(region_numeric["v_eff"][k][index, r]).item())
            wi = int(jnp.asarray(region_numeric["worst_idx"][k][index, r]).item())
            key_row: Dict[str, Any] = {"n_points": n_pts}
            if flags["status"] == "insufficient_points" or insuf:
                key_row.update(
                    {
                        "status": "insufficient_points",
                        "admissibility_score": None,
                        "admissibility_level": "Insufficient points",
                        "score_for_admissibility": None,
                        "worst_point_index": None,
                        "reason": flags.get("reason"),
                    }
                )
                per_key[k] = key_row
                continue
            is_implied = str(k).startswith("constitutive/") and str(k).endswith(
                "/implied_delta"
            )
            if is_implied and vis_status in ("insufficient_points", "not_identifiable"):
                key_row.update(
                    {
                        "status": vis_status,
                        "admissibility_score": None,
                        "admissibility_level": (
                            "Insufficient points"
                            if vis_status == "insufficient_points"
                            else "Not identifiable"
                        ),
                        "nu_tilde": viscosity_meta.get("nu_tilde"),
                        "mu_tilde": viscosity_meta.get("mu_tilde"),
                        "fit_quality": viscosity_meta.get("fit_quality_median"),
                        "n_identifiable": n_ok,
                        "reason": viscosity_meta.get("reason"),
                        "worst_point_index": None,
                    }
                )
                per_key[k] = key_row
                continue
            if is_implied:
                key_row["nu_tilde"] = viscosity_meta.get("nu_tilde")
                key_row["mu_tilde"] = viscosity_meta.get("mu_tilde")
                key_row["fit_quality"] = viscosity_meta.get("fit_quality_median")
                key_row["n_identifiable"] = n_ok
            scale_k = float(scale.get(k, 1.0e-2) or 1.0e-2)
            if not (scale_k > 0.0):
                scale_k = 1.0e-2
            score = (
                float("nan")
                if not (v_eff == v_eff)  # NaN check
                else float(score_from_r_max(v_eff, scale_k=scale_k))
            )
            key_row.update(
                {
                    "status": "ok",
                    "score_for_admissibility": v_eff if v_eff == v_eff else float("nan"),
                    "admissibility_score": score,
                    "admissibility_level": admissibility_level(score),
                    "admissibility_metric": (
                        f"p{percentile:g}" if metric == "percentile" else "max"
                    ),
                    "worst_point_index": None if wi < 0 else wi,
                    "scale_k": scale_k,
                }
            )
            per_key[k] = key_row
        block: Dict[str, Any] = {
            "n_points": n_pts,
            "status": flags["status"],
            "reason": flags.get("reason"),
            "mask_sha256": normalized["mask_sha256"][name],
            "per_key": per_key,
            "viscosity": viscosity_meta,
        }
        if metric == "percentile":
            for key in ("percentile_status", "percentile_tail_count", "percentile"):
                if key in flags:
                    block[key] = flags[key]
        out_regions[name] = block
    return {"provenance": prov, "regions": out_regions}


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
    # Nondimensional constitutive implied_delta / ref_delta use the fixed gauge
    # (same as ResidualEngine._state_derived_scale_per_key).
    from moju.monitor.auditor import _suffix_is_nd_closure

    for key in flat_keys:
        if key in scale:
            continue
        if "/" in key:
            prefix, rest = key.split("/", 1)
            if prefix == "constitutive" and _suffix_is_nd_closure(rest):
                scale[key] = float(plan.floor)
                source[key] = "fixed"
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
