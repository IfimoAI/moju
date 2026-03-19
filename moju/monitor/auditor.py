"""
ResidualEngine: residuals for governing laws, constitutive, and scaling/similarity audits.

- compute_residuals(state_pred, state_ref=None, *, log_to_python=True)
- build_loss: cascaded RMS over laws only (training).
- audit / visualize: same metrics (RMS, R_norm, admissibility) for all residual keys.

Constitutive and scaling/similarity audits are tied to Models.* and Groups.* functions via
standard closure types (ref_delta, chain_dx, chain_dt). Metrics are consistency indicators,
not certification.
"""

from __future__ import annotations

import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Union

import jax
import jax.numpy as jnp

from moju.piratio.groups import Groups
from moju.piratio.laws import Laws
from moju.monitor.closure_registry import (
    GROUP_FNS,
    MODEL_FNS,
    compute_chain,
    compute_chain_weak,
    compute_ref_delta,
)


def _kwargs_from_state(
    state: Dict[str, Any], constants: Dict[str, Any], state_map: Dict[str, str]
) -> Dict[str, Any]:
    """Build kwargs for a law/group from state_map (arg_name -> state_key)."""
    out = {}
    for arg_name, key in state_map.items():
        val = state.get(key)
        if val is None:
            val = constants.get(key)
        if val is None:
            raise KeyError(f"Key {key!r} not found in state or constants (arg {arg_name})")
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
    merged = {**state, **constants}
    for spec in groups_spec:
        state_map = spec["state_map"]
        output_key = spec["output_key"]
        kwargs = _kwargs_from_state(merged, constants, state_map)
        fn = _get_fn(spec, Groups)
        state[output_key] = fn(**kwargs)
        merged[output_key] = state[output_key]
    return state


def _rms_scalar(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.mean(x**2))


def admissibility_level(score: float) -> str:
    if score >= 0.90:
        return "High Admissibility"
    if score >= 0.70:
        return "Moderate Admissibility"
    if score >= 0.40:
        return "Low Admissibility"
    return "Non-Admissible"


def _rms_per_key(
    residuals_flat: Dict[str, jnp.ndarray],
    *,
    to_python: bool = True,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, arr in residuals_flat.items():
        r = _rms_scalar(arr)
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


def build_loss(
    residual_dict: Dict[str, Any],
    option: str = "cascaded",
    law_weights: Optional[Dict[str, float]] = None,
) -> jnp.ndarray:
    if option != "cascaded":
        raise ValueError(f"Only option='cascaded' is implemented, got {option!r}")
    laws = residual_dict.get("laws", {})
    if not laws:
        return jnp.array(0.0)
    names = list(laws.keys())
    n = len(names)
    weights = law_weights or {}
    w = jnp.array([weights.get(name, 1.0 / n) for name in names])
    rms_vals = jnp.array([_rms_scalar(jnp.asarray(laws[name])) for name in names])
    return jnp.sum(w * rms_vals)


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
    if not log:
        return {"per_key": {}, "overall_admissibility_score": 0.0, "overall_admissibility_level": "Non-Admissible"}
    ref = r_ref if r_ref is not None else log[0].get("rms", {})
    if not ref:
        ref = log[0].get("rms", {})
    last_report_per_key: Dict[str, Any] = {}
    for entry in log:
        rms = entry.get("rms", {})
        r_norm = {}
        admissibility = {}
        for k, v in rms.items():
            r_ref_k = ref.get(k)
            if r_ref_k is not None and r_ref_k > 0:
                r_norm[k] = v / r_ref_k
                admissibility[k] = 1.0 / (1.0 + r_norm[k])
            else:
                r_norm[k] = float("inf") if v != 0 else 0.0
                admissibility[k] = 0.0
            last_report_per_key[k] = {
                "rms": v,
                "r_norm": r_norm[k],
                "admissibility_score": admissibility[k],
                "admissibility_level": admissibility_level(admissibility[k]),
            }
        entry["r_norm"] = r_norm
        entry["admissibility_score"] = admissibility
        # Category scores: geometric mean of per-key scores within each category.
        category_keys: Dict[str, List[str]] = {"laws": [], "constitutive": [], "scaling": []}
        for k in admissibility.keys():
            if "/" in k:
                prefix = k.split("/", 1)[0]
                if prefix in category_keys:
                    category_keys[prefix].append(k)
        category_scores: Dict[str, float] = {}
        for cat, keys in category_keys.items():
            if not keys:
                continue
            # Include zero scores; omit only when key missing.
            gm = 1.0
            for k in keys:
                gm *= float(admissibility.get(k, 0.0)) ** (1.0 / len(keys))
            category_scores[cat] = gm

        entry["category_admissibility_score"] = category_scores

        # Overall: geometric mean across present categories (edge cases: 0 -> NaN, 1 -> that score)
        cats_present = list(category_scores.keys())
        if not cats_present:
            entry["overall_admissibility_score"] = float("nan")
        elif len(cats_present) == 1:
            entry["overall_admissibility_score"] = category_scores[cats_present[0]]
        else:
            gm = 1.0
            for cat in cats_present:
                gm *= float(category_scores[cat]) ** (1.0 / len(cats_present))
            entry["overall_admissibility_score"] = gm
    overall = log[-1].get("overall_admissibility_score", 0.0) if log else 0.0
    report = {
        "per_key": last_report_per_key,
        "per_category": log[-1].get("category_admissibility_score", {}) if log else {},
        "overall_admissibility_score": overall,
        "overall_admissibility_level": admissibility_level(overall),
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


def visualize(
    log: List[Dict[str, Any]],
    keys: Optional[List[str]] = None,
    backend: str = "matplotlib",
) -> Any:
    if backend == "none":
        return None
    if backend != "matplotlib":
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    if not log:
        return None
    first_rms = log[0].get("rms", {})
    plot_keys = keys or list(first_rms.keys())
    if not plot_keys:
        return None
    indices = [i for i in range(len(log))]
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax_rms, ax_adm = axes[0], axes[1]
    for k in plot_keys:
        rms_vals = [e.get("rms", {}).get(k) for e in log]
        rms_vals = [v for v in rms_vals if v is not None]
        if len(rms_vals) == len(log):
            ax_rms.plot(indices, rms_vals, label=k)
        adm_vals = [e.get("admissibility_score", {}).get(k) for e in log]
        adm_vals = [v for v in adm_vals if v is not None]
        if len(adm_vals) == len(log):
            ax_adm.plot(indices, adm_vals, label=k)
    ax_rms.set_ylabel("RMS")
    ax_rms.legend(loc="best", fontsize=8)
    ax_rms.set_title("RMS per key")
    ax_adm.set_ylabel("Admissibility score")
    ax_adm.set_xlabel("Step / index")
    ax_adm.legend(loc="best", fontsize=8)
    ax_adm.set_title("Admissibility score per key")
    plt.tight_layout()
    return fig


class ResidualEngine:
    """
    Governing laws (Laws.*), optional group specs to enrich state, and model/group closures.

    Entry points:
      - Path A (recommended): provide (model, params, collocation) and a state_builder
        so moju can build state_pred (and derivatives) internally.
      - Path B (advanced): provide state_pred directly.

    Closure policy:
      - If a Model/Group spec has no spatial and no temporal variation, it is omitted from the report.
      - Spatial-only -> chain_dx only; temporal-only -> chain_dt only; both -> both.
      - ref_delta is included only when state_ref is provided and variation exists.

    Audit spec shape (constitutive_audit / scaling_audit items):
      {
        "name": "sutherland_mu",               # Models.<name> or Groups.<name>
        "output_key": "mu",                    # state key for F output; expects d_mu_dx / d_mu_dt for chain
        "state_map": {"T": "T", "mu0": "mu0", "T0": "T0", "S": "S"},  # fn arg -> state key
        "predicted_spatial": ["T"],            # state keys varying in x
        "predicted_temporal": ["T"],           # state keys varying in t
      }

    Derivative convention in state_pred: d_<state_key>_dx and d_<state_key>_dt.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        constants: Optional[Dict[str, Any]] = None,
        laws: Optional[List[Dict[str, Any]]] = None,
        groups: Optional[List[Dict[str, Any]]] = None,
        *,
        constitutive_audit: Optional[List[Dict[str, Any]]] = None,
        scaling_audit: Optional[List[Dict[str, Any]]] = None,
        constitutive_custom: Optional[List[Dict[str, Any]]] = None,
        scaling_custom: Optional[List[Dict[str, Any]]] = None,
        state_builder: Optional[
            Callable[[Any, Any, Dict[str, Any], Dict[str, Any]], Dict[str, Any]]
        ] = None,
        enable_omit_messages: bool = True,
        primary_fields: Optional[List[str]] = None,
    ):
        # MonitorConfig convenience
        if config is not None:
            from moju.monitor.config import MonitorConfig

            if isinstance(config, MonitorConfig):
                constants = config.constants
                laws = config.laws
                groups = config.groups
                constitutive_audit = [s.to_dict() for s in config.constitutive_audit]
                scaling_audit = [s.to_dict() for s in config.scaling_audit]
                constitutive_custom = config.constitutive_custom
                scaling_custom = config.scaling_custom
                primary_fields = list(config.primary_fields)
                if config.state_builder is not None and state_builder is None:
                    state_builder = config.state_builder
            else:
                raise TypeError("config must be a MonitorConfig")

        self.constants = dict(constants or {})
        self.laws_spec = list(laws or [])
        self.groups_spec = list(groups or [])
        self.constitutive_audit = list(constitutive_audit or [])
        self.scaling_audit = list(scaling_audit or [])
        self.constitutive_custom = list(constitutive_custom or [])
        self.scaling_custom = list(scaling_custom or [])
        self.state_builder = state_builder
        self.enable_omit_messages = bool(enable_omit_messages)
        self.primary_fields = list(primary_fields or ["T", "u", "v", "w", "p", "rho"])

        # Config-time validation (low effort)
        def _validate_specs(specs: Sequence[Dict[str, Any]], registry: Dict[str, Any], category: str) -> None:
            for spec in specs:
                if "name" not in spec:
                    raise ValueError(f"{category} spec missing 'name'")
                name = spec["name"]
                reg = registry.get(name)
                if reg is None:
                    raise ValueError(f"{category} spec name {name!r} is not registered")
                if "output_key" not in spec:
                    raise ValueError(f"{category}:{name} missing 'output_key'")
                if "state_map" not in spec or not isinstance(spec["state_map"], dict):
                    raise ValueError(f"{category}:{name} missing 'state_map' dict")
                _, arg_names = reg
                missing_args = [an for an in arg_names if an not in spec["state_map"]]
                if missing_args:
                    raise ValueError(f"{category}:{name} state_map missing args: {missing_args}")
                for k in spec.get("predicted_spatial", []) or []:
                    if k not in spec["state_map"].values():
                        raise ValueError(f"{category}:{name} predicted_spatial key {k!r} not in state_map values")
                for k in spec.get("predicted_temporal", []) or []:
                    if k not in spec["state_map"].values():
                        raise ValueError(f"{category}:{name} predicted_temporal key {k!r} not in state_map values")

        _validate_specs(self.constitutive_audit, MODEL_FNS, "constitutive")
        _validate_specs(self.scaling_audit, GROUP_FNS, "scaling")

        self._log: List[Dict[str, Any]] = []
        self._index = 0

    @property
    def log(self) -> List[Dict[str, Any]]:
        return self._log

    def _state_builder(self, state_pred: Dict[str, Any]) -> Dict[str, Any]:
        return _build_state(state_pred, self.constants, self.groups_spec)

    def compute_residuals(
        self,
        state_pred: Optional[Dict[str, Any]] = None,
        state_ref: Optional[Dict[str, Any]] = None,
        *,
        model: Any = None,
        params: Any = None,
        collocation: Optional[Dict[str, Any]] = None,
        log_to_python: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute residuals.

        Path A: pass (model, params, collocation) and configure engine.state_builder.
        Path B: pass state_pred directly.
        """
        residuals: Dict[str, Any] = {"laws": {}}

        if state_pred is None:
            if self.state_builder is None:
                raise ValueError("Path A requires ResidualEngine(state_builder=...)")
            if model is None or params is None or collocation is None:
                raise ValueError("Path A requires model, params, and collocation")
            state_pred = self.state_builder(model, params, collocation, self.constants)

        state_pred_built = self._state_builder(state_pred)
        merged = {**state_pred_built, **self.constants}

        for spec in self.laws_spec:
            name = spec["name"]
            state_map = spec["state_map"]
            kwargs = _kwargs_from_state(merged, self.constants, state_map)
            fn = _get_fn(spec, Laws)
            residuals["laws"][name] = fn(**kwargs)

        omitted_msgs: List[str] = []
        inferred_msgs: List[str] = []

        def _maybe_log_omit(msg: str) -> None:
            if self.enable_omit_messages:
                omitted_msgs.append(msg)

        def _maybe_log_infer(msg: str) -> None:
            if self.enable_omit_messages:
                inferred_msgs.append(msg)

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
                closure_mode = str(spec.get("closure_mode") or "pointwise")
                quadrature_weights = dict(spec.get("quadrature_weights") or {})
                if closure_mode not in ("pointwise", "weak"):
                    raise ValueError(f"{category}:{name} closure_mode must be 'pointwise' or 'weak', got {closure_mode!r}")
                # Sensible defaults (medium effort): when not provided, infer from collocation and common keys
                if "predicted_spatial" in spec:
                    predicted_spatial = list(spec.get("predicted_spatial") or [])
                else:
                    predicted_spatial = []
                    if collocation is not None and "x" in collocation:
                        for k in self.primary_fields:
                            if k in state_map.values():
                                predicted_spatial = [k]
                                break
                    _maybe_log_infer(f"{category}:{name} inferred predicted_spatial={predicted_spatial}")

                if "predicted_temporal" in spec:
                    predicted_temporal = list(spec.get("predicted_temporal") or [])
                else:
                    predicted_temporal = []
                    if collocation is not None and "t" in collocation:
                        for k in self.primary_fields:
                            if k in state_map.values():
                                predicted_temporal = [k]
                                break
                    _maybe_log_infer(f"{category}:{name} inferred predicted_temporal={predicted_temporal}")

                if not predicted_spatial and not predicted_temporal:
                    _maybe_log_omit(
                        f"{category}:{name} omitted: no predicted_spatial or predicted_temporal"
                    )
                    continue

                reg = registry.get(name)
                if reg is None:
                    # unknown function name -> omit silently (config validation should catch)
                    continue
                fn, arg_names = reg

                if state_ref is not None and output_key is not None:
                    arr = compute_ref_delta(
                        fn=fn,
                        arg_names=arg_names,
                        output_key=output_key,
                        state_map=state_map,
                        state_pred=merged,
                        state_ref={**self._state_builder(state_ref), **self.constants},
                        constants=self.constants,
                    )
                    if arr is not None:
                        out[f"{name}/ref_delta"] = jnp.asarray(arr)

                if predicted_spatial and output_key is not None:
                    if closure_mode == "weak":
                        arr = compute_chain_weak(
                            fn=fn,
                            arg_names=arg_names,
                            output_key=output_key,
                            state_map=state_map,
                            state_pred=merged,
                            constants=self.constants,
                            predicted_varying=predicted_spatial,
                            deriv="x",
                            weight_key=quadrature_weights.get("x"),
                        )
                        _maybe_log_infer(f"{category}:{name} using weak chain_dx")
                    else:
                        arr = compute_chain(
                            fn=fn,
                            arg_names=arg_names,
                            output_key=output_key,
                            state_map=state_map,
                            state_pred=merged,
                            constants=self.constants,
                            predicted_varying=predicted_spatial,
                            deriv="x",
                        )
                    if arr is not None:
                        out[f"{name}/chain_dx"] = jnp.asarray(arr)

                if predicted_temporal and output_key is not None:
                    if closure_mode == "weak":
                        arr = compute_chain_weak(
                            fn=fn,
                            arg_names=arg_names,
                            output_key=output_key,
                            state_map=state_map,
                            state_pred=merged,
                            constants=self.constants,
                            predicted_varying=predicted_temporal,
                            deriv="t",
                            weight_key=quadrature_weights.get("t"),
                        )
                        _maybe_log_infer(f"{category}:{name} using weak chain_dt")
                    else:
                        arr = compute_chain(
                            fn=fn,
                            arg_names=arg_names,
                            output_key=output_key,
                            state_map=state_map,
                            state_pred=merged,
                            constants=self.constants,
                            predicted_varying=predicted_temporal,
                            deriv="t",
                        )
                    if arr is not None:
                        out[f"{name}/chain_dt"] = jnp.asarray(arr)

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

        if self.scaling_audit or self.scaling_custom:
            s = _run_specs(self.scaling_audit, registry=GROUP_FNS, category="scaling")
            if self.scaling_custom:
                for spec in self.scaling_custom:
                    cname = spec["name"]
                    arr = spec["fn"](merged, self.constants)
                    if arr is not None:
                        s[f"custom/{cname}"] = jnp.asarray(arr)
            if s:
                residuals["scaling"] = s

        if state_ref is not None:
            state_ref_built = self._state_builder(state_ref)
            common = set(state_pred_built.keys()) & set(state_ref_built.keys())
            residuals["data"] = {
                k: jnp.asarray(state_ref_built[k]) - jnp.asarray(state_pred_built[k])
                for k in common
            }

        flat = _flatten_residual_dict(residuals)
        rms_per_key = _rms_per_key(flat, to_python=log_to_python)
        entry: Dict[str, Any] = {"index": self._index, "rms": rms_per_key}
        if omitted_msgs:
            entry["omitted"] = omitted_msgs
        if inferred_msgs:
            entry["inferred"] = inferred_msgs
        self._log.append(entry)
        self._index += 1
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
            for spec in self.constitutive_audit + self.scaling_audit:
                keys |= set((spec.get("state_map") or {}).values())
                ok = spec.get("output_key")
                if ok:
                    keys.add(ok)
        return keys

    def required_derivative_keys(self) -> Set[str]:
        keys: Set[str] = set()
        for spec in self.constitutive_audit + self.scaling_audit:
            output_key = spec.get("output_key")
            if not output_key:
                continue
            pred_x = list(spec.get("predicted_spatial") or [])
            pred_t = list(spec.get("predicted_temporal") or [])
            if pred_x:
                keys.add(f"d_{output_key}_dx")
                for k in pred_x:
                    keys.add(f"d_{k}_dx")
            if pred_t:
                keys.add(f"d_{output_key}_dt")
                for k in pred_t:
                    keys.add(f"d_{k}_dt")
        return keys


def list_constitutive_models():
    from moju.monitor.closure_registry import list_models
    return list_models()


def list_scaling_closure_ids():
    from moju.monitor.closure_registry import list_groups
    return list_groups()
