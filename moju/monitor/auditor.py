"""
ResidualEngine: residuals for governing laws, constitutive closures, and scaling closures.

- compute_residuals(state_pred, state_ref=None, *, log_to_python=True)
- build_loss: cascaded RMS over laws only (training).
- audit / visualize: same metrics (RMS, R_norm, admissibility) for all residual keys.

Constitutive and scaling residuals are closure-based (no key_ref). See constitutive_closures
and scaling_closures modules. Metrics are consistency indicators, not certification.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp

from moju.piratio.groups import Groups
from moju.piratio.laws import Laws
from moju.monitor.constitutive_closures import run_constitutive_closures
from moju.monitor.scaling_closures import run_scaling_closures


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
    last_report_per_key = {}
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
        keys_with_score = [k for k in admissibility if admissibility[k] > 0]
        if keys_with_score:
            n = len(keys_with_score)
            w_dict = weights or {}
            w_list = [w_dict.get(k, 1.0 / n) for k in keys_with_score]
            total_w = sum(w_list)
            if total_w > 0:
                w_list = [w / total_w for w in w_list]
            geom_mean = 1.0
            for i, k in enumerate(keys_with_score):
                geom_mean *= admissibility[k] ** w_list[i]
            entry["overall_admissibility_score"] = geom_mean
        else:
            entry["overall_admissibility_score"] = 0.0
    overall = log[-1].get("overall_admissibility_score", 0.0) if log else 0.0
    report = {
        "per_key": last_report_per_key,
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
    Governing laws (Laws.*), optional group specs to enrich state, constitutive_audit and
    scaling_audit for closure residuals. No key_ref.

    constitutive_audit: list of Models registry names, e.g. ["sutherland_mu", "thermal_diffusivity"].
    scaling_audit: list of closure ids, e.g. ["pe_identity", "fo_definition"].
    constitutive_custom / scaling_custom: [{"name": str, "fn": (state, constants) -> array|None}].
    """

    def __init__(
        self,
        constants: Optional[Dict[str, Any]] = None,
        laws: Optional[List[Dict[str, Any]]] = None,
        groups: Optional[List[Dict[str, Any]]] = None,
        *,
        constitutive_audit: Optional[List[str]] = None,
        scaling_audit: Optional[List[str]] = None,
        constitutive_custom: Optional[List[Dict[str, Any]]] = None,
        scaling_custom: Optional[List[Dict[str, Any]]] = None,
    ):
        self.constants = dict(constants or {})
        self.laws_spec = list(laws or [])
        self.groups_spec = list(groups or [])
        self.constitutive_audit = list(constitutive_audit or [])
        self.scaling_audit = list(scaling_audit or [])
        self.constitutive_custom = list(constitutive_custom or [])
        self.scaling_custom = list(scaling_custom or [])
        self._log: List[Dict[str, Any]] = []
        self._index = 0

    @property
    def log(self) -> List[Dict[str, Any]]:
        return self._log

    def _state_builder(self, state_pred: Dict[str, Any]) -> Dict[str, Any]:
        return _build_state(state_pred, self.constants, self.groups_spec)

    def compute_residuals(
        self,
        state_pred: Dict[str, Any],
        state_ref: Optional[Dict[str, Any]] = None,
        *,
        log_to_python: bool = True,
    ) -> Dict[str, Any]:
        residuals: Dict[str, Any] = {"laws": {}}
        state_pred_built = self._state_builder(state_pred)
        merged = {**state_pred_built, **self.constants}

        for spec in self.laws_spec:
            name = spec["name"]
            state_map = spec["state_map"]
            kwargs = _kwargs_from_state(merged, self.constants, state_map)
            fn = _get_fn(spec, Laws)
            residuals["laws"][name] = fn(**kwargs)

        if self.constitutive_audit or self.constitutive_custom:
            c = run_constitutive_closures(
                self.constitutive_audit,
                merged,
                self.constants,
                self.constitutive_custom or None,
            )
            if c:
                residuals["constitutive"] = c

        if self.scaling_audit or self.scaling_custom:
            s = run_scaling_closures(
                self.scaling_audit,
                merged,
                self.constants,
                self.scaling_custom or None,
            )
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
        self._log.append({"index": self._index, "rms": rms_per_key})
        self._index += 1
        return residuals


def list_constitutive_models():
    from moju.monitor.constitutive_closures import list_constitutive_models as _lcm
    return _lcm()


def list_scaling_closure_ids():
    from moju.monitor.scaling_closures import list_scaling_closures
    return list_scaling_closures()
