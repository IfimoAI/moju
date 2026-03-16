"""
ResidualEngine: single place for residuals, physics loss, and monitoring.

You configure laws, groups, and models; then one class computes residuals,
builds a physics-only loss (build_loss), and keeps a log for audit and visualize.

- compute_residuals(state_pred, state_ref=None, key_ref=None) returns a residual dict
  and logs per-key RMS to the internal log.
- build_loss(residual_dict, ...) returns a physics-only scalar; add data loss in JAX/torch.
- audit(log) computes R_norm, admissibility score, and overall admissibility score from the log and writes them back.
- visualize(log, ...) plots RMS and metrics per key.

key_ref is for Groups and Models only (reference values for group/model outputs).
Data residual is computed and logged only when state_ref is provided.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp

from moju.piratio.groups import Groups
from moju.piratio.laws import Laws
from moju.piratio.models import Models


def _kwargs_from_state(
    state: Dict[str, Any], constants: Dict[str, Any], state_map: Dict[str, str]
) -> Dict[str, Any]:
    """Build kwargs for a law/group/model from state_map (arg_name -> state_key)."""
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
    """Resolve callable from spec: use spec['fn'] if present, else getattr(builtin_class, spec['name'])."""
    if "fn" in spec:
        return spec["fn"]
    return getattr(builtin_class, spec["name"])


def _build_state(
    state_pred: Dict[str, Any],
    constants: Dict[str, Any],
    models_spec: List[Dict],
    groups_spec: List[Dict],
) -> Dict[str, Any]:
    """Run Models (in order) then Groups, writing outputs into state."""
    state = dict(state_pred)
    merged = {**state, **constants}

    for spec in models_spec:
        state_map = spec["state_map"]
        output_key = spec["output_key"]
        kwargs = _kwargs_from_state(merged, constants, state_map)
        fn = _get_fn(spec, Models)
        state[output_key] = fn(**kwargs)
        merged[output_key] = state[output_key]

    for spec in groups_spec:
        state_map = spec["state_map"]
        output_key = spec["output_key"]
        kwargs = _kwargs_from_state(merged, constants, state_map)
        fn = _get_fn(spec, Groups)
        state[output_key] = fn(**kwargs)
        merged[output_key] = state[output_key]

    return state


def _rms_scalar(x: jnp.ndarray) -> jnp.ndarray:
    """RMS over all elements and batch: sqrt(mean(x**2))."""
    return jnp.sqrt(jnp.mean(x**2))


# Admissibility score bands: 0.00-<0.40 Non-Admissible, 0.40-<0.70 Low, 0.70-<0.90 Moderate, 0.90-1.00 High
def admissibility_level(score: float) -> str:
    """
    Map an admissibility score to one of four levels.

    Ranges: 0.00-<0.40 Non-Admissible; 0.40-<0.70 Low Admissibility;
    0.70-<0.90 Moderate Admissibility; 0.90-1.00 High Admissibility.
    """
    if score >= 0.90:
        return "High Admissibility"
    if score >= 0.70:
        return "Moderate Admissibility"
    if score >= 0.40:
        return "Low Admissibility"
    return "Non-Admissible"


def _rms_per_key(residuals_flat: Dict[str, jnp.ndarray]) -> Dict[str, float]:
    """Compute one RMS per key across all batches; return host-side dict."""
    out = {}
    for key, arr in residuals_flat.items():
        r = _rms_scalar(arr)
        out[key] = float(jax.device_get(r))
    return out


def _flatten_residual_dict(residuals: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
    """Flatten residual dict to key -> array for RMS logging. Keys like 'laws/mass_incompressible'."""
    flat = {}
    for category, content in residuals.items():
        if isinstance(content, dict):
            for name, arr in content.items():
                if hasattr(arr, "shape"):
                    flat[f"{category}/{name}"] = jnp.asarray(arr)
                else:
                    flat[f"{category}/{name}"] = jnp.asarray(arr)
        else:
            if hasattr(content, "shape"):
                flat[category] = jnp.asarray(content)
            else:
                flat[category] = jnp.asarray(content)
    return flat


def build_loss(
    residual_dict: Dict[str, Any],
    option: str = "cascaded",
    law_weights: Optional[Dict[str, float]] = None,
) -> jnp.ndarray:
    """
    Physics-only loss from the residual dict (no data residual or loss passed in).

    Cascaded option: loss = sum over laws of weight_i * rms(residual_dict["laws"][law_i]).
    Default weights are equal per law. User adds data loss to the output in JAX or torch.

    :param residual_dict: Dict with at least "laws" key mapping law name -> residual array.
    :param option: "cascaded" (implemented); weighted by law_weights.
    :param law_weights: Optional dict law_name -> weight; else equal weights.
    :return: Scalar physics loss (differentiable).
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
    """
    Compute R_norm, admissibility score, and overall admissibility score from the log; write them back into the same log.

    R_norm = RMS(r) / RMS(r_ref). Admissibility score = 1 / (1 + R_norm). Overall = geometric mean of admissibility scores.
    When state_ref was provided, data residual is in the log and included in metrics.

    Optional export: if export_dir is set, creates a session folder (audit_YYYYMMDD_HHMM), writes a PDF report
    and optionally residuals.json, then zips the folder. Requires reportlab: pip install moju[report].

    :param log: List of entries, each with "rms" dict (key -> float). First entry used as r_ref if r_ref is None.
    :param r_ref: Optional reference RMS per key; if None, use first log entry's rms.
    :param weights: Optional weights per key for geometric mean; else equal.
    :param export_dir: If set, write PDF report (and optionally residuals) to a session folder and zip it.
    :param save_residuals: If True and last_residual_dict is provided, save residuals.json in the session folder.
    :param last_residual_dict: Optional residual dict from the last compute_residuals call; required for save_residuals.
    :param model_name: Optional model name for the report header.
    :param model_id: Optional model ID for the report header.
    :return: Report dict {per_key: {rms, r_norm, admissibility_score, admissibility_level}, overall_admissibility_score, overall_admissibility_level}.
    """
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
    """
    Create simple plots of RMS per key and admissibility score per key as training proceeds.

    :param log: List of log entries (each with rms, and optionally r_norm, admissibility_score).
    :param keys: Optional list of keys to plot; if None, plot all keys present in the first entry.
    :param backend: "matplotlib" (default) or "none" to skip plotting.
    :return: Figure or None if backend is "none" or matplotlib not available.
    """
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
    Single place for residuals, physics loss, and monitoring.

    Configure laws, groups, models, and constants; then call compute_residuals(state_pred, ...)
    to get a residual dict and log per-key RMS. Use build_loss(residual_dict) for training;
    use audit(log) and visualize(log) for monitoring.

    state_pred is required. state_ref and key_ref are optional.
    key_ref is for Groups and Models only. Data residual is computed only when state_ref is provided.

    Custom physics: in any law/group/model spec you can pass an optional "fn": callable.
    If present, that JAX-differentiable function is used instead of the built-in (Laws/Groups/Models).name.
    Kwargs are built from state_map; use jax.numpy inside your fn. See docs for examples.
    """

    def __init__(
        self,
        constants: Optional[Dict[str, Any]] = None,
        laws: Optional[List[Dict[str, Any]]] = None,
        groups: Optional[List[Dict[str, Any]]] = None,
        models: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        :param constants: Dict of constant name -> value (e.g. L, mu0, T0).
        :param laws: List of specs, each {"name": str, "state_map": {arg_name: state_key}}.
          Optional "fn": callable — if provided, use this JAX-differentiable function instead of
          the built-in Laws.name; kwargs come from state_map. Use for custom physics laws.
        :param groups: List of specs, each {"name": str, "state_map": {...}, "output_key": str}.
          Optional "fn": callable — if provided, use instead of built-in Groups.name.
        :param models: List of specs, each {"name": str, "state_map": {...}, "output_key": str}.
          Optional "fn": callable — if provided, use instead of built-in Models.name.
        """
        self.constants = dict(constants or {})
        self.laws_spec = list(laws or [])
        self.groups_spec = list(groups or [])
        self.models_spec = list(models or [])
        self._log: List[Dict[str, Any]] = []
        self._index = 0

    @property
    def log(self) -> List[Dict[str, Any]]:
        """The log object (list of entries with rms, and after audit: r_norm, admissibility_score, overall_admissibility_score)."""
        return self._log

    def _state_builder(self, state_pred: Dict[str, Any]) -> Dict[str, Any]:
        return _build_state(
            state_pred,
            self.constants,
            self.models_spec,
            self.groups_spec,
        )

    def compute_residuals(
        self,
        state_pred: Dict[str, Any],
        state_ref: Optional[Dict[str, Any]] = None,
        key_ref: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Compute residual dict and append per-key RMS to the internal log.

        - When state_ref is None and key_ref is not provided: only law residuals.
        - When key_ref is provided (no state_ref): law + group + model residuals (group/model vs key_ref).
        - When state_ref is provided: law + group + model (vs state_ref or key_ref) + data residual.

        :param state_pred: Predicted state (required). Dict of array-like.
        :param state_ref: Optional reference state; when provided, data residual is computed and logged.
        :param key_ref: Optional reference values for group/model outputs (key -> value). For Groups/Models only.
        :return: Residual dict with keys "laws", optionally "groups", "models", "data". All JAX arrays; differentiable.
        """
        residuals: Dict[str, Any] = {"laws": {}}
        state_pred_built = self._state_builder(state_pred)
        merged = {**state_pred_built, **self.constants}

        for spec in self.laws_spec:
            name = spec["name"]
            state_map = spec["state_map"]
            kwargs = _kwargs_from_state(merged, self.constants, state_map)
            fn = _get_fn(spec, Laws)
            residuals["laws"][name] = fn(**kwargs)

        if key_ref is not None:
            residuals["groups"] = {}
            residuals["models"] = {}
            for spec in self.groups_spec:
                output_key = spec["output_key"]
                state_map = spec["state_map"]
                kwargs = _kwargs_from_state(merged, self.constants, state_map)
                fn = _get_fn(spec, Groups)
                out = fn(**kwargs)
                ref_val = key_ref.get(output_key)
                if ref_val is not None:
                    ref_arr = jnp.asarray(ref_val)
                    if hasattr(out, "shape"):
                        ref_arr = jnp.broadcast_to(ref_arr, out.shape)
                    residuals["groups"][output_key] = out - ref_arr
                else:
                    residuals["groups"][output_key] = out
            for spec in self.models_spec:
                output_key = spec["output_key"]
                state_map = spec["state_map"]
                kwargs = _kwargs_from_state(merged, self.constants, state_map)
                fn = _get_fn(spec, Models)
                out = fn(**kwargs)
                ref_val = key_ref.get(output_key)
                if ref_val is not None:
                    ref_arr = jnp.asarray(ref_val)
                    if hasattr(out, "shape"):
                        ref_arr = jnp.broadcast_to(ref_arr, out.shape)
                    residuals["models"][output_key] = out - ref_arr
                else:
                    residuals["models"][output_key] = out

        if state_ref is not None:
            state_ref_built = self._state_builder(state_ref)
            common = set(state_pred_built.keys()) & set(state_ref_built.keys())
            residuals["data"] = {
                k: jnp.asarray(state_ref_built[k]) - jnp.asarray(state_pred_built[k])
                for k in common
            }

        flat = _flatten_residual_dict(residuals)
        rms_per_key = _rms_per_key(flat)
        self._log.append({"index": self._index, "rms": rms_per_key})
        self._index += 1
        return residuals
