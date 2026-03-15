"""
MojuCore: single place for residuals, physics loss, and monitoring.

You configure laws, groups, and models; then one class computes residuals,
builds a physics-only loss (build_loss), and keeps a log for audit and visualize.

- compute_residuals(state_pred, state_ref=None, key_ref=None) returns a residual dict
  and logs per-key RMS to the internal log.
- build_loss(residual_dict, ...) returns a physics-only scalar; add data loss in JAX/torch.
- audit(log) computes R_norm, S, overall score from the log and writes them back.
- visualize(log, ...) plots RMS and metrics per key.

key_ref is for Groups and Models only (reference values for group/model outputs).
Data residual is computed and logged only when state_ref is provided.
"""

from __future__ import annotations

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
        name = spec["name"]
        state_map = spec["state_map"]
        output_key = spec["output_key"]
        kwargs = _kwargs_from_state(merged, constants, state_map)
        fn = getattr(Models, name)
        state[output_key] = fn(**kwargs)
        merged[output_key] = state[output_key]

    for spec in groups_spec:
        name = spec["name"]
        state_map = spec["state_map"]
        output_key = spec["output_key"]
        kwargs = _kwargs_from_state(merged, constants, state_map)
        fn = getattr(Groups, name)
        state[output_key] = fn(**kwargs)
        merged[output_key] = state[output_key]

    return state


def _rms_scalar(x: jnp.ndarray) -> jnp.ndarray:
    """RMS over all elements and batch: sqrt(mean(x**2))."""
    return jnp.sqrt(jnp.mean(x**2))


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
) -> Dict[str, Any]:
    """
    Compute R_norm, S, and overall physics score from the log; write them back into the same log.

    R_norm = RMS(r) / RMS(r_ref), S = 1 / (1 + R_norm). Overall score = geometric mean of S.
    When state_ref was provided, data residual is in the log and included in metrics.

    :param log: List of entries, each with "rms" dict (key -> float). First entry used as r_ref if r_ref is None.
    :param r_ref: Optional reference RMS per key; if None, use first log entry's rms.
    :param weights: Optional weights per key for geometric mean; else equal.
    :return: Report dict {per_key: {rms, r_norm, S}, overall_physics_score}.
    """
    if not log:
        return {"per_key": {}, "overall_physics_score": 0.0}
    ref = r_ref if r_ref is not None else log[0].get("rms", {})
    if not ref:
        ref = log[0].get("rms", {})
    last_report_per_key = {}
    for entry in log:
        rms = entry.get("rms", {})
        r_norm = {}
        S = {}
        for k, v in rms.items():
            r_ref_k = ref.get(k)
            if r_ref_k is not None and r_ref_k > 0:
                r_norm[k] = v / r_ref_k
                S[k] = 1.0 / (1.0 + r_norm[k])
            else:
                r_norm[k] = float("inf") if v != 0 else 0.0
                S[k] = 0.0
            last_report_per_key[k] = {"rms": v, "r_norm": r_norm[k], "S": S[k]}
        entry["r_norm"] = r_norm
        entry["S"] = S
        keys_with_s = [k for k in S if S[k] > 0]
        if keys_with_s:
            n = len(keys_with_s)
            w_dict = weights or {}
            w_list = [w_dict.get(k, 1.0 / n) for k in keys_with_s]
            total_w = sum(w_list)
            if total_w > 0:
                w_list = [w / total_w for w in w_list]
            geom_mean = 1.0
            for i, k in enumerate(keys_with_s):
                geom_mean *= S[k] ** w_list[i]
            entry["overall_physics_score"] = geom_mean
        else:
            entry["overall_physics_score"] = 0.0
    overall = log[-1].get("overall_physics_score", 0.0) if log else 0.0
    return {"per_key": last_report_per_key, "overall_physics_score": overall}


def visualize(
    log: List[Dict[str, Any]],
    keys: Optional[List[str]] = None,
    backend: str = "matplotlib",
) -> Any:
    """
    Create simple plots of RMS per key and metrics per key (R_norm, S) as training proceeds.

    :param log: List of log entries (each with rms, and optionally r_norm, S).
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
    ax_rms, ax_s = axes[0], axes[1]
    for k in plot_keys:
        rms_vals = [e.get("rms", {}).get(k) for e in log]
        rms_vals = [v for v in rms_vals if v is not None]
        if len(rms_vals) == len(log):
            ax_rms.plot(indices, rms_vals, label=k)
        s_vals = [e.get("S", {}).get(k) for e in log]
        s_vals = [v for v in s_vals if v is not None]
        if len(s_vals) == len(log):
            ax_s.plot(indices, s_vals, label=k)
    ax_rms.set_ylabel("RMS")
    ax_rms.legend(loc="best", fontsize=8)
    ax_rms.set_title("RMS per key")
    ax_s.set_ylabel("S")
    ax_s.set_xlabel("Step / index")
    ax_s.legend(loc="best", fontsize=8)
    ax_s.set_title("Score S per key")
    plt.tight_layout()
    return fig


class MojuCore:
    """
    Single place for residuals, physics loss, and monitoring.

    Configure laws, groups, models, and constants; then call compute_residuals(state_pred, ...)
    to get a residual dict and log per-key RMS. Use build_loss(residual_dict) for training;
    use audit(log) and visualize(log) for monitoring.

    state_pred is required. state_ref and key_ref are optional.
    key_ref is for Groups and Models only. Data residual is computed only when state_ref is provided.
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
        :param groups: List of specs, each {"name": str, "state_map": {...}, "output_key": str}.
        :param models: List of specs, each {"name": str, "state_map": {...}, "output_key": str}.
        """
        self.constants = dict(constants or {})
        self.laws_spec = list(laws or [])
        self.groups_spec = list(groups or [])
        self.models_spec = list(models or [])
        self._log: List[Dict[str, Any]] = []
        self._index = 0

    @property
    def log(self) -> List[Dict[str, Any]]:
        """The log object (list of entries with rms, and after audit: r_norm, S, overall_physics_score)."""
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
            fn = getattr(Laws, name)
            residuals["laws"][name] = fn(**kwargs)

        if key_ref is not None:
            residuals["groups"] = {}
            residuals["models"] = {}
            for spec in self.groups_spec:
                name = spec["name"]
                output_key = spec["output_key"]
                state_map = spec["state_map"]
                kwargs = _kwargs_from_state(merged, self.constants, state_map)
                fn = getattr(Groups, name)
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
                name = spec["name"]
                output_key = spec["output_key"]
                state_map = spec["state_map"]
                kwargs = _kwargs_from_state(merged, self.constants, state_map)
                fn = getattr(Models, name)
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
