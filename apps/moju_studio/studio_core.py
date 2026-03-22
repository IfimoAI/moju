"""
Moju helpers for Studio: registries, preflight, residual flattening, code snippets.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import jax.numpy as jnp

from moju.monitor import MonitorConfig, ResidualEngine
from moju.monitor.config import MonitorConfig as MC
from moju.piratio.laws import Laws


def list_registered_law_names() -> List[str]:
    names = []
    for n in dir(Laws):
        if n.startswith("_"):
            continue
        attr = getattr(Laws, n, None)
        if callable(attr):
            names.append(n)
    return sorted(names)


def jnp_constants(cfg: MonitorConfig) -> MonitorConfig:
    """Coerce numeric constants to jnp arrays for JAX-safe merges."""
    out = {}
    for k, v in cfg.constants.items():
        if isinstance(v, (int, float, list)):
            out[k] = jnp.asarray(v)
        else:
            out[k] = v
    return replace(cfg, constants=out)


def monitor_config_from_merged_dict(d: Dict[str, Any], state_builder: Optional[Callable] = None) -> MonitorConfig:
    """Build MonitorConfig from dict (e.g. merged JSON + UI)."""
    cfg = MC.from_dict(d)
    if state_builder is not None:
        cfg = replace(cfg, state_builder=state_builder)
    return jnp_constants(cfg)


STUDIO_NPZ_SHIM_ATTR = "__studio_npz_shim__"
STUDIO_RECOMPUTES_WITH_CONSTANTS_ATTR = "_moju_studio_recomputes_with_constants"


def is_studio_npz_shim_state_builder(builder: Any) -> bool:
    return bool(getattr(builder, STUDIO_NPZ_SHIM_ATTR, False))


def mark_recomputing_state_builder(
    fn: Callable[..., Dict[str, Any]],
) -> Callable[..., Dict[str, Any]]:
    """
    Optional tag for documentation/tests: marks a ``state_builder`` as recomputing from
    ``constants`` (and colloc/model).

    Studio π gating only requires a **non-NPZ-shim** ``state_builder`` (e.g. set
    ``st.session_state['studio_recomputing_state_builder']`` to that callable); this helper
    does not change gating logic by itself.
    """
    setattr(fn, STUDIO_RECOMPUTES_WITH_CONSTANTS_ATTR, True)
    return fn


def make_session_state_builder(pred: Dict[str, Any]) -> Callable[..., Dict[str, Any]]:
    """
    Path A shim: return uploaded arrays; groups + constants merges happen inside the engine.
    Ignores ``constants`` for tensor keys already in ``pred`` — **not valid for π-constant**
    scale-invariance in Studio (see :func:`validate_studio_pi_gating`).
    """

    pred_j = {k: jnp.asarray(v) for k, v in pred.items()}

    def state_builder(model, params, collocation, constants):  # noqa: ARG001
        return {k: jnp.asarray(v) for k, v in pred_j.items()}

    setattr(state_builder, STUDIO_NPZ_SHIM_ATTR, True)
    return state_builder


def validate_studio_pi_gating(
    *,
    use_path_b: bool,
    scaling_audit_specs: List[Dict[str, Any]],
    state_builder: Optional[Callable[..., Dict[str, Any]]],
) -> None:
    """
    Enforce Studio policy: surface π-constant only when predictions can be recomputed under
    scaled constants and compare keys are set.

    Raises ``ValueError`` with a clear message if the configuration is invalid for Studio.
    """
    pi_specs = [s for s in scaling_audit_specs if s.get("invariance_pi_constant")]
    if not pi_specs:
        return

    if use_path_b:
        raise ValueError(
            "π-constant (scale-invariance) scaling audits require **Path A** "
            "(a `state_builder` forward, not uploaded `state_pred` only). "
            "Switch **Execution path** to Path A, or disable π-constant on scaling audits."
        )

    if state_builder is None:
        raise ValueError("Path A requires a state_builder for π-constant audits.")

    if is_studio_npz_shim_state_builder(state_builder):
        raise ValueError(
            "Moju Studio: π-constant residuals are disabled with the **NPZ Path A shim**, "
            "because `state_pred` does not change when constants are scaled — results would "
            "not reflect scale invariance of a recomputing model. "
            "Use a `state_builder` that recomputes from `constants` (Python API, or set "
            "`st.session_state['studio_recomputing_state_builder']` to that callable), "
            "or turn off π-constant on scaling audits."
        )

    for spec in pi_specs:
        name = spec.get("name", "?")
        ck = list(spec.get("invariance_compare_keys") or [])
        if not ck:
            raise ValueError(
                f"scaling audit {name!r}: π-constant requires non-empty `invariance_compare_keys` "
                "(which state keys to compare after scaling constants)."
            )


def flatten_residuals(residuals: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror auditor flat keys for plotting."""
    flat: Dict[str, Any] = {}
    for category, content in residuals.items():
        if isinstance(content, dict):
            for name, arr in content.items():
                flat[f"{category}/{name}"] = arr
        elif hasattr(content, "shape"):
            flat[category] = content
    return flat


def preflight_engine(
    engine: ResidualEngine,
    state_keys: Set[str],
) -> Tuple[List[str], List[str]]:
    """
    Return (missing_state_keys, missing_derivative_keys) relative to uploaded state keys.
    Does not include group-generated keys.
    """
    req_s = engine.required_state_keys()
    req_d = engine.required_derivative_keys()
    miss_s = sorted(k for k in req_s if k not in state_keys)
    miss_d = sorted(k for k in req_d if k not in state_keys)
    return miss_s, miss_d


def audit_report_to_jsonable(report: Dict[str, Any]) -> Dict[str, Any]:
    """Strip non-JSON types from audit() output for download."""

    def _conv(o: Any) -> Any:
        if isinstance(o, dict):
            return {k: _conv(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_conv(x) for x in o]
        if hasattr(o, "item") and hasattr(o, "shape") and o.shape == ():
            return float(o.item())
        if isinstance(o, float):
            return o
        if isinstance(o, int):
            return o
        if o is None:
            return None
        if isinstance(o, str):
            return o
        return str(o)

    return _conv(report)


def generate_python_snippet(cfg: MonitorConfig, *, path_b: bool) -> str:
    """Minimal reproducibility snippet (user fills state_pred)."""
    d = cfg.to_dict()
    lines = [
        "from moju.monitor import ResidualEngine, MonitorConfig, audit, visualize",
        "from moju.monitor.config import AuditSpec",
        "import jax.numpy as jnp",
        "",
        "# Build config (expand AuditSpecs as needed; implied_fn not serializable).",
        f"cfg = MonitorConfig.from_dict({json.dumps(d, indent=2, default=str)})",
        "engine = ResidualEngine(config=cfg)",
        "",
    ]
    if path_b:
        lines += [
            "state_pred = { ... }  # your arrays",
            "state_ref = None  # optional",
            "residuals = engine.compute_residuals(state_pred, state_ref, auto_path_b_derivatives=False)",
            "report = audit(engine.log)",
        ]
    else:
        lines += [
            "# Path A: define state_builder on cfg and call:",
            "engine.compute_residuals(None, model=..., params=..., collocation=...)",
            "report = audit(engine.log)",
        ]
    return "\n".join(lines)
