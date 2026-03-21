"""
ResidualEngine: residuals for governing laws, constitutive, and scaling/similarity audits.

- compute_residuals(state_pred, state_ref=None, *, log_to_python=True)
- build_loss: cascaded RMS over laws only (training).
- audit / visualize: same metrics (RMS, R_norm, admissibility) for all residual keys;
  visualize builds a multi-panel matplotlib dashboard (no extra user data required).

Constitutive and scaling/similarity audits are tied to Models.* and Groups.* functions via
standard closure types (ref_delta, implied_delta, chain_dx/chain_dy/chain_dz, chain_dt). Metrics are consistency
indicators, not certification.
"""

from __future__ import annotations

import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Union

import jax
import jax.numpy as jnp

from moju.piratio.groups import Groups
from moju.piratio.laws import Laws
from moju.monitor.closure_registry import (
    GROUP_FNS,
    MODEL_FNS,
    compute_chain,
    compute_chain_weak,
    compute_implied_delta,
    compute_ref_delta,
)
from moju.monitor.derivative_keys import CHAIN_SPATIAL_DERIVS, collect_audit_derivative_keys
from moju.monitor.pi_constant_recipes import (
    GROUP_PI_CONSTANT_RECIPES,
    apply_pi_constant_recipe,
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


_SCALE_EPS = 1e-12


def _state_derived_scale_per_key(
    flat_keys: Iterable[str],
    merged: Dict[str, Any],
    laws_spec: List[Dict[str, Any]],
    constitutive_audit: List[Dict[str, Any]],
    scaling_audit: List[Dict[str, Any]],
    state_ref_built: Optional[Dict[str, Any]] = None,
    *,
    to_python: bool = True,
) -> Dict[str, float]:
    """
    State-derived scale per residual key for R_norm = RMS(r_k) / scale_k.
    Used when r_ref is not supplied; provides a scale relative to solution size.
    """
    out: Dict[str, float] = {}
    ref = state_ref_built if state_ref_built is not None else merged

    for k in flat_keys:
        if "/" not in k:
            scale = _SCALE_EPS + _rms_scalar(jnp.asarray(1.0))
            out[k] = float(jax.device_get(scale)) if to_python else float(scale)
            continue
        prefix, rest = k.split("/", 1)
        if prefix == "laws":
            name = rest
            spec = next((s for s in laws_spec if s.get("name") == name), None)
            if spec is None:
                scale = _SCALE_EPS + _rms_scalar(jnp.asarray(1.0))
            else:
                state_map = spec.get("state_map") or {}
                parts = []
                for sk in state_map.values():
                    if sk in merged:
                        v = jnp.asarray(merged[sk])
                        parts.append(jnp.ravel(v))
                if parts:
                    big = jnp.concatenate(parts)
                    scale = _SCALE_EPS + _rms_scalar(big)
                else:
                    scale = _SCALE_EPS + _rms_scalar(jnp.asarray(1.0))
            out[k] = float(jax.device_get(scale)) if to_python else float(scale)
            continue
        if prefix == "constitutive":
            name = rest.split("/")[0]
            spec = next((s for s in constitutive_audit if s.get("name") == name), None)
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
            name = rest.split("/")[0]
            spec = next((s for s in scaling_audit if s.get("name") == name), None)
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
                ivk_s = spec.get("implied_value_key")
                if ivk_s and ivk_s in merged:
                    parts.append(jnp.ravel(jnp.asarray(merged[ivk_s])))
                if parts:
                    big = jnp.concatenate(parts)
                    scale = _SCALE_EPS + _rms_scalar(big)
                else:
                    scale = _SCALE_EPS + _rms_scalar(jnp.asarray(1.0))
            out[k] = float(jax.device_get(scale)) if to_python else float(scale)
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
        scale = _SCALE_EPS + _rms_scalar(jnp.asarray(1.0))
        out[k] = float(jax.device_get(scale)) if to_python else float(scale)
    return out


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


def _compute_log_step_metrics(
    log: List[Dict[str, Any]],
    r_ref: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Per-log-entry admissibility metrics (same rules as ``audit``), without mutating ``log``.

    Returns one dict per entry with keys: ``r_norm``, ``admissibility_score``,
    ``category_admissibility_score``, ``overall_admissibility_score``, and
    ``per_key_report`` (flat key -> {rms, r_norm, admissibility_score, admissibility_level}).
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
            if scale_k <= 0:
                scale_k = 1.0
            r_norm[k] = v / scale_k
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
            gm = 1.0
            for kk in keys:
                gm *= float(admissibility.get(kk, 0.0)) ** (1.0 / len(keys))
            category_scores[cat] = gm
        cats_present = list(category_scores.keys())
        if not cats_present:
            overall = float("nan")
        elif len(cats_present) == 1:
            overall = category_scores[cats_present[0]]
        else:
            gm = 1.0
            for cat in cats_present:
                gm *= float(category_scores[cat]) ** (1.0 / len(cats_present))
            overall = gm
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
    Physics admissibility from logged RMS and scales.

    Reporting uses three levels (no extra aggregation API): (1) per residual key in
    ``per_key``; (2) geometric mean within each category — ``per_category`` keys
    ``laws``, ``constitutive``, ``scaling``, ``data``; (3) overall score — geometric mean of
    the category scores that are present.
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


def _closure_kind_for_key(flat_key: str) -> Optional[str]:
    """Last path segment for constitutive/scaling keys (e.g. chain_dx, ref_delta)."""
    parts = flat_key.split("/")
    if len(parts) < 2:
        return None
    if parts[0] not in ("constitutive", "scaling"):
        return None
    return parts[-1]


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


def _build_visualize_bundle(
    log: List[Dict[str, Any]],
    keys: Optional[List[str]],
    r_ref: Optional[Dict[str, float]],
    max_legend_keys: int,
) -> Optional[Dict[str, Any]]:
    """
    Shared arrays and metadata for :func:`visualize` (matplotlib or plotly).

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
    legend_keys = plot_keys[: max(1, int(max_legend_keys))]
    buckets = _keys_by_category(plot_keys)
    ordered_keys: List[str] = []
    for cat in ("laws", "constitutive", "scaling", "data"):
        ordered_keys.extend(sorted(buckets[cat]))
    if not ordered_keys:
        ordered_keys = list(plot_keys)
    mat = np.zeros((len(ordered_keys), n))
    for j in range(n):
        for i, kk in enumerate(ordered_keys):
            v = metrics[j]["admissibility_score"].get(kk)
            mat[i, j] = float(v) if v is not None else float("nan")
    short_labels = [k if len(k) < 42 else k[:39] + "…" for k in ordered_keys]
    last_rn = metrics[-1]["r_norm"]
    ranked = sorted(
        ((k, float(last_rn[k])) for k in plot_keys if k in last_rn),
        key=lambda t: t[1],
        reverse=True,
    )
    last_adm = metrics[-1]["admissibility_score"]
    kind_vals: Dict[str, List[float]] = {}
    for k in plot_keys:
        kind = _closure_kind_for_key(k)
        if kind is None:
            continue
        if k in last_adm:
            kind_vals.setdefault(kind, []).append(float(last_adm[k]))
    kinds_order = (
        sorted(kind_vals.keys(), key=lambda x: (-len(kind_vals[x]), x)) if kind_vals else []
    )
    means_closure = [float(np.mean(kind_vals[kk])) for kk in kinds_order]
    om = [len(e.get("omitted") or []) for e in log]
    inf = [len(e.get("inferred") or []) for e in log]
    top5 = [k for k, _ in ranked[:5]] if ranked else plot_keys[:5]
    cat_colors = {
        "laws": "#1f77b4",
        "constitutive": "#ff7f0e",
        "scaling": "#2ca02c",
        "data": "#9467bd",
    }
    last_cat = metrics[-1]["category_admissibility_score"]
    cats_fin = [
        (c, float(last_cat[c]))
        for c in cat_colors
        if c in last_cat and np.isfinite(last_cat[c])
    ]
    rms_title = "RMS per key"
    if len(plot_keys) > len(legend_keys):
        rms_title = f"RMS per key (showing {len(legend_keys)} of {len(plot_keys)} keys)"
    return {
        "log": log,
        "metrics": metrics,
        "n": n,
        "indices": indices,
        "plot_keys": plot_keys,
        "legend_keys": legend_keys,
        "buckets": buckets,
        "ordered_keys": ordered_keys,
        "mat": mat,
        "short_labels": short_labels,
        "ranked": ranked,
        "kind_vals": kind_vals,
        "kinds_order": kinds_order,
        "means_closure": means_closure,
        "om": om,
        "inf": inf,
        "top5": top5,
        "cats_fin": cats_fin,
        "rms_title": rms_title,
        "cat_colors": cat_colors,
        "np": np,
    }


def visualize(
    log: List[Dict[str, Any]],
    keys: Optional[List[str]] = None,
    backend: str = "matplotlib",
    *,
    r_ref: Optional[Dict[str, float]] = None,
    max_legend_keys: int = 16,
) -> Any:
    """
    Multi-panel monitor dashboard from ``ResidualEngine`` log entries (``rms``, ``scale``).

    Uses the same R_norm / admissibility rules as :func:`audit` via
    :func:`_compute_log_step_metrics` and **does not mutate** ``log``.

    **Backends**

    - ``matplotlib`` — static figure (requires ``matplotlib``).
    - ``plotly`` — interactive figure (zoom/pan/hover; requires ``pip install plotly`` or ``moju[viz]``).
    - ``none`` — returns ``None``.

    Panels (when data allows): RMS and admissibility per key; overall + category trajectories;
    admissibility heatmap; top ``R_norm`` keys; closure-type summary; omitted/inferred counts;
    twin RMS / ``R_norm`` for top keys; category radar; worst key per category.

    Parameters
    ----------
    log
        Entries from ``ResidualEngine.log`` (after ``compute_residuals``).
    keys
        Subset of flat residual keys to plot; default = all keys in the first entry.
    backend
        ``matplotlib``, ``plotly``, or ``none``.
    r_ref
        Optional per-key reference scale overrides (same as :func:`audit`).
    max_legend_keys
        Cap legend entries on per-key line plots for readability.
    """
    if backend == "none":
        return None

    bundle = _build_visualize_bundle(log, keys, r_ref, max_legend_keys)
    if bundle is None:
        return None

    if backend == "plotly":
        try:
            from moju.monitor.visualize_plotly import build_plotly_monitor_figure

            return build_plotly_monitor_figure(bundle)
        except ImportError:
            return None

    if backend != "matplotlib":
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    np = bundle["np"]
    log = bundle["log"]
    metrics = bundle["metrics"]
    n = bundle["n"]
    indices = bundle["indices"]
    plot_keys = bundle["plot_keys"]
    legend_keys = bundle["legend_keys"]
    buckets = bundle["buckets"]
    ordered_keys = bundle["ordered_keys"]
    mat = bundle["mat"]
    short_labels = bundle["short_labels"]
    ranked = bundle["ranked"]
    kinds_order = bundle["kinds_order"]
    means_closure = bundle["means_closure"]
    om = bundle["om"]
    inf = bundle["inf"]
    top5 = bundle["top5"]
    cats_fin = bundle["cats_fin"]
    rms_title = bundle["rms_title"]
    cat_colors = bundle["cat_colors"]

    fig = plt.figure(figsize=(13, 22))
    gs = fig.add_gridspec(
        8,
        2,
        height_ratios=[1.0, 1.0, 0.9, 1.35, 1.0, 0.75, 1.0, 0.85],
        hspace=0.5,
        wspace=0.28,
    )

    # --- 1–2: RMS and admissibility per key ---
    ax_rms = fig.add_subplot(gs[0, :])
    ax_adm = fig.add_subplot(gs[1, :], sharex=ax_rms)
    for k in legend_keys:
        rms_vals = [e.get("rms", {}).get(k) for e in log]
        if all(v is not None for v in rms_vals):
            ax_rms.plot(indices, rms_vals, label=k, alpha=0.85)
        adm_vals = [metrics[i]["admissibility_score"].get(k) for i in range(n)]
        if all(v is not None for v in adm_vals):
            ax_adm.plot(indices, adm_vals, label=k, alpha=0.85)
    ax_rms.set_ylabel("RMS")
    ax_rms.set_title(rms_title)
    ax_rms.legend(loc="upper right", fontsize=7, ncol=2)
    ax_rms.grid(True, alpha=0.25)
    ax_adm.set_ylabel("Admissibility")
    ax_adm.set_title("Admissibility per key (1/(1+R_norm))")
    ax_adm.legend(loc="lower right", fontsize=7, ncol=2)
    ax_adm.grid(True, alpha=0.25)
    ax_adm.set_ylim(0.0, 1.02)

    # --- 3: Overall + category geometric means ---
    ax_sum = fig.add_subplot(gs[2, :], sharex=ax_rms)
    overall_s = [metrics[i]["overall_admissibility_score"] for i in range(n)]
    if any(np.isfinite(overall_s)):
        ax_sum.plot(indices, overall_s, color="black", linewidth=2.2, label="Overall")
    for cat, col in cat_colors.items():
        ys = []
        for i in range(n):
            v = metrics[i]["category_admissibility_score"].get(cat)
            ys.append(float(v) if v is not None else float("nan"))
        if any(np.isfinite(ys)):
            ax_sum.plot(indices, ys, color=col, linestyle="--", alpha=0.9, label=cat)
    ax_sum.set_ylabel("Admissibility")
    ax_sum.set_title("Overall and per-category geometric mean admissibility")
    ax_sum.legend(loc="best", fontsize=8, ncol=5)
    ax_sum.grid(True, alpha=0.25)
    ax_sum.set_ylim(0.0, 1.02)

    # --- 4: Heatmap (keys × step) ---
    ax_hm = fig.add_subplot(gs[3, :])
    mat_m = np.ma.masked_invalid(mat)
    im = ax_hm.imshow(
        mat_m,
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax_hm.set_yticks(range(len(ordered_keys)))
    ax_hm.set_yticklabels(short_labels, fontsize=6)
    step = max(1, n // 20)
    xt = list(range(0, n, step))
    ax_hm.set_xticks(xt)
    ax_hm.set_xticklabels([str(int(i)) for i in xt])
    ax_hm.set_xlabel("Log index")
    ax_hm.set_title("Admissibility heatmap (keys × step)")
    fig.colorbar(im, ax=ax_hm, fraction=0.02, pad=0.01, label="Admissibility")

    # --- 5: Top offenders (last step) ---
    ax_bar = fig.add_subplot(gs[4, :])
    top_n = min(15, len(ranked))
    if top_n:
        rk, rv = zip(*ranked[:top_n])
        y_pos = np.arange(top_n)
        ax_bar.barh(y_pos, rv, color="#c44e52", alpha=0.85)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels([x if len(x) < 50 else x[:47] + "…" for x in rk], fontsize=7)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel("R_norm (last step)")
        ax_bar.set_title("Top residual keys by R_norm (last step)")
        ax_bar.grid(True, axis="x", alpha=0.25)
    else:
        ax_bar.set_visible(False)

    # --- 6: Closure-type mean admissibility (last step) ---
    ax_cl = fig.add_subplot(gs[5, 0])
    if kinds_order:
        xpos = np.arange(len(kinds_order))
        ax_cl.bar(xpos, means_closure, color="#4c72b0", alpha=0.88)
        ax_cl.set_xticks(xpos)
        ax_cl.set_xticklabels(kinds_order, rotation=35, ha="right", fontsize=7)
        ax_cl.set_ylabel("Mean admissibility")
        ax_cl.set_title("Constitutive/scaling by closure type (last step)")
        ax_cl.set_ylim(0.0, 1.02)
        ax_cl.grid(True, axis="y", alpha=0.25)
    else:
        ax_cl.text(0.5, 0.5, "No constitutive/scaling keys", ha="center", va="center")
        ax_cl.set_axis_off()

    # --- 7: Omitted / inferred counts ---
    ax_om = fig.add_subplot(gs[5, 1])
    if max(om + inf, default=0) > 0:
        width = 0.35
        ax_om.bar(indices - width / 2, om, width, label="omitted", color="#8c564b", alpha=0.85)
        ax_om.bar(indices + width / 2, inf, width, label="inferred", color="#17becf", alpha=0.85)
        ax_om.set_xlabel("Log index")
        ax_om.set_ylabel("Count")
        ax_om.set_title("Omitted / inferred log messages")
        ax_om.legend(fontsize=8)
        ax_om.grid(True, axis="y", alpha=0.25)
    else:
        ax_om.text(0.5, 0.5, "No omitted/inferred entries", ha="center", va="center")
        ax_om.set_axis_off()

    # --- 8: Twin RMS / R_norm for top keys by R_norm ---
    ax_twin = fig.add_subplot(gs[6, :], sharex=ax_rms)
    ax_t2 = ax_twin.twinx()
    for k in top5:
        rms_s = [e.get("rms", {}).get(k) for e in log]
        rn_s = [metrics[i]["r_norm"].get(k) for i in range(n)]
        if all(v is not None for v in rms_s):
            ax_twin.plot(indices, rms_s, label=f"{k} RMS", linestyle="-", alpha=0.9)
        if all(v is not None for v in rn_s):
            ax_t2.plot(indices, rn_s, label=f"{k} R_norm", linestyle="--", alpha=0.75)
    ax_twin.set_ylabel("RMS", color="#333")
    ax_t2.set_ylabel("R_norm", color="#666")
    ax_twin.set_title("Top keys: RMS (solid) vs R_norm (dashed, right axis)")
    ax_twin.grid(True, alpha=0.25)
    h1, l1 = ax_twin.get_legend_handles_labels()
    h2, l2 = ax_t2.get_legend_handles_labels()
    ax_twin.legend(h1 + h2, l1 + l2, loc="best", fontsize=6, ncol=2)

    # --- 9: Radar (category scores, last step) ---
    ax_rad = fig.add_subplot(gs[7, 0], projection="polar")
    if len(cats_fin) >= 2:
        labels = [c[0] for c in cats_fin]
        vals = [c[1] for c in cats_fin]
        angles = np.linspace(0.0, 2 * np.pi, len(labels), endpoint=False)
        vals_c = vals + vals[:1]
        angles_c = np.concatenate([angles, [angles[0]]])
        ax_rad.plot(angles_c, vals_c, "o-", linewidth=2, color="#bcbd22")
        ax_rad.fill(angles_c, vals_c, alpha=0.25, color="#bcbd22")
        ax_rad.set_xticks(angles)
        ax_rad.set_xticklabels(labels, fontsize=8)
        ax_rad.set_ylim(0.0, 1.0)
        ax_rad.set_title("Category admissibility (last step)", y=1.08, fontsize=10)
    else:
        ax_rad.set_visible(False)

    # Spare panel: min admissibility per category (worst key) over time
    ax_worst = fig.add_subplot(gs[7, 1], sharex=ax_rms)
    for cat, col in cat_colors.items():
        ys = []
        bk = buckets[cat]
        if not bk:
            continue
        for i in range(n):
            adm = metrics[i]["admissibility_score"]
            mnv = min((float(adm[k]) for k in bk if k in adm), default=float("nan"))
            ys.append(mnv)
        if any(np.isfinite(ys)):
            ax_worst.plot(indices, ys, color=col, label=f"min in {cat}", alpha=0.9)
    ax_worst.set_ylabel("Admissibility")
    ax_worst.set_title("Worst key per category (min admissibility)")
    ax_worst.legend(loc="best", fontsize=7)
    ax_worst.grid(True, alpha=0.25)
    ax_worst.set_ylim(0.0, 1.02)

    ax_twin.set_xlabel("Log index / step")
    ax_worst.set_xlabel("Log index / step")
    fig.suptitle("Moju monitor visualization", fontsize=12, y=0.995)
    return fig


class ResidualEngine:
    """
    Governing laws (Laws.*), optional group specs to enrich state, and model/group closures.

    Entry points:
      - Path A (recommended): provide (model, params, collocation) and a state_builder
        so moju can build state_pred (and derivatives) internally.
      - Path B (advanced): provide state_pred directly.

    Closure policy:
      - chain_dx / chain_dt run only when predicted_spatial / predicted_temporal are non-empty.
      - ref_delta runs when state_ref is provided (independent of predicted_*).
      - implied_delta runs when implied_value_key or implied_fn is set; omitted if implied is missing.
      - A spec with no chain, no ref_delta, and no implied_delta does nothing (optional omit log).

    Audit spec shape (constitutive_audit / scaling_audit items):
      {
        "name": "sutherland_mu",               # Models.<name> or Groups.<name>
        "output_key": "mu",                    # state key for F output; expects d_mu_dx / d_mu_dt for chain
        "state_map": {"T": "T", "mu0": "mu0", "T0": "T0", "S": "S"},  # fn arg -> state key
        "predicted_spatial": ["T"],            # state keys varying in x
        "predicted_temporal": ["T"],           # state keys varying in t
      }

    Derivative convention in state_pred: d_<state_key>_dx, _dy, _dz, _dt as required by audits.
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
            from moju.monitor.config import MonitorConfig, audit_spec_to_engine_dict

            if isinstance(config, MonitorConfig):
                constants = config.constants
                laws = config.laws
                groups = config.groups
                constitutive_audit = [audit_spec_to_engine_dict(s) for s in config.constitutive_audit]
                scaling_audit = [audit_spec_to_engine_dict(s) for s in config.scaling_audit]
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
                ivk = spec.get("implied_value_key")
                ifn = spec.get("implied_fn")
                if ivk and ifn is not None:
                    raise ValueError(
                        f"{category}:{name} use only one of implied_value_key and implied_fn, not both"
                    )
                csa = list(spec.get("chain_spatial_axes") or ["x"])
                allowed = set(CHAIN_SPATIAL_DERIVS)
                bad = [a for a in csa if a not in allowed]
                if bad:
                    raise ValueError(
                        f"{category}:{name} chain_spatial_axes must be subset of {sorted(allowed)}, "
                        f"invalid: {bad}"
                    )
                if not csa:
                    raise ValueError(f"{category}:{name} chain_spatial_axes must be non-empty")

        _validate_specs(self.constitutive_audit, MODEL_FNS, "constitutive")
        _validate_specs(self.scaling_audit, GROUP_FNS, "scaling")
        self._validate_pi_constant_specs()

        self._log: List[Dict[str, Any]] = []
        self._index = 0

    @property
    def log(self) -> List[Dict[str, Any]]:
        return self._log

    def _validate_pi_constant_specs(self) -> None:
        for spec in self.scaling_audit:
            if not spec.get("invariance_pi_constant"):
                continue
            name = spec["name"]
            if name not in GROUP_PI_CONSTANT_RECIPES:
                raise ValueError(
                    f"scaling:{name} invariance_pi_constant requires a built-in π-constant recipe; "
                    f"supported: {sorted(GROUP_PI_CONSTANT_RECIPES.keys())}"
                )
            recipe = GROUP_PI_CONSTANT_RECIPES[name]
            sm = spec.get("state_map") or {}
            for arg_name, _ in recipe:
                if arg_name not in sm:
                    raise ValueError(
                        f"scaling:{name} π-constant recipe needs state_map entry for arg {arg_name!r}"
                    )
            cmp_keys = list(spec.get("invariance_compare_keys") or [])
            if not cmp_keys:
                raise ValueError(
                    f"scaling:{name} invariance_pi_constant requires non-empty invariance_compare_keys"
                )
            c = float(spec.get("invariance_scale_c", 10.0))
            if c <= 1.0:
                raise ValueError(f"scaling:{name} invariance_scale_c must be > 1, got {c}")
            for arg_name, _ in recipe:
                sk = sm[arg_name]
                if sk not in self.constants:
                    raise ValueError(
                        f"scaling:{name} π-constant requires key {sk!r} in ResidualEngine.constants "
                        f"(arg {arg_name!r})"
                    )
            if self.state_builder is None:
                raise ValueError(
                    f"scaling:{name} invariance_pi_constant requires ResidualEngine(state_builder=...) (Path A only)"
                )

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
    ) -> Dict[str, Any]:
        """
        Compute residuals.

        Path A: pass (model, params, collocation) and configure engine.state_builder.
        Path B: pass state_pred directly.

        If ``auto_path_b_derivatives`` is True, uses default ``PathBGridConfig``; if a
        ``PathBGridConfig`` instance, uses that layout. Fills missing ``d_*_dx``/``_dy``/``_dz``/``_dt``
        keys required by audits via finite differences (after group state build, before laws).
        Warnings are appended to the log ``inferred`` list when enabled.

        If ``fill_law_fd`` is True, ``auto_path_b_derivatives`` must also be enabled; missing
        **registered** ``Laws.*`` inputs (e.g. ``phi_laplacian``, ``u_grad``) are filled from
        primitives on the same grid when possible (see ``law_fd_recipes``).
        """
        path_a = state_pred is None
        pi_specs = [s for s in self.scaling_audit if s.get("invariance_pi_constant")]
        if pi_specs and not path_a:
            raise ValueError(
                "π-constant scaling audit (invariance_pi_constant) requires Path A: "
                "call compute_residuals(..., model=..., params=..., collocation=...) "
                "without passing state_pred."
            )

        residuals: Dict[str, Any] = {"laws": {}}

        if state_pred is None:
            if self.state_builder is None:
                raise ValueError("Path A requires ResidualEngine(state_builder=...)")
            if model is None or params is None or collocation is None:
                raise ValueError("Path A requires model, params, and collocation")
            state_pred = self.state_builder(model, params, collocation, self.constants)

        state_pred_built = self._state_builder(state_pred)
        merged = {**state_pred_built, **self.constants}

        omitted_msgs: List[str] = []
        inferred_msgs: List[str] = []

        def _maybe_log_omit(msg: str) -> None:
            if self.enable_omit_messages:
                omitted_msgs.append(msg)

        def _maybe_log_infer(msg: str) -> None:
            if self.enable_omit_messages:
                inferred_msgs.append(msg)

        if fill_law_fd and not auto_path_b_derivatives:
            raise ValueError(
                "fill_law_fd=True requires auto_path_b_derivatives=True or a PathBGridConfig"
            )

        if auto_path_b_derivatives:
            from moju.monitor.path_b_derivatives import PathBGridConfig, fill_path_b_derivatives

            if auto_path_b_derivatives is True:
                grid = PathBGridConfig()
            elif isinstance(auto_path_b_derivatives, PathBGridConfig):
                grid = auto_path_b_derivatives
            else:
                raise TypeError(
                    "auto_path_b_derivatives must be False, True, or a PathBGridConfig instance"
                )
            state_pred_built, pb_warn = fill_path_b_derivatives(
                state_pred_built,
                constitutive_audit=self.constitutive_audit,
                scaling_audit=self.scaling_audit,
                laws_spec=self.laws_spec,
                constants=self.constants,
                grid=grid,
                copy=False,
                fill_law_recipes=bool(fill_law_fd),
            )
            merged = {**state_pred_built, **self.constants}
            for w in pb_warn:
                _maybe_log_infer(f"path_b_derivatives: {w}")

        for spec in self.laws_spec:
            name = spec["name"]
            state_map = spec["state_map"]
            kwargs = _kwargs_from_state(merged, self.constants, state_map)
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

                has_implied = bool(spec.get("implied_value_key")) or spec.get("implied_fn") is not None
                if (
                    not predicted_spatial
                    and not predicted_temporal
                    and state_ref is None
                    and not has_implied
                ):
                    _maybe_log_omit(
                        f"{category}:{name} omitted: no chain, ref_delta, or implied_delta applicable"
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

                if has_implied:
                    arr = compute_implied_delta(
                        fn=fn,
                        arg_names=arg_names,
                        state_map=state_map,
                        state_pred=merged,
                        constants=self.constants,
                        implied_value_key=spec.get("implied_value_key"),
                        implied_fn=spec.get("implied_fn"),
                    )
                    if arr is not None:
                        out[f"{name}/implied_delta"] = jnp.asarray(arr)

                if predicted_spatial and output_key is not None:
                    spatial_axes = list(spec.get("chain_spatial_axes") or ["x"])
                    for spatial_axis in spatial_axes:
                        if spatial_axis not in CHAIN_SPATIAL_DERIVS:
                            continue
                        chain_key = f"chain_d{spatial_axis}"
                        if closure_mode == "weak":
                            arr = compute_chain_weak(
                                fn=fn,
                                arg_names=arg_names,
                                output_key=output_key,
                                state_map=state_map,
                                state_pred=merged,
                                constants=self.constants,
                                predicted_varying=predicted_spatial,
                                deriv=spatial_axis,
                                weight_key=quadrature_weights.get(spatial_axis),
                            )
                            _maybe_log_infer(
                                f"{category}:{name} using weak {chain_key}"
                            )
                        else:
                            arr = compute_chain(
                                fn=fn,
                                arg_names=arg_names,
                                output_key=output_key,
                                state_map=state_map,
                                state_pred=merged,
                                constants=self.constants,
                                predicted_varying=predicted_spatial,
                                deriv=spatial_axis,
                            )
                        if arr is not None:
                            out[f"{name}/{chain_key}"] = jnp.asarray(arr)

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

        pi_constant_scales: Dict[str, float] = {}
        if self.scaling_audit or self.scaling_custom:
            s = _run_specs(self.scaling_audit, registry=GROUP_FNS, category="scaling")
            if self.scaling_custom:
                for spec in self.scaling_custom:
                    cname = spec["name"]
                    arr = spec["fn"](merged, self.constants)
                    if arr is not None:
                        s[f"custom/{cname}"] = jnp.asarray(arr)
            if path_a and self.state_builder is not None:
                for spec in self.scaling_audit:
                    if not spec.get("invariance_pi_constant"):
                        continue
                    name = spec["name"]
                    c = float(spec.get("invariance_scale_c", 10.0))
                    if c <= 1.0:
                        raise ValueError(f"scaling:{name} invariance_scale_c must be > 1, got {c}")
                    recipe = GROUP_PI_CONSTANT_RECIPES[name]
                    state_map = spec.get("state_map") or {}
                    constants_scaled = apply_pi_constant_recipe(
                        self.constants, recipe, state_map, c
                    )
                    state_pred_pi = self.state_builder(model, params, collocation, constants_scaled)
                    merged_scaled = {
                        **self._state_builder(state_pred_pi, constants_scaled),
                        **constants_scaled,
                    }
                    fn, arg_names = GROUP_FNS[name]
                    kb = _kwargs_from_state(merged, self.constants, state_map)
                    ks = _kwargs_from_state(merged_scaled, constants_scaled, state_map)
                    val_b = fn(**{an: kb[an] for an in arg_names})
                    val_s = fn(**{an: ks[an] for an in arg_names})
                    if not jnp.allclose(
                        jnp.asarray(val_b), jnp.asarray(val_s), rtol=1e-4, atol=1e-6
                    ):
                        raise ValueError(
                            f"scaling:{name} π-constant scaled inputs did not preserve group value "
                            f"(baseline {val_b!r} vs scaled {val_s!r})"
                        )
                    compare_keys = list(spec.get("invariance_compare_keys") or [])
                    parts_r: List[jnp.ndarray] = []
                    parts_scale: List[jnp.ndarray] = []
                    for ck in compare_keys:
                        if ck not in merged:
                            raise KeyError(
                                f"scaling:{name} invariance_compare_keys: {ck!r} missing from baseline merged state"
                            )
                        if ck not in merged_scaled:
                            raise KeyError(
                                f"scaling:{name} invariance_compare_keys: {ck!r} missing from scaled merged state"
                            )
                        vb = jnp.asarray(merged[ck])
                        vs = jnp.asarray(merged_scaled[ck])
                        parts_r.append(jnp.ravel(vs - vb))
                        parts_scale.append(jnp.ravel(jnp.abs(vs)))
                    r_pi = jnp.concatenate(parts_r) if parts_r else jnp.array([0.0])
                    stacked_abs = jnp.concatenate(parts_scale) if parts_scale else jnp.array([0.0])
                    flat_pi = f"{name}/pi_constant"
                    s[flat_pi] = r_pi
                    scale_key = f"scaling/{flat_pi}"
                    mean_abs = float(jax.device_get(jnp.mean(stacked_abs)))
                    pi_constant_scales[scale_key] = float(_SCALE_EPS + mean_abs)
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
        state_ref_built_for_scale = None
        if state_ref is not None:
            state_ref_built_for_scale = self._state_builder(state_ref)
        scale_per_key = _state_derived_scale_per_key(
            flat.keys(),
            merged,
            self.laws_spec,
            self.constitutive_audit,
            self.scaling_audit,
            state_ref_built_for_scale,
            to_python=log_to_python,
        )
        scale_per_key.update(pi_constant_scales)
        entry: Dict[str, Any] = {"index": self._index, "rms": rms_per_key, "scale": scale_per_key}
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
                ivk = spec.get("implied_value_key")
                if ivk:
                    keys.add(ivk)
        return keys

    def required_derivative_keys(self) -> Set[str]:
        sx, st = collect_audit_derivative_keys(
            list(self.constitutive_audit), list(self.scaling_audit)
        )
        return sx | st


def list_constitutive_models():
    from moju.monitor.closure_registry import list_models
    return list_models()


def list_scaling_closure_ids():
    from moju.monitor.closure_registry import list_groups
    return list_groups()
