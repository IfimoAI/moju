"""
Constitutive Divergence card — where catalog ``F(pred)`` and the law-implied
quantity disagree.

For every constitutive audit row, the engine stashes a sidecar entry on
``residuals["closure_debug"][basename]``::

    {
        "pred":       F(pred_args),                       # always present
        "implied":    implied,
        "raw":        pred - implied,                     # dimensional difference (diagnostic)
        "delta":      raw / (|pred| + eps),               # array fed to R_eff (== log key value)
        "mode":       "subtract",
        "output_key": str,
        "law_name":   str | None,
        "model_name": str,
    }

``closure_debug["delta"]`` is exactly :func:`_normalized_divergence` applied to
``pred`` and ``implied``, so the divergence and consistency plots, the log key
``…/implied_delta``, and the array RMSed into ``R_eff`` are all the same numbers.

This module produces four visualisation modes plus a composite 2x2 dashboard.

Modes
-----

- ``"spatial"`` — three-panel heatmap row: model output, law-implied side,
  signed normalised divergence (diverging colormap centered on zero).
- ``"scatter"`` — pred vs implied scatter with the y = x identity line,
  colored by ``|divergence|``; reports % of points within ±0.1, ±0.5, ±1 %.
- ``"distribution"`` — histogram of normalised divergence with admissibility
  threshold bands (green / amber / red).
- ``"hotspot"`` — top-k worst points by ``|divergence|``, plotted in 2-D /
  3-D over collocation coordinates ``(x, y, [z], [t])``.

The helper :func:`build_spatial_normalized_divergence_figure` renders **only**
the normalised divergence (single axes), for the monitor wide row.

Composite entry: :func:`build_constitutive_divergence_dashboard` returns a
2x2 :class:`plotly.graph_objects.Figure` with one panel of each mode for
the basename whose final-step R_norm is largest (or the explicit basename
passed by the caller).
"""
from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

from moju.monitor.auditor import (
    CONSTITUTIVE_AXIS_PAD_FRAC,
    CONSTITUTIVE_BAND_FRAC_HIGH,
    CONSTITUTIVE_BAND_FRAC_LOW,
    CONSTITUTIVE_BAND_FRAC_MOD,
)
from moju.monitor.visualize_theme import (
    MOJU_LIGHT,
    apply_theme,
    get_theme,
    themed_axis_style,
    themed_colorbar,
)
from moju.monitor.visualize_components import _empty_card  # type: ignore[attr-defined]
from moju.monitor.visualize_labels import (
    pretty_residual_key,
    truncate_display_label,
)


_DIVERGENCE_EPS: float = 1e-30
_SLICE_MIN_FINITE_FRACTION: float = 0.01
_SLICE_SPIKE_RATIO: float = 10.0

# User-facing divergence panel wording (distinct from closure_debug JSON keys ``pred`` / ``implied``).
USER_CONSTIT_MODEL: str = "Model"
USER_CONSTIT_IMPLIED: str = "Implied"


def _user_constitutive_side_labels() -> Tuple[str, str]:
    return USER_CONSTIT_MODEL, USER_CONSTIT_IMPLIED


def _title_case_token(text: Any) -> str:
    return str(text or "").replace("_", " ").strip().title()


def constitutive_term_label(
    debug_entry: Optional[Dict[str, Any]] = None,
    residual_basename: Optional[str] = None,
) -> str:
    """User-facing constitutive term name for panel titles and value axes."""
    entry = debug_entry or {}
    for key in ("display_name", "display_label", "label"):
        val = entry.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    name = entry.get("model_name")
    if isinstance(name, str) and name.strip():
        s = name.strip()
    elif residual_basename:
        s = str(residual_basename).split("/", 1)[0].strip()
    else:
        ok = entry.get("output_key")
        s = str(ok).strip() if ok else ""
    if not s:
        return "Constitutive Term"
    for suffix in ("_from_re", "_from_st", "_from_pe"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return _title_case_token(s)


def divergence_y_quantity_label(debug_entry: Optional[Dict[str, Any]] = None) -> str:
    """Y-axis label for constitutive quantity on spatial / dissonance panels."""
    return constitutive_term_label(debug_entry)


def _constitutive_dissonance_title(
    *,
    is_transient: bool,
    is_worst_slice: bool,
    t_value: Optional[float] = None,
    time_slice_criterion: Optional[str] = None,
    spatial_slice_criterion: Optional[str] = None,
) -> str:
    details: List[str] = []
    if is_transient:
        if time_slice_criterion == "mean":
            t_str = f"t ≈ {t_value:.4g}, mean t slice (max degenerate)" if t_value is not None else "mean t slice (max degenerate)"
        else:
            t_str = f"worst t ≈ {t_value:.4g}" if t_value is not None else "worst t"
        details.append(t_str)
    if is_worst_slice:
        if spatial_slice_criterion == "mean":
            details.append("mean slice (max degenerate)")
        else:
            details.append("worst slice")
    return "Constitutive Consistency" + (f" ({', '.join(details)})" if details else "")


def constitutive_divergence_title(term_label: str) -> str:
    return f"Constitutive Divergence ({term_label})"


def constitutive_divergence_title_for_bundle(
    bundle: Dict[str, Any],
    residual_basename: Optional[str] = None,
) -> str:
    debug = _closure_debug(bundle)
    bn = residual_basename or _auto_select_basename(bundle, debug)
    entry = debug.get(bn or "", {}) if debug else {}
    return constitutive_divergence_title(constitutive_term_label(entry, bn))



def _spatial_position_axis_title(bundle_spatial: Any) -> str:
    """Match monitor spatial labelling."""
    sp = bundle_spatial if isinstance(bundle_spatial, dict) else None
    if not sp or sp.get("kind") != "1d":
        return "Position x"
    ax = sp.get("position_axis") or "x"
    return f"Position {ax}"


def infer_divergence_abscissa(
    bundle: Dict[str, Any],
    n: int,
    *,
    hint_axis: Optional[str] = None,
) -> Tuple[np.ndarray, str]:
    """
    Return ``(xs, x_title)`` aligned to length ``n`` for divergence line plots.

    Priority: spatial 1-D ``bundle['spatial']['x']``; then ``coord_snapshot`` axes
    (hint axis first among x/y/z/t, then defaults); fallback index with ``Sample index``.

    For each axis the unique grid vector (``coord_snapshot[axis + "_grid"]``) is tried before
    the raw flattened vector so that plots built from a flattened meshgrid collocation still
    resolve to the physical 0-to-L coordinate range.
    """
    if n <= 0:
        return np.asarray([], dtype=float), "Sample index"
    spatial_any = bundle.get("spatial") or {}
    if isinstance(spatial_any, dict) and spatial_any.get("kind") == "1d":
        x_sp = spatial_any.get("x")
        if x_sp is not None:
            arr = np.asarray(x_sp, dtype=float).ravel()
            if arr.shape[0] == n:
                return arr, _spatial_position_axis_title(spatial_any)
    cs: Any = {}
    log = bundle.get("log") or []
    if log and isinstance(log[-1], dict):
        cs = log[-1].get("coord_snapshot") or {}
    if isinstance(cs, dict):
        axes_order = ("x", "y", "z", "t")
        hinted = hint_axis.strip().lower() if isinstance(hint_axis, str) and hint_axis.strip() else None
        probe = []
        if hinted and hinted in axes_order:
            probe.append(hinted)
        for ax in axes_order:
            if ax not in probe:
                probe.append(ax)
        titles = {"x": "Position x", "y": "Position y", "z": "Position z", "t": "Time t"}
        for ax in probe:
            # Try unique grid axis first (set when collocation is a flattened meshgrid).
            v_grid = cs.get(ax + "_grid")
            if v_grid is not None:
                arr = np.asarray(v_grid, dtype=float).ravel()
                if arr.shape[0] == n:
                    return arr, titles.get(ax, f"Position {ax}")
            v = cs.get(ax)
            arr = np.asarray(v, dtype=float).ravel() if v is not None else np.asarray([])
            if arr.shape[0] == n:
                return arr, titles.get(ax, f"Position {ax}")

    return np.arange(n, dtype=float), "Sample index"


def _closure_debug(bundle: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Resolve the ``closure_debug`` sidecar in a visualization bundle."""
    if not isinstance(bundle, dict):
        return {}
    cd = bundle.get("closure_debug")
    if isinstance(cd, dict) and cd:
        return cd
    residuals = bundle.get("residuals")
    if isinstance(residuals, dict):
        nested = residuals.get("closure_debug")
        if isinstance(nested, dict):
            return nested
    return {}


def _final_rnorm_by_basename(bundle: Dict[str, Any]) -> Dict[str, float]:
    """Map constitutive ``residual_basename`` → final-step R_norm."""
    plot_keys: Optional[List[str]] = bundle.get("plot_keys")
    r_norm_mat = bundle.get("r_norm_mat")
    if not plot_keys or r_norm_mat is None:
        return {}
    mat = np.asarray(r_norm_mat, dtype=float)
    if mat.ndim != 2 or mat.shape[1] != len(plot_keys):
        return {}
    finals = mat[-1, :]
    out: Dict[str, float] = {}
    for k, v in zip(plot_keys, finals):
        if not isinstance(k, str) or not k.startswith("constitutive/"):
            continue
        if not np.isfinite(v):
            continue
        # strip "constitutive/" prefix and "/implied_delta" / "/ref_delta" suffix
        bn = k[len("constitutive/"):]
        if bn.endswith("/implied_delta"):
            bn = bn[: -len("/implied_delta")]
        elif bn.endswith("/ref_delta"):
            bn = bn[: -len("/ref_delta")]
        out[bn] = max(out.get(bn, 0.0), float(v))
    return out


def _auto_select_basename(bundle: Dict[str, Any], debug: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """Pick the basename with the worst final-step R_norm; fall back to first."""
    finals = _final_rnorm_by_basename(bundle)
    if finals:
        candidates = [(bn, r) for bn, r in finals.items() if bn in debug]
        if candidates:
            candidates.sort(key=lambda kv: -kv[1])
            return candidates[0][0]
    return next(iter(debug.keys()), None)


def primary_closure_debug_field_length(bundle: Dict[str, Any]) -> Optional[int]:
    """Flattened sample count for selected ``closure_debug`` basename, if any."""
    debug = _closure_debug(bundle)
    if not debug:
        return None
    bn = _auto_select_basename(bundle, debug)
    if bn is None or bn not in debug:
        return None
    entry = debug[bn]
    arr = _coerce_to_numpy(entry.get("pred"))
    arr = np.asarray(arr, dtype=float)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.size == 0:
        return None
    return int(arr.size) if arr.ndim <= 2 else None


def _coerce_to_numpy(x: Any) -> np.ndarray:
    try:
        return np.asarray(x, dtype=float)
    except Exception:  # noqa: BLE001
        return np.asarray([], dtype=float)


def _sides_for_divergence(
    debug_entry: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """
    Return ``(side_model, side_implied, label_model, label_implied)``.

    User-facing axis labels **Model** / **Implied** (``closure_debug`` retains ``pred`` /
    ``implied`` keys).  ``closure_debug["raw"]`` is the dimensional difference
    ``pred − implied``; the log key ``…/implied_delta`` (and ``closure_debug["delta"]``)
    is the model-normalised fractional residual ``raw / (|pred| + ε)`` that is fed to
    ``R_eff``.  ``_normalized_divergence(pred, implied)`` reproduces ``delta`` exactly.
    """
    lab_m, lab_i = _user_constitutive_side_labels()
    a = _coerce_to_numpy(debug_entry.get("pred"))
    b = _coerce_to_numpy(debug_entry.get("implied"))
    return a, b, lab_m, lab_i


def _normalized_divergence(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalised constitutive divergence: (a − b) / (|a| + ε).

    ``a`` is the model/catalog side; the denominator is |a| so the result is a
    fractional deviation relative to the local model magnitude.
    ``ε = _DIVERGENCE_EPS`` guards against division by zero.
    """
    return (a - b) / (np.abs(a) + _DIVERGENCE_EPS)


def _flatten(a: np.ndarray) -> np.ndarray:
    """Flatten while broadcasting scalars to 1-element arrays."""
    if a.ndim == 0:
        return a.reshape(1)
    return a.reshape(-1)


def _coords_for_points(
    bundle: Dict[str, Any], n_points: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Best-effort flat (x, y, z, t) arrays of length ``n_points``."""
    log = bundle.get("log") or []
    if not log:
        return None, None, None, None
    cs = log[-1].get("coord_snapshot") if isinstance(log[-1], dict) else None
    if not isinstance(cs, dict):
        return None, None, None, None
    out: List[Optional[np.ndarray]] = []
    for axis in ("x", "y", "z", "t"):
        v = cs.get(axis)
        arr = np.asarray(v, dtype=float).ravel() if v is not None else None
        if arr is not None and arr.shape[0] != n_points:
            arr = None
        out.append(arr)
    return out[0], out[1], out[2], out[3]


def _closure_coords_for_reduce(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Coordinate dict for :func:`~moju.monitor.spatial_rnorm_panels._reduce_spatial_array` (snapshot + spatial).

    When the log snapshot contains ``t_grid`` (unique time axis from a flattened meshgrid),
    that vector is used as ``pred["t"]`` so :func:`_reduce_spatial_array` can correctly
    identify and slice the time dimension after data has been reshaped to ``(n_t, n_x)``.
    """
    pred: Dict[str, Any] = {}
    log = bundle.get("log") or []
    if log and isinstance(log[-1], dict):
        cs = log[-1].get("coord_snapshot")
        if isinstance(cs, dict):
            for axis in ("x", "y", "z", "t"):
                v = cs.get(axis)
                if v is None:
                    continue
                pred[str(axis)] = np.asarray(v, dtype=float).ravel()
            # Override flattened t with the unique t_grid vector when available so that
            # _reduce_spatial_array can match shape[0] of a reshaped (n_t, n_x) array.
            t_grid = cs.get("t_grid")
            if t_grid is not None:
                pred["t"] = np.asarray(t_grid, dtype=float).ravel()
    sp_any = bundle.get("spatial")
    if isinstance(sp_any, dict):
        kind = sp_any.get("kind")
        if kind == "1d" and pred.get("x") is None and sp_any.get("x") is not None:
            pred["x"] = np.asarray(sp_any["x"], dtype=float).ravel()
        if kind == "2d":
            coords_dict = sp_any.get("coords") or {}
            cx = coords_dict.get("x") if coords_dict else sp_any.get("x")
            cy = coords_dict.get("y") if coords_dict else sp_any.get("y")
            if cx is not None:
                xv = np.asarray(cx, dtype=float).ravel()
                if pred.get("x") is None or xv.size > np.asarray(pred["x"]).size:
                    pred["x"] = xv
            if cy is not None and pred.get("y") is None:
                pred["y"] = np.asarray(cy, dtype=float).ravel()
    return pred


def worst_div_mean_abs_row_index(div: np.ndarray) -> int:
    """Pick row ``y`` index maximizing mean ``|Δ_norm|`` over ``x`` (worst horizontal stripe)."""
    d = np.asarray(div, dtype=float)
    if d.ndim != 2 or d.shape[0] < 1:
        return 0
    row_means = np.nanmean(np.abs(d), axis=1)
    return int(np.nanargmax(row_means))


def worst_div_max_abs_row_index(div: np.ndarray) -> int:
    """Pick row ``y`` index maximizing max ``|Δ_norm|`` over ``x`` (worst-point row)."""
    d = np.asarray(div, dtype=float)
    if d.ndim != 2 or d.shape[0] < 1:
        return 0
    row_maxes = np.nanmax(np.abs(d), axis=1)
    return int(np.nanargmax(row_maxes))


def _worst_time_slice_index(a: np.ndarray, b: np.ndarray) -> int:
    """Return the time-axis index (axis 0) that maximises mean ``|δ|`` over all spatial axes.

    ``a`` is ``pred``, ``b`` is ``implied``, both shaped ``(n_t, ...)``.  Returns ``0``
    when the arrays are empty or entirely NaN.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim < 1 or a.shape[0] == 0:
        return 0
    div = _normalized_divergence(a, b)
    if div.ndim == 1:
        slice_means = np.abs(div)
    else:
        reduce_axes = tuple(range(1, div.ndim))
        slice_means = np.nanmean(np.abs(div), axis=reduce_axes)
    if not np.any(np.isfinite(slice_means)):
        return 0
    return int(np.nanargmax(slice_means))


def _worst_time_slice_index_max(a: np.ndarray, b: np.ndarray) -> int:
    """Return the time-axis index (axis 0) that maximises max ``|δ|`` over all spatial axes."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim < 1 or a.shape[0] == 0:
        return 0
    div = _normalized_divergence(a, b)
    if div.ndim == 1:
        slice_maxes = np.abs(div)
    else:
        reduce_axes = tuple(range(1, div.ndim))
        slice_maxes = np.nanmax(np.abs(div), axis=reduce_axes)
    if not np.any(np.isfinite(slice_maxes)):
        return 0
    return int(np.nanargmax(slice_maxes))


def _is_degenerate_worst_slice(abs_div: np.ndarray) -> bool:
    """True when a max-|δ| slice pick would be driven by sparse NaNs or a lone spike."""
    vals = np.abs(np.asarray(abs_div, dtype=float)).ravel()
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return True
    if finite.size / max(vals.size, 1) < _SLICE_MIN_FINITE_FRACTION:
        return True
    if finite.size < 2:
        return False
    peak = float(np.max(finite))
    rest = np.delete(finite, int(np.argmax(finite)))
    if rest.size == 0:
        return False
    rest_max = float(np.nanmax(rest))
    if rest_max >= CONSTITUTIVE_BAND_FRAC_HIGH:
        return False
    return peak >= _SLICE_SPIKE_RATIO * max(rest_max, CONSTITUTIVE_BAND_FRAC_HIGH)


def _select_worst_time_slice_index(a: np.ndarray, b: np.ndarray) -> Tuple[int, str]:
    """Pick worst time slice by max |δ|, falling back to mean |δ| when degenerate."""
    t_max = _worst_time_slice_index_max(a, b)
    div = _normalized_divergence(a, b)
    if div.ndim >= 1 and 0 <= t_max < div.shape[0]:
        div_at_t = np.abs(div[t_max])
    else:
        div_at_t = np.abs(div)
    if _is_degenerate_worst_slice(div_at_t):
        return _worst_time_slice_index(a, b), "mean"
    return t_max, "max"


def _select_worst_row_index(div: np.ndarray) -> Tuple[int, str]:
    """Pick worst spatial row by max |δ|, falling back to mean |δ| when degenerate."""
    y_max = worst_div_max_abs_row_index(div)
    if _is_degenerate_worst_slice(div[y_max]):
        return worst_div_mean_abs_row_index(div), "mean"
    return y_max, "max"


def _x_abscissa_0_to_L(cx: np.ndarray, bundle: Dict[str, Any]) -> Tuple[np.ndarray, float, str]:
    """
    Map physical sample positions ``cx`` to ``[0, L]`` where ``L`` is ``spatial.domain_length`` if set,
    otherwise ``max(cx)-min(cx)`` (so the axis spans the collocated extent).
    """
    x = np.asarray(cx, dtype=float).ravel()
    n = x.size
    if n == 0:
        return np.asarray([], dtype=float), 1.0, "Position x"
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    span = hi - lo
    sp = bundle.get("spatial") if isinstance(bundle.get("spatial"), dict) else {}
    L: Optional[float] = None
    if isinstance(sp, dict) and sp.get("domain_length") is not None:
        try:
            L_try = float(sp["domain_length"])
            if np.isfinite(L_try) and L_try > 0:
                L = L_try
        except (TypeError, ValueError):
            L = None
    if L is None:
        L = float(span) if np.isfinite(span) and span > 0 else 1.0
    if not np.isfinite(span) or span <= 0:
        return np.linspace(0.0, float(L), n, dtype=float), float(L), "Position x"
    x_plot = (x - lo) / span * float(L)
    return x_plot, float(L), "Position x"


def _max_delta_label(
    x_plot: np.ndarray,
    a_1d: np.ndarray,
    b_1d: np.ndarray,
) -> str:
    """Return a formatted string 'max X.XX% \u0394 @ x=Y.YY' for the plotted slice.

    Uses the same normalisation as the divergence formula: (|a-b|)/(|a|+\u03b5).
    Returns an empty string if the arrays are empty or all-NaN.
    """
    a = np.asarray(a_1d, dtype=float).ravel()
    b = np.asarray(b_1d, dtype=float).ravel()
    xs = np.asarray(x_plot, dtype=float).ravel()
    if a.size == 0:
        return ""
    delta_pct = np.abs(a - b) / (np.abs(a) + _DIVERGENCE_EPS) * 100.0
    valid = np.isfinite(delta_pct)
    if not np.any(valid):
        return ""
    idx = int(np.nanargmax(delta_pct))
    max_pct = float(delta_pct[idx])
    x_at = float(xs[idx]) if xs.size > idx else float("nan")
    if np.isfinite(x_at):
        return f"max {max_pct:.2f}% \u0394 @ x={x_at:.3g}"
    return f"max {max_pct:.2f}% \u0394"


def _dissonance_y_range(
    model_a: np.ndarray,
    implied_b: Optional[np.ndarray] = None,
) -> List[float]:
    """Y-axis limits for Constitutive Consistency: at least model ±1.5%, expand for implied outliers."""
    a = np.asarray(model_a, dtype=float).ravel()
    w_pad = CONSTITUTIVE_AXIS_PAD_FRAC * (np.abs(a) + _DIVERGENCE_EPS)
    y_lo = float(np.nanmin(a - w_pad))
    y_hi = float(np.nanmax(a + w_pad))
    if implied_b is not None:
        b = np.asarray(implied_b, dtype=float).ravel()
        y_lo = min(y_lo, float(np.nanmin(b)))
        y_hi = max(y_hi, float(np.nanmax(b)))
    span = y_hi - y_lo
    margin = span * 0.05 if span > 0 else abs(float(np.nanmean(a))) * CONSTITUTIVE_AXIS_PAD_FRAC or 0.1
    return [y_lo - margin, y_hi + margin]


def _build_dissonance_band_traces(
    x_plot: np.ndarray,
    a_1d: np.ndarray,
    theme: Any,
) -> Tuple[List[Any], List[float]]:
    """Build acceptability-band fill traces and y-axis range for the dissonance subplot.

    ``a_1d`` must be the **final 1D plotted slice** (post-reduction / worst-row
    extraction), never the pre-reduction full array.

    Band half-widths are spatially varying, derived from:

        δ(x) = |m(x) − mref(x)| / (|mref(x)| + ε) × 100

    where ``mref(x) = a(x)`` (model value) is the reference/normaliser and
    ``ε = _DIVERGENCE_EPS``.  The half-width at threshold p% is therefore:

        w_p(x) = (p / 100) × (|a(x)| + ε)

    Returns
    -------
    band_traces : list[go.Scatter]
        Five fill traces ordered for bottom-to-top rendering:
        alarm lower, alarm upper, warning lower, warning upper, acceptable.
        All have ``showlegend=False`` and blank names so they are invisible to
        ``_add_dissonance_inline_legend``.
    y_range : [float, float]
        ``[y_lo, y_hi]`` covering at least model ±1.5 % (``CONSTITUTIVE_AXIS_PAD_FRAC``).
        Callers with an implied slice should prefer :func:`_dissonance_y_range`.
    """
    import plotly.graph_objects as go

    t = get_theme(theme)
    a = np.asarray(a_1d, dtype=float).ravel()
    xs = np.asarray(x_plot, dtype=float).ravel()

    # Element-wise half-widths: model value a(x) is the reference/normaliser
    w01 = CONSTITUTIVE_BAND_FRAC_HIGH * (np.abs(a) + _DIVERGENCE_EPS)  # ±0.1% green
    w05 = CONSTITUTIVE_BAND_FRAC_MOD * (np.abs(a) + _DIVERGENCE_EPS)   # ±0.5% amber
    w1 = CONSTITUTIVE_BAND_FRAC_LOW * (np.abs(a) + _DIVERGENCE_EPS)    # ±1% red / outer

    xs_rev = xs[::-1]

    def _poly(y_hi: np.ndarray, y_lo: np.ndarray) -> Tuple[List[float], List[float]]:
        return (
            [*xs.tolist(), *xs_rev.tolist()],
            [*y_hi.tolist(), *y_lo[::-1].tolist()],
        )

    # Colors as rgba strings for semi-transparent fills
    c_ok = "rgba(16,185,129,0.15)"     # adm_high emerald
    c_warn = "rgba(245,158,11,0.20)"   # adm_med  amber
    c_alarm = "rgba(239,68,68,0.18)"   # adm_low  red

    def _fill(px: List[float], py: List[float], color: str) -> Any:
        return go.Scatter(
            x=px,
            y=py,
            fill="toself",
            fillcolor=color,
            line=dict(width=0),
            mode="lines",
            name="",
            showlegend=False,
            hoverinfo="skip",
        )

    band_traces = [
        _fill(*_poly(a - w05, a - w1), c_alarm),   # red lower
        _fill(*_poly(a + w1, a + w05), c_alarm),   # red upper
        _fill(*_poly(a - w01, a - w05), c_warn),   # amber lower
        _fill(*_poly(a + w05, a + w01), c_warn),   # amber upper
        _fill(*_poly(a + w01, a - w01), c_ok),     # green centre
    ]

    y_range = _dissonance_y_range(a)

    return band_traces, y_range


def _build_dissonance_tier_lines(
    x_plot: np.ndarray,
    a_1d: np.ndarray,
    theme: Any,
) -> List[Any]:
    """Six faint dotted boundary curves at the ±0.1 %, ±0.5 %, and ±1 % tier edges.

    Rendered behind the band fills so they serve as visible tick-marks at
    the acceptability/warning and warning/alarm boundaries.  Each trace
    carries a hover label (``"+0.1% Δ"``, ``"−0.5% Δ"`` etc.) and has
    ``showlegend=False`` so it does not pollute the inline legend.
    """
    import plotly.graph_objects as go

    t = get_theme(theme)
    a = np.asarray(a_1d, dtype=float).ravel()
    xs = np.asarray(x_plot, dtype=float).ravel()

    w01 = CONSTITUTIVE_BAND_FRAC_HIGH * (np.abs(a) + _DIVERGENCE_EPS)
    w05 = CONSTITUTIVE_BAND_FRAC_MOD * (np.abs(a) + _DIVERGENCE_EPS)
    w1 = CONSTITUTIVE_BAND_FRAC_LOW * (np.abs(a) + _DIVERGENCE_EPS)

    muted = t.palette.muted

    def _boundary(y_vals: np.ndarray, label: str) -> Any:
        return go.Scatter(
            x=xs,
            y=y_vals,
            mode="lines",
            line=dict(color=muted, dash="dot", width=1.0),
            name="",          # blank name keeps it out of inline legend
            showlegend=False,
            hovertemplate=f"{label}<extra></extra>",
            meta={"tier_label": label},
        )

    return [
        _boundary(a - w1, "\u22121% \u0394"),
        _boundary(a - w05, "\u22120.5% \u0394"),
        _boundary(a - w01, "\u22120.1% \u0394"),
        _boundary(a + w01, "+0.1% \u0394"),
        _boundary(a + w05, "+0.5% \u0394"),
        _boundary(a + w1, "+1% \u0394"),
    ]


def prepare_constitutive_model_implied_vs_x_embed(
    bundle: Dict[str, Any],
    residual_basename: Optional[str] = None,
    *,
    prefer_last_t: bool = True,
    theme: Any = MOJU_LIGHT,
) -> Optional[Dict[str, Any]]:
    """
    Build two line traces (Model vs x, Implied vs x) for the monitor: when a time axis is
    present, selects the **worst-divergence time slice** (the slice whose max |δ| over all
    spatial axes is largest, falling back to mean |δ| when that slice is degenerate) rather
    than always using the last time step.  For 2-D/3-D data the worst-y/z row pick uses the
    same max-with-mean-fallback rule on top of the time slice.  Steady-state data (no time
    axis) is unchanged.  Abscissa is mapped to ``[0, L]`` when possible.

    Returns ``None`` if nothing to plot.

    ``prefer_last_t`` controls whether the time-axis reduction is attempted at all; when
    ``False`` no time slicing is done (kept for call-site backward compatibility).
    """
    from moju.monitor.spatial_rnorm_panels import _reduce_spatial_array

    import plotly.graph_objects as go

    t = get_theme(theme)
    debug = _closure_debug(bundle)
    if not debug:
        return None
    bn = residual_basename or _auto_select_basename(bundle, debug)
    if bn is None or bn not in debug:
        return None
    entry = debug[bn]
    a_full = _coerce_to_numpy(entry.get("pred"))
    b_full = _coerce_to_numpy(entry.get("implied"))
    if a_full.size == 0 or b_full.size == 0:
        return None
    if a_full.shape != b_full.shape:
        try:
            a_full, b_full = np.broadcast_arrays(a_full, b_full)
        except ValueError:
            return None

    # Reshape flattened meshgrid data (n_t*n_x,) → (n_t, n_x) so the time axis is
    # explicit before we pick the worst-t slice.
    cs_last = ((bundle.get("log") or [{}])[-1] or {}).get("coord_snapshot") or {}
    gs = cs_last.get("grid_shape")
    if a_full.ndim == 1 and gs and int(gs[0]) * int(gs[1]) == a_full.size:
        nt, nx = int(gs[0]), int(gs[1])
        a_full = a_full.reshape(nt, nx)
        b_full = b_full.reshape(nt, nx)

    coords_pred = _closure_coords_for_reduce(bundle)
    label_model, label_implied = _user_constitutive_side_labels()
    y_qty = divergence_y_quantity_label(entry)
    hint_ax = bundle.get("spatial_coord_hint")
    note_parts: List[str] = []
    is_transient_slice = False
    t_value: Optional[float] = None
    time_slice_criterion: Optional[str] = None
    spatial_slice_criterion: Optional[str] = None

    # --- Worst-t slice: max |δ| with mean fallback when degenerate ---
    if prefer_last_t and coords_pred.get("t") is not None and a_full.ndim >= 2:
        t_vec = np.asarray(coords_pred["t"]).reshape(-1)
        nt = int(t_vec.shape[0])
        if a_full.shape[0] == nt:
            t_idx, time_slice_criterion = _select_worst_time_slice_index(a_full, b_full)
            a_full = np.asarray(a_full[t_idx], dtype=float)
            b_full = np.asarray(b_full[t_idx], dtype=float)
            t_value = float(t_vec[t_idx])
            is_transient_slice = True
            if time_slice_criterion == "mean":
                note_parts.append(f"t ≈ {t_value:.4g}, mean t slice (max degenerate)")
            else:
                note_parts.append(f"worst t ≈ {t_value:.4g}")

    # After time slicing pass prefer_last_t=False so _reduce_spatial_array does not try to
    # slice time again (the time dimension is already gone).
    try:
        a = np.asarray(_reduce_spatial_array(a_full, coords_pred, prefer_last_t=False), dtype=float)
        b = np.asarray(_reduce_spatial_array(b_full, coords_pred, prefer_last_t=False), dtype=float)
    except (TypeError, ValueError):
        return None

    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
        b = b[0]

    if a.ndim == 1 and b.ndim == 1:
        n = int(a.shape[0])
        xs, _ = infer_divergence_abscissa(bundle, n, hint_axis=str(hint_ax) if hint_ax else None)
        xs = np.asarray(xs, dtype=float).ravel()
        if xs.shape[0] != n:
            xs = np.arange(n, dtype=float)
        x_plot, _L, x_ax_title = _x_abscissa_0_to_L(xs, bundle)
        band_traces, _ = _build_dissonance_band_traces(x_plot, a, t)
        tier_lines = _build_dissonance_tier_lines(x_plot, a, t)
        y_range = _dissonance_y_range(a, b)
        line_traces = [
            go.Scatter(
                x=x_plot,
                y=a,
                mode="lines",
                line=dict(color=t.palette.line_primary, dash="dash", width=2.5),
                name=label_model,
                showlegend=True,
            ),
            go.Scatter(
                x=x_plot,
                y=b,
                mode="lines",
                line=dict(color=t.palette.title_color, width=2.5),
                name=label_implied,
                showlegend=True,
            ),
        ]
        max_delta_label = _max_delta_label(x_plot, a, b)
        subtitle = ", ".join(note_parts) if note_parts else ""
        return {
            "traces": tier_lines + band_traces + line_traces,
            "x_title": x_ax_title,
            "y_title": y_qty,
            "term_label": y_qty,
            "title": _constitutive_dissonance_title(
                is_transient=is_transient_slice,
                is_worst_slice=False,
                t_value=t_value,
                time_slice_criterion=time_slice_criterion,
                spatial_slice_criterion=spatial_slice_criterion,
            ),
            "subtitle": subtitle,
            "row_note": "1-D profile",
            "is_transient_slice": is_transient_slice,
            "is_worst_slice": False,
            "t_value": t_value,
            "time_slice_criterion": time_slice_criterion,
            "spatial_slice_criterion": spatial_slice_criterion,
            "y_range": y_range,
            "max_delta_label": max_delta_label,
        }

    if a.ndim == 2 and b.ndim == 2:
        div = _normalized_divergence(a, b)
        y_ix, spatial_slice_criterion = _select_worst_row_index(div)
        la = np.asarray(a[y_ix], dtype=float).ravel()
        lb = np.asarray(b[y_ix], dtype=float).ravel()
        nx = la.shape[0]
        coords = bundle.get("spatial") or {}
        cxx: Optional[np.ndarray] = None
        cds_raw = coords.get("coords") or {} if isinstance(coords, dict) else {}
        xv0 = cds_raw.get("x") if cds_raw else coords.get("x")
        if xv0 is not None:
            cxx = np.asarray(xv0, dtype=float).ravel()
        if cxx is None and coords_pred.get("x") is not None:
            px = np.asarray(coords_pred["x"], dtype=float).ravel()
            if px.size == nx:
                cxx = px
        if cxx is None or cxx.size != nx:
            cxx_infer, _ = infer_divergence_abscissa(bundle, nx, hint_axis=str(hint_ax) if hint_ax else None)
            cxx = np.asarray(cxx_infer, dtype=float).ravel()
            if cxx.size != nx:
                cxx = np.linspace(0.0, 1.0, nx, dtype=float)
        x_plot, _L, x_ax_title = _x_abscissa_0_to_L(cxx, bundle)
        if spatial_slice_criterion == "mean":
            note_parts.append("mean slice (max degenerate)")
        else:
            note_parts.append("worst slice")
        subtitle = ", ".join(note_parts)
        cy_val = None
        cy_src = cds_raw.get("y") if cds_raw else coords.get("y")
        if cy_src is not None:
            cyy = np.asarray(cy_src, dtype=float).ravel()
            if cyy.size > y_ix:
                cy_val = float(cyy[y_ix])
        band_traces_2d, _ = _build_dissonance_band_traces(x_plot, la, t)
        tier_lines_2d = _build_dissonance_tier_lines(x_plot, la, t)
        y_range_2d = _dissonance_y_range(la, lb)
        line_traces_2d = [
            go.Scatter(
                x=x_plot,
                y=la,
                mode="lines",
                line=dict(color=t.palette.line_primary, dash="dash", width=2.5),
                name=label_model,
                showlegend=True,
            ),
            go.Scatter(
                x=x_plot,
                y=lb,
                mode="lines",
                line=dict(color=t.palette.title_color, width=2.5),
                name=label_implied,
                showlegend=True,
            ),
        ]
        max_delta_label_2d = _max_delta_label(x_plot, la, lb)
        row_note = f"y* row index {y_ix}"
        if cy_val is not None and np.isfinite(cy_val):
            row_note = f"y* ≈ {cy_val:.4g} ({row_note})"
        return {
            "traces": tier_lines_2d + band_traces_2d + line_traces_2d,
            "x_title": x_ax_title,
            "y_title": y_qty,
            "term_label": y_qty,
            "title": _constitutive_dissonance_title(
                is_transient=is_transient_slice,
                is_worst_slice=True,
                t_value=t_value,
                time_slice_criterion=time_slice_criterion,
                spatial_slice_criterion=spatial_slice_criterion,
            ),
            "subtitle": subtitle,
            "row_note": row_note,
            "is_transient_slice": is_transient_slice,
            "is_worst_slice": True,
            "t_value": t_value,
            "time_slice_criterion": time_slice_criterion,
            "spatial_slice_criterion": spatial_slice_criterion,
            "y_range": y_range_2d,
            "max_delta_label": max_delta_label_2d,
        }

    return None


# ---------------------------------------------------------------------------
# Mode builders
# ---------------------------------------------------------------------------


class _SpatialDivPrep(NamedTuple):
    """Shared spatial divergence slice (1-D lines or 2-D heatmaps) for card + monitor."""

    basename: str
    label_model: str
    label_implied: str
    is_1d: bool
    a: np.ndarray
    b: np.ndarray
    div: np.ndarray
    xs_1d: Optional[np.ndarray]
    x_title: str
    y_title: str
    y_qty: str
    term_label: str
    cx: Optional[np.ndarray]
    cy: Optional[np.ndarray]
    abs_lim: float


def _coord_axis_title(axis: str) -> str:
    titles = {"x": "Position x", "y": "Position y", "z": "Position z", "t": "Time t"}
    return titles.get(str(axis), f"Position {axis}")


def _coord_vector_for_axis(bundle: Dict[str, Any], axis: str, expected_len: int) -> Optional[np.ndarray]:
    """Best-effort coordinate vector for a named axis from spatial metadata or log coord snapshots."""
    spatial_any = bundle.get("spatial") or {}
    if isinstance(spatial_any, dict):
        coords = spatial_any.get("coords") or {}
        v = coords.get(axis) if isinstance(coords, dict) else None
        if v is None:
            v = spatial_any.get(axis)
        if v is not None:
            arr = np.asarray(v, dtype=float).ravel()
            if arr.shape[0] == expected_len:
                return arr
    log = bundle.get("log") or []
    if log and isinstance(log[-1], dict):
        cs = log[-1].get("coord_snapshot") or {}
        if isinstance(cs, dict):
            # Prefer unique grid axis vector (from meshgrid detection) over raw flattened coords.
            grid_key = axis + "_grid"
            v_grid = cs.get(grid_key)
            if v_grid is not None:
                arr = np.asarray(v_grid, dtype=float).ravel()
                if arr.shape[0] == expected_len:
                    return arr
            if cs.get(axis) is not None:
                arr = np.asarray(cs.get(axis), dtype=float).ravel()
                if arr.shape[0] == expected_len:
                    return arr
    return None


def _infer_2d_divergence_axes(bundle: Dict[str, Any], shape: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, str]:
    """Infer heatmap x/y coordinate vectors and labels from the reduced array shape."""
    n_rows, n_cols = int(shape[0]), int(shape[1])
    hint = str(bundle.get("spatial_coord_hint") or "").strip().lower()

    x_order: List[str] = []
    if hint in ("x", "y", "z", "t"):
        x_order.append(hint)
    for axis in ("x", "y", "z", "t"):
        if axis not in x_order:
            x_order.append(axis)

    x_axis = "x"
    cx: Optional[np.ndarray] = None
    for axis in x_order:
        cx = _coord_vector_for_axis(bundle, axis, n_cols)
        if cx is not None:
            x_axis = axis
            break

    y_order = ["y", "z", "t", "x"]
    if x_axis in y_order:
        y_order.remove(x_axis)
        y_order.append(x_axis)
    cy: Optional[np.ndarray] = None
    y_axis = "y"
    for axis in y_order:
        cy = _coord_vector_for_axis(bundle, axis, n_rows)
        if cy is not None:
            y_axis = axis
            break

    return cx, cy, _coord_axis_title(x_axis), _coord_axis_title(y_axis)


def _prepare_spatial_divergence(
    bundle: Dict[str, Any],
    residual_basename: Optional[str] = None,
) -> Union[_SpatialDivPrep, Tuple[str, str]]:
    """
    Resolve closure sidecars and return arrays for model, implied, and normalised divergence,
    or ``(title, message)`` for an empty card.
    """
    debug = _closure_debug(bundle)
    if not debug:
        return ("Constitutive divergence (spatial)", "No closure_debug sidecar available")
    bn = residual_basename or _auto_select_basename(bundle, debug)
    if bn is None or bn not in debug:
        return ("Constitutive divergence (spatial)", "No constitutive basename to render")
    entry = debug[bn]
    a_full = _coerce_to_numpy(entry.get("pred"))
    b_full = _coerce_to_numpy(entry.get("implied"))
    if a_full.size == 0 or b_full.size == 0:
        return (bn, "No spatial data for divergence")
    if a_full.shape != b_full.shape:
        try:
            a_full, b_full = np.broadcast_arrays(a_full, b_full)
        except ValueError:
            return (bn, "Model and Implied shapes differ; cannot render side-by-side heatmaps")

    label_model, label_implied = _user_constitutive_side_labels()
    hint_ax = bundle.get("spatial_coord_hint")

    a = a_full
    b = b_full
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
        b = b[0]
    y_qty = divergence_y_quantity_label(entry)
    term_label = constitutive_term_label(entry, bn)

    # Reshape flattened meshgrid data (n_t*n_x,) → (n_t, n_x) so the heatmap path is taken.
    if a.ndim == 1:
        cs_last = ((bundle.get("log") or [{}])[-1] or {}).get("coord_snapshot") or {}
        gs = cs_last.get("grid_shape")
        if gs and int(gs[0]) * int(gs[1]) == a.size:
            nt, nx = int(gs[0]), int(gs[1])
            a = a.reshape(nt, nx)
            b = b.reshape(nt, nx)
    if a.ndim == 1:
        div = _normalized_divergence(a, b)
        xs, x_title = infer_divergence_abscissa(bundle, int(a.shape[0]), hint_axis=str(hint_ax) if hint_ax else None)
        return _SpatialDivPrep(
            basename=bn,
            label_model=label_model,
            label_implied=label_implied,
            is_1d=True,
            a=a,
            b=b,
            div=div,
            xs_1d=np.asarray(xs, dtype=float),
            x_title=x_title,
            y_title="Normalised divergence",
            y_qty=y_qty,
            term_label=term_label,
            cx=None,
            cy=None,
            abs_lim=1.0,
        )

    if a.ndim != 2:
        return (bn, f"Cannot render divergence for ndim={a.ndim} arrays")

    div = _normalized_divergence(a, b)
    abs_lim = float(np.nanpercentile(np.abs(div), 95)) if div.size else 1.0
    if not np.isfinite(abs_lim) or abs_lim == 0.0:
        abs_lim = 1.0

    cx, cy, x_title, y_title = _infer_2d_divergence_axes(bundle, a.shape)

    return _SpatialDivPrep(
        basename=bn,
        label_model=label_model,
        label_implied=label_implied,
        is_1d=False,
        a=a,
        b=b,
        div=div,
        xs_1d=None,
        x_title=x_title,
        y_title=y_title,
        y_qty=y_qty,
        term_label=term_label,
        cx=cx,
        cy=cy,
        abs_lim=abs_lim,
    )


def build_spatial_normalized_divergence_figure(
    bundle: Dict[str, Any],
    residual_basename: Optional[str] = None,
    *,
    theme: Any = MOJU_LIGHT,
    height: Optional[int] = None,
    title: Optional[str] = None,
) -> Any:
    """
    Single-plot normalised constitutive divergence (heatmap or line): no model/implied subpanels.
    Intended for monitor embedding on a wide ``colspan`` cell.
    """
    import plotly.graph_objects as go

    t = get_theme(theme)
    pre = _prepare_spatial_divergence(bundle, residual_basename)
    if not isinstance(pre, _SpatialDivPrep):
        return _empty_card(pre[0], pre[1], theme)

    if pre.is_1d:
        xs = pre.xs_1d
        if xs is None:
            return _empty_card(pre.basename, "Missing abscissa for divergence", theme)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=pre.div,
                mode="lines",
                line=dict(color=t.palette.adm_low, width=2),
                name="Δ",
                showlegend=False,
            )
        )
        fig.update_xaxes(title_text=pre.x_title, **themed_axis_style(theme, show_grid=False, zero_line=False))
        fig.update_yaxes(title_text="Normalised divergence", **themed_axis_style(theme))
        fig.update_layout(showlegend=False)
        return apply_theme(
            fig,
            theme,
            title=title or constitutive_divergence_title(pre.term_label),
            height=height or t.layout.card_height,
        )

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=pre.div,
            x=pre.cx,
            y=pre.cy,
            colorscale=t.colorscales.divergence,
            zmid=0,
            zmin=-pre.abs_lim,
            zmax=pre.abs_lim,
            colorbar=themed_colorbar(theme, title="Normalized delta"),
            hovertemplate="x=%{x:.3g}<br>y=%{y:.3g}<br>Δ=%{z:.3g}<extra></extra>",
            showscale=True,
        )
    )
    fig.update_xaxes(title_text=pre.x_title, **themed_axis_style(theme, show_grid=False, zero_line=False))
    fig.update_yaxes(title_text=pre.y_title, **themed_axis_style(theme, show_grid=False, zero_line=False))
    return apply_theme(
        fig,
        theme,
        title=title or constitutive_divergence_title(pre.term_label),
        height=height or t.layout.card_height,
    )


def build_spatial_divergence_panel(
    bundle: Dict[str, Any],
    residual_basename: Optional[str] = None,
    *,
    theme: Any = MOJU_LIGHT,
    height: Optional[int] = None,
    title: Optional[str] = None,
) -> Any:
    """Three-panel spatial heatmap: model, implied, normalised divergence."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    t = get_theme(theme)
    pre = _prepare_spatial_divergence(bundle, residual_basename)
    if not isinstance(pre, _SpatialDivPrep):
        return _empty_card(pre[0], pre[1], theme)

    label_model, label_implied = pre.label_model, pre.label_implied
    bn = pre.basename

    if pre.is_1d:
        xs = pre.xs_1d
        if xs is None:
            return _empty_card(pre.basename, "Missing abscissa for divergence", theme)
        fig = make_subplots(
            rows=1,
            cols=3,
            shared_yaxes=False,
            subplot_titles=(label_model, label_implied, "Normalised divergence"),
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=pre.a,
                mode="lines",
                line=dict(color=t.palette.line_primary),
                name=label_model,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=pre.b,
                mode="lines",
                line=dict(color=t.palette.cat_constitutive),
                name=label_implied,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=pre.div,
                mode="lines",
                line=dict(color=t.palette.adm_low),
                name="Δ",
            ),
            row=1,
            col=3,
        )
        for col in (1, 2, 3):
            fig.update_xaxes(row=1, col=col, title_text=pre.x_title, **themed_axis_style(theme, show_grid=False, zero_line=False))
            y_lab = "Normalised divergence" if col == 3 else pre.y_qty
            fig.update_yaxes(row=1, col=col, title_text=y_lab, **themed_axis_style(theme))
        fig.update_layout(showlegend=False)
        return apply_theme(
            fig,
            theme,
            title=title or constitutive_divergence_title(pre.term_label),
            height=height or t.layout.card_height,
        )

    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        subplot_titles=(label_model, label_implied, "Normalised divergence"),
    )
    fig.add_trace(
        go.Heatmap(
            z=pre.a,
            x=pre.cx,
            y=pre.cy,
            colorscale=t.colorscales.sequential,
            colorbar=themed_colorbar(theme, title=label_model),
            hovertemplate="x=%{x:.3g}<br>y=%{y:.3g}<br>val=%{z:.3g}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=pre.b,
            x=pre.cx,
            y=pre.cy,
            colorscale=t.colorscales.sequential_alt,
            colorbar=themed_colorbar(theme, title=label_implied),
            hovertemplate="x=%{x:.3g}<br>y=%{y:.3g}<br>val=%{z:.3g}<extra></extra>",
            showscale=True,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Heatmap(
            z=pre.div,
            x=pre.cx,
            y=pre.cy,
            colorscale=t.colorscales.divergence,
            zmid=0,
            zmin=-pre.abs_lim,
            zmax=pre.abs_lim,
            colorbar=themed_colorbar(theme, title="Normalized delta"),
            hovertemplate="x=%{x:.3g}<br>y=%{y:.3g}<br>Δ=%{z:.3g}<extra></extra>",
            showscale=True,
        ),
        row=1,
        col=3,
    )
    for col in (1, 2, 3):
        fig.update_xaxes(row=1, col=col, title_text=pre.x_title, **themed_axis_style(theme, show_grid=False, zero_line=False))
        fig.update_yaxes(row=1, col=col, title_text=pre.y_title, **themed_axis_style(theme, show_grid=False, zero_line=False))
    return apply_theme(
        fig,
        theme,
        title=title or constitutive_divergence_title(pre.term_label),
        height=height or t.layout.card_height,
    )


def build_scatter_divergence_panel(
    bundle: Dict[str, Any],
    residual_basename: Optional[str] = None,
    *,
    theme: Any = MOJU_LIGHT,
    height: Optional[int] = None,
    title: Optional[str] = None,
) -> Any:
    """Scatter of model vs implied; y=x identity line; colored by |Δ|."""
    import plotly.graph_objects as go

    t = get_theme(theme)
    debug = _closure_debug(bundle)
    if not debug:
        return _empty_card(title or "Constitutive scatter", "No closure_debug sidecar available", theme)
    bn = residual_basename or _auto_select_basename(bundle, debug)
    if bn is None or bn not in debug:
        return _empty_card(title or "Constitutive scatter", "No constitutive basename to render", theme)
    entry = debug[bn]
    a, b, lab_a, lab_b = _sides_for_divergence(entry)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return _empty_card(title or bn, "Cannot align model / implied for scatter", theme)
    af = _flatten(a)
    bf = _flatten(b)
    mask = np.isfinite(af) & np.isfinite(bf)
    af = af[mask]
    bf = bf[mask]
    if af.size == 0:
        return _empty_card(title or bn, "No finite points", theme)
    div = _normalized_divergence(af, bf)
    abs_div = np.abs(div)

    # Identity line spanning the data extent
    lo = float(np.nanmin([np.nanmin(af), np.nanmin(bf)]))
    hi = float(np.nanmax([np.nanmax(af), np.nanmax(bf)]))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = 0.0, 1.0
    if lo == hi:
        hi = lo + 1.0
    pad = 0.05 * (hi - lo)

    # Tolerance band statistics
    pct_01 = float(np.mean(abs_div <= CONSTITUTIVE_BAND_FRAC_HIGH) * 100.0)
    pct_05 = float(np.mean(abs_div <= CONSTITUTIVE_BAND_FRAC_MOD) * 100.0)
    pct_1 = float(np.mean(abs_div <= CONSTITUTIVE_BAND_FRAC_LOW) * 100.0)
    cx, cy, cz, ct = _coords_for_points(bundle, af.size)
    custom = np.column_stack([
        cx if cx is not None else np.full(af.size, np.nan),
        cy if cy is not None else np.full(af.size, np.nan),
        cz if cz is not None else np.full(af.size, np.nan),
        ct if ct is not None else np.full(af.size, np.nan),
        div,
    ])
    hover = (
        f"<b>{lab_a}</b>=%{{x:.4g}}<br>"
        f"<b>{lab_b}</b>=%{{y:.4g}}<br>"
        "Δ_norm=%{customdata[4]:.3g}<br>"
        "x=%{customdata[0]:.3g} y=%{customdata[1]:.3g}"
        " z=%{customdata[2]:.3g} t=%{customdata[3]:.3g}"
        "<extra></extra>"
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[lo - pad, hi + pad],
            y=[lo - pad, hi + pad],
            mode="lines",
            line=dict(color=t.palette.muted, dash="dash", width=1.4),
            hoverinfo="skip",
            showlegend=False,
            name="y = x",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=af,
            y=bf,
            mode="markers",
            marker=dict(
                size=6,
                color=abs_div,
                colorscale=t.colorscales.divergence,
                cmin=0.0,
                cmax=float(np.nanpercentile(abs_div, 98)) if abs_div.size else 1.0,
                colorbar=themed_colorbar(theme, title="|Δ_norm|"),
                line=dict(color=t.palette.bar_line, width=0.4),
            ),
            customdata=custom,
            hovertemplate=hover,
            showlegend=False,
        )
    )
    annot = (
        f"Within ±0.1%: {pct_01:.1f}%<br>"
        f"Within ±0.5%: {pct_05:.1f}%<br>"
        f"Within ±1%: {pct_1:.1f}%"
    )
    fig.add_annotation(
        text=annot,
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        showarrow=False,
        align="left",
        font=t.font_dict(size=11, color=t.palette.muted),
        bgcolor=t.palette.summary_bg,
        bordercolor=t.palette.summary_border,
        borderwidth=1,
        borderpad=6,
    )
    fig.update_xaxes(title_text=lab_a, range=[lo - pad, hi + pad], **themed_axis_style(theme))
    fig.update_yaxes(title_text=lab_b, range=[lo - pad, hi + pad], **themed_axis_style(theme))
    return apply_theme(fig, theme, title=title or f"{pretty_residual_key(bn)} (scatter)", height=height or t.layout.card_height)


def build_distribution_divergence_panel(
    bundle: Dict[str, Any],
    residual_basename: Optional[str] = None,
    *,
    theme: Any = MOJU_LIGHT,
    height: Optional[int] = None,
    title: Optional[str] = None,
) -> Any:
    """Histogram of normalised divergence with admissibility threshold bands."""
    import plotly.graph_objects as go

    t = get_theme(theme)
    debug = _closure_debug(bundle)
    if not debug:
        return _empty_card(title or "Divergence distribution", "No closure_debug sidecar available", theme)
    bn = residual_basename or _auto_select_basename(bundle, debug)
    if bn is None or bn not in debug:
        return _empty_card(title or "Divergence distribution", "No constitutive basename to render", theme)
    entry = debug[bn]
    a, b, _, _ = _sides_for_divergence(entry)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return _empty_card(title or bn, "Cannot align model / implied for distribution", theme)
    div = _normalized_divergence(_flatten(a), _flatten(b))
    div = div[np.isfinite(div)]
    if div.size == 0:
        return _empty_card(title or bn, "No finite divergence values", theme)
    abs_lim = float(np.nanpercentile(np.abs(div), 99)) if div.size else 1.0
    if not np.isfinite(abs_lim) or abs_lim == 0.0:
        abs_lim = 1.0
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=div,
            nbinsx=72,
            marker=dict(color=t.palette.cat_constitutive, line=dict(color=t.palette.bar_line, width=0.4)),
            opacity=0.92,
            name="Δ_norm",
            hovertemplate="bin %{x:.3g}<br>count: %{y}<extra></extra>",
        )
    )
    # Threshold bands aligned with admissibility tiers (±0.1% / ±0.5% / ±1%)
    fh = CONSTITUTIVE_BAND_FRAC_HIGH
    fm = CONSTITUTIVE_BAND_FRAC_MOD
    fl = CONSTITUTIVE_BAND_FRAC_LOW
    band_specs = [
        (-fh, fh, t.palette.adm_high, "±0.1%"),
        (-fm, -fh, t.palette.adm_med, ""),
        (fh, fm, t.palette.adm_med, "±0.5%"),
        (-fl, -fm, t.palette.adm_low, ""),
        (fm, fl, t.palette.adm_low, "±1%"),
    ]
    for x0, x1, color, _label in band_specs:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.08, line_width=0)
    fig.add_vline(x=0.0, line=dict(color=t.palette.muted, width=1, dash="dot"))

    pct_band = lambda thresh: float(np.mean(np.abs(div) <= thresh) * 100.0)
    annot = (
        f"|Δ| ≤ 0.1%: {pct_band(fh):.1f}%<br>"
        f"|Δ| ≤ 0.5%: {pct_band(fm):.1f}%<br>"
        f"|Δ| ≤ 1%: {pct_band(fl):.1f}%"
    )
    fig.add_annotation(
        text=annot,
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        showarrow=False,
        align="left",
        font=t.font_dict(size=11, color=t.palette.muted),
        bgcolor=t.palette.summary_bg,
        bordercolor=t.palette.summary_border,
        borderwidth=1,
        borderpad=6,
    )
    rng = max(abs_lim, 0.12)
    fig.update_xaxes(title_text="(Model − Implied) / scale", range=[-rng, rng], **themed_axis_style(theme))
    fig.update_yaxes(title_text="count", **themed_axis_style(theme))
    fig.update_layout(showlegend=False)
    return apply_theme(fig, theme, title=title or f"{pretty_residual_key(bn)} (distribution)", height=height or t.layout.card_height)


def build_hotspot_divergence_panel(
    bundle: Dict[str, Any],
    residual_basename: Optional[str] = None,
    *,
    theme: Any = MOJU_LIGHT,
    height: Optional[int] = None,
    title: Optional[str] = None,
    top_k: int = 32,
) -> Any:
    """Top-``k`` worst points spatially over collocation coordinates (2-D or 3-D)."""
    import plotly.graph_objects as go

    t = get_theme(theme)
    debug = _closure_debug(bundle)
    if not debug:
        return _empty_card(title or "Divergence hotspots", "No closure_debug sidecar available", theme)
    bn = residual_basename or _auto_select_basename(bundle, debug)
    if bn is None or bn not in debug:
        return _empty_card(title or "Divergence hotspots", "No constitutive basename to render", theme)
    entry = debug[bn]
    a, b, lab_a, lab_b = _sides_for_divergence(entry)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return _empty_card(title or bn, "Cannot align model / implied for hotspots", theme)
    af = _flatten(a)
    bf = _flatten(b)
    div = _normalized_divergence(af, bf)
    abs_div = np.abs(div)
    mask = np.isfinite(abs_div)
    if not mask.any():
        return _empty_card(title or bn, "No finite divergence", theme)
    af = af[mask]
    bf = bf[mask]
    div = div[mask]
    abs_div = abs_div[mask]
    cx, cy, cz, ct = _coords_for_points(bundle, af.size)

    order = np.argsort(-abs_div)[:top_k]
    af = af[order]
    bf = bf[order]
    div = div[order]
    abs_div = abs_div[order]
    cx_ = cx[order] if cx is not None else None
    cy_ = cy[order] if cy is not None else None
    cz_ = cz[order] if cz is not None else None
    ct_ = ct[order] if ct is not None else None

    size_scaled = 8 + 24 * (abs_div / (abs_div.max() + _DIVERGENCE_EPS))
    abs_lim = float(np.nanpercentile(abs_div, 95)) if abs_div.size else 1.0
    if not np.isfinite(abs_lim) or abs_lim == 0.0:
        abs_lim = 1.0

    has_2d = cx_ is not None and cy_ is not None
    has_3d = has_2d and cz_ is not None

    customdata = np.column_stack([
        cx_ if cx_ is not None else np.full(af.size, np.nan),
        cy_ if cy_ is not None else np.full(af.size, np.nan),
        cz_ if cz_ is not None else np.full(af.size, np.nan),
        ct_ if ct_ is not None else np.full(af.size, np.nan),
        af,
        bf,
        div,
    ])
    hover = (
        "x=%{customdata[0]:.3g} y=%{customdata[1]:.3g}"
        " z=%{customdata[2]:.3g} t=%{customdata[3]:.3g}<br>"
        f"{lab_a}=%{{customdata[4]:.3g}}<br>"
        f"{lab_b}=%{{customdata[5]:.3g}}<br>"
        "Δ_norm=%{customdata[6]:.3g}"
        "<extra></extra>"
    )

    fig = go.Figure()
    if has_3d:
        fig.add_trace(
            go.Scatter3d(
                x=cx_,
                y=cy_,
                z=cz_,
                mode="markers",
                marker=dict(
                    size=size_scaled,
                    color=div,
                    cmin=-abs_lim,
                    cmax=abs_lim,
                    colorscale=t.colorscales.divergence,
                    colorbar=themed_colorbar(theme, title="Δ_norm"),
                    line=dict(color=t.palette.bar_line, width=0.3),
                ),
                customdata=customdata,
                hovertemplate=hover,
                showlegend=False,
            )
        )
        fig.update_layout(
            scene=dict(
                xaxis=dict(title="Position x", title_font=t.section_title_font_dict(), tickfont=t.tick_font_dict()),
                yaxis=dict(title="Position y", title_font=t.section_title_font_dict(), tickfont=t.tick_font_dict()),
                zaxis=dict(title="Position z", title_font=t.section_title_font_dict(), tickfont=t.tick_font_dict()),
                aspectmode="data",
            ),
        )
    elif has_2d:
        fig.add_trace(
            go.Scatter(
                x=cx_,
                y=cy_,
                mode="markers",
                marker=dict(
                    size=size_scaled,
                    color=div,
                    cmin=-abs_lim,
                    cmax=abs_lim,
                    colorscale=t.colorscales.divergence,
                    colorbar=themed_colorbar(theme, title="Δ_norm"),
                    line=dict(color=t.palette.bar_line, width=0.4),
                ),
                customdata=customdata,
                hovertemplate=hover,
                showlegend=False,
            )
        )
        fig.update_xaxes(title_text="Position x", **themed_axis_style(theme))
        fig.update_yaxes(title_text="Position y", **themed_axis_style(theme))
    else:
        hint_ax = bundle.get("spatial_coord_hint")
        bx, xt = infer_divergence_abscissa(bundle, af.size, hint_axis=str(hint_ax) if hint_ax else None)
        fig.add_trace(
            go.Bar(
                x=bx,
                y=abs_div,
                marker=dict(color=div, cmin=-abs_lim, cmax=abs_lim, colorscale=t.colorscales.divergence, colorbar=themed_colorbar(theme, title="Δ_norm")),
                customdata=customdata,
                hovertemplate=hover,
                showlegend=False,
            )
        )
        fig.update_xaxes(title_text=xt, **themed_axis_style(theme))
        fig.update_yaxes(title_text="|Δ_norm|", **themed_axis_style(theme))
    return apply_theme(fig, theme, title=title or f"{pretty_residual_key(bn)} (top-{top_k} hotspots)", height=height or t.layout.card_height)


# ---------------------------------------------------------------------------
# Composite + dispatcher
# ---------------------------------------------------------------------------


_MODES = ("spatial", "scatter", "distribution", "hotspot")


def build_constitutive_divergence_card(
    bundle: Dict[str, Any],
    *,
    residual_basename: Optional[str] = None,
    mode: str = "spatial",
    theme: Any = MOJU_LIGHT,
    height: Optional[int] = None,
    title: Optional[str] = None,
    **mode_kwargs: Any,
) -> Any:
    """
    Single-mode constitutive divergence card.

    Parameters
    ----------
    bundle:
        Visualisation bundle (must contain ``closure_debug``).
    residual_basename:
        Specific constitutive ``residual_basename`` (e.g.
        ``"thermal_diffusivity/law_fourier_conduction"``).  When ``None``,
        auto-selects the basename with the worst final-step R_norm.
    mode:
        One of ``"spatial"``, ``"scatter"``, ``"distribution"``, ``"hotspot"``.
    """
    if mode not in _MODES:
        raise ValueError(f"Unknown divergence mode {mode!r}; expected one of {_MODES}")
    if mode == "spatial":
        return build_spatial_divergence_panel(
            bundle, residual_basename, theme=theme, height=height, title=title, **mode_kwargs
        )
    if mode == "scatter":
        return build_scatter_divergence_panel(
            bundle, residual_basename, theme=theme, height=height, title=title, **mode_kwargs
        )
    if mode == "distribution":
        return build_distribution_divergence_panel(
            bundle, residual_basename, theme=theme, height=height, title=title, **mode_kwargs
        )
    return build_hotspot_divergence_panel(
        bundle, residual_basename, theme=theme, height=height, title=title, **mode_kwargs
    )


def build_constitutive_divergence_dashboard(
    bundle: Dict[str, Any],
    *,
    residual_basename: Optional[str] = None,
    theme: Any = MOJU_LIGHT,
    height: Optional[int] = None,
    title: Optional[str] = None,
) -> Any:
    """
    Composite 2x2 constitutive divergence dashboard.

    Top row: spatial divergence | scatter (pred vs implied).
    Bottom row: distribution (histogram) | hotspot map.

    When ``residual_basename`` is ``None``, the worst-performing constitutive
    basename (largest final-step R_norm) is selected automatically so the
    operator sees the most informative divergence first.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    t = get_theme(theme)
    debug = _closure_debug(bundle)
    if not debug:
        return _empty_card(title or "Constitutive divergence", "No closure_debug sidecar available", theme)
    bn = residual_basename or _auto_select_basename(bundle, debug)
    if bn is None or bn not in debug:
        return _empty_card(title or "Constitutive divergence", "No constitutive basename to render", theme)

    sub_specs = [[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]]
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=sub_specs,
        horizontal_spacing=0.10,
        vertical_spacing=0.14,
        subplot_titles=("Spatial divergence", "Model vs Implied", "Divergence distribution", "Top hotspots"),
    )

    panels = [
        build_spatial_divergence_panel(bundle, bn, theme=theme),
        build_scatter_divergence_panel(bundle, bn, theme=theme),
        build_distribution_divergence_panel(bundle, bn, theme=theme),
        build_hotspot_divergence_panel(bundle, bn, theme=theme),
    ]
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for panel, (row, col) in zip(panels, positions):
        for trace in panel.data:
            # Subplots only support a subset of trace types; if hotspot uses
            # Scatter3d we keep the existing layout but skip nesting
            if trace.type in ("scatter3d", "volume", "surface"):
                continue
            # Avoid duplicating colorbars across the composite
            try:
                if hasattr(trace, "marker") and getattr(trace.marker, "colorbar", None) is not None:
                    trace.marker.colorbar = None  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
            try:
                if hasattr(trace, "colorbar") and getattr(trace, "colorbar", None) is not None:
                    trace.colorbar = None  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
            try:
                if hasattr(trace, "showscale"):
                    trace.showscale = False  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
            fig.add_trace(trace, row=row, col=col)

    for row in (1, 2):
        for col in (1, 2):
            fig.update_xaxes(row=row, col=col, **themed_axis_style(theme, show_grid=False, zero_line=False))
            fig.update_yaxes(row=row, col=col, **themed_axis_style(theme, show_grid=False, zero_line=False))
    fig.update_layout(showlegend=False)
    target_height = height or int(t.layout.card_height * 1.9)
    return apply_theme(fig, theme, title=title or f"Constitutive divergence — {pretty_residual_key(bn)}", height=target_height)


def list_constitutive_basenames(bundle: Dict[str, Any]) -> List[str]:
    """Return all available constitutive basenames in the bundle, sorted by worst R_norm first."""
    debug = _closure_debug(bundle)
    if not debug:
        return []
    finals = _final_rnorm_by_basename(bundle)
    keys = list(debug.keys())
    keys.sort(key=lambda k: -finals.get(k, 0.0))
    return keys


__all__ = [
    "USER_CONSTIT_MODEL",
    "USER_CONSTIT_IMPLIED",
    "build_constitutive_divergence_card",
    "build_constitutive_divergence_dashboard",
    "build_distribution_divergence_panel",
    "build_hotspot_divergence_panel",
    "build_scatter_divergence_panel",
    "build_spatial_divergence_panel",
    "build_spatial_normalized_divergence_figure",
    "constitutive_divergence_title",
    "constitutive_divergence_title_for_bundle",
    "constitutive_term_label",
    "divergence_y_quantity_label",
    "infer_divergence_abscissa",
    "list_constitutive_basenames",
    "prepare_constitutive_model_implied_vs_x_embed",
    "primary_closure_debug_field_length",
    "worst_div_max_abs_row_index",
    "worst_div_mean_abs_row_index",
]
