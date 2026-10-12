"""
Local under-resolution sensor (diagnostic only — never changes tiers).

Compares a fine stencil estimate to a coarser one (lower order or every-other-point)
and reports relative disagreement per region. Emit warnings when derivatives look
unreliable relative to local field energy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import jax.numpy as jnp

from moju.monitor.fd_stencils import diff1_uniform, validate_fd_order
from moju.monitor.path_b_derivatives import _uniform_1d_spacing


DEFAULT_RESOLUTION_RATIO = 0.5

# Fraction of 1D spectral power in the Nyquist bin above which we warn about
# checkerboard / grid-scale energy. Diagnostic only — never changes tiers.
DEFAULT_GRID_SCALE_ENERGY_THRESHOLD = 0.05


def _relative_disagreement(a: Any, b: Any, field: Any, *, eps: float = 1e-12) -> jnp.ndarray:
    """Pointwise |a-b| / (|field| * k + eps) style relative to local amplitude of a,b."""
    aa = jnp.asarray(a)
    bb = jnp.asarray(b)
    scale = jnp.maximum(jnp.abs(aa), jnp.abs(bb))
    denom = scale + float(eps)
    return jnp.abs(aa - bb) / denom


def sensor_order_disagreement_1d(
    values: Any,
    h: float,
    *,
    order_hi: int = 4,
    order_lo: int = 2,
    axis: int = -1,
) -> jnp.ndarray:
    """Relative disagreement between two FD orders for the first derivative."""
    order_hi = validate_fd_order(order_hi)
    order_lo = validate_fd_order(order_lo)
    a = jnp.asarray(values)
    d_hi = diff1_uniform(a, float(h), order=order_hi, axis=axis)
    d_lo = diff1_uniform(a, float(h), order=order_lo, axis=axis)
    return _relative_disagreement(d_hi, d_lo, a)


def sensor_shortest_wavelength_energy_1d(
    values: Any,
    *,
    axis: int = -1,
) -> float:
    """
    Max Nyquist-bin spectral energy fraction among lines along ``axis``.

    Averaging across lines first would cancel checkerboards; each line is
    demeaned and FFTed independently, then the max Nyquist fraction is returned.
    Diagnostic scalar in ``[0, 1]``.
    """
    a = jnp.moveaxis(jnp.asarray(values, dtype=float), axis, -1)
    lines = a.reshape(-1, a.shape[-1])
    n = int(lines.shape[-1])
    if n < 4:
        return float("nan")
    demeaned = lines - jnp.mean(lines, axis=-1, keepdims=True)
    spec = jnp.fft.rfft(demeaned, axis=-1)
    power = jnp.abs(spec) ** 2
    totals = jnp.sum(power, axis=-1) + 1e-30
    fracs = power[:, -1] / totals
    return float(jnp.max(fracs))


def sensor_grid_scale_energy_nd(
    values: Any,
    *,
    axes: Optional[Sequence[int]] = None,
) -> float:
    """
    Max Nyquist-bin energy fraction over the given axes (2D: max of row/col means).

    Used to flag checkerboard / grid-scale artifacts. Diagnostic only.
    """
    a = jnp.asarray(values, dtype=float)
    if axes is None:
        axes = tuple(range(a.ndim))
    fracs: List[float] = []
    for ax in axes:
        fracs.append(sensor_shortest_wavelength_energy_1d(a, axis=int(ax)))
    finite = [f for f in fracs if f == f]  # not NaN
    if not finite:
        return float("nan")
    return float(max(finite))


def build_resolution_sensor_report(
    state: Mapping[str, Any],
    *,
    field_keys: Optional[Sequence[str]] = None,
    key_x: str = "x",
    key_y: str = "y",
    order_hi: int = 4,
    order_lo: int = 2,
    threshold: float = DEFAULT_RESOLUTION_RATIO,
    grid_scale_threshold: float = DEFAULT_GRID_SCALE_ENERGY_THRESHOLD,
    region_reduce: str = "max",
) -> Dict[str, Any]:
    """
    Per-field resolution sensor summary (diagnostic).

    For each scalar / first component of a vector field with a recoverable 1D/2D
    rectilinear grid, compare FD order ``order_hi`` vs ``order_lo`` and warn when
    the reduced disagreement exceeds ``threshold``.

    Also warns when Nyquist-bin spectral energy exceeds ``grid_scale_threshold``
    (checkerboard / grid-scale artifacts). Never changes admissibility tiers.
    """
    keys = list(field_keys) if field_keys is not None else [
        k
        for k in ("T", "phi", "u", "p")
        if k in state and state.get(k) is not None
    ]
    per_field: Dict[str, Any] = {}
    warnings: List[str] = []
    x = state.get(key_x)
    y = state.get(key_y)

    for key in keys:
        val = state.get(key)
        if val is None:
            continue
        arr = jnp.asarray(val)
        # Use first vector component for sensor if needed
        if arr.ndim >= 1 and key == "u" and arr.shape[-1] in (2, 3) and arr.ndim >= 2:
            sample = arr[..., 0]
        else:
            sample = arr
        entry: Dict[str, Any] = {"shape": list(sample.shape)}
        disagree = None
        try:
            if sample.ndim == 1 and x is not None:
                xa = jnp.reshape(jnp.asarray(x), (-1,))
                h = _uniform_1d_spacing(xa)
                if h is not None and int(xa.shape[0]) == int(sample.shape[0]):
                    disagree = sensor_order_disagreement_1d(
                        sample, float(h), order_hi=order_hi, order_lo=order_lo, axis=0
                    )
            elif sample.ndim == 2 and x is not None and y is not None:
                xa = jnp.asarray(x)
                ya = jnp.asarray(y)
                if xa.ndim == 2 and ya.ndim == 2:
                    x1 = xa[:, 0]
                    y1 = ya[0, :]
                elif xa.ndim == 1 and ya.ndim == 1:
                    x1, y1 = xa, ya
                else:
                    x1 = y1 = None
                if x1 is not None and y1 is not None:
                    hx = _uniform_1d_spacing(x1)
                    hy = _uniform_1d_spacing(y1)
                    if hx is not None and hy is not None:
                        dx = sensor_order_disagreement_1d(
                            sample, float(hx), order_hi=order_hi, order_lo=order_lo, axis=0
                        )
                        dy = sensor_order_disagreement_1d(
                            sample, float(hy), order_hi=order_hi, order_lo=order_lo, axis=1
                        )
                        disagree = jnp.maximum(dx, dy)
        except Exception as err:  # noqa: BLE001
            entry["error"] = str(err)
            per_field[key] = entry
            continue

        # Grid-scale / checkerboard energy (independent of order-disagreement path)
        try:
            if sample.ndim >= 1:
                gse = sensor_grid_scale_energy_nd(sample)
                entry["grid_scale_energy"] = gse
                entry["grid_scale_threshold"] = float(grid_scale_threshold)
                entry["grid_scale_flag"] = bool(
                    gse == gse and gse > float(grid_scale_threshold)
                )
                if entry["grid_scale_flag"]:
                    warnings.append(
                        f"resolution_sensor: field {key!r} Nyquist-bin energy "
                        f"{gse:.3g} exceeds threshold {grid_scale_threshold} — "
                        f"possible checkerboard / grid-scale artifact (diagnostic only)"
                    )
        except Exception as err:  # noqa: BLE001
            entry["grid_scale_error"] = str(err)

        if disagree is None:
            entry["status"] = "skipped" if "status" not in entry else entry["status"]
            per_field[key] = entry
            continue

        # Prefer interior (away from one-sided closures) for the unreliable flag
        pad = max(order_hi, 2)
        interior = disagree
        if disagree.ndim == 1 and int(disagree.shape[0]) > 2 * pad:
            edge = jnp.concatenate([disagree[:pad], disagree[-pad:]])
            interior = disagree[pad:-pad]
            entry["edge_max"] = float(jnp.nanmax(edge))
            entry["interior_max"] = float(jnp.nanmax(interior))
        elif disagree.ndim == 2 and min(disagree.shape) > 2 * pad:
            edge_vals = jnp.concatenate(
                [
                    disagree[:pad, :].ravel(),
                    disagree[-pad:, :].ravel(),
                    disagree[:, :pad].ravel(),
                    disagree[:, -pad:].ravel(),
                ]
            )
            interior = disagree[pad:-pad, pad:-pad]
            entry["edge_max"] = float(jnp.nanmax(edge_vals))
            entry["interior_max"] = float(jnp.nanmax(interior))
        if region_reduce == "max":
            red = float(jnp.nanmax(interior))
        else:
            red = float(jnp.nanmean(interior))
        entry["disagreement_reduce"] = red
        entry["threshold"] = float(threshold)
        entry["unreliable"] = bool(red > float(threshold))
        per_field[key] = entry
        if entry["unreliable"]:
            warnings.append(
                f"resolution_sensor: field {key!r} interior order-disagreement "
                f"{red:.3g} exceeds threshold {threshold} — derivatives may be unreliable"
            )

    return {
        "per_field": per_field,
        "warnings": warnings,
        "order_hi": order_hi,
        "order_lo": order_lo,
        "threshold": float(threshold),
        "grid_scale_threshold": float(grid_scale_threshold),
        "note": "Diagnostic only; does not change admissibility tiers.",
    }
