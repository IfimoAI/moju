"""
Build spatial panels (laws + constitutive) from residuals and ``state_pred``.

By default each cell is :math:`|r|` along the mesh (**absolute residual**). With
``normalize_spatial=True``, values are :math:`|r|/s_k` (**R_norm-style**, same per-key
scale as audit logs).

Used by :func:`moju.monitor.auditor.visualize` when ``residuals`` and coordinates are
available without explicit ``spatial_law_panel`` / ``spatial_rnorm_panel``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import jax.numpy as jnp


def flatten_residuals(residuals: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror auditor flat keys for spatial plotting."""
    flat: Dict[str, Any] = {}
    for category, content in residuals.items():
        if isinstance(content, dict):
            for name, arr in content.items():
                flat[f"{category}/{name}"] = arr
        elif hasattr(content, "shape"):
            flat[category] = content
    return flat


def infer_default_coord_axis_from_residuals(
    residuals: Dict[str, Any],
    *,
    coord_key: str = "x",
    prefer_last_t: bool = True,
) -> Optional[np.ndarray]:
    """
    When no mesh coordinates are supplied, infer a 1D axis length from law/constitutive
    residual arrays and return ``linspace(0, 1, n)`` as a neutral normalized axis.

    All considered arrays must agree on the inferred spatial length; otherwise returns
    ``None`` (caller should pass ``state_pred`` / ``coord_snapshot``).

    ``coord_key`` matches the visualize ``spatial_coord_key`` (typically ``"x"``); the
    returned axis length is inferred from array shapes, not from the key name.
    """
    _ = coord_key
    flat = flatten_residuals(residuals)
    n_spatial: Optional[int] = None
    for fk, v in sorted(flat.items()):
        if not (fk.startswith("laws/") or fk.startswith("constitutive/")):
            continue
        try:
            a = np.asarray(jnp.asarray(v), dtype=float)
        except (TypeError, ValueError):
            continue
        if a.ndim == 0 or a.size == 0:
            continue
        if prefer_last_t and a.ndim >= 2:
            n = int(a.shape[-1])
        elif a.ndim == 1:
            n = int(a.shape[0])
        else:
            n = int(a.shape[-1])
        if n < 1:
            continue
        if n_spatial is None:
            n_spatial = n
        elif n_spatial != n:
            return None
    if n_spatial is None:
        return None
    return np.linspace(0.0, 1.0, n_spatial, dtype=float)


def _per_key_scale_flat(
    flat_key: str,
    *,
    log_entry: Optional[Dict[str, Any]],
    first_rms: Optional[Dict[str, Any]],
    r_ref: Optional[Dict[str, float]],
) -> float:
    """Scalar scale for R_norm-style normalization (matches audit log rules)."""
    if log_entry is None:
        return 1.0
    rms = log_entry.get("rms") or {}
    entry_scale = log_entry.get("scale") or {}
    fr = first_rms or {}
    ref = r_ref or {}
    if flat_key in ref and ref[flat_key] is not None and float(ref[flat_key]) > 0:
        return float(ref[flat_key])
    if flat_key in entry_scale and entry_scale[flat_key] is not None and float(entry_scale[flat_key]) > 0:
        return float(entry_scale[flat_key])
    if flat_key in fr and fr[flat_key] is not None and float(fr[flat_key]) > 0:
        return float(fr[flat_key])
    return 1.0


def _is_law_linked_implied_constitutive_key(flat_key: str) -> bool:
    """True for law-linked implied constitutive residuals."""
    k = str(flat_key)
    if not k.startswith("constitutive/"):
        return False
    parts = k.split("/")
    if "implied_delta" not in parts:
        return False
    return any(p.startswith("law_") for p in parts)


def _spatial_slice_vs_coord(
    arr: Any,
    pred: Dict[str, Any],
    *,
    coord_key: str,
    prefer_last_t: bool,
) -> Optional[np.ndarray]:
    coord = pred.get(coord_key)
    if coord is None:
        return None
    c_arr = np.asarray(jnp.asarray(coord)).ravel()
    nc = int(c_arr.shape[0])
    a = np.asarray(jnp.asarray(arr), dtype=float)
    if a.ndim == 0:
        return None
    t = pred.get("t")
    if prefer_last_t and t is not None and a.ndim >= 2:
        nt = int(np.asarray(jnp.asarray(t)).reshape(-1).shape[0])
        if a.shape[0] == nt:
            a = a[-1]
    while a.ndim > 1 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 1 and a.shape[0] == nc:
        return a
    if a.ndim >= 2 and a.shape[-1] == nc:
        flat = np.reshape(a, (-1, nc))
        return np.nanmean(flat, axis=0)
    if a.ndim >= 2 and a.shape[0] == nc:
        flat = np.reshape(np.moveaxis(a, 0, -1), (-1, nc))
        return np.nanmean(flat, axis=0)
    return None


def _reduce_spatial_array(arr: Any, pred: Dict[str, Any], *, prefer_last_t: bool) -> np.ndarray:
    a = np.asarray(jnp.asarray(arr), dtype=float)
    if a.ndim == 0:
        return a.reshape(1)
    t = pred.get("t")
    if prefer_last_t and t is not None and a.ndim >= 2:
        nt = int(np.asarray(jnp.asarray(t)).reshape(-1).shape[0])
        if a.shape[0] == nt:
            a = np.asarray(a[-1])
    while a.ndim > 1 and a.shape[0] == 1:
        a = a[0]
    return np.asarray(a, dtype=float)


def _spatial_field_2d_grid(
    arr: Any,
    pred: Dict[str, Any],
    *,
    prefer_last_t: bool,
    nx: int,
    ny: int,
) -> Optional[np.ndarray]:
    try:
        a = _reduce_spatial_array(arr, pred, prefer_last_t=prefer_last_t)
    except (TypeError, ValueError):
        return None
    if a.shape == (ny, nx):
        return a
    if a.shape == (nx, ny):
        return a.T
    return None


def _spatial_field_3d_grid(
    arr: Any,
    pred: Dict[str, Any],
    *,
    prefer_last_t: bool,
    nx: int,
    ny: int,
    nz: int,
) -> Optional[np.ndarray]:
    try:
        a = _reduce_spatial_array(arr, pred, prefer_last_t=prefer_last_t)
    except (TypeError, ValueError):
        return None
    if a.shape == (nx, ny, nz):
        return a
    return None


def _build_nd_spatial_panels(
    flat: Dict[str, Any],
    pred: Dict[str, Any],
    *,
    prefer_last_t: bool,
    log_entry: Optional[Dict[str, Any]],
    first_rms: Optional[Dict[str, Any]],
    r_ref: Optional[Dict[str, float]],
    dim: int,
    normalize_spatial: bool,
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if dim == 3:
        if pred.get("x") is None or pred.get("y") is None or pred.get("z") is None:
            return None, None
        x_ = np.asarray(jnp.asarray(pred["x"])).ravel()
        y_ = np.asarray(jnp.asarray(pred["y"])).ravel()
        z_ = np.asarray(jnp.asarray(pred["z"])).ravel()
        nx, ny, nz = int(x_.size), int(y_.size), int(z_.size)

        def grid_fn(arr: Any) -> Optional[np.ndarray]:
            return _spatial_field_3d_grid(
                arr, pred, prefer_last_t=prefer_last_t, nx=nx, ny=ny, nz=nz
            )

        coords = {"x": x_, "y": y_, "z": z_}
    elif dim == 2:
        if pred.get("x") is None or pred.get("y") is None:
            return None, None
        x_ = np.asarray(jnp.asarray(pred["x"])).ravel()
        y_ = np.asarray(jnp.asarray(pred["y"])).ravel()
        nx, ny = int(x_.size), int(y_.size)

        def grid_fn(arr: Any) -> Optional[np.ndarray]:
            return _spatial_field_2d_grid(
                arr, pred, prefer_last_t=prefer_last_t, nx=nx, ny=ny
            )

        coords = {"x": x_, "y": y_}
    else:
        return None, None

    law_values: Dict[str, np.ndarray] = {}
    implied_const_rows: List[tuple[str, np.ndarray]] = []
    other_const_rows: List[tuple[str, np.ndarray]] = []
    for k, v in sorted(flat.items()):
        field = grid_fn(v)
        if field is None:
            continue
        row = np.abs(np.asarray(field, dtype=float))
        if normalize_spatial:
            scale_k = _per_key_scale_flat(k, log_entry=log_entry, first_rms=first_rms, r_ref=r_ref)
            denom = max(float(scale_k), 1e-30)
            row = row / denom
        sk = str(k)
        if sk.startswith("laws/") and len(law_values) < 12:
            law_values[sk] = row
        elif sk.startswith("constitutive/"):
            if _is_law_linked_implied_constitutive_key(sk):
                implied_const_rows.append((sk, row))
            else:
                other_const_rows.append((sk, row))

    const_values: Dict[str, np.ndarray] = {}
    for k, row in implied_const_rows:
        if len(const_values) >= 12:
            break
        const_values[k] = row
    for k, row in other_const_rows:
        if len(const_values) >= 12:
            break
        if k not in const_values:
            const_values[k] = row

    if not law_values and not const_values:
        return None, None
    law_panel = {**coords, "values": law_values} if law_values else None
    implied_panel = {**coords, "values": const_values} if const_values else None
    return law_panel, implied_panel


def build_spatial_rnorm_panels_from_residuals(
    residuals: Optional[Dict[str, Any]],
    pred: Dict[str, Any],
    *,
    coord_key: str = "x",
    prefer_last_t: bool = True,
    log_entry: Optional[Dict[str, Any]] = None,
    first_rms: Optional[Dict[str, Any]] = None,
    r_ref: Optional[Dict[str, float]] = None,
    log_step_index: Optional[int] = None,
    normalize_spatial: bool = False,
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Law and constitutive spatial panels: 3D volume, 2D grid, or 1D along ``coord_key``.

    Default ``normalize_spatial=False``: :math:`|r|` (absolute residual). If True,
    :math:`|r|/s_k` using the same scales as audit R_norm.
    """

    def _tag_step(panel: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if panel is None or log_step_index is None:
            return panel
        return {**panel, "log_step_index": int(log_step_index)}

    if residuals is None:
        return None, None
    flat = flatten_residuals(residuals)
    pred_work = dict(pred)
    if pred_work.get(coord_key) is None:
        inf = infer_default_coord_axis_from_residuals(
            residuals, coord_key=str(coord_key), prefer_last_t=prefer_last_t
        )
        if inf is not None:
            pred_work[str(coord_key)] = inf

    p3 = _build_nd_spatial_panels(
        flat,
        pred_work,
        prefer_last_t=prefer_last_t,
        log_entry=log_entry,
        first_rms=first_rms,
        r_ref=r_ref,
        dim=3,
        normalize_spatial=normalize_spatial,
    )
    if p3[0] is not None or p3[1] is not None:
        return _tag_step(p3[0]), _tag_step(p3[1])

    p2 = _build_nd_spatial_panels(
        flat,
        pred_work,
        prefer_last_t=prefer_last_t,
        log_entry=log_entry,
        first_rms=first_rms,
        r_ref=r_ref,
        dim=2,
        normalize_spatial=normalize_spatial,
    )
    if p2[0] is not None or p2[1] is not None:
        return _tag_step(p2[0]), _tag_step(p2[1])

    coord = pred_work.get(coord_key)
    if coord is None:
        return None, None
    coord_arr = np.asarray(jnp.asarray(coord), dtype=float).ravel()

    law_values: Dict[str, np.ndarray] = {}
    implied_const_rows: List[tuple[str, np.ndarray]] = []
    other_const_rows: List[tuple[str, np.ndarray]] = []
    for k, v in sorted(flat.items()):
        sl = _spatial_slice_vs_coord(v, pred_work, coord_key=coord_key, prefer_last_t=prefer_last_t)
        if sl is None:
            continue
        row = np.abs(np.asarray(sl, dtype=float))
        if normalize_spatial:
            scale_k = _per_key_scale_flat(k, log_entry=log_entry, first_rms=first_rms, r_ref=r_ref)
            denom = max(float(scale_k), 1e-30)
            row = row / denom
        if k.startswith("laws/") and len(law_values) < 12:
            law_values[k] = row
        elif k.startswith("constitutive/"):
            if _is_law_linked_implied_constitutive_key(k):
                implied_const_rows.append((k, row))
            else:
                other_const_rows.append((k, row))

    const_values: Dict[str, np.ndarray] = {}
    for k, row in implied_const_rows:
        if len(const_values) >= 12:
            break
        const_values[k] = row
    for k, row in other_const_rows:
        if len(const_values) >= 12:
            break
        if k not in const_values:
            const_values[k] = row

    pos = {"position_axis": coord_key}
    law_panel = {**pos, "x": coord_arr, "values": law_values} if law_values else None
    implied_panel = {**pos, "x": coord_arr, "values": const_values} if const_values else None
    return _tag_step(law_panel), _tag_step(implied_panel)
