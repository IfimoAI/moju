"""
Helpers to ingest high-fidelity reference data into `state_ref` dictionaries.

These functions are intentionally lightweight adapters that:
- map dataset variables to moju state keys
- normalize coordinate/dimension naming (x/t/y/z)
- optionally interpolate onto a target grid (when xarray is available)

Core `moju` does not depend on xarray; xarray support is optional.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np


try:  # optional
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore


_CANON_COORDS = ("x", "t", "y", "z")


@dataclass(frozen=True)
class XarrayRefSpec:
    """Configuration for xarray -> state_ref ingestion."""

    var_map: Mapping[str, str]
    coord_map: Mapping[str, str] = None  # canonical -> dataset coord name
    dims_map: Mapping[str, str] = None  # canonical -> dataset dim name (fallback)
    target: Optional[Mapping[str, Any]] = None  # canonical coord -> 1D values
    method: str = "linear"


def _require_xarray() -> Any:
    if xr is None:  # pragma: no cover
        raise ImportError(
            "xarray is required for this loader. Install with: pip install moju[ref]"
        )
    return xr


def _as_dataset(obj: Any) -> Any:
    _xr = _require_xarray()
    if isinstance(obj, _xr.Dataset):
        return obj
    if isinstance(obj, _xr.DataArray):
        return obj.to_dataset(name=obj.name or "var")
    raise TypeError("Expected xarray.Dataset or xarray.DataArray")


def _normalize_maps(
    coord_map: Optional[Mapping[str, str]],
    dims_map: Optional[Mapping[str, str]],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    cm = dict(coord_map or {})
    dm = dict(dims_map or {})
    for c in _CANON_COORDS:
        cm.setdefault(c, c)
        dm.setdefault(c, c)
    return cm, dm


def from_xarray(
    data: Any,
    *,
    var_map: Mapping[str, str],
    coord_map: Optional[Mapping[str, str]] = None,
    dims_map: Optional[Mapping[str, str]] = None,
    target: Optional[Mapping[str, Any]] = None,
    method: str = "linear",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Convert an xarray Dataset/DataArray to a `state_ref` dict.

    Parameters
    - data: xr.Dataset or xr.DataArray
    - var_map: mapping of moju state_key -> dataset variable name
    - coord_map: mapping canonical coord name (x/t/y/z) -> dataset coord name
    - dims_map: mapping canonical coord name (x/t/y/z) -> dataset dim name (fallback)
    - target: optional mapping canonical coord -> target coordinate values (for interpolation)
    - method: interpolation method passed to xarray.interp (e.g. 'linear', 'nearest')
    - strict: if True, missing vars/coords raise; else best-effort
    """
    _xr = _require_xarray()
    ds = _as_dataset(data)
    cm, dm = _normalize_maps(coord_map, dims_map)

    # Normalize/rename coords when possible (best effort).
    rename_coords: Dict[str, str] = {}
    for canon, ds_coord in cm.items():
        if ds_coord in ds.coords and ds_coord != canon:
            rename_coords[ds_coord] = canon
    if rename_coords:
        ds = ds.rename(rename_coords)

    # Some datasets encode axes as dims only; allow dims_map to rename dims to canonical.
    rename_dims: Dict[str, str] = {}
    for canon, ds_dim in dm.items():
        if ds_dim in ds.dims and ds_dim != canon:
            rename_dims[ds_dim] = canon
    if rename_dims:
        ds = ds.rename_dims(rename_dims)

    if target is not None:
        # Ensure target coords exist as coords or dims
        interp_kwargs: Dict[str, Any] = {}
        for canon, values in target.items():
            if canon not in _CANON_COORDS:
                if strict:
                    raise KeyError(f"Unknown canonical coord {canon!r}; expected one of {_CANON_COORDS}")
                continue
            if canon not in ds.coords and canon not in ds.dims:
                if strict:
                    raise KeyError(
                        f"Cannot interpolate: canonical coord {canon!r} not present as a coord or dim"
                    )
                continue
            interp_kwargs[canon] = values
        if interp_kwargs:
            ds = ds.interp(interp_kwargs, method=method)

    out: Dict[str, Any] = {}
    for state_key, var_name in var_map.items():
        if var_name not in ds:
            if strict:
                raise KeyError(f"Variable {var_name!r} not found in dataset (for state key {state_key!r})")
            continue
        out[state_key] = ds[var_name].data
    return out


def from_numpy_grids(
    *,
    variables: Mapping[str, np.ndarray],
    coords: Optional[Mapping[str, np.ndarray]] = None,
    var_map: Optional[Mapping[str, str]] = None,
    target: Optional[Mapping[str, Any]] = None,
    method: str = "linear",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Convert raw numpy arrays (plus optional coords) to a `state_ref` dict.

    If xarray is installed and `coords` is provided, this function can optionally
    perform interpolation by round-tripping through xarray.

    Parameters
    - variables: mapping of variable name -> ndarray
    - coords: optional mapping canonical coord name (x/t/y/z) -> 1D coord arrays
    - var_map: optional mapping moju state_key -> variable name (defaults to identity)
    - target/method: optional interpolation onto target coords (requires xarray)
    """
    vm = dict(var_map or {k: k for k in variables.keys()})

    if target is None or coords is None or xr is None:
        # No interpolation path; just map variables to state keys.
        out: Dict[str, Any] = {}
        for state_key, var_name in vm.items():
            arr = variables.get(var_name)
            if arr is None:
                if strict:
                    raise KeyError(f"Variable {var_name!r} not found in variables (for state key {state_key!r})")
                continue
            out[state_key] = arr
        return out

    _xr = _require_xarray()
    coords_canon = {k: np.asarray(v) for k, v in (coords or {}).items() if k in _CANON_COORDS}
    data_vars: Dict[str, Any] = {}
    for var_name, arr in variables.items():
        data_vars[var_name] = (tuple(coords_canon.keys()), np.asarray(arr))
    ds = _xr.Dataset(data_vars=data_vars, coords=coords_canon)
    return from_xarray(ds, var_map=vm, target=target, method=method, strict=strict)

