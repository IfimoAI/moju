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
from typing import Any, Dict, Mapping, Optional, Tuple

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


def from_meshio(
    path: str,
    *,
    var_map: Mapping[str, str],
    cell_or_point: str = "point",
    coords_key: str = "_coords",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load a mesh-based snapshot (VTK/VTU/OpenFOAM via meshio) into a `state_ref` dict.

    Notes
    - For unstructured meshes, this function does not interpolate to collocation points.
      It returns arrays as stored in the file, plus point coordinates under `coords_key`.
    - Use `cell_or_point='point'` for point_data and `'cell'` for cell_data.
    """
    try:
        import meshio  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "meshio is required for this loader. Install with: pip install moju[ref_vtk] "
            "or pip install moju[ref_foam]"
        ) from e

    mesh = meshio.read(path)
    if cell_or_point not in ("point", "cell"):
        raise ValueError("cell_or_point must be 'point' or 'cell'")

    out: Dict[str, Any] = {}
    out[coords_key] = np.asarray(getattr(mesh, "points", None)) if getattr(mesh, "points", None) is not None else None

    data = mesh.point_data if cell_or_point == "point" else mesh.cell_data
    for state_key, field_name in var_map.items():
        if field_name not in data:
            if strict:
                raise KeyError(f"Field {field_name!r} not found in meshio {cell_or_point}_data")
            continue
        val = data[field_name]
        # meshio cell_data can be dict[str, list[np.ndarray]] per cell block; handle common single-block case.
        if cell_or_point == "cell" and isinstance(val, list):
            if len(val) == 1:
                val = val[0]
        out[state_key] = np.asarray(val)
    return out


def from_vtu(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Thin wrapper around `from_meshio` for VTU files."""
    return from_meshio(*args, **kwargs)


def from_vtk(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Thin wrapper around `from_meshio` for VTK files."""
    return from_meshio(*args, **kwargs)


def from_openfoam(
    case_path: str,
    *,
    var_map: Mapping[str, str],
    time: Optional[str] = None,
    cell_or_point: str = "cell",
    coords_key: str = "_coords",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load an OpenFOAM snapshot into `state_ref` via meshio.

    v1 scope: single time snapshot. Many OpenFOAM exports are best consumed by first
    converting to VTK/VTU (e.g. via `foamToVTK`) and using `from_vtu`.
    """
    path = case_path
    if time is not None:
        path = f"{case_path}:{time}"
    return from_meshio(path, var_map=var_map, cell_or_point=cell_or_point, coords_key=coords_key, strict=strict)


def from_hdf5(
    path: str,
    *,
    var_map: Mapping[str, str],
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load reference arrays from an HDF5 file into a `state_ref` dict.

    var_map maps moju state keys to dataset paths within the HDF5 file.
    Example: {"T": "fields/T", "u": "fields/u"}.
    """
    try:
        import h5py  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "h5py is required for this loader. Install with: pip install moju[ref_hdf5]"
        ) from e

    out: Dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        for state_key, ds_path in var_map.items():
            if ds_path not in f:
                if strict:
                    raise KeyError(f"HDF5 dataset path {ds_path!r} not found (for state key {state_key!r})")
                continue
            out[state_key] = np.asarray(f[ds_path][...])
    return out

