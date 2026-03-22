"""
Load state bundles (NPZ, NPY, HDF5, NetCDF) and JSON MonitorConfig fragments for Moju Studio.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Tuple

import jax.numpy as jnp
import numpy as np

# Cap auto-discovery of datasets/variables to avoid loading huge files by accident.
_MAX_AUTO_KEYS = 512


def load_npz_bytes(data: bytes) -> Dict[str, Any]:
    """Load a .npz into a dict of jax arrays (scalar-safe)."""
    bio = BytesIO(data)
    npz = np.load(bio, allow_pickle=False)
    out: Dict[str, Any] = {}
    for k in npz.files:
        out[str(k)] = jnp.asarray(npz[k])
    npz.close()
    return out


def load_npy_bytes(data: bytes, key: str) -> Dict[str, Any]:
    """
    Load a single-array ``.npy`` into a one-key dict.

    ``key`` is the name used in ``state_pred`` / ``state_ref`` (must be non-empty).
    """
    name = (key or "").strip()
    if not name:
        raise ValueError("NPY uploads require a non-empty array key name (state field name).")
    bio = BytesIO(data)
    arr = np.load(bio, allow_pickle=False)
    if not isinstance(arr, np.ndarray):
        raise ValueError("NPY file must contain an ndarray.")
    if arr.dtype == object:
        raise ValueError("NPY must contain a numeric ndarray, not object dtype.")
    return {name: jnp.asarray(arr)}


def _import_h5py():
    try:
        import h5py  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "HDF5 support requires h5py. Install with: pip install 'moju[studio-science]'"
        ) from e
    return h5py


def _import_xarray():
    try:
        import xarray as xr  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "NetCDF support requires xarray (and netCDF4). "
            "Install with: pip install 'moju[studio-science]'"
        ) from e
    return xr


def _numpy_ok_for_state(arr: np.ndarray) -> bool:
    if arr.dtype == object:
        return False
    return bool(np.issubdtype(arr.dtype, np.number) or arr.dtype == bool)


def load_h5_bytes(data: bytes, dataset_paths: str = "") -> Dict[str, Any]:
    """
    Load HDF5 numeric datasets into a dict (string keys → jax arrays).

    If ``dataset_paths`` is non-empty, it must be a comma-separated list of dataset
    paths inside the file (e.g. ``pressure,/fields/T``). If empty, all numeric leaf
    datasets are loaded (keys are HDF5 paths; capped at 512 datasets).
    """
    h5py = _import_h5py()
    out: Dict[str, Any] = {}
    bio = BytesIO(data)
    with h5py.File(bio, "r") as f:
        paths = [p.strip() for p in (dataset_paths or "").split(",") if p.strip()]
        if paths:
            for p in paths:
                ds = f[p]
                if not isinstance(ds, h5py.Dataset):
                    raise ValueError(f"HDF5 path {p!r} is not a dataset.")
                arr = np.asarray(ds[...])
                if not _numpy_ok_for_state(arr):
                    raise ValueError(f"HDF5 dataset {p!r} is not a numeric array.")
                key = p.lstrip("/")
                out[key] = jnp.asarray(arr)
        else:

            def visitor(name: str, obj: Any) -> None:
                if not isinstance(obj, h5py.Dataset):
                    return
                arr = np.asarray(obj[...])
                if not _numpy_ok_for_state(arr):
                    return
                if len(out) >= _MAX_AUTO_KEYS:
                    raise ValueError(
                        f"HDF5: more than {_MAX_AUTO_KEYS} numeric datasets; "
                        "provide a comma-separated list of dataset paths to load."
                    )
                key = name.lstrip("/") or str(obj.name).lstrip("/")
                out[key] = jnp.asarray(arr)

            f.visititems(visitor)
    if not out:
        raise ValueError("No numeric datasets found in HDF5 file.")
    return out


def load_netcdf_bytes(data: bytes, variable_names: str = "") -> Dict[str, Any]:
    """
    Load NetCDF variables into a dict (variable name → jax array).

    If ``variable_names`` is non-empty, use a comma-separated list (e.g. ``T,u``).
    If empty, load all numeric data variables (capped at 512).
    """
    xr = _import_xarray()
    bio = BytesIO(data)
    # Try common engines for in-memory NetCDF3/4.
    last_err: Exception | None = None
    opened = None
    for engine in ("netcdf4", "scipy"):
        try:
            bio.seek(0)
            opened = xr.open_dataset(bio, engine=engine)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    if opened is None:
        raise ValueError(
            f"Could not open NetCDF (tried netcdf4, scipy). Last error: {last_err!r}"
        )
    try:
        loaded = opened.load()
    finally:
        opened.close()

    names = [n.strip() for n in (variable_names or "").split(",") if n.strip()]
    out: Dict[str, Any] = {}
    if names:
        for n in names:
            if n not in loaded:
                raise ValueError(
                    f"NetCDF: variable {n!r} not found. Available: {list(loaded.data_vars)}"
                )
            arr = np.asarray(loaded[n].values)
            if not _numpy_ok_for_state(arr):
                raise ValueError(f"NetCDF variable {n!r} is not a numeric array.")
            out[n] = jnp.asarray(arr)
    else:
        for n in loaded.data_vars:
            if len(out) >= _MAX_AUTO_KEYS:
                raise ValueError(
                    f"NetCDF: more than {_MAX_AUTO_KEYS} numeric data variables; "
                    "provide a comma-separated list of variable names."
                )
            arr = np.asarray(loaded[n].values)
            if not _numpy_ok_for_state(arr):
                continue
            out[str(n)] = jnp.asarray(arr)
    if not out:
        raise ValueError("No numeric data variables found in NetCDF file.")
    return out


def load_state_bundle_bytes(
    data: bytes,
    *,
    filename: str = "",
    npy_key: str = "field",
    science_selection: str = "",
) -> Dict[str, Any]:
    """
    Dispatch on file extension and load into the same ``dict[str, array]`` as ``load_npz_bytes``.

    Extensions: ``.npz``, ``.npy``, ``.h5``, ``.hdf5``, ``.nc``, ``.nc4``.
    """
    ext = Path(filename or "").suffix.lower().lstrip(".")
    if ext == "npz":
        return load_npz_bytes(data)
    if ext == "npy":
        return load_npy_bytes(data, npy_key)
    if ext in ("h5", "hdf5"):
        return load_h5_bytes(data, science_selection)
    if ext in ("nc", "nc4"):
        return load_netcdf_bytes(data, science_selection)
    raise ValueError(
        f"Unsupported state file extension {ext!r} for {filename!r}. "
        "Use .npz, .npy, .h5, .hdf5, .nc, or .nc4."
    )


def constants_json_to_dict(raw: str) -> Dict[str, Any]:
    """Parse JSON object; convert numeric leaves to jnp.ndarray (0-dim scalars allowed)."""
    d = json.loads(raw or "{}")
    if not isinstance(d, dict):
        raise ValueError("constants JSON must be an object at the root")

    def _leaf(v: Any) -> Any:
        if isinstance(v, (int, float)):
            return jnp.asarray(v)
        if isinstance(v, list):
            return jnp.asarray(v)
        if isinstance(v, dict):
            return {kk: _leaf(vv) for kk, vv in v.items()}
        return v

    return {k: _leaf(v) for k, v in d.items()}


def merge_monitor_config_fragment(
    base: Dict[str, Any], fragment: Dict[str, Any]
) -> Dict[str, Any]:
    """Shallow merge for top-level keys (laws, groups, audits replace if present in fragment)."""
    out = dict(base)
    for k, v in fragment.items():
        if k == "constants" and isinstance(v, dict):
            cur = dict(out.get("constants") or {})
            for ck, cv in v.items():
                cur[ck] = cv
            out["constants"] = cur
        else:
            out[k] = v
    return out


def parse_monitor_config_json(raw: str) -> Dict[str, Any]:
    """Parse JSON; returns a dict suitable for MonitorConfig.from_dict after constants coercion."""
    d = json.loads(raw or "{}")
    if not isinstance(d, dict):
        raise ValueError("MonitorConfig JSON must be an object")
    if "constants" in d and isinstance(d["constants"], dict):
        d = dict(d)
        d["constants"] = {k: _coerce_constant_leaf(v) for k, v in d["constants"].items()}
    return d


def _coerce_constant_leaf(v: Any) -> Any:
    if isinstance(v, (int, float)):
        return jnp.asarray(v)
    if isinstance(v, list):
        return jnp.asarray(v)
    return v


def validate_non_empty_state(state: Dict[str, Any]) -> Tuple[bool, str]:
    if not state:
        return False, "State is empty."
    return True, ""
