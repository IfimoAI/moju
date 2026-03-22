"""Tests for Moju Studio loaders (no Streamlit required)."""

import io
import json

import numpy as np
import pytest

from apps.moju_studio.studio_io import (
    constants_json_to_dict,
    load_h5_bytes,
    load_netcdf_bytes,
    load_npz_bytes,
    load_npy_bytes,
    load_state_bundle_bytes,
    merge_monitor_config_fragment,
    parse_monitor_config_json,
)


def test_load_npz_bytes_roundtrip():
    bio = io.BytesIO()
    np.savez(bio, T=np.array([1.0, 2.0]), phi_laplacian=np.zeros((2, 3)))
    raw = bio.getvalue()
    d = load_npz_bytes(raw)
    assert set(d.keys()) == {"T", "phi_laplacian"}
    assert tuple(d["T"].shape) == (2,)


def test_constants_json_to_dict():
    d = constants_json_to_dict('{"rho": 1.2, "arr": [1, 2]}')
    assert float(d["rho"]) == pytest.approx(1.2)
    assert d["arr"].shape == (2,)


def test_merge_monitor_config_fragment_constants():
    base = {"laws": [], "constants": {"a": 1}}
    out = merge_monitor_config_fragment(base, {"constants": {"b": 2}})
    assert out["constants"] == {"a": 1, "b": 2}


def test_parse_monitor_config_json():
    d = parse_monitor_config_json(json.dumps({"laws": [], "scaling_audit": []}))
    assert d["laws"] == []


def test_load_npy_bytes():
    bio = io.BytesIO()
    np.save(bio, np.array([[1.0, 2.0], [3.0, 4.0]]))
    raw = bio.getvalue()
    d = load_npy_bytes(raw, "velocity")
    assert set(d.keys()) == {"velocity"}
    assert tuple(d["velocity"].shape) == (2, 2)


def test_load_npy_bytes_requires_key():
    bio = io.BytesIO()
    np.save(bio, np.ones(2))
    raw = bio.getvalue()
    with pytest.raises(ValueError, match="key"):
        load_npy_bytes(raw, "  ")


def test_load_state_bundle_bytes_bad_extension():
    with pytest.raises(ValueError, match="Unsupported"):
        load_state_bundle_bytes(b"xyz", filename="state.csv")


def test_load_state_bundle_bytes_dispatch():
    bio = io.BytesIO()
    np.save(bio, np.ones(3))
    raw = bio.getvalue()
    d = load_state_bundle_bytes(raw, filename="x.npy", npy_key="u")
    assert "u" in d


def test_load_h5_bytes_roundtrip():
    try:
        import h5py
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"h5py not usable in this environment: {e}")
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.create_dataset("pressure", data=np.array([1.0, 2.0]))
        g = f.create_group("nested")
        g.create_dataset("T", data=np.zeros((2, 3)))
    raw = buf.getvalue()
    d = load_h5_bytes(raw, "pressure")
    assert set(d.keys()) == {"pressure"}
    d_all = load_h5_bytes(raw, "")
    assert "pressure" in d_all and "nested/T" in d_all


def test_load_netcdf_bytes_roundtrip():
    try:
        import xarray
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"xarray not usable in this environment: {e}")
    ds = xarray.Dataset({"T": (["i"], np.array([1.0, 2.0, 3.0]))})
    buf = io.BytesIO()
    try:
        ds.to_netcdf(buf, engine="netcdf4")
    except Exception:
        pytest.skip("netcdf4 engine not available for write")
    raw = buf.getvalue()
    d = load_netcdf_bytes(raw, "T")
    assert set(d.keys()) == {"T"}
    assert tuple(d["T"].shape) == (3,)
    d_all = load_netcdf_bytes(raw, "")
    assert "T" in d_all


def test_studio_core_laplace_path_b():
    from apps.moju_studio.studio_core import flatten_residuals, monitor_config_from_merged_dict, preflight_engine

    import jax.numpy as jnp

    from moju.monitor import ResidualEngine, audit

    frag = {
        "laws": [{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_laplacian"}}],
        "groups": [],
        "constitutive_audit": [],
        "scaling_audit": [],
        "constants": {},
    }
    cfg = monitor_config_from_merged_dict(frag)
    engine = ResidualEngine(config=cfg)
    pred = {"phi_laplacian": jnp.zeros((4, 5))}
    residuals = engine.compute_residuals(pred, None)
    rep = audit(engine.log)
    assert "overall_admissibility_score" in rep
    flat = flatten_residuals(residuals)
    assert any(k.startswith("laws/") for k in flat)
    miss_s, _miss_d = preflight_engine(engine, set(pred.keys()))
    assert not miss_s
