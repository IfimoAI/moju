import numpy as np
import pytest

from moju.monitor.state_ref import from_numpy_grids


def test_from_numpy_grids_identity_map_no_interp():
    variables = {"T": np.ones((3,), dtype=float)}
    state_ref = from_numpy_grids(variables=variables)
    assert "T" in state_ref
    assert np.allclose(state_ref["T"], 1.0)


def test_from_numpy_grids_var_map_no_interp():
    variables = {"T_cfd": np.arange(4, dtype=float)}
    state_ref = from_numpy_grids(variables=variables, var_map={"T": "T_cfd"})
    assert np.allclose(state_ref["T"], np.arange(4, dtype=float))


def test_from_xarray_interp_and_map():
    try:
        import xarray as xr  # type: ignore
    except Exception as e:
        pytest.skip(f"xarray import failed in this environment: {e!r}")
    from moju.monitor.state_ref import from_xarray

    ds = xr.Dataset(
        data_vars={"T_cfd": (("t", "x"), np.arange(6, dtype=float).reshape(2, 3))},
        coords={"t": np.array([0.0, 1.0]), "x": np.array([0.0, 0.5, 1.0])},
    )

    out = from_xarray(
        ds,
        var_map={"T": "T_cfd"},
        target={"t": np.array([0.0]), "x": np.array([0.5])},
        method="linear",
    )
    assert "T" in out
    assert np.asarray(out["T"]).shape == (1, 1)

