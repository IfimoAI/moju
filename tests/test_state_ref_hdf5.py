import numpy as np
import pytest


def test_from_hdf5_reads_dataset(tmp_path):
    try:
        import h5py  # type: ignore
    except Exception as e:
        pytest.skip(f"h5py import failed in this environment: {e!r}")
    from moju.monitor.state_ref import from_hdf5

    p = tmp_path / "data.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("fields/T", data=np.arange(5, dtype=float))

    out = from_hdf5(str(p), var_map={"T": "fields/T"})
    assert np.allclose(out["T"], np.arange(5, dtype=float))

