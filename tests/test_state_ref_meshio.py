import numpy as np
import pytest


def test_from_meshio_roundtrip_vtu_point_data(tmp_path):
    meshio = pytest.importorskip("meshio")
    from moju.monitor.state_ref import from_meshio

    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    cells = [("line", np.array([[0, 1]], dtype=int))]
    point_data = {"phi": np.array([3.0, 4.0], dtype=float)}

    mesh = meshio.Mesh(points=points, cells=cells, point_data=point_data)
    p = tmp_path / "m.vtu"
    meshio.write(str(p), mesh)

    out = from_meshio(str(p), var_map={"phi": "phi"}, cell_or_point="point")
    assert "_coords" in out
    assert out["_coords"] is not None
    assert np.allclose(out["_coords"], points)
    assert np.allclose(out["phi"], point_data["phi"])

