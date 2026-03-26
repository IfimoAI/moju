"""Tests for moju.monitor.spatial_rnorm_panels."""

import numpy as np


def test_build_spatial_rnorm_panels_1d():
    from moju.monitor.spatial_rnorm_panels import build_spatial_rnorm_panels_from_residuals

    x = np.linspace(0, 1, 4)
    residuals = {"laws": {"a": np.ones(4) * 0.2}, "constitutive": {"m/c": np.ones(4) * 0.1}}
    pred = {"x": x}
    log_entry = {"rms": {"laws/a": 1.0, "constitutive/m/c": 1.0}, "scale": {}}
    law, cn = build_spatial_rnorm_panels_from_residuals(
        residuals,
        pred,
        log_entry=log_entry,
        first_rms={},
        r_ref={},
        log_step_index=0,
    )
    assert law is not None and cn is not None
    assert law.get("log_step_index") == 0
    assert np.asarray(law["x"]).shape == (4,)
    assert "laws/a" in law["values"]


def test_build_spatial_rnorm_panels_1d_list_coord_from_snapshot_style():
    """JSON log coord_snapshot uses Python lists; panel builder should coerce like ndarray."""
    from moju.monitor.spatial_rnorm_panels import build_spatial_rnorm_panels_from_residuals

    residuals = {"laws": {"a": np.ones(4) * 0.2}, "constitutive": {"m/c": np.ones(4) * 0.1}}
    pred = {"x": [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]}
    log_entry = {"rms": {"laws/a": 1.0, "constitutive/m/c": 1.0}, "scale": {}}
    law, cn = build_spatial_rnorm_panels_from_residuals(
        residuals,
        pred,
        log_entry=log_entry,
        first_rms={},
        r_ref={},
        log_step_index=0,
    )
    assert law is not None and cn is not None
    assert np.asarray(law["x"]).shape == (4,)


def test_build_spatial_rnorm_panels_default_absolute_residual_ignores_scale():
    """Default normalize_spatial=False: panel values are |r|, not |r|/scale."""
    from moju.monitor.spatial_rnorm_panels import build_spatial_rnorm_panels_from_residuals

    x = np.linspace(0, 1, 4)
    residuals = {"laws": {"a": np.ones(4) * 0.8}}
    pred = {"x": x}
    log_entry = {"rms": {"laws/a": 1.0}, "scale": {"laws/a": 10.0}}
    law, _ = build_spatial_rnorm_panels_from_residuals(
        residuals,
        pred,
        log_entry=log_entry,
        first_rms={},
        r_ref={},
    )
    assert law is not None
    assert np.allclose(np.asarray(law["values"]["laws/a"]), 0.8)


def test_infer_default_coord_from_residuals_only():
    from moju.monitor.spatial_rnorm_panels import infer_default_coord_axis_from_residuals

    r = {"laws": {"a": np.ones(6) * 0.1}, "constitutive": {"b": np.ones(6) * 0.05}}
    ax = infer_default_coord_axis_from_residuals(r)
    assert ax is not None and len(ax) == 6
    assert np.isclose(ax[0], 0.0) and np.isclose(ax[-1], 1.0)


def test_build_spatial_panels_residuals_without_state_pred_or_snapshot():
    """visualize(..., residuals=...) without coordinates still builds 1D panels."""
    from moju.monitor.spatial_rnorm_panels import build_spatial_rnorm_panels_from_residuals

    residuals = {"laws": {"a": np.ones(5) * 0.2}, "constitutive": {"m/c": np.ones(5) * 0.1}}
    law, cn = build_spatial_rnorm_panels_from_residuals(
        residuals,
        {},
        log_entry={"rms": {"laws/a": 1.0, "constitutive/m/c": 1.0}, "scale": {}},
        first_rms={},
        r_ref={},
    )
    assert law is not None and cn is not None
    assert np.asarray(law["x"]).shape == (5,)


def test_build_spatial_rnorm_panels_normalize_spatial_divides_by_scale():
    from moju.monitor.spatial_rnorm_panels import build_spatial_rnorm_panels_from_residuals

    x = np.linspace(0, 1, 4)
    residuals = {"laws": {"a": np.ones(4) * 0.8}}
    pred = {"x": x}
    log_entry = {"rms": {"laws/a": 1.0}, "scale": {"laws/a": 10.0}}
    law, _ = build_spatial_rnorm_panels_from_residuals(
        residuals,
        pred,
        log_entry=log_entry,
        first_rms={},
        r_ref={},
        normalize_spatial=True,
    )
    assert law is not None
    assert np.allclose(np.asarray(law["values"]["laws/a"]), 0.08)
