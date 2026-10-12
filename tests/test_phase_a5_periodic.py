"""Phase A5: per-axis periodic FD, d2/mixed boundary orders, grid-scale sensor."""

from __future__ import annotations

import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from moju.monitor.fd_stencils import (
    diff1_uniform,
    diff2_uniform,
    mixed_diff_uniform,
)
from moju.monitor.path_b_derivatives import PathBGridConfig, fill_path_b_derivatives
from moju.monitor.path_b_spectral import validate_spectral_grid_config
from moju.monitor.resolution_sensor import (
    DEFAULT_GRID_SCALE_ENERGY_THRESHOLD,
    build_resolution_sensor_report,
    sensor_grid_scale_energy_nd,
)


def _obs_order(e_c: float, e_f: float, ratio: float = 2.0) -> float:
    return math.log(e_c / e_f) / math.log(ratio)


@pytest.mark.parametrize("order", [2, 4, 6])
def test_periodic_analytic_full_domain(order):
    n = 64
    L = 2.0 * math.pi
    x = jnp.linspace(0.0, L, n, endpoint=False)
    h = L / n
    f = jnp.sin(2.0 * x)
    d1 = diff1_uniform(f, h, order=order, periodic=True)
    d2 = diff2_uniform(f, h, order=order, periodic=True)
    e1 = float(jnp.max(jnp.abs(d1 - 2.0 * jnp.cos(2.0 * x))))
    e2 = float(jnp.max(jnp.abs(d2 + 4.0 * jnp.sin(2.0 * x))))
    # Full-domain including wrap; tighter for higher order
    tol = {2: 5e-2, 4: 5e-4, 6: 5e-6}[order]
    assert e1 < tol
    assert e2 < tol


def test_periodic_per_axis_mixed():
    n = 48
    L = 2.0 * math.pi
    xs = jnp.linspace(0.0, 1.0, n)  # bounded
    ys = jnp.linspace(0.0, L, n, endpoint=False)  # periodic
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    hx = float(xs[1] - xs[0])
    hy = L / n
    f = jnp.exp(x) * jnp.sin(y)
    # ∂²/∂x∂y = exp(x) cos(y)
    exact = jnp.exp(x) * jnp.cos(y)
    out = mixed_diff_uniform(
        f, hx, hy, order=4, periodic_x=False, periodic_y=True
    )
    # Interior in x, full in y
    pad = 2
    err = float(jnp.max(jnp.abs(out[pad:-pad, :] - exact[pad:-pad, :])))
    assert err < 5e-4


def test_spectral_rejects_partial_periodic():
    with pytest.raises(ValueError, match="periodic"):
        validate_spectral_grid_config(
            PathBGridConfig(diff_method="spectral", periodic=(False, True))
        )


def test_path_b_fd_respects_y_periodic():
    n = 32
    L = 2.0 * math.pi
    xs = jnp.linspace(0.0, 1.0, n)
    ys = jnp.linspace(0.0, L, n, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    phi = jnp.exp(x) * jnp.sin(y)
    state = {"x": x, "y": y, "phi": phi}
    grid = PathBGridConfig(
        spatial_dimension=2,
        steady=True,
        fd_order=4,
        periodic=(False, True),
    )
    out, _ = fill_path_b_derivatives(
        state,
        laws_spec=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_laplacian"}}],
        grid=grid,
        fill_law_recipes=True,
    )
    lap = out["phi_laplacian"]
    # Δ(e^x sin y) = 0
    pad = 2
    err = float(jnp.max(jnp.abs(lap[pad:-pad, :])))
    assert err < 1e-3


def test_d2_boundary_matching_order4_converges():
    half = 2
    errs = []
    for n in (24, 48, 96):
        x = jnp.linspace(0.0, 1.0, n)
        h = float(x[1] - x[0])
        f = jnp.exp(x) * (x**3)
        exact = jnp.exp(x) * (x**3 + 6.0 * x**2 + 6.0 * x)
        a2 = diff2_uniform(f, h, order=4)
        e = jnp.abs(a2 - exact)
        band = jnp.concatenate([e[:half], e[-half:]])
        errs.append(float(jnp.sqrt(jnp.mean(jnp.square(band)))))
    o = _obs_order(errs[-2], errs[-1], ratio=2.0)
    assert o > 3.5, f"d2 boundary observed order {o}, errs={errs}"


def test_mixed_corner_matching_order4_converges():
    half = 2
    errs = []
    for n in (24, 48, 96):
        xs = jnp.linspace(0.0, 1.0, n)
        ys = jnp.linspace(0.0, 1.0, n)
        x, y = jnp.meshgrid(xs, ys, indexing="ij")
        hx = float(xs[1] - xs[0])
        hy = float(ys[1] - ys[0])
        f = jnp.exp(x + y) * x * y
        exact = jnp.exp(x + y) * (x * y + x + y + 1.0)
        a = mixed_diff_uniform(f, hx, hy, order=4)
        e = a - exact
        mask = jnp.zeros_like(e, dtype=bool)
        mask = mask.at[:half, :].set(True)
        mask = mask.at[-half:, :].set(True)
        mask = mask.at[:, :half].set(True)
        mask = mask.at[:, -half:].set(True)
        errs.append(float(jnp.sqrt(jnp.mean(jnp.square(e[mask])))))
    o = _obs_order(errs[-2], errs[-1], ratio=2.0)
    assert o > 3.5, f"mixed boundary observed order {o}, errs={errs}"


def test_grid_scale_sensor_checkerboard_and_gaussian():
    n = 32
    x = jnp.linspace(0.0, 1.0, n)
    y = jnp.linspace(0.0, 1.0, n)
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    checker = ((jnp.arange(n)[:, None] + jnp.arange(n)[None, :]) % 2).astype(float)
    gse_c = sensor_grid_scale_energy_nd(checker)
    assert gse_c > DEFAULT_GRID_SCALE_ENERGY_THRESHOLD
    report_c = build_resolution_sensor_report(
        {"x": x, "y": y, "T": checker},
        field_keys=["T"],
        threshold=0.99,  # ignore order-disagreement for this assert
    )
    assert report_c["per_field"]["T"]["grid_scale_flag"] is True
    assert any("Nyquist" in w or "grid-scale" in w for w in report_c["warnings"])

    # Narrow Gaussian: low Nyquist fraction relative to checkerboard
    i0, j0 = n // 2, n // 2
    ii = jnp.arange(n)[:, None]
    jj = jnp.arange(n)[None, :]
    gauss = jnp.exp(-0.5 * (((ii - i0) / 1.2) ** 2 + ((jj - j0) / 1.2) ** 2))
    report_g = build_resolution_sensor_report(
        {"x": xx, "y": yy, "T": gauss},
        field_keys=["T"],
    )
    # Sensor still runs; checkerboard must flag more strongly than a smooth bump
    assert report_g["per_field"]["T"]["grid_scale_energy"] < gse_c
    # Diagnostic never claims to change tiers
    assert "does not change" in report_g["note"].lower() or "Diagnostic" in report_g["note"]
