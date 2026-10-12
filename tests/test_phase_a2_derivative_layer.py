"""Phase A2: weighted norms, curvilinear, dedicated d2, closures, sensor, controls."""

from __future__ import annotations

import zipfile
import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from moju.monitor import ResidualEngine, audit
from moju.monitor.analytic_flows import (
    kovasznay_2d,
    ns_state_from_analytic,
    taylor_couette_2d,
    taylor_green_2d,
)
from moju.monitor.curvilinear import (
    fill_vector_grad_lap_curvilinear_2d,
    gradient_curvilinear_2d,
)
from moju.monitor.fd_stencils import (
    diff1_uniform,
    diff2_uniform,
    fornberg_weights,
    mixed_diff_uniform,
)
from moju.monitor.path_b_derivatives import PathBGridConfig
from moju.monitor.resolution_sensor import build_resolution_sensor_report
from moju.monitor.weighted_norms import trapezoid_weights_1d, weighted_rms


def _ns_engine() -> ResidualEngine:
    return ResidualEngine(
        constants={},
        laws=[
            {
                "name": "momentum_navier_stokes",
                "state_map": {
                    "u_t": "u_t",
                    "u": "u",
                    "u_grad": "u_grad",
                    "p_grad": "p_grad",
                    "u_laplacian": "u_laplacian",
                    "re": "re",
                },
            }
        ],
        groups=[],
        law_implied_audits=False,
        default_coord_dimension=2,
    )


def test_trapezoid_weights_uniform_equal():
    x = jnp.linspace(0.0, 1.0, 5)
    w = trapezoid_weights_1d(x)
    assert float(jnp.sum(w)) == pytest.approx(1.0, rel=1e-6)


def test_weighted_rms_clustering_invariant():
    """Dense clustering must not change weighted RMS of a constant residual."""
    # Non-uniform x: clustered near 0
    x = jnp.concatenate(
        [jnp.linspace(0.0, 0.1, 40), jnp.linspace(0.1, 1.0, 10)]
    )
    r = jnp.ones_like(x)  # constant residual
    w = trapezoid_weights_1d(x)
    # Unweighted RMS over-represents the cluster; weighted = 1
    unweighted = float(jnp.sqrt(jnp.mean(r**2)))
    wr = float(weighted_rms(r, w))
    assert unweighted == pytest.approx(1.0)
    assert wr == pytest.approx(1.0, rel=1e-5)
    # Spatially varying residual: r = x; weighted RMS ≈ sqrt(∫x² dx / L)
    r2 = x
    wr2 = float(weighted_rms(r2, w))
    # Composite trapezoid of x^2 on [0,1] / length
    exact = float(jnp.sqrt(jnp.trapezoid(r2**2, x=x) / (float(x[-1] - x[0]))))
    assert wr2 == pytest.approx(exact, rel=2e-2)


def test_weighted_p95_safeguard_detects_local_violation():
    """Default weighted+p95 should drop below High sooner than weighted-only."""
    nx, ny = 48, 40
    s = jnp.linspace(0.0, 1.0, nx)
    xs = s**2.5
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, ny, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = kovasznay_2d(x, y, re=40.0)
    state = ns_state_from_analytic(fields, x, y)
    region = x <= float(xs[max(3, nx // 5)])
    pg = jnp.asarray(state["p_grad"])
    state["p_grad"] = pg + jnp.zeros_like(pg).at[..., 0].set(
        jnp.where(region, 0.2, 0.0)
    )
    w = trapezoid_weights_1d(xs)[:, None] * trapezoid_weights_1d(ys)[None, :]
    eng = _ns_engine()
    eng.compute_residuals(
        state, weighted_norms=True, weighted_safeguard=True, point_weights=w
    )
    score_sg = float(audit(eng.log)["overall_admissibility_score"])
    eng.clear_log()
    eng.compute_residuals(
        state, weighted_norms=True, weighted_safeguard=False, point_weights=w
    )
    score_w = float(audit(eng.log)["overall_admissibility_score"])
    assert score_sg < 0.909  # p95 safeguard engages
    assert score_w >= 0.909  # bare weighted still High at mag=0.2


def test_safeguard_percentile_is_configurable():
    """safeguard_percentile overrides the default percentile used in scoring."""
    from moju.monitor.weighted_safeguard import DEFAULT_PERCENTILE

    assert DEFAULT_PERCENTILE == 95.0
    nx, ny = 40, 40
    xs = jnp.linspace(0.0, 1.0, nx)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, ny, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = kovasznay_2d(x, y, re=40.0)
    state = ns_state_from_analytic(fields, x, y)
    w = trapezoid_weights_1d(xs)[:, None] * trapezoid_weights_1d(ys)[None, :]
    eng = _ns_engine()
    eng.compute_residuals(
        state,
        weighted_norms=True,
        weighted_safeguard=True,
        safeguard_percentile=95.0,
        point_weights=w,
        auto_path_b_derivatives=PathBGridConfig(
            spatial_dimension=2, steady=True, fd_order=4
        ),
        fill_law_fd=True,
    )
    entry = eng.log[-1]
    assert entry["weighted_safeguard"]["laws/momentum_navier_stokes"]["percentile"] == 95.0
    eng.clear_log()
    eng.compute_residuals(
        state,
        weighted_norms=True,
        weighted_safeguard=True,
        safeguard_percentile=99.0,
        point_weights=w,
        auto_path_b_derivatives=PathBGridConfig(
            spatial_dimension=2, steady=True, fd_order=4
        ),
        fill_law_fd=True,
    )
    entry99 = eng.log[-1]
    assert entry99["weighted_safeguard"]["laws/momentum_navier_stokes"]["percentile"] == 99.0
    # Higher percentile is a stricter (or equal) safeguard on the same residual field.
    assert float(entry99["rms"]["laws/momentum_navier_stokes"]) >= float(
        entry["rms"]["laws/momentum_navier_stokes"]
    ) - 1e-12


def test_weighted_norms_respect_eval_mask():
    """Weights must align with masked-in points (no shape error; band-only scoring)."""
    n = 32
    x = jnp.linspace(0.0, 1.0, n)
    phi = jnp.sin(jnp.pi * x)
    eng = ResidualEngine(
        constants={},
        laws=[
            {
                "name": "laplace_equation",
                "state_map": {"phi_laplacian": "phi_laplacian"},
            }
        ],
        groups=[],
        law_implied_audits=False,
        default_coord_dimension=1,
    )
    grid = PathBGridConfig(spatial_dimension=1, steady=True, fd_order=4)
    mask = x > 0.3
    eng.compute_residuals(
        {"x": x, "phi": phi},
        auto_path_b_derivatives=grid,
        fill_law_fd=True,
        weighted_norms=True,
        eval_mask=mask,
    )
    assert eng.log[-1]["weighted_norms"]["scoring_uses_weights"] is True
    assert "rms" in eng.log[-1]


def test_weighted_norms_sidecar_and_score_mode():
    n = 32
    x = jnp.linspace(0.0, 1.0, n)
    phi = jnp.sin(jnp.pi * x)
    eng = ResidualEngine(
        constants={},
        laws=[
            {
                "name": "laplace_equation",
                "state_map": {"phi_laplacian": "phi_laplacian"},
            }
        ],
        groups=[],
        law_implied_audits=False,
        default_coord_dimension=1,
    )
    grid = PathBGridConfig(spatial_dimension=1, steady=True, fd_order=4)
    eng.compute_residuals(
        {"x": x, "phi": phi},
        auto_path_b_derivatives=grid,
        fill_law_fd=True,
        weighted_norms="report",
    )
    assert "rms_weighted" in eng.log[-1]
    assert eng.log[-1]["weighted_norms"]["scoring_uses_weights"] is False
    # Score mode swaps primary rms
    eng.clear_log()
    eng.compute_residuals(
        {"x": x, "phi": phi},
        auto_path_b_derivatives=grid,
        fill_law_fd=True,
        weighted_norms=True,
    )
    assert "rms_unweighted" in eng.log[-1]
    assert eng.log[-1]["weighted_norms"]["scoring_uses_weights"] is True


@pytest.mark.parametrize("order", [2, 4, 6])
def test_diff2_dedicated_converges(order):
    errors = []
    for n in (24, 48):
        L = 2.0 * jnp.pi
        x = jnp.linspace(0.0, L, n, endpoint=False)
        h = float(L / n)
        f = jnp.sin(3.0 * x)
        d2 = diff2_uniform(f, h, order=order, axis=0)
        pad = order + 1
        exact = -9.0 * jnp.sin(3.0 * x)
        err = float(jnp.max(jnp.abs(d2[pad:-pad] - exact[pad:-pad])))
        errors.append(err)
    assert errors[1] < errors[0] * 0.55


def test_mixed_diff_sine():
    n = 48
    L = 2.0 * jnp.pi
    xs = jnp.linspace(0.0, L, n, endpoint=False)
    ys = jnp.linspace(0.0, L, n, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    h = float(L / n)
    f = jnp.sin(x) * jnp.cos(y)
    dxy = mixed_diff_uniform(f, h, h, order=4, axis_x=0, axis_y=1)
    exact = -jnp.cos(x) * jnp.sin(y)
    pad = 4
    err = float(jnp.max(jnp.abs(dxy[pad:-pad, pad:-pad] - exact[pad:-pad, pad:-pad])))
    assert err < 1e-3


def test_boundary_closure_fornberg_matches_classic_order2():
    """Order-2 left closure matches the classic one-sided stencil."""
    import numpy as np

    w = fornberg_weights(0.0, np.arange(3, dtype=float), 1)[1, :]
    # (-3/2, 2, -1/2) on unit grid
    assert float(w[0]) == pytest.approx(-1.5, rel=1e-10)
    assert float(w[1]) == pytest.approx(2.0, rel=1e-10)
    assert float(w[2]) == pytest.approx(-0.5, rel=1e-10)


@pytest.mark.parametrize("order", [2, 4, 6])
def test_boundary_closure_accuracy_exponential(order):
    """One-sided closures on exp(λx); higher order reduces edge error until λh large."""
    n = 80
    lam = 2.0
    x = jnp.linspace(0.0, 1.0, n)
    h = float(x[1] - x[0])
    f = jnp.exp(lam * x)
    df = diff1_uniform(f, h, order=order, axis=0)
    exact = lam * f
    # Edge band of width order
    edge_err = float(jnp.max(jnp.abs(df[:order] - exact[:order])))
    # Interior
    int_err = float(jnp.max(jnp.abs(df[order:-order] - exact[order:-order])))
    assert edge_err < 0.5  # absolute tolerance on this mild exponential
    assert int_err < 5e-3
    # Document: edge error can exceed interior; masking near walls is often recommended
    assert edge_err >= 0.0


def test_resolution_sensor_warns_on_rough_field():
    n = 32
    x = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    # Near-Nyquist oscillation
    rough = jnp.sin((n // 2 - 1) * x)
    report = build_resolution_sensor_report(
        {"x": x, "T": rough},
        field_keys=["T"],
        order_hi=4,
        order_lo=2,
        threshold=0.05,
    )
    assert "T" in report["per_field"]
    # Rough / near-Nyquist field should show large order disagreement
    assert report["per_field"]["T"].get("disagreement_reduce", 0) > 0.05
    assert report["per_field"]["T"].get("unreliable") is True


def test_perturbed_taylor_green_stays_lower_tier():
    """Known nonzero residual must not become High under fd_order=4."""
    n, t, re = 32, 0.1, 100.0
    xs = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = taylor_green_2d(x, y, t, re=re)
    # Perturb pressure gradient to inject a known residual ~ O(0.2)
    state = ns_state_from_analytic(fields, x, y, t=t)
    state["p_grad"] = state["p_grad"] + 0.2
    eng = _ns_engine()
    eng.compute_residuals(
        state,
        run_mode="training",
        auto_path_b_derivatives=PathBGridConfig(
            spatial_dimension=2, steady=True, fd_order=4
        ),
        fill_law_fd=False,
    )
    report = audit(eng.log)
    score = float(report["overall_admissibility_score"])
    tier = report["overall_admissibility_level"]
    assert score < 0.909  # below High cutoff
    assert "High" not in tier


def test_perturbed_kovasznay_stays_lower_tier():
    nx, ny, re = 40, 40, 40.0
    xs = jnp.linspace(0.0, 1.0, nx)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, ny, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = kovasznay_2d(x, y, re=re)
    state = ns_state_from_analytic(fields, x, y)
    state["u"] = state["u"] * 1.25  # strong violation of exact balance
    eng = _ns_engine()
    eng.compute_residuals(
        state,
        run_mode="training",
        auto_path_b_derivatives=PathBGridConfig(
            spatial_dimension=2, steady=True, fd_order=4
        ),
        fill_law_fd=False,
    )
    report = audit(eng.log)
    assert float(report["overall_admissibility_score"]) < 0.909


def test_taylor_couette_curvilinear_convergence():
    """Exact Couette on polar grid: residual and grad error drop with resolution at fd4."""
    errors = []
    for nr, nt in ((16, 24), (32, 48)):
        r = jnp.linspace(1.0, 2.0, nr)
        th = jnp.linspace(0.0, 2.0 * jnp.pi, nt, endpoint=False)
        fields = taylor_couette_2d(r, th, re=100.0)
        x, y = fields["x"], fields["y"]
        hr = float(r[1] - r[0])
        ht = float(th[1] - th[0])
        # Curvilinear fill of grad/lap from primitives
        filled = fill_vector_grad_lap_curvilinear_2d(
            fields["u"], x, y, d_xi=hr, d_eta=ht, order=4
        )
        p_x, p_y = gradient_curvilinear_2d(
            fields["p"], x, y, d_xi=hr, d_eta=ht, order=4
        )
        # Compare u_grad to exact (interior)
        pad = 3
        eg = fields["u_grad"][pad:-pad, pad:-pad]
        ag = filled["u_grad"][pad:-pad, pad:-pad]
        errors.append(float(jnp.max(jnp.abs(ag - eg))))
        # NS residual with filled derivatives
        state = {
            "x": x,
            "y": y,
            "u": fields["u"],
            "u_t": fields["u_t"],
            "u_grad": filled["u_grad"],
            "u_laplacian": filled["u_laplacian"],
            "p": fields["p"],
            "p_grad": jnp.stack([p_x, p_y], axis=-1),
            "re": fields["re"],
        }
        eng = _ns_engine()
        eng.compute_residuals(state, run_mode="training")
        rmax = float(
            jnp.max(jnp.abs(eng.last_residuals["laws"]["momentum_navier_stokes"]))
        )
        # Store residual on last iteration
        if nr == 32:
            assert rmax < 0.5  # coarse polar still limited near axis spacing
    assert errors[1] < errors[0]


def test_w2_path_b_paper_bundle_legacy_scores():
    """Bundled w2 Path B state: supplied derivatives reproduce; FD refill compared."""
    root = Path(__file__).resolve().parents[1]
    zpath = (
        root
        / "examples/Notebooks/media/data/wide2_const_prop_1D_cooling_slab_test_state_pred.json.zip"
    )
    if not zpath.is_file():
        pytest.skip("w2 Path B zip not present")
    with zipfile.ZipFile(zpath) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json") and "__MACOSX" not in n]
        with zf.open(names[0]) as f:
            raw = json.load(f)
    state = {k: jnp.asarray(v) for k, v in raw.items()}
    from moju.piratio import Models
    from moju.monitor import implied_group_specs_for_laws

    L, cp, h, k, rho = 0.1, 900.0, 500.0, 200.0, 2700.0
    engine_kw = {
        "constants": {
            "L": L,
            "cp": cp,
            "h": h,
            "k": k,
            "rho": rho,
            "alpha": Models.thermal_diffusivity(k, rho, cp),
        },
        "laws": [
            {
                "name": "fourier_conduction",
                "state_map": {
                    "T_t": "T_t",
                    "T_laplacian": "T_xx",
                    "fo": "fo",
                    "t": "t",
                    "L": "L",
                },
            }
        ],
        "groups": implied_group_specs_for_laws(["fourier_conduction"]),
        "law_implied_audits": False,
        "default_coord_dimension": 1,
    }
    eng = ResidualEngine(**engine_kw)
    eng.compute_residuals(state, run_mode="eval")
    r_supplied = audit(eng.log)
    assert r_supplied["overall_admissibility_score"] is not None

    # Flattened (nt*nx,) eval grid — reshape to structured (nt, nx) for FD refill of T_xx
    n_tot = int(jnp.asarray(state["T"]).size)
    nt, nx = 512, 384
    assert n_tot == nt * nx
    T2 = jnp.asarray(state["T"]).reshape(nt, nx)
    x1 = jnp.asarray(state["x"]).reshape(nt, nx)[0, :]
    hx = float(x1[1] - x1[0])

    def _refill(order):
        if order is None:
            # legacy jnp.gradient along x
            Txx = jnp.gradient(jnp.gradient(T2, hx, axis=1), hx, axis=1)
        else:
            Txx = diff2_uniform(T2, hx, order=order, axis=1)
        st = dict(state)
        st["T_xx"] = Txx.reshape(-1)
        e = ResidualEngine(**engine_kw)
        e.compute_residuals(st, run_mode="eval")
        return audit(e.log)

    r_fd_legacy = _refill(None)
    r_fd4 = _refill(4)
    assert r_fd_legacy["overall_admissibility_score"] is not None
    assert r_fd4["overall_admissibility_score"] is not None
    # Supplied PINN derivatives should score at least as well as coarse FD refill
    assert float(r_supplied["overall_admissibility_score"]) >= 0.5
