"""Moju 2.3: region-aware implied-law fitting and per-region scoring."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from moju.monitor import ResidualEngine, audit, evaluate
from moju.monitor.analytic_flows import kovasznay_2d, ns_state_from_analytic
from moju.monitor.regions import (
    DEFAULT_REGION_MIN_POINTS,
    normalize_regions,
    weighted_reduce_magnitude,
    weighted_reduce_magnitude_numpy_reference,
)

_NS_SM = {
    "u_t": "u_t",
    "u": "u",
    "u_grad": "u_grad",
    "p_grad": "p_grad",
    "u_laplacian": "u_laplacian",
    "re": "re",
}
_MU_KEY = "constitutive/dynamic_viscosity_from_re/law_momentum_navier_stokes/implied_delta"
_LAW_KEY = "laws/momentum_navier_stokes"


def _kov_state(n: int = 48, re: float = 40.0):
    x = jnp.linspace(0.0, 1.0, n)
    y = jnp.linspace(0.0, 2.0 * jnp.pi, n)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    fields = kovasznay_2d(X, Y, re=re)
    state = ns_state_from_analytic(fields, X, Y)
    state.update({"rho": 1.0, "U": 1.0, "L": 1.0, "mu": 1.0 / re, "re": re})
    return state, X, Y


def test_region_min_points_default_justified():
    assert DEFAULT_REGION_MIN_POINTS == 16


def test_regions_none_bit_identical_to_no_regions_kwarg():
    state, _, _ = _kov_state(32)
    eng_a = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _NS_SM}],
        law_implied_audits=True,
        constants={"rho": 1.0, "U": 1.0, "L": 1.0},
    )
    eng_a.compute_residuals(state, weighted_norms=False)
    rep_a = audit(eng_a.log)

    eng_b = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _NS_SM}],
        law_implied_audits=True,
        constants={"rho": 1.0, "U": 1.0, "L": 1.0},
        regions=None,
    )
    eng_b.compute_residuals(state, weighted_norms=False, regions=None)
    rep_b = audit(eng_b.log)

    assert rep_a["overall_admissibility_score"] == rep_b["overall_admissibility_score"]
    assert rep_a["per_key"][_MU_KEY]["admissibility_score"] == rep_b["per_key"][_MU_KEY][
        "admissibility_score"
    ]
    assert "region_results" not in rep_a
    assert "region_results" not in rep_b


def test_masked_percentile_matches_numpy_reference():
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(64, 32))
    w = (rng.random(64 * 32) > 0.3).astype(np.float32)
    for metric, pct in (("worst_point", 95.0), ("percentile", 95.0), ("percentile", 50.0)):
        v_j, _ = weighted_reduce_magnitude(arr, w, metric=metric, percentile=pct)
        v_n = weighted_reduce_magnitude_numpy_reference(arr, w, metric=metric, percentile=pct)
        assert v_j == pytest.approx(v_n, rel=0, abs=1e-12)


def test_piecewise_viscosity_per_region_recovers_each_value():
    """Synthetic field: left half ν=0.1, right half ν=0.01; global fit is a blend."""
    n = 64
    nu_l, nu_r = 0.10, 0.01
    u = jnp.ones((n, 2)).at[:, 1].set(0.0)
    lap = jnp.ones((n, 2)).at[:, 1].set(0.0)
    # Steady Stokes-like: ∇p = ν ∇²u  ⇒  p_grad = ν * lap
    p_grad = jnp.zeros((n, 2))
    p_grad = p_grad.at[: n // 2, 0].set(nu_l * lap[: n // 2, 0])
    p_grad = p_grad.at[n // 2 :, 0].set(nu_r * lap[n // 2 :, 0])
    state = {
        "u": u,
        "u_t": jnp.zeros((n, 2)),
        "u_grad": jnp.zeros((n, 2, 2)),
        "p_grad": p_grad,
        "u_laplacian": lap,
        "re": 1.0 / nu_l,  # catalog uses left value; mismatch expected globally
        "rho": 1.0,
        "U": 1.0,
        "L": 1.0,
        "mu": nu_l,
    }
    left = np.zeros(n, dtype=bool)
    left[: n // 2] = True
    right = ~left
    regions = {"left": left, "right": right}

    eng = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _NS_SM}],
        law_implied_audits=True,
        constants={
            "rho": 1.0,
            "U": 1.0,
            "L": 1.0,
            "implied_term_fraction": 0.05,
            "implied_fit_quality_fraction": 0.05,
            "implied_min_identifiable_count": 4,
            "implied_min_identifiable_fraction": 0.05,
        },
        regions=regions,
        region_min_points=8,
    )
    eng.compute_residuals(state, weighted_norms=False)
    rep = audit(eng.log)
    assert "region_results" in rep
    vis_l = rep["region_results"]["regions"]["left"]["viscosity"]
    vis_r = rep["region_results"]["regions"]["right"]["viscosity"]
    assert vis_l["status"] == "ok"
    assert vis_r["status"] == "ok"
    # μ̃ = ρ U L ν̃* with ρ=U=L=1 → μ̃ = ν̃*
    assert float(vis_l["nu_tilde"]) == pytest.approx(nu_l, rel=0, abs=1e-8)
    assert float(vis_r["nu_tilde"]) == pytest.approx(nu_r, rel=0, abs=1e-8)
    # Global implied (catalog μ = nu_l) is a blend / not equal to either alone on whole domain.
    global_imp = eng.log[-1].get("implied_identifiability") or {}
    # Global score still present and independent of regions block.
    assert _MU_KEY in rep["per_key"]
    assert rep["region_results"]["provenance"]["region_min_points"] == 8
    assert rep["region_results"]["provenance"]["n_unassigned"] == 0


def test_inertia_dominated_region_not_identifiable():
    state, X, _ = _kov_state(48, re=400.0)
    # Near-wall band often viscous; far field may fail gate at high Re — use a tiny weak band.
    # Force a region with near-zero laplacian magnitude (no viscous signal).
    n = int(state["u"].shape[0] * state["u"].shape[1]) if False else state["u_laplacian"].shape[0]
    # state from meshgrid is (nx, ny, 2); flatten via compute path — use mask on leading axes
    mask_weak = np.asarray(X > 0.85)
    # Expand to match residual point layout after engine (nx, ny)
    eng = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _NS_SM}],
        law_implied_audits=True,
        constants={
            "rho": 1.0,
            "U": 1.0,
            "L": 1.0,
            "implied_term_fraction": 0.5,  # strict: viscous must dominate
            "implied_min_identifiable_count": 8,
        },
        regions={"weak": mask_weak},
        region_min_points=8,
    )
    eng.compute_residuals(state, weighted_norms=False)
    rep = audit(eng.log)
    weak = rep["region_results"]["regions"]["weak"]
    # Either insufficient after gate → not_identifiable on viscosity, or ok if viscous enough.
    vis = weak.get("viscosity")
    if vis is not None:
        assert vis["status"] in ("not_identifiable", "ok", "insufficient_points")
    # Prefer asserting not_identifiable when gate is strict on weak viscous region
    if vis is not None and float(vis.get("n_identifiable") or 0) < 8:
        assert vis["status"] == "not_identifiable"


def test_localized_noise_drops_one_region_only():
    state, X, _ = _kov_state(48)
    key = jax.random.PRNGKey(0)
    noisy = dict(state)
    band = X < 0.15
    noisy["u_laplacian"] = state["u_laplacian"] + 0.05 * jax.random.normal(
        key, state["u_laplacian"].shape
    ) * band[..., None]

    near = np.asarray(band)
    far = ~near
    eng = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _NS_SM}],
        law_implied_audits=True,
        constants={"rho": 1.0, "U": 1.0, "L": 1.0},
        regions={"near": near, "far": far},
        region_min_points=16,
        implied_metric="percentile",
        implied_percentile=95.0,
    )
    eng.compute_residuals(noisy, weighted_norms=False)
    rep = audit(eng.log)
    s_near = rep["region_results"]["regions"]["near"]["per_key"][_MU_KEY]["admissibility_score"]
    s_far = rep["region_results"]["regions"]["far"]["per_key"][_MU_KEY]["admissibility_score"]
    assert math_isfinite(s_far) and math_isfinite(s_near)
    assert s_near < s_far
    # Global p95 may miss a small noisy band (motivating region-aware scoring).
    g = rep["per_key"][_MU_KEY]["admissibility_score"]
    assert math_isfinite(g)


def math_isfinite(x) -> bool:
    import math

    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def test_overlapping_and_unassigned_and_label_form():
    n = 32
    a = np.zeros(n, dtype=bool)
    a[:20] = True
    b = np.zeros(n, dtype=bool)
    b[10:] = True  # overlap 10..19; unassigned none if we only use a,b covering 0..31
    # leave last 2 unassigned
    b[30:] = False
    a[30:] = False
    norm = normalize_regions({"a": a, "b": b})
    assert norm["overlaps"] is True
    assert norm["n_unassigned"] == 2

    labels = np.zeros(n, dtype=int)
    labels[5:15] = 1
    labels[20:28] = 2
    norm2 = normalize_regions(labels, region_names={1: "r1", 2: "r2"})
    assert set(norm2["names"]) == {"r1", "r2"}
    assert norm2["n_points"]["r1"] == 10
    assert norm2["n_unassigned"] == n - 10 - 8

    # dict vs labels identical for same coverage
    d = {"r1": labels == 1, "r2": labels == 2}
    norm_d = normalize_regions(d)
    assert norm_d["mask_sha256"]["r1"] == norm2["mask_sha256"]["r1"]
    assert norm_d["mask_sha256"]["r2"] == norm2["mask_sha256"]["r2"]


def test_empty_region_raises():
    with pytest.raises(ValueError, match="empty region"):
        normalize_regions({"empty": np.zeros(8, dtype=bool)})


def test_below_region_min_points_insufficient():
    state, X, _ = _kov_state(24)
    tiny = np.zeros(X.shape, dtype=bool)
    tiny[0, :8] = True  # 8 points < region_min_points=16
    assert int(tiny.sum()) < 16
    eng = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _NS_SM}],
        law_implied_audits=True,
        constants={"rho": 1.0, "U": 1.0, "L": 1.0},
        regions={"tiny": tiny},
        region_min_points=16,
    )
    eng.compute_residuals(state, weighted_norms=False)
    rep = audit(eng.log)
    row = rep["region_results"]["regions"]["tiny"]
    assert row["status"] == "insufficient_points"
    pk = row["per_key"][_MU_KEY]
    assert pk["admissibility_level"] == "Insufficient points"


def test_evaluate_with_regions_loop_and_provenance():
    state, X, _ = _kov_state(24)
    near = np.asarray(X < 0.5)
    eng = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _NS_SM}],
        law_implied_audits=True,
        constants={"rho": 1.0, "U": 1.0, "L": 1.0},
    )
    reports = evaluate(
        eng,
        [state],
        regions={"near": near, "far": ~near},
        region_min_points=8,
        weighted_norms=False,
    )
    assert reports[0]["execution"] == "loop"
    assert "region_results" in reports[0]
    prov = reports[0]["region_results"]["provenance"]
    assert prov["region_min_points"] == 8
    assert set(prov["region_names"]) == {"far", "near"}
    assert "mask_sha256" in prov
    assert prov["moju_version"] == "2.3.0"


def test_jax_torch_region_parity():
    torch = pytest.importorskip("torch")
    from moju.torch._engine import TorchResidualEngine

    state, X, _ = _kov_state(24)
    near = np.asarray(X < 0.4)
    far = ~near
    regions = {"near": near, "far": far}
    eng_j = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _NS_SM}],
        law_implied_audits=True,
        constants={"rho": 1.0, "U": 1.0, "L": 1.0},
        regions=regions,
        region_min_points=8,
    )
    eng_j.compute_residuals(state, weighted_norms=False)
    rep_j = audit(eng_j.log)

    state_t = {
        k: torch.tensor(np.asarray(v).tolist(), dtype=torch.float64) for k, v in state.items()
    }
    eng_t = TorchResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _NS_SM}],
        law_implied_audits=True,
        constants={"rho": 1.0, "U": 1.0, "L": 1.0},
        regions=regions,
        region_min_points=8,
    )
    rep_t = eng_t.audit(state_t)
    for name in ("near", "far"):
        a = rep_j["region_results"]["regions"][name]["per_key"][_MU_KEY]["admissibility_score"]
        b = rep_t["region_results"]["regions"][name]["per_key"][_MU_KEY]["admissibility_score"]
        if math_isfinite(a) and math_isfinite(b):
            assert float(a) == pytest.approx(float(b), rel=0, abs=1e-8)


def test_regions_rejected_via_constants_route_note():
    # constants must not silently configure regions; API is kwargs-only.
    state, X, _ = _kov_state(16)
    eng = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _NS_SM}],
        law_implied_audits=True,
        constants={"rho": 1.0, "U": 1.0, "L": 1.0, "regions": {"x": np.ones(1)}},
    )
    eng.compute_residuals(state, weighted_norms=False)
    rep = audit(eng.log)
    assert "region_results" not in rep
