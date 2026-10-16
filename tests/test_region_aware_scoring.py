"""Moju 2.3: region-aware implied-law fitting and per-region scoring."""

from __future__ import annotations

import json
import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from moju.monitor import ResidualEngine, audit, evaluate, export_monitor_log
from moju.monitor.analytic_flows import kovasznay_2d, ns_state_from_analytic
from moju.monitor.monitor_log_export import monitor_log_export_to_jsonable
from moju.monitor.regions import (
    DEFAULT_REGION_MIN_POINTS,
    normalize_regions,
    percentile_tail_adequacy,
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


def math_isfinite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _kov_state(n: int = 48, re: float = 40.0):
    x = jnp.linspace(0.0, 1.0, n)
    y = jnp.linspace(0.0, 2.0 * jnp.pi, n)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    fields = kovasznay_2d(X, Y, re=re)
    state = ns_state_from_analytic(fields, X, Y)
    state.update({"rho": 1.0, "U": 1.0, "L": 1.0, "mu": 1.0 / re, "re": re})
    return state, X, Y


def _stokes_piecewise(n: int, nu_l: float, nu_r: float, noise_std: float = 0.0, seed: int = 0):
    """Steady Stokes-like: ∇p = ν ∇²u on a 1-D collocation of n points."""
    u = jnp.ones((n, 2)).at[:, 1].set(0.0)
    lap = jnp.ones((n, 2)).at[:, 1].set(0.0)
    p_grad = jnp.zeros((n, 2))
    p_grad = p_grad.at[: n // 2, 0].set(nu_l * lap[: n // 2, 0])
    p_grad = p_grad.at[n // 2 :, 0].set(nu_r * lap[n // 2 :, 0])
    if noise_std > 0.0:
        key = jax.random.PRNGKey(seed)
        p_grad = p_grad + noise_std * jax.random.normal(key, p_grad.shape)
    left = np.zeros(n, dtype=bool)
    left[: n // 2] = True
    right = ~left
    state = {
        "u": u,
        "u_t": jnp.zeros((n, 2)),
        "u_grad": jnp.zeros((n, 2, 2)),
        "p_grad": p_grad,
        "u_laplacian": lap,
        "re": 1.0 / nu_l,
        "rho": 1.0,
        "U": 1.0,
        "L": 1.0,
        "mu": nu_l,
    }
    return state, {"left": left, "right": right}


def test_region_min_points_default_justified():
    """16 is a fit floor (≥ 2× implied_min_identifiable_count), not percentile adequacy."""
    assert DEFAULT_REGION_MIN_POINTS == 16


@pytest.mark.parametrize(
    "n,pct,expected",
    [
        (19, 95.0, "equivalent_to_worst_point"),  # tail = 0.95 < 1
        (20, 95.0, "low_tail_count"),  # tail = 1.0
        (99, 95.0, "low_tail_count"),  # tail = 4.95
        (100, 95.0, "ok"),  # tail = 5.0
        (10, 90.0, "low_tail_count"),  # tail = 1.0
        (9, 90.0, "equivalent_to_worst_point"),  # tail = 0.9
    ],
)
def test_percentile_tail_adequacy_three_states(n, pct, expected):
    out = percentile_tail_adequacy(n, pct)
    assert out["percentile_status"] == expected
    assert out["percentile_tail_count"] == pytest.approx(n * (1.0 - pct / 100.0))


def test_percentile_status_on_regions_and_global_report():
    """Small / mid / large regions get all three flags; global also reported."""
    n = 200
    state, _ = _stokes_piecewise(n, 0.10, 0.01)
    # Override region sizes for the three flag bands at p95: 15 / 40 / 145.
    tiny = np.zeros(n, dtype=bool)
    tiny[:15] = True
    mid = np.zeros(n, dtype=bool)
    mid[15:55] = True
    big = np.zeros(n, dtype=bool)
    big[55:] = True  # 145 pts → tail=7.25 → ok
    regions = {"tiny": tiny, "mid": mid, "big": big}
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
        implied_metric="percentile",
        implied_percentile=95.0,
    )
    eng.compute_residuals(state, weighted_norms=False)
    rep = audit(eng.log, last_residual_dict=eng.last_residuals)
    assert rep["region_results"]["regions"]["tiny"]["percentile_status"] == (
        "equivalent_to_worst_point"
    )
    assert rep["region_results"]["regions"]["mid"]["percentile_status"] == "low_tail_count"
    assert rep["region_results"]["regions"]["big"]["percentile_status"] == "ok"
    assert "percentile_adequacy" in rep
    assert rep["percentile_adequacy"]["percentile_status"] == "ok"  # n=200, p95 → tail=10
    assert rep["percentile_adequacy"]["percentile_tail_count"] == pytest.approx(10.0)


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


def test_audit_regions_kwargs_per_call_overrides_constructor():
    """audit(..., regions=...) with last_residual_dict; per-call > constructor."""
    state, regions = _stokes_piecewise(64, 0.10, 0.01)
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
        regions={"left": regions["left"]},  # constructor: left only
        region_min_points=8,
    )
    eng.compute_residuals(state, weighted_norms=False)
    # Per-call audit with both regions and a different min_points.
    labels = np.zeros(64, dtype=int)
    labels[:32] = 1
    labels[32:] = 2
    rep = audit(
        eng.log,
        last_residual_dict=eng.last_residuals,
        regions=labels,
        region_names={1: "A", 2: "B"},
        region_min_points=10,
    )
    assert set(rep["region_results"]["regions"].keys()) == {"A", "B"}
    assert rep["region_results"]["provenance"]["region_min_points"] == 10
    assert "left" not in rep["region_results"]["regions"]


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
    state, regions = _stokes_piecewise(64, 0.10, 0.01)
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
    vis_l = rep["region_results"]["regions"]["left"]["viscosity"]
    vis_r = rep["region_results"]["regions"]["right"]["viscosity"]
    assert vis_l["status"] == "ok"
    assert vis_r["status"] == "ok"
    assert float(vis_l["nu_tilde"]) == pytest.approx(0.10, rel=0, abs=1e-8)
    assert float(vis_r["nu_tilde"]) == pytest.approx(0.01, rel=0, abs=1e-8)
    assert _MU_KEY in rep["per_key"]
    assert rep["region_results"]["provenance"]["region_min_points"] == 8
    assert rep["region_results"]["provenance"]["n_unassigned"] == 0


def test_noisy_piecewise_viscosity_recovery():
    """Modest noise on piecewise ν; recover ν̃ ± fit quality per region."""
    nu_l, nu_r = 0.10, 0.01
    state, regions = _stokes_piecewise(128, nu_l, nu_r, noise_std=0.002, seed=7)
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
    vis_l = rep["region_results"]["regions"]["left"]["viscosity"]
    vis_r = rep["region_results"]["regions"]["right"]["viscosity"]
    assert vis_l["status"] == "ok"
    assert vis_r["status"] == "ok"
    assert float(vis_l["nu_tilde"]) == pytest.approx(nu_l, rel=0.05, abs=0.005)
    assert float(vis_r["nu_tilde"]) == pytest.approx(nu_r, rel=0.15, abs=0.002)
    assert math_isfinite(vis_l["fit_quality_median"])
    assert math_isfinite(vis_r["fit_quality_median"])
    # Stash for the pre-merge report (printed by pytest -s if desired).
    print(
        "noisy_recovery",
        {
            "left": {
                "nu_tilde": float(vis_l["nu_tilde"]),
                "fit_quality_median": float(vis_l["fit_quality_median"]),
            },
            "right": {
                "nu_tilde": float(vis_r["nu_tilde"]),
                "fit_quality_median": float(vis_r["fit_quality_median"]),
            },
        },
    )


def test_inertia_dominated_region_not_identifiable():
    """Region with zero viscous basis → Not identifiable."""
    n = 64
    state, regions = _stokes_piecewise(n, 0.10, 0.01)
    # Zero the laplacian (basis) on the right half → no identifiable viscous signal.
    state = dict(state)
    state["u_laplacian"] = state["u_laplacian"].at[n // 2 :].set(0.0)
    # Keep p_grad consistent on left; right has nonzero p_grad but zero basis.
    eng = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _NS_SM}],
        law_implied_audits=True,
        constants={
            "rho": 1.0,
            "U": 1.0,
            "L": 1.0,
            "implied_term_fraction": 0.2,
            "implied_fit_quality_fraction": 0.2,
            "implied_min_identifiable_count": 8,
            "implied_min_identifiable_fraction": 0.2,
        },
        regions=regions,
        region_min_points=8,
    )
    eng.compute_residuals(state, weighted_norms=False)
    rep = audit(eng.log)
    weak = rep["region_results"]["regions"]["right"]
    vis = weak["viscosity"]
    assert vis is not None
    assert vis["status"] == "not_identifiable"
    assert weak["per_key"][_MU_KEY]["admissibility_level"] == "Not identifiable"


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

    # Clean baseline for far/global comparison.
    eng_clean = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _NS_SM}],
        law_implied_audits=True,
        constants={"rho": 1.0, "U": 1.0, "L": 1.0},
        regions={"near": near, "far": far},
        region_min_points=16,
        implied_metric="percentile",
        implied_percentile=95.0,
    )
    eng_clean.compute_residuals(state, weighted_norms=False)
    rep_clean = audit(eng_clean.log)
    s_far_clean = rep_clean["region_results"]["regions"]["far"]["per_key"][_MU_KEY][
        "admissibility_score"
    ]
    g_clean = rep_clean["per_key"][_MU_KEY]["admissibility_score"]

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
    # Far region unchanged vs clean baseline (localized noise).
    assert float(s_far) == pytest.approx(float(s_far_clean), rel=0, abs=1e-10)
    # Global p95 may miss a small noisy band.
    g = rep["per_key"][_MU_KEY]["admissibility_score"]
    assert math_isfinite(g)
    assert float(g) == pytest.approx(float(g_clean), rel=0.05, abs=1e-3) or float(g) > float(
        s_near
    )


def test_overlapping_and_unassigned_and_label_form():
    n = 32
    a = np.zeros(n, dtype=bool)
    a[:20] = True
    b = np.zeros(n, dtype=bool)
    b[10:] = True
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

    d = {"r1": labels == 1, "r2": labels == 2}
    norm_d = normalize_regions(d)
    assert norm_d["mask_sha256"]["r1"] == norm2["mask_sha256"]["r1"]
    assert norm_d["mask_sha256"]["r2"] == norm2["mask_sha256"]["r2"]


def test_label_array_vs_dict_identical_audit_scores():
    state, regions = _stokes_piecewise(64, 0.10, 0.01)
    labels = np.zeros(64, dtype=int)
    labels[:32] = 1
    labels[32:] = 2
    common = dict(
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
        region_min_points=8,
    )
    eng_d = ResidualEngine(**common, regions=regions)
    eng_d.compute_residuals(state, weighted_norms=False)
    rep_d = audit(eng_d.log)

    eng_l = ResidualEngine(**common, regions=labels, region_names={1: "left", 2: "right"})
    eng_l.compute_residuals(state, weighted_norms=False)
    rep_l = audit(eng_l.log)

    for name in ("left", "right"):
        a = rep_d["region_results"]["regions"][name]["per_key"][_MU_KEY]["admissibility_score"]
        b = rep_l["region_results"]["regions"][name]["per_key"][_MU_KEY]["admissibility_score"]
        assert float(a) == pytest.approx(float(b), rel=0, abs=1e-12)
        assert (
            rep_d["region_results"]["regions"][name]["mask_sha256"]
            == rep_l["region_results"]["regions"][name]["mask_sha256"]
        )


def test_empty_region_raises():
    with pytest.raises(ValueError, match="empty region"):
        normalize_regions({"empty": np.zeros(8, dtype=bool)})


def test_below_region_min_points_insufficient():
    state, X, _ = _kov_state(24)
    tiny = np.zeros(X.shape, dtype=bool)
    tiny[0, :8] = True
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


def test_evaluate_with_regions_forces_loop_not_batched_jit_vmap():
    """Regions force the evaluate loop path (batched jit/vmap is default-scoring only)."""
    state, X, _ = _kov_state(24)
    near = np.asarray(X < 0.5)
    eng = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _NS_SM}],
        law_implied_audits=True,
        constants={"rho": 1.0, "U": 1.0, "L": 1.0},
    )
    reports = evaluate(
        eng,
        [state, state],
        regions={"near": near, "far": ~near},
        region_min_points=8,
        weighted_norms=False,
    )
    assert reports[0]["execution"] == "loop"
    assert reports[1]["execution"] == "loop"
    assert "region_results" in reports[0]
    prov = reports[0]["region_results"]["provenance"]
    assert prov["region_min_points"] == 8
    assert set(prov["region_names"]) == {"far", "near"}
    assert "mask_sha256" in prov
    assert prov["moju_version"] == "2.3.0"
    # Regions force loop even when candidates are same-shaped (batched jit/vmap
    # is reserved for default implied scoring with regions=None).


def test_mask_hashes_and_region_provenance_in_report_and_json():
    state, regions = _stokes_piecewise(64, 0.10, 0.01)
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
        implied_metric="percentile",
        implied_percentile=95.0,
    )
    eng.compute_residuals(state, weighted_norms=False)
    rep = audit(eng.log, last_residual_dict=eng.last_residuals)
    prov = rep["region_results"]["provenance"]
    assert set(prov["mask_sha256"].keys()) == {"left", "right"}
    for name in ("left", "right"):
        h = prov["mask_sha256"][name]
        assert isinstance(h, str) and len(h) == 64
        assert rep["region_results"]["regions"][name]["mask_sha256"] == h
        assert "percentile_status" in rep["region_results"]["regions"][name]
    assert "percentile_adequacy" in rep

    export = export_monitor_log(eng.log, scope="audit", force=True)
    summary = export["summary"]
    assert "region_results" in summary
    assert "regions_provenance" in summary
    assert summary["regions_provenance"]["mask_sha256"]["left"] == prov["mask_sha256"]["left"]
    blob = monitor_log_export_to_jsonable(export)
    text = json.dumps(blob)
    assert "mask_sha256" in text
    assert "region_min_points" in text
    assert "percentile_status" in text
    assert "percentile_adequacy" in text


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
    state, X, _ = _kov_state(16)
    eng = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _NS_SM}],
        law_implied_audits=True,
        constants={"rho": 1.0, "U": 1.0, "L": 1.0, "regions": {"x": np.ones(1)}},
    )
    eng.compute_residuals(state, weighted_norms=False)
    rep = audit(eng.log)
    assert "region_results" not in rep
