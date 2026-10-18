"""Batched evaluate with regions: jit/vmap path, equality, parity, timing."""

from __future__ import annotations

import math
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from moju.monitor import ResidualEngine, audit, evaluate
from moju.monitor.analytic_flows import kovasznay_2d, ns_state_from_analytic
from moju.monitor.evaluate_batch import get_batch_trace_count, reset_batch_trace_count

_NS_SM = {
    "u_t": "u_t",
    "u": "u",
    "u_grad": "u_grad",
    "p_grad": "p_grad",
    "u_laplacian": "u_laplacian",
    "re": "re",
}
_MU_KEY = "constitutive/dynamic_viscosity_from_re/law_momentum_navier_stokes/implied_delta"


def _kov_batch(n: int = 24, re: float = 40.0, batch: int = 8):
    x = jnp.linspace(0.0, 1.0, n)
    y = jnp.linspace(0.0, 2.0 * jnp.pi, n)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    fields = kovasznay_2d(X, Y, re=re)
    base = ns_state_from_analytic(fields, X, Y)
    base.update({"rho": 1.0, "U": 1.0, "L": 1.0, "mu": 1.0 / re, "re": re})
    # Small per-item perturbation so the batch is not a trivial clone.
    states = []
    for i in range(batch):
        s = dict(base)
        s["p_grad"] = base["p_grad"] + (1e-6 * i) * jnp.ones_like(base["p_grad"])
        states.append(s)
    near = np.asarray(X < 0.5)
    regions = {"near": near, "far": ~near}
    return states, regions, X


def _region_snapshot(rep: dict) -> dict:
    rr = rep["region_results"]["regions"]
    out = {}
    for name, block in rr.items():
        pk = block["per_key"][_MU_KEY]
        vis = block.get("viscosity") or {}
        raw_score = pk.get("admissibility_score")
        try:
            score_f = float(raw_score) if raw_score is not None else float("nan")
        except (TypeError, ValueError):
            score_f = float("nan")
        out[name] = {
            "score": score_f if math.isfinite(score_f) else float("nan"),
            "v_eff": float(pk.get("score_for_admissibility", float("nan")))
            if pk.get("score_for_admissibility") is not None
            else float("nan"),
            "worst": pk.get("worst_point_index"),
            "status": pk.get("status"),
            "level": pk.get("admissibility_level"),
            "nu": float(vis.get("nu_tilde", float("nan")))
            if vis.get("nu_tilde") is not None
            else float("nan"),
            "fq": float(vis.get("fit_quality_median", float("nan")))
            if vis.get("fit_quality_median") is not None
            else float("nan"),
            "vis_status": vis.get("status"),
            "percentile_status": block.get("percentile_status"),
            "n_points": block.get("n_points"),
        }
    return out


def test_batched_evaluate_regions_equals_per_item_audit():
    states, regions, _ = _kov_batch(batch=8)
    eng = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _NS_SM}],
        law_implied_audits=True,
        constants={"rho": 1.0, "U": 1.0, "L": 1.0},
    )
    reset_batch_trace_count()
    batched = evaluate(
        eng,
        states,
        regions=regions,
        region_min_points=8,
        weighted_norms=False,
    )
    assert get_batch_trace_count() == 1
    assert all(r["execution"] == "batched" for r in batched)

    # Per-item loop via explicit debug flag (same region rules, host path).
    looped = evaluate(
        eng,
        states,
        regions=regions,
        region_min_points=8,
        weighted_norms=False,
        evaluate_debug_loop=True,
    )
    assert all(r["execution"] == "loop" for r in looped)

    for i in range(8):
        a = _region_snapshot(batched[i])
        b = _region_snapshot(looped[i])
        for name in ("near", "far"):
            assert a[name]["status"] == b[name]["status"]
            assert a[name]["vis_status"] == b[name]["vis_status"]
            assert a[name]["worst"] == b[name]["worst"]
            assert a[name]["n_points"] == b[name]["n_points"]
            if math.isfinite(a[name]["score"]) and math.isfinite(b[name]["score"]):
                assert a[name]["score"] == pytest.approx(b[name]["score"], rel=0, abs=1e-8)
            if math.isfinite(a[name]["nu"]) and math.isfinite(b[name]["nu"]):
                assert a[name]["nu"] == pytest.approx(b[name]["nu"], rel=0, abs=1e-8)


def test_batched_evaluate_regions_single_traced_call():
    states, regions, _ = _kov_batch(batch=8)
    eng = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _NS_SM}],
        law_implied_audits=True,
        constants={"rho": 1.0, "U": 1.0, "L": 1.0},
    )
    reset_batch_trace_count()
    evaluate(eng, states, regions=regions, region_min_points=8, weighted_norms=False)
    assert get_batch_trace_count() == 1
    # Second call reuses the compiled function but still counts as one invoke.
    evaluate(eng, states, regions=regions, region_min_points=8, weighted_norms=False)
    assert get_batch_trace_count() == 2


def test_global_identical_with_and_without_regions():
    states, regions, _ = _kov_batch(batch=4)
    eng = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _NS_SM}],
        law_implied_audits=True,
        constants={"rho": 1.0, "U": 1.0, "L": 1.0},
    )
    with_r = evaluate(eng, states, regions=regions, region_min_points=8, weighted_norms=False)
    without = evaluate(eng, states, weighted_norms=False)
    for a, b in zip(with_r, without):
        assert a["overall_admissibility_score"] == pytest.approx(
            b["overall_admissibility_score"], rel=0, abs=1e-12
        )
        assert a["per_key"][_MU_KEY]["admissibility_score"] == pytest.approx(
            b["per_key"][_MU_KEY]["admissibility_score"], rel=0, abs=1e-12
        )
        assert "region_results" in a
        assert "region_results" not in b


def test_jax_torch_region_parity_on_evaluate_batch_item():
    torch = pytest.importorskip("torch")
    from moju.torch._engine import TorchResidualEngine

    states, regions, _ = _kov_batch(batch=2, n=20)
    eng_j = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _NS_SM}],
        law_implied_audits=True,
        constants={"rho": 1.0, "U": 1.0, "L": 1.0},
    )
    reps_j = evaluate(eng_j, states, regions=regions, region_min_points=8, weighted_norms=False)
    assert reps_j[0]["execution"] == "batched"

    state_t = {
        k: torch.tensor(np.asarray(v).tolist(), dtype=torch.float64)
        for k, v in states[0].items()
    }
    eng_t = TorchResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _NS_SM}],
        law_implied_audits=True,
        constants={"rho": 1.0, "U": 1.0, "L": 1.0},
        regions=regions,
        region_min_points=8,
    )
    rep_t = eng_t.audit(state_t)
    snap_j = _region_snapshot(reps_j[0])
    for name in ("near", "far"):
        a = snap_j[name]["score"]
        b = float(
            rep_t["region_results"]["regions"][name]["per_key"][_MU_KEY]["admissibility_score"]
        )
        if math.isfinite(a) and math.isfinite(b):
            assert a == pytest.approx(b, rel=0, abs=1e-7)


def test_timing_batched_vs_loop_reported(capsys):
    """Report wall times; do not assert a speedup (JIT warmup varies)."""
    states, regions, _ = _kov_batch(batch=8, n=32)
    eng = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _NS_SM}],
        law_implied_audits=True,
        constants={"rho": 1.0, "U": 1.0, "L": 1.0},
    )
    # Warmup compile
    evaluate(eng, states[:2], regions=regions, region_min_points=8, weighted_norms=False)

    t0 = time.perf_counter()
    evaluate(eng, states, regions=regions, region_min_points=8, weighted_norms=False)
    t_batch = time.perf_counter() - t0

    t0 = time.perf_counter()
    evaluate(
        eng,
        states,
        regions=regions,
        region_min_points=8,
        weighted_norms=False,
        evaluate_debug_loop=True,
    )
    t_loop = time.perf_counter() - t0

    print(
        f"[timing] evaluate regions batch={t_batch:.4f}s loop={t_loop:.4f}s "
        f"ratio_loop/batch={t_loop / max(t_batch, 1e-9):.2f} (n=8)"
    )
    assert t_batch > 0 and t_loop > 0


def test_regions_without_debug_loop_never_silently_falls_back():
    states, regions, _ = _kov_batch(batch=2)
    eng = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _NS_SM}],
        law_implied_audits=True,
        constants={"rho": 1.0, "U": 1.0, "L": 1.0},
        best_effort_partial=True,  # forces batch unavailability
    )
    with pytest.raises(RuntimeError, match="batched jit/vmap"):
        evaluate(eng, states, regions=regions, region_min_points=8)
