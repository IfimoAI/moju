"""Batched evaluate for opt-in implied scoring (percentile / noise-norm), no regions."""

from __future__ import annotations

import math
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from moju.monitor import ResidualEngine, evaluate
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


def _batch(n: int = 24, batch: int = 8):
    x = jnp.linspace(0.0, 1.0, n)
    y = jnp.linspace(0.0, 2.0 * jnp.pi, n)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    fields = kovasznay_2d(X, Y, re=40.0)
    base = ns_state_from_analytic(fields, X, Y)
    base.update({"rho": 1.0, "U": 1.0, "L": 1.0, "mu": 1.0 / 40.0, "re": 40.0})
    states = []
    for i in range(batch):
        s = dict(base)
        s["p_grad"] = base["p_grad"] + (1e-6 * i) * jnp.ones_like(base["p_grad"])
        states.append(s)
    return states


def _eng():
    return ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _NS_SM}],
        law_implied_audits=True,
        constants={"rho": 1.0, "U": 1.0, "L": 1.0},
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"implied_metric": "percentile", "implied_percentile": 95.0},
        {"implied_noise_normalization": True},
        {
            "implied_metric": "percentile",
            "implied_percentile": 95.0,
            "implied_noise_normalization": True,
        },
    ],
)
def test_batched_optin_equals_per_item_audit(kwargs):
    states = _batch(batch=8)
    eng = _eng()
    reset_batch_trace_count()
    batched = evaluate(eng, states, weighted_norms=False, **kwargs)
    assert get_batch_trace_count() == 1
    assert all(r["execution"] == "batched" for r in batched)
    looped = evaluate(
        eng, states, weighted_norms=False, evaluate_debug_loop=True, **kwargs
    )
    assert all(r["execution"] == "loop" for r in looped)
    for i in range(8):
        a = batched[i]["per_key"][_MU_KEY]
        b = looped[i]["per_key"][_MU_KEY]
        assert a["admissibility_metric"] == b["admissibility_metric"]
        assert float(a["admissibility_score"]) == pytest.approx(
            float(b["admissibility_score"]), rel=0, abs=1e-5
        )
        assert float(a["score_for_admissibility"]) == pytest.approx(
            float(b["score_for_admissibility"]), rel=0, abs=1e-5
        )


def test_batched_optin_single_traced_call():
    states = _batch(batch=8)
    eng = _eng()
    reset_batch_trace_count()
    evaluate(
        eng,
        states,
        weighted_norms=False,
        implied_metric="percentile",
        implied_percentile=95.0,
    )
    assert get_batch_trace_count() == 1
    evaluate(
        eng,
        states,
        weighted_norms=False,
        implied_metric="percentile",
        implied_percentile=95.0,
    )
    assert get_batch_trace_count() == 2


def test_defaults_still_batched_and_match_loop():
    """Default worst-point path stays on batch and matches the debug loop."""
    states = _batch(batch=4)
    eng = _eng()
    batched = evaluate(eng, states, weighted_norms=False)
    looped = evaluate(eng, states, weighted_norms=False, evaluate_debug_loop=True)
    assert all(r["execution"] == "batched" for r in batched)
    for a, b in zip(batched, looped):
        assert a["overall_admissibility_score"] == pytest.approx(
            b["overall_admissibility_score"], rel=0, abs=1e-8
        )
        assert a["per_key"][_MU_KEY]["admissibility_score"] == pytest.approx(
            b["per_key"][_MU_KEY]["admissibility_score"], rel=0, abs=1e-8
        )


def test_timing_optin_batched_vs_loop_reported():
    states = _batch(batch=8, n=32)
    eng = _eng()
    kw = {
        "implied_metric": "percentile",
        "implied_percentile": 95.0,
        "implied_noise_normalization": True,
        "weighted_norms": False,
    }
    evaluate(eng, states[:2], **kw)  # warmup
    t0 = time.perf_counter()
    evaluate(eng, states, **kw)
    t_batch = time.perf_counter() - t0
    t0 = time.perf_counter()
    evaluate(eng, states, evaluate_debug_loop=True, **kw)
    t_loop = time.perf_counter() - t0
    print(
        f"[timing] evaluate opt-in batch={t_batch:.4f}s loop={t_loop:.4f}s "
        f"ratio_loop/batch={t_loop / max(t_batch, 1e-9):.2f} (n=8)"
    )
    assert t_batch > 0 and t_loop > 0
