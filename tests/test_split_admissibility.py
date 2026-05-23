"""Split admissibility: RMS for laws, worst-point for ND constitutive closure keys."""

import math

import pytest

from moju.monitor.auditor import (
    ADM_HIGH_THRESHOLD,
    ADM_LOW_THRESHOLD,
    DEFAULT_NONDIM_R_NORM_SCALE_K,
    ResidualEngine,
    _compute_log_step_metrics,
    _key_uses_worst_point_admissibility,
    _max_per_key,
    _rms_per_key,
    audit,
)


def test_key_uses_worst_point_admissibility() -> None:
    assert _key_uses_worst_point_admissibility("constitutive/a/implied_delta")
    assert _key_uses_worst_point_admissibility("constitutive/foo/law_x/ref_delta")
    assert not _key_uses_worst_point_admissibility("laws/a")
    assert not _key_uses_worst_point_admissibility("constitutive/custom/residual")


def test_hotspot_cheating_rms_high_max_non_admissible() -> None:
    """999 points at δ=0 and one at 2%: RMS looks good, worst-point fails."""
    import jax.numpy as jnp

    arr = jnp.zeros(1000)
    arr = arr.at[0].set(0.02)
    flat = {"constitutive/cheat/implied_delta": arr}
    rms = _rms_per_key(flat)["constitutive/cheat/implied_delta"]
    r_max = _max_per_key(flat)["constitutive/cheat/implied_delta"]
    assert r_max == pytest.approx(0.02)
    assert rms < 0.001  # ~0.02/sqrt(1000)

    scale = DEFAULT_NONDIM_R_NORM_SCALE_K
    adm_rms = 1.0 / (1.0 + rms / scale)
    adm_max = 1.0 / (1.0 + r_max / scale)
    assert adm_rms >= ADM_HIGH_THRESHOLD
    assert adm_max < ADM_LOW_THRESHOLD

    log = [
        {
            "index": 0,
            "rms": {"constitutive/cheat/implied_delta": float(rms)},
            "r_max": {"constitutive/cheat/implied_delta": float(r_max)},
            "scale": {"constitutive/cheat/implied_delta": scale},
            "run_mode": "training",
        }
    ]
    m = _compute_log_step_metrics(log)[0]
    pk = m["per_key_report"]["constitutive/cheat/implied_delta"]
    assert pk["admissibility_metric"] == "max"
    assert pk["score_for_admissibility"] == pytest.approx(0.02)
    assert pk["admissibility_score"] == pytest.approx(adm_max)
    assert m["category_admissibility_score"]["constitutive"] == pytest.approx(adm_max)

    report = audit(log)
    assert report["per_key"]["constitutive/cheat/implied_delta"]["admissibility_score"] == pytest.approx(
        adm_max
    )
    summary = report.get("constitutive_closure_summary")
    assert summary is not None
    assert "worst-point" in summary


def test_ref_delta_uses_r_max_for_admissibility() -> None:
    import jax.numpy as jnp

    arr = jnp.array([0.001, 0.015, 0.002])
    flat = {"constitutive/foo/ref_delta": arr}
    rms = _rms_per_key(flat)["constitutive/foo/ref_delta"]
    r_max = _max_per_key(flat)["constitutive/foo/ref_delta"]
    assert r_max == pytest.approx(0.015)

    scale = DEFAULT_NONDIM_R_NORM_SCALE_K
    log = [
        {
            "run_mode": "eval",
            "rms": {"constitutive/foo/ref_delta": float(rms)},
            "r_max": {"constitutive/foo/ref_delta": float(r_max)},
            "scale": {"constitutive/foo/ref_delta": scale},
        }
    ]
    m = _compute_log_step_metrics(log)[0]
    pk = m["per_key_report"]["constitutive/foo/ref_delta"]
    assert pk["admissibility_metric"] == "max"
    expected = 1.0 / (1.0 + 0.015 / scale)
    assert pk["admissibility_score"] == pytest.approx(expected)


def test_backward_compat_no_r_max_falls_back_to_rms() -> None:
    log = [
        {
            "index": 0,
            "rms": {"constitutive/alpha/implied_delta": 0.001},
            "scale": {"constitutive/alpha/implied_delta": 0.01},
        }
    ]
    m = _compute_log_step_metrics(log)[0]
    pk = m["per_key_report"]["constitutive/alpha/implied_delta"]
    assert pk["admissibility_metric"] == "max"
    assert pk["score_for_admissibility"] == pytest.approx(0.001)
    expected = 1.0 / (1.0 + 0.001 / 0.01)
    assert pk["admissibility_score"] == pytest.approx(expected)


def test_constitutive_category_min_with_two_keys() -> None:
    scale = 0.01
    log = [
        {
            "run_mode": "training",
            "rms": {
                "constitutive/a/implied_delta": 0.001,
                "constitutive/b/implied_delta": 0.001,
            },
            "r_max": {
                "constitutive/a/implied_delta": 0.001,
                "constitutive/b/implied_delta": 0.008,
            },
            "scale": {
                "constitutive/a/implied_delta": scale,
                "constitutive/b/implied_delta": scale,
            },
        }
    ]
    m = _compute_log_step_metrics(log)[0]
    adm_a = 1.0 / (1.0 + 0.001 / scale)
    adm_b = 1.0 / (1.0 + 0.008 / scale)
    assert m["category_admissibility_score"]["constitutive"] == pytest.approx(min(adm_a, adm_b))


def test_compute_residuals_logs_r_max() -> None:
    """Engine logs sparse r_max for implied_delta keys."""
    import jax.numpy as jnp

    from moju.piratio.models import Models

    P, R = jnp.array(101325.0), jnp.array(287.0)
    T = jnp.array(290.0)
    rho = Models.ideal_gas_rho(P, R, T)
    engine = ResidualEngine(
        laws=[],
        constitutive_audit=[
            {
                "name": "ideal_gas_rho",
                "output_key": "rho",
                "state_map": {"P": "P", "R": "R", "T": "T"},
                "implied_value_key": "rho_implied",
            }
        ],
    )
    engine.compute_residuals({"P": P, "R": R, "T": T, "rho": rho, "rho_implied": rho})
    entry = engine.log[-1]
    assert "r_max" in entry
    key = "constitutive/ideal_gas_rho/implied_delta"
    assert key in entry["r_max"]
    assert math.isfinite(float(entry["r_max"][key]))
    assert key in entry["rms"]
