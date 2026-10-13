"""Moju 2.2: opt-in robust scoring for implied constitutive rows."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from moju.monitor import ResidualEngine, audit, implied_group_specs_for_laws
from moju.monitor.analytic_flows import kovasznay_2d, ns_state_from_analytic
from moju.monitor.implied_scoring import (
    DEFAULT_IMPLIED_METRIC,
    noise_normalized_delta,
    parse_implied_scoring_options,
    reduce_implied_magnitude,
)
from moju.piratio import Models

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "examples" / "Notebooks" / "reference" / "32x32x32_opt"
WIDE2_ZIP = (
    ROOT
    / "examples"
    / "Notebooks"
    / "media"
    / "data"
    / "wide2_const_prop_1D_cooling_slab_test_state_pred.json.zip"
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
_ALPHA_KEY = "constitutive/thermal_diffusivity/law_fourier_conduction/implied_delta"


def _ns_engine(constants: dict | None = None) -> ResidualEngine:
    c = {"rho": 1.0, "U": 1.0, "L": 1.0}
    if constants:
        c.update(constants)
    return ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _NS_SM}],
        law_implied_audits=True,
        constants=c,
    )


def _kov_state(n: int = 48):
    x = jnp.linspace(0.0, 1.0, n)
    y = jnp.linspace(0.0, 2.0 * jnp.pi, n)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    fields = kovasznay_2d(X, Y, re=40.0)
    state = ns_state_from_analytic(fields, X, Y)
    state.update({"rho": 1.0, "U": 1.0, "L": 1.0, "mu": 1.0 / 40.0, "re": 40.0})
    return state, X, Y


def _audit_mu(state, constants=None):
    eng = _ns_engine(constants)
    eng.compute_residuals(state, weighted_norms=False)
    return audit(eng.log), eng


def test_default_implied_scoring_options_match_2_1():
    opts = parse_implied_scoring_options({})
    assert opts["implied_metric"] == DEFAULT_IMPLIED_METRIC == "worst_point"
    assert opts["implied_noise_normalization"] is False
    assert opts["implied_percentile"] == 95.0


def test_paper_path_a_reference_constitutive_score_under_defaults():
    """Path A published 32×32×32 reference: default worst-point formula matches stored scores."""
    from moju.monitor.auditor import admissibility_level

    cat_path = REF / "category_scores.np.npy"
    if not cat_path.is_file():
        cat_path = REF / "category_scores.npy"
    if not cat_path.is_file() or not (REF / "training_admissibility.npy").is_file():
        pytest.skip("Path A reference bundle missing")

    ref_cat = np.load(cat_path, allow_pickle=True).item()
    ref_adm = np.load(REF / "training_admissibility.npy", allow_pickle=True).item()
    const_key = (
        "constitutive/thermal_diffusivity/law_fourier_conduction/implied_delta"
    )
    assert const_key in ref_cat
    ref_score = float(ref_cat[const_key])
    r_max = float(ref_adm["constitutive worst-point r_max"][-1])
    scale_k = 1.0e-2
    score = 1.0 / (1.0 + r_max / scale_k)
    assert score == pytest.approx(ref_score, rel=0, abs=1e-9)
    assert float(ref_adm["constitutive admissibility"][-1]) == pytest.approx(
        ref_score, rel=0, abs=1e-12
    )
    assert admissibility_level(ref_score).startswith("High")
    # Defaults still select worst-point aggregation (2.1 behaviour).
    assert parse_implied_scoring_options({})["implied_metric"] == "worst_point"
    assert parse_implied_scoring_options({})["implied_noise_normalization"] is False


def test_paper_path_b_wide2_defaults_frozen_constitutive_score():
    """Path B bundled wide2 eval state: defaults freeze constitutive score/tier."""
    if not WIDE2_ZIP.is_file():
        pytest.skip("wide2 Path B zip missing")
    with zipfile.ZipFile(WIDE2_ZIP) as zf:
        names = [n for n in zf.namelist() if n.endswith(".json") and "__MACOSX" not in n]
        with zf.open(names[0]) as f:
            raw = json.load(f)
    state = {k: jnp.asarray(v) for k, v in raw.items()}
    L, cp, h, k, rho = 0.1, 900.0, 500.0, 200.0, 2700.0
    eng = ResidualEngine(
        constants={
            "L": L,
            "cp": cp,
            "h": h,
            "k": k,
            "rho": rho,
            "alpha": Models.thermal_diffusivity(k, rho, cp),
        },
        laws=[
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
        groups=implied_group_specs_for_laws(["fourier_conduction"]),
    )
    eng.compute_residuals(state, run_mode="eval")
    report = audit(eng.log)
    pk = report["per_key"][_ALPHA_KEY]
    # Frozen golden under Moju 2.1 / 2.2 defaults (worst-point, no noise norm).
    assert pk["admissibility_score"] == pytest.approx(0.8898391009192613, rel=0, abs=1e-12)
    assert pk["admissibility_level"].startswith("Moderate")
    assert pk["admissibility_metric"] == "max"
    assert pk.get("implied_noise_normalization") is False
    assert pk.get("implied_metric") == "worst_point"
    assert float(pk["r_max_raw"]) == pytest.approx(float(pk["r_max"]), rel=0, abs=1e-15)
    assert report["per_category"]["constitutive"] == pytest.approx(
        pk["admissibility_score"], rel=0, abs=1e-15
    )


def test_defaults_match_explicit_worst_point_no_noise_on_kovasznay():
    state, _, _ = _kov_state()
    rep_def, _ = _audit_mu(state)
    rep_exp, _ = _audit_mu(
        state,
        {
            "implied_metric": "worst_point",
            "implied_noise_normalization": False,
        },
    )
    a = rep_def["per_key"][_MU_KEY]
    b = rep_exp["per_key"][_MU_KEY]
    assert a["admissibility_score"] == pytest.approx(b["admissibility_score"], abs=0.0)
    assert a["score_for_admissibility"] == pytest.approx(b["score_for_admissibility"], abs=0.0)
    assert a["admissibility_level"].startswith("High")


def test_near_wall_noise_percentile_keeps_high_while_default_drops():
    state, X, _ = _kov_state(48)
    key = jax.random.PRNGKey(1)
    lap_noise = jax.random.normal(key, state["u_laplacian"].shape)
    mask = (X < 0.12)[..., None]
    noisy = dict(state)
    noisy["u_laplacian"] = state["u_laplacian"] + 0.02 * lap_noise * mask

    rep_def, _ = _audit_mu(noisy)
    rep_p95, _ = _audit_mu(noisy, {"implied_metric": "percentile", "implied_percentile": 95})
    assert not rep_def["per_key"][_MU_KEY]["admissibility_level"].startswith("High")
    assert rep_p95["per_key"][_MU_KEY]["admissibility_level"].startswith("High")
    # Raw worst-point always retained alongside the percentile statistic.
    assert float(rep_p95["per_key"][_MU_KEY]["r_max_raw"]) == pytest.approx(
        float(rep_def["per_key"][_MU_KEY]["r_max_raw"]), rel=0, abs=1e-12
    )
    assert float(rep_p95["per_key"][_MU_KEY]["score_for_admissibility"]) < float(
        rep_p95["per_key"][_MU_KEY]["r_max_raw"]
    )


def test_genuine_half_domain_viscosity_mismatch_stays_low_under_opt_in():
    state, X, _ = _kov_state(48)
    bad = dict(state)
    bad["u_laplacian"] = state["u_laplacian"].at[X < 0.5].multiply(0.5)
    for opts in (
        None,
        {"implied_metric": "percentile"},
        {"implied_noise_normalization": True},
        {"implied_metric": "percentile", "implied_noise_normalization": True},
    ):
        rep, _ = _audit_mu(bad, opts)
        level = rep["per_key"][_MU_KEY]["admissibility_level"]
        assert level.startswith("Non-Admissible") or level.startswith("Low"), level


def test_near_gate_points_opt_in_keeps_high_on_light_noise():
    """
    Regional balance near the term-fraction gate: light noise on the weak side
    must not drop the tier under percentile scoring.
    """
    n = 64
    nu = 0.2
    u = jnp.ones((n, 2)).at[:, 1].set(0.0)
    lap = jnp.zeros((n, 2)).at[:32, 0].set(1.0).at[32:, 0].set(0.12)
    adv_x = jnp.zeros((n,)).at[:32].set(0.05).at[32:].set(0.5)
    u_grad = jnp.zeros((n, 2, 2)).at[:, 0, 0].set(adv_x)
    adv = jnp.stack([adv_x, jnp.zeros((n,))], axis=-1)
    p_grad = nu * lap - adv
    state = {
        "u": u,
        "u_t": jnp.zeros((n, 2)),
        "u_grad": u_grad,
        "p_grad": p_grad,
        "u_laplacian": lap,
        "re": 1.0 / nu,
        "rho": 1.0,
        "U": 1.0,
        "L": 1.0,
        "mu": nu,
    }
    # Light noise on the weaker half (near gate).
    key = jax.random.PRNGKey(3)
    noisy = dict(state)
    noise = 0.01 * jax.random.normal(key, lap.shape)
    noisy["u_laplacian"] = lap.at[32:].add(noise[32:])

    rep_def, _ = _audit_mu(noisy)
    rep_opt, _ = _audit_mu(
        noisy,
        {"implied_metric": "percentile", "implied_noise_normalization": True},
    )
    assert rep_opt["per_key"][_MU_KEY]["admissibility_level"].startswith("High")
    # Default may still be High on this mild case; opt-in must not be worse-tiered.
    from moju.monitor.auditor import ADM_HIGH_THRESHOLD

    assert float(rep_opt["per_key"][_MU_KEY]["admissibility_score"]) >= ADM_HIGH_THRESHOLD
    assert float(rep_opt["per_key"][_MU_KEY]["admissibility_score"]) >= float(
        rep_def["per_key"][_MU_KEY]["admissibility_score"]
    ) - 1e-9


def test_gate_boundary_noise_normalization_formula_softens_ratio_delta():
    """δ_nn = raw / (|F| + |raw|) softens Fourier-style ratio residuals."""
    F = jnp.array([1.0, 1.0, 1.0])
    raw = jnp.array([0.01, 0.1, 1.0])
    implied = F - raw
    d_raw = raw / (jnp.abs(F) + 1e-30)
    d_nn = noise_normalized_delta(pred=F, implied=implied, raw=raw, debug={})
    assert float(jnp.max(jnp.abs(d_nn))) < float(jnp.max(jnp.abs(d_raw)))
    assert float(d_nn[0]) == pytest.approx(0.01 / 1.01, rel=1e-6)


def test_report_surfaces_record_implied_scoring_metadata(tmp_path):
    from moju.monitor.monitor_log_export import export_monitor_log
    from moju.monitor.report import _implied_scoring_pdf_note, write_audit_pdf

    state, X, _ = _kov_state(32)
    key = jax.random.PRNGKey(0)
    noisy = dict(state)
    noisy["u_laplacian"] = state["u_laplacian"] + 0.02 * jax.random.normal(
        key, state["u_laplacian"].shape
    ) * (X < 0.1)[..., None]
    rep, eng = _audit_mu(
        noisy,
        {"implied_metric": "percentile", "implied_noise_normalization": True},
    )
    pk = rep["per_key"][_MU_KEY]
    assert pk["implied_metric"] == "percentile"
    assert pk["implied_noise_normalization"] is True
    assert pk["implied_percentile"] == 95.0
    assert "r_max_raw" in pk
    assert _MU_KEY in rep["implied_scoring"]

    export = export_monitor_log(eng.log, scope="audit", persist=False)
    assert export["summary"]["per_key"][_MU_KEY]["implied_metric"] == "percentile"
    assert _MU_KEY in export["summary"]["implied_scoring"]

    note = _implied_scoring_pdf_note(pk)
    assert "noise_norm" in note and "p95" in note
    assert "raw max" in note
    pytest.importorskip("reportlab")
    out = tmp_path / "r.pdf"
    write_audit_pdf(rep, str(out))
    assert out.read_bytes()[:5] == b"%PDF-"


def test_jax_torch_parity_opt_in_implied_scoring():
    torch = pytest.importorskip("torch")
    from moju.torch._engine import TorchResidualEngine

    state, X, _ = _kov_state(24)
    key = jax.random.PRNGKey(2)
    noisy = dict(state)
    noisy["u_laplacian"] = state["u_laplacian"] + 0.015 * jax.random.normal(
        key, state["u_laplacian"].shape
    ) * (X < 0.15)[..., None]
    opts = {
        "rho": 1.0,
        "U": 1.0,
        "L": 1.0,
        "implied_metric": "percentile",
        "implied_percentile": 95.0,
        "implied_noise_normalization": True,
    }
    eng_j = _ns_engine(opts)
    eng_j.compute_residuals(noisy, weighted_norms=False)
    rep_j = audit(eng_j.log)
    state_t = {
        k: torch.tensor(np.asarray(v).tolist(), dtype=torch.float64) for k, v in noisy.items()
    }
    eng_t = TorchResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _NS_SM}],
        law_implied_audits=True,
        constants=opts,
    )
    # Torch audit path uses JAX ResidualEngine with the same constants.
    rep_t = eng_t.audit(state_t)
    a = rep_j["per_key"][_MU_KEY]
    b = rep_t["per_key"][_MU_KEY]
    assert a["admissibility_metric"] == b["admissibility_metric"]
    assert a["admissibility_score"] == pytest.approx(b["admissibility_score"], rel=0, abs=1e-8)
    assert float(a["r_max_raw"]) == pytest.approx(float(b["r_max_raw"]), rel=0, abs=1e-8)


def test_reduce_percentile_below_max():
    arr = jnp.array([0.0, 0.0, 0.0, 1.0])
    assert reduce_implied_magnitude(arr, metric="worst_point") == pytest.approx(1.0)
    assert reduce_implied_magnitude(arr, metric="percentile", percentile=75) < 1.0
