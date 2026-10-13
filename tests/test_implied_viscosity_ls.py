"""Implied viscosity from momentum LSQ (NS / Stokes / Burgers) — Moju 2.1."""

from __future__ import annotations

import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from moju.monitor import ResidualEngine, audit
from moju.monitor.analytic_flows import kovasznay_2d, ns_state_from_analytic, taylor_green_2d
from moju.monitor.closure_registry import MODEL_FNS
from moju.monitor.law_implied_diagnostics import (
    DEFAULT_IMPLIED_TERM_FRACTION,
    implied_mu_momentum_navier_stokes,
)
from moju.piratio.laws import Laws
from moju.piratio.models import Models


def _ns_sm():
    return {
        "u_t": "u_t",
        "u": "u",
        "u_grad": "u_grad",
        "p_grad": "p_grad",
        "u_laplacian": "u_laplacian",
        "re": "re",
    }


def test_models_dynamic_viscosity_uses_reference_U_not_local_speed():
    mu = Models.dynamic_viscosity_from_re(1.0, 1.0, 1.0, 40.0)
    assert float(mu) == pytest.approx(1.0 / 40.0)
    mu2 = Models.dynamic_viscosity_from_re(1.0, jnp.array(2.0), 1.0, 40.0)
    assert float(mu2) == pytest.approx(2.0 / 40.0)


def test_dynamic_viscosity_deprecated_u_alias_matches_U_and_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        mu_u = Models.dynamic_viscosity_from_re(1.0, L=1.0, re=40.0, u=2.0)
    assert float(mu_u) == pytest.approx(2.0 / 40.0)
    deprec = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprec, "expected DeprecationWarning for u="
    assert "U=" in str(deprec[0].message) and "3.0" in str(deprec[0].message)

    mu_U = Models.dynamic_viscosity_from_re(1.0, U=2.0, L=1.0, re=40.0)
    assert float(mu_u) == pytest.approx(float(mu_U))

    with pytest.raises(TypeError, match="only one"):
        Models.dynamic_viscosity_from_re(1.0, U=1.0, L=1.0, re=40.0, u=2.0)


def test_model_fns_binds_U_not_deprecated_u():
    _fn, args = MODEL_FNS["dynamic_viscosity_from_re"]
    assert args == ["rho", "U", "L", "re"]


def test_legacy_local_speed_bias_on_kovasznay_reproduced():
    """Document old β·ρ·|u|·L bias vs true μ while law residual is ~machine eps."""
    nx = ny = 32
    x = jnp.linspace(0.0, 1.0, nx)
    y = jnp.linspace(0.0, 2.0 * jnp.pi, ny)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    fields = kovasznay_2d(X, Y, re=40.0)
    law_r = Laws.momentum_navier_stokes(
        fields["u_t"],
        fields["u"],
        fields["u_grad"],
        fields["p_grad"],
        fields["u_laplacian"],
        fields["re"],
    )
    assert float(jnp.max(jnp.abs(law_r))) < 1e-5

    adv = jnp.einsum("...ij,...j->...i", fields["u_grad"], fields["u"])
    rhs = fields["u_t"] + adv + fields["p_grad"]
    num = jnp.sum(rhs * fields["u_laplacian"], axis=-1)
    den = jnp.sum(fields["u_laplacian"] ** 2, axis=-1)
    beta = num / (den + 1e-30)
    u_mag = jnp.sqrt(jnp.sum(fields["u"] ** 2, axis=-1))
    mu_legacy = beta * 1.0 * u_mag * 1.0
    mu_true = 1.0 / 40.0
    rel = jnp.abs(mu_legacy - mu_true) / mu_true
    assert float(jnp.max(rel)) > 0.5
    assert float(jnp.mean(rel[u_mag < 0.5])) > 0.1


def test_kovasznay_implied_mu_matches_true_with_reference_U():
    nx = ny = 48
    x = jnp.linspace(0.0, 1.0, nx)
    y = jnp.linspace(0.0, 2.0 * jnp.pi, ny)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    fields = kovasznay_2d(X, Y, re=40.0)
    state = ns_state_from_analytic(fields, X, Y)
    state.update({"rho": 1.0, "U": 1.0, "L": 1.0, "mu": 1.0 / 40.0})
    eng = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _ns_sm()}],
        law_implied_audits=True,
        constants={"rho": 1.0, "U": 1.0, "L": 1.0},
    )
    r = eng.compute_residuals(state)
    key = "dynamic_viscosity_from_re/law_momentum_navier_stokes/implied_delta"
    d = r["constitutive"][key]
    assert float(jnp.nanmax(jnp.abs(d))) < 1e-4
    dbg = r["closure_debug"]["dynamic_viscosity_from_re/law_momentum_navier_stokes"]
    assert dbg["identifiable"] is True
    assert float(dbg["excluded_fraction"]) < 0.5
    assert float(jnp.nanmean(jnp.abs(dbg["implied"] - 1.0 / 40.0))) < 1e-4
    assert float(jnp.abs(dbg["mu_tilde_global"] - 1.0 / 40.0)) < 1e-4
    assert float(jnp.abs(dbg["re_tilde_global"] - 40.0)) < 1e-3


def _kovasznay_exact_implied_rel_error(n: int):
    """Return (max_rel, mean_rel, kept_fraction) vs exact ν on gate-kept points."""
    re = 40.0
    nu = 1.0 / re
    x = jnp.linspace(0.0, 1.0, n)
    y = jnp.linspace(0.0, 2.0 * jnp.pi, n)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    fields = kovasznay_2d(X, Y, re=re)
    state = ns_state_from_analytic(fields, X, Y)
    state.update({"rho": 1.0, "U": 1.0, "L": 1.0, "mu": nu})
    eng = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _ns_sm()}],
        law_implied_audits=True,
        constants={"rho": 1.0, "U": 1.0, "L": 1.0},
    )
    r = eng.compute_residuals(state)
    dbg = r["closure_debug"]["dynamic_viscosity_from_re/law_momentum_navier_stokes"]
    imp = jnp.asarray(dbg["implied"])
    kept = jnp.isfinite(imp)
    rel = jnp.abs(imp[kept] - nu) / nu
    return float(jnp.max(rel)), float(jnp.mean(rel)), float(jnp.mean(kept)), dbg


def test_kovasznay_implied_nu_relative_error_machine_eps_two_resolutions():
    """
    Exact NS + exact derivatives: implied ν recovers exact ν on kept points
    within float tolerance; kept fraction is 100% at default gate settings.
    """
    for n in (16, 64):
        max_rel, mean_rel, kept_frac, dbg = _kovasznay_exact_implied_rel_error(n)
        assert dbg["identifiable"] is True
        assert kept_frac == pytest.approx(1.0)
        assert float(dbg["excluded_fraction"]) == pytest.approx(0.0)
        # Measured ~3e-15; allow 1e-12 for platform float noise.
        assert max_rel < 1e-12
        assert mean_rel < 1e-12


def test_kovasznay_fd_implied_nu_error_decreases_with_resolution():
    """Path B FD4 fill: mean relative error on kept points decreases n=32 → n=64."""
    from moju.monitor.path_b_derivatives import PathBGridConfig

    re = 40.0
    nu = 1.0 / re

    def mean_rel(n: int) -> float:
        x = jnp.linspace(0.0, 1.0, n)
        y = jnp.linspace(0.0, 2.0 * jnp.pi, n)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        fields = kovasznay_2d(X, Y, re=re)
        state = {
            "x": X,
            "y": Y,
            "u": fields["u"],
            "p": fields["p"],
            "re": fields["re"],
            "u_t": fields["u_t"],
            "rho": 1.0,
            "U": 1.0,
            "L": 1.0,
            "mu": nu,
        }
        grid = PathBGridConfig(
            layout="meshgrid", spatial_dimension=2, steady=True, fd_order=4
        )
        eng = ResidualEngine(
            laws=[{"name": "momentum_navier_stokes", "state_map": _ns_sm()}],
            law_implied_audits=True,
            constants={"rho": 1.0, "U": 1.0, "L": 1.0},
        )
        r = eng.compute_residuals(
            state, auto_path_b_derivatives=grid, fill_law_fd=True
        )
        dbg = r["closure_debug"]["dynamic_viscosity_from_re/law_momentum_navier_stokes"]
        assert dbg["identifiable"] is True
        imp = jnp.asarray(dbg["implied"])
        kept = jnp.isfinite(imp)
        assert float(jnp.mean(kept)) > 0.9
        return float(jnp.mean(jnp.abs(imp[kept] - nu) / nu))

    e32 = mean_rel(32)
    e64 = mean_rel(64)
    # Measured ~0.097 → ~0.009; require clear decrease.
    assert e64 < e32 * 0.5
    assert e64 < 0.05


def test_taylor_green_implied_mu_matches_true():
    # Re=10 keeps |ν∇²u|/max_term above the default 0.1 gate on most of the grid.
    n = 32
    xs = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = taylor_green_2d(x, y, 0.1, re=10.0)
    state = ns_state_from_analytic(fields, x, y, t=0.1)
    state.update({"rho": 1.0, "U": 1.0, "L": 1.0, "mu": 0.1})
    eng = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _ns_sm()}],
        law_implied_audits=True,
        constants={"rho": 1.0, "U": 1.0, "L": 1.0},
    )
    r = eng.compute_residuals(state)
    key = "dynamic_viscosity_from_re/law_momentum_navier_stokes/implied_delta"
    assert float(jnp.nanmax(jnp.abs(r["constitutive"][key]))) < 1e-4
    law_r = Laws.momentum_navier_stokes(
        state["u_t"], state["u"], state["u_grad"], state["p_grad"], state["u_laplacian"], state["re"]
    )
    assert float(jnp.max(jnp.abs(law_r))) < 1e-5


def test_regional_viscous_dominant_vs_negligible_gate():
    """
    Exact NS balance everywhere: left half viscous-dominant, right half inertia-dominant.

    Gate keeps dominant points (ν̃ ≈ true); excludes negligible region; logs excluded_fraction.
    """
    n = 64
    nu = 0.2
    re = 1.0 / nu
    # u = (1, 0) ⇒ (u·∇)u = first Jacobian column
    u = jnp.ones((n, 2)).at[:, 1].set(0.0)
    lap = jnp.zeros((n, 2)).at[:32, 0].set(1.0).at[32:, 0].set(0.05)
    adv_x = jnp.zeros((n,)).at[:32].set(0.05).at[32:].set(5.0)
    u_grad = jnp.zeros((n, 2, 2)).at[:, 0, 0].set(adv_x)
    adv = jnp.stack([adv_x, jnp.zeros((n,))], axis=-1)
    # u_t = 0 ⇒ adv + p_grad = nu * lap  (machine-exact residual)
    p_grad = nu * lap - adv
    u_t = jnp.zeros((n, 2))
    state = {
        "u": u,
        "u_t": u_t,
        "u_grad": u_grad,
        "p_grad": p_grad,
        "u_laplacian": lap,
        "re": re,
        "rho": 1.0,
        "U": 1.0,
        "L": 1.0,
        "mu": nu,  # ρ U L / Re = nu when ρ=U=L=1
    }
    eng = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _ns_sm()}],
        law_implied_audits=True,
        constants={
            "rho": 1.0,
            "U": 1.0,
            "L": 1.0,
            "implied_term_fraction": DEFAULT_IMPLIED_TERM_FRACTION,
        },
    )
    r = eng.compute_residuals(state)
    dbg = r["closure_debug"]["dynamic_viscosity_from_re/law_momentum_navier_stokes"]
    tf = jnp.asarray(dbg["term_fraction"])
    assert float(jnp.min(tf[:32])) >= DEFAULT_IMPLIED_TERM_FRACTION
    assert float(jnp.max(tf[32:])) < DEFAULT_IMPLIED_TERM_FRACTION
    assert dbg["identifiable"] is True
    assert 0.4 <= float(dbg["excluded_fraction"]) <= 0.6
    # Dominant region recovers true ν*; negligible region is NaN-masked.
    imp = jnp.asarray(dbg["implied"])
    assert float(jnp.nanmax(jnp.abs(imp[:32] - nu))) < 1e-5
    assert bool(jnp.all(jnp.isnan(imp[32:])))
    assert float(jnp.abs(dbg["mu_tilde_global"] - nu)) < 1e-5
    assert "excluded_fraction" in eng.log[-1]["implied_identifiability"][
        "dynamic_viscosity_from_re/law_momentum_navier_stokes"
    ]


def _tg_re100_not_identifiable_engine():
    n = 32
    xs = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    x, y = jnp.meshgrid(xs, xs, indexing="ij")
    fields = taylor_green_2d(x, y, 0.1, re=100.0)
    state = ns_state_from_analytic(fields, x, y, t=0.1)
    state.update({"rho": 1.0, "U": 1.0, "L": 1.0, "mu": 0.01})
    eng = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _ns_sm()}],
        law_implied_audits=True,
        constants={"rho": 1.0, "U": 1.0, "L": 1.0},
    )
    eng.compute_residuals(state, weighted_norms=False)
    return eng


_NI_KEY = "constitutive/dynamic_viscosity_from_re/law_momentum_navier_stokes/implied_delta"
_NI_BN = "dynamic_viscosity_from_re/law_momentum_navier_stokes"


def test_not_identifiable_does_not_zero_overall_when_law_is_high():
    """Not-identifiable implied μ must not drag overall to Non-Admissible."""
    eng = _tg_re100_not_identifiable_engine()
    report = audit(eng.log)
    pk = report["per_key"][_NI_KEY]
    assert pk["admissibility_level"] == "Not identifiable"
    assert report["per_key"]["laws/momentum_navier_stokes"]["admissibility_level"].startswith("High")
    assert report["overall_admissibility_level"].startswith("High")


def test_not_identifiable_visible_in_python_report_object():
    eng = _tg_re100_not_identifiable_engine()
    report = audit(eng.log)
    pk = report["per_key"][_NI_KEY]
    assert pk["admissibility_level"] == "Not identifiable"
    assert pk["identifiability_status"] == "not_identifiable"
    assert float(pk["excluded_fraction"]) > 0.5
    assert "could not be evaluated" in str(pk["identifiability_reason"]).lower()
    assert "passed" in str(pk["identifiability_reason"]).lower()
    id_map = report["implied_identifiability"][_NI_BN]
    assert id_map["status"] == "not_identifiable"
    assert float(id_map["excluded_fraction"]) == pytest.approx(float(pk["excluded_fraction"]))
    assert "constitutive" not in report["per_category"] or _NI_KEY not in (
        report.get("per_category") or {}
    )


def test_not_identifiable_visible_in_json_export():
    from moju.monitor.monitor_log_export import export_monitor_log, monitor_log_export_to_jsonable

    eng = _tg_re100_not_identifiable_engine()
    audit(eng.log)  # enrich step metrics / levels
    export = export_monitor_log(eng.log, scope="audit", persist=False)
    summary = export["summary"]
    pk = summary["per_key"][_NI_KEY]
    assert pk["admissibility_level"] == "Not identifiable"
    assert float(pk["excluded_fraction"]) > 0.5
    assert "identifiability_reason" in pk
    assert summary["implied_identifiability"][_NI_BN]["status"] == "not_identifiable"
    ni_worst = [r for r in summary["worst_keys_rows"] if r.get("key") == _NI_KEY]
    assert ni_worst, "Not identifiable row must appear in worst_keys_rows"
    assert ni_worst[0]["admissibility_level"] == "Not identifiable"
    blob = monitor_log_export_to_jsonable(export)
    assert blob["summary"]["per_key"][_NI_KEY]["admissibility_level"] == "Not identifiable"


def test_not_identifiable_visible_in_pdf_report(tmp_path, monkeypatch):
    pytest.importorskip("reportlab")
    import moju.monitor.report as report_mod
    from moju.monitor.report import _not_identifiable_pdf_notes, write_audit_pdf

    eng = _tg_re100_not_identifiable_engine()
    report = audit(eng.log)
    notes = _not_identifiable_pdf_notes(report)
    assert notes, "PDF notes must include Not identifiable implied rows"
    assert any("Not identifiable" in n for n in notes)
    assert any("could not be evaluated" in n for n in notes)
    assert any("96" in n or "excluded fraction" in n for n in notes)

    # Capture Paragraph text fed into the PDF story (glyph encoding hides ASCII in the file).
    captured: list[str] = []
    RealParagraph = report_mod.Paragraph

    def _capturing_paragraph(text, *args, **kwargs):
        captured.append(str(text))
        return RealParagraph(text, *args, **kwargs)

    monkeypatch.setattr(report_mod, "Paragraph", _capturing_paragraph)
    out = tmp_path / "ni.pdf"
    write_audit_pdf(report, str(out))
    assert out.read_bytes()[:5] == b"%PDF-"
    joined = "\n".join(captured)
    assert "Not identifiable" in joined
    assert "Implied property identifiability" in joined
    assert "could not be evaluated" in joined


def test_not_identifiable_visible_in_plotly_dashboard():
    pytest.importorskip("plotly")
    from moju.monitor.auditor import build_monitor_visualize_bundle, visualize
    from moju.monitor.visualize_components import build_worst_keys_table_card

    eng = _tg_re100_not_identifiable_engine()
    audit(eng.log)
    bundle = build_monitor_visualize_bundle(eng.log, keys=None, r_ref=None, mode="eval")
    assert bundle is not None
    ni_rows = [r for r in (bundle.get("worst_keys_rows") or []) if r.get("key") == _NI_KEY]
    assert ni_rows, "Plotly bundle must list Not identifiable in worst_keys_rows"
    assert ni_rows[0]["admissibility_level"] == "Not identifiable"
    card = build_worst_keys_table_card(bundle)
    cells = card.data[0].cells.values
    flat = " ".join(str(x) for col in cells for x in col)
    assert "Not identifiable" in flat
    out = visualize(
        eng.log, backend="plotly", dashboard_mode="single-figure", visualize_layout="split"
    )
    assert isinstance(out, dict) and "worst_keys" in out
    wk_cells = out["worst_keys"].data[0].cells.values
    wk_flat = " ".join(str(x) for col in wk_cells for x in col)
    assert "Not identifiable" in wk_flat
    # Main figure summary should mention not-evaluable constitutive row
    fig = out["monitor"]
    layout_ann = list(getattr(fig.layout, "annotations", None) or [])
    joined = " ".join(str(getattr(a, "text", a)) for a in layout_ann)
    assert "not evaluable" in joined.lower() or "Not identifiable" in joined


def test_fit_quality_threshold_separately_configurable():
    """implied_fit_quality_fraction can differ from implied_term_fraction."""
    n = 32
    nu = 0.2
    u = jnp.ones((n, 2)).at[:, 1].set(0.0)
    lap = jnp.zeros((n, 2)).at[:, 0].set(1.0)
    adv_x = jnp.full((n,), 0.05)
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
    eng = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _ns_sm()}],
        law_implied_audits=True,
        constants={
            "rho": 1.0,
            "U": 1.0,
            "L": 1.0,
            "implied_term_fraction": 0.1,
            "implied_fit_quality_fraction": 0.99,
        },
    )
    r = eng.compute_residuals(state)
    dbg = r["closure_debug"]["dynamic_viscosity_from_re/law_momentum_navier_stokes"]
    assert float(dbg["fit_quality_threshold"]) == pytest.approx(0.99)
    assert float(dbg["conditioning_threshold"]) == pytest.approx(0.1)


def test_all_negligible_viscous_marked_not_identifiable():
    """Inertia-dominated exact balance on the whole grid → not identifiable."""
    n = 32
    nu = 0.01
    u = jnp.ones((n, 2)).at[:, 1].set(0.0)
    lap = jnp.zeros((n, 2)).at[:, 0].set(0.05)
    adv_x = jnp.full((n,), 5.0)
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
    eng = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _ns_sm()}],
        law_implied_audits=True,
        constants={"rho": 1.0, "U": 1.0, "L": 1.0},
    )
    eng.compute_residuals(state)
    dbg = eng.log[-1]["implied_identifiability"][
        "dynamic_viscosity_from_re/law_momentum_navier_stokes"
    ]
    # Also pull closure_debug from last compute via residuals path
    r = eng.compute_residuals(state)
    cd = r["closure_debug"]["dynamic_viscosity_from_re/law_momentum_navier_stokes"]
    assert cd["identifiable"] is False
    assert cd["status"] == "not_identifiable"
    assert float(cd["excluded_fraction"]) >= 0.9
    report = audit(eng.log)
    pk = report["per_key"][
        "constitutive/dynamic_viscosity_from_re/law_momentum_navier_stokes/implied_delta"
    ]
    assert pk["admissibility_level"] == "Not identifiable"
    assert dbg["identifiable"] is False


def test_stokes_implied_mu_with_reference_U():
    state = {
        "p_grad": jnp.array([1.0, 0.0]),
        "u_lap": jnp.array([2.0, 0.0]),
        "re": 2.0,
        "rho": 1.0,
        "U": 1.0,
        "L": 1.0,
        "mu": 0.5,
    }
    eng = ResidualEngine(
        laws=[{
            "name": "stokes_flow",
            "state_map": {"p_grad": "p_grad", "u_laplacian": "u_lap", "re": "re"},
        }],
        law_implied_audits=True,
    )
    r = eng.compute_residuals(state)
    key = "dynamic_viscosity_from_re/law_stokes_flow/implied_delta"
    assert float(jnp.nanmax(jnp.abs(r["constitutive"][key]))) < 1e-5


def test_burgers_implied_nu_still_matches_and_reports_re_tilde():
    re, U, L = 2.0, 1.0, 1.0
    nu_cat = U * L / re
    u_lap = jnp.array([4.0, 2.0])
    state = {
        "u": jnp.array([1.0, 0.5]),
        "u_grad": jnp.zeros((2, 2)),
        "u_lap": u_lap,
        "u_t": nu_cat * u_lap,
        "re": re,
        "U": U,
        "L": L,
        "nu": nu_cat,
    }
    eng = ResidualEngine(
        laws=[{
            "name": "burgers_equation",
            "state_map": {
                "u_t": "u_t", "u": "u", "u_grad": "u_grad",
                "u_laplacian": "u_lap", "re": "re", "U": "U", "L": "L",
            },
        }],
        law_implied_audits=True,
    )
    r = eng.compute_residuals(state)
    key = "kinematic_viscosity_from_re/law_burgers_equation/implied_delta"
    assert float(jnp.nanmax(jnp.abs(r["constitutive"][key]))) < 1e-5
    dbg = r["closure_debug"]["kinematic_viscosity_from_re/law_burgers_equation"]
    assert float(jnp.abs(dbg["re_tilde_global"] - re)) < 1e-4


def test_jax_torch_parity_ns_implied_mu():
    torch = pytest.importorskip("torch")
    from moju.torch._engine import TorchResidualEngine

    nx = ny = 24
    x = jnp.linspace(0.0, 1.0, nx)
    y = jnp.linspace(0.0, 2.0 * jnp.pi, ny)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    fields = kovasznay_2d(X, Y, re=40.0)
    state = ns_state_from_analytic(fields, X, Y)
    state.update({"rho": 1.0, "U": 1.0, "L": 1.0, "mu": 1.0 / 40.0})
    laws = [{"name": "momentum_navier_stokes", "state_map": _ns_sm()}]
    const = {"rho": 1.0, "U": 1.0, "L": 1.0}
    eng_j = ResidualEngine(laws=laws, law_implied_audits=True, constants=const)
    r_j = eng_j.compute_residuals(state)
    key = "dynamic_viscosity_from_re/law_momentum_navier_stokes/implied_delta"
    state_t = {k: torch.tensor(jnp.asarray(v).tolist(), dtype=torch.float64) for k, v in state.items()}
    eng_t = TorchResidualEngine(laws=laws, law_implied_audits=True, constants=const)
    r_t = eng_t.compute_residuals_torch(state_t)
    dj = jnp.asarray(r_j["constitutive"][key])
    dt = jnp.asarray(r_t["constitutive"][key].detach().numpy())
    both = jnp.isfinite(dj) & jnp.isfinite(dt)
    assert float(jnp.mean(both)) > 0.5
    assert float(jnp.max(jnp.abs(dj[both] - dt[both]))) < 1e-5


def test_implied_fn_direct_matches_true_mu_on_kov():
    nx = ny = 32
    x = jnp.linspace(0.0, 1.0, nx)
    y = jnp.linspace(0.0, 2.0 * jnp.pi, ny)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    fields = kovasznay_2d(X, Y, re=40.0)
    state = {**fields, "rho": 1.0, "U": 1.0, "L": 1.0}
    fn = implied_mu_momentum_navier_stokes(_ns_sm())
    out = fn(state, {})
    assert out is not None
    assert out.meta["identifiable"] is True
    assert float(jnp.nanmax(jnp.abs(out.value - 1.0 / 40.0))) < 1e-4
