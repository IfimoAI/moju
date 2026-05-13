"""Tests for ResidualEngine, build_loss, audit, visualize, constitutive closures."""

import pytest
import jax
import jax.numpy as jnp
from moju.monitor import (
    AuditSpec,
    MonitorConfig,
    ResidualEngine,
    admissibility_level,
    audit,
    build_loss,
    list_constitutive_models,
    list_scaling_closure_ids,
    visualize,
)
from moju.monitor.auditor import (
    ADM_HIGH_THRESHOLD,
    DEFAULT_VISUALIZE_TITLE_EVAL,
    DEFAULT_VISUALIZE_TITLE_TEST,
    DEFAULT_VISUALIZE_TITLE_TRAINING,
)
from moju.monitor.closure_registry import (
    apply_closure_discrepancy_normalize,
    MODEL_FNS,
    compute_implied_delta,
)
from moju.piratio.models import Models


class TestAdmissibilityLevel:
    def test_four_levels(self):
        assert admissibility_level(0.0) == "Non-Admissible"
        assert admissibility_level(0.49) == "Non-Admissible"
        assert admissibility_level(0.5) == "Low Admissibility"
        assert admissibility_level(0.74) == "Low Admissibility"
        assert admissibility_level(0.75) == "Moderate Admissibility"
        assert admissibility_level(0.95) == "Moderate Admissibility"
        assert admissibility_level(0.9500001) == "High Admissibility"
        assert admissibility_level(1.0) == "High Admissibility"

    def test_non_finite_unknown(self):
        assert admissibility_level(float("nan")) == "Unknown"
        assert admissibility_level(float("inf")) == "Unknown"


class TestNanTolerantAuditMetrics:
    def test_compute_log_step_metrics_category_zero_if_any_key_nan(self):
        import math

        from moju.monitor.auditor import _compute_log_step_metrics

        log = [
            {
                "index": 0,
                "rms": {
                    "constitutive/a/implied_delta": 1.0,
                    "constitutive/b/implied_delta": float("nan"),
                },
                "scale": {
                    "constitutive/a/implied_delta": 1.0,
                    "constitutive/b/implied_delta": 1.0,
                },
            }
        ]
        m = _compute_log_step_metrics(log)
        assert m[0]["category_admissibility_score"]["constitutive"] == 0.0
        assert "laws" not in m[0]["category_admissibility_score"]
        assert m[0]["overall_admissibility_score"] == 0.0

    def test_compute_log_step_metrics_all_nan_category_scores_zero_overall_zero(self):
        from moju.monitor.auditor import _compute_log_step_metrics

        log = [
            {
                "index": 0,
                "rms": {"constitutive/x/implied_delta": float("nan")},
                "scale": {"constitutive/x/implied_delta": 1.0},
            }
        ]
        m = _compute_log_step_metrics(log)
        assert m[0]["category_admissibility_score"]["constitutive"] == 0.0
        assert m[0]["overall_admissibility_score"] == 0.0

        rep = audit(log)
        assert rep["per_category"]["constitutive"] == 0.0
        assert rep["overall_admissibility_score"] == 0.0
        assert rep["overall_admissibility_level"] == "Non-Admissible"

    def test_training_overall_uses_only_laws_and_constitutive(self):
        from moju.monitor.auditor import _compute_log_step_metrics

        log = [
            {
                "run_mode": "training",
                "rms": {
                    "laws/laplace_equation": 0.01,
                    "constitutive/x/implied_delta": 0.5,
                    "scaling/pe/ref_delta": 5.0,
                },
                "scale": {
                    "laws/laplace_equation": 1.0,
                    "constitutive/x/implied_delta": 1.0,
                    "scaling/pe/ref_delta": 1.0,
                },
            }
        ]
        m = _compute_log_step_metrics(log)
        cl = m[0]["category_admissibility_score"]["laws"]
        cc = m[0]["category_admissibility_score"]["constitutive"]
        expected = min(cl, cc)
        assert abs(m[0]["overall_admissibility_score"] - expected) < 1e-9
        assert expected == cc

    def test_compute_log_step_metrics_eval_rolls_up_available_categories(self):
        import math

        from moju.monitor.auditor import _compute_log_step_metrics

        log_eval = [
            {
                "run_mode": "eval",
                "rms": {
                    "laws/a": 0.1,
                    "constitutive/b/implied_delta": 0.1,
                    "scaling/pe/ref_delta": 5.0,
                },
                "scale": {
                    "laws/a": 1.0,
                    "constitutive/b/implied_delta": 1.0,
                    "scaling/pe/ref_delta": 1.0,
                },
            }
        ]
        m_ev = _compute_log_step_metrics(log_eval)
        ce = m_ev[0]["category_admissibility_score"]
        expected_eval = min(float(ce["laws"]), float(ce["constitutive"]), float(ce["scaling"]))
        assert abs(m_ev[0]["overall_admissibility_score"] - expected_eval) < 1e-9

        log_eval_no_scaling = [
            {
                "run_mode": "eval",
                "rms": {
                    "laws/a": 0.1,
                    "constitutive/b/implied_delta": 0.1,
                },
                "scale": {
                    "laws/a": 1.0,
                    "constitutive/b/implied_delta": 1.0,
                },
            }
        ]
        m_ev2 = _compute_log_step_metrics(log_eval_no_scaling)
        ce2 = m_ev2[0]["category_admissibility_score"]
        expected_eval2 = min(float(ce2["laws"]), float(ce2["constitutive"]))
        assert abs(m_ev2[0]["overall_admissibility_score"] - expected_eval2) < 1e-9

        log_legacy = [
            {
                "rms": {
                    "laws/a": 0.1,
                    "constitutive/b/implied_delta": 0.1,
                    "scaling/pe/ref_delta": 5.0,
                },
                "scale": {
                    "laws/a": 1.0,
                    "constitutive/b/implied_delta": 1.0,
                    "scaling/pe/ref_delta": 1.0,
                },
            }
        ]
        m_leg = _compute_log_step_metrics(log_legacy)
        assert math.isfinite(m_leg[0]["overall_admissibility_score"])
        ce_leg = m_leg[0]["category_admissibility_score"]
        assert m_leg[0]["overall_admissibility_score"] == min(float(v) for v in ce_leg.values())

    def test_overall_admissibility_is_capped_by_weak_category(self):
        from moju.monitor.auditor import _compute_log_step_metrics, admissibility_level

        log = [
            {
                "run_mode": "eval",
                "rms": {
                    "laws/a": 0.01,
                    "constitutive/b/implied_delta": 0.01,
                    "data/T": 2.0,
                },
                "scale": {
                    "laws/a": 1.0,
                    "constitutive/b/implied_delta": 1.0,
                    "data/T": 1.0,
                },
            }
        ]
        m = _compute_log_step_metrics(log)
        cats = m[0]["category_admissibility_score"]
        assert cats["laws"] > 0.95
        assert cats["constitutive"] > 0.95
        assert cats["data"] < 0.5
        assert m[0]["overall_admissibility_score"] == cats["data"]
        assert admissibility_level(m[0]["overall_admissibility_score"]) == "Non-Admissible"

    def test_rms_scalar_uses_nanmean(self):
        import math

        from moju.monitor.auditor import _rms_scalar

        x = jnp.array([1.0, jnp.nan, 3.0])
        r = float(_rms_scalar(x))
        assert math.isclose(r, math.sqrt(5.0), rel_tol=1e-5)


class TestResidualEngineResidualDict:
    def test_laws_only_when_no_audits(self, rtol, atol):
        core = ResidualEngine(
            constants={"L": 0.1},
            laws=[{"name": "mass_incompressible", "state_map": {"u_grad": "u_grad"}}],
        )
        state_pred = {"u_grad": jnp.array([[0.0, 1.0], [-1.0, 0.0]])}
        residuals = core.compute_residuals(state_pred)
        assert "laws" in residuals
        assert jnp.allclose(residuals["laws"]["mass_incompressible"], 0.0, rtol=rtol, atol=atol)
        assert "constitutive" not in residuals
        assert "scaling" not in residuals
        assert "data" not in residuals

    def test_log_appends_rms_per_key(self):
        core = ResidualEngine(
            laws=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_laplacian"}}],
        )
        state_pred = {"phi_laplacian": jnp.array(1.0)}
        core.compute_residuals(state_pred)
        core.compute_residuals(state_pred)
        assert len(core.log) == 2
        assert "laws/laplace_equation" in core.log[0]["rms"]

    def test_coord_snapshot_logged_from_state_coords(self):
        core = ResidualEngine(
            laws=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_laplacian"}}],
        )
        x = jnp.linspace(0.0, 1.0, 5)
        state_pred = {"phi_laplacian": jnp.array(1.0), "x": x}
        core.compute_residuals(state_pred)
        snap = core.log[-1].get("coord_snapshot") or {}
        assert "x" in snap
        assert len(snap["x"]) == 5
        assert all(isinstance(v, float) for v in snap["x"])

    def test_last_residuals_set_after_compute(self):
        core = ResidualEngine(
            laws=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_laplacian"}}],
        )
        state_pred = {"phi_laplacian": jnp.array(1.0)}
        r = core.compute_residuals(state_pred)
        assert core.last_residuals is r
        core.clear_log()
        assert core.last_residuals is None

    def test_clear_log_resets_entries_and_index(self):
        core = ResidualEngine(
            laws=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_laplacian"}}],
        )
        state_pred = {"phi_laplacian": jnp.array(1.0)}
        core.compute_residuals(state_pred)
        core.compute_residuals(state_pred)
        assert len(core.log) == 2
        core.clear_log()
        assert len(core.log) == 0
        core.compute_residuals(state_pred)
        assert len(core.log) == 1
        assert core.log[0].get("index") == 0

    def test_constitutive_sutherland_ref_delta_nonzero(self, rtol, atol):
        core = ResidualEngine(
            constants={"mu0": 1.8e-5, "T0": 273.0, "S": 110.4},
            laws=[],
            constitutive_audit=[
                {
                    "name": "sutherland_mu",
                    "output_key": "mu",
                    "state_map": {"T": "T", "mu0": "mu0", "T0": "T0", "S": "S"},
                }
            ],
        )
        T_pred = 300.0
        T_ref = 280.0
        mu_pred = 1.8e-5 * (T_pred / 273) ** 1.5 * (273 + 110.4) / (T_pred + 110.4)
        mu_ref = 1.8e-5 * (T_ref / 273) ** 1.5 * (273 + 110.4) / (T_ref + 110.4)
        state_pred = {"mu": mu_pred, "T": T_pred}
        state_ref = {"mu": mu_ref, "T": T_ref}
        residuals = core.compute_residuals(
            state_pred, state_ref=state_ref, run_mode="eval"
        )
        assert "constitutive" in residuals
        assert "sutherland_mu/ref_delta" in residuals["constitutive"]
        assert abs(float(residuals["constitutive"]["sutherland_mu/ref_delta"])) > 0.0

    def test_state_ref_adds_data_residual(self, rtol, atol):
        core = ResidualEngine(
            laws=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_laplacian"}}],
        )
        state_pred = {"phi_laplacian": jnp.array(0.5)}
        state_ref = {"phi_laplacian": jnp.array(0.0)}
        residuals = core.compute_residuals(
            state_pred, state_ref=state_ref, run_mode="eval"
        )
        assert "data" in residuals
        assert jnp.allclose(residuals["data"]["phi_laplacian"], -0.5, rtol=rtol, atol=atol)

    def test_state_ref_ignored_in_training_mode(self, rtol, atol):
        core = ResidualEngine(
            laws=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_laplacian"}}],
        )
        state_pred = {"phi_laplacian": jnp.array(0.5)}
        state_ref = {"phi_laplacian": jnp.array(0.0)}
        residuals = core.compute_residuals(state_pred, state_ref=state_ref, run_mode="training")
        assert "data" not in residuals

class TestBuildLoss:
    def test_cascaded_loss_scalar(self, rtol, atol):
        residual_dict = {
            "laws": {
                "mass_incompressible": jnp.array(0.0),
                "laplace_equation": jnp.array(0.0),
            }
        }
        loss = build_loss(residual_dict)
        assert jnp.allclose(loss, 0.0, rtol=rtol, atol=atol)

    def test_cascaded_loss_nonzero(self, rtol, atol):
        residual_dict = {"laws": {"laplace_equation": jnp.array(3.0)}}
        loss = build_loss(residual_dict)
        assert jnp.allclose(loss, 3.0, rtol=rtol, atol=atol)

    def test_build_loss_law_weights(self, rtol, atol):
        residual_dict = {"laws": {"a": jnp.array(1.0), "b": jnp.array(0.0)}}
        loss = build_loss(residual_dict, law_weights={"a": 1.0, "b": 0.0})
        assert jnp.allclose(loss, 1.0, rtol=rtol, atol=atol)

    def test_build_loss_differentiable(self):
        def loss_fn(phi_laplacian):
            return build_loss({"laws": {"laplace_equation": phi_laplacian}})

        grad = jax.grad(loss_fn)(jnp.array(2.0))
        assert grad is not None


class TestBuildLossBatch:
    def test_batch_law_residuals(self, rtol, atol):
        residual_dict = {"laws": {"mass_incompressible": jnp.zeros((5,))}}
        loss = build_loss(residual_dict)
        assert jnp.allclose(loss, 0.0, rtol=rtol, atol=atol)


class TestREff:
    """R_eff uses jittered RMS × Q^p (p=R_EFF_Q_POWER); Q>1 when |r| is uneven."""

    def test_r_eff_matches_rms_when_uniform_or_scalar(self):
        from moju.monitor.auditor import _rms_scalar, _r_eff_scalar

        u = jnp.ones((4,))
        assert jnp.allclose(_r_eff_scalar(u), _rms_scalar(u))
        assert jnp.allclose(_r_eff_scalar(jnp.array(2.0)), _rms_scalar(jnp.array(2.0)))

    def test_r_eff_exceeds_rms_and_lowers_admissibility_when_uneven(self):
        from moju.monitor.auditor import _rms_scalar, _r_eff_scalar

        spike = jnp.array([0.0, 0.0, 0.0, 10.0])
        r_eff = float(_r_eff_scalar(spike))
        r_rms = float(_rms_scalar(spike))
        assert r_eff > r_rms
        uniform = jnp.ones((4,))
        adm_u = 1.0 / (1.0 + float(_r_eff_scalar(uniform)))
        adm_spike = 1.0 / (1.0 + r_eff)
        assert adm_spike < adm_u


class TestAudit:
    def test_audit_writes_back_to_log(self):
        log = [
            {"index": 0, "rms": {"laws/a": 2.0, "constitutive/x": 1.0}},
            {"index": 1, "rms": {"laws/a": 1.0, "constitutive/x": 0.5}},
        ]
        report = audit(log)
        assert "per_key" in report
        assert "r_norm" in log[0]

    def test_audit_r_ref_from_first_entry(self):
        log = [{"index": 0, "rms": {"k": 10.0}}, {"index": 1, "rms": {"k": 5.0}}]
        audit(log)
        assert log[1]["r_norm"]["k"] == 0.5

    def test_audit_uses_r_ref_as_scale_when_supplied(self):
        log = [{"index": 0, "rms": {"k": 100.0}}, {"index": 1, "rms": {"k": 4.0}}]
        audit(log, r_ref={"k": 8.0})
        assert log[1]["r_norm"]["k"] == 0.5
        assert log[0]["r_norm"]["k"] == 100.0 / 8.0

    def test_audit_uses_entry_scale_when_present(self):
        log = [
            {"index": 0, "rms": {"laws/a": 2.0}, "scale": {"laws/a": 4.0}},
            {"index": 1, "rms": {"laws/a": 1.0}, "scale": {"laws/a": 4.0}},
        ]
        audit(log)
        assert log[1]["r_norm"]["laws/a"] == 0.25
        assert log[0]["r_norm"]["laws/a"] == 0.5

    def test_engine_log_has_scale_and_audit_uses_it(self, rtol, atol):
        core = ResidualEngine(
            laws=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_xx"}}],
        )
        state_pred = {"phi_xx": jnp.array(1.0)}
        core.compute_residuals(state_pred)
        assert "scale" in core.log[-1]
        assert "laws/laplace_equation" in core.log[-1]["scale"]
        report = audit(core.log)
        r_norm = core.log[-1]["r_norm"]["laws/laplace_equation"]
        scale_k = core.log[-1]["scale"]["laws/laplace_equation"]
        rms = core.log[-1]["rms"]["laws/laplace_equation"]
        assert abs(r_norm - rms / scale_k) < 1e-6

    def test_default_scale_is_unit_for_laws_and_implied_delta(self, rtol, atol):
        """ND law and implied_delta keys use scale_k ≈ DEFAULT_NONDIM_R_NORM_SCALE_K."""
        from moju.monitor.auditor import DEFAULT_NONDIM_R_NORM_SCALE_K

        P, R, T = jnp.array(1e6), jnp.array(287.0), jnp.array(300.0)
        rho = Models.ideal_gas_rho(P, R, T)
        core = ResidualEngine(
            laws=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_xx"}}],
            constitutive_audit=[
                {
                    "name": "ideal_gas_rho",
                    "output_key": "rho",
                    "state_map": {"P": "P", "R": "R", "T": "T"},
                    "implied_value_key": "rho_implied",
                }
            ],
        )
        state_pred = {
            "phi_xx": jnp.array(2.0),
            "P": P,
            "R": R,
            "T": T,
            "rho": rho,
            "rho_implied": rho,
        }
        core.compute_residuals(state_pred)
        sc = core.log[-1]["scale"]
        assert abs(sc["laws/laplace_equation"] - DEFAULT_NONDIM_R_NORM_SCALE_K) < 1e-9
        assert abs(sc["constitutive/ideal_gas_rho/implied_delta"] - DEFAULT_NONDIM_R_NORM_SCALE_K) < 1e-9

    def test_constitutive_implied_scale_is_default_nondim(self, rtol, atol):
        from moju.monitor.auditor import DEFAULT_NONDIM_R_NORM_SCALE_K

        P, R = jnp.array(101325.0), jnp.array(287.0)
        T = jnp.array(290.0)
        rho = Models.ideal_gas_rho(P, R, T)
        core = ResidualEngine(
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
        state_pred = {"P": P, "R": R, "T": T, "rho": rho, "rho_implied": rho}
        core.compute_residuals(state_pred)
        sk = core.log[-1]["scale"]["constitutive/ideal_gas_rho/implied_delta"]
        assert abs(sk - DEFAULT_NONDIM_R_NORM_SCALE_K) < 1e-9

    def test_audit_export_dir_pdf_with_new_categories(self, tmp_path):
        pytest.importorskip("reportlab")
        log = [
            {"index": 0, "rms": {"laws/a": 1.0, "constitutive/m/implied_delta": 0.5, "scaling/pe/ref_delta": 0.1}},
            {"index": 1, "rms": {"laws/a": 0.5, "constitutive/m/implied_delta": 0.25, "scaling/pe/ref_delta": 0.05}},
        ]
        report = audit(log, export_dir=str(tmp_path))
        assert "per_key" in report
        dirs = [d for d in tmp_path.iterdir() if d.is_dir() and d.name.startswith("audit_")]
        assert len(dirs) == 1
        assert (dirs[0] / "report.pdf").exists()


class TestVisualize:
    def test_visualize_empty_log_returns_none(self):
        assert visualize([], backend="plotly") is None

    def test_visualize_backend_none_returns_none(self):
        assert visualize([{"index": 0, "rms": {"k": 1.0}}], backend="none") is None

    def test_visualize_matplotlib_backend_raises(self):
        with pytest.raises(ValueError, match="matplotlib"):
            visualize([{"index": 0, "rms": {"k": 1.0}, "scale": {}}], backend="matplotlib")

    def test_visualize_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown visualize backend"):
            visualize([{"index": 0, "rms": {"k": 1.0}, "scale": {}}], backend="not_a_backend")

    def test_visualize_multi_panel_figure(self):
        pytest.importorskip("plotly")
        log = [
            {
                "index": 0,
                "rms": {
                    "laws/a": 1.0,
                    "constitutive/m/implied_delta": 0.5,
                    "scaling/pe/ref_delta": 0.1,
                    "data/T": 0.2,
                },
                "scale": {},
                "omitted": ["demo omit"],
                "inferred": [],
            },
            {
                "index": 1,
                "rms": {
                    "laws/a": 0.5,
                    "constitutive/m/implied_delta": 0.25,
                    "scaling/pe/ref_delta": 0.05,
                    "data/T": 0.1,
                },
                "scale": {},
            },
        ]
        fig = visualize(log, mode="training")
        assert fig is not None
        assert hasattr(fig, "data")
        assert len(fig.data) >= 7

    def test_visualize_r_norm_scale_linear(self):
        pytest.importorskip("plotly")
        log = [
            {
                "index": 0,
                "rms": {
                    "laws/a": 1.0,
                    "constitutive/m/implied_delta": 0.5,
                    "scaling/pe/ref_delta": 0.1,
                },
                "scale": {},
            },
            {
                "index": 1,
                "rms": {
                    "laws/a": 0.5,
                    "constitutive/m/implied_delta": 0.25,
                    "scaling/pe/ref_delta": 0.05,
                },
                "scale": {},
            },
        ]
        fig_p = visualize(log, mode="training", r_norm_scale="linear")
        assert fig_p is not None

    def test_build_monitor_visualize_bundle_and_studio_plotly_cards(self):
        pytest.importorskip("plotly")
        import numpy as np

        from moju.monitor.auditor import build_monitor_visualize_bundle
        from moju.monitor.visualize_plotly import (
            MOJU_STUDIO_DASHBOARD_CARD_HEIGHT,
            build_plotly_category_admissibility_bar_figure,
            build_plotly_law_rnorm_final_bar_figure,
            build_plotly_spatial_rnorm_heatmap_card,
        )

        log = [
            {
                "index": 0,
                "rms": {
                    "laws/a": 1.0,
                    "constitutive/m/implied_delta": 0.5,
                    "scaling/pe/ref_delta": 0.1,
                },
                "scale": {},
            },
            {
                "index": 1,
                "rms": {
                    "laws/a": 0.5,
                    "constitutive/m/implied_delta": 0.25,
                    "scaling/pe/ref_delta": 0.05,
                },
                "scale": {},
            },
        ]
        x = np.linspace(0, 1, 5)
        spatial_law = {"x": x, "values": {"laws/a": np.ones(5) * 0.2}}
        spatial_c = {"x": x, "values": {"constitutive/m/implied_delta": np.ones(5) * 0.1}}
        bundle = build_monitor_visualize_bundle(
            log,
            r_ref=None,
            max_legend_keys=16,
            spatial_law_panel=spatial_law,
            spatial_rnorm_panel=spatial_c,
            mode="training",
        )
        assert bundle is not None
        f1 = build_plotly_law_rnorm_final_bar_figure(bundle)
        f2 = build_plotly_category_admissibility_bar_figure(bundle)
        assert f1 is not None and f2 is not None
        assert f1.layout.height == MOJU_STUDIO_DASHBOARD_CARD_HEIGHT
        assert f2.layout.height == MOJU_STUDIO_DASHBOARD_CARD_HEIGHT
        law_bar = next((t for t in f1.data if getattr(t, "type", None) == "bar"), None)
        assert law_bar is not None
        marker_color = getattr(getattr(law_bar, "marker", None), "color", None)
        if isinstance(marker_color, (list, tuple)):
            assert all(c == "#8B5CF6" for c in marker_color)
        else:
            assert marker_color == "#8B5CF6"
        f3 = build_plotly_spatial_rnorm_heatmap_card(bundle["spatial"], colorscale="Jet")
        f4 = build_plotly_spatial_rnorm_heatmap_card(bundle["spatial_rnorm"], colorscale="Jet")
        assert f3 is not None and f4 is not None
        assert f3.layout.height == MOJU_STUDIO_DASHBOARD_CARD_HEIGHT
        assert f4.layout.height == MOJU_STUDIO_DASHBOARD_CARD_HEIGHT

    def test_build_monitor_visualize_bundle_uses_residuals_when_panels_none(self):
        import numpy as np

        from moju.monitor.auditor import build_monitor_visualize_bundle

        log = [
            {
                "index": 0,
                "rms": {"laws/a": 1.0, "constitutive/m/c": 0.5},
                "scale": {"laws/a": 1.0, "constitutive/m/c": 1.0},
                "coord_snapshot": {"x": [0.0, 0.5, 1.0]},
            },
        ]
        residuals = {"laws": {"a": np.ones(3) * 0.1}, "constitutive": {"m/c": np.ones(3) * 0.05}}
        bundle = build_monitor_visualize_bundle(
            log,
            spatial_law_panel=None,
            spatial_rnorm_panel=None,
            mode="training",
            residuals=residuals,
            state_pred={},
        )
        assert bundle is not None
        assert bundle.get("spatial") is not None
        assert bundle.get("spatial_rnorm") is not None

    def test_maybe_build_spatial_panels_fills_missing_constitutive_side(self):
        import numpy as np

        from moju.monitor.auditor import _maybe_build_spatial_panels

        x = np.linspace(0, 1, 4)
        explicit_law = {"x": x, "values": {"laws/a": np.ones(4) * 0.2}}
        log = [
            {
                "index": 0,
                "rms": {"laws/a": 1.0, "constitutive/m/c": 0.5},
                "scale": {"laws/a": 1.0, "constitutive/m/c": 1.0},
            },
        ]
        residuals = {"constitutive": {"m/c": np.ones(4) * 0.1}}
        law_p, rn_p = _maybe_build_spatial_panels(
            log,
            explicit_law,
            None,
            residuals,
            {"x": x},
            None,
            "x",
            True,
        )
        assert law_p is not None
        assert rn_p is not None
        assert "values" in rn_p and any(k.startswith("constitutive/") for k in (rn_p.get("values") or {}))

    def test_visualize_eval_mode_uses_last_log_entry(self):
        pytest.importorskip("plotly")
        log = [
            {"index": 0, "rms": {"laws/a": 10.0}, "scale": {"laws/a": 1.0}},
            {"index": 1, "rms": {"laws/a": 0.1}, "scale": {"laws/a": 1.0}},
        ]
        fig = visualize(log, mode="eval")
        assert fig is not None
        assert hasattr(fig, "data")

    def test_visualize_spatial_law_panel_plotly(self):
        pytest.importorskip("plotly")
        import numpy as np

        log = [
            {"index": 0, "rms": {"laws/a": 1.0}, "scale": {"laws/a": 1.0}},
            {"index": 1, "rms": {"laws/a": 0.5}, "scale": {"laws/a": 1.0}},
        ]
        x = np.linspace(0, 1, 5)
        fig = visualize(
            log,
            mode="training",
            spatial_law_panel={"x": x, "values": {"a": np.ones(5) * 0.2}},
        )
        assert fig is not None
        assert len(fig.data) >= 7
        hm = [t for t in fig.data if getattr(t, "type", None) == "heatmap"]
        assert len(hm) >= 1

    def test_visualize_category_adm_high_scores_plotly(self):
        pytest.importorskip("plotly")
        log = [
            {
                "index": 0,
                "rms": {"laws/a": 0.0196, "constitutive/b": 0.0198},
                "scale": {"laws/a": 1.0, "constitutive/b": 1.0},
            },
            {
                "index": 1,
                "rms": {"laws/a": 0.0196, "constitutive/b": 0.0198},
                "scale": {"laws/a": 1.0, "constitutive/b": 1.0},
            },
        ]
        fig = visualize(log, mode="training")
        assert fig is not None
        assert len(fig.data) >= 5

    def test_build_visualize_bundle_category_training(self):
        from moju.monitor.auditor import _build_visualize_bundle

        log = [
            {"index": 0, "rms": {"laws/x": 1.0, "constitutive/a/b": 0.5, "scaling/y/z": 0.2}, "scale": {}},
            {"index": 1, "rms": {"laws/x": 0.5, "constitutive/a/b": 0.25, "scaling/y/z": 0.1}, "scale": {}},
        ]
        b = _build_visualize_bundle(log, None, None, 8, spatial_parsed=None, mode="training")
        assert b is not None
        ct = b["category_training"]
        assert "laws/x" in ct["laws"]["keys"]
        assert "constitutive/a/b" in ct["constitutive"]["keys"]
        assert "scaling" not in ct
        wk = b.get("worst_keys_rows") or []
        assert len(wk) >= 1
        assert wk[0]["key"] in ("laws/x", "constitutive/a/b", "scaling/y/z")

    def test_visualize_eval_title_shows_overall_rollup(self):
        pytest.importorskip("plotly")
        log = [
            {
                "index": 0,
                "run_mode": "eval",
                "rms": {"laws/a": 1.0, "constitutive/b": 0.5},
                "scale": {"laws/a": 1.0, "constitutive/b": 1.0},
            },
        ]
        fig = visualize(log, backend="plotly", mode="eval")
        lt = str(fig.layout.title.text or "")
        assert "Overall admissibility (final)" in lt
        assert "not defined" not in lt.lower()

    def test_visualize_split_layout_returns_monitor_and_worst_keys(self):
        pytest.importorskip("plotly")
        log = [
            {"index": 0, "rms": {"laws/a": 1.0, "constitutive/b": 0.5}, "scale": {"laws/a": 2.0, "constitutive/b": 2.0}},
            {"index": 1, "rms": {"laws/a": 0.5, "constitutive/b": 0.25}, "scale": {"laws/a": 2.0, "constitutive/b": 2.0}},
        ]
        out = visualize(log, backend="plotly", mode="training", visualize_layout="split")
        assert isinstance(out, dict)
        assert "monitor" in out and "worst_keys" in out
        wk = out["worst_keys"]
        assert any(getattr(tr, "type", None) == "table" for tr in getattr(wk, "data", []))

    def test_visualize_spatial_heatmap_default_colorscale_viridis(self):
        pytest.importorskip("plotly")
        import numpy as np

        from moju.monitor.visualize_plotly import DEFAULT_SPATIAL_HEATMAP_COLORSCALE

        log = [
            {"index": 0, "rms": {"laws/a": 1.0}, "scale": {"laws/a": 1.0}},
            {"index": 1, "rms": {"laws/a": 0.5}, "scale": {"laws/a": 1.0}},
        ]
        x = np.linspace(0, 1, 5)
        fig = visualize(
            log,
            mode="training",
            spatial_law_panel={"x": x, "values": {"a": np.ones(5) * 0.2}},
        )
        hm = [t for t in fig.data if getattr(t, "type", None) == "heatmap"]
        assert hm
        cs = getattr(hm[0], "colorscale", None)
        assert cs == DEFAULT_SPATIAL_HEATMAP_COLORSCALE or (
            isinstance(cs, (list, tuple)) and len(cs) > 0 and str(cs[0][1]).lower().startswith("#440154")
        )

    def test_visualize_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            visualize([{"index": 0, "rms": {"k": 1.0}, "scale": {}}], mode="invalid")

    def test_visualize_plotly_returns_figure(self):
        pytest.importorskip("plotly")
        import numpy as np

        log = [
            {
                "index": 0,
                "rms": {
                    "laws/a": 1.0,
                    "constitutive/m/implied_delta": 0.5,
                    "scaling/pe/ref_delta": 0.1,
                    "data/T": 0.2,
                },
                "scale": {},
                "omitted": ["demo omit"],
                "inferred": [],
            },
            {
                "index": 1,
                "rms": {
                    "laws/a": 0.5,
                    "constitutive/m/implied_delta": 0.25,
                    "scaling/pe/ref_delta": 0.05,
                    "data/T": 0.1,
                },
                "scale": {},
            },
        ]
        x = np.linspace(0, 1, 5)
        spatial_law = {"x": x, "values": {"laws/a": np.ones(5) * 0.2}}
        spatial_c = {"x": x, "values": {"constitutive/m/implied_delta": np.ones(5) * 0.1}}
        fig = visualize(
            log,
            backend="plotly",
            mode="training",
            spatial_law_panel=spatial_law,
            spatial_rnorm_panel=spatial_c,
        )
        assert fig is not None
        assert hasattr(fig, "data")
        assert len(fig.data) >= 7
        hm = [t for t in fig.data if getattr(t, "type", None) == "heatmap"]
        assert len(hm) >= 2

    def test_visualize_autofill_spatial_from_residuals(self):
        pytest.importorskip("plotly")
        import numpy as np

        log = [
            {"index": 0, "rms": {"laws/a": 1.0, "constitutive/m/c": 0.5}, "scale": {"laws/a": 1.0, "constitutive/m/c": 1.0}},
        ]
        x = np.linspace(0, 1, 5)
        residuals = {"laws": {"a": np.ones(5) * 0.1}, "constitutive": {"m/c": np.ones(5) * 0.05}}
        pred = {"x": x}
        fig = visualize(
            log,
            mode="training",
            residuals=residuals,
            state_pred=pred,
        )
        assert fig is not None
        hm = [t for t in fig.data if getattr(t, "type", None) == "heatmap"]
        assert len(hm) >= 2

    def test_visualize_uses_engine_without_explicit_residuals(self):
        pytest.importorskip("plotly")

        core = ResidualEngine(
            laws=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_laplacian"}}],
        )
        x = jnp.linspace(0.0, 1.0, 5)
        state_pred = {"phi_laplacian": jnp.ones(5) * 0.1, "x": x}
        core.compute_residuals(state_pred)
        log = list(core.log)
        assert log[-1].get("coord_snapshot")
        fig = visualize(log, mode="training", engine=core)
        assert fig is not None
        hm = [t for t in fig.data if getattr(t, "type", None) == "heatmap"]
        assert len(hm) >= 1

    def test_visualize_uses_coord_snapshot_when_state_pred_empty(self):
        pytest.importorskip("plotly")
        import numpy as np

        log = [
            {
                "index": 0,
                "rms": {"laws/a": 1.0, "constitutive/m/c": 0.5},
                "scale": {"laws/a": 1.0, "constitutive/m/c": 1.0},
                "coord_snapshot": {"x": [0.0, 0.25, 0.5, 0.75, 1.0]},
            },
        ]
        residuals = {"laws": {"a": np.ones(5) * 0.1}, "constitutive": {"m/c": np.ones(5) * 0.05}}
        fig = visualize(
            log,
            mode="training",
            residuals=residuals,
            state_pred={},
        )
        assert fig is not None
        hm = [t for t in fig.data if getattr(t, "type", None) == "heatmap"]
        assert len(hm) >= 2

    def test_visualize_plotly_default_titles(self):
        pytest.importorskip("plotly")
        log = [
            {
                "index": 0,
                "rms": {
                    "laws/a": 1.0,
                    "constitutive/m/implied_delta": 0.5,
                    "scaling/pe/ref_delta": 0.1,
                },
                "scale": {},
            },
            {
                "index": 1,
                "rms": {
                    "laws/a": 0.5,
                    "constitutive/m/implied_delta": 0.25,
                    "scaling/pe/ref_delta": 0.05,
                },
                "scale": {},
            },
        ]
        fig_tr = visualize(log, backend="plotly", mode="training")
        ttr = str(fig_tr.layout.title.text or "")
        assert DEFAULT_VISUALIZE_TITLE_TRAINING in ttr
        assert "Overall admissibility (final)" in ttr
        fig_te = visualize(log, backend="plotly", mode="eval")
        tte = str(fig_te.layout.title.text or "")
        assert DEFAULT_VISUALIZE_TITLE_EVAL in tte
        assert DEFAULT_VISUALIZE_TITLE_TEST in tte
        assert "Overall admissibility (final)" in tte

    def test_visualize_mode_test_alias_matches_eval(self):
        pytest.importorskip("plotly")
        from moju.monitor.auditor import build_monitor_visualize_bundle

        log = [{"index": 0, "rms": {"laws/a": 1.0, "constitutive/m/c": 0.5}, "scale": {}}]
        fig_e = visualize(log, backend="plotly", mode="eval")
        fig_t = visualize(log, backend="plotly", mode="test")
        assert fig_e.layout.height == fig_t.layout.height
        assert len(fig_e.data) == len(fig_t.data)
        b_e = build_monitor_visualize_bundle(log, mode="eval")
        b_t = build_monitor_visualize_bundle(log, mode="test")
        assert b_e is not None and b_t is not None
        assert b_e["mode"] == "eval" and b_t["mode"] == "eval"

    def test_visualize_plotly_default_theme_is_light(self):
        pytest.importorskip("plotly")
        log = [{"index": 0, "rms": {"laws/a": 1.0}, "scale": {}}]
        fig = visualize(log, backend="plotly", mode="training")
        assert fig is not None
        assert getattr(fig.layout, "paper_bgcolor", None) == "#ffffff"
        assert getattr(fig.layout, "plot_bgcolor", None) == "#ffffff"

    def test_visualize_plotly_theme_dark_rejected(self):
        pytest.importorskip("plotly")
        log = [{"index": 0, "rms": {"laws/a": 1.0}, "scale": {}}]
        with pytest.raises(ValueError, match="theme='light' only"):
            visualize(log, backend="plotly", mode="training", theme="dark")

    def test_visualize_plotly_eval_mode_spatial_row_placeholders_without_coords(self):
        pytest.importorskip("plotly")
        log = [{"index": 0, "rms": {"laws/a": 1.0, "constitutive/m/c": 0.5}, "scale": {}}]
        fig = visualize(log, backend="plotly", mode="eval")
        assert fig is not None
        hm = [t for t in fig.data if getattr(t, "type", None) == "heatmap"]
        assert len(hm) == 0
        sc = [t for t in fig.data if getattr(t, "type", None) == "scatter"]
        assert any("spatial" in str(getattr(t, "text", "") or "").lower() for t in sc)

    def test_visualize_plotly_eval_mode_heatmaps_with_residuals_only(self):
        pytest.importorskip("plotly")
        import numpy as np

        log = [{"index": 0, "rms": {"laws/a": 1.0, "constitutive/m/c": 0.5}, "scale": {}}]
        residuals = {"laws": {"a": np.ones(5) * 0.1}, "constitutive": {"m/c": np.ones(5) * 0.05}}
        fig = visualize(log, backend="plotly", mode="eval", residuals=residuals)
        assert fig is not None
        hm = [t for t in fig.data if getattr(t, "type", None) == "heatmap"]
        assert len(hm) >= 2

    def test_visualize_plotly_eval_mode_dual_spatial_heatmaps(self):
        pytest.importorskip("plotly")
        import numpy as np

        log = [{"index": 0, "rms": {"laws/a": 1.0, "constitutive/m/c": 0.5}, "scale": {}}]
        x = np.linspace(0, 1, 5)
        spatial_law = {"x": x, "values": {"laws/a": np.ones(5) * 0.2}}
        spatial_c = {"x": x, "values": {"constitutive/m/c": np.ones(5) * 0.1}}
        fig = visualize(
            log,
            backend="plotly",
            mode="eval",
            spatial_law_panel=spatial_law,
            spatial_rnorm_panel=spatial_c,
        )
        assert fig is not None
        hm = [t for t in fig.data if getattr(t, "type", None) == "heatmap"]
        assert len(hm) == 2

    def test_parse_spatial_law_panel_1d_2d_3d(self):
        import numpy as np

        from moju.monitor.auditor import _parse_spatial_law_panel

        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 1, 4)
        z = np.linspace(0, 1, 5)
        o1 = _parse_spatial_law_panel({"x": x, "values": {"a": np.ones(3)}})
        assert o1 is not None and o1["kind"] == "1d"
        assert o1["Z"].shape == (1, 3)
        o2 = _parse_spatial_law_panel({"x": x, "y": y, "values": {"laplace": np.ones((4, 3))}})
        assert o2 is not None and o2["kind"] == "2d"
        assert o2["Z"].shape == (1, 4, 3)
        o3 = _parse_spatial_law_panel({"x": x, "y": y, "z": z, "values": {"laplace": np.ones((3, 4, 5))}})
        assert o3 is not None and o3["kind"] == "3d"
        assert o3["V"].shape == (1, 3, 4, 5)
        assert _parse_spatial_law_panel({"x": x, "y": y, "values": {"a": np.ones(3)}}) is None
        o2i = _parse_spatial_law_panel(
            {"x": x, "y": y, "values": {"a": np.ones((4, 3))}, "log_step_index": 7}
        )
        assert o2i is not None and o2i.get("log_step_index") == 7

    def test_build_plotly_spatial_card_2d_and_volume_3d(self):
        pytest.importorskip("plotly")
        import numpy as np

        from moju.monitor.auditor import _parse_spatial_law_panel
        from moju.monitor.visualize_plotly import build_plotly_spatial_rnorm_heatmap_card

        x, y = np.linspace(0, 1, 4), np.linspace(0, 1, 5)
        sp2 = _parse_spatial_law_panel({"x": x, "y": y, "values": {"laws/a": np.ones((5, 4)) * 0.2}})
        f2 = build_plotly_spatial_rnorm_heatmap_card(sp2)
        assert f2 is not None and f2.data
        assert getattr(f2.data[0], "type", None) == "heatmap"
        z = np.linspace(0, 1, 3)
        sp3 = _parse_spatial_law_panel(
            {
                "x": np.linspace(0, 1, 3),
                "y": np.linspace(0, 1, 4),
                "z": z,
                "values": {"laws/a": np.ones((3, 4, 3)) * 0.15},
            }
        )
        f3 = build_plotly_spatial_rnorm_heatmap_card(sp3, card_title="Vol")
        assert f3 is not None
        assert any(getattr(t, "type", None) == "volume" for t in f3.data)

    def test_build_plotly_spatial_card_1d_moves_key_to_subtitle_and_updates_colorbar(self):
        pytest.importorskip("plotly")
        import numpy as np

        from moju.monitor.auditor import _parse_spatial_law_panel
        from moju.monitor.visualize_plotly import build_plotly_spatial_rnorm_heatmap_card

        x = np.linspace(0, 1, 5)
        sp1 = _parse_spatial_law_panel({"x": x, "values": {"fourier_conduction": np.ones(5) * 0.2}})
        fig = build_plotly_spatial_rnorm_heatmap_card(sp1, card_title="Spatial |residual|")
        assert fig is not None

        assert fig.layout.yaxis.showticklabels is False

        title_text = fig.layout.title.text or ""
        assert "Fourier Conduction" in title_text

        hm = next(t for t in fig.data if getattr(t, "type", None) == "heatmap")
        cb_title_obj = getattr(getattr(hm, "colorbar", None), "title", None)
        cb_title_text = getattr(cb_title_obj, "text", None)
        assert cb_title_text == "log10(|residual| + ε)"

    def test_visualize_plotly_figure_title_override(self):
        pytest.importorskip("plotly")
        log = [
            {"index": 0, "rms": {"laws/a": 1.0, "constitutive/m/c": 0.5}, "scale": {}},
            {"index": 1, "rms": {"laws/a": 0.5, "constitutive/m/c": 0.25}, "scale": {}},
        ]
        fig = visualize(log, backend="plotly", mode="training", figure_title="Custom dashboard title")
        lt = str(fig.layout.title.text or "")
        assert "Custom dashboard title" in lt
        assert "Overall admissibility (final)" in lt

    def test_visualize_plotly_overall_admissibility_line_is_black(self):
        pytest.importorskip("plotly")
        log = [
            {"index": 0, "rms": {"laws/a": 1.0, "constitutive/m/c": 0.5}, "scale": {}},
            {"index": 1, "rms": {"laws/a": 0.5, "constitutive/m/c": 0.25}, "scale": {}},
        ]
        fig = visualize(log, backend="plotly", mode="training", dashboard_mode="single-figure")
        assert fig is not None
        tr = next((t for t in fig.data if getattr(t, "name", "") == "Overall Admissibility"), None)
        assert tr is not None
        line = getattr(tr, "line", None)
        color = getattr(line, "color", None) if line is not None else None
        assert str(color).lower() in {"#000000", "rgb(0, 0, 0)"}

    def test_visualize_plotly_worst_violation_annotation_only_when_multiple_keys(self):
        pytest.importorskip("plotly")
        # Case 1: single key per category -> no "Worst violation:" annotation.
        log1 = [
            {"index": 0, "rms": {"laws/a": 1.0, "constitutive/m/c": 0.5}, "scale": {}},
            {"index": 1, "rms": {"laws/a": 0.5, "constitutive/m/c": 0.25}, "scale": {}},
        ]
        fig1 = visualize(log1, backend="plotly", mode="training", dashboard_mode="single-figure")
        anns1 = list(getattr(fig1.layout, "annotations", []) or [])
        worst_anns1 = [a for a in anns1 if "Worst violation:" in str(getattr(a, "text", "") or "")]
        assert len(worst_anns1) == 0

        # Case 2: two law keys, one constitutive key -> exactly one "Worst violation:" annotation.
        log2 = [
            {
                "index": 0,
                "rms": {"laws/a": 1.0, "laws/b": 0.8, "constitutive/m/c": 0.5},
                "scale": {},
            },
            {
                "index": 1,
                "rms": {"laws/a": 0.5, "laws/b": 0.3, "constitutive/m/c": 0.25},
                "scale": {},
            },
        ]
        fig2 = visualize(log2, backend="plotly", mode="training", dashboard_mode="single-figure")
        anns2 = list(getattr(fig2.layout, "annotations", []) or [])
        worst_anns2 = [a for a in anns2 if "Worst violation:" in str(getattr(a, "text", "") or "")]
        assert len(worst_anns2) == 1

    def test_visualize_training_all_residuals_bar_and_category_colors(self):
        pytest.importorskip("plotly")
        log = [
            {
                "index": 0,
                "rms": {
                    "laws/fourier_conduction": 1.0,
                    "constitutive/thermal_diffusivity/implied_delta": 0.5,
                },
                "scale": {},
            },
            {
                "index": 1,
                "rms": {
                    "laws/fourier_conduction": 0.5,
                    "constitutive/thermal_diffusivity/implied_delta": 0.25,
                },
                "scale": {},
            },
        ]
        fig = visualize(log, backend="plotly", mode="training")
        assert fig is not None

        # Overall Admissibility trend: y-axis 0–100 with integer ticks (title still Admissibility (%)).
        yaxes = [getattr(fig.layout, k) for k in dir(fig.layout) if k.startswith("yaxis")]
        assert any(
            getattr(getattr(ax, "title", None), "text", None) == "Admissibility (%)"
            and getattr(ax, "tickformat", None) == ".0f"
            for ax in yaxes
        )

        # Residual diagnostics include law/constitutive traces.
        line_names = [
            str(getattr(tr, "name", "") or "")
            for tr in fig.data
            if getattr(tr, "type", None) == "scatter" and getattr(tr, "mode", None) == "lines"
        ]
        assert any("Fourier Conduction" in n for n in line_names)
        assert any("Thermal diffusivity (implied)" in n for n in line_names)

        # Color mapping: laws=purple, constitutive=teal on non-overall residual traces.
        color_by_name = {}
        for tr in fig.data:
            if getattr(tr, "type", None) == "scatter" and getattr(tr, "mode", None) == "lines":
                nm = str(getattr(tr, "name", "") or "")
                line = getattr(tr, "line", None)
                color = getattr(line, "color", None)
                if nm:
                    color_by_name[nm] = color
        assert color_by_name.get("Fourier Conduction") == "#8B5CF6"
        assert color_by_name.get("Thermal diffusivity (implied)") == "#14B8A6"


    def test_visualize_dash_tabs_payload_contract(self):
        pytest.importorskip("plotly")
        log = [
            {"index": 0, "rms": {"laws/a": 1.0, "constitutive/b/implied_delta": 0.5}, "scale": {"laws/a": 2.0}},
            {"index": 1, "rms": {"laws/a": 0.5, "constitutive/b/implied_delta": 0.25}, "scale": {"laws/a": 2.0}},
        ]
        out = visualize(log, backend="plotly", mode="training", dashboard_mode="dash-tabs", baseline_score=0.6)
        assert isinstance(out, dict)
        assert out.get("mode") == "dash-tabs"
        assert out.get("toggles", {}).get("mode") == ["training", "eval"]
        tabs = out.get("tabs") or {}
        assert set(["kpi", "admissibility", "forensic_heatmaps", "convergence"]).issubset(set(tabs.keys()))
        kpi_fig = tabs["kpi"]
        assert any(getattr(tr, "type", None) == "indicator" for tr in getattr(kpi_fig, "data", []))

        log_ev = [
            {"index": 0, "run_mode": "eval", "rms": {"laws/a": 1.0, "constitutive/b/implied_delta": 0.5}, "scale": {}},
        ]
        out_ev = visualize(log_ev, backend="plotly", mode="eval", dashboard_mode="dash-tabs")
        kpi_ev = (out_ev.get("tabs") or {}).get("kpi")
        assert kpi_ev is not None
        assert any(getattr(tr, "type", None) == "indicator" for tr in getattr(kpi_ev, "data", []))
        anns_ev = list(getattr(kpi_ev.layout, "annotations", []) or [])
        assert any(
            "rollup" in str(getattr(a, "text", "")).lower()
            or "overall admissibility" in str(getattr(a, "text", "")).lower()
            for a in anns_ev
        )

    def test_visualize_enterprise_threshold_line_and_scale_hover(self):
        pytest.importorskip("plotly")
        log = [
            {"index": 0, "rms": {"laws/a": 1.0, "constitutive/b/implied_delta": 0.5}, "scale": {"laws/a": 2.0, "constitutive/b/implied_delta": 3.0}},
            {"index": 1, "rms": {"laws/a": 0.5, "constitutive/b/implied_delta": 0.25}, "scale": {"laws/a": 2.0, "constitutive/b/implied_delta": 3.0}},
        ]
        fig = visualize(log, backend="plotly", mode="training", dashboard_mode="single-figure")
        assert fig is not None
        lt = str(getattr(getattr(fig.layout, "title", None), "text", None) or "")
        assert DEFAULT_VISUALIZE_TITLE_TRAINING in lt
        assert "Overall admissibility (final)" in lt
        inds = [t for t in fig.data if getattr(t, "type", None) == "indicator"]
        assert len(inds) == 2
        anns = list(getattr(fig.layout, "annotations", []) or [])
        assert not any(
            "Overall admissibility (final)" in str(getattr(a, "text", "") or "") for a in anns
        ), "overall line is merged into layout title, not a separate annotation"
        assert not any("Final Step:" in str(getattr(a, "text", "") or "") for a in anns)
        shapes = list(getattr(fig.layout, "shapes", []) or [])
        _thr_x = ADM_HIGH_THRESHOLD * 100.0
        assert any(
            abs(float(getattr(s, "x0", -1)) - _thr_x) < 1e-6
            for s in shapes
            if getattr(s, "type", None) == "line"
        )
        hover_templates = [str(getattr(t, "hovertemplate", "")) for t in fig.data]
        assert any("scale_k=" in h for h in hover_templates)

    def test_visualize_eval_mode_combined_residual_bars_and_spatial_row_four(self):
        """Eval single-figure: one combined residual bar chart beside category; spatial maps on row 4."""
        pytest.importorskip("plotly")
        import numpy as np

        log = [
            {"index": 0, "rms": {"laws/a": 1.0, "constitutive/m/c": 0.5, "scaling/pe/x": 0.2}, "scale": {}},
            {"index": 1, "rms": {"laws/a": 0.8, "constitutive/m/c": 0.4, "scaling/pe/x": 0.1}, "scale": {}},
        ]
        x = np.linspace(0, 1, 8)
        spatial_law = {"x": x, "values": {"laws/a": np.ones(8) * 0.01}}
        spatial_c = {"x": x, "values": {"constitutive/m/c": np.ones(8) * 0.5}}
        fig = visualize(
            log,
            backend="plotly",
            mode="eval",
            dashboard_mode="single-figure",
            spatial_law_panel=spatial_law,
            spatial_rnorm_panel=spatial_c,
        )
        bars = [t for t in fig.data if getattr(t, "type", None) == "bar"]
        assert len(bars) == 2, "category breakdown + single combined residual bar chart"
        combined = bars[-1]
        mcol = getattr(getattr(combined, "marker", None), "color", None)
        assert isinstance(mcol, (list, tuple)) and len(mcol) >= 2
        assert len(set(mcol)) >= 2, "per-key colors distinguish categories"
        yref = getattr(combined, "yaxis", "y")
        yaxis_name = "yaxis" if yref == "y" else f"yaxis{yref[1:]}"
        yax = getattr(fig.layout, yaxis_name)
        assert getattr(yax, "type", None) == "log"
        assert getattr(yax, "exponentformat", None) == "power"
        anns = [str(getattr(a, "text", "") or "") for a in (fig.layout.annotations or [])]
        summary = next((a for a in anns if a.startswith("Summary:")), "")
        assert "Training trend improving" not in summary
        assert "Training trend degrading" not in summary
        primary = next(a for a in fig.layout.annotations if "Primary Issue:" in str(getattr(a, "text", "") or ""))
        assert float(primary.y) > 1.0
        assert getattr(primary, "yanchor", None) == "bottom"
        hms = [
            t
            for t in fig.data
            if getattr(t, "type", None) == "heatmap"
            and isinstance(getattr(t, "meta", None), dict)
        ]
        assert len(hms) == 2
        assert all(int(t.meta.get("subplot_row", 0)) == 4 for t in hms)
        domains = []
        for hm in hms:
            xref = getattr(hm, "xaxis", "x")
            axis_name = "xaxis" if xref == "x" else f"xaxis{xref[1:]}"
            dom = getattr(getattr(fig.layout, axis_name), "domain")
            domains.append((float(dom[0]), float(dom[1])))
        widths = [b - a for a, b in domains]
        assert min(widths) > 0.4
        assert abs(widths[0] - widths[1]) < 1e-9

    def test_visualize_training_summary_keeps_training_trend_line(self):
        pytest.importorskip("plotly")

        log = [
            {"index": 0, "rms": {"laws/a": 1.0, "constitutive/m/c": 0.5}, "scale": {}},
            {"index": 1, "rms": {"laws/a": 0.5, "constitutive/m/c": 0.25}, "scale": {}},
        ]
        fig = visualize(log, backend="plotly", mode="training", dashboard_mode="single-figure")
        anns = [str(getattr(a, "text", "") or "") for a in (fig.layout.annotations or [])]
        summary = next((a for a in anns if a.startswith("Summary:")), "")
        assert "Training trend improving" in summary
        primary = next(a for a in fig.layout.annotations if "Primary Issue:" in str(getattr(a, "text", "") or ""))
        assert float(primary.y) > 1.0
        assert getattr(primary, "yanchor", None) == "bottom"

    def test_visualize_single_figure_subplot_title_annotations_match_panels(self):
        """Explicit panel titles (merged grid + domain cells) plus spatial colorbar/hover wording."""
        pytest.importorskip("plotly")
        import numpy as np

        from moju.monitor.visualize_plotly import SPATIAL_HEATMAP_COLORBAR_TITLE_LOG, _monitor_flat_subplot_titles

        expect_tr = {
            "Overall Admissibility",
            "Category Breakdown",
            "Governing Residuals",
            "Constitutive Residuals",
            "Governing Residual",
            "Constitutive Residual",
        }
        assert {t for t in _monitor_flat_subplot_titles(n_rows=5, is_eval=False, nr_panel_title="NR") if t} == expect_tr
        expect_ev = {
            "Category Breakdown",
            "Normalized Residuals",
            "Governing Residual",
            "Constitutive Residual",
        }
        assert {t for t in _monitor_flat_subplot_titles(n_rows=4, is_eval=True, nr_panel_title="Normalized Residuals") if t} == expect_ev

        log = [{"index": 0, "rms": {"laws/a": 1.0, "constitutive/m/c": 0.5}, "scale": {}}]
        x = np.linspace(0, 1, 8)
        spatial_law = {"x": x, "values": {"laws/a": np.ones(8) * 0.01}}
        spatial_c = {"x": x, "values": {"constitutive/m/c": np.ones(8) * 0.5}}
        fig_tr = visualize(
            log,
            backend="plotly",
            mode="training",
            dashboard_mode="single-figure",
            spatial_law_panel=spatial_law,
            spatial_rnorm_panel=spatial_c,
        )
        ann_tr = {str(getattr(a, "text", "") or "") for a in (fig_tr.layout.annotations or [])}
        assert expect_tr <= ann_tr
        fig_ev = visualize(
            log,
            backend="plotly",
            mode="eval",
            dashboard_mode="single-figure",
            spatial_law_panel=spatial_law,
            spatial_rnorm_panel=spatial_c,
        )
        ann_ev = {str(getattr(a, "text", "") or "") for a in (fig_ev.layout.annotations or [])}
        assert expect_ev <= ann_ev
        hms = [t for t in fig_tr.data if getattr(t, "type", None) == "heatmap" and getattr(t, "colorbar", None)]
        assert hms
        cb = hms[0].to_plotly_json().get("colorbar") or {}
        cb_title = cb.get("title")
        if isinstance(cb_title, dict):
            cb_title = cb_title.get("text") or ""
        cb_title = str(cb_title or "")
        assert SPATIAL_HEATMAP_COLORBAR_TITLE_LOG.lower() in cb_title.lower()
        ht = str(getattr(hms[0], "hovertemplate", "") or "")
        assert SPATIAL_HEATMAP_COLORBAR_TITLE_LOG.lower() in ht.lower() and "%{z" in ht

    def test_visualize_single_figure_spatial_heatmaps_have_independent_colorbar_ranges(self):
        pytest.importorskip("plotly")
        import numpy as np

        log = [{"index": 0, "rms": {"laws/a": 1.0, "constitutive/m/c": 0.5}, "scale": {}}]
        x = np.linspace(0, 1, 8)
        spatial_law = {"x": x, "values": {"laws/a": np.ones(8) * 0.01}}
        spatial_c = {"x": x, "values": {"constitutive/m/c": np.ones(8) * 0.5}}
        fig = visualize(
            log,
            backend="plotly",
            mode="training",
            dashboard_mode="single-figure",
            spatial_law_panel=spatial_law,
            spatial_rnorm_panel=spatial_c,
        )
        hms = [
            t
            for t in fig.data
            if getattr(t, "type", None) == "heatmap"
            and isinstance(getattr(t, "meta", None), dict)
            and int(t.meta.get("subplot_row", 0)) == 5
        ]
        assert len(hms) == 2
        domains = []
        for hm in hms:
            xref = getattr(hm, "xaxis", "x")
            axis_name = "xaxis" if xref == "x" else f"xaxis{xref[1:]}"
            dom = getattr(getattr(fig.layout, axis_name), "domain")
            domains.append((float(dom[0]), float(dom[1])))
        widths = [b - a for a, b in domains]
        assert min(widths) > 0.4
        assert abs(widths[0] - widths[1]) < 1e-9
        z0 = (getattr(hms[0], "zmin", None), getattr(hms[0], "zmax", None))
        z1 = (getattr(hms[1], "zmin", None), getattr(hms[1], "zmax", None))
        assert z0[0] is not None and z0[1] is not None
        assert z1[0] is not None and z1[1] is not None
        assert z0 != z1

    def test_visualize_constitutive_dissonance_titles_term_and_slice_context(self):
        pytest.importorskip("plotly")
        import numpy as np

        nx, ny = 4, 3
        pred = np.stack([np.ones((ny, nx)), np.full((ny, nx), 3.0)], axis=0)
        implied = np.stack([np.ones((ny, nx)), np.full((ny, nx), 2.0)], axis=0)
        raw = pred - implied
        log = [
            {
                "index": 0,
                "rms": {
                    "laws/a": 1.0,
                    "constitutive/thermal_diffusivity/law_fourier_conduction/implied_delta": 0.5,
                },
                "scale": {},
                "coord_snapshot": {
                    "t": [0.0, 1.0],
                    "x": list(np.linspace(0, 1, nx)),
                    "y": list(np.linspace(0, 1, ny)),
                },
            }
        ]
        residuals = {
            "closure_debug": {
                "thermal_diffusivity/law_fourier_conduction": {
                    "pred": pred,
                    "implied": implied,
                    "raw": raw,
                    "scale_a": None,
                    "scale_b": None,
                    "ref": None,
                    "mode": "subtract",
                    "output_key": "alpha",
                    "law_name": "fourier_conduction",
                    "model_name": "thermal_diffusivity",
                }
            }
        }
        fig = visualize(
            log,
            backend="plotly",
            mode="training",
            dashboard_mode="single-figure",
            residuals=residuals,
        )
        ann = {str(getattr(a, "text", "") or "") for a in (fig.layout.annotations or [])}
        assert "Constitutive Divergence (Thermal Diffusivity)" in ann
        assert "Constitutive Dissonance (max t, worst slice)" in ann
        dissonance_subplot = fig.get_subplot(6, 5)
        xaxis_title = str(getattr(getattr(dissonance_subplot.xaxis, "title", None), "text", "") or "")
        assert xaxis_title == "Position x"
        assert "(0 to L)" not in xaxis_title
        yaxis_title = str(getattr(getattr(dissonance_subplot.yaxis, "title", None), "text", "") or "")
        assert yaxis_title == "Thermal Diffusivity"

        assert fig.layout.showlegend is False
        line_traces = [tr for tr in fig.data if getattr(tr, "name", "") in {"Model", "Implied"}]
        assert len(line_traces) == 2
        assert all(getattr(tr, "showlegend", None) is False for tr in line_traces)
        legend_anns = [
            a
            for a in (fig.layout.annotations or [])
            if str(getattr(a, "text", "") or "") in {"Model", "Implied"}
        ]
        assert {str(getattr(a, "text", "") or "") for a in legend_anns} == {"Model", "Implied"}
        assert all(str(getattr(a, "xref", "") or "").endswith(" domain") for a in legend_anns)
        assert all(str(getattr(a, "yref", "") or "").endswith(" domain") for a in legend_anns)
        assert all(0.30 < float(getattr(a, "x", 0.0)) < 0.70 for a in legend_anns)
        assert all(0.90 < float(getattr(a, "y", 0.0)) < 1.0 for a in legend_anns)
        legend_shapes = [
            s
            for s in (fig.layout.shapes or [])
            if str(getattr(s, "xref", "") or "").endswith(" domain")
            and str(getattr(s, "yref", "") or "").endswith(" domain")
            and 0.90 < float(getattr(s, "y0", 0.0)) < 1.0
        ]
        assert len(legend_shapes) >= 2

    def test_visualize_forensic_tab_heatmap_has_data_driven_zlim(self):
        pytest.importorskip("plotly")
        log = [
            {"index": 0, "rms": {"laws/a": 1.0, "constitutive/b/implied_delta": 0.5}, "scale": {"laws/a": 2.0}},
            {"index": 1, "rms": {"laws/a": 0.2, "constitutive/b/implied_delta": 0.25}, "scale": {"laws/a": 2.0}},
        ]
        out = visualize(log, backend="plotly", mode="training", dashboard_mode="dash-tabs")
        ff = (out.get("tabs") or {}).get("forensic_heatmaps")
        assert ff is not None
        hm = next((t for t in ff.data if getattr(t, "type", None) == "heatmap"), None)
        assert hm is not None
        assert getattr(hm, "zmin", None) is not None
        assert getattr(hm, "zmax", None) is not None

    def test_visualize_training_kpi_domain_does_not_overlap_overall_chart(self):
        pytest.importorskip("plotly")
        log = [
            {"index": 0, "rms": {"laws/a": 1.0, "constitutive/b/implied_delta": 0.5}, "scale": {}},
            {"index": 1, "rms": {"laws/a": 0.5, "constitutive/b/implied_delta": 0.25}, "scale": {}},
        ]
        fig = visualize(log, backend="plotly", mode="training", dashboard_mode="single-figure", baseline_score=0.7)
        assert fig is not None
        ind = next((tr for tr in fig.data if getattr(tr, "type", None) == "indicator"), None)
        assert ind is not None
        kpi_y0, kpi_y1 = float(ind.domain.y[0]), float(ind.domain.y[1])
        trend_trace = next(
            (
                tr
                for tr in fig.data
                if getattr(tr, "type", None) == "scatter"
                and getattr(tr, "name", "") == "Overall Admissibility"
            ),
            None,
        )
        assert trend_trace is not None
        trend_yaxis_ref = getattr(trend_trace, "yaxis", "y")
        trend_axis_name = "yaxis" if trend_yaxis_ref == "y" else f"yaxis{trend_yaxis_ref[1:]}"
        trend_axis = getattr(fig.layout, trend_axis_name, None)
        assert trend_axis is not None and getattr(trend_axis, "domain", None) is not None
        ty0, ty1 = float(trend_axis.domain[0]), float(trend_axis.domain[1])
        assert kpi_y1 <= ty0 or kpi_y0 >= ty1

    def test_visualize_training_high_variance_log_no_pink_overlay(self):
        """Volatile admissibility logs still build single-figure; no high-variance pink background tint."""
        pytest.importorskip("plotly")
        log = [
            {"index": i, "rms": {"laws/a": (10.0 if i % 2 == 0 else 0.01)}, "scale": {"laws/a": 1.0}}
            for i in range(6)
        ]
        fig = visualize(log, backend="plotly", mode="training", dashboard_mode="single-figure")
        assert fig is not None
        shapes = list(getattr(fig.layout, "shapes", []) or [])
        assert not any(
            getattr(s, "type", None) == "rect"
            and abs(float(getattr(s, "opacity", 0) or 0) - 0.07) < 1e-6
            for s in shapes
        )

    def test_visualize_overall_admissibility_y_grid_and_white_underlay(self):
        """Trend panel: major y-grid, full-domain white underlay, no pink opacity-0.07 overlay."""
        pytest.importorskip("plotly")
        from moju.monitor.visualize_plotly import _ENTERPRISE_THEME

        log = [
            {"index": 0, "rms": {"laws/a": 1.0, "constitutive/m/c": 0.5}, "scale": {}},
            {"index": 1, "rms": {"laws/a": 0.5, "constitutive/m/c": 0.25}, "scale": {}},
        ]
        fig = visualize(log, backend="plotly", mode="training", dashboard_mode="single-figure")
        trend_trace = next(
            (
                tr
                for tr in fig.data
                if getattr(tr, "type", None) == "scatter"
                and getattr(tr, "name", "") == "Overall Admissibility"
                and getattr(tr, "mode", "") == "lines"
            ),
            None,
        )
        assert trend_trace is not None
        yref = getattr(trend_trace, "yaxis", "y")
        yaxis_name = "yaxis" if yref == "y" else f"yaxis{yref[1:]}"
        yax = getattr(fig.layout, yaxis_name, None)
        assert yax is not None
        assert getattr(yax, "showgrid", None) is True
        assert getattr(yax, "gridcolor", None) == _ENTERPRISE_THEME["grid_color"]
        shapes = list(getattr(fig.layout, "shapes", []) or [])
        full_domain_white = [
            s
            for s in shapes
            if getattr(s, "type", None) == "rect"
            and str(getattr(s, "xref", "") or "").endswith(" domain")
            and str(getattr(s, "yref", "") or "").endswith(" domain")
            and float(getattr(s, "x0", -1)) == 0.0
            and float(getattr(s, "x1", -1)) == 1.0
            and float(getattr(s, "y0", -1)) == 0.0
            and float(getattr(s, "y1", -1)) == 1.0
            and str(getattr(s, "fillcolor", "") or "").lower() in {"#ffffff", "rgb(255, 255, 255)"}
        ]
        assert len(full_domain_white) == 1
        assert not any(
            getattr(s, "type", None) == "rect"
            and abs(float(getattr(s, "opacity", 0) or 0) - 0.07) < 1e-6
            for s in shapes
        )

    def test_visualize_forensic_tab_heatmap_colorbar_uses_domain_alignment(self):
        pytest.importorskip("plotly")
        log = [
            {"index": 0, "rms": {"laws/a": 1.0, "constitutive/b/implied_delta": 0.5}, "scale": {}},
            {"index": 1, "rms": {"laws/a": 0.5, "constitutive/b/implied_delta": 0.25}, "scale": {}},
        ]
        out = visualize(log, backend="plotly", mode="training", dashboard_mode="dash-tabs")
        figf = (out.get("tabs") or {}).get("forensic_heatmaps")
        assert figf is not None
        hm = next((t for t in figf.data if getattr(t, "type", None) == "heatmap"), None)
        assert hm is not None
        cbd = hm.to_plotly_json().get("colorbar") or {}
        assert cbd.get("xanchor") == "left"
        assert cbd.get("x") is not None and float(cbd["x"]) <= 1.0

    def test_visualize_spatial_right_heatmap_insets_x_domain_for_colorbar(self):
        pytest.importorskip("plotly")
        import numpy as np

        log = [
            {"index": 0, "rms": {"laws/a": 1.0, "constitutive/m/c": 0.5}, "scale": {}},
            {"index": 1, "rms": {"laws/a": 0.5, "constitutive/m/c": 0.25}, "scale": {}},
        ]
        residuals = {"laws": {"a": np.ones(5) * 0.1}, "constitutive": {"m/c": np.ones(5) * 0.05}}
        fig = visualize(log, backend="plotly", mode="training", residuals=residuals)
        right_hm = next(
            (
                t
                for t in fig.data
                if getattr(t, "type", None) == "heatmap"
                and isinstance(getattr(t, "meta", None), dict)
                and int(t.meta.get("subplot_col", 0)) == 5
            ),
            None,
        )
        assert right_hm is not None
        xref = getattr(right_hm, "xaxis", "x")
        ax_name = "xaxis" if xref == "x" else f"xaxis{xref[1:]}"
        xax = getattr(fig.layout, ax_name, None)
        assert xax is not None and xax.domain is not None
        assert float(xax.domain[1]) < 0.98
        cbx = float((right_hm.to_plotly_json().get("colorbar") or {}).get("x", 0))
        assert cbx > float(xax.domain[1])

    def test_visualize_training_indicator_traces_have_no_cartesian_axis_refs(self):
        """go.Indicator must not receive xaxis/yaxis (PlotlyKeyError in some subplot paths)."""
        pytest.importorskip("plotly")
        log = [
            {"index": 0, "rms": {"laws/a": 1.0, "constitutive/b/implied_delta": 0.5}, "scale": {}},
            {"index": 1, "rms": {"laws/a": 0.5, "constitutive/b/implied_delta": 0.25}, "scale": {}},
        ]
        fig = visualize(log, backend="plotly", mode="training", dashboard_mode="single-figure")
        assert fig is not None
        for tr in fig.data:
            if getattr(tr, "type", None) == "indicator":
                assert getattr(tr, "xaxis", None) is None
                assert getattr(tr, "yaxis", None) is None

    def test_visualize_eval_mode_style_parity_threshold_and_watermark(self):
        pytest.importorskip("plotly")
        log = [{"index": 0, "rms": {"laws/a": 1.0, "constitutive/b/implied_delta": 0.5}, "scale": {"laws/a": 2.0}}]
        fig = visualize(log, backend="plotly", mode="eval")
        assert fig is not None
        assert getattr(fig.layout, "paper_bgcolor", None) == "#ffffff"
        assert getattr(fig.layout, "plot_bgcolor", None) == "#ffffff"
        anns = list(getattr(fig.layout, "annotations", []) or [])
        assert not any("Ifimo Lab: Moju Forensic Suite" in str(getattr(a, "text", "")) for a in anns)
        fig_b = visualize(log, backend="plotly", mode="eval", show_branding=True)
        anns_b = list(getattr(fig_b.layout, "annotations", []) or [])
        assert any("Ifimo Lab: Moju Forensic Suite" in str(getattr(a, "text", "")) for a in anns_b)
        shapes = list(getattr(fig.layout, "shapes", []) or [])
        _thr_x = ADM_HIGH_THRESHOLD * 100.0
        assert any(
            getattr(s, "type", None) == "line" and abs(float(getattr(s, "x0", -1)) - _thr_x) < 1e-6
            for s in shapes
        )
        hovers = [str(getattr(tr, "hovertemplate", "")) for tr in fig.data]
        assert any("scale_k=" in ht for ht in hovers)

    def test_compute_log_step_metrics_includes_data_category(self):
        log = [
            {
                "index": 0,
                "rms": {"laws/a": 1.0, "data/T": 0.5},
                "scale": {"laws/a": 1.0, "data/T": 1.0},
            }
        ]
        from moju.monitor.auditor import _compute_log_step_metrics

        m = _compute_log_step_metrics(log)
        assert "data" in m[0]["category_admissibility_score"]
        assert "laws" in m[0]["category_admissibility_score"]


class TestResidualEngineStateBuilder:
    def test_groups_enrich_state(self, rtol, atol):
        core = ResidualEngine(
            constants={"L": 0.1},
            groups=[{"name": "re", "state_map": {"u": "u", "L": "L", "rho": "rho", "mu": "mu"}, "output_key": "re"}],
        )
        state_pred = {"u": 1.0, "rho": 1000.0, "mu": 1.0}
        state = core._state_builder(state_pred)
        assert "re" in state
        assert jnp.allclose(state["re"], 100.0, rtol=rtol, atol=atol)


class TestCustomFn:
    def test_custom_law_fn(self, rtol, atol):
        def my_residual(x):
            return x - 1.0

        core = ResidualEngine(
            laws=[{"name": "my_law", "state_map": {"x": "x"}, "fn": my_residual}],
        )
        state_pred = {"x": jnp.array(2.0)}
        residuals = core.compute_residuals(state_pred)
        assert jnp.allclose(residuals["laws"]["my_law"], 1.0, rtol=rtol, atol=atol)
        assert jnp.allclose(build_loss(residuals), 1.0, rtol=rtol, atol=atol)

    def test_constitutive_custom_closure(self, rtol, atol):
        core = ResidualEngine(
            laws=[],
            constitutive_custom=[{"name": "my_c", "fn": lambda s, c: s["a"] - 2.0 * s["b"]}],
        )
        state_pred = {"a": jnp.array(4.0), "b": jnp.array(2.0)}
        residuals = core.compute_residuals(state_pred)
        assert jnp.allclose(residuals["constitutive"]["custom/my_c"], 0.0, rtol=rtol, atol=atol)

    def test_custom_group_fn_in_state(self, rtol, atol):
        def my_group(a, b):
            return a * b

        core = ResidualEngine(
            groups=[{"name": "my_ab", "state_map": {"a": "a", "b": "b"}, "output_key": "ab", "fn": my_group}],
        )
        state = core._state_builder({"a": jnp.array(3.0), "b": jnp.array(4.0)})
        assert jnp.allclose(state["ab"], 12.0, rtol=rtol, atol=atol)


class TestClosureDiscrepancyNormalize:
    def test_symmetric_zero_when_pred_matches_implied(self, rtol, atol):
        pred = jnp.array(3.0)
        implied = jnp.array(3.0)
        r = apply_closure_discrepancy_normalize(pred - implied, pred, implied)
        assert jnp.allclose(r, 0.0, rtol=rtol, atol=atol)

    def test_symmetric_scale_doubling_invariant(self, rtol, atol):
        pred = jnp.array(4.0)
        implied = jnp.array(2.0)
        r1 = apply_closure_discrepancy_normalize(pred - implied, pred, implied)
        r2 = apply_closure_discrepancy_normalize(
            2 * (pred - implied), 2 * pred, 2 * implied
        )
        assert jnp.allclose(r1, r2, rtol=rtol, atol=atol)

    def test_ref_scale_uses_ref_denominator(self, rtol, atol):
        pred = jnp.array(10.0)
        implied = jnp.array(6.0)
        ref = jnp.array(2.0)
        r = apply_closure_discrepancy_normalize(pred - implied, pred, implied, ref=ref)
        assert jnp.allclose(r, 4.0 / (1e-30 + 2.0), rtol=rtol, atol=atol)


class TestImpliedDeltaClosure:
    def test_compute_implied_delta_value_key(self, rtol, atol):
        fn, arg_names = MODEL_FNS["ideal_gas_rho"]
        P, R, T = jnp.array(1e5), jnp.array(287.0), jnp.array(300.0)
        rho_m = Models.ideal_gas_rho(P, R, T)
        merged = {"P": P, "R": R, "T": T, "rho_alt": rho_m}
        r = compute_implied_delta(
            fn=fn,
            arg_names=arg_names,
            state_map={"P": "P", "R": "R", "T": "T"},
            state_pred=merged,
            constants={},
            implied_value_key="rho_alt",
        )
        assert r is not None
        assert jnp.allclose(r, 0.0, rtol=rtol, atol=atol)

    def test_compute_implied_delta_missing_key_returns_none(self):
        fn, arg_names = MODEL_FNS["ideal_gas_rho"]
        r = compute_implied_delta(
            fn=fn,
            arg_names=arg_names,
            state_map={"P": "P", "R": "R", "T": "T"},
            state_pred={"P": 1.0, "R": 1.0, "T": 1.0},
            constants={},
            implied_value_key="rho_missing",
        )
        assert r is None

    def test_compute_implied_delta_balance_fourier(self, rtol, atol):
        fn, arg_names = MODEL_FNS["thermal_diffusivity"]
        k, rho, cp = jnp.array(1.0), jnp.array(1.0), jnp.array(1.0)
        alpha_m = fn(k, rho, cp)
        T_lap = jnp.array(1.0)
        T_t = alpha_m * T_lap
        merged = {"k": k, "rho": rho, "cp": cp, "T_t": T_t, "T_xx": T_lap}

        def balance(st, _c, pred):
            tt = jnp.asarray(st["T_t"])
            lap = jnp.asarray(st["T_xx"])
            p = jnp.asarray(pred)
            d = p * lap
            return tt - d, tt, d

        r = compute_implied_delta(
            fn=fn,
            arg_names=arg_names,
            state_map={"k": "k", "rho": "rho", "cp": "cp"},
            state_pred=merged,
            constants={},
            implied_balance_fn=balance,
        )
        assert r is not None
        assert jnp.allclose(r, 0.0, rtol=rtol, atol=atol)

    def test_compute_implied_delta_rejects_multiple_implied_modes(self):
        fn, arg_names = MODEL_FNS["ideal_gas_rho"]
        with pytest.raises(ValueError, match="at most one of implied"):
            compute_implied_delta(
                fn=fn,
                arg_names=arg_names,
                state_map={"P": "P", "R": "R", "T": "T"},
                state_pred={"P": 1.0, "R": 1.0, "T": 1.0, "x": 1.0},
                constants={},
                implied_value_key="x",
                implied_balance_fn=lambda s, c, p: (0.0, 0.0, 0.0),
            )

    def test_engine_implied_delta_ideal_gas(self, rtol, atol):
        P, R = jnp.array(101325.0), jnp.array(287.0)
        T = jnp.array(290.0)
        rho = Models.ideal_gas_rho(P, R, T)
        core = ResidualEngine(
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
        res = core.compute_residuals(
            {"P": P, "R": R, "T": T, "rho": rho, "rho_implied": rho}
        )
        assert "constitutive" in res
        assert jnp.allclose(
            res["constitutive"]["ideal_gas_rho/implied_delta"], 0.0, rtol=rtol, atol=atol
        )

    def test_ref_delta_without_predicted_spatial(self, rtol, atol):
        """ref_delta runs when state_ref is set even if predicted_* are empty."""
        core = ResidualEngine(
            laws=[],
            constitutive_audit=[
                {
                    "name": "ideal_gas_rho",
                    "output_key": "rho",
                    "state_map": {"P": "P", "R": "R", "T": "T"},
                }
            ],
        )
        P, R, T = jnp.array(1e5), jnp.array(287.0), jnp.array(300.0)
        rho1 = Models.ideal_gas_rho(P, R, T)
        rho2 = Models.ideal_gas_rho(P * 1.01, R, T)
        state_pred = {"P": P, "R": R, "T": T, "rho": rho1}
        state_ref = {"P": P * 1.01, "R": R, "T": T, "rho": rho2}
        res = core.compute_residuals(
            state_pred, state_ref=state_ref, run_mode="eval"
        )
        assert "ideal_gas_rho/ref_delta" in res["constitutive"]

    def test_implied_both_key_and_fn_raises(self):
        with pytest.raises(ValueError, match="only one of implied"):
            ResidualEngine(
                laws=[],
                constitutive_audit=[
                    {
                        "name": "ideal_gas_rho",
                        "output_key": "rho",
                        "state_map": {"P": "P", "R": "R", "T": "T"},
                        "implied_value_key": "x",
                        "implied_fn": lambda s, c: s.get("x"),
                    }
                ],
            )

    def test_law_linked_implied_contributes_to_audit_scoring(self):
        """Law-linked implied rows are included in constitutive category/overall scoring."""
        engine = ResidualEngine(
            laws=[
                {
                    "name": "fourier_conduction",
                    "state_map": {
                        "T_t": "T_t",
                        "T_laplacian": "T_laplacian",
                        "fo": "fo",
                        "t": "t",
                        "L": "L",
                    },
                }
            ],
            groups=[{"name": "fo", "output_key": "fo", "state_map": {"alpha": "alpha", "t": "t", "L": "L"}}],
            law_implied_audits=True,
        )
        state_pred = {
            "T_t": jnp.array([1.0]),
            "T_laplacian": jnp.array([1.0]),
            "alpha": jnp.array([1.0]),
            "t": jnp.array([1.0]),
            "L": jnp.array([1.0]),
            "k": jnp.array([1.0]),
            "rho": jnp.array([1.0]),
            "cp": jnp.array([1.0]),
        }
        _ = engine.compute_residuals(state_pred)
        rep = audit(engine.log)
        assert "constitutive" in rep["per_category"]
        assert rep["per_category"]["constitutive"] >= 0.0

class TestRegistryHelpers:
    def test_list_constitutive_models(self):
        names = list_constitutive_models()
        assert "sutherland_mu" in names
        assert "thermal_diffusivity" in names
        assert "smagorinsky_nu_t" in names
        assert "k_epsilon_nu_t" in names
        assert "k_omega_nu_t" in names

    def test_list_scaling_closure_ids(self):
        ids = list_scaling_closure_ids()
        assert "pe" in ids
        assert "fo" in ids


class TestMonitorConfig:
    def test_to_from_dict_roundtrip(self):
        cfg = MonitorConfig(
            constants={"cp": 1.0},
            laws=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_xx"}}],
            constitutive_audit=[
                AuditSpec(
                    name="sutherland_mu",
                    output_key="mu",
                    state_map={"T": "T", "mu0": "mu0", "T0": "T0", "S": "S"},
                )
            ],
        )
        d = cfg.to_dict()
        cfg2 = MonitorConfig.from_dict(d)
        assert cfg2.to_dict() == d

    def test_from_dict_rejects_scaling_audit(self):
        with pytest.raises(ValueError, match="scaling_audit"):
            MonitorConfig.from_dict({"laws": [], "scaling_audit": []})


class TestRequiredKeys:
    def test_required_state_keys_union(self):
        engine = ResidualEngine(
            laws=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_xx"}}],
            groups=[
                {
                    "name": "pe",
                    "output_key": "Pe",
                    "state_map": {"re": "Re", "pr": "Pr"},
                }
            ],
        )
        state_keys = engine.required_state_keys()
        assert "phi_xx" in state_keys
        assert "Re" in state_keys and "Pr" in state_keys and "Pe" in state_keys


class TestKwargsFromStateLawHint:
    def test_missing_derived_key_includes_fd_hint(self):
        from moju.monitor.auditor import _kwargs_from_state

        with pytest.raises(KeyError, match="fill_law_fd"):
            _kwargs_from_state(
                {},
                {},
                {"T_laplacian": "T_laplacian"},
                law_context="fourier_conduction",
            )

    def test_missing_non_derived_key_omits_fd_sentence(self):
        from moju.monitor.auditor import _kwargs_from_state

        with pytest.raises(KeyError) as ei:
            _kwargs_from_state(
                {},
                {},
                {"Fo": "Fo"},
                law_context="fourier_conduction",
            )
        assert "fill_law_fd" not in str(ei.value)

    def test_placeholder_string_for_derived_key_treated_as_missing(self):
        from moju.monitor.auditor import _kwargs_from_state

        with pytest.raises(KeyError, match="fill_law_fd"):
            _kwargs_from_state(
                {},
                {"T_laplacian": "T_laplacian"},
                {"T_laplacian": "T_laplacian"},
                law_context="fourier_conduction",
            )

    def test_non_placeholder_string_raises_clear_type_error(self):
        from moju.monitor.auditor import _kwargs_from_state

        with pytest.raises(TypeError, match="resolved to string value"):
            _kwargs_from_state(
                {},
                {"T_laplacian": "oops"},
                {"T_laplacian": "T_laplacian"},
                law_context="fourier_conduction",
            )
