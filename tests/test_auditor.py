"""Tests for ResidualEngine, build_loss, audit, visualize, constitutive/scaling closures."""

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
    list_pi_constant_group_names,
    list_scaling_closure_ids,
    visualize,
)
from moju.monitor.auditor import DEFAULT_VISUALIZE_TITLE_TEST, DEFAULT_VISUALIZE_TITLE_TRAINING
from moju.monitor.closure_registry import MODEL_FNS, compute_implied_delta
from moju.piratio.models import Models


class TestAdmissibilityLevel:
    def test_four_levels(self):
        assert admissibility_level(0.0) == "Non-Admissible"
        assert admissibility_level(0.39) == "Non-Admissible"
        assert admissibility_level(0.40) == "Low Admissibility"
        assert admissibility_level(0.69) == "Low Admissibility"
        assert admissibility_level(0.70) == "Moderate Admissibility"
        assert admissibility_level(0.89) == "Moderate Admissibility"
        assert admissibility_level(0.90) == "High Admissibility"
        assert admissibility_level(1.0) == "High Admissibility"

    def test_non_finite_unknown(self):
        assert admissibility_level(float("nan")) == "Unknown"
        assert admissibility_level(float("inf")) == "Unknown"


class TestNanTolerantAuditMetrics:
    def test_compute_log_step_metrics_geometric_mean_ignores_nan_keys(self):
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
        assert math.isclose(m[0]["category_admissibility_score"]["constitutive"], 0.5, rel_tol=1e-9)
        assert "laws" not in m[0]["category_admissibility_score"]
        assert math.isclose(m[0]["overall_admissibility_score"], 0.5, rel_tol=1e-9)

    def test_compute_log_step_metrics_omits_category_when_all_nan(self):
        from moju.monitor.auditor import _compute_log_step_metrics

        log = [
            {
                "index": 0,
                "rms": {"constitutive/x/implied_delta": float("nan")},
                "scale": {"constitutive/x/implied_delta": 1.0},
            }
        ]
        m = _compute_log_step_metrics(log)
        assert "constitutive" not in m[0]["category_admissibility_score"]
        import math

        assert math.isnan(m[0]["overall_admissibility_score"])

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

    def test_constitutive_sutherland_closure(self, rtol, atol):
        core = ResidualEngine(
            constants={"mu0": 1.8e-5, "T0": 273.0, "S": 110.4},
            laws=[],
            constitutive_audit=[
                {
                    "name": "sutherland_mu",
                    "output_key": "mu",
                    "state_map": {"T": "T", "mu0": "mu0", "T0": "T0", "S": "S"},
                    "predicted_spatial": ["T"],
                }
            ],
        )
        T = 300.0
        mu_true = 1.8e-5 * (T / 273) ** 1.5 * (273 + 110.4) / (T + 110.4)
        state_pred = {
            "mu": mu_true,
            "T": T,
            "d_T_dx": jnp.array(1.0),
            "d_mu_dx": jnp.array(0.0),  # inconsistent on purpose
        }
        residuals = core.compute_residuals(state_pred)
        assert "constitutive" in residuals
        assert "sutherland_mu/chain_dx" in residuals["constitutive"]
        assert abs(float(residuals["constitutive"]["sutherland_mu/chain_dx"])) > 0.0

    def test_scaling_pe_identity_zero(self, rtol, atol):
        core = ResidualEngine(
            laws=[],
            scaling_audit=[
                {
                    "name": "pe",
                    "output_key": "Pe",
                    "state_map": {"re": "Re", "pr": "Pr"},
                    "predicted_spatial": ["Re"],
                }
            ],
        )
        Re, Pr = 100.0, 0.7
        # Provide d_Pe_dx consistent with chain rule for Pe = Re*Pr and dRe/dx = 1, dPr/dx = 0.
        state_pred = {"Pe": Re * Pr, "Re": Re, "Pr": Pr, "d_Re_dx": 1.0, "d_Pe_dx": Pr}
        residuals = core.compute_residuals(state_pred)
        assert "scaling" in residuals
        assert jnp.allclose(residuals["scaling"]["pe/chain_dx"], 0.0, rtol=rtol, atol=atol)

    def test_scaling_pe_weak_chain_dx_weighted_rms(self, rtol, atol):
        core = ResidualEngine(
            laws=[],
            scaling_audit=[
                {
                    "name": "pe",
                    "output_key": "Pe",
                    "state_map": {"re": "Re", "pr": "Pr"},
                    "predicted_spatial": ["Re"],
                    "closure_mode": "weak",
                    "quadrature_weights": {"x": "w_x"},
                }
            ],
        )
        # Pe = Re * Pr, Pr constant, dRe/dx = 1, but we set dPe/dx = 0 -> residual = -Pr everywhere.
        Pr = 5.0
        state_pred = {
            "Pe": jnp.array([10.0, 11.0, 12.0]),
            "Re": jnp.array([2.0, 2.0, 2.0]),
            "Pr": Pr,
            "d_Re_dx": jnp.ones((3,)),
            "d_Pe_dx": jnp.zeros((3,)),
            "w_x": jnp.array([1.0, 2.0, 1.0]),
        }
        residuals = core.compute_residuals(state_pred)
        assert "scaling" in residuals
        # Weighted RMS of constant residual -Pr is |Pr|.
        assert jnp.allclose(residuals["scaling"]["pe/chain_dx"], abs(Pr), rtol=rtol, atol=atol)

    def test_scaling_pe_identity_nonzero(self, rtol, atol):
        core = ResidualEngine(
            laws=[],
            scaling_audit=[
                {
                    "name": "pe",
                    "output_key": "Pe",
                    "state_map": {"re": "Re", "pr": "Pr"},
                    "predicted_spatial": ["Re"],
                }
            ],
        )
        state_pred = {"Pe": 100.0, "Re": 10.0, "Pr": 5.0, "d_Re_dx": 1.0, "d_Pe_dx": 0.0}
        residuals = core.compute_residuals(state_pred)
        # For Pe = Re*Pr, chain expects dPe/dx = Pr * dRe/dx = 5.
        assert jnp.allclose(residuals["scaling"]["pe/chain_dx"], -5.0, rtol=rtol, atol=atol)

    def test_state_ref_adds_data_residual(self, rtol, atol):
        core = ResidualEngine(
            laws=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_laplacian"}}],
        )
        state_pred = {"phi_laplacian": jnp.array(0.5)}
        state_ref = {"phi_laplacian": jnp.array(0.0)}
        residuals = core.compute_residuals(state_pred, state_ref=state_ref)
        assert "data" in residuals
        assert jnp.allclose(residuals["data"]["phi_laplacian"], -0.5, rtol=rtol, atol=atol)

    def test_inferred_predicted_spatial_logged(self):
        core = ResidualEngine(
            constants={"mu0": 1.8e-5, "T0": 273.0, "S": 110.4},
            laws=[],
            constitutive_audit=[
                {
                    "name": "sutherland_mu",
                    "output_key": "mu",
                    "state_map": {"T": "T", "mu0": "mu0", "T0": "T0", "S": "S"},
                    # predicted_spatial intentionally omitted to trigger inference
                }
            ],
        )
        T = 300.0
        mu_true = 1.8e-5 * (T / 273) ** 1.5 * (273 + 110.4) / (T + 110.4)
        state_pred = {"mu": mu_true, "T": T, "d_T_dx": jnp.array(1.0), "d_mu_dx": jnp.array(0.0)}
        core.compute_residuals(state_pred, collocation={"x": jnp.array([0.0])})
        assert "inferred" in core.log[-1]
        assert any("constitutive:sutherland_mu inferred predicted_spatial=['T']" in s for s in core.log[-1]["inferred"])


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

    def test_audit_export_dir_pdf_with_new_categories(self, tmp_path):
        pytest.importorskip("reportlab")
        log = [
            {"index": 0, "rms": {"laws/a": 1.0, "constitutive/m/chain_dx": 0.5, "scaling/pe/chain_dx": 0.1}},
            {"index": 1, "rms": {"laws/a": 0.5, "constitutive/m/chain_dx": 0.25, "scaling/pe/chain_dx": 0.05}},
        ]
        report = audit(log, export_dir=str(tmp_path))
        assert "per_key" in report
        dirs = [d for d in tmp_path.iterdir() if d.is_dir() and d.name.startswith("audit_")]
        assert len(dirs) == 1
        assert (dirs[0] / "report.pdf").exists()


class TestVisualize:
    def test_visualize_empty_log_returns_none(self):
        assert visualize([], backend="matplotlib") is None

    def test_visualize_backend_none_returns_none(self):
        assert visualize([{"index": 0, "rms": {"k": 1.0}}], backend="none") is None

    def test_visualize_multi_panel_figure(self):
        pytest.importorskip("matplotlib")
        log = [
            {
                "index": 0,
                "rms": {
                    "laws/a": 1.0,
                    "constitutive/m/chain_dx": 0.5,
                    "scaling/pe/chain_dx": 0.1,
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
                    "constitutive/m/chain_dx": 0.25,
                    "scaling/pe/chain_dx": 0.05,
                    "data/T": 0.1,
                },
                "scale": {},
            },
        ]
        fig = visualize(log, backend="matplotlib", mode="training")
        assert fig is not None
        # Training: top row + R_norm lines + R_norm heatmaps (+ colorbars) + optional spatial
        assert len(fig.axes) >= 6
        from matplotlib.projections.polar import PolarAxes

        assert not any(isinstance(ax, PolarAxes) for ax in fig.axes)

    def test_visualize_r_norm_scale_linear(self):
        pytest.importorskip("matplotlib")
        log = [
            {
                "index": 0,
                "rms": {
                    "laws/a": 1.0,
                    "constitutive/m/chain_dx": 0.5,
                    "scaling/pe/chain_dx": 0.1,
                },
                "scale": {},
            },
            {
                "index": 1,
                "rms": {
                    "laws/a": 0.5,
                    "constitutive/m/chain_dx": 0.25,
                    "scaling/pe/chain_dx": 0.05,
                },
                "scale": {},
            },
        ]
        fig = visualize(log, backend="matplotlib", mode="training", r_norm_scale="linear")
        assert fig is not None
        pytest.importorskip("plotly")
        fig_p = visualize(log, backend="plotly", mode="training", r_norm_scale="linear")
        assert fig_p is not None

    def test_build_monitor_visualize_bundle_and_studio_plotly_cards(self):
        pytest.importorskip("plotly")
        import numpy as np

        from moju.monitor.auditor import build_monitor_visualize_bundle
        from moju.monitor.visualize_plotly import (
            build_plotly_category_admissibility_bar_figure,
            build_plotly_law_rnorm_final_bar_figure,
            build_plotly_spatial_rnorm_heatmap_card,
        )

        log = [
            {
                "index": 0,
                "rms": {
                    "laws/a": 1.0,
                    "constitutive/m/chain_dx": 0.5,
                    "scaling/pe/chain_dx": 0.1,
                },
                "scale": {},
            },
            {
                "index": 1,
                "rms": {
                    "laws/a": 0.5,
                    "constitutive/m/chain_dx": 0.25,
                    "scaling/pe/chain_dx": 0.05,
                },
                "scale": {},
            },
        ]
        x = np.linspace(0, 1, 5)
        spatial_law = {"x": x, "values": {"laws/a": np.ones(5) * 0.2}}
        spatial_c = {"x": x, "values": {"constitutive/m/chain_dx": np.ones(5) * 0.1}}
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
        f3 = build_plotly_spatial_rnorm_heatmap_card(bundle["spatial"], colorscale="Jet")
        f4 = build_plotly_spatial_rnorm_heatmap_card(bundle["spatial_rnorm"], colorscale="Jet")
        assert f3 is not None and f4 is not None

    def test_visualize_test_mode_uses_last_log_entry(self):
        pytest.importorskip("matplotlib")
        log = [
            {"index": 0, "rms": {"laws/a": 10.0}, "scale": {"laws/a": 1.0}},
            {"index": 1, "rms": {"laws/a": 0.1}, "scale": {"laws/a": 1.0}},
        ]
        fig = visualize(log, backend="matplotlib", mode="test")
        assert fig is not None

    def test_visualize_spatial_law_panel_matplotlib(self):
        pytest.importorskip("matplotlib")
        import numpy as np

        log = [
            {"index": 0, "rms": {"laws/a": 1.0}, "scale": {"laws/a": 1.0}},
            {"index": 1, "rms": {"laws/a": 0.5}, "scale": {"laws/a": 1.0}},
        ]
        x = np.linspace(0, 1, 5)
        fig = visualize(
            log,
            backend="matplotlib",
            mode="training",
            spatial_law_panel={"x": x, "values": {"a": np.ones(5) * 0.2}},
        )
        assert fig is not None
        assert len(fig.axes) >= 7

    def test_visualize_matplotlib_training_top_axes_below_title_band(self):
        """Reserved layout rect should keep the top row of axes out of the title strip."""
        pytest.importorskip("matplotlib")
        log = [
            {
                "index": 0,
                "rms": {"laws/a": 1.0, "constitutive/m/c": 0.5},
                "scale": {},
            },
            {
                "index": 1,
                "rms": {"laws/a": 0.5, "constitutive/m/c": 0.25},
                "scale": {},
            },
        ]
        fig = visualize(log, backend="matplotlib", mode="training")
        assert fig is not None
        assert fig.axes
        max_y1 = max(ax.get_position().y1 for ax in fig.axes)
        assert max_y1 < 0.94, f"expected axes below title band, got max y1={max_y1}"

    def test_visualize_matplotlib_category_adm_bar_autoscale(self):
        pytest.importorskip("matplotlib")
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
        fig = visualize(log, backend="matplotlib", mode="training")
        assert fig is not None
        ax_cat = next(
            ax for ax in fig.axes if ax.get_title() and "Category admissibility" in ax.get_title()
        )
        x0, x1 = ax_cat.get_xlim()
        assert x1 - x0 < 0.5
        assert x1 <= 1.0 + 1e-6

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

    def test_visualize_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            visualize([{"index": 0, "rms": {"k": 1.0}, "scale": {}}], mode="invalid")

    def test_visualize_plotly_returns_figure(self):
        pytest.importorskip("plotly")
        log = [
            {
                "index": 0,
                "rms": {
                    "laws/a": 1.0,
                    "constitutive/m/chain_dx": 0.5,
                    "scaling/pe/chain_dx": 0.1,
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
                    "constitutive/m/chain_dx": 0.25,
                    "scaling/pe/chain_dx": 0.05,
                    "data/T": 0.1,
                },
                "scale": {},
            },
        ]
        fig = visualize(log, backend="plotly", mode="training")
        assert fig is not None
        assert hasattr(fig, "data")
        assert len(fig.data) >= 7
        hm = [t for t in fig.data if getattr(t, "type", None) == "heatmap"]
        assert len(hm) >= 2

    def test_visualize_plotly_default_titles(self):
        pytest.importorskip("plotly")
        log = [
            {
                "index": 0,
                "rms": {
                    "laws/a": 1.0,
                    "constitutive/m/chain_dx": 0.5,
                    "scaling/pe/chain_dx": 0.1,
                },
                "scale": {},
            },
            {
                "index": 1,
                "rms": {
                    "laws/a": 0.5,
                    "constitutive/m/chain_dx": 0.25,
                    "scaling/pe/chain_dx": 0.05,
                },
                "scale": {},
            },
        ]
        fig_tr = visualize(log, backend="plotly", mode="training")
        assert DEFAULT_VISUALIZE_TITLE_TRAINING in (fig_tr.layout.title.text or "")
        fig_te = visualize(log, backend="plotly", mode="test")
        assert DEFAULT_VISUALIZE_TITLE_TEST in (fig_te.layout.title.text or "")

    def test_visualize_plotly_figure_title_override(self):
        pytest.importorskip("plotly")
        log = [
            {"index": 0, "rms": {"laws/a": 1.0, "constitutive/m/c": 0.5}, "scale": {}},
            {"index": 1, "rms": {"laws/a": 0.5, "constitutive/m/c": 0.25}, "scale": {}},
        ]
        fig = visualize(log, backend="plotly", mode="training", figure_title="Custom dashboard title")
        assert "Custom dashboard title" in (fig.layout.title.text or "")

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

    def test_scaling_custom_closure(self, rtol, atol):
        core = ResidualEngine(
            laws=[],
            scaling_custom=[{"name": "diff", "fn": lambda s, c: s["x"] - s["y"]}],
        )
        state_pred = {"x": jnp.array(1.0), "y": jnp.array(3.0)}
        residuals = core.compute_residuals(state_pred)
        assert jnp.allclose(residuals["scaling"]["custom/diff"], -2.0, rtol=rtol, atol=atol)

    def test_custom_group_fn_in_state(self, rtol, atol):
        def my_group(a, b):
            return a * b

        core = ResidualEngine(
            groups=[{"name": "my_ab", "state_map": {"a": "a", "b": "b"}, "output_key": "ab", "fn": my_group}],
        )
        state = core._state_builder({"a": jnp.array(3.0), "b": jnp.array(4.0)})
        assert jnp.allclose(state["ab"], 12.0, rtol=rtol, atol=atol)


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
                    "predicted_spatial": [],
                    "predicted_temporal": [],
                }
            ],
        )
        P, R, T = jnp.array(1e5), jnp.array(287.0), jnp.array(300.0)
        rho1 = Models.ideal_gas_rho(P, R, T)
        rho2 = Models.ideal_gas_rho(P * 1.01, R, T)
        state_pred = {"P": P, "R": R, "T": T, "rho": rho1}
        state_ref = {"P": P * 1.01, "R": R, "T": T, "rho": rho2}
        res = core.compute_residuals(state_pred, state_ref=state_ref)
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
                    predicted_temporal=["T"],
                )
            ],
        )
        d = cfg.to_dict()
        cfg2 = MonitorConfig.from_dict(d)
        assert cfg2.to_dict() == d


class TestRequiredKeys:
    def test_required_state_and_derivative_keys(self):
        engine = ResidualEngine(
            laws=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_xx"}}],
            scaling_audit=[
                {
                    "name": "pe",
                    "output_key": "Pe",
                    "state_map": {"re": "Re", "pr": "Pr"},
                    "predicted_spatial": ["Re", "Pr"],
                    "predicted_temporal": ["Re"],
                }
            ],
        )
        state_keys = engine.required_state_keys()
        assert "phi_xx" in state_keys
        assert "Re" in state_keys and "Pr" in state_keys and "Pe" in state_keys

        deriv_keys = engine.required_derivative_keys()
        assert "d_Pe_dx" in deriv_keys
        assert "d_Re_dx" in deriv_keys
        assert "d_Pr_dx" in deriv_keys
        assert "d_Pe_dt" in deriv_keys
        assert "d_Re_dt" in deriv_keys

    def test_default_inference_uses_primary_fields(self, rtol, atol):
        engine = ResidualEngine(
            laws=[],
            primary_fields=["u", "T"],
            scaling_audit=[
                {
                    "name": "pe",
                    "output_key": "Pe",
                    "state_map": {"re": "u", "pr": "T"},
                }
            ],
        )
        # No predicted_* provided: with collocation including x and t, engine should pick 'u' first.
        state_pred = {"u": 1.0, "T": 2.0, "Pe": 2.0, "d_u_dx": 0.0, "d_Pe_dx": 0.0}
        residuals = engine.compute_residuals(state_pred, collocation={"x": 0.0, "t": 0.0})
        # With u chosen for predicted_spatial, chain_dx exists only if derivative keys are present; here it is present.
        assert "scaling" in residuals


def _re_pi_spec(*, compare_keys=("out",), scale_c=10.0):
    return {
        "name": "re",
        "output_key": "Re",
        "state_map": {"u": "u", "L": "L", "rho": "rho", "mu": "mu"},
        "invariance_pi_constant": True,
        "invariance_compare_keys": list(compare_keys),
        "invariance_scale_c": scale_c,
    }


class TestPiConstantClosure:
    def test_list_pi_constant_group_names(self):
        names = list_pi_constant_group_names()
        assert "re" in names and "pr" in names and "pe" in names
        assert "st" in names and "gr" in names and "we" in names
        assert len(names) == 22

    def test_apply_pi_constant_recipe_re_preserves_re(self):
        from moju.monitor.pi_constant_recipes import (
            GROUP_PI_CONSTANT_RECIPES,
            apply_pi_constant_recipe,
        )

        const = {"L": 1.0, "mu": 2.0, "rho": 1.0, "u": 1.0}
        sm = {"u": "u", "L": "L", "rho": "rho", "mu": "mu"}
        out = apply_pi_constant_recipe(const, GROUP_PI_CONSTANT_RECIPES["re"], sm, 10.0)
        Re0 = const["rho"] * const["u"] * const["L"] / const["mu"]
        Re1 = float(out["rho"] * out["u"] * out["L"] / out["mu"])
        assert abs(Re0 - Re1) < 1e-9

    def test_apply_pi_constant_recipe_c_must_exceed_one(self):
        from moju.monitor.pi_constant_recipes import (
            GROUP_PI_CONSTANT_RECIPES,
            apply_pi_constant_recipe,
        )

        const = {"L": 1.0, "mu": 2.0, "rho": 1.0, "u": 1.0}
        sm = {"u": "u", "L": "L", "rho": "rho", "mu": "mu"}
        with pytest.raises(ValueError, match="c > 1"):
            apply_pi_constant_recipe(const, GROUP_PI_CONSTANT_RECIPES["re"], sm, 1.0)

    def test_engine_init_requires_recipe_or_compare_keys(self):
        with pytest.raises(ValueError, match="invariance_compare_keys"):
            ResidualEngine(
                constants={"L": 1.0, "mu": 2.0, "rho": 1.0, "u": 1.0},
                laws=[],
                groups=[],
                scaling_audit=[_re_pi_spec(compare_keys=())],
                state_builder=lambda m, p, col, ct: {"out": jnp.array(1.0)},
            )

    def test_all_registered_groups_have_pi_recipe(self):
        from moju.monitor.pi_constant_recipes import assert_pi_recipes_cover_all_groups

        assert_pi_recipes_cover_all_groups()

    def test_engine_init_invariance_c_must_exceed_one(self):
        with pytest.raises(ValueError, match="invariance_scale_c"):
            ResidualEngine(
                constants={"L": 1.0, "mu": 2.0, "rho": 1.0, "u": 1.0},
                laws=[],
                groups=[],
                scaling_audit=[_re_pi_spec(scale_c=1.0)],
                state_builder=lambda m, p, col, ct: {"out": ct["L"] / ct["mu"]},
            )

    def test_path_b_forbidden_when_pi_enabled(self):
        def sb(model, params, collocation, constants):
            return {"out": constants["L"] / constants["mu"]}

        engine = ResidualEngine(
            constants={"L": 1.0, "mu": 2.0, "rho": 1.0, "u": 1.0},
            laws=[],
            groups=[],
            scaling_audit=[_re_pi_spec()],
            state_builder=sb,
        )
        with pytest.raises(ValueError, match="Path A"):
            engine.compute_residuals({"out": jnp.array(0.5), "Re": jnp.array(0.25), "u": jnp.array(1.0)})

    def test_path_a_pi_residual_zero_when_invariant(self, rtol, atol):
        def sb(model, params, collocation, constants):
            return {"out": constants["L"] / constants["mu"]}

        engine = ResidualEngine(
            constants={"L": 1.0, "mu": 2.0, "rho": 1.0, "u": 1.0},
            laws=[],
            groups=[],
            scaling_audit=[_re_pi_spec()],
            state_builder=sb,
        )
        residuals = engine.compute_residuals(None, model=0, params=0, collocation={})
        r = residuals["scaling"]["re/pi_constant"]
        assert jnp.allclose(r, 0.0, rtol=rtol, atol=atol)

    def test_pi_constant_scale_uses_mean_abs_scaled_branch(self):
        def sb(model, params, collocation, constants):
            return {"out": constants["L"]}

        engine = ResidualEngine(
            constants={"L": 1.0, "mu": 2.0, "rho": 1.0, "u": 1.0},
            laws=[],
            groups=[],
            scaling_audit=[_re_pi_spec()],
            state_builder=sb,
        )
        engine.compute_residuals(None, model=0, params=0, collocation={})
        entry = engine.log[-1]
        scale = entry["scale"]["scaling/re/pi_constant"]
        assert scale > 5.0
        rms = entry["rms"]["scaling/re/pi_constant"]
        assert rms > 0.01


class TestAuditSpecPiFieldsRoundtrip:
    def test_monitor_config_scaling_audit_pi_fields(self):
        spec = AuditSpec(
            name="re",
            output_key="Re",
            state_map={"u": "u", "L": "L", "rho": "rho", "mu": "mu"},
            invariance_pi_constant=True,
            invariance_compare_keys=["out"],
            invariance_scale_c=7.0,
        )
        cfg = MonitorConfig(constants={}, scaling_audit=[spec])
        d = cfg.to_dict()
        cfg2 = MonitorConfig.from_dict(d)
        s2 = cfg2.scaling_audit[0]
        assert s2.invariance_pi_constant is True
        assert s2.invariance_compare_keys == ["out"]
        assert s2.invariance_scale_c == 7.0


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
