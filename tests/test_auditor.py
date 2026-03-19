"""Tests for ResidualEngine, build_loss, audit, visualize, constitutive/scaling closures."""

import pytest
import jax
import jax.numpy as jnp
from moju.monitor import (
    ResidualEngine,
    admissibility_level,
    audit,
    build_loss,
    list_constitutive_models,
    list_scaling_closure_ids,
    visualize,
)


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


class TestRegistryHelpers:
    def test_list_constitutive_models(self):
        names = list_constitutive_models()
        assert "sutherland_mu" in names
        assert "thermal_diffusivity" in names

    def test_list_scaling_closure_ids(self):
        ids = list_scaling_closure_ids()
        assert "pe" in ids
        assert "fo" in ids
