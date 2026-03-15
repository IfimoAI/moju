"""Tests for MojuCore, build_loss, audit, and visualize."""

import pytest
import jax
import jax.numpy as jnp
from moju.piratio import MojuCore, build_loss, audit, visualize


class TestMojuCoreResidualDict:
    """compute_residuals returns correct residual dict structure."""

    def test_laws_only_when_no_ref(self, rtol, atol):
        """When state_ref and key_ref are None, only law residuals are computed."""
        core = MojuCore(
            constants={"L": 0.1},
            laws=[{"name": "mass_incompressible", "state_map": {"u_grad": "u_grad"}}],
            groups=[],
            models=[],
        )
        state_pred = {"u_grad": jnp.array([[0.0, 1.0], [-1.0, 0.0]])}
        residuals = core.compute_residuals(state_pred)
        assert "laws" in residuals
        assert "mass_incompressible" in residuals["laws"]
        assert jnp.allclose(residuals["laws"]["mass_incompressible"], 0.0, rtol=rtol, atol=atol)
        assert "groups" not in residuals
        assert "models" not in residuals
        assert "data" not in residuals

    def test_log_appends_rms_per_key(self):
        """Log gets one entry per compute_residuals with rms per key."""
        core = MojuCore(
            constants={},
            laws=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_laplacian"}}],
            groups=[],
            models=[],
        )
        state_pred = {"phi_laplacian": jnp.array(1.0)}
        core.compute_residuals(state_pred)
        core.compute_residuals(state_pred)
        assert len(core.log) == 2
        assert core.log[0]["index"] == 0
        assert core.log[1]["index"] == 1
        assert "rms" in core.log[0]
        assert "laws/laplace_equation" in core.log[0]["rms"]

    def test_key_ref_adds_groups_and_models_residuals(self, rtol, atol):
        """When key_ref is provided, group and model residuals are computed vs key_ref."""
        core = MojuCore(
            constants={"mu0": 1.8e-5, "T0": 273.0, "S": 110.4},
            laws=[],
            groups=[{"name": "re", "state_map": {"u": "u", "L": "L", "rho": "rho", "mu": "mu"}, "output_key": "re"}],
            models=[{"name": "sutherland_mu", "state_map": {"T": "T", "mu0": "mu0", "T0": "T0", "S": "S"}, "output_key": "mu"}],
        )
        state_pred = {"u": 1.0, "L": 0.1, "rho": 1.2, "T": 300.0}
        key_ref = {"re": 100.0, "mu": 2.0e-5}
        residuals = core.compute_residuals(state_pred, key_ref=key_ref)
        assert "groups" in residuals
        assert "re" in residuals["groups"]
        assert "models" in residuals
        assert "mu" in residuals["models"]
        assert "data" not in residuals

    def test_state_ref_adds_data_residual(self, rtol, atol):
        """When state_ref is provided, data residual is computed over common keys."""
        core = MojuCore(
            constants={},
            laws=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_laplacian"}}],
            groups=[],
            models=[],
        )
        state_pred = {"phi_laplacian": jnp.array(0.5)}
        state_ref = {"phi_laplacian": jnp.array(0.0)}
        residuals = core.compute_residuals(state_pred, state_ref=state_ref)
        assert "data" in residuals
        assert "phi_laplacian" in residuals["data"]
        assert jnp.allclose(residuals["data"]["phi_laplacian"], -0.5, rtol=rtol, atol=atol)


class TestBuildLoss:
    """build_loss is physics-only and cascaded over laws."""

    def test_cascaded_loss_scalar(self, rtol, atol):
        """build_loss returns a scalar."""
        residual_dict = {
            "laws": {
                "mass_incompressible": jnp.array(0.0),
                "laplace_equation": jnp.array(0.0),
            }
        }
        loss = build_loss(residual_dict)
        assert loss.shape == ()
        assert jnp.allclose(loss, 0.0, rtol=rtol, atol=atol)

    def test_cascaded_loss_nonzero(self, rtol, atol):
        """build_loss is sum of weighted RMS of law residuals."""
        residual_dict = {
            "laws": {
                "laplace_equation": jnp.array(3.0),
            }
        }
        loss = build_loss(residual_dict)
        assert float(loss) >= 0
        assert jnp.allclose(loss, 3.0, rtol=rtol, atol=atol)

    def test_build_loss_law_weights(self, rtol, atol):
        """law_weights customizes per-law weights."""
        residual_dict = {
            "laws": {
                "a": jnp.array(1.0),
                "b": jnp.array(0.0),
            }
        }
        loss = build_loss(residual_dict, law_weights={"a": 1.0, "b": 0.0})
        assert jnp.allclose(loss, 1.0, rtol=rtol, atol=atol)

    def test_build_loss_differentiable(self):
        """build_loss output is differentiable w.r.t. residual inputs."""
        def loss_fn(phi_laplacian):
            rd = {"laws": {"laplace_equation": phi_laplacian}}
            return build_loss(rd)
        grad = jax.grad(loss_fn)(jnp.array(2.0))
        assert grad is not None


class TestBuildLossBatch:
    """build_loss reduces over batch."""

    def test_batch_law_residuals(self, rtol, atol):
        """Residual dict with batch dimension; loss is scalar."""
        residual_dict = {
            "laws": {
                "mass_incompressible": jnp.zeros((5,)),
            }
        }
        loss = build_loss(residual_dict)
        assert loss.shape == ()
        assert jnp.allclose(loss, 0.0, rtol=rtol, atol=atol)


class TestAudit:
    """audit computes R_norm, S, writes back to log."""

    def test_audit_writes_back_to_log(self):
        """audit adds r_norm, S, overall_physics_score to each log entry."""
        log = [
            {"index": 0, "rms": {"laws/a": 2.0, "laws/b": 1.0}},
            {"index": 1, "rms": {"laws/a": 1.0, "laws/b": 0.5}},
        ]
        report = audit(log)
        assert "per_key" in report
        assert "overall_physics_score" in report
        assert "r_norm" in log[0]
        assert "S" in log[0]
        assert "overall_physics_score" in log[0]
        assert "r_norm" in log[1]
        assert "S" in log[1]

    def test_audit_r_ref_from_first_entry(self):
        """When r_ref is None, first entry's rms is used as reference."""
        log = [
            {"index": 0, "rms": {"k": 10.0}},
            {"index": 1, "rms": {"k": 5.0}},
        ]
        audit(log)
        assert log[1]["r_norm"]["k"] == 0.5
        assert log[1]["S"]["k"] == 1.0 / (1.0 + 0.5)


class TestVisualize:
    """visualize produces plots or returns None."""

    def test_visualize_empty_log_returns_none(self):
        """Empty log returns None."""
        fig = visualize([], backend="matplotlib")
        assert fig is None

    def test_visualize_backend_none_returns_none(self):
        """backend='none' returns None without plotting."""
        log = [{"index": 0, "rms": {"k": 1.0}}]
        fig = visualize(log, backend="none")
        assert fig is None

    def test_visualize_with_log_no_crash(self):
        """With log entries, visualize does not raise (may return None if no matplotlib)."""
        log = [
            {"index": 0, "rms": {"laws/a": 1.0}},
            {"index": 1, "rms": {"laws/a": 0.5}},
        ]
        fig = visualize(log, backend="matplotlib")
        assert fig is None or hasattr(fig, "savefig")


class TestMojuCoreStateBuilder:
    """State builder runs models then groups."""

    def test_state_builder_adds_model_output(self, rtol, atol):
        """Model output is written to state under output_key."""
        core = MojuCore(
            constants={"mu0": 1.8e-5, "T0": 273.0, "S": 110.4},
            laws=[],
            groups=[],
            models=[{"name": "sutherland_mu", "state_map": {"T": "T", "mu0": "mu0", "T0": "T0", "S": "S"}, "output_key": "mu"}],
        )
        state_pred = {"T": 300.0}
        state = core._state_builder(state_pred)
        assert "mu" in state
        expected = 1.8e-5 * (300 / 273) ** 1.5 * (273 + 110.4) / (300 + 110.4)
        assert abs(float(state["mu"]) - expected) < 1e-10

    def test_state_builder_adds_group_output(self, rtol, atol):
        """Group output is written to state under output_key."""
        core = MojuCore(
            constants={"L": 0.1},
            laws=[],
            groups=[{"name": "re", "state_map": {"u": "u", "L": "L", "rho": "rho", "mu": "mu"}, "output_key": "re"}],
            models=[],
        )
        state_pred = {"u": 1.0, "rho": 1000.0, "mu": 1.0}
        state = core._state_builder(state_pred)
        assert "re" in state
        assert jnp.allclose(state["re"], 100.0, rtol=rtol, atol=atol)
