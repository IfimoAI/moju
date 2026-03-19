"""Smoke test: README Quick Start snippet runs and produces expected outputs."""

import pytest
import jax.numpy as jnp


def test_readme_quick_start_runs():
    """Code from README Quick Start runs without error and returns plausible values."""
    import moju
    from moju.piratio import Groups, Models

    assert moju.__version__
    Re = Groups.re(u=1.0, L=0.1, rho=1000.0, mu=1e-3)
    rho = Models.ideal_gas_rho(P=101325.0, R=287.0, T=300.0)

    assert Re is not None
    assert rho is not None
    assert float(Re) > 0
    assert 1.0 < float(rho) < 2.0  # air at 300 K


def test_readme_five_minute_example_runs():
    """README 5-minute example: ResidualEngine, build_loss, audit with one law."""
    import jax.numpy as jnp
    from moju.monitor import ResidualEngine, build_loss, audit, MonitorConfig

    cfg = MonitorConfig(laws=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_xx"}}])
    engine = ResidualEngine(config=cfg)
    state_pred = {"phi_xx": jnp.array(0.0)}
    residuals = engine.compute_residuals(state_pred)
    loss = build_loss(residuals)
    report = audit(engine.log)

    assert jnp.ndim(loss) == 0
    assert "overall_admissibility_score" in report
    assert "overall_admissibility_level" in report
    assert "per_category" in report
    assert "per_key" in report
    assert "laws" in report["per_category"]


def test_readme_laws_example_runs():
    """README Laws example: mass_incompressible with zero gradient."""
    import jax.numpy as jnp
    from moju.piratio import Laws

    u_grad = jnp.array([[0.0, 0.0], [0.0, 0.0]])
    residual = Laws.mass_incompressible(u_grad)
    assert jnp.allclose(residual, 0.0)


def test_readme_operators_example_runs():
    """README Operators example: gradient and Laplacian of sum(x^2)."""
    import jax.numpy as jnp
    from moju.piratio import Operators

    def scalar_field(params, x):
        return jnp.sum(x ** 2)

    params = {}
    x = jnp.array([1.0, 2.0])
    grad = Operators.gradient(scalar_field, params, x)
    lap = Operators.laplacian(scalar_field, params, x)

    assert jnp.allclose(grad, jnp.array([2.0, 4.0]))
    assert jnp.allclose(lap, 4.0)
