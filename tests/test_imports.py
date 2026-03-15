"""Test package imports and public API consistency."""

import pytest


def test_import_moju():
    """moju package imports and has version."""
    import moju
    assert hasattr(moju, "__version__")
    assert isinstance(moju.__version__, str)
    assert len(moju.__version__) >= 3  # e.g. "0.1.0"


def test_import_piratio_all():
    """moju.piratio exports Groups, Models, Laws, Operators."""
    from moju.piratio import Groups, Models, Laws, Operators
    assert Groups is not None
    assert Models is not None
    assert Laws is not None
    assert Operators is not None


def test_piratio_module_has_all():
    """piratio __all__ matches public API."""
    import moju.piratio as piratio
    assert hasattr(piratio, "__all__")
    assert set(piratio.__all__) == {"Groups", "Models", "Laws", "Operators"}


def test_import_monitor_all():
    """moju.monitor exports ResidualEngine, admissibility_level, build_loss, audit, visualize."""
    from moju.monitor import ResidualEngine, admissibility_level, build_loss, audit, visualize
    assert ResidualEngine is not None
    assert callable(admissibility_level)
    assert callable(build_loss)
    assert callable(audit)
    assert callable(visualize)


def test_monitor_module_has_all():
    """monitor __all__ matches public API."""
    import moju.monitor as monitor
    assert hasattr(monitor, "__all__")
    assert set(monitor.__all__) == {"ResidualEngine", "admissibility_level", "build_loss", "audit", "visualize"}


def test_groups_has_re_and_pr():
    """Groups exposes at least re and pr (core dimensionless numbers)."""
    from moju.piratio import Groups
    assert hasattr(Groups, "re")
    assert hasattr(Groups, "pr")
    assert callable(Groups.re)
    assert callable(Groups.pr)


def test_models_has_ideal_gas_and_sutherland():
    """Models exposes ideal_gas_rho and sutherland_mu."""
    from moju.piratio import Models
    assert hasattr(Models, "ideal_gas_rho")
    assert hasattr(Models, "sutherland_mu")
    assert callable(Models.ideal_gas_rho)
    assert callable(Models.sutherland_mu)


def test_laws_has_mass_incompressible_and_momentum_ns():
    """Laws exposes mass_incompressible and momentum_navier_stokes."""
    from moju.piratio import Laws
    assert hasattr(Laws, "mass_incompressible")
    assert hasattr(Laws, "momentum_navier_stokes")
    assert callable(Laws.mass_incompressible)
    assert callable(Laws.momentum_navier_stokes)


def test_operators_has_gradient_and_laplacian():
    """Operators exposes gradient and laplacian."""
    from moju.piratio import Operators
    assert hasattr(Operators, "gradient")
    assert hasattr(Operators, "laplacian")
    assert callable(Operators.gradient)
    assert callable(Operators.laplacian)
