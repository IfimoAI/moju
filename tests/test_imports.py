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
    assert set(piratio.__all__) == {
        "Groups", "Models", "Laws", "Operators",
        "NondimScales", "dimensional_to_nd", "nd_to_dimensional",
    }


def test_import_monitor_all():
    """moju.monitor exports ResidualEngine, audit helpers, closure listers."""
    from moju.monitor import (
        ResidualEngine,
        admissibility_level,
        audit,
        build_loss,
        build_minimal_residual_engine,
        build_law_spec_identity,
        build_monitor_visualize_bundle,
        build_spatial_rnorm_panels_from_residuals,
        collect_group_input_state_keys,
        enrich_derived_state_from_constitutive_audits,
        export_monitor_log,
        get_monitor_log_export,
        implied_group_specs_for_laws,
        merge_implied_groups_first,
        monitor_log_export_to_bundle,
        monitor_log_export_to_jsonable,
        MODEL_DERIVED_REGISTRY,
        ModelDerivedBridge,
        visualize,
        list_constitutive_models,
        list_scaling_closure_ids,
    )
    assert callable(build_minimal_residual_engine)
    assert callable(build_law_spec_identity)
    assert callable(build_monitor_visualize_bundle)
    assert callable(build_spatial_rnorm_panels_from_residuals)
    assert callable(implied_group_specs_for_laws)
    assert callable(merge_implied_groups_first)
    assert ResidualEngine is not None
    assert callable(list_constitutive_models)
    assert callable(list_scaling_closure_ids)


def test_monitor_module_has_all():
    """monitor __all__ matches public API."""
    import moju.monitor as monitor
    assert hasattr(monitor, "__all__")
    assert set(monitor.__all__) == {
        "audit_meta",
        "build_audit_meta",
        "format_audit_meta_plain_summary",
        "ResidualEngine",
        "admissibility_level",
        "configure_r_eff",
        "build_loss",
        "build_minimal_residual_engine",
        "audit",
        "build_monitor_visualize_bundle",
        "export_monitor_log",
        "get_monitor_log_export",
        "monitor_log_export_to_bundle",
        "monitor_log_export_to_jsonable",
        "visualize",
        "build_spatial_rnorm_panels_from_residuals",
        "build_law_spec_identity",
        "implied_group_specs_for_laws",
        "merge_implied_groups_first",
        "list_constitutive_models",
        "list_scaling_closure_ids",
        "AuditSpec",
        "MonitorConfig",
        "audit_spec_to_engine_dict",
        "apply_derived_state_chain",
        "eval_derived_expr",
        "all_ref_keys_from_chain",
        "keys_produced_by_chain",
        "PathBGridConfig",
        "fill_path_b_derivatives",
        "fill_path_b_spectral",
        "fill_law_fd_from_primitives",
        "list_law_fd_supported_laws",
        "effective_audit_specs_for_fragment",
        "law_implied_unsupported_reasons",
        "list_laws_with_implied_diagnostics",
        "merge_law_implied_audit_specs",
        "merge_fragment_law_implied_audit_specs",
        "MODEL_DERIVED_REGISTRY",
        "ModelDerivedBridge",
        "collect_group_input_state_keys",
        "enrich_derived_state_from_constitutive_audits",
        "pretty_residual_key",
        "pretty_category_name",
    }


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


def test_models_has_smagorinsky_nu_t():
    """Models exposes Smagorinsky eddy viscosity helper."""
    from moju.piratio import Models
    assert hasattr(Models, "smagorinsky_nu_t")
    assert callable(Models.smagorinsky_nu_t)


def test_models_has_k_epsilon_and_k_omega_nu_t():
    from moju.piratio import Models
    assert hasattr(Models, "k_epsilon_nu_t") and callable(Models.k_epsilon_nu_t)
    assert hasattr(Models, "k_omega_nu_t") and callable(Models.k_omega_nu_t)


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
