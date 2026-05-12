"""Monitor: ResidualEngine, build_loss, audit, visualize for residuals and training monitoring."""

from moju.monitor.auditor import (
    ResidualEngine,
    admissibility_level,
    audit,
    build_loss,
    build_minimal_residual_engine,
    build_monitor_visualize_bundle,
    list_constitutive_models,
    list_scaling_closure_ids,
    visualize,
)
from moju.monitor.law_group_inference import (
    build_law_spec_identity,
    implied_group_specs_for_laws,
    merge_implied_groups_first,
)
from moju.monitor.spatial_rnorm_panels import build_spatial_rnorm_panels_from_residuals
from moju.monitor.config import AuditSpec, MonitorConfig, audit_spec_to_engine_dict
from moju.monitor.derived_state_chain import (
    apply_derived_state_chain,
    eval_derived_expr,
    all_ref_keys_from_chain,
    keys_produced_by_chain,
)
from moju.monitor.law_fd_recipes import (
    fill_law_fd_from_primitives,
    list_law_fd_supported_laws,
)
from moju.monitor.law_implied_diagnostics import (
    effective_audit_specs_for_fragment,
    law_implied_unsupported_reasons,
    list_laws_with_implied_diagnostics,
    merge_fragment_law_implied_audit_specs,
    merge_law_implied_audit_specs,
)
from moju.monitor.model_derived_registry import (
    MODEL_DERIVED_REGISTRY,
    ModelDerivedBridge,
    collect_group_input_state_keys,
    enrich_derived_state_from_constitutive_audits,
)
from moju.monitor.path_b_derivatives import PathBGridConfig, fill_path_b_derivatives
from moju.monitor.closure_registry import apply_closure_discrepancy_normalize
from moju.monitor.visualize_labels import pretty_category_name, pretty_residual_key

__all__ = [
    "ResidualEngine",
    "admissibility_level",
    "build_loss",
    "build_minimal_residual_engine",
    "audit",
    "build_monitor_visualize_bundle",
    "visualize",
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
    "apply_closure_discrepancy_normalize",
    "pretty_residual_key",
    "pretty_category_name",
    "build_spatial_rnorm_panels_from_residuals",
    "build_law_spec_identity",
    "implied_group_specs_for_laws",
    "merge_implied_groups_first",
]
