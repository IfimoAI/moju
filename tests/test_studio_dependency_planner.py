"""Tests for Studio dependency planner (preflight / Config preview)."""

from apps.moju_studio.config_forms import preflight_checklist_with_dependency_plan
from apps.moju_studio.studio_auto_config import build_studio_auto_fragment
from apps.moju_studio.studio_core import dependency_plan_for_path_b_run
from apps.moju_studio.studio_dependency_planner import (
    BUILTIN_ALIASES_TO_CANONICAL,
    coord_keys_for_law_fd_recipe,
    collect_audit_derivative_keys_from_fragment,
    collect_required_state_keys_from_fragment,
    expand_keys_with_aliases,
    format_planner_preflight_warning,
    plan_dependencies,
)
from moju.monitor.config import MonitorConfig
from moju.monitor.law_fd_recipes import LawFDArgRecipe
from moju.monitor.path_b_derivatives import PathBGridConfig


def _frag_fourier_only():
    return build_studio_auto_fragment(
        law_names=["fourier_conduction"],
        model_names=[],
        group_names=[],
        pred_keys=set(),
        constant_keys=set(),
    )


def _frag_mass_and_ns():
    return build_studio_auto_fragment(
        law_names=["mass_incompressible", "momentum_navier_stokes"],
        model_names=[],
        group_names=[],
        pred_keys=set(),
        constant_keys=set(),
    )


def _frag_re_group():
    return build_studio_auto_fragment(
        law_names=[],
        model_names=[],
        group_names=["re"],
        pred_keys=set(),
        constant_keys=set(),
    )


def test_collect_required_state_keys_fourier():
    frag = _frag_fourier_only()
    keys = collect_required_state_keys_from_fragment(frag)
    assert "T_laplacian" in keys
    assert "T_t" in keys
    assert "fo" in keys
    assert "alpha" in keys  # primitive for implied Groups.fo
    assert {"k", "rho", "cp"} <= keys  # law-linked thermal_diffusivity audit


def test_collect_required_state_keys_derived_chain_adds_refs_drops_outputs():
    frag = {
        "laws": [],
        "groups": [{"name": "fo", "output_key": "fo", "state_map": {"alpha": "alpha", "t": "t", "L": "L"}}],
        "constitutive_audit": [],
        "scaling_audit": [],
        "derived_state_chain": [
            {
                "output_key": "alpha",
                "expr": {
                    "op": "div",
                    "a": {"op": "ref", "key": "kappa"},
                    "b": {
                        "op": "mul",
                        "a": {"op": "ref", "key": "rho"},
                        "b": {"op": "ref", "key": "cp"},
                    },
                },
            },
        ],
    }
    keys = collect_required_state_keys_from_fragment(frag)
    assert "kappa" in keys and "rho" in keys and "cp" in keys
    assert "alpha" not in keys  # produced by chain, not required from NPZ
    assert "fo" in keys and "t" in keys and "L" in keys


def test_fourier_fragment_includes_implied_fo_group():
    frag = _frag_fourier_only()
    fo_specs = [g for g in frag["groups"] if g.get("name") == "fo"]
    assert len(fo_specs) == 1
    assert fo_specs[0]["output_key"] == "fo"
    assert fo_specs[0]["state_map"] == {"alpha": "alpha", "t": "t", "L": "L"}


def test_fourier_studio_patch_planner_requires_d_alpha_dx():
    """Law-linked thermal_diffusivity row gets ``predicted_spatial: [alpha]`` when α is in NPZ."""
    frag = build_studio_auto_fragment(
        law_names=["fourier_conduction"],
        model_names=[],
        group_names=[],
        pred_keys={"T", "x", "t", "L", "alpha"},
        constant_keys=set(),
    )
    dkeys = collect_audit_derivative_keys_from_fragment(frag)
    assert "d_alpha_dx" in dkeys


def test_plan_fourier_t_laplacian_derivable_with_fd():
    frag = _frag_fourier_only()
    # ``fo`` computed by implied group; supply primitives ``alpha``, ``t``, ``L``.
    pred = {"T", "x", "t", "alpha", "L"}
    p = plan_dependencies(
        frag,
        pred_keys=pred,
        constant_keys={"k", "rho", "cp"},
        auto_path_b_derivatives=True,
        fill_law_fd=True,
        path_b_grid=PathBGridConfig(layout="meshgrid", spatial_dimension=1, steady=True),
    )
    assert "T_laplacian" in p.derivable_law_fd_if_enabled
    assert not p.has_blocking_gaps()


def test_plan_fourier_fo_not_required_in_npz_when_implied_group():
    frag = _frag_fourier_only()
    pred = {"T", "x", "t", "alpha", "L"}
    p = plan_dependencies(
        frag,
        pred_keys=pred,
        constant_keys={"k", "rho", "cp"},
        auto_path_b_derivatives=True,
        fill_law_fd=True,
        path_b_grid=PathBGridConfig(layout="meshgrid", spatial_dimension=1, steady=True),
    )
    assert "fo" not in p.missing_state_direct
    assert not p.has_blocking_gaps()


def test_plan_markdown_coordinate_t_hint_when_t_missing():
    frag = _frag_fourier_only()
    pred = {"T", "x", "alpha", "L"}
    p = plan_dependencies(
        frag,
        pred_keys=pred,
        constant_keys=set(),
        auto_path_b_derivatives=True,
        fill_law_fd=True,
        path_b_grid=PathBGridConfig(layout="meshgrid", spatial_dimension=1, steady=True),
    )
    assert "t" in p.missing_state_direct
    md = p.to_markdown()
    assert "Note (`t`)" in md
    assert "mesh time coordinate" in md
    assert "key_t" in md


def test_plan_fourier_blocked_without_primitive():
    frag = _frag_fourier_only()
    pred = {"x", "t", "alpha", "L"}
    p = plan_dependencies(
        frag,
        pred_keys=pred,
        constant_keys=set(),
        auto_path_b_derivatives=True,
        fill_law_fd=True,
    )
    assert p.has_blocking_gaps()
    assert "T_laplacian" in p.missing_state_direct or any(
        "T_laplacian" in b for b in p.law_fd_blocked
    )


def test_plan_fourier_fd_disabled_shows_blocked():
    frag = _frag_fourier_only()
    pred = {"T", "x", "t", "alpha", "L"}
    p = plan_dependencies(
        frag,
        pred_keys=pred,
        constant_keys=set(),
        auto_path_b_derivatives=False,
        fill_law_fd=False,
    )
    assert p.has_blocking_gaps()
    assert any("T_laplacian" in b for b in p.law_fd_blocked)


def test_alias_temperature_satisfies_T_primitive():
    frag = _frag_fourier_only()
    pred = {"temperature", "x", "t", "alpha", "L"}
    p = plan_dependencies(
        frag,
        pred_keys=pred,
        constant_keys=set(),
        auto_path_b_derivatives=True,
        fill_law_fd=True,
        path_b_grid=PathBGridConfig(layout="meshgrid", spatial_dimension=1, steady=True),
    )
    assert any("temperature" in w for w in p.alias_warnings)
    assert "T_laplacian" in p.derivable_law_fd_if_enabled


def test_expand_keys_with_aliases():
    eff, w = expand_keys_with_aliases({"temperature"}, set())
    assert "T" in eff
    assert w


def test_builtin_alias_map_has_temperature():
    assert BUILTIN_ALIASES_TO_CANONICAL["temperature"] == "T"


def test_coord_keys_laplacian_respects_spatial_dimension():
    recipe = LawFDArgRecipe("laplacian")
    g1 = PathBGridConfig(spatial_dimension=1)
    g2 = PathBGridConfig(spatial_dimension=2)
    g3 = PathBGridConfig(spatial_dimension=3)
    g_auto = PathBGridConfig(spatial_dimension="auto")
    assert coord_keys_for_law_fd_recipe(recipe, g1) == {"x"}
    assert coord_keys_for_law_fd_recipe(recipe, g2) == {"x", "y"}
    assert coord_keys_for_law_fd_recipe(recipe, g3) == {"x", "y", "z"}
    assert coord_keys_for_law_fd_recipe(recipe, g_auto) == {"x", "y", "z"}


def test_coord_keys_dt_and_dtt_include_time_key():
    for kind in ("dt", "dtt"):
        recipe = LawFDArgRecipe(kind)
        g = PathBGridConfig(key_t="tau")
        assert coord_keys_for_law_fd_recipe(recipe, g) == {"tau"}


def test_mass_and_ns_requires_u_and_p_keys():
    frag = _frag_mass_and_ns()
    keys = collect_required_state_keys_from_fragment(frag)
    assert "u_grad" in keys
    assert "u_t" in keys
    assert "p_grad" in keys
    assert "u_laplacian" in keys
    assert "re" in keys
    assert "u" in keys  # Reynolds group primitive


def test_re_group_requires_re_output_and_inputs():
    frag = _frag_re_group()
    keys = collect_required_state_keys_from_fragment(frag)
    assert "re" in keys
    assert "u" in keys
    assert "L" in keys
    assert "rho" in keys
    assert "mu" in keys


def test_dependency_plan_for_path_b_run_wraps_monitor_config():
    frag = _frag_fourier_only()
    cfg = MonitorConfig.from_dict(frag)
    p = dependency_plan_for_path_b_run(
        cfg,
        {"T", "x", "t", "alpha", "L"},
        auto_path_b_derivatives=True,
        fill_law_fd=True,
        path_b_grid=PathBGridConfig(spatial_dimension=1, steady=True),
    )
    assert "T_laplacian" in p.derivable_law_fd_if_enabled


def test_preflight_checklist_appends_planner_section():
    txt = preflight_checklist_with_dependency_plan(
        ["T"],
        [],
        [],
        "### Dependency plan\n\n- item",
    )
    assert "Dependency planner" in txt
    assert "item" in txt
    assert "NPZ keys only" in txt


def test_preflight_checklist_available_keys_marks_constants():
    txt = preflight_checklist_with_dependency_plan(
        ["T", "L"],
        [],
        ["T"],
        "### Plan\n",
        available_keys=["T", "L"],
    )
    assert "NPZ ∪ Constants" in txt
    assert "- [x] T" in txt
    assert "- [x] L" in txt


def test_format_planner_preflight_warning_when_blocking():
    frag = _frag_fourier_only()
    pred = {"x", "t", "alpha", "L"}
    p = plan_dependencies(
        frag,
        pred_keys=pred,
        constant_keys=set(),
        auto_path_b_derivatives=True,
        fill_law_fd=True,
    )
    assert p.has_blocking_gaps()
    msg = format_planner_preflight_warning(p)
    assert "Law-FD" in msg or "Unresolved" in msg


def test_fourier_pred_plus_constants_no_false_missing_state():
    """NPZ has T,x,t; alpha,L,k,rho,cp in Constants — planner should not list T_t/T_laplacian/fo as unresolved."""
    frag = _frag_fourier_only()
    pred = {"T", "x", "t"}
    p = plan_dependencies(
        frag,
        pred_keys=pred,
        constant_keys={"alpha", "L", "k", "rho", "cp"},
        auto_path_b_derivatives=True,
        fill_law_fd=True,
        path_b_grid=PathBGridConfig(layout="meshgrid", spatial_dimension=1, steady=True),
    )
    assert "T_t" in p.derivable_law_fd_if_enabled
    assert not p.missing_state_direct
    assert not p.has_blocking_gaps()


def test_power_law_mu_model_adds_constitutive_derivatives_when_pred_has_fields():
    frag = build_studio_auto_fragment(
        law_names=[],
        model_names=["power_law_mu"],
        group_names=[],
        pred_keys={"gamma_dot", "K", "n", "mu_pl", "x"},
        constant_keys=set(),
    )
    dkeys = collect_audit_derivative_keys_from_fragment(frag)
    assert "d_mu_pl_dx" in dkeys
