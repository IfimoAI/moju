"""Tests for Studio model-derived derived_state_chain enrichment."""

from apps.moju_studio.studio_auto_config import build_studio_auto_fragment
from apps.moju_studio.studio_dependency_planner import (
    collect_required_state_keys_from_fragment,
    plan_dependencies,
)
from apps.moju_studio.studio_model_derived_registry import (
    MODEL_DERIVED_REGISTRY,
    collect_group_input_state_keys,
    enrich_fragment_from_model_audits,
)
from moju.monitor.path_b_derivatives import PathBGridConfig


def test_registry_contains_thermal_diffusivity():
    assert "thermal_diffusivity" in MODEL_DERIVED_REGISTRY


def test_collect_group_input_state_keys_fo():
    groups = [{"name": "fo", "output_key": "fo", "state_map": {"alpha": "alpha", "t": "t", "L": "L"}}]
    assert collect_group_input_state_keys(groups) == {"alpha", "t", "L"}


def test_enrich_fourier_with_thermal_diffusivity_appends_alpha_step():
    frag = build_studio_auto_fragment(
        law_names=["fourier_conduction"],
        model_names=["thermal_diffusivity"],
        group_names=[],
        pred_keys=set(),
        constant_keys=set(),
    )
    assert frag.get("derived_state_chain") == []
    enriched = enrich_fragment_from_model_audits(frag)
    chain = enriched["derived_state_chain"]
    assert len(chain) == 1
    assert chain[0]["output_key"] == "alpha"
    expr = chain[0]["expr"]
    assert expr["op"] == "div"
    assert expr["a"] == {"op": "ref", "key": "k"}
    assert expr["b"]["op"] == "mul"


def test_enrich_custom_state_map_kappa():
    frag = build_studio_auto_fragment(
        law_names=["fourier_conduction"],
        model_names=["thermal_diffusivity"],
        group_names=[],
        pred_keys=set(),
        constant_keys=set(),
    )
    audits = list(frag["constitutive_audit"])
    audits[0] = {
        **audits[0],
        "state_map": {"k": "kappa", "rho": "rho", "cp": "cp"},
    }
    frag["constitutive_audit"] = audits
    enriched = enrich_fragment_from_model_audits(frag)
    assert enriched["derived_state_chain"][0]["expr"]["a"] == {"op": "ref", "key": "kappa"}


def test_enrich_skips_when_alpha_already_in_chain():
    frag = build_studio_auto_fragment(
        law_names=["fourier_conduction"],
        model_names=["thermal_diffusivity"],
        group_names=[],
        pred_keys=set(),
        constant_keys=set(),
    )
    frag["derived_state_chain"] = [
        {"output_key": "alpha", "expr": {"op": "ref", "key": "T"}},
    ]
    enriched = enrich_fragment_from_model_audits(frag)
    assert len(enriched["derived_state_chain"]) == 1


def test_enrich_skips_without_thermal_audit():
    """No ``thermal_diffusivity`` audit row => no alpha-from-T derived step."""
    frag = build_studio_auto_fragment(
        law_names=["laplace_equation"],
        model_names=[],
        group_names=[],
        pred_keys=set(),
        constant_keys=set(),
    )
    enriched = enrich_fragment_from_model_audits(frag)
    assert enriched["derived_state_chain"] == []


def test_enrich_skips_when_output_not_needed_by_groups():
    """thermal_diffusivity present but no group asks for alpha — e.g. no fo in fragment."""
    frag = {
        "laws": [],
        "groups": [],
        "constitutive_audit": [
            {
                "name": "thermal_diffusivity",
                "output_key": "alpha",
                "state_map": {"k": "k", "rho": "rho", "cp": "cp"},
            }
        ],
        "derived_state_chain": [],
    }
    enriched = enrich_fragment_from_model_audits(frag)
    assert enriched["derived_state_chain"] == []


def test_planner_alpha_not_required_after_enrich():
    frag = build_studio_auto_fragment(
        law_names=["fourier_conduction"],
        model_names=["thermal_diffusivity"],
        group_names=[],
        pred_keys=set(),
        constant_keys=set(),
    )
    enriched = enrich_fragment_from_model_audits(frag)
    keys = collect_required_state_keys_from_fragment(enriched)
    assert "alpha" not in keys
    assert {"k", "rho", "cp"} <= keys


def test_plan_fourier_with_enrich_no_blocking_for_alpha():
    frag = build_studio_auto_fragment(
        law_names=["fourier_conduction"],
        model_names=["thermal_diffusivity"],
        group_names=[],
        pred_keys=set(),
        constant_keys=set(),
    )
    enriched = enrich_fragment_from_model_audits(frag)
    pred = {"T", "x", "t", "k", "rho", "cp", "L"}
    p = plan_dependencies(
        enriched,
        pred_keys=pred,
        constant_keys=set(),
        auto_path_b_derivatives=True,
        fill_law_fd=True,
        path_b_grid=PathBGridConfig(layout="meshgrid", spatial_dimension=1, steady=True),
    )
    assert "alpha" not in p.missing_state_direct
    assert not p.has_blocking_gaps()
