"""Tests for Moju Studio config_forms (no Streamlit)."""

from apps.moju_studio.config_forms import (
    build_audit_spec_dict,
    build_law_spec,
    law_parameter_names,
    merge_simple_config_with_json_override,
    model_parameter_names,
    path_b_grid_from_options,
    preflight_checklist_text,
    reindex_log_entries,
)


def test_law_parameter_names_laplace():
    names = law_parameter_names("laplace_equation")
    assert names == ["phi_laplacian"]


def test_build_law_spec():
    spec = build_law_spec("laplace_equation", {"phi_laplacian": "phi_laplacian"})
    assert spec == {"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_laplacian"}}


def test_model_parameter_names_nonempty():
    m = "ideal_gas_rho"
    names = model_parameter_names(m)
    assert isinstance(names, list)
    assert len(names) >= 1


def test_build_constitutive_audit_spec_minimal():
    args = model_parameter_names("sutherland_mu")
    assert set(args) == {"T", "mu0", "T0", "S"}
    d = build_audit_spec_dict(
        category="constitutive",
        name="sutherland_mu",
        output_key="mu",
        state_map={a: a for a in args},
        predicted_spatial=["T"],
        implied_value_key="mu_ref",
    )
    assert d["name"] == "sutherland_mu"
    assert d["implied_value_key"] == "mu_ref"
    assert d["closure_mode"] == "pointwise"
    assert "chain_spatial_axes" in d


def test_merge_simple_with_override_empty_lists():
    simple = {"laws": [{"name": "laplace_equation", "state_map": {"phi_laplacian": "p"}}]}
    out = merge_simple_config_with_json_override(simple, '{"laws": []}')
    assert out["laws"] == []


def test_merge_simple_override_primary_fields():
    simple = {"laws": [], "primary_fields": ["T"]}
    out = merge_simple_config_with_json_override(simple, '{"primary_fields": ["u", "v"]}')
    assert out["primary_fields"] == ["u", "v"]


def test_path_b_grid_from_options():
    g = path_b_grid_from_options(layout="separable", spatial_dimension=2, steady=False)
    assert g.layout == "separable"
    assert g.steady is False


def test_reindex_log_entries():
    old = [{"index": 0, "rms": {}}]
    new = [{"index": 0, "rms": {"a": 1.0}}]
    out = reindex_log_entries(old, new)
    assert len(out) == 2
    assert out[1]["index"] == 1


def test_preflight_checklist_text():
    t = preflight_checklist_text(["a", "b"], ["d_T_dx"], ["a", "d_T_dx"])
    assert "[x] a" in t
    assert "[ ] b" in t or "b" in t
