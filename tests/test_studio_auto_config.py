"""Tests for Moju Studio auto MonitorConfig builder (no Streamlit)."""

import pytest

from apps.moju_studio.studio_auto_config import (
    STUDIO_LAW_NAMES,
    STUDIO_MODEL_NAMES,
    build_studio_auto_fragment,
)
from moju.monitor.law_implied_diagnostics import list_laws_with_implied_diagnostics


def test_build_fragment_laplace_only():
    d = build_studio_auto_fragment(
        law_names=["laplace_equation"],
        model_names=[],
        group_names=[],
        pred_keys={"phi_laplacian", "x"},
        constant_keys=set(),
    )
    assert len(d["laws"]) == 1
    assert d["laws"][0]["name"] == "laplace_equation"
    assert d["laws"][0]["state_map"] == {"phi_laplacian": "phi_laplacian"}
    assert d["constitutive_audit"] == []
    assert d["scaling_audit"] == []


def test_build_fragment_thermal_diffusivity_predicted_spatial():
    d = build_studio_auto_fragment(
        law_names=[],
        model_names=["thermal_diffusivity"],
        group_names=[],
        pred_keys={"alpha", "k", "rho", "cp", "x"},
        constant_keys=set(),
    )
    ca = d["constitutive_audit"][0]
    assert ca["name"] == "thermal_diffusivity"
    assert ca["output_key"] == "alpha"
    assert set(ca["predicted_spatial"]) == {"k", "rho", "cp"}


def test_fo_scaling_audit_excludes_mesh_coords_from_chain_lists():
    """Avoid ``d_fo_dt`` on coordinate ``t`` when ``t`` is in NPZ keys."""
    d = build_studio_auto_fragment(
        law_names=[],
        model_names=[],
        group_names=["fo"],
        pred_keys={"alpha", "t", "L", "x", "T"},
        constant_keys=set(),
    )
    sa = next(s for s in d["scaling_audit"] if s["name"] == "fo")
    assert "t" not in sa["predicted_temporal"]
    assert "t" not in sa["predicted_spatial"]
    assert "x" not in sa["predicted_spatial"]
    assert "alpha" in sa["predicted_temporal"]


def test_user_selected_fo_group_output_key_matches_law_fo():
    """``fo`` must stay lowercase so laws and implied groups agree; avoid requiring ``Fo`` in NPZ."""
    d = build_studio_auto_fragment(
        law_names=[],
        model_names=[],
        group_names=["fo"],
        pred_keys=set(),
        constant_keys=set(),
    )
    fo_rows = [g for g in d["groups"] if g.get("name") == "fo"]
    assert len(fo_rows) == 1
    assert fo_rows[0]["output_key"] == "fo"
    sa = [s for s in d["scaling_audit"] if s.get("name") == "fo"]
    assert len(sa) == 1
    assert sa[0]["output_key"] == "fo"


def test_build_fragment_rejects_unknown_law():
    with pytest.raises(ValueError, match="allowlist"):
        build_studio_auto_fragment(
            law_names=["not_a_real_law"],
            model_names=[],
            group_names=[],
            pred_keys=set(),
            constant_keys=set(),
        )


def test_allowlists_non_empty():
    assert "laplace_equation" in STUDIO_LAW_NAMES
    assert "thermal_diffusivity" in STUDIO_MODEL_NAMES


def test_fourier_implied_scaling_audit_fd_on_composition_when_T_in_pred():
    """Implied ``fo`` scaling audit enables spatial chain without ``d_fo_dx`` in NPZ."""
    d = build_studio_auto_fragment(
        law_names=["fourier_conduction"],
        model_names=[],
        group_names=[],
        pred_keys={"T", "x", "t", "L", "alpha"},
        constant_keys=set(),
    )
    assert d.get("law_implied_audits") is True
    ca = [s for s in d["constitutive_audit"] if s.get("name") == "thermal_diffusivity"]
    assert len(ca) == 1
    assert "law_fourier_conduction" in ca[0].get("residual_basename", "")
    assert ca[0].get("implied_fn") is not None
    assert "alpha" in ca[0].get("predicted_spatial", [])
    sa = [s for s in d["scaling_audit"] if s.get("name") == "fo"]
    assert len(sa) == 1
    assert sa[0]["chain_output"] == "fd_on_composition"
    assert "alpha" in sa[0]["predicted_spatial"]


def test_fourier_T_in_pred_adds_alpha_to_predicted_spatial_for_fo_chain():
    d = build_studio_auto_fragment(
        law_names=["fourier_conduction"],
        model_names=[],
        group_names=[],
        pred_keys={"T", "x", "t", "L"},
        constant_keys=set(),
    )
    sa = next(s for s in d["scaling_audit"] if s["name"] == "fo")
    assert "alpha" in sa["predicted_spatial"]
    assert sa["chain_output"] == "fd_on_composition"


def test_fourier_law_td_no_alpha_in_pred_skips_constitutive_chain_patch():
    d = build_studio_auto_fragment(
        law_names=["fourier_conduction"],
        model_names=[],
        group_names=[],
        pred_keys={"T", "x", "t", "L"},
        constant_keys=set(),
    )
    ca = next(s for s in d["constitutive_audit"] if s.get("name") == "thermal_diffusivity")
    assert "alpha" not in ca.get("predicted_spatial", [])


def test_fourier_user_thermal_diffusivity_model_skips_law_chain_patch():
    d = build_studio_auto_fragment(
        law_names=["fourier_conduction"],
        model_names=["thermal_diffusivity"],
        group_names=[],
        pred_keys={"T", "x", "t", "L", "alpha", "k", "rho", "cp"},
        constant_keys=set(),
    )
    law_rows = [
        s
        for s in d["constitutive_audit"]
        if s.get("name") == "thermal_diffusivity" and "law_fourier_conduction" in s.get("residual_basename", "")
    ]
    assert len(law_rows) == 1
    assert "alpha" not in law_rows[0].get("predicted_spatial", [])


def test_studio_includes_law_linked_implied_for_supported_laws():
    supported = set(list_laws_with_implied_diagnostics())
    for law_name in sorted(supported.intersection(set(STUDIO_LAW_NAMES))):
        d = build_studio_auto_fragment(
            law_names=[law_name],
            model_names=[],
            group_names=[],
            pred_keys=set(),
            constant_keys=set(),
        )
        implied_rows = [
            s for s in (d.get("constitutive_audit", []) + d.get("scaling_audit", []))
            if s.get("implied_fn") is not None
        ]
        assert implied_rows, f"Studio missing implied row for {law_name}"
