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
    assert "scaling_audit" not in d


def test_build_fragment_thermal_diffusivity_audit_minimal():
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
    assert "predicted_spatial" not in ca


def test_user_selected_fo_group_output_key_matches_law_fo():
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


def test_fourier_law_linked_implied_no_chain_fields():
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
    assert ca[0].get("implied_balance_fn") is None
    assert "predicted_spatial" not in ca[0]


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
            s
            for s in d.get("constitutive_audit", [])
            if s.get("implied_fn") is not None or s.get("implied_balance_fn") is not None
        ]
        assert implied_rows, f"Studio missing implied row for {law_name}"
