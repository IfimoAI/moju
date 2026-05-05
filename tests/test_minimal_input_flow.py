"""Tests for law-first minimal-input monitor flow."""

import jax.numpy as jnp

from moju.monitor import build_minimal_residual_engine
from moju.monitor.path_b_derivatives import PathBGridConfig


def test_build_minimal_engine_infers_fo_group_from_fourier_law():
    eng = build_minimal_residual_engine(law_names=["fourier_conduction"])
    outs = {str(g.get("output_key")) for g in eng.groups_spec}
    assert "fo" in outs


def test_build_minimal_engine_user_group_override_wins():
    eng = build_minimal_residual_engine(
        law_names=["fourier_conduction"],
        groups=[{"name": "fo", "output_key": "fo", "state_map": {"alpha": "alpha_alt", "t": "t", "L": "L"}}],
    )
    fo = next(g for g in eng.groups_spec if str(g.get("output_key")) == "fo")
    assert fo["state_map"]["alpha"] == "alpha_alt"


def test_best_effort_partial_skips_missing_law_inputs_and_logs_unresolved():
    eng = build_minimal_residual_engine(
        law_names=["fourier_conduction"],
        best_effort_partial=True,
    )
    state = {
        "T": jnp.array([300.0, 301.0, 302.0]),
        "t": jnp.array([1.0, 2.0, 3.0]),
        "x": jnp.array([0.0, 0.5, 1.0]),
    }
    _ = eng.compute_residuals(
        state,
        auto_path_b_derivatives=True,
        fill_law_fd=True,
        log_to_python=True,
    )
    entry = eng.log[-1]
    assert "unresolved_dependencies" in entry
    assert any(d.get("stage") == "law" for d in entry["unresolved_dependencies"])


def test_best_effort_partial_does_not_change_default_engine_behavior():
    eng = build_minimal_residual_engine(law_names=["fourier_conduction"], best_effort_partial=False)
    state = {
        "T": jnp.array([300.0, 301.0, 302.0]),
        "t": jnp.array([1.0, 2.0, 3.0]),
        "x": jnp.array([0.0, 0.5, 1.0]),
    }
    raised = False
    try:
        _ = eng.compute_residuals(
            state,
            auto_path_b_derivatives=True,
            fill_law_fd=True,
            log_to_python=True,
        )
    except KeyError:
        raised = True
    assert raised


def test_minimal_engine_defaults_coord_dimension_to_1d():
    eng = build_minimal_residual_engine(law_names=["laplace_equation"])
    assert eng.default_coord_dimension == 1


def test_minimal_engine_uses_problem_coord_dimension_and_allows_per_call_override():
    # Configure problem as 3D, but only provide 1D coords; default auto FD should skip/omit.
    eng = build_minimal_residual_engine(
        law_names=["laplace_equation"],
        best_effort_partial=True,
        coord_dimension=3,
    )
    state = {"phi": jnp.array([1.0, 2.0, 3.0]), "x": jnp.array([0.0, 0.5, 1.0])}
    _ = eng.compute_residuals(
        state,
        auto_path_b_derivatives=True,
        fill_law_fd=True,
        log_to_python=True,
    )
    entry = eng.log[-1]
    assert "omitted" in entry
    # Explicit per-call override to 1D should allow FD fill and law computation.
    residuals = eng.compute_residuals(
        state,
        auto_path_b_derivatives=PathBGridConfig(spatial_dimension=1),
        fill_law_fd=True,
        log_to_python=True,
    )
    assert "laws" in residuals and "laplace_equation" in residuals["laws"]
