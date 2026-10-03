"""derivatives='supplied_only', MissingSuppliedDerivativeError, and derivative provenance."""

import importlib.util
import math
from pathlib import Path

import jax.numpy as jnp
import pytest

from moju.monitor import (
    KeyDeclaration,
    MissingSuppliedDerivativeError,
    ResidualEngine,
    audit,
    audit_meta,
    fill_path_b_derivatives,
    fill_path_b_spectral,
)
from moju.monitor.derivative_provenance import is_derivative_key

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load(filename):
    spec = importlib.util.spec_from_file_location(filename[:-3], _EXAMPLES / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


heat = _load("cookbook_supplied_derivatives_heat.py")


def test_cookbook_supplied_and_fd_labels():
    out = heat.main()
    assert out["supplied"]["derivative_provenance"] == {"T_t": "supplied", "T_laplacian": "supplied"}
    assert out["fd"]["derivative_provenance"] == {"T_t": "supplied", "T_laplacian": "finite_difference"}
    err = out["missing_error"]
    assert isinstance(err, MissingSuppliedDerivativeError) and isinstance(err, KeyError)
    assert (err.law, err.arg, err.key) == ("fourier_conduction", "T_t", "T_t")


def test_supplied_only_rejects_fill_options():
    eng = heat.engine()
    st = heat.state_builder(heat.surrogate, 1.0, {"x": jnp.linspace(0, 1, 9), "t": jnp.asarray(0.1)}, {})
    with pytest.raises(ValueError, match="supplied_only"):
        eng.compute_residuals(st, auto_path_b_derivatives=True)
    with pytest.raises(ValueError, match="derivatives must be"):
        ResidualEngine(derivatives="maybe")


def test_supplied_only_raises_even_in_best_effort():
    eng = ResidualEngine(
        laws=heat.LAWS,
        groups=heat.GROUPS,
        law_implied_audits=False,
        derivatives="supplied_only",
        best_effort_partial=True,
    )
    st = heat.state_builder(heat.surrogate, 1.0, {"x": jnp.linspace(0, 1, 9), "t": jnp.asarray(0.1)}, {})
    del st["T_laplacian"]
    with pytest.raises(MissingSuppliedDerivativeError, match="T_laplacian"):
        eng.compute_residuals(st)


def test_is_derivative_key_rules():
    assert is_derivative_key("fourier_conduction", "T_t", "temp_rate")
    assert is_derivative_key("custom", "a", "phi_grad")
    assert not is_derivative_key("custom", "a", "phi")
    assert is_derivative_key("custom", "a", "acc", {"acc": KeyDeclaration("length", time_order=2)})
    assert not is_derivative_key("custom", "a", "y_t", {"y_t": KeyDeclaration("length")})


def test_fill_functions_return_provenance():
    n = 32
    x = jnp.linspace(0.0, 2 * math.pi, n, endpoint=False)
    laws = [{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_laplacian"}}]
    st = {"x": x, "phi": jnp.sin(x)}
    out2 = fill_path_b_spectral(st, laws_spec=laws)
    assert len(out2) == 2
    _, _, prov = fill_path_b_spectral(st, laws_spec=laws, return_provenance=True)
    assert prov == {"phi_laplacian": "spectral"}
    _, _, prov_fd = fill_path_b_derivatives(st, laws_spec=laws, fill_law_recipes=True, return_provenance=True)
    assert prov_fd == {"phi_laplacian": "finite_difference"}


def test_provenance_in_audit_meta_and_pdf(tmp_path):
    rep = heat.main()["fd"]
    assert "T_laplacian finite difference" in rep["audit_meta"]["plain_sections"]["pipeline"]
    pytest.importorskip("reportlab")
    from moju.monitor.report import write_audit_pdf

    write_audit_pdf(rep, str(tmp_path / "p.pdf"))
    assert (tmp_path / "p.pdf").stat().st_size > 0
