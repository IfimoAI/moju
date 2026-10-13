"""Implied viscosity from momentum LSQ (NS / Stokes / Burgers) — Moju 2.1."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from moju.monitor import ResidualEngine, audit
from moju.monitor.analytic_flows import kovasznay_2d, ns_state_from_analytic, taylor_green_2d
from moju.monitor.law_implied_diagnostics import (
    DEFAULT_IMPLIED_TERM_FRACTION,
    implied_mu_momentum_navier_stokes,
)
from moju.piratio.laws import Laws
from moju.piratio.models import Models


def _ns_sm():
    return {
        "u_t": "u_t",
        "u": "u",
        "u_grad": "u_grad",
        "p_grad": "p_grad",
        "u_laplacian": "u_laplacian",
        "re": "re",
    }


def test_models_dynamic_viscosity_uses_reference_U_not_local_speed():
    mu = Models.dynamic_viscosity_from_re(1.0, 1.0, 1.0, 40.0)
    assert float(mu) == pytest.approx(1.0 / 40.0)
    # Local speed must not enter the catalog model (U is scalar reference).
    mu2 = Models.dynamic_viscosity_from_re(1.0, jnp.array(2.0), 1.0, 40.0)
    assert float(mu2) == pytest.approx(2.0 / 40.0)


def test_legacy_local_speed_bias_on_kovasznay_reproduced():
    """Document old β·ρ·|u|·L bias vs true μ while law residual is ~machine eps."""
    nx = ny = 32
    x = jnp.linspace(0.0, 1.0, nx)
    y = jnp.linspace(0.0, 2.0 * jnp.pi, ny)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    fields = kovasznay_2d(X, Y, re=40.0)
    law_r = Laws.momentum_navier_stokes(
        fields["u_t"],
        fields["u"],
        fields["u_grad"],
        fields["p_grad"],
        fields["u_laplacian"],
        fields["re"],
    )
    assert float(jnp.max(jnp.abs(law_r))) < 1e-5

    adv = jnp.einsum("...ij,...j->...i", fields["u_grad"], fields["u"])
    rhs = fields["u_t"] + adv + fields["p_grad"]
    num = jnp.sum(rhs * fields["u_laplacian"], axis=-1)
    den = jnp.sum(fields["u_laplacian"] ** 2, axis=-1)
    beta = num / (den + 1e-30)
    u_mag = jnp.sqrt(jnp.sum(fields["u"] ** 2, axis=-1))
    mu_legacy = beta * 1.0 * u_mag * 1.0
    mu_true = 1.0 / 40.0
    rel = jnp.abs(mu_legacy - mu_true) / mu_true
    # Where local speed differs from U_ref=1, legacy estimate departs strongly.
    assert float(jnp.max(rel)) > 0.5
    assert float(jnp.mean(rel[u_mag < 0.5])) > 0.1


def test_kovasznay_implied_mu_matches_true_with_reference_U():
    nx = ny = 48
    x = jnp.linspace(0.0, 1.0, nx)
    y = jnp.linspace(0.0, 2.0 * jnp.pi, ny)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    fields = kovasznay_2d(X, Y, re=40.0)
    state = ns_state_from_analytic(fields, X, Y)
    state.update({"rho": 1.0, "U": 1.0, "L": 1.0, "mu": 1.0 / 40.0})
    eng = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _ns_sm()}],
        law_implied_audits=True,
        constants={"rho": 1.0, "U": 1.0, "L": 1.0},
    )
    r = eng.compute_residuals(state)
    key = "dynamic_viscosity_from_re/law_momentum_navier_stokes/implied_delta"
    d = r["constitutive"][key]
    assert float(jnp.nanmax(jnp.abs(d))) < 1e-4
    dbg = r["closure_debug"]["dynamic_viscosity_from_re/law_momentum_navier_stokes"]
    assert dbg["identifiable"] is True
    assert float(dbg["excluded_fraction"]) < 0.5
    assert float(jnp.nanmean(jnp.abs(dbg["implied"] - 1.0 / 40.0))) < 1e-4
    assert float(jnp.abs(dbg["mu_tilde_global"] - 1.0 / 40.0)) < 1e-4
    assert float(jnp.abs(dbg["re_tilde_global"] - 40.0)) < 1e-3


def test_taylor_green_implied_mu_matches_true():
    n = 32
    xs = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    ys = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    fields = taylor_green_2d(x, y, 0.1, re=100.0)
    state = ns_state_from_analytic(fields, x, y, t=0.1)
    state.update({"rho": 1.0, "U": 1.0, "L": 1.0, "mu": 0.01})
    eng = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _ns_sm()}],
        law_implied_audits=True,
        constants={"rho": 1.0, "U": 1.0, "L": 1.0},
    )
    r = eng.compute_residuals(state)
    key = "dynamic_viscosity_from_re/law_momentum_navier_stokes/implied_delta"
    assert float(jnp.nanmax(jnp.abs(r["constitutive"][key]))) < 1e-4
    law_r = Laws.momentum_navier_stokes(
        state["u_t"], state["u"], state["u_grad"], state["p_grad"], state["u_laplacian"], state["re"]
    )
    assert float(jnp.max(jnp.abs(law_r))) < 1e-5


def test_negligible_viscous_term_excluded_or_not_identifiable():
    # Vanishing Laplacian: operator-energy gate → not identifiable (no stable μ).
    state = {
        "u": jnp.ones((16, 2)),
        "u_t": jnp.zeros((16, 2)),
        "u_grad": jnp.zeros((16, 2, 2)),
        "p_grad": jnp.ones((16, 2)),
        "u_laplacian": jnp.full((16, 2), 1e-12),
        "re": 1.0,
        "rho": 1.0,
        "U": 1.0,
        "L": 1.0,
        "mu": 1.0,
    }
    eng = ResidualEngine(
        laws=[{"name": "momentum_navier_stokes", "state_map": _ns_sm()}],
        law_implied_audits=True,
        constants={"implied_term_fraction": DEFAULT_IMPLIED_TERM_FRACTION},
    )
    r = eng.compute_residuals(state)
    dbg = r["closure_debug"]["dynamic_viscosity_from_re/law_momentum_navier_stokes"]
    assert dbg["identifiable"] is False
    assert dbg["status"] == "not_identifiable"
    assert dbg["excluded_fraction"] >= 0.9
    report = audit(eng.log)
    pk = report["per_key"][
        "constitutive/dynamic_viscosity_from_re/law_momentum_navier_stokes/implied_delta"
    ]
    assert pk["admissibility_level"] == "Not identifiable"


def test_stokes_implied_mu_with_reference_U():
    # Exact Stokes: ∇p = (1/Re) ∇²u with Re=2 → β=0.5, μ=ρ U L β = 0.5
    state = {
        "p_grad": jnp.array([1.0, 0.0]),
        "u_lap": jnp.array([2.0, 0.0]),
        "re": 2.0,
        "rho": 1.0,
        "U": 1.0,
        "L": 1.0,
        "mu": 0.5,
    }
    eng = ResidualEngine(
        laws=[{
            "name": "stokes_flow",
            "state_map": {"p_grad": "p_grad", "u_laplacian": "u_lap", "re": "re"},
        }],
        law_implied_audits=True,
    )
    r = eng.compute_residuals(state)
    key = "dynamic_viscosity_from_re/law_stokes_flow/implied_delta"
    assert float(jnp.nanmax(jnp.abs(r["constitutive"][key]))) < 1e-5


def test_burgers_implied_nu_still_matches_and_reports_re_tilde():
    re, U, L = 2.0, 1.0, 1.0
    nu_cat = U * L / re
    u_lap = jnp.array([4.0, 2.0])
    state = {
        "u": jnp.array([1.0, 0.5]),
        "u_grad": jnp.zeros((2, 2)),
        "u_lap": u_lap,
        "u_t": nu_cat * u_lap,
        "re": re,
        "U": U,
        "L": L,
        "nu": nu_cat,
    }
    eng = ResidualEngine(
        laws=[{
            "name": "burgers_equation",
            "state_map": {
                "u_t": "u_t", "u": "u", "u_grad": "u_grad",
                "u_laplacian": "u_lap", "re": "re", "U": "U", "L": "L",
            },
        }],
        law_implied_audits=True,
    )
    r = eng.compute_residuals(state)
    key = "kinematic_viscosity_from_re/law_burgers_equation/implied_delta"
    assert float(jnp.nanmax(jnp.abs(r["constitutive"][key]))) < 1e-5
    dbg = r["closure_debug"]["kinematic_viscosity_from_re/law_burgers_equation"]
    assert float(jnp.abs(dbg["re_tilde_global"] - re)) < 1e-4


def test_jax_torch_parity_ns_implied_mu():
    torch = pytest.importorskip("torch")
    from moju.torch._engine import TorchResidualEngine

    nx = ny = 24
    x = jnp.linspace(0.0, 1.0, nx)
    y = jnp.linspace(0.0, 2.0 * jnp.pi, ny)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    fields = kovasznay_2d(X, Y, re=40.0)
    state = ns_state_from_analytic(fields, X, Y)
    state.update({"rho": 1.0, "U": 1.0, "L": 1.0, "mu": 1.0 / 40.0})
    laws = [{"name": "momentum_navier_stokes", "state_map": _ns_sm()}]
    const = {"rho": 1.0, "U": 1.0, "L": 1.0}
    eng_j = ResidualEngine(laws=laws, law_implied_audits=True, constants=const)
    r_j = eng_j.compute_residuals(state)
    key = "dynamic_viscosity_from_re/law_momentum_navier_stokes/implied_delta"
    state_t = {k: torch.tensor(jnp.asarray(v).tolist(), dtype=torch.float64) for k, v in state.items()}
    eng_t = TorchResidualEngine(laws=laws, law_implied_audits=True, constants=const)
    r_t = eng_t.compute_residuals_torch(state_t)
    dj = jnp.asarray(r_j["constitutive"][key])
    dt = jnp.asarray(r_t["constitutive"][key].detach().numpy())
    # Compare finite points only
    both = jnp.isfinite(dj) & jnp.isfinite(dt)
    assert float(jnp.mean(both)) > 0.5
    assert float(jnp.max(jnp.abs(dj[both] - dt[both]))) < 1e-5


def test_implied_fn_direct_matches_true_mu_on_kov():
    nx = ny = 32
    x = jnp.linspace(0.0, 1.0, nx)
    y = jnp.linspace(0.0, 2.0 * jnp.pi, ny)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    fields = kovasznay_2d(X, Y, re=40.0)
    state = {
        **fields,
        "rho": 1.0,
        "U": 1.0,
        "L": 1.0,
    }
    fn = implied_mu_momentum_navier_stokes(_ns_sm())
    out = fn(state, {})
    assert out is not None
    assert out.meta["identifiable"] is True
    assert float(jnp.nanmax(jnp.abs(out.value - 1.0 / 40.0))) < 1e-4
