"""Law-linked implied constitutive diagnostics."""

import jax.numpy as jnp

from moju.monitor import ResidualEngine, audit
from moju.monitor.law_implied_diagnostics import (
    law_implied_unsupported_reasons,
    list_laws_with_implied_diagnostics,
    merge_fragment_law_implied_audit_specs,
    merge_law_implied_audit_specs,
)
from moju.piratio.groups import Groups
from moju.piratio.laws import Laws


def test_merge_fragment_drops_duplicate_basename_rows():
    laws = [
        {
            "name": "fourier_conduction",
            "state_map": {"T_t": "T_t", "T_laplacian": "T_xx", "fo": "Fo", "t": "t", "L": "L"},
        }
    ]
    lic, _ = merge_law_implied_audit_specs(laws, enabled=True)
    bn = lic[0]["residual_basename"]
    cfg = [{"name": "thermal_diffusivity", "residual_basename": bn, "output_key": "alpha"}]
    mc, rest = merge_fragment_law_implied_audit_specs(lic, cfg)
    assert mc == lic
    assert rest == []


def test_merge_fourier_prepends_thermal_diffusivity():
    laws = [
        {
            "name": "fourier_conduction",
            "state_map": {"T_t": "T_t", "T_laplacian": "T_xx", "fo": "Fo", "t": "t", "L": "L"},
        }
    ]
    c, s = merge_law_implied_audit_specs(laws, enabled=True)
    assert len(c) == 1
    assert c[0]["name"] == "thermal_diffusivity"
    assert c[0]["residual_basename"] == "thermal_diffusivity/law_fourier_conduction"
    assert c[0]["implied_fn"] is not None
    assert s == []


def test_merge_disabled_returns_empty():
    c, s = merge_law_implied_audit_specs(
        [{"name": "fourier_conduction", "state_map": {"T_t": "T_t", "T_laplacian": "lap"}}],
        enabled=False,
    )
    assert c == [] and s == []


def test_fourier_implied_delta_near_zero_on_consistent_state():
    alpha = jnp.array(1.2e-5)
    L = jnp.array(0.02)
    t = jnp.array(5.0)
    Fo = Groups.fo(alpha=alpha, t=t, L=L)
    T_lap = jnp.array(1.0)
    T_t = alpha * T_lap

    engine = ResidualEngine(
        laws=[
            {
                "name": "fourier_conduction",
                "state_map": {
                    "T_t": "T_t",
                    "T_laplacian": "T_xx",
                    "fo": "Fo",
                    "t": "t",
                    "L": "L",
                },
                "fn": Laws.fourier_conduction,
            }
        ],
        groups=[{"name": "fo", "output_key": "Fo", "state_map": {"alpha": "alpha", "t": "t", "L": "L"}}],
        law_implied_audits=True,
    )
    state = {
        "T_t": T_t,
        "T_xx": T_lap,
        "Fo": Fo,
        "t": t,
        "L": L,
        # k/(rho*cp) must match alpha so model thermal_diffusivity agrees with T_t/T_laplacian.
        "k": jnp.array(1.2e-5 * 2700.0 * 900.0),
        "rho": jnp.array(2700.0),
        "cp": jnp.array(900.0),
        "alpha": alpha,
    }
    r = engine.compute_residuals(state)
    key = "thermal_diffusivity/law_fourier_conduction/implied_delta"
    assert key in r["constitutive"]
    assert float(jnp.max(jnp.abs(r["constitutive"][key]))) < 1e-4


def test_ref_delta_gated_by_include_ref_delta():
    """Specs with include_ref_delta=False skip F(pred)-F(ref)."""
    from moju.monitor.closure_registry import MODEL_FNS

    fn, arg_names = MODEL_FNS["thermal_diffusivity"]
    spec = {
        "name": "thermal_diffusivity",
        "output_key": "alpha",
        "state_map": {"k": "k", "rho": "rho", "cp": "cp"},
        "implied_fn": lambda st, c: jnp.array(1.0),
        "residual_basename": "thermal_diffusivity/custom",
        "include_ref_delta": False,
    }
    engine = ResidualEngine(constitutive_audit=[spec])
    alpha = jnp.array(1.0)
    base = {"k": jnp.array(1.0), "rho": jnp.array(1.0), "cp": jnp.array(1.0), "alpha": alpha}
    r = engine.compute_residuals(dict(base), state_ref=dict(base))
    assert "thermal_diffusivity/custom/ref_delta" not in r.get("constitutive", {})
    assert "thermal_diffusivity/custom/implied_delta" in r["constitutive"]


def test_list_laws_nonempty():
    assert "fourier_conduction" in list_laws_with_implied_diagnostics()


def test_each_supported_law_produces_implied_row():
    for law_name in list_laws_with_implied_diagnostics():
        c, s = merge_law_implied_audit_specs(
            [{"name": law_name, "state_map": {}}],
            enabled=True,
        )
        assert len(c) + len(s) >= 1


def test_supported_rows_have_unique_residual_basenames():
    laws = [{"name": n, "state_map": {}} for n in list_laws_with_implied_diagnostics()]
    c, s = merge_law_implied_audit_specs(laws, enabled=True)
    basenames = [str(d.get("residual_basename")) for d in (c + s)]
    assert len(basenames) == len(set(basenames))


def test_supported_rows_are_constitutive_only():
    laws = [{"name": n, "state_map": {}} for n in list_laws_with_implied_diagnostics()]
    c, s = merge_law_implied_audit_specs(laws, enabled=True)
    assert c
    assert s == []


def test_unsupported_reasons_present_and_disjoint_from_supported():
    unsupported = law_implied_unsupported_reasons()
    supported = set(list_laws_with_implied_diagnostics())
    assert unsupported
    assert all(bool(v and str(v).strip()) for v in unsupported.values())
    assert supported.isdisjoint(set(unsupported.keys()))


def test_fick_implied_mass_diffusivity_near_zero():
    engine = ResidualEngine(
        laws=[{"name": "fick_diffusion", "state_map": {"phi_t": "phi_t", "phi_laplacian": "phi_lap", "fo_mass": "fo_mass", "t": "t", "L": "L"}}],
        law_implied_audits=True,
    )
    D = jnp.array(2.0)
    state = {
        "phi_t": jnp.array(6.0),
        "phi_lap": jnp.array(3.0),
        "fo_mass": jnp.array(0.5),
        "t": jnp.array(1.0),
        "L": jnp.array(2.0),
        "D": D,
    }
    r = engine.compute_residuals(state)
    key = "mass_diffusivity/law_fick_diffusion/implied_delta"
    assert key in r["constitutive"]
    assert float(jnp.max(jnp.abs(r["constitutive"][key]))) < 1e-6


def test_wave_implied_speed_near_zero():
    engine = ResidualEngine(
        laws=[{"name": "wave_equation", "state_map": {"phi_tt": "phi_tt", "phi_laplacian": "phi_lap", "st_wave": "st_wave", "omega": "omega", "L": "L"}}],
        law_implied_audits=True,
    )
    c = jnp.array(3.0)
    state = {
        "phi_tt": jnp.array(9.0),
        "phi_lap": jnp.array(1.0),
        "st_wave": jnp.array(1.0),
        "omega": jnp.array(3.0),
        "L": jnp.array(1.0),
        "c": c,
    }
    r = engine.compute_residuals(state)
    key = "wave_speed_from_st/law_wave_equation/implied_delta"
    assert key in r["constitutive"]
    assert float(jnp.max(jnp.abs(r["constitutive"][key]))) < 1e-6


def test_advection_diffusion_implied_kappa_near_zero():
    engine = ResidualEngine(
        laws=[{"name": "advection_diffusion", "state_map": {"phi_t": "phi_t", "u": "u", "phi_grad": "phi_grad", "phi_laplacian": "phi_lap", "pe": "pe"}}],
        law_implied_audits=True,
    )
    u = jnp.array([2.0, 0.0])
    state = {
        "phi_t": jnp.array(0.0),
        "u": u,
        "phi_grad": jnp.array([1.0, 0.0]),
        "phi_lap": jnp.array(4.0),
        "pe": jnp.array(2.0),
        "L": jnp.array(1.0),
        "kappa": jnp.array(1.0),
    }
    r = engine.compute_residuals(state)
    key = "scalar_diffusivity_from_pe/law_advection_diffusion/implied_delta"
    assert key in r["constitutive"]
    assert float(jnp.max(jnp.abs(r["constitutive"][key]))) < 1e-6


def test_ns_stokes_burgers_implied_mu_keys_present():
    checks = [
        ("momentum_navier_stokes", {"u_t": "u_t", "u": "u", "u_grad": "u_grad", "p_grad": "p_grad", "u_laplacian": "u_lap", "re": "re"},
         "dynamic_viscosity_from_re/law_momentum_navier_stokes/implied_delta"),
        ("stokes_flow", {"p_grad": "p_grad", "u_laplacian": "u_lap", "re": "re"},
         "dynamic_viscosity_from_re/law_stokes_flow/implied_delta"),
        ("burgers_equation", {"u_t": "u_t", "u": "u", "u_grad": "u_grad", "u_laplacian": "u_lap", "re": "re", "U": "U", "L": "L"},
         "dynamic_viscosity_from_re/law_burgers_equation/implied_delta"),
    ]
    common = {
        "u": jnp.array([2.0, 0.0]),
        "u_grad": jnp.array([[0.0, 0.0], [0.0, 0.0]]),
        "u_t": jnp.array([0.0, 0.0]),
        "p_grad": jnp.array([2.0, 0.0]),
        "u_lap": jnp.array([2.0, 0.0]),
        "re": jnp.array(1.0),
        "rho": jnp.array(1.0),
        "L": jnp.array(1.0),
        "U": jnp.array(1.0),
        "mu": jnp.array(2.0),
    }
    for law_name, sm, key in checks:
        engine = ResidualEngine(laws=[{"name": law_name, "state_map": sm}], law_implied_audits=True)
        r = engine.compute_residuals(dict(common))
        assert key in r["constitutive"]
