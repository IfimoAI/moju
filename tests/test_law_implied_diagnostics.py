"""Law-linked implied constitutive diagnostics."""

import jax.numpy as jnp

from moju.monitor import ResidualEngine, audit
from moju.monitor.law_implied_diagnostics import (
    classify_laws_for_implied_diagnostics,
    law_implied_unsupported_reasons,
    list_unclassified_laws_for_implied_diagnostics,
    list_laws_with_implied_diagnostics,
    merge_fragment_law_implied_audit_specs,
    merge_law_implied_audit_specs,
    supported_auto_implied_laws_for,
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
    assert c[0].get("implied_balance_fn") is None
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


def test_fourier_implied_delta_nan_when_laplacian_zero():
    """Direct implied alpha is undefined where T_laplacian vanishes."""
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
    alpha = jnp.array(1.0)
    state = {
        "T_t": jnp.array(0.3),
        "T_xx": jnp.array(0.0),
        "Fo": jnp.array(1.0),
        "t": jnp.array(1.0),
        "L": jnp.array(1.0),
        "k": jnp.array(1.0),
        "rho": jnp.array(1.0),
        "cp": jnp.array(1.0),
        "alpha": alpha,
    }
    r = engine.compute_residuals(state)
    key = "thermal_diffusivity/law_fourier_conduction/implied_delta"
    arr = r["constitutive"][key]
    assert jnp.all(jnp.isnan(arr))


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


def test_law_linked_rows_use_implied_fn_not_balance_fn():
    laws = [{"name": n, "state_map": {}} for n in list_laws_with_implied_diagnostics()]
    c, _ = merge_law_implied_audit_specs(laws, enabled=True)
    assert c
    assert all(row.get("implied_fn") is not None for row in c)
    assert all(row.get("implied_balance_fn") is None for row in c)


def test_unsupported_reasons_present_and_disjoint_from_supported():
    unsupported = law_implied_unsupported_reasons()
    supported = set(list_laws_with_implied_diagnostics())
    assert unsupported
    assert all(bool(v and str(v).strip()) for v in unsupported.values())
    assert supported.isdisjoint(set(unsupported.keys()))


def test_all_laws_are_classified_for_implied_diagnostics():
    cls = classify_laws_for_implied_diagnostics()
    assert cls
    assert set(cls.values()) <= {"supported", "user_specified_only", "unclassified"}
    assert list_unclassified_laws_for_implied_diagnostics() == ()


def test_faraday_law_has_explicit_unsupported_reason():
    unsupported = law_implied_unsupported_reasons()
    assert "faraday_law" in unsupported
    assert "curl" in unsupported["faraday_law"].lower()


def test_supported_auto_implied_laws_for_selected_specs():
    supported, manual = supported_auto_implied_laws_for(
        [
            {"name": "fourier_conduction", "state_map": {}},
            {"name": "laplace_equation", "state_map": {}},
        ]
    )
    assert "fourier_conduction" in supported
    assert "laplace_equation" in manual


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


def test_ns_stokes_burgers_implied_mu_balance_near_zero():
    checks = [
        ("momentum_navier_stokes", {"u_t": "u_t", "u": "u", "u_grad": "u_grad", "p_grad": "p_grad", "u_laplacian": "u_lap", "re": "re"},
         "dynamic_viscosity_from_re/law_momentum_navier_stokes/implied_delta"),
        ("stokes_flow", {"p_grad": "p_grad", "u_laplacian": "u_lap", "re": "re"},
         "dynamic_viscosity_from_re/law_stokes_flow/implied_delta"),
        ("burgers_equation", {"u_t": "u_t", "u": "u", "u_grad": "u_grad", "u_laplacian": "u_lap", "re": "re", "U": "U", "L": "L"},
         "dynamic_viscosity_from_re/law_burgers_equation/implied_delta"),
    ]
    for law_name, sm, key in checks:
        common = {
            "u": jnp.array([2.0, 0.0]),
            "u_grad": jnp.array([[0.0, 0.0], [0.0, 0.0]]),
            "p_grad": jnp.array([2.0, 0.0]),
            "u_lap": jnp.array([2.0, 0.0]),
            "re": jnp.array(1.0),
            "rho": jnp.array(1.0),
            "L": jnp.array(1.0),
            "U": jnp.array(1.0),
            "mu": jnp.array(2.0),
        }
        if law_name == "burgers_equation":
            common["u_t"] = jnp.array([2.0, 0.0])
        else:
            common["u_t"] = jnp.array([0.0, 0.0])
        engine = ResidualEngine(laws=[{"name": law_name, "state_map": sm}], law_implied_audits=True)
        r = engine.compute_residuals(dict(common))
        assert key in r["constitutive"]
        assert float(jnp.max(jnp.abs(r["constitutive"][key]))) < 1e-5


def test_law_linked_implied_rows_use_subtract_debug_mode():
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
            }
        ],
        groups=[{"name": "fo", "output_key": "Fo", "state_map": {"alpha": "alpha", "t": "t", "L": "L"}}],
        law_implied_audits=True,
    )
    state = {
        "T_t": jnp.array([2.0, 4.0]),
        "T_xx": jnp.array([1.0, 2.0]),
        "Fo": jnp.array([2.0, 2.0]),
        "t": jnp.array([1.0, 1.0]),
        "L": jnp.array([1.0, 1.0]),
        "k": jnp.array([2.0, 2.0]),
        "rho": jnp.array([1.0, 1.0]),
        "cp": jnp.array([1.0, 1.0]),
        "alpha": jnp.array([2.0, 2.0]),
    }
    r = engine.compute_residuals(state)
    debug = r["closure_debug"]["thermal_diffusivity/law_fourier_conduction"]
    assert debug["mode"] == "subtract"
    assert jnp.allclose(debug["pred"], jnp.array([2.0, 2.0]))
    assert jnp.allclose(debug["implied"], jnp.array([2.0, 2.0]))


def test_hookes_implied_stress_near_zero_on_consistent_1d():
    """1-D Hooke's: E=2, nu=0, strain=[0.5] -> stress_model=[1.0] == stress_pinn -> delta≈0."""
    E = jnp.array(2.0)
    nu = jnp.array(0.0)
    strain = jnp.array([0.5])
    stress = jnp.array([1.0])           # consistent with E*strain
    stiffness = jnp.array([[2.0]])      # law arg: C = [[E]] for 1D

    engine = ResidualEngine(
        laws=[
            {
                "name": "hookes_law_residual",
                "state_map": {
                    "stress": "stress",
                    "strain": "strain",
                    "stiffness_tensor": "C",
                },
            }
        ],
        law_implied_audits=True,
    )
    state = {"stress": stress, "strain": strain, "C": stiffness, "E": E, "nu": nu}
    r = engine.compute_residuals(state)
    key = "isotropic_linear_stress/law_hookes_law_residual/implied_delta"
    assert key in r["constitutive"], f"key missing; constitutive keys: {list(r.get('constitutive', {}).keys())}"
    assert float(jnp.max(jnp.abs(r["constitutive"][key]))) < 1e-5


def test_hookes_implied_stress_nonzero_on_inconsistent_state():
    """Deliberately wrong E gives non-zero implied_delta."""
    strain = jnp.array([0.5])
    stress = jnp.array([1.0])
    stiffness = jnp.array([[2.0]])
    E_wrong = jnp.array(10.0)          # inconsistent with stress=1, strain=0.5
    nu = jnp.array(0.0)

    engine = ResidualEngine(
        laws=[
            {
                "name": "hookes_law_residual",
                "state_map": {"stress": "stress", "strain": "strain", "stiffness_tensor": "C"},
            }
        ],
        law_implied_audits=True,
    )
    state = {"stress": stress, "strain": strain, "C": stiffness, "E": E_wrong, "nu": nu}
    r = engine.compute_residuals(state)
    key = "isotropic_linear_stress/law_hookes_law_residual/implied_delta"
    assert key in r["constitutive"]
    assert float(jnp.max(jnp.abs(r["constitutive"][key]))) > 0.1


def test_hookes_implied_row_present_in_registry():
    """merge_law_implied_audit_specs includes isotropic_linear_stress for hookes_law_residual."""
    c, _ = merge_law_implied_audit_specs(
        [{"name": "hookes_law_residual", "state_map": {"stress": "stress", "strain": "strain", "stiffness_tensor": "C"}}],
        enabled=True,
    )
    assert any(r["name"] == "isotropic_linear_stress" for r in c)
    assert any(r.get("implied_fn") is not None for r in c)


def test_mass_compressible_ideal_gas_implied_near_zero():
    """ideal_gas_rho audit: rho_pinn consistent with P/(R*T) gives delta≈0."""
    P = jnp.array(101325.0)
    R = jnp.array(287.0)
    T = jnp.array(300.0)
    rho = P / (R * T)

    # mass_compressible needs: rho, rho_t, u, rho_grad, u_grad
    engine = ResidualEngine(
        laws=[
            {
                "name": "mass_compressible",
                "state_map": {
                    "rho": "rho",
                    "rho_t": "rho_t",
                    "u": "u",
                    "rho_grad": "rho_grad",
                    "u_grad": "u_grad",
                },
            }
        ],
        law_implied_audits=True,
    )
    state = {
        "rho": rho,
        "rho_t": jnp.array(0.0),
        "u": jnp.array([1.0]),
        "rho_grad": jnp.array([0.0]),
        "u_grad": jnp.array([[0.0]]),
        # model inputs for ideal_gas_rho
        "P": P,
        "R": R,
        "T": T,
    }
    r = engine.compute_residuals(state)
    key = "ideal_gas_rho/law_mass_compressible/implied_delta"
    assert key in r["constitutive"], f"key missing; keys: {list(r.get('constitutive', {}).keys())}"
    assert float(jnp.max(jnp.abs(r["constitutive"][key]))) < 1e-5


def test_mass_compressible_boussinesq_implied_near_zero():
    """boussinesq_rho audit: rho_pinn = rho0*(1-beta*dT) gives delta≈0."""
    rho0 = jnp.array(1000.0)
    beta = jnp.array(2e-4)
    dT = jnp.array(10.0)
    rho = rho0 * (1.0 - beta * dT)

    engine = ResidualEngine(
        laws=[
            {
                "name": "mass_compressible",
                "state_map": {
                    "rho": "rho",
                    "rho_t": "rho_t",
                    "u": "u",
                    "rho_grad": "rho_grad",
                    "u_grad": "u_grad",
                },
            }
        ],
        law_implied_audits=True,
    )
    state = {
        "rho": rho,
        "rho_t": jnp.array(0.0),
        "u": jnp.array([0.0]),
        "rho_grad": jnp.array([0.0]),
        "u_grad": jnp.array([[0.0]]),
        # model inputs for boussinesq_rho
        "rho0": rho0,
        "beta": beta,
        "dT": dT,
    }
    r = engine.compute_residuals(state)
    key = "boussinesq_rho/law_mass_compressible/implied_delta"
    assert key in r["constitutive"], f"key missing; keys: {list(r.get('constitutive', {}).keys())}"
    assert float(jnp.max(jnp.abs(r["constitutive"][key]))) < 1e-5


def test_mass_compressible_implied_skipped_when_eos_keys_absent():
    """If P, R, T (and rho0, beta, dT) are absent, both EOS rows return None and are omitted."""
    engine = ResidualEngine(
        laws=[
            {
                "name": "mass_compressible",
                "state_map": {
                    "rho": "rho",
                    "rho_t": "rho_t",
                    "u": "u",
                    "rho_grad": "rho_grad",
                    "u_grad": "u_grad",
                },
            }
        ],
        law_implied_audits=True,
    )
    state = {
        "rho": jnp.array(1.0),
        "rho_t": jnp.array(0.0),
        "u": jnp.array([0.0]),
        "rho_grad": jnp.array([0.0]),
        "u_grad": jnp.array([[0.0]]),
        # EOS keys intentionally absent
    }
    r = engine.compute_residuals(state)
    constitutive = r.get("constitutive", {})
    assert "ideal_gas_rho/law_mass_compressible/implied_delta" not in constitutive
    assert "boussinesq_rho/law_mass_compressible/implied_delta" not in constitutive


def test_mass_compressible_rows_in_registry():
    """merge_law_implied_audit_specs returns two rows for mass_compressible."""
    c, _ = merge_law_implied_audit_specs(
        [{"name": "mass_compressible", "state_map": {"rho": "rho", "rho_t": "rho_t", "u": "u", "rho_grad": "rho_grad", "u_grad": "u_grad"}}],
        enabled=True,
    )
    names = [r["name"] for r in c]
    assert "ideal_gas_rho" in names
    assert "boussinesq_rho" in names
    assert all(r.get("implied_fn") is not None for r in c)


def test_fourier_implied_works_with_user_fns_materializing_k_rho_alpha():
    """
    Users can avoid precomputing constitutive inputs by supplying callables keyed by output state.
    Here: k(T), rho(T), and alpha(k,rho,cp) are built via user_fns so implied thermal_diffusivity runs.
    """
    cp = jnp.array(900.0)
    engine = ResidualEngine(
        constants={"cp": cp},
        laws=[
            {
                "name": "fourier_conduction",
                "state_map": {
                    "T_t": "T_t",
                    "T_laplacian": "T_xx",
                    "fo": "fo",
                    "t": "t",
                    "L": "L",
                },
                "fn": Laws.fourier_conduction,
            }
        ],
        groups=[
            {
                "name": "fo",
                "output_key": "fo",
                "state_map": {"alpha": "alpha", "t": "t", "L": "L"},
                "fn": Groups.fo,
            }
        ],
        user_fns={
            "k": lambda T: 200.0 * (1.0 + 0.001 * (jnp.asarray(T) - 400.0)),
            "rho": lambda T: 2700.0 * (1.0 - 0.0001 * (jnp.asarray(T) - 400.0)),
            "alpha": lambda k, rho, cp: jnp.asarray(k) / (jnp.asarray(rho) * jnp.asarray(cp)),
        },
        law_implied_audits=True,
    )
    T_xx = jnp.array(2.0)
    T_t = jnp.array(1.0)
    state = {
        "T": jnp.array(420.0),
        "T_t": T_t,
        "T_xx": T_xx,
        "t": jnp.array(5.0),
        "L": jnp.array(0.02),
    }
    r = engine.compute_residuals(state)
    key = "thermal_diffusivity/law_fourier_conduction/implied_delta"
    assert key in r["constitutive"]


