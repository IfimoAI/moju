"""
Tests for moju.piratio.nondim — NondimScales, dimensional_to_nd, nd_to_dimensional.

Coverage:
- per-rule scaling correctness for each field family
- affine temperature offset
- all four time_scale modes
- auto p_ref (rho_ref * U_ref²)
- passthrough keys (dimensionless groups and law constants)
- extra_rules float and callable overrides
- roundtrip dimensional → ND → dimensional consistency
- unknown-key warning and suppression
- full realistic NS state dict
- NondimScales validation (bad time_scale, missing diffusivity)
"""
import math
import warnings

import jax.numpy as jnp
import pytest

from moju.piratio import NondimScales, dimensional_to_nd, nd_to_dimensional
from moju.piratio.nondim import _FIELD_SCALE_RULES, _PASSTHROUGH_KEYS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _close(a, b, rtol=1e-5, atol=1e-7):
    return bool(jnp.allclose(jnp.asarray(a), jnp.asarray(b), rtol=rtol, atol=atol))


# Default scales used in most tests
_SCALES = NondimScales(
    L_ref=0.1,
    U_ref=2.0,
    rho_ref=1000.0,
    dT_ref=50.0,
    T0=300.0,
    phi_ref=5.0,
    E_ref=2e11,
    time_scale="convective",
)


# ---------------------------------------------------------------------------
# NondimScales dataclass
# ---------------------------------------------------------------------------

class TestNondimScales:
    def test_t_ref_convective(self):
        s = NondimScales(L_ref=0.1, U_ref=2.0)
        assert math.isclose(s.t_ref, 0.05)

    def test_t_ref_fourier(self):
        s = NondimScales(L_ref=0.1, alpha_ref=1e-6, time_scale="fourier")
        expected = 0.1 ** 2 / 1e-6
        assert math.isclose(s.t_ref, expected)

    def test_t_ref_mass_fourier(self):
        s = NondimScales(L_ref=0.2, D_ref=2e-9, time_scale="mass_fourier")
        expected = 0.2 ** 2 / 2e-9
        assert math.isclose(s.t_ref, expected)

    def test_t_ref_wave(self):
        s = NondimScales(L_ref=1.0, c_ref=340.0, time_scale="wave")
        expected = 1.0 / 340.0
        assert math.isclose(s.t_ref, expected)

    def test_p_ref_auto(self):
        s = NondimScales(L_ref=1.0, U_ref=3.0, rho_ref=2.0)
        assert math.isclose(s._p_ref, 2.0 * 9.0)

    def test_p_ref_explicit(self):
        s = NondimScales(L_ref=1.0, p_ref=1e5)
        assert math.isclose(s._p_ref, 1e5)

    def test_invalid_time_scale(self):
        with pytest.raises(ValueError, match="time_scale"):
            NondimScales(L_ref=1.0, time_scale="bogus")

    def test_fourier_without_alpha(self):
        with pytest.raises(ValueError, match="alpha_ref"):
            NondimScales(L_ref=1.0, time_scale="fourier")

    def test_mass_fourier_without_D(self):
        with pytest.raises(ValueError, match="D_ref"):
            NondimScales(L_ref=1.0, time_scale="mass_fourier")

    def test_wave_without_c(self):
        with pytest.raises(ValueError, match="c_ref"):
            NondimScales(L_ref=1.0, time_scale="wave")

    def test_frozen_immutable(self):
        s = NondimScales(L_ref=1.0)
        with pytest.raises(Exception):
            s.L_ref = 2.0  # type: ignore


# ---------------------------------------------------------------------------
# Per-rule scaling: dimensional_to_nd forward correctness
# ---------------------------------------------------------------------------

class TestCoordinateScaling:
    def test_x_scaled(self):
        nd = dimensional_to_nd({"x": 0.05}, _SCALES, warn_unknown=False)
        assert _close(nd["x"], 0.05 / 0.1)  # x/L_ref = 0.5

    def test_y_scaled(self):
        nd = dimensional_to_nd({"y": 0.2}, _SCALES, warn_unknown=False)
        assert _close(nd["y"], 0.2 / 0.1)

    def test_z_scaled(self):
        nd = dimensional_to_nd({"z": 0.3}, _SCALES, warn_unknown=False)
        assert _close(nd["z"], 0.3 / 0.1)


class TestTimeScaling:
    def test_t_convective(self):
        s = NondimScales(L_ref=0.1, U_ref=2.0)
        t_ref = 0.1 / 2.0
        nd = dimensional_to_nd({"t": 0.01}, s, warn_unknown=False)
        assert _close(nd["t"], 0.01 / t_ref)

    def test_t_fourier(self):
        s = NondimScales(L_ref=0.1, alpha_ref=1e-6, time_scale="fourier")
        t_ref = s.t_ref
        nd = dimensional_to_nd({"t": 5.0}, s, warn_unknown=False)
        assert _close(nd["t"], 5.0 / t_ref)

    def test_t_mass_fourier(self):
        s = NondimScales(L_ref=0.05, D_ref=2e-9, time_scale="mass_fourier")
        t_ref = s.t_ref
        nd = dimensional_to_nd({"t": 100.0}, s, warn_unknown=False)
        assert _close(nd["t"], 100.0 / t_ref)

    def test_t_wave(self):
        s = NondimScales(L_ref=1.0, c_ref=340.0, time_scale="wave")
        t_ref = s.t_ref
        nd = dimensional_to_nd({"t": 0.002}, s, warn_unknown=False)
        assert _close(nd["t"], 0.002 / t_ref)


class TestVelocityScaling:
    """Plan test: test_convective_velocity_scaling"""

    def test_u_scaled(self):
        nd = dimensional_to_nd({"u": 2.0}, _SCALES, warn_unknown=False)
        assert _close(nd["u"], 2.0 / 2.0)  # 1.0

    def test_v_scaled(self):
        nd = dimensional_to_nd({"v": 1.0}, _SCALES, warn_unknown=False)
        assert _close(nd["v"], 1.0 / 2.0)  # 0.5

    def test_w_scaled(self):
        nd = dimensional_to_nd({"w": 4.0}, _SCALES, warn_unknown=False)
        assert _close(nd["w"], 4.0 / 2.0)  # 2.0

    def test_u_t_convective(self):
        # u_t* = (t_ref/U_ref) * ∂u/∂t = (L/(U²)) * ∂u/∂t
        t_ref = _SCALES.t_ref  # 0.1/2.0 = 0.05
        u_t_dim = 10.0  # [m/s²]
        nd = dimensional_to_nd({"u_t": u_t_dim}, _SCALES, warn_unknown=False)
        expected = u_t_dim * t_ref / _SCALES.U_ref
        assert _close(nd["u_t"], expected)

    def test_u_grad_scaled(self):
        # u_grad* = (L_ref/U_ref) * ∂u/∂x
        u_grad_dim = jnp.ones((2, 2)) * 20.0
        nd = dimensional_to_nd({"u_grad": u_grad_dim}, _SCALES, warn_unknown=False)
        expected = u_grad_dim * _SCALES.L_ref / _SCALES.U_ref
        assert _close(nd["u_grad"], expected)

    def test_u_laplacian_scaled(self):
        # u_laplacian* = (L_ref²/U_ref) * ∇²u
        u_lap_dim = jnp.array([5.0, -3.0])
        nd = dimensional_to_nd({"u_laplacian": u_lap_dim}, _SCALES, warn_unknown=False)
        expected = u_lap_dim * _SCALES.L_ref ** 2 / _SCALES.U_ref
        assert _close(nd["u_laplacian"], expected)


class TestPressureScaling:
    """Plan test: test_pressure_auto_p_ref"""

    def test_p_auto_p_ref(self):
        # p_ref = rho_ref * U_ref² = 1000 * 4 = 4000 Pa
        p_dim = 8000.0
        nd = dimensional_to_nd({"p": p_dim}, _SCALES, warn_unknown=False)
        p_ref = _SCALES._p_ref  # 4000
        assert _close(nd["p"], p_dim / p_ref)

    def test_p_explicit_p_ref(self):
        s = NondimScales(L_ref=0.1, p_ref=1e5)
        p_dim = 5e4
        nd = dimensional_to_nd({"p": p_dim}, s, warn_unknown=False)
        assert _close(nd["p"], 0.5)

    def test_p_grad_scaled(self):
        """Plan test: p_grad scales correctly with auto p_ref."""
        p_ref = _SCALES._p_ref  # 4000
        # p_grad* = (L_ref/p_ref) * dp/dx
        pg_dim = jnp.array([2e5, -1e5])
        nd = dimensional_to_nd({"p_grad": pg_dim}, _SCALES, warn_unknown=False)
        expected = pg_dim * _SCALES.L_ref / p_ref
        assert _close(nd["p_grad"], expected)


class TestDensityScaling:
    def test_rho_scaled(self):
        nd = dimensional_to_nd({"rho": 1200.0}, _SCALES, warn_unknown=False)
        assert _close(nd["rho"], 1200.0 / 1000.0)

    def test_rho_t_scaled(self):
        t_ref = _SCALES.t_ref
        nd = dimensional_to_nd({"rho_t": 50.0}, _SCALES, warn_unknown=False)
        expected = 50.0 * t_ref / _SCALES.rho_ref
        assert _close(nd["rho_t"], expected)

    def test_rho_grad_scaled(self):
        nd = dimensional_to_nd({"rho_grad": jnp.array([100.0])}, _SCALES, warn_unknown=False)
        expected = jnp.array([100.0]) * _SCALES.L_ref / _SCALES.rho_ref
        assert _close(nd["rho_grad"], expected)


class TestTemperatureScaling:
    """Plan tests: test_temperature_affine_offset, test_fourier_time_scale"""

    def test_T_affine_offset(self):
        """T* = (T - T0) / dT_ref → T=350, T0=300, dT_ref=50 → T*=1."""
        s = NondimScales(L_ref=1.0, dT_ref=50.0, T0=300.0)
        nd = dimensional_to_nd({"T": 350.0}, s, warn_unknown=False)
        assert _close(nd["T"], 1.0)

    def test_T_affine_below_T0(self):
        s = NondimScales(L_ref=1.0, dT_ref=50.0, T0=300.0)
        nd = dimensional_to_nd({"T": 275.0}, s, warn_unknown=False)
        assert _close(nd["T"], -0.5)

    def test_T_t_convective(self):
        t_ref = _SCALES.t_ref
        nd = dimensional_to_nd({"T_t": 1000.0}, _SCALES, warn_unknown=False)
        expected = 1000.0 * t_ref / _SCALES.dT_ref
        assert _close(nd["T_t"], expected)

    def test_T_t_fourier(self):
        """Plan test: time_scale='fourier' → T_t uses L²/α."""
        s = NondimScales(
            L_ref=0.1, dT_ref=100.0, alpha_ref=1e-6, time_scale="fourier"
        )
        t_ref = s.t_ref  # L²/α = 0.01/1e-6 = 1e4
        T_t_dim = 0.5  # K/s
        nd = dimensional_to_nd({"T_t": T_t_dim}, s, warn_unknown=False)
        expected = T_t_dim * t_ref / s.dT_ref
        assert _close(nd["T_t"], expected)

    def test_T_grad_scaled(self):
        nd = dimensional_to_nd({"T_grad": 500.0}, _SCALES, warn_unknown=False)
        expected = 500.0 * _SCALES.L_ref / _SCALES.dT_ref
        assert _close(nd["T_grad"], expected)

    def test_T_laplacian_scaled(self):
        nd = dimensional_to_nd({"T_laplacian": 2e4}, _SCALES, warn_unknown=False)
        expected = 2e4 * _SCALES.L_ref ** 2 / _SCALES.dT_ref
        assert _close(nd["T_laplacian"], expected)


class TestGenericScalarScaling:
    def test_phi_scaled(self):
        nd = dimensional_to_nd({"phi": 10.0}, _SCALES, warn_unknown=False)
        assert _close(nd["phi"], 10.0 / _SCALES.phi_ref)

    def test_phi_t_scaled(self):
        nd = dimensional_to_nd({"phi_t": 20.0}, _SCALES, warn_unknown=False)
        expected = 20.0 * _SCALES.t_ref / _SCALES.phi_ref
        assert _close(nd["phi_t"], expected)

    def test_phi_tt_wave_scale(self):
        """phi_tt uses t_ref² — correct for wave_equation."""
        s = NondimScales(
            L_ref=1.0, phi_ref=2.0, c_ref=340.0, time_scale="wave"
        )
        t_ref = s.t_ref
        phi_tt_dim = 100.0  # [phi/s²]
        nd = dimensional_to_nd({"phi_tt": phi_tt_dim}, s, warn_unknown=False)
        expected = phi_tt_dim * t_ref ** 2 / s.phi_ref
        assert _close(nd["phi_tt"], expected)


class TestSolidMechanicsScaling:
    def test_stress_scaled(self):
        nd = dimensional_to_nd({"stress": 1e9}, _SCALES, warn_unknown=False)
        assert _close(nd["stress"], 1e9 / _SCALES.E_ref)

    def test_stiffness_tensor_scaled(self):
        C = jnp.eye(6) * 2e11
        nd = dimensional_to_nd({"stiffness_tensor": C}, _SCALES, warn_unknown=False)
        assert _close(nd["stiffness_tensor"], C / _SCALES.E_ref)

    def test_strain_unchanged(self):
        """Strain is dimensionless → scale factor = 1."""
        eps = jnp.array([0.001, 0.0, -0.0005])
        nd = dimensional_to_nd({"strain": eps}, _SCALES, warn_unknown=False)
        assert _close(nd["strain"], eps)


class TestSchrodingerScaling:
    def test_psi_laplacian_scaled(self):
        """psi_laplacian* = L² · ∇²ψ  (already L²-scaled form expected by law)."""
        psi_lap_dim = jnp.array([1e10, -2e10])
        nd = dimensional_to_nd({"psi_laplacian": psi_lap_dim}, _SCALES, warn_unknown=False)
        expected = psi_lap_dim * _SCALES.L_ref ** 2
        assert _close(nd["psi_laplacian"], expected)


class TestTurbulenceScaling:
    def test_nu_eff_scaled(self):
        # ν* = ν / (U_ref * L_ref)
        nu_dim = 1e-5  # [m²/s]
        nd = dimensional_to_nd({"nu_eff": nu_dim}, _SCALES, warn_unknown=False)
        expected = nu_dim / (_SCALES.U_ref * _SCALES.L_ref)
        assert _close(nd["nu_eff"], expected)

    def test_strain_rate_magnitude_scaled(self):
        # |S|* = (L_ref / U_ref) · |S|
        srm_dim = 50.0  # [1/s]
        nd = dimensional_to_nd({"strain_rate_magnitude": srm_dim}, _SCALES, warn_unknown=False)
        expected = srm_dim * _SCALES.L_ref / _SCALES.U_ref
        assert _close(nd["strain_rate_magnitude"], expected)

    def test_Delta_scaled(self):
        nd = dimensional_to_nd({"Delta": 0.01}, _SCALES, warn_unknown=False)
        assert _close(nd["Delta"], 0.01 / _SCALES.L_ref)


# ---------------------------------------------------------------------------
# Passthrough keys
# ---------------------------------------------------------------------------

class TestPassthroughKeys:
    """Plan test: test_passthrough_keys_unchanged"""

    def test_re_passes_through(self):
        nd = dimensional_to_nd({"re": 1000.0}, _SCALES)
        assert nd["re"] == 1000.0

    def test_pr_passes_through(self):
        nd = dimensional_to_nd({"pr": 7.0}, _SCALES)
        assert nd["pr"] == 7.0

    def test_fo_passes_through(self):
        nd = dimensional_to_nd({"fo": 0.3}, _SCALES)
        assert nd["fo"] == 0.3

    def test_law_constant_L_passes_through(self):
        nd = dimensional_to_nd({"L": 0.1}, _SCALES)
        assert nd["L"] == 0.1

    def test_law_constant_mu_passes_through(self):
        nd = dimensional_to_nd({"mu": 1e-3}, _SCALES)
        assert nd["mu"] == 1e-3

    def test_passthrough_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dimensional_to_nd({"re": 1000.0, "pr": 7.0, "L": 0.1}, _SCALES)
        assert len(w) == 0

    def test_all_groups_in_passthrough(self):
        """Spot-check that common group names are all in the frozenset."""
        for key in ("re", "pr", "gr", "ma", "we", "fo", "sc", "le", "da", "ec"):
            assert key in _PASSTHROUGH_KEYS, f"Expected {key!r} in _PASSTHROUGH_KEYS"


# ---------------------------------------------------------------------------
# extra_rules
# ---------------------------------------------------------------------------

class TestExtraRules:
    """Plan test: test_extra_rules_override"""

    def test_float_extra_rule_applied(self):
        """User-supplied float scale is applied correctly."""
        nd = dimensional_to_nd(
            {"my_field": 4.0},
            _SCALES,
            extra_rules={"my_field": 0.5},
            warn_unknown=False,
        )
        assert _close(nd["my_field"], 2.0)

    def test_float_extra_rule_overrides_builtin(self):
        """extra_rules float takes precedence over built-in rule for same key."""
        nd = dimensional_to_nd(
            {"u": 2.0},
            _SCALES,
            extra_rules={"u": 0.25},  # custom factor instead of 1/U_ref
            warn_unknown=False,
        )
        assert _close(nd["u"], 2.0 * 0.25)

    def test_callable_extra_rule_applied(self):
        """Callable extra rule fn(value, scales) → value is applied."""
        def my_rule(v, s):
            return jnp.asarray(v) * 10.0  # arbitrary

        nd = dimensional_to_nd(
            {"my_E_field": 1000.0},
            _SCALES,
            extra_rules={"my_E_field": my_rule},
            warn_unknown=False,
        )
        assert _close(nd["my_E_field"], 10000.0)

    def test_callable_extra_rule_not_inverted(self):
        """nd_to_dimensional copies callable-rule keys unchanged."""
        def my_rule(v, s):
            return jnp.asarray(v) * 10.0

        state_nd = {"my_E_field": jnp.array(10000.0)}
        back = nd_to_dimensional(
            state_nd, _SCALES, extra_rules={"my_E_field": my_rule}
        )
        # Cannot invert callable → copied unchanged
        assert _close(back["my_E_field"], 10000.0)


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------

class TestWarnings:
    """Plan tests: test_unknown_key_warns, test_unknown_key_no_warn"""

    def test_unknown_key_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            nd = dimensional_to_nd({"weird_key": 42.0}, _SCALES, warn_unknown=True)
        assert any("weird_key" in str(wi.message) for wi in w)
        assert nd["weird_key"] == 42.0  # value is still copied

    def test_unknown_key_no_warn(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dimensional_to_nd({"weird_key": 42.0}, _SCALES, warn_unknown=False)
        user_warnings = [wi for wi in w if issubclass(wi.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_no_warning_for_recognized_keys(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dimensional_to_nd({"u": 1.0, "T": 310.0, "rho": 1000.0}, _SCALES)
        user_warnings = [wi for wi in w if issubclass(wi.category, UserWarning)]
        assert len(user_warnings) == 0


# ---------------------------------------------------------------------------
# Roundtrip: dimensional → ND → dimensional
# ---------------------------------------------------------------------------

class TestRoundtrip:
    """Plan test: test_roundtrip_nd_to_dimensional"""

    _MULT_KEYS = [
        "x", "y", "z", "t", "u", "v", "w",
        "u_t", "u_grad", "u_laplacian",
        "p", "p_grad",
        "rho", "rho_t", "rho_grad",
        "T_t", "T_grad", "T_laplacian",
        "phi", "phi_t", "phi_grad", "phi_laplacian", "phi_tt",
        "stress", "stiffness_tensor", "strain",
        "nu_eff", "strain_rate_magnitude", "Delta",
    ]

    def _make_state(self):
        """Build a representative state with one scalar per key."""
        return {k: jnp.array(1.0) for k in self._MULT_KEYS}

    def test_multiplicative_roundtrip(self):
        """nd_to_dimensional(dimensional_to_nd(state)) ≈ state for all multiplicative keys."""
        state = self._make_state()
        nd = dimensional_to_nd(state, _SCALES, warn_unknown=False)
        back = nd_to_dimensional(nd, _SCALES)
        for key in self._MULT_KEYS:
            assert _close(back[key], state[key]), (
                f"Roundtrip failed for key {key!r}: "
                f"original={state[key]}, recovered={back[key]}"
            )

    def test_temperature_affine_roundtrip(self):
        """Affine temperature scaling must also round-trip correctly."""
        T_dim = jnp.array(350.0)
        nd = dimensional_to_nd({"T": T_dim}, _SCALES, warn_unknown=False)
        back = nd_to_dimensional(nd, _SCALES)
        assert _close(back["T"], T_dim)

    def test_passthrough_roundtrip(self):
        nd = dimensional_to_nd({"re": 1000.0, "L": 0.1}, _SCALES)
        back = nd_to_dimensional(nd, _SCALES)
        assert back["re"] == 1000.0
        assert back["L"] == 0.1

    def test_extra_rules_float_roundtrip(self):
        state = {"my_B": jnp.array(2.0)}
        nd = dimensional_to_nd(state, _SCALES, extra_rules={"my_B": 3.0}, warn_unknown=False)
        back = nd_to_dimensional(nd, _SCALES, extra_rules={"my_B": 3.0})
        assert _close(back["my_B"], state["my_B"])

    def test_psi_laplacian_roundtrip(self):
        psi_lap = jnp.array(1e10)
        nd = dimensional_to_nd({"psi_laplacian": psi_lap}, _SCALES, warn_unknown=False)
        back = nd_to_dimensional(nd, _SCALES)
        assert _close(back["psi_laplacian"], psi_lap)


# ---------------------------------------------------------------------------
# Full NS state dict
# ---------------------------------------------------------------------------

class TestFullNSStateDict:
    """
    Plan test: test_full_ns_state_dict.

    Supply a realistic 2-D NS state (u, v, p_grad, u_grad, u_laplacian,
    rho, x, y, t) and verify every key against the analytic formula.
    """

    def setup_method(self):
        self.s = NondimScales(
            L_ref=0.1,
            U_ref=1.0,
            rho_ref=1000.0,
            dT_ref=50.0,
            T0=293.0,
            time_scale="convective",
        )
        self.p_ref = self.s._p_ref  # 1000 * 1² = 1000 Pa
        self.t_ref = self.s.t_ref   # 0.1 / 1.0 = 0.1 s

        self.state = {
            "x": jnp.array(0.05),          # [m]
            "y": jnp.array(0.03),          # [m]
            "t": jnp.array(0.02),          # [s]
            "u": jnp.array(0.5),           # [m/s]
            "v": jnp.array(-0.2),          # [m/s]
            "rho": jnp.array(998.0),       # [kg/m³]
            "T": jnp.array(303.0),         # [K]
            "p_grad": jnp.array([500.0, -300.0]),    # [Pa/m]
            "u_grad": jnp.ones((2, 2)) * 3.0,        # [1/s]
            "u_laplacian": jnp.array([10.0, -5.0]),  # [m/s / m²]
            "re": 1000.0,   # dimensionless — passthrough
            "L": 0.1,       # [m] law constant — passthrough
        }

    def test_all_keys_present_in_output(self):
        nd = dimensional_to_nd(self.state, self.s)
        for key in self.state:
            assert key in nd

    def test_x_value(self):
        nd = dimensional_to_nd(self.state, self.s)
        assert _close(nd["x"], 0.05 / 0.1)

    def test_t_value(self):
        nd = dimensional_to_nd(self.state, self.s)
        assert _close(nd["t"], 0.02 / self.t_ref)

    def test_u_value(self):
        nd = dimensional_to_nd(self.state, self.s)
        assert _close(nd["u"], 0.5 / 1.0)

    def test_rho_value(self):
        nd = dimensional_to_nd(self.state, self.s)
        assert _close(nd["rho"], 998.0 / 1000.0)

    def test_T_value(self):
        """T=303, T0=293, dT_ref=50 → T*=0.2"""
        nd = dimensional_to_nd(self.state, self.s)
        assert _close(nd["T"], (303.0 - 293.0) / 50.0)

    def test_p_grad_value(self):
        nd = dimensional_to_nd(self.state, self.s)
        expected = jnp.array([500.0, -300.0]) * self.s.L_ref / self.p_ref
        assert _close(nd["p_grad"], expected)

    def test_u_grad_value(self):
        nd = dimensional_to_nd(self.state, self.s)
        expected = jnp.ones((2, 2)) * 3.0 * self.s.L_ref / self.s.U_ref
        assert _close(nd["u_grad"], expected)

    def test_u_laplacian_value(self):
        nd = dimensional_to_nd(self.state, self.s)
        expected = jnp.array([10.0, -5.0]) * self.s.L_ref ** 2 / self.s.U_ref
        assert _close(nd["u_laplacian"], expected)

    def test_re_passthrough(self):
        nd = dimensional_to_nd(self.state, self.s)
        assert nd["re"] == 1000.0

    def test_L_passthrough(self):
        nd = dimensional_to_nd(self.state, self.s)
        assert nd["L"] == 0.1


# ---------------------------------------------------------------------------
# _FIELD_SCALE_RULES completeness
# ---------------------------------------------------------------------------

class TestRulesRegistry:
    def test_all_rules_are_two_tuples(self):
        for key, rule in _FIELD_SCALE_RULES.items():
            assert (
                isinstance(rule, tuple) and len(rule) == 2
            ), f"Rule for {key!r} must be (fwd, inv) tuple"

    def test_forward_and_inverse_callable(self):
        for key, (fwd, inv) in _FIELD_SCALE_RULES.items():
            assert callable(fwd), f"Forward fn for {key!r} not callable"
            assert callable(inv), f"Inverse fn for {key!r} not callable"

    def test_expected_keys_present(self):
        expected = {
            "x", "y", "z", "t",
            "u", "v", "w", "u_t", "u_grad", "u_laplacian",
            "p", "p_grad",
            "rho", "rho_t", "rho_grad",
            "T", "T_t", "T_grad", "T_laplacian",
            "phi", "phi_t", "phi_grad", "phi_laplacian", "phi_tt",
            "psi_laplacian",
            "stress", "stiffness_tensor", "strain",
            "nu_eff", "nu_molecular",
            "strain_rate_magnitude", "Delta",
        }
        for key in expected:
            assert key in _FIELD_SCALE_RULES, f"{key!r} missing from _FIELD_SCALE_RULES"
