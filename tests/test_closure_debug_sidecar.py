"""
Tests for the ``closure_debug`` sidecar produced by the residual engines.

Covers:
- :func:`moju.monitor.closure_registry.compute_implied_delta_with_debug` for
  both subtract and balance modes.
- :class:`moju.monitor.auditor.ResidualEngine.compute_residuals` populating
  ``residuals["closure_debug"]`` and the public
  :attr:`engine.last_residuals` accessor.
- :class:`moju.torch.TorchResidualEngine` mirroring the same sidecar.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest


def test_compute_implied_delta_with_debug_subtract_mode() -> None:
    from moju.monitor.closure_registry import (
        MODEL_FNS,
        compute_implied_delta_with_debug,
    )

    fn, arg_names = MODEL_FNS["ideal_gas_rho"]
    P, R, T = jnp.array(1.0e5), jnp.array(287.0), jnp.array(300.0)
    rho = fn(P, R, T)
    merged = {"P": P, "R": R, "T": T, "rho_implied": rho}
    delta, debug = compute_implied_delta_with_debug(
        fn=fn,
        arg_names=arg_names,
        state_map={"P": "P", "R": "R", "T": "T"},
        state_pred=merged,
        constants={},
        implied_value_key="rho_implied",
        output_key="rho",
    )
    assert delta is not None
    assert debug is not None
    assert debug["mode"] == "subtract"
    assert debug["pred"] is not None
    assert debug["implied"] is not None
    assert debug["scale_a"] is None
    assert debug["scale_b"] is None


def test_subtract_debug_broadcasts_scalar_pred_to_implied_shape() -> None:
    from moju.monitor.closure_registry import (
        MODEL_FNS,
        compute_implied_delta_with_debug,
    )

    fn, arg_names = MODEL_FNS["thermal_diffusivity"]
    merged = {
        "k": jnp.array(2.0),
        "rho": jnp.array(1.0),
        "cp": jnp.array(1.0),
        "alpha_implied": jnp.array([2.0, 2.0, 2.0]),
    }
    delta, debug = compute_implied_delta_with_debug(
        fn=fn,
        arg_names=arg_names,
        state_map={"k": "k", "rho": "rho", "cp": "cp"},
        state_pred=merged,
        constants={},
        implied_value_key="alpha_implied",
        output_key="alpha",
    )
    assert delta is not None
    assert debug is not None
    assert debug["mode"] == "subtract"
    assert debug["raw"].shape == (3,)
    assert debug["pred"].shape == (3,)
    assert debug["implied"].shape == (3,)
    assert jnp.allclose(debug["pred"], jnp.array([2.0, 2.0, 2.0]))


def test_compute_implied_delta_with_debug_balance_mode() -> None:
    from moju.monitor.closure_registry import (
        MODEL_FNS,
        compute_implied_delta_with_debug,
    )

    fn, arg_names = MODEL_FNS["thermal_diffusivity"]
    k, rho, cp = jnp.array(1.0), jnp.array(1.0), jnp.array(1.0)
    alpha_m = fn(k, rho, cp)
    T_lap = jnp.array(1.0)
    T_t = alpha_m * T_lap
    merged = {"k": k, "rho": rho, "cp": cp, "T_t": T_t, "T_xx": T_lap}

    def balance(st, _c, pred):
        tt = jnp.asarray(st["T_t"])
        lap = jnp.asarray(st["T_xx"])
        p = jnp.asarray(pred)
        d = p * lap
        return tt - d, tt, d

    delta, debug = compute_implied_delta_with_debug(
        fn=fn,
        arg_names=arg_names,
        state_map={"k": "k", "rho": "rho", "cp": "cp"},
        state_pred=merged,
        constants={},
        implied_balance_fn=balance,
        output_key="alpha",
    )
    assert delta is not None
    assert debug is not None
    assert debug["mode"] == "balance"
    assert debug["scale_a"] is not None
    assert debug["scale_b"] is not None


def test_engine_populates_closure_debug_sidecar() -> None:
    from moju.piratio.models import Models
    from moju.monitor.auditor import ResidualEngine

    P, R, T = jnp.array(1.0e5), jnp.array(287.0), jnp.array(300.0)
    rho = Models.ideal_gas_rho(P, R, T)
    engine = ResidualEngine(
        laws=[],
        constitutive_audit=[
            {
                "name": "ideal_gas_rho",
                "output_key": "rho",
                "state_map": {"P": "P", "R": "R", "T": "T"},
                "implied_value_key": "rho_implied",
            }
        ],
    )
    res = engine.compute_residuals({"P": P, "R": R, "T": T, "rho": rho, "rho_implied": rho})
    assert "closure_debug" in res
    cd = res["closure_debug"]
    assert isinstance(cd, dict) and cd
    # Basename is the model name by default
    assert "ideal_gas_rho" in cd
    entry = cd["ideal_gas_rho"]
    assert entry["mode"] == "subtract"
    assert entry["pred"] is not None
    assert entry["category"] == "constitutive"
    assert entry["model_name"] == "ideal_gas_rho"
    # last_residuals exposes the same sidecar
    assert engine.last_residuals.get("closure_debug") == cd


def test_engine_flatten_skips_closure_debug() -> None:
    """closure_debug must not leak into the flattened RMS/log payload."""
    from moju.piratio.models import Models
    from moju.monitor.auditor import ResidualEngine

    P, R, T = jnp.array(1.0e5), jnp.array(287.0), jnp.array(300.0)
    rho = Models.ideal_gas_rho(P, R, T)
    engine = ResidualEngine(
        laws=[],
        constitutive_audit=[
            {
                "name": "ideal_gas_rho",
                "output_key": "rho",
                "state_map": {"P": "P", "R": "R", "T": "T"},
                "implied_value_key": "rho_implied",
            }
        ],
    )
    engine.compute_residuals({"P": P, "R": R, "T": T, "rho": rho, "rho_implied": rho})
    last_log = engine.log[-1]
    assert all(not k.startswith("closure_debug") for k in (last_log.get("rms") or {}))
    assert all(not k.startswith("closure_debug") for k in (last_log.get("scale") or {}))


def test_torch_engine_closure_debug_sidecar() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("jax2torch")
    from moju.piratio.models import Models
    from moju.torch import TorchResidualEngine

    # ideal_gas_rho subtract mode: pred == implied → debug present, delta ~ 0
    Pv = torch.tensor([1.0e5, 1.1e5], dtype=torch.float64)
    Rv = torch.tensor([287.0, 287.0], dtype=torch.float64)
    Tv = torch.tensor([300.0, 290.0], dtype=torch.float64)
    rho_arr = np.asarray(Models.ideal_gas_rho(jnp.array([1.0e5, 1.1e5]), jnp.array(287.0), jnp.array([300.0, 290.0])))
    rho_t = torch.tensor(np.asarray(rho_arr), dtype=torch.float64)

    def implied_rho(state, _const):
        return state["rho_implied"]

    engine = TorchResidualEngine(
        laws=[],
        constitutive_audit=[
            {
                "name": "ideal_gas_rho",
                "output_key": "rho",
                "state_map": {"P": "P", "R": "R", "T": "T"},
                "implied_fn_torch": implied_rho,
                "residual_basename": "ideal_gas_rho",
            }
        ],
    )
    state = {"P": Pv, "R": Rv, "T": Tv, "rho": rho_t, "rho_implied": rho_t}
    res = engine.compute_residuals_torch(state)
    assert "closure_debug" in res
    cd = res["closure_debug"]
    assert "ideal_gas_rho" in cd
    entry = cd["ideal_gas_rho"]
    assert entry["mode"] == "subtract"
    assert entry["pred"] is not None
    assert entry["implied"] is not None
