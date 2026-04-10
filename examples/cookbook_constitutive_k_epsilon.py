#!/usr/bin/env python3
"""Cookbook: constitutive **ref_delta** for ``Models.k_epsilon_nu_t`` (Path B)."""

from __future__ import annotations

from typing import Any, Dict

import jax.numpy as jnp

from moju.monitor import AuditSpec, MonitorConfig, ResidualEngine, audit
from moju.piratio.models import Models


def main() -> Dict[str, Any]:
    eps0 = jnp.array(1e-12)
    C_mu = jnp.array(0.09)
    x = jnp.linspace(0.0, 1.0, 40)
    k = 0.2 + 0.15 * x
    epsilon = 0.4 + 0.1 * x
    nu_t = Models.k_epsilon_nu_t(C_mu, k, epsilon, eps0)

    state_pred = {"k": k, "epsilon": epsilon, "nu_t": nu_t, "C_mu": C_mu}
    cfg = MonitorConfig(
        constants={"eps0": eps0},
        laws=[],
        constitutive_audit=[
            AuditSpec(
                name="k_epsilon_nu_t",
                output_key="nu_t",
                state_map={"C_mu": "C_mu", "k": "k", "epsilon": "epsilon", "eps0": "eps0"},
            )
        ],
    )
    engine = ResidualEngine(config=cfg)
    residuals = engine.compute_residuals(
        state_pred, state_ref=dict(state_pred), run_mode="eval"
    )
    report = audit(engine.log)
    flat_key = "constitutive/k_epsilon_nu_t/ref_delta"
    rms = engine.log[-1]["rms"][flat_key]
    return {"report": report, "ref_rms": float(rms), "residuals": residuals, "engine": engine, "flat_key": flat_key}


if __name__ == "__main__":
    out = main()
    print("ref_delta RMS:", out["ref_rms"])
