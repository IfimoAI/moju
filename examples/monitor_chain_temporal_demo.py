#!/usr/bin/env python3
"""
Minimal monitor demo (Path B): temporal chain closure for a Model.

We audit Models.sutherland_mu(T, mu0, T0, S) using the temporal chain closure:
  dmu/dt ?= (dmu/dT) * dT/dt
"""

import jax.numpy as jnp

from moju.monitor import ResidualEngine, audit


def main():
    constants = {"mu0": 1.8e-5, "T0": 273.0, "S": 110.4}
    engine = ResidualEngine(
        constants=constants,
        laws=[],
        constitutive_audit=[
            {
                "name": "sutherland_mu",
                "output_key": "mu",
                "state_map": {"T": "T", "mu0": "mu0", "T0": "T0", "S": "S"},
                "predicted_temporal": ["T"],
            }
        ],
    )

    # Synthetic time snapshot: pick T and dT/dt, and set mu = sutherland_mu(T).
    T = jnp.array(300.0)
    dT_dt = jnp.array(2.0)
    mu = constants["mu0"] * (T / constants["T0"]) ** 1.5 * (constants["T0"] + constants["S"]) / (
        T + constants["S"]
    )

    # Provide an intentionally wrong dmu/dt so chain residual is nonzero.
    state_pred = {"T": T, "mu": mu, "d_T_dt": dT_dt, "d_mu_dt": jnp.array(0.0)}
    residuals = engine.compute_residuals(state_pred, log_to_python=True)
    print("sutherland_mu/chain_dt:", float(residuals["constitutive"]["sutherland_mu/chain_dt"]))

    report = audit(engine.log)
    print("Overall score:", report["overall_admissibility_score"])
    print("Per-category:", report.get("per_category", {}))


if __name__ == "__main__":
    main()

