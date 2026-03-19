#!/usr/bin/env python3
"""
Minimal monitor demo (Path B): spatial chain closure for a Group.

We audit the identity Pe = Re*Pr via the chain closure on Groups.pe(re, pr).
The chain residual checks:
  dPe/dx  ?=  (dPe/dRe)*dRe/dx + (dPe/dPr)*dPr/dx
For Pe = Re*Pr, dPe/dRe = Pr and dPe/dPr = Re.
"""

import jax.numpy as jnp

from moju.monitor import ResidualEngine, audit


def main():
    engine = ResidualEngine(
        laws=[],
        scaling_audit=[
            {
                "name": "pe",
                "output_key": "Pe",
                "state_map": {"re": "Re", "pr": "Pr"},
                "predicted_spatial": ["Re", "Pr"],
            }
        ],
    )

    # Re(x)=10+x, Pr(x)=2 (constant). Then Pe=Re*Pr, dPe/dx should be Pr*dRe/dx = 2*1 = 2.
    state_pred = {
        "Re": jnp.array(10.0),
        "Pr": jnp.array(2.0),
        "Pe": jnp.array(20.0),
        "d_Re_dx": jnp.array(1.0),
        "d_Pr_dx": jnp.array(0.0),
        "d_Pe_dx": jnp.array(2.0),
    }
    residuals = engine.compute_residuals(state_pred, log_to_python=True)
    print("Residual keys:", residuals.get("scaling", {}).keys())
    print("pe/chain_dx:", float(residuals["scaling"]["pe/chain_dx"]))

    report = audit(engine.log)
    print("Overall score:", report["overall_admissibility_score"])
    print("Per-category:", report.get("per_category", {}))


if __name__ == "__main__":
    main()

