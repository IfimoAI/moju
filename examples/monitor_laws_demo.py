#!/usr/bin/env python3
"""
Minimal monitor demo: governing law residual + audit.
"""

import jax.numpy as jnp

from moju.monitor import ResidualEngine, audit, build_loss


def main():
    engine = ResidualEngine(
        laws=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_laplacian"}}]
    )
    state_pred = {"phi_laplacian": jnp.array(1.0)}
    residuals = engine.compute_residuals(state_pred, log_to_python=True)
    print("laws/laplace_equation:", float(residuals["laws"]["laplace_equation"]))
    print("loss:", float(build_loss(residuals)))
    report = audit(engine.log)
    print("overall:", report["overall_admissibility_score"])


if __name__ == "__main__":
    main()

