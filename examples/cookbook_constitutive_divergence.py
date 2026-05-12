"""
Cookbook: visualise where the law-implied and catalog-model constitutive
terms diverge for a Fourier-conduction audit.

Pipeline
--------

1. Build a tiny 1D slab cooling state (``T`` over ``x`` and ``t``).
2. Run :class:`ResidualEngine` with the **balance-mode** constitutive audit
   for ``thermal_diffusivity`` against ``Laws.fourier_conduction``.
3. Inspect the ``closure_debug`` sidecar on
   :attr:`engine.last_residuals` to confirm ``pred`` / ``scale_a`` / ``scale_b``
   are populated.
4. Render the four-mode Constitutive Divergence card and the 2×2 composite
   dashboard.
5. Export the dashboard as a self-contained HTML.

Run
---

``python examples/cookbook_constitutive_divergence.py``
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from moju.monitor import ResidualEngine, build_monitor_visualize_bundle
from moju.monitor.visualize_constitutive import (
    build_constitutive_divergence_card,
    build_constitutive_divergence_dashboard,
    list_constitutive_basenames,
)
from moju.monitor.visualize_export import export_dashboard_html


def _slab_state(nx: int = 32, nt: int = 16, alpha_true: float = 1.4e-5) -> dict:
    """Synthetic 1D unsteady conduction snapshot at one time step.

    Uses an analytic Gaussian temperature profile so we can compute exact
    ``T_t`` and ``T_xx``; the catalog ``thermal_diffusivity(k, rho, cp)`` will
    therefore equal the implied ``alpha = T_t / T_xx`` everywhere.
    """
    x = np.linspace(-1.0, 1.0, nx)
    t = 1.0  # single snapshot time
    sigma2 = 0.05 + alpha_true * t * 2.0  # diffuses with time
    T = np.exp(-(x ** 2) / sigma2)
    # Analytic derivatives of T = exp(-x^2 / σ²(t)), with σ²(t) = σ0² + 2·α·t
    T_t = T * (x ** 2) / (sigma2 ** 2) * 2.0 * alpha_true
    T_xx = T * (4.0 * x ** 2 - 2.0 * sigma2) / (sigma2 ** 2)
    # Material primitives: choose k, rho, cp consistent with alpha_true
    rho_val, cp_val = 1.0, 1.0
    k_val = alpha_true * rho_val * cp_val
    return {
        "T": jnp.asarray(T),
        "T_t": jnp.asarray(T_t),
        "T_xx": jnp.asarray(T_xx),
        "T_laplacian": jnp.asarray(T_xx),
        "x": jnp.asarray(x),
        "t": jnp.asarray(t),
        "k": jnp.asarray(k_val),
        "rho": jnp.asarray(rho_val),
        "cp": jnp.asarray(cp_val),
    }


def main() -> None:
    state = _slab_state()

    # Fourier balance closure: ∂T/∂t = α · ∇²T
    def fourier_balance(st, _const, alpha_pred):
        tt = jnp.asarray(st["T_t"])
        lap = jnp.asarray(st["T_laplacian"])
        a = jnp.asarray(alpha_pred)
        d = a * lap
        return tt - d, tt, d

    engine = ResidualEngine(
        laws=[],
        constitutive_audit=[
            {
                "name": "thermal_diffusivity",
                "output_key": "alpha",
                "state_map": {"k": "k", "rho": "rho", "cp": "cp"},
                "implied_balance_fn": fourier_balance,
                "residual_basename": "thermal_diffusivity/law_fourier_conduction",
            }
        ],
    )
    residuals = engine.compute_residuals(state)

    debug = residuals.get("closure_debug", {})
    print(f"closure_debug entries: {sorted(debug.keys())}")
    for key, entry in debug.items():
        print(
            f"  - {key}: mode={entry['mode']} pred.shape="
            f"{np.asarray(entry['pred']).shape} "
            f"scale_a.shape={None if entry['scale_a'] is None else np.asarray(entry['scale_a']).shape}"
        )

    bundle = build_monitor_visualize_bundle(engine.log, engine=engine, mode="eval")
    print("Available constitutive basenames:", list_constitutive_basenames(bundle))

    for mode in ("spatial", "scatter", "distribution", "hotspot"):
        fig = build_constitutive_divergence_card(bundle, mode=mode)
        out = Path(f"constitutive_divergence_{mode}.html")
        export_dashboard_html(fig, out, title=f"Constitutive divergence — {mode}")
        print(f"Wrote {out}")

    composite = build_constitutive_divergence_dashboard(bundle)
    out_full = Path("constitutive_divergence_dashboard.html")
    export_dashboard_html(composite, out_full, title="Constitutive divergence dashboard")
    print(f"Wrote {out_full}")


if __name__ == "__main__":
    main()
