"""
CFD snapshot cookbook (1D heat-like scalar), end-to-end.

Workflow:
  xarray/NetCDF → regrid → compute gradients (FD + optional smoothing) → audit → interpret score

Run (recommended):
  pip install moju[ref] moju[report]
  # Optional for better smoothing:
  pip install scipy
  python examples/cfd_snapshot_cookbook_heat_1d.py
"""

from __future__ import annotations

import numpy as np

import jax.numpy as jnp

from moju.monitor import AuditSpec, MonitorConfig, ResidualEngine, audit
from moju.monitor.state_ref import from_xarray


def _smooth_1d(y: np.ndarray, *, window: int = 7) -> np.ndarray:
    window = int(window)
    if window < 3:
        return y
    if window % 2 == 0:
        window += 1
    try:
        from scipy.signal import savgol_filter  # type: ignore

        return savgol_filter(y, window_length=window, polyorder=2, mode="interp")
    except Exception:
        # Numpy-only fallback: simple moving average
        k = np.ones((window,), dtype=float) / float(window)
        ypad = np.pad(y, (window // 2, window // 2), mode="edge")
        return np.convolve(ypad, k, mode="valid")


def _fd_dx(u: np.ndarray, x: np.ndarray) -> np.ndarray:
    # Coordinate-aware finite differences (1D).
    return np.gradient(u, x, edge_order=1)


def main(n: int = 200, noise: float = 0.05, smooth_window: int = 9, export_dir: str = "exports"):
    import xarray as xr  # requires moju[ref] (xarray + pandas)

    # 1) Pretend we loaded a CFD scalar field T(t,x) from NetCDF.
    t = np.array([0.0], dtype=float)
    x = np.linspace(0.0, 1.0, n)
    T_clean = np.sin(2.0 * np.pi * x)
    rng = np.random.default_rng(0)
    T_noisy = T_clean + noise * rng.standard_normal(size=T_clean.shape)

    ds = xr.Dataset(
        data_vars={"T_cfd": (("t", "x"), T_noisy[None, :])},
        coords={"t": t, "x": x},
    )

    # 2) Regrid/interpolate to collocation coordinates (here we keep same grid).
    x_col = np.linspace(0.0, 1.0, n)
    state_ref = from_xarray(
        ds,
        var_map={"T": "T_cfd"},
        target={"t": np.array([0.0]), "x": x_col},
        method="linear",
    )

    # 3) Build a Path B state_pred from the snapshot (here, we treat the snapshot as the prediction too).
    T = np.asarray(state_ref["T"]).reshape(-1)
    T_smooth = _smooth_1d(T, window=smooth_window)
    dT_dx = _fd_dx(T_smooth, x_col)

    # Provide quadrature weights for weak-form integration along x.
    # For structured 1D, trapezoidal-like weights from spacing are a reasonable default.
    w_x = np.gradient(x_col)

    # 4) Configure an audit: use a simple scaling identity Pe = Re*Pr as a stand-in closure.
    cfg = MonitorConfig(
        laws=[],
        scaling_audit=[
            AuditSpec(
                name="pe",
                output_key="Pe",
                state_map={"re": "Re", "pr": "Pr"},
                predicted_spatial=["Re"],
                closure_mode="weak",
                quadrature_weights={"x": "w_x"},
            )
        ],
    )
    engine = ResidualEngine(config=cfg)

    # Synthetic dimensionless numbers driven by T: Re(x) = 1 + T(x), Pr constant, Pe = Re*Pr.
    Pr = 1.0
    Re = 1.0 + T
    Pe = Re * Pr
    dRe_dx = dT_dx
    dPe_dx = dRe_dx * Pr

    state_pred = {
        "Re": jnp.asarray(Re),
        "Pr": jnp.asarray(Pr),
        "Pe": jnp.asarray(Pe),
        "d_Re_dx": jnp.asarray(dRe_dx),
        "d_Pe_dx": jnp.asarray(dPe_dx),
        "w_x": jnp.asarray(w_x),
    }

    engine.compute_residuals(state_pred, state_ref={"Re": jnp.asarray(Re)})
    report = audit(engine.log, export_dir=export_dir, model_name="cfd_snapshot_cookbook_heat_1d")

    pe_score = report["per_key"].get("scaling/pe/chain_dx", {})
    print("Weak chain_dx RMS (Pe closure):", pe_score.get("rms"))
    print("Admissibility score:", pe_score.get("admissibility_score"))
    print("Overall admissibility:", report.get("overall_admissibility_score"))


if __name__ == "__main__":
    main()

