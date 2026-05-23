"""
CFD snapshot cookbook (1D heat-like scalar), end-to-end.

Workflow:
  xarray/NetCDF → regrid → optional smoothing → eval audit (data residual vs reference field) → interpret score

Run (recommended):
  pip install "moju[io]"
  # Optional for better smoothing:
  pip install scipy
  python examples/cfd_snapshot_cookbook_heat_1d.py
"""

from __future__ import annotations

import numpy as np

import jax.numpy as jnp

from moju.monitor import MonitorConfig, ResidualEngine, audit
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


def main(n: int = 200, noise: float = 0.05, smooth_window: int = 9, export_dir: str = "exports"):
    import xarray as xr  # requires moju[io]

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

    # 3) Build state_pred from the snapshot (here, prediction = smoothed field; reference = noisy).
    T = np.asarray(state_ref["T"]).reshape(-1)
    T_smooth = _smooth_1d(T, window=smooth_window)

    # 4) Configure a minimal audit: eval compares prediction to reference on overlapping keys (data residuals).
    cfg = MonitorConfig(laws=[])
    engine = ResidualEngine(config=cfg)

    state_pred = {"T": jnp.asarray(T_smooth)}
    state_ref = {"T": jnp.asarray(T)}

    engine.compute_residuals(state_pred, state_ref=state_ref, run_mode="eval")
    report = audit(engine.log, export_dir=export_dir, model_name="cfd_snapshot_cookbook_heat_1d")

    t_score = report["per_key"].get("data/T", {})
    print("Data residual R_eff (T):", t_score.get("rms"))
    print("Admissibility score:", t_score.get("admissibility_score"))
    print("Overall admissibility:", report.get("overall_admissibility_score"))


if __name__ == "__main__":
    main()

