# Monitor: training vs eval (`run_mode`)

`ResidualEngine.compute_residuals` supports **`run_mode="training"`** (default) and **`run_mode="eval"`**.

Per flat residual key, the log’s **`rms`** field is **R_eff** = √(mean(r²)+δ²)·**Q**^**p** with **δ²** = **`R_EFF_RMS_JITTER_SQ`** in `moju.monitor.auditor`, exponent **p** = **`R_EFF_Q_POWER`** (default **2.0**), **Q** = RMS(m)/mean(m), **m_i** = √(r_i²+ε²) over collocation values (**Q = 1** when |r| is uniform, or for a single point). **R_norm** = **R_eff**/scale_k as elsewhere in the monitor. Default **`scale_k`** for **laws/** and nondimensional **implied_delta** / **ref_delta** is **1.0×10⁻²** (`DEFAULT_NONDIM_R_NORM_SCALE_K` in `moju.monitor.auditor`). Optional **`audit` / `visualize`** argument **`r_ref`** overrides **`scale_k`** per key.

For minimal workflows, `build_minimal_residual_engine(law_names=[...], coord_dimension=1|2|3)` can auto-wire identity law specs plus inferred `Groups.*` rows and run in best-effort partial mode (skips unresolved rows and logs `unresolved_dependencies`). The configured `coord_dimension` is reused only when you explicitly ask for Path B finite-difference inference with `compute_residuals(..., auto_path_b_derivatives=True)`.

## Minimal inputs by dimension (quick helper)

For direct Path B use, you may provide law inputs and derivatives yourself. For example, a 1D Fourier slab-cooling state can include `T`, `T_t`, `T_laplacian`, `x`, `t`, `L`, `k`, `rho`, `cp`, and `alpha`; no finite-difference inference is needed when those derivative keys are already present.

If you do use Path B finite-difference inference (`auto_path_b_derivatives=True`), provide:

- **1D:** coordinate `x`
- **2D:** coordinates `x`, `y`
- **3D:** coordinates `x`, `y`, `z`
- **Unsteady terms** (e.g., `_t`, `_tt`): add coordinate `t` in any dimension

Also provide the primitive field(s) used by your selected law(s), such as `T` or `u`, plus any required material/property terms not inferable from your supplied state/constants.

## Training (`run_mode="training"`)

Use inside optimization loops.

- **Laws**, **groups** (dimensionless numbers merged into state), and **constitutive** residuals (including law-linked implied rows) run as before.
- **`state_ref` is ignored** for:
  - constitutive **`ref_delta`**
  - the **`data/`** block (per-key prediction − reference on overlapping keys)

Each log entry stores **`run_mode`**. **`audit()`** / **`_compute_log_step_metrics`** compute **overall admissibility** as the minimum of the present **laws** and **constitutive** category scores for **training** entries. Legacy entries **without** **`run_mode`** use the minimum finite score across all present categories (including **`data`** or legacy **`scaling/`** keys if present in old logs).

**Plotly `visualize(..., mode="training")`** shows **two** KPI cards: Governing and Constitutive. When closure debug data is present, the dashboard also renders a constitutive row with a **Divergence** heatmap (normalised as `(model − implied) / (|model| + ε)`) and a **Constitutive Consistency** line plot with spatially varying ±1 % / ±5 % acceptability bands centred on the model prediction.

## Eval (`run_mode="eval"`)

Use when you have a reference state or want **`ref_delta`** / **`data/`** comparisons.

- Pass **`state_ref`** to enable constitutive **`ref_delta`** and **`data/`** residuals.
- **`audit()`** rolls up **overall admissibility** for eval as the minimum **finite** per-category score present in that step (**laws**, **constitutive**, **`data`**, and legacy **`scaling`** buckets if old logs still contain `scaling/...` keys).

**`visualize(..., mode="eval")`** uses **two** KPI cards (Governing, Constitutive), matching training layout. Category breakdowns still list whatever categories exist in the log. **`mode="test"`** is accepted as an alias for **`eval`** (no deprecation warning).

## PDF reports

`write_audit_pdf` omits **data** sections when `report["monitor_run_mode"] == "training"` (set by `audit()` from the last log entry).

## Migration

Callers who passed **`state_ref` every step** during training must switch to **`run_mode="eval"`** for that comparison (or a dedicated eval call after training).

**Removed:** `MonitorConfig.scaling_audit`, π-constant / similarity-audit machinery, and related `AuditSpec` fields. Passing **`scaling_audit`** or **`pi_constant_*`** keys to **`MonitorConfig.from_dict`** raises **`ValueError`**. Use **`groups`** specs for `Groups.*` outputs in state; run similarity sweeps outside **`ResidualEngine`** if needed.
