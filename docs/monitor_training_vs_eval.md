# Monitor: training vs eval (`run_mode`)

`ResidualEngine.compute_residuals` supports **`run_mode="training"`** (default) and **`run_mode="eval"`**.

Per flat residual key, the log’s **`rms`** field is **R_eff** = √(mean(r²)+δ²)·**Q^0.5** with **δ²** = **`R_EFF_RMS_JITTER_SQ`** in `moju.monitor.auditor`, **Q** = RMS(m)/mean(m), **m_i** = √(r_i²+ε²) over collocation values (**Q = 1** when |r| is uniform, or for a single point). **R_norm** = **R_eff**/scale_k as elsewhere in the monitor. Default **`scale_k`** for **laws/** and nondimensional **implied_delta** / **ref_delta** is **2×10⁻²** (`DEFAULT_NONDIM_R_NORM_SCALE_K` in `moju.monitor.auditor`). Optional **`audit` / `visualize`** argument **`r_ref`** overrides **`scale_k`** per key.

## Training (`run_mode="training"`)

Use inside optimization loops.

- **Laws** and **constitutive `implied_delta`** (including law-linked implied rows) run as before.
- **`state_ref` is ignored** for:
  - **constitutive** and **scaling** **`ref_delta`**
  - the **`data/`** block (per-key prediction − reference on overlapping keys)
- **π-constant** scaling (`invariance_pi_constant`) is **skipped** (Path A only when enabled in eval).

Each log entry stores **`run_mode`**. **`audit()`** / **`_compute_log_step_metrics`** compute **overall admissibility** as the geometric mean of **laws** and **constitutive** only for **training** entries; for **eval** entries the overall is **not defined** (**`nan`**). Legacy entries **without** **`run_mode`** keep a single overall as the geometric mean of all present categories.

**Plotly `visualize(..., mode="training")`** shows **two** KPI cards: Governing and Constitutive.

## Eval (`run_mode="eval"`)

Use after training (or any time you have a reference or want π-constant checks).

- Pass **`state_ref`** to enable **`ref_delta`** and **`data/`** residuals.
- **π-constant** runs only for **Path A** (`state_builder` + `model`, `params`, `collocation`; no `state_pred` argument).
- **`audit()`** does **not** define a single **overall** admissibility for eval logs (**`nan`** / **Unknown**); **`per_category`** and **`per_key`** still include **`data/`** when present.

**`visualize(..., mode="eval")`** shows **three** KPI cards (Governing, Constitutive, Scaling) when those category scores exist—**no** Data KPI (data category remains in the breakdown / per-key). When the log entry has **`run_mode="eval"`**, the title subtitle explains that roll-up overall is **not defined** (instead of an **N/A** “final overall” line). **`dashboard_mode="dash-tabs"`** KPI tab shows the same **category indicators** plus a short **`run_mode`** note (not an empty placeholder). **`mode="test"`** is still accepted and behaves the same (no deprecation warning).

## Law → π-constant defaults (opt-in)

`MonitorConfig` fields (all optional, default off):

| Field | Purpose |
|-------|---------|
| `pi_constant_law_defaults_enabled` | When true, eval engine builder appends π-constant `scaling_audit` rows from law names. |
| `pi_constant_default_c` | Scale factor `c>1` (default `10`). |
| `pi_constant_law_group_overrides` | Per-law list of group names **replacing** the built-in primary map. |
| `pi_constant_extra_groups` | Extra `Groups.*` names to append. |
| `pi_constant_default_compare_keys` | `invariance_compare_keys`; if empty, uses `primary_fields`. |

Build an eval engine without polluting your training config:

```python
from moju.monitor import MonitorConfig, build_residual_engine_for_pi_constant_eval

eval_engine = build_residual_engine_for_pi_constant_eval(
    base_config,
    state_builder=my_state_builder,
    constants={"alpha": ..., "t": ..., "L": ...},  # merge into engine.constants; must satisfy π-constant recipes
)
residuals = eval_engine.compute_residuals(
    None, model, params, collocation, run_mode="eval"
)
```

Registry: `LAW_PRIMARY_PI_GROUPS` and helpers in `moju.monitor.law_group_defaults`.

## PDF reports

`write_audit_pdf` omits **scaling** and **data** sections when `report["monitor_run_mode"] == "training"` (set by `audit()` from the last log entry).

## Migration

Callers who passed **`state_ref` every step** during training must switch to **`run_mode="eval"`** for that comparison (or a dedicated eval call after training).
