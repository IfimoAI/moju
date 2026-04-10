# Monitor: training vs eval (`run_mode`)

`ResidualEngine.compute_residuals` supports **`run_mode="training"`** (default) and **`run_mode="eval"`**.

## Training (`run_mode="training"`)

Use inside optimization loops.

- **Laws** and **constitutive `implied_delta`** (including law-linked implied rows) run as before.
- **`state_ref` is ignored** for:
  - **constitutive** and **scaling** **`ref_delta`**
  - the **`data/`** block (per-key prediction − reference on overlapping keys)
- **π-constant** scaling (`invariance_pi_constant`) is **skipped** (Path A only when enabled in eval).

Each log entry stores **`run_mode`**. **`audit()`** / **`_compute_log_step_metrics`** compute **overall admissibility** as the geometric mean of **laws** and **constitutive** only for those entries.

**Plotly `visualize(..., mode="training")`** shows **two** KPI cards: Governing and Constitutive.

## Eval (`run_mode="eval"`)

Use after training (or any time you have a reference or want π-constant checks).

- Pass **`state_ref`** to enable **`ref_delta`** and **`data/`** residuals.
- **π-constant** runs only for **Path A** (`state_builder` + `model`, `params`, `collocation`; no `state_pred` argument).
- Overall admissibility uses **all** categories present in the RMS keys (`laws`, `constitutive`, `scaling`, `data`).

**`visualize(..., mode="eval")`** can show **four** KPI cards (Governing, Constitutive, Scaling, Data) when those category scores exist. **`mode="test"`** is still accepted and behaves the same (no deprecation warning).

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
