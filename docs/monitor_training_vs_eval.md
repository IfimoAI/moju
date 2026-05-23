# Monitor: training vs eval (`run_mode`)

`ResidualEngine.compute_residuals` supports **`run_mode="training"`** (default) and **`run_mode="eval"`**.

Per flat residual key, the log's **`rms`** field is **R_eff** by default — **sqrt(mean(r^2)+δ^2)** with **δ²** = **`R_EFF_RMS_JITTER_SQ`** in `moju.monitor.auditor` (smooth RMS at zero residual). Optionally **R_eff** = RMS_δ **· Q^p**, **Q** = RMS(m)/mean(m), **m_i** = sqrt(r_i^2 + ε²) over collocation values; set **`p`** globally with **`configure_r_eff(q_power=…)`** (default **p** = **0**, so **Q** is omitted). Typical hotspot-sensitive monitoring uses **`configure_r_eff(q_power=2.0)`**. **R_norm** = **R_eff**/scale_k as elsewhere in the monitor. Default **`law_scale_mode="auto"`** sets governing **laws/** **`scale_k`** from term-balance RMS in merged state (floored at **`DEFAULT_NONDIM_R_NORM_SCALE_K ≈ 1e-2`**). Closure **`implied_delta` / `ref_delta`** stay fixed **`≈ 1e-2`**. Set **`law_scale_mode="fixed"`** for the legacy gauge on all laws. Each log entry stores **`scale_source`** per key (`auto`, `auto_fallback`, `fixed`, `state_derived`). Optional **`audit` / `visualize`** argument **`r_ref`** overrides **`scale_k`** per key.

## Path B state units (`state_units`)

Default **`state_units="nondimensional"`** — uploaded arrays are already in the scaling **`Laws.*`** expect.

Opt in with **`state_units="dimensional"`** (SI Path B) on **`ResidualEngine`** or per **`compute_residuals(...)`** call:

1. **Infer** **`NondimScales`** from selected laws + state (`infer_nondim_scales` in `moju.monitor.nondim_inference`).
2. Run **groups** on **physical** state (Re, Fo, … need dimensional inputs).
3. **`dimensional_to_nd`** on field variables (derivatives scaled consistently).
4. Optional Path B FD, then laws and auto **`scale_k`**.

Partial overrides: **`nondim_scales={"L_ref": 0.05, "T0": 300.0}`** on engine, **`MonitorConfig`**, or the compute call. Logged: **`nondim_scale_source`**, **`nondim_scales`**. **`state_ref`** gets the same conversion in **`run_mode="eval"`**. Moju Studio: checkbox **State in physical units (SI)** on the Run tab.

Do **not** rely on magnitude heuristics to detect SI vs ND — declare **`state_units`** explicitly.

## Admissibility metrics

| What | Governing **laws/** | Constitutive **`implied_delta`** / **`ref_delta`** |
|------|---------------------|-----------------------------------------------------|
| **Training loss / logged `rms`** | RMS **R_eff** | RMS **R_eff** (unchanged) |
| **Audit admissibility scalar** | RMS **R_eff** | **Worst-point** **`r_max = max \|δ\|`** |
| **Category rollup** | Geometric mean | **Minimum** |
| **Overall (training)** | `min(laws, constitutive)` | (same) |

**Why:** a model can satisfy governing laws on average while violating constitutive closure at isolated collocation points. Worst-point scoring on **`implied_delta`** / **`ref_delta`** catches that cheat; RMS on laws reflects typical PDE compliance.

Each **`compute_residuals`** log entry stores **`rms`** (RMS per key, used by training plots and **`build_loss`**) and a sparse **`r_max`** dict (worst-point magnitudes for ND closure keys only). **`audit()`** / **`per_key_report`** add **`admissibility_metric`** (`"rms"` or `"max"`), **`score_for_admissibility`**, and optional **`r_max`**. Legacy logs without **`r_max`** fall back to RMS for closure admissibility. The dashboard **Summary** box and audit PDF **constitutive closure summary** lead with worst-point error and list RMS as a diagnostic.

## Calibrating `scale_k` (`law_scale_mode`, `r_ref`)

**`R_norm = R_eff / scale_k`** (or **`r_max / scale_k`** for closure admissibility). Admissibility is **`1 / (1 + R_norm)`**.

### `law_scale_mode` (governing laws)

| Mode | Behavior |
|------|----------|
| **`"auto"`** (default) | Term-balance RMS from merged state per law (`moju.monitor.law_scale_recipes`); floor at **`≈ 1e-2`** |
| **`"fixed"`** | **`scale_k ≈ 1e-2`** for all **laws/** (pedagogical / tier-aligned demos) |

Closure **`implied_delta` / `ref_delta`** always use fixed **`≈ 1e-2`** (fractional δ; tiers match Consistency bands).

Pass **`r_ref`** (flat key → positive float) to **`audit(log, r_ref=...)`** and **`visualize(..., r_ref=...)`** to override **`scale_k`** per key after logging. Precedence: **`r_ref`** > logged **`entry["scale"]`** > first-step RMS fallback > 1.

### Which keys use which default

| Key family | Default `scale_k` | Notes |
|------------|-------------------|--------|
| **`laws/*`** | **Auto** term-balance (fallback **`≈ 1e-2`**) | Set **`law_scale_mode="fixed"`** for legacy gauge |
| **`constitutive/.../implied_delta`**, **`ref_delta`** | Fixed **`≈ 1e-2`** | Fractional δ; tiers stable |
| **Other `constitutive/*`**, **`data/*`** | State- or reference-derived RMS | Unchanged |

### When to override

Consider per-key **`r_ref`** or **`law_scale_mode="fixed"`** when:

1. You want governing-law tier numbers aligned to the **`1e-2`** closure gauge (tutorials).
2. You have a **baseline** and want **`R_norm ≈ 1`** at that baseline for specific laws.
3. Auto term-balance is misleading for your custom nondimensionalization.

Usually **keep defaults** for **`implied_delta` / `ref_delta`**: interpret closure via **Consistency bands** and the **worst-point summary**; those bands are pure fractions and do not depend on **`r_ref`**.

### How to choose a value

Pick **`scale_k`** = the violation magnitude you treat as **1× reference error** for that flat key (e.g. `"laws/momentum_navier_stokes"`):

- **From a trusted baseline:** set **`r_ref[key] = rms_baseline`** from a reference run’s logged **`rms`**. A model matching that baseline then lands near **50% admissibility** under **`A = 1/(1 + r/scale_k)`**.
- **From an explicit tolerance:** if you accept law RMS up to **`τ`** in your units, use **`r_ref[key] = τ`**.
- **Same dict for audit and visualize** so KPI cards, PDF, and plots stay consistent.

Example:

```python
r_ref = {
    "laws/momentum_navier_stokes": 0.05,  # τ or baseline RMS for this law
}
report = audit(engine.log, r_ref=r_ref)
fig = visualize(engine.log, last_residuals=engine.last_residuals, r_ref=r_ref, mode="training")
```

### Caveats

- **`r_ref`** retunes **admissibility percentages and tier labels** for the keys you override; it does **not** change constitutive **±0.1 / 0.5 / 1% plot bands** (those stay fractional).
- Global tier cutoffs (**`ADM_HIGH_THRESHOLD`**, etc.) assume fixed **`scale_k`** for **closure** keys. Under **`law_scale_mode="auto"`**, treat **High / Moderate / Low** on **laws** as calibration-dependent; use **`scale_source`** in logs and **`per_key_report`**.
- **`build_loss`** and logged **`rms`** are **unchanged** by **`r_ref`**; overrides affect **audit / visualize scoring only**.

Moju Studio exposes optional **`r_ref`** JSON for the next dashboard run (see **`apps/moju_studio/README.md`**).

## Explain this audit (`audit_meta`)

Use **`audit_meta(log)`** (or **`build_audit_meta`**) for a plain-language summary of how **`scale_k`** and nondimensional conversion were chosen for a logged step — without parsing raw **`scale_source`** / **`nondim_scales`** dicts.

```python
from moju.monitor import audit, audit_meta

engine.compute_residuals(state_pred, run_mode="training")
report = audit(engine.log)
print(report["audit_meta"]["plain_summary"])

# Standalone on any log entry (default: last step)
meta = audit_meta(engine.log, r_ref={"laws/fourier_conduction": 0.05})
print(meta["plain_sections"]["scaling"])
```

Each **`compute_residuals`** entry now stores **`monitor_settings`**: **`law_scale_mode`**, **`state_units`**. Legacy logs without **`scale_source`** degrade gracefully (`"unknown (legacy log)"`). **`audit(..., r_ref=...)`** attaches **`report["audit_meta"]`** with **`r_ref`** precedence matching **`per_key_report`**. Moju Studio **Dashboard** expander **How scoring was calibrated** and audit PDF section **Scoring calibration** render the same metadata.

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

**Plotly `visualize(..., mode="training")`** shows **two** KPI cards: Governing and Constitutive. The Governing and Constitutive per-key residual time-series panels now plot **`R_eff`** (the raw effective residual that the training loss minimises) — y-axis label `Effective residual (R_eff)` (linear) or `log10(R_eff + ε)` (log). Hovertemplates read `R_eff=…`. The worst-violation marker within each category ranks keys by terminal `R_eff` rather than `R_norm`. When closure debug data is present, the dashboard also renders a constitutive row with a **Divergence** heatmap (the model-normalised fractional residual `δ = (model − implied) / (|model| + ε)`) and a **Constitutive Consistency** line plot with spatially varying **±0.1 % / ±0.5 % / ±1 %** acceptability bands centred on the model prediction. The dashboard **Summary** box and audit PDF include a one-line **constitutive closure summary** (worst-point fractional error, band, RMS diagnostic, worst-point admissibility) when `implied_delta` keys are present. **Category admissibility** on the Constitutive KPI card reflects **minimum** worst-point scores across closure keys.

### Spatial residual heatmaps (training + eval)

Spatial residual panels (training row 5 and eval row 4) always show the per-point absolute residual `|r|`, the same per-point quantity whose RMS feeds `R_eff`. The legacy `spatial_normalize` keyword has been **removed** from `visualize()`, `build_monitor_visualize_bundle()`, `build_visualize_bundle()`, and `build_spatial_rnorm_panels_from_residuals()` — callers that previously passed `spatial_normalize=False` (the default) see no change; callers that previously asked for `|r| / scale_k` must drop the kwarg.

## Eval (`run_mode="eval"`)

Use when you have a reference state or want **`ref_delta`** / **`data/`** comparisons.

- Pass **`state_ref`** to enable constitutive **`ref_delta`** and **`data/`** residuals.
- **`audit()`** rolls up **overall admissibility** for eval as the minimum **finite** per-category score present in that step (**laws**, **constitutive**, **`data`**, and legacy **`scaling`** buckets if old logs still contain `scaling/...` keys).

**`visualize(..., mode="eval")`** uses **two** KPI cards (Governing, Constitutive), matching training layout. The eval **combined bar chart** (row 3, last column) stays on **`R_norm`** — keys are scale-normalised so different residual families can be compared at a glance. Category breakdowns still list whatever categories exist in the log. **`mode="test"`** is accepted as an alias for **`eval`** (no deprecation warning).

## PDF reports

`write_audit_pdf` omits **data** sections when `report["monitor_run_mode"] == "training"` (set by `audit()` from the last log entry).

## Migration

Callers who passed **`state_ref` every step** during training must switch to **`run_mode="eval"`** for that comparison (or a dedicated eval call after training).

**Removed:** `MonitorConfig.scaling_audit`, π-constant / similarity-audit machinery, and related `AuditSpec` fields. Passing **`scaling_audit`** or **`pi_constant_*`** keys to **`MonitorConfig.from_dict`** raises **`ValueError`**. Use **`groups`** specs for `Groups.*` outputs in state; run similarity sweeps outside **`ResidualEngine`** if needed.
