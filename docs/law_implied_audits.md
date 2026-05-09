# Law-linked implied audits

When you select a governing law in `ResidualEngine` or `MonitorConfig`, Moju can **automatically** add matching **constitutive** audit rows for supported laws. Each row uses the standard **`implied_delta`** closure.

## Subtract mode (generic constitutive audits)

Let \(F\) be the catalog model output (`Models.*` from state) and \(\tilde F\) an alternate value from **`implied_value_key`** or **`implied_fn`**. The stored tensor is **always** the nondimensional discrepancy

\[
R^* = \frac{F - \tilde F}{\varepsilon + |F| + |\tilde F|}
\]

unless a reference field resolves (audit **`implied_delta_ref_key`**, or **`{output_key}_ref`** in merged state/constants), in which case

\[
R^* = \frac{F - \tilde F}{\varepsilon + |\text{ref}|}.
\]

## Balance mode (law-linked coefficient audits)

For several law-linked rows, Moju uses **`implied_balance_fn(state, constants, pred)`** with **`pred = F(...)`** from the catalog model. The closure returns **`(raw, scale_a, scale_b)`** where **`raw`** is the **governing-equation residual** written with **`pred`** as the constitutive coefficient (e.g. **Fourier:** \(T_t - \alpha_{\text{model}}\,T_{\text{laplacian}}\)), and **`scale_a` / `scale_b`** are the two term magnitudes used for symmetric normalization:

\[
R^* = \frac{\text{raw}}{\varepsilon + |\text{scale}_a| + |\text{scale}_b|}
\]

(or **`/ (ε + |ref|)`** when a ref tensor resolves). There is **no** division of fields to recover an “implied” coefficient for these rows.

There is **no** raw SI-difference mode. **`Models.*`** still uses your physical state keys; the **monitor residual** is always normalized as above.

This answers: *“Does the constitutive closure in the catalog agree with what the PDE fields imply locally?”* without requiring **`state_ref`**. It is **not** a claim that the closure matches experiment—only that it matches the **same predicted state** you pass to the law.
These implied residual keys are included in normal category/overall admissibility scoring by default (same as other constitutive residual keys).

**Training vs eval:** **`implied_delta`** law-linked rows run in both **`run_mode="training"`** (default) and **`run_mode="eval"`**. **`ref_delta`** on those rows (and separate **`data/`** pred−ref) runs only when you call **`compute_residuals(..., run_mode="eval", state_ref=...)`**. See [monitor_training_vs_eval.md](monitor_training_vs_eval.md).

## Configuration

| Mechanism | Behavior |
|-----------|----------|
| `MonitorConfig(law_implied_audits=True)` (default) | Prepend law-linked rows before your `constitutive_audit`. |
| `ResidualEngine(..., law_implied_audits=False)` | Skip prepending (dict-only construction). |
| Expert JSON / `merge_simple_config_with_json_override` | Optional `"law_implied_audits": false` to disable. |

Rows are merged in **`merge_law_implied_audit_specs(laws_spec, enabled=...)`**. Inspect coverage with **`list_laws_with_implied_diagnostics()`** and intentional best-effort gaps with **`law_implied_unsupported_reasons()`**.

## Residual keys and `ref_delta`

Each auto row sets **`residual_basename`** so keys stay unique when multiple laws or manual audits use the same `Models.*` / `Groups.*` name:

- Example: `constitutive/thermal_diffusivity/law_fourier_conduction/implied_delta`

If **`state_ref`** is passed and **`run_mode="eval"`**, **`ref_delta`** is computed for the same row **unless** the spec sets **`include_ref_delta: false`**. With **`run_mode="training"`**, **`state_ref`** is ignored for **`ref_delta`** (use an eval pass after training). Nondimensional rules match **`implied_delta`** (symmetric scale, or **`ref_delta_ref_key`** / **`{output_key}_ref`** for the \(|\text{ref}|\) denominator).

## Constitutive-only policy

Moju uses strict constitutive semantics for law-linked implied rows:

- law-linked implied rows are `category: "constitutive"` only;
- group/scaling implied rows are not auto-added;
- scaling-linked laws are mapped through constitutive target models (e.g. viscosity, diffusivity, wave speed) when available.

For **Fourier conduction**, Fo and α are linked by **α = Fo·L²/t** for fixed **t**, **L**. To avoid double-counting and keep constitutive semantics, the registry adds only **`thermal_diffusivity`** implied residual.

## Covered laws (registry)

The mapping lives in **`moju/monitor/law_implied_diagnostics.py`** (`_LAW_IMPLIED_ROWS`). At a glance:

| Law | Auto implied constitutive audit | Notes |
|-----|-------------------------------|-------|
| `fourier_conduction` | `thermal_diffusivity` | Balance \(T_t - \alpha_{\text{model}}\,T_{\text{laplacian}}\); model uses **k**, **rho**, **cp**. |
| `fick_diffusion` | `mass_diffusivity` | Balance \(\phi_t - D_{\text{model}}\,\phi_{\text{laplacian}}\); model uses **fo_mass**, **t**, **L**. |
| `wave_equation` | `wave_speed_from_st` | Balance \(\phi_{tt} - c_{\text{model}}^2\,\phi_{\text{laplacian}}\); model uses **omega**, **L**, **st_wave**. |
| `advection_diffusion` | `scalar_diffusivity_from_pe` | Balance \(\phi_t + \mathbf u\!\cdot\!\nabla\phi - (\kappa_{\text{model}}/(|\mathbf u|L))\,\phi_{\text{laplacian}}\); model uses **u**, **L**, **pe**. |
| `momentum_navier_stokes` | `dynamic_viscosity_from_re` | Vector **`Laws.momentum_navier_stokes`** with **Re** from **ρ|u|L/μ_model**. |
| `stokes_flow` | `dynamic_viscosity_from_re` | Vector **`Laws.stokes_flow`** with **Re** from **ρ|u|L/μ_model** ( **`u`**, **ρ**, **L** may come from state if omitted from the law `state_map`). |
| `burgers_equation` | `dynamic_viscosity_from_re` | Vector **`Laws.burgers_equation`** with **Re** from **ρ|u|L/μ_model**. |
| `momentum_incompressible_newtonian_laplacian` | `turbulent_viscous_acceleration_*` | Three auto rows: k-ω, k-ε, and Smagorinsky; **subtract** mode **`pred − implied_fn`**. |
| `momentum_compressible_newtonian_laplacian` | `turbulent_viscous_acceleration_compressible_*` | Three auto rows: compressible k-ω, k-ε, and Smagorinsky; **subtract** mode. |

## Unsupported laws (best effort)

Laws without an entry add **no** law-linked implied rows. These gaps are intentional and documented in **`law_implied_unsupported_reasons()`** (same module). Typical reasons include:

- no single constitutive/scaling target to rearrange (`mass_incompressible`, `mass_compressible`),
- geometry/material specific inversion not encoded as one catalog closure (`laplace_beltrami`, `hookes_law_residual`),
- model-context-dependent closure choice (`darcy_flow`, `brinkman_extension`),
- laws requiring domain-specific closure/model choices not yet encoded in the constitutive registry.

If you still want implied constitutive checks for unsupported laws, add explicit rows under `constitutive_audit` (for example with `implied_fn` or a custom `implied_balance_fn`).

## Studio and dependency planning

**`build_studio_auto_fragment`** prepends the same dict rows (with **`implied_balance_fn`** or **`implied_fn`** attached) and sets **`law_implied_audits: true`**.

**`apps/moju_studio/studio_dependency_planner`** uses **`effective_audit_specs_for_fragment(fragment)`** so required keys (e.g. **k**, **rho**, **cp** for Fourier) appear in preflight even though JSON exports omit callables.

## User functions for constitutive terms (Python-only)

If a law-linked implied constitutive row requires inputs like **`k`**, **`rho`**, or **`mu`**, you can supply them either as arrays/scalars in `state_pred` / `constants`, **or** as Python callables keyed by the **output state key** via `ResidualEngine(user_fns=...)`.

Example: let Moju materialize `k`, `rho`, and `alpha` from `T` (so the Fourier implied `thermal_diffusivity` check can run without you precomputing those fields):

```python
engine = ResidualEngine(
    constants={"cp": 900.0},
    laws=[{"name": "fourier_conduction", "state_map": {"T_t": "T_t", "T_laplacian": "T_xx", "fo": "fo", "t": "t", "L": "L"}}],
    groups=[{"name": "fo", "output_key": "fo", "state_map": {"alpha": "alpha", "t": "t", "L": "L"}}],
    user_fns={
        "k": lambda T: 200.0 * (1.0 + 0.001 * (T - 400.0)),
        "rho": lambda T: 2700.0 * (1.0 - 0.0001 * (T - 400.0)),
        "alpha": lambda k, rho, cp: k / (rho * cp),
    },
)
```

Notes:
- `user_fns` are **Python-only** (not JSON-serializable); Studio config previews remain conservative and will still list the raw keys (e.g. `k`, `rho`) as required unless you provide them in NPZ/constants.

## Custom extensions

Expert users can attach **`implied_balance_fn`** or **`implied_fn`** on manual **`constitutive_audit`** dict rows (same shapes as law-linked rows). Use **`residual_basename`** for unique log keys when reusing a `Models.*` name.
