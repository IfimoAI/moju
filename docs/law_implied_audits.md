# Law-linked implied audits

When you select a governing law in `ResidualEngine` or `MonitorConfig`, Moju can **automatically** add matching **constitutive** or **scaling** audit rows. Each row uses the standard **`implied_delta`** closure.

Let \(F\) be the catalog model output (`Models.*` from state) and \(\tilde F\) the **implied** value from rearranging the law (`implied_fn` / `implied_value_key`). The stored tensor is **always** the nondimensional discrepancy

\[
R^* = \frac{F - \tilde F}{\varepsilon + |F| + |\tilde F|}
\]

unless a reference field resolves (audit **`implied_delta_ref_key`**, or **`{output_key}_ref`** in merged state/constants), in which case

\[
R^* = \frac{F - \tilde F}{\varepsilon + |\text{ref}|}.
\]

There is **no** raw SI-difference mode. **`Models.*`** still uses your physical state keys; the **monitor residual** is always normalized as above.

**Implied \(\tilde F\)** is computed by **rearranging the law** using the fields referenced by that law’s **`state_map`** (e.g. α\_implied = T_t / T_laplacian for Fourier conduction).

This answers: *“Does the constitutive closure in the catalog agree with what the PDE fields imply locally?”* without requiring **`state_ref`**. It is **not** a claim that the closure matches experiment—only that it matches the **same predicted state** you pass to the law.
These implied residual keys are included in normal category/overall admissibility scoring by default (same as other constitutive/scaling residual keys).

## Configuration

| Mechanism | Behavior |
|-----------|----------|
| `MonitorConfig(law_implied_audits=True)` (default) | Prepend law-linked rows before your `constitutive_audit` / `scaling_audit`. |
| `ResidualEngine(..., law_implied_audits=False)` | Skip prepending (dict-only construction). |
| Expert JSON / `merge_simple_config_with_json_override` | Optional `"law_implied_audits": false` to disable. |

Rows are merged in **`merge_law_implied_audit_specs(laws_spec, enabled=...)`**. Inspect coverage with **`list_laws_with_implied_diagnostics()`** and intentional best-effort gaps with **`law_implied_unsupported_reasons()`**.

## Residual keys and `ref_delta`

Each auto row sets **`residual_basename`** so keys stay unique when multiple laws or manual audits use the same `Models.*` / `Groups.*` name:

- Example: `constitutive/thermal_diffusivity/law_fourier_conduction/implied_delta`

If **`state_ref`** is passed to **`compute_residuals`**, **`ref_delta`** is computed for the same row **unless** the spec sets **`include_ref_delta: false`**. It uses the same nondimensional rules as **`implied_delta`** (symmetric scale, or **`ref_delta_ref_key`** / **`{output_key}_ref`** for the \(|\text{ref}|\) denominator).

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
| `fourier_conduction` | `thermal_diffusivity` | α\_implied = T_t / T_laplacian; model side uses **k**, **rho**, **cp**. |
| `fick_diffusion` | `mass_diffusivity` | D\_implied = φ_t / φ_laplacian; model side uses **fo_mass**, **t**, **L**. |
| `wave_equation` | `wave_speed_from_st` | c\_implied = sqrt(φ_tt / φ_laplacian); model side uses **omega**, **L**, **st_wave**. |
| `advection_diffusion` | `scalar_diffusivity_from_pe` | κ\_implied = U·L·(φ_t + u·∇φ)/φ_laplacian; model side uses **u**, **L**, **pe**. |
| `momentum_navier_stokes` | `dynamic_viscosity_from_re` | μ\_implied from momentum-balance-implied Re and μ = ρ|u|L/Re. |
| `stokes_flow` | `dynamic_viscosity_from_re` | μ\_implied from Stokes-balance-implied Re and μ = ρ|u|L/Re. |
| `burgers_equation` | `dynamic_viscosity_from_re` | μ\_implied from Burgers-balance-implied Re and μ = ρ|u|L/Re. |

## Unsupported laws (best effort)

Laws without an entry add **no** law-linked implied rows. These gaps are intentional and documented in **`law_implied_unsupported_reasons()`** (same module). Typical reasons include:

- no single constitutive/scaling target to rearrange (`mass_incompressible`, `mass_compressible`),
- geometry/material specific inversion not encoded as one catalog closure (`laplace_beltrami`, `hookes_law_residual`),
- model-context-dependent closure choice (`darcy_flow`, `brinkman_extension`),
- laws requiring domain-specific closure/model choices not yet encoded in the constitutive registry.

## Studio and dependency planning

**`build_studio_auto_fragment`** prepends the same dict rows (with **`implied_fn`** attached) and sets **`law_implied_audits: true`**.

**`apps/moju_studio/studio_dependency_planner`** uses **`effective_audit_specs_for_fragment(fragment)`** so required keys (e.g. **k**, **rho**, **cp** for Fourier) appear in preflight even though JSON exports omit **`implied_fn`**.

## Custom extensions

Public helpers such as **`implied_re_momentum_navier_stokes`**, **`implied_mu_from_re_balance`**, etc., can be wrapped in your own **`AuditSpec(..., implied_fn=...)`** if you need a different **`state_map`** or an extra audit not yet in the registry.
