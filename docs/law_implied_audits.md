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

## Direct implied-term mode (law-linked coefficient audits)

For supported law-linked rows, Moju now recovers a direct **implied constitutive term** from the governing-law fields and compares it to the catalog model output. For example, Fourier conduction compares:

\[
\alpha_{\text{model}} = \mathrm{Models.thermal\_diffusivity}(k,\rho,c_p)
\quad\text{vs}\quad
\alpha_{\text{implied}} = \frac{T_t}{T_{\text{laplacian}}}.
\]

The stored tensor is the standard subtract-mode nondimensional discrepancy:

\[
R^* = \frac{F - \tilde F}{\varepsilon + |F| + |\tilde F|}
\]

where **`a`** is the model/catalog term and **`ε = 1e-30`** guards against division by zero. This means the closure debug sidecar contains:

- **`pred`**: the model constitutive term \(F\);
- **`implied`**: the law-implied constitutive term \(\tilde F\);
- **`mode`**: **`"subtract"`**.

Divisions are masked: points with ill-conditioned denominators are stored as **`NaN`** instead of huge finite implied values. Downstream reductions use nan-aware statistics, and all-invalid rows become non-finite diagnostics.

There is **no** raw SI-difference mode. **`Models.*`** still uses your physical state keys; the **monitor residual** is always normalized as above.

This answers: *“Does the constitutive closure in the catalog agree with what the PDE fields imply locally?”* without requiring **`state_ref`**. It is **not** a claim that the closure matches experiment—only that it matches the **same predicted state** you pass to the law.
These implied residual keys are included in normal category/overall admissibility scoring by default (same as other constitutive residual keys).

### How `visualize()` renders constitutive results

When `visualize()` is called after training or eval, the constitutive row of the monitor dashboard shows two sub-panels:

- **Constitutive Divergence** (heatmap or line): the normalised delta `(pred − implied) / (|pred| + ε)` across all spatial collocation points. The colour scale is diverging and centred on zero; values near zero indicate the model is consistent with the law-implied term at that location.
- **Constitutive Consistency** (line plot): the model (`pred`) and law-implied (`implied`) constitutive values as separate lines for the last time slice (transient data) or the worst-divergence row (2D data). Spatially varying acceptability bands centred on the model curve show ±1 % (green, acceptable), ±1–5 % (amber, warning), and ±5–6 % (red, alarm) tolerance zones based on the local model magnitude. Faint dotted tier boundary lines at the ±1 % and ±5 % edges carry hover labels (`+1% Δ`, `−5% Δ`, etc.).

In the README's minimal 1D slab-cooling example, the user supplies a Path B `state_pred`
with `T`, `T_t`, `T_laplacian`, coordinates, and material properties. Because those
derivative keys are already present, no finite-difference inference is needed. Selecting
`fourier_conduction` through `build_minimal_residual_engine(...)` is enough for Moju to
prepend the `thermal_diffusivity/law_fourier_conduction` implied audit.

**Training vs eval:** **`implied_delta`** law-linked rows run in both **`run_mode="training"`** (default) and **`run_mode="eval"`**. **`ref_delta`** on those rows (and separate **`data/`** pred−ref) runs only when you call **`compute_residuals(..., run_mode="eval", state_ref=...)`**. See [monitor_training_vs_eval.md](monitor_training_vs_eval.md).

## Configuration

| Mechanism | Behavior |
|-----------|----------|
| `MonitorConfig(law_implied_audits=True)` (default) | Prepend law-linked rows before your `constitutive_audit`. |
| `ResidualEngine(..., law_implied_audits=False)` | Skip prepending (dict-only construction). |
| Expert JSON / `merge_simple_config_with_json_override` | Optional `"law_implied_audits": false` to disable. |

Rows are merged in **`merge_law_implied_audit_specs(laws_spec, enabled=...)`**. Inspect coverage with **`list_laws_with_implied_diagnostics()`** and intentional best-effort gaps with **`law_implied_unsupported_reasons()`**.

## Auto materialization (`derived_state_chain`)

Law-linked rows only define **which** constitutive audits exist and how **`implied_fn`** recovers the law-implied term. Dimensionless **groups** may still need the same quantity as a **state key** (e.g. **`alpha`** for **`Groups.fo`**).

When a constitutive audit’s **`name`** is registered in **`moju.monitor.model_derived_registry.MODEL_DERIVED_REGISTRY`** and its **`output_key`** appears in some **`groups`** **`state_map` value**, :class:`ResidualEngine` **appends** a matching JSON DSL step to **`derived_state_chain`** at engine construction (same rules as Moju Studio’s **`enrich_fragment_from_model_audits`**). Supported catalog bridges today: **`thermal_diffusivity`** (\(\alpha = k/(\rho c_p)\)), **`mass_diffusivity`** (\(D = \mathrm{Fo}_{\text{mass}}\,L^2/t\)), **`wave_speed_from_st`** (\(c = \omega L/\mathrm{St}\)).

**`user_fns`:** Before **`apply_derived_state_chain`**, the engine runs **`user_fns`** for any **reference keys** used in those expressions (e.g. materialize **`k`** from **`T`** via **`user_fns['k']`**) so nonlinear conductivities work without duplicating **`alpha`** in the NPZ.

**`pred` for implied audits** remains **only** the catalog **`Models.*`** evaluation on the audit’s **`state_map`**; auto materialization does not substitute a different **`pred`**.

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
| `fourier_conduction` | `thermal_diffusivity` | Implied \(\alpha = T_t / T_{\text{laplacian}}\); model uses **k**, **rho**, **cp**. |
| `fick_diffusion` | `mass_diffusivity` | Implied \(D = \phi_t / \phi_{\text{laplacian}}\); model uses **fo_mass**, **t**, **L**. |
| `wave_equation` | `wave_speed_from_st` | Implied \(c = \sqrt{\phi_{tt}/\phi_{\text{laplacian}}}\); negative or ill-conditioned ratios become **NaN**. |
| `advection_diffusion` | `scalar_diffusivity_from_pe` | Implied \(\kappa = (\phi_t + \mathbf u\!\cdot\!\nabla\phi)|\mathbf u|L / \phi_{\text{laplacian}}\). |
| `momentum_navier_stokes` | `dynamic_viscosity_from_re` | Implied **μ** from least-squares projection of \(u_t + u\cdot\nabla u + \nabla p\) onto \(\nabla^2u\). |
| `stokes_flow` | `dynamic_viscosity_from_re` | Implied **μ** from least-squares projection of \(\nabla p\) onto \(\nabla^2u\). |
| `burgers_equation` | `dynamic_viscosity_from_re` | Implied **μ** from projected kinematic viscosity in \(u_t + u\cdot\nabla u = \nu\nabla^2u\). |
| `momentum_incompressible_newtonian_laplacian` | `turbulent_viscous_acceleration_*` | Three auto rows: k-ω, k-ε, and Smagorinsky; **subtract** mode **`pred − implied_fn`**. |
| `momentum_compressible_newtonian_laplacian` | `turbulent_viscous_acceleration_compressible_*` | Three auto rows: compressible k-ω, k-ε, and Smagorinsky; **subtract** mode. |

## Unsupported laws (best effort)

Laws without an entry add **no** law-linked implied rows. These gaps are intentional and documented in **`law_implied_unsupported_reasons()`** (same module). Typical reasons include:

- no single constitutive/scaling target to rearrange (`mass_incompressible`, `mass_compressible`),
- geometry/material specific inversion not encoded as one catalog closure (`laplace_beltrami`, `hookes_law_residual`),
- model-context-dependent closure choice (`darcy_flow`, `brinkman_extension`),
- laws requiring domain-specific closure/model choices not yet encoded in the constitutive registry.

If you still want implied constitutive checks for unsupported laws, add explicit rows under `constitutive_audit` (for example with `implied_fn` or a custom `implied_balance_fn` for advanced balance-style diagnostics).

## Studio and dependency planning

**`build_studio_auto_fragment`** prepends the same dict rows (with **`implied_fn`** attached) and sets **`law_implied_audits: true`**.

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

Expert users can attach **`implied_fn`** on manual **`constitutive_audit`** dict rows (same subtract-style shape as law-linked rows). Advanced users may still attach **`implied_balance_fn`** directly to `ResidualEngine` specs for custom diagnostics, but law-linked built-ins use direct implied terms. Use **`residual_basename`** for unique log keys when reusing a `Models.*` name.
