# moju — Physics-AI supervision for engineering-grade simulations

```bash
pip install moju
```

**Moju makes AI models physically admissible and auditable.** It is a lightweight framework for enforcing physics constraints during training, composing dimensionless groups and constitutive models with governing laws, and auditing how well predictions satisfy physics.

*Physics you know, in the AI you train. Dimensionless scaling, constitutive models, and equation residuals in one JAX library.*

---

## Why moju?

Most Physics AI tools focus on adding a physics loss. Moju goes further:

- **Structured physics** — Models, Groups, and Laws as composable building blocks (Reynolds number, viscosity, conservation equations).
- **Automatic residual construction** — `ResidualEngine.compute_residuals(...)` builds law, constitutive, and scaling residuals from your state.
- **Physics admissibility scoring** — `audit(log)` returns per-category and overall scores so you see how well predictions satisfy the physics.
- **Works across PINNs, CFD surrogates, and other state predictors** — Differentiable end-to-end; use in training loops or as a standalone audit toolkit.

---

## The big idea

Moju treats physics as composable building blocks:

```
Predictions (state_pred)
        ↓
Constitutive models (Models.*) + Dimensionless groups (Groups.*)
        ↓
Governing laws (Laws.*)
        ↓
ResidualEngine.compute_residuals(...)  →  residuals
        ↓
loss = build_loss(residuals)     report = audit(engine.log)
```

Instead of hand-wiring `loss = data_loss + physics_loss`, you get residuals from the engine, a physics loss from `build_loss(residuals)`, and an admissibility report from `audit(engine.log)`.

---

## 5-minute example

Run this after `pip install moju`:

```python
import jax.numpy as jnp
from moju.monitor import ResidualEngine, build_loss, audit, MonitorConfig, AuditSpec
from moju.piratio import Models, Groups

mu0 = jnp.array(1.8e-5)
T0 = jnp.array(273.0)
S = jnp.array(110.4)

T = jnp.array(300.0)
mu = Models.sutherland_mu(T=T, mu0=mu0, T0=T0, S=S)

Re = jnp.array(10.0)
Pr = jnp.array(2.0)
Pe = Groups.pe(re=Re, pr=Pr)

cfg = MonitorConfig(
    laws=[{"name": "laplace_equation", "state_map": {"phi_laplacian": "phi_xx"}}],
    constitutive_audit=[
        AuditSpec(
            name="sutherland_mu",
            output_key="mu",
            state_map={"T": "T", "mu0": "mu0", "T0": "T0", "S": "S"},
            predicted_spatial=["T"],
        )
    ],
    scaling_audit=[
        AuditSpec(
            name="pe",
            output_key="Pe",
            state_map={"re": "Re", "pr": "Pr"},
            predicted_spatial=["Re"],
        )
    ],
)

engine = ResidualEngine(config=cfg)

state_pred = {
    "phi_xx": jnp.array(0.0),
    "T": T,
    "mu0": mu0,
    "T0": T0,
    "S": S,
    "mu": mu,
    "d_T_dx": jnp.array(0.0),
    "d_mu_dx": jnp.array(0.0),
    "Re": Re,
    "Pr": Pr,
    "Pe": Pe,
    "d_Re_dx": jnp.array(0.0),
    "d_Pe_dx": jnp.array(0.0),
}

residuals = engine.compute_residuals(state_pred)
loss = build_loss(residuals)
report = audit(engine.log)

print("Physics loss:", float(loss))
print("Overall admissibility:", report["overall_admissibility_score"], report["overall_admissibility_level"])
print("Per category:", report["per_category"])
```

---

## What you get

Moju gives you physics diagnostics, not just a loss. The audit report looks like this:

| Category              | Score |
| --------------------- | ----- |
| Governing laws        | 0.92  |
| Constitutive         | 0.94  |
| Scaling and similarity | 0.96 |

**Overall admissibility score** — geometric mean across categories (e.g. 0.94).  
**Overall admissibility level** — e.g. "High Admissibility".

Report keys: `report["per_category"]` (`laws`, `constitutive`, `scaling`), `report["overall_admissibility_score"]`, `report["overall_admissibility_level"]`. Per-key RMS, R_norm, and admissibility are in `report["per_key"]`.

**Admissibility levels:** (1) each residual key has its own score in `per_key`; (2) each category score in `per_category` is the **geometric mean** of the scores for all keys in that category (governing laws, constitutive models, scaling/groups); (3) the **overall** score is the geometric mean of those category scores that are present. New metrics use the same pipeline—for example optional **π-constant** checks on a scaling audit add a key `scaling/<name>/pi_constant` and are included in the scaling category mean.

---

## Use cases

- **Physics-Informed Neural Networks (PINNs)** — Residuals and loss from governing equations; audit score each step.
- **CFD surrogate models** — Compare to high-fidelity data via `state_ref`; constitutive and scaling audits.
- **Digital twins** — Continuous audit of predictions against physics and data.
- **Scale-invariant modeling** — Dimensionless groups (Re, Pr, Pe, …) and scaling-similarity audits.

---

## Core concepts

| Concept         | Meaning |
| --------------- | ------- |
| **Models**      | Constitutive relationships (e.g. viscosity μ(T), density ρ(P,T)). |
| **Groups**      | Dimensionless quantities (Re, Pr, Pe, Ma, …). |
| **Laws**        | Governing equations (mass, momentum, energy, …); residuals go into `build_loss`. |
| **ResidualEngine** | Builds state from config and optional predictions; runs laws and optional constitutive/scaling audits; produces residuals and a log. |
| **build_loss**  | Builds a scalar physics loss from residuals (laws only). |
| **audit**       | Takes the engine log; returns per-key and per-category admissibility and overall score. |

---

## Installation

```bash
pip install moju
```

Optional extras:

- `pip install moju[ref]` — xarray-based `state_ref` loaders and interpolation.
- `pip install moju[ref_vtk]` — VTK/VTU loaders (meshio).
- `pip install moju[ref_foam]` — OpenFOAM snapshot loaders (meshio).
- `pip install moju[ref_hdf5]` — HDF5 loaders (h5py).
- `pip install moju[report]` — PDF Physics Admissibility Report from `audit(..., export_dir=...)`.

---

## Philosophy

Moju does not define physics. Moju provides a structured way to **enforce** and **audit** it. You bring your governing equations, constitutive models, and dimensionless groups; moju gives you residuals, a differentiable loss, and an admissibility score. JAX-native and fully differentiable so it fits into training loops and high-stakes workflows.

---

## Learn more

**API at a glance** — Two namespaces: **moju.piratio** (Groups, Models, Laws, Operators) and **moju.monitor** (ResidualEngine, build_loss, audit, visualize). Use `MonitorConfig` and `AuditSpec` for typed config; `engine.required_state_keys()` and `engine.required_derivative_keys()` for introspection.

**Examples**

- Quick scaling and laws: `Groups.re(...)`, `Models.ideal_gas_rho(...)`, `Laws.mass_incompressible(u_grad)` — see snippets in the full docs.
- Monitor with laws + scaling audit: `python examples/monitor_chain_spatial_demo.py`, `python examples/monitor_chain_temporal_demo.py`.
- End-to-end NN → residuals → PDF: `python examples/monitor_heat_end_to_end.py`, `python examples/monitor_burgers_end_to_end.py`.
- CFD snapshot → state_ref → audit: `examples/cfd_snapshot_cookbook_heat_1d.py`; reference loaders: `examples/monitor_state_ref_from_vtu_demo.py`, `from_openfoam`, `from_hdf5`.

**Paths** — Path A: pass `(model, params, collocation)` and a `state_builder` to build `state_pred`. Path B: pass `state_pred` directly (e.g. from CFD or finite differences). Constitutive and scaling audits use specs tied to `Models.*` and `Groups.*` (ref_delta, chain_dx, chain_dt). R_norm is scale-based (state-derived by default; override with `audit(log, r_ref=...)`).

**Docs** — [VERSIONING.md](VERSIONING.md). Online docs: overview, Groups, Models, Laws, Operators.

---

## License

MIT License. Developed by Ifimo Lab, a division of Ifimo Analytics.
