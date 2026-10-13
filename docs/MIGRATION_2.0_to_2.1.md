# Migration: Moju 2.0 → 2.1

2.1 is a **minor** release. Tier cutoffs and Path B / weighted-norm **defaults** from 2.0 are unchanged. The change is limited to **law-linked implied viscosity** (and the catalog model those rows call).

## What broke / what scores change

| Key / API | Change |
|-----------|--------|
| `Models.dynamic_viscosity_from_re(rho, U, L, re)` | Argument renamed/semantics: **`U`** is the declared reference velocity (was local field **`u`** / ``|u|``). |
| Law-linked NS/Stokes audit `state_map` | Uses ``U`` instead of ``u`` for the catalog μ model. |
| `…/dynamic_viscosity_from_re/law_momentum_navier_stokes/implied_delta` | Implied μ from momentum LSQ × ``ρ U L``; scores change vs 2.0. |
| `…/dynamic_viscosity_from_re/law_stokes_flow/implied_delta` | Same for Stokes. |
| `…/kinematic_viscosity_from_re/law_burgers_equation/implied_delta` | Same ν̃ LSQ as before; points may be NaN-masked by the conditioning gate. |

Governing **law** residuals (e.g. `laws/momentum_navier_stokes`) are unaffected. The 1.3.1 / 2.0 golden path with ``law_implied_audits=False`` is unaffected.

## Why

Old behaviour built ``μ̃ = β · ρ · |u|_local · L``. On exact Kovasznay / Taylor–Green data the momentum residual can be ~machine precision while ``μ̃`` still tracks local speed and disagrees with the true material viscosity near walls and stagnation. 2.1 recovers viscosity from the balance and converts with **user-declared** ``U``, ``L``.

## What you should pass

For NS / Stokes implied-μ audits, supply reference scales in `constants` or state:

```python
constants={"rho": 1.0, "U": 1.0, "L": 1.0, "re": 40.0}
```

Optional gate knobs (same places):

| Constant | Default | Meaning |
|----------|---------|---------|
| `implied_term_fraction` | `0.1` | Min fit quality / operator-energy factor to keep a point |
| `implied_min_identifiable_fraction` | `0.05` | Min fraction of points kept to score the row |
| `implied_min_identifiable_count` | `8` | Absolute floor (scaled down on tiny grids) |

When the row is not identifiable, `audit()` reports admissibility level **`Not identifiable`** for that key; see `log[-1]["implied_identifiability"]` and `closure_debug[...]` for `excluded_fraction`, `mu_tilde_global`, `re_tilde_global`.

## Reproduce 2.0 implied-μ (not recommended)

There is no bit-exact switch for the old ``|u|_local`` conversion. Pin ``moju==2.0.0`` if you must match prior implied-μ scores. Prefer updating call sites to pass ``U`` and accept the corrected material-property semantics.
