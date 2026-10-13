# Migration: Moju 2.0 → 2.1

2.1 is a **minor** release. Tier cutoffs and Path B / weighted-norm **defaults** from 2.0 are unchanged. The change is limited to **law-linked implied viscosity** (and the catalog model those rows call).

## What changed

| Key / API | Change |
|-----------|--------|
| `Models.dynamic_viscosity_from_re(rho, U, L, re)` | Reference velocity is **`U`** (material-property semantics). Old keyword **`u=`** still works as a **deprecated alias** (warns; removal in **3.0**). |
| Law-linked NS/Stokes audit `state_map` | Catalog args are ``rho``, ``U``, ``L``, ``re`` (not the velocity field). |
| `…/dynamic_viscosity_from_re/law_momentum_navier_stokes/implied_delta` | Implied μ from momentum LSQ × ``ρ U L``; scores change vs 2.0 local-speed bias. |
| `…/dynamic_viscosity_from_re/law_stokes_flow/implied_delta` | Same for Stokes. |
| `…/kinematic_viscosity_from_re/law_burgers_equation/implied_delta` | Same ν̃ LSQ; points may be NaN-masked by the conditioning gate. |

### Corrected values (vs 2.0)

| Quantity | 2.0 (incorrect) | 2.1 |
|----------|-----------------|-----|
| Implied μ (NS/Stokes) | ``β · ρ · \|u\|_local · L`` | ``β · ρ · U · L`` with declared ``U`` |
| Catalog μ | ``ρ \|u\|_local L / Re`` | ``ρ U L / Re`` |
| Implied Re | from local speed | ``ρ U L / μ̃`` or ``U L / ν̃`` |

Governing **law** residuals are unaffected. Paths with ``law_implied_audits=False`` are unaffected. Fourier / thermal-diffusivity implied audits (paper slab-cooling) are unaffected.

## Declaring ``U`` and ``L``

```python
constants = {"rho": 1.0, "U": 1.0, "L": 1.0}  # same scales as Re in the law
engine = ResidualEngine(
    laws=[{"name": "momentum_navier_stokes", "state_map": {...}}],
    constants=constants,
    law_implied_audits=True,
)
```

Deprecated temporary call (warns):

```python
Models.dynamic_viscosity_from_re(rho, L=L, re=re, u=U_ref)  # use U=U_ref instead
```

## Conditioning gate and “Not identifiable”

A collocation point is kept only if **both** hold (``implied_term_fraction``, default **0.1**):

1. **Term fraction:** ``|β ∇²u| / max_i |term_i| ≥ threshold``  
   (`term_i` ∈ {``u_t``, advection, ``∇p``, optional body force, viscous term}).
2. **Fit quality:** ``|β ∇²u| / (|β ∇²u| + |rhs − β ∇²u|) ≥ threshold``.

Excluded points are NaN in ``δ``; ``closure_debug`` / ``log[-1]["implied_identifiability"]`` report ``excluded_fraction``. If too few points remain (``implied_min_identifiable_fraction`` default 0.05, ``implied_min_identifiable_count`` default 8), admissibility for that key is **`Not identifiable`**. That key is omitted from constitutive/overall rollup (it does not force Non-Admissible).

| Constant | Default | Meaning |
|----------|---------|---------|
| `implied_term_fraction` | `0.1` | Threshold for term fraction **and** fit quality |
| `implied_min_identifiable_fraction` | `0.05` | Min fraction of points kept to score the row |
| `implied_min_identifiable_count` | `8` | Absolute floor (scaled down on tiny grids) |

## Reproduce 2.0 implied-μ (not recommended)

Pin ``moju==2.0.0`` if you must match prior local-speed implied-μ scores. Prefer ``U`` and the corrected material-property semantics.
