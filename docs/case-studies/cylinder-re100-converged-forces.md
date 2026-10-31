# Case study: converged forces, unconverged near-wall physics

*2D flow past a circular cylinder at Re = 100, simulated in Nektar++. Audited with Moju.*

![Vortex shedding (vorticity) over the actual Nektar++ mesh](img/wake-over-mesh.jpg)

## TL;DR

- We ran the same mesh at polynomial order 4 and 6. Lift, drag and shedding frequency agree within 0.01%. By the usual convergence check, the two runs are the same.
- Moju's implied-viscosity check says otherwise. Near the wall, the order-4 field behaves as if the fluid's viscosity were wrong: 2.8% off at the 95th percentile and 7.5% at the worst point, against 0.08% and 0.27% at order 6.
- Moju scores the order-4 field 26.9% (Non-Admissible) and the order-6 field 92.6% (High Admissibility).
- A convergence study only tells you whether two runs agree with each other. A physics audit tells you whether each one satisfies the physics. That matters when simulation fields become training data for neural surrogates: a model trained on near-wall errors learns them too, right where the wall shear and pressure that make the forces come from.

## Setup

| Item | Value |
|---|---|
| Solver | Nektar++ 5.10.0, `IncNavierStokesSolver` (spectral/hp elements) |
| Flow | 2D, incompressible, Re = 100 (D = 1, U = 1, ν = 0.01), unconfined, settled periodic shedding |
| Mesh | 6,883 quadrilaterals, identical for both runs |
| Resolutions | polynomial order 4 (`NUMMODES=5`) and 6 (`NUMMODES=7`), i.e. p-refinement on the same mesh |
| Domain | 30D upstream, 25D downstream, 30D to each side |

![The Nektar++ mesh near the cylinder and the full computational domain](img/mesh-and-domain.jpg)

## What converged

Statistics over the settled window:

| Quantity | Order 4 | Order 6 | Change |
|---|---:|---:|---:|
| Strouhal number (zero-crossing) | 0.16524 | 0.16523 | 0.0042% |
| Mean drag coefficient | 1.33446 | 1.33451 | 0.0042% |
| RMS lift coefficient | 0.22940 | 0.22938 | 0.0090% |

![Lift over four shedding cycles at order 4 and 6](img/forces-p4-vs-p6.jpg)

## What the audit checks

Moju turns each predicted state into residuals of the governing laws you declare and into **law-linked constitutive checks** ([docs](../law_implied_audits.md)). For incompressible Navier–Stokes, the constitutive check asks: *what viscosity does the local momentum balance imply at this point, and does it match the viscosity you declared?*

- At each sample point, Moju solves the momentum balance for a local viscosity ν̃ (a least-squares projection onto ∇²u) and computes the fractional mismatch δ = (ν − ν̃)/ν.
- It only uses points where the viscous term is large enough for ν to be identifiable. Elsewhere, inertia or pressure dominate and the implied value is ill-conditioned, so those points are reported as not identifiable, not scored. In this case all 380 near-wall samples are identifiable at both orders.
- The score uses the 95th percentile of |δ| with noise normalization (the opt-in `implied_metric="percentile"` setting, added in Moju 2.2.0). The worst point is reported alongside it but does not set the score.

The fields passed to Moju are the solver's own output: velocity and pressure at five closely spaced snapshots (for ∂u/∂t), and spatial derivatives from Nektar++'s `FieldConvert`.

## Results

| | Order 4 | Order 6 |
|---|---:|---:|
| Implied-viscosity mismatch, 95th percentile | 2.8% | 0.08% |
| Implied-viscosity mismatch, worst point | 7.5% | 0.27% |
| Moju admissibility (implied viscosity) | **26.9%, Non-Admissible** | **92.6%, High** |

![Near-wall implied-viscosity mismatch at order 4 and 6, shared log scale](img/near-wall-implied-viscosity.jpg)

![Moju admissibility scores and tiers](img/moju-admissibility.jpg)

Two things are worth noting:

- **The governing laws look fine at both orders.** Near-wall continuity and momentum residuals score High (about 0.95 to 1.00). The problem only shows up when you ask whether the field's viscous balance matches the stated fluid.
- **A global summary hides it.** The median implied viscosity over the identifiable near-wall points is 0.010000 at both orders, essentially the declared 0.01. The error is local, and a central summary washes it out.

## Why we care

We are generating simulation data intended for training specialized neural surrogates. Integrated quantities such as forces and frequencies are the usual convergence criteria, and they converge long before the near-wall derivatives do. A surrogate learns the whole field, so the data has to be right where the physics is hardest, not only where the integrals are.

## Limitations

- This is one benchmark: 2D, laminar, Re = 100. Nothing here is a claim about 3D, turbulent or high-Reynolds-number flow.
- A tier means consistency with the stated physics on the sampled points. It is not a certification and does not replace validation against experiments.
- Scores were computed with Moju 2.4.2 with strict evaluation coverage.
- The simulation fields themselves are not published in this repository.

## Try it on your own fields

```bash
pip install moju
```

- Law-linked implied audits: [`docs/law_implied_audits.md`](../law_implied_audits.md)
- Auditing CFD snapshots: [`examples/cfd_snapshot_cookbook_heat_1d.py`](../../examples/cfd_snapshot_cookbook_heat_1d.py)
- Navier–Stokes term diagnostics: [`examples/cookbook_taylor_green_diagnostics.py`](../../examples/cookbook_taylor_green_diagnostics.py)
- Getting derivatives right: [`docs/getting_derivatives_right.md`](../getting_derivatives_right.md)

Simulations and case study by Ifimo Lab. Questions and critique are welcome in [Issues](https://github.com/IfimoAI/moju/issues).

## Correction (2026-10-30)

An earlier version of this page reported values affected by a sampling-geometry error in our post-processing: the corner order passed to Nektar++'s `FieldConvert` plane interpolation was wrong, so the near-wall sample coordinates did not match the interpolated field. After the fix, the order-4 mismatch is 2.8% (95th percentile) and 7.5% (worst point) instead of 4.3% and 8.0%; order 6 is 0.08% and 0.27% instead of 0.10% and 0.26%; the scores are 26.9% and 92.6% instead of 19.5% and 90.6%, and all 380 near-wall samples are identifiable instead of 156. The forces and the conclusion are unchanged.
