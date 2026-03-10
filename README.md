# moju
Physics-AI supervision for engineering-grade simulations.

**Moju** is a low code **Physics AI Workbench** and **Digital Twin** framework. It provides the essential utilities for "supervision" required to transform black box neural networks into trustworthy engineering tools.

## Core Pillars

* **Data Ingest (The Bridge):** Automated extraction of unstructured OpenFOAM data into JAX compatible tensors.
* **Dimensionless Scaling (PiRatio):** Built in Buckingham Pi abstraction for scale invariant modeling.
* **The Truth Machine (Auditor):** Real time validation of AI predictions against Navier Stokes residuals and entropy consistency.

## Quick Start

1. Create your environment:
   `conda create -n moju python=3.10`
2. Install the core:
   `pip install moju`
3. Initialize the bridge:
   `moju ingest ./openfoam_project`

## Quick Start

1. Create your environment:
   `conda create -n moju python=3.10`
2. Install the core:
   `pip install moju`
3. Initialize the bridge:
   `moju ingest ./openfoam_project`

## Example imports

```python
# Package and version
import moju
print(moju.__version__)  # "0.1.0"

# PiRatio: dimensionless groups and physical models
from moju.piratio import Groups, Models

# Dimensionless groups (JAX-jitted, work with scalars or arrays)
Re = Groups.re(u=1.0, L=0.1, rho=1000.0, mu=1e-3)   # Reynolds number
Pr = Groups.pr(mu=1e-3, cp=4186.0, k=0.6)           # Prandtl number
Nu = Groups.nu(h=100.0, L=0.1, k=0.6)               # Nusselt number
Ma = Groups.ma(u=100.0, a=343.0)                     # Mach number

# Physical models (differentiable, for loss terms or constraints)
mu = Models.sutherland_mu(T=300.0, mu0=1.8e-5, T0=273.0, S=110.4)  # Air viscosity
rho = Models.ideal_gas_rho(P=101325.0, R=287.0, T=300.0)          # Ideal gas density
q_rad = Models.stefan_boltzmann_flux(epsilon=0.9, T=400.0)        # Radiative flux
```

## Technical Manifesto

Moju is built on the principle that **Physics is the Ground Truth**. We provide the "Glass Box" transparency that practicing engineers need to deploy AI in high stakes environments like data center thermal management.

## License

MIT License. Open for the community. Developed by Abiodun Olaoye, PhD.
