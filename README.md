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

## Technical Manifesto

Moju is built on the principle that **Physics is the Ground Truth**. We provide the "Glass Box" transparency that practicing engineers need to deploy AI in high stakes environments like data center thermal management.

## License

MIT License. Open for the community. Developed by Abiodun Olaoye, PhD.
