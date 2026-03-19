#!/usr/bin/env python3
"""
Generate a sample Physical Admissibility Report PDF for review.

Run from repo root after: pip install moju[report] or pip install reportlab
  python scripts/generate_sample_audit_pdf.py

Output: examples/sample_physical_admissibility_report.pdf
"""

import os
import sys

# Add repo root so moju.monitor.report is importable
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from moju.monitor.report import write_audit_pdf


def _sample_report() -> dict:
    """Build a sample report dict matching the structure returned by audit()."""
    return {
        "overall_admissibility_score": 0.92,
        "overall_admissibility_level": "High Admissibility",
        "per_key": {
            "laws/mass_incompressible": {
                "rms": 0.02,
                "r_norm": 0.08,
                "admissibility_score": 0.92,
                "admissibility_level": "High Admissibility",
            },
            "laws/navier_stokes": {
                "rms": 0.05,
                "r_norm": 0.14,
                "admissibility_score": 0.88,
                "admissibility_level": "Moderate Admissibility",
            },
            "scaling/re/chain_dx": {
                "rms": 0.01,
                "r_norm": 0.05,
                "admissibility_score": 0.95,
                "admissibility_level": "High Admissibility",
            },
            "scaling/pe/chain_dx": {
                "rms": 0.12,
                "r_norm": 0.43,
                "admissibility_score": 0.70,
                "admissibility_level": "Moderate Admissibility",
            },
            "scaling/pr/chain_dx": {
                "rms": 0.25,
                "r_norm": 0.83,
                "admissibility_score": 0.55,
                "admissibility_level": "Low Admissibility",
            },
            "constitutive/sutherland_mu/chain_dt": {
                "rms": 0.02,
                "r_norm": 0.09,
                "admissibility_score": 0.92,
                "admissibility_level": "High Admissibility",
            },
            "constitutive/ideal_gas_rho/ref_delta": {
                "rms": 0.01,
                "r_norm": 0.02,
                "admissibility_score": 0.98,
                "admissibility_level": "High Admissibility",
            },
            "data/u": {
                "rms": 0.08,
                "r_norm": 0.25,
                "admissibility_score": 0.80,
                "admissibility_level": "Moderate Admissibility",
            },
            "laws/heat_diffusion": {
                "rms": 0.45,
                "r_norm": 1.86,
                "admissibility_score": 0.35,
                "admissibility_level": "Non-Admissible",
            },
        },
    }


def main() -> None:
    examples_dir = os.path.join(_repo_root, "examples")
    os.makedirs(examples_dir, exist_ok=True)
    out_path = os.path.join(examples_dir, "sample_physical_admissibility_report.pdf")
    report = _sample_report()
    write_audit_pdf(
        report,
        out_path,
        model_name="NeuralTurb-Flowv3",
        model_id="223355592",
    )
    print(f"Sample PDF written to: {out_path}")


if __name__ == "__main__":
    main()
