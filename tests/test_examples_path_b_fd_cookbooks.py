"""Smoke tests: Path B FD cookbooks run without error."""

from pathlib import Path

import runpy


_ROOT = Path(__file__).resolve().parents[1]


def test_cookbook_path_b_fd_audit_pe():
    runpy.run_path(
        str(_ROOT / "examples" / "cookbook_path_b_fd_audit_pe.py"),
        run_name="__main__",
    )


def test_cookbook_path_b_fd_law_laplace():
    runpy.run_path(
        str(_ROOT / "examples" / "cookbook_path_b_fd_law_laplace.py"),
        run_name="__main__",
    )
