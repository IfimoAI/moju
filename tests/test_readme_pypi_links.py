"""README links must be absolute so they work on PyPI (long description)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

# Repo-root relative prefixes that break on https://pypi.org/project/moju/
_FORBIDDEN_PREFIXES = ("examples/", "docs/", "scripts/", "apps/")


def test_readme_has_no_pypi_breaking_relative_links():
    text = README.read_text()
    targets = re.findall(r"\]\(([^)]+)\)", text)
    bad = []
    for target in targets:
        target = target.strip()
        if target.startswith("#") or target.startswith("http"):
            continue
        if target in ("VERSIONING.md", "CHANGELOG.md") or any(
            target.startswith(p) for p in _FORBIDDEN_PREFIXES
        ):
            bad.append(target)
    assert not bad, f"Use absolute GitHub URLs in README.md: {bad}"
