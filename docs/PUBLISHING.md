# Publishing moju to PyPI

## Prerequisites

- PyPI account: https://pypi.org/account/register/
- API token: PyPI → Account Settings → API tokens → Add API token (scope: entire account or project `moju`)

## Build (from repo root on `main`)

```bash
pip install build
python -m build
```

This produces `dist/moju-<version>.tar.gz` and `dist/moju-<version>-py3-none-any.whl` (e.g. `moju-0.4.2`).

## Upload

**Test first (optional):** https://test.pypi.org/account/register/

```bash
pip install twine
twine upload --repository testpypi dist/*
# When prompted: username = __token__, password = pypi-... (Test PyPI token)
# Then: pip install -i https://test.pypi.org/simple/ moju
```

**Production:**

```bash
twine upload dist/*
# username = __token__, password = your PyPI API token
```

After a successful upload, anyone can run `pip install moju`.

## Version bump for future releases

1. Update **both** to the same version string (e.g. `0.5.0`): `version` in [pyproject.toml](../pyproject.toml) and `__version__` in [moju/__init__.py](../moju/__init__.py) so the package and runtime version stay in sync.
2. Add or update the release section in [CHANGELOG.md](../CHANGELOG.md) (e.g. `## [0.5.0] - YYYY-MM-DD`).
3. Re-run `python -m build` and upload (e.g. `twine upload dist/moju-<version>*` or use the GitHub Actions workflow). Use tag `v` + version (e.g. `v0.5.0`) when creating the Git tag and GitHub Release.

The GitHub Actions workflow (on release or manual run) skips the PyPI upload if the built version is already published, so retroactive or duplicate releases do not fail.

## GitHub release and tags

After merging the release branch into `main` and pushing:

1. Create an annotated tag for the version: `git tag -a v0.4.2 -m "Release 0.4.2"` (on the commit that has the version bump).
2. Push the tag: `git push origin v0.4.2`.
3. Create a GitHub Release from the tag (Releases → Draft a new release → choose tag `v0.4.2`, set as latest, publish). This triggers the Publish to PyPI workflow if the version is not already on PyPI.
