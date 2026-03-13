# Publishing moju to PyPI

## Prerequisites

- PyPI account: https://pypi.org/account/register/
- API token: PyPI → Account Settings → API tokens → Add API token (scope: entire account or project `moju`)

## Build (from repo root on `main`)

```bash
pip install build
python -m build
```

This produces `dist/moju-<version>.tar.gz` and `dist/moju-<version>-py3-none-any.whl` (e.g. `moju-0.1.2`).

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

1. Update `version` in [pyproject.toml](../pyproject.toml).
2. Set the same value for `__version__` in [moju/__init__.py](../moju/__init__.py) so the package and runtime version stay in sync.
3. Re-run `python -m build` and upload (e.g. `twine upload dist/moju-<version>*` or use the GitHub Actions workflow).
