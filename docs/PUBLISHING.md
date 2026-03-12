# Publishing moju to PyPI

## Prerequisites

- PyPI account: https://pypi.org/account/register/
- API token: PyPI → Account Settings → API tokens → Add API token (scope: entire account or project `moju`)

## Build (from repo root on `main`)

```bash
pip install build
python -m build
```

This produces `dist/moju-0.1.0.tar.gz` and `dist/moju-0.1.0-py3-none-any.whl`.

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
2. Re-run `python -m build` and `twine upload dist/*`.
