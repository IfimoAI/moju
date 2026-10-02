# Contributing to Moju

Thank you for your interest in Moju. Bug reports, documentation fixes, new examples, and code changes are welcome through [GitHub issues](https://github.com/IfimoAI/moju/issues) and pull requests.

## License of contributions

Moju is released under the [MIT License](LICENSE). By submitting a contribution, you agree that it is licensed under the same MIT License.

## Developer Certificate of Origin (DCO)

Every commit in a pull request must carry a `Signed-off-by` line. The sign-off certifies that you wrote the change or otherwise have the right to submit it under the project's license, as described in the [Developer Certificate of Origin 1.1](https://developercertificate.org/).

Add the sign-off with `-s`:

```bash
git commit -s -m "Fix Path B spacing check for 2D grids"
```

This appends a line using your configured `user.name` and `user.email`:

```text
Signed-off-by: Your Name <you@example.com>
```

To sign off commits you already made on your branch:

```bash
git rebase --signoff main
```

If you use AI coding tools, you remain responsible for reviewing the change and for the sign-off.

## Development setup

```bash
git clone https://github.com/IfimoAI/moju.git
cd moju
pip install -e ".[dev]"        # add ,torch or ,io for those extras
pytest
```

## Pull requests

- Keep each pull request focused on one change.
- Add or update tests under `tests/` for behavior changes.
- Update `README.md`, `docs/`, and the `Unreleased` section of `CHANGELOG.md` when user-facing behavior changes.
- See [`VERSIONING.md`](VERSIONING.md) for the versioning policy.
