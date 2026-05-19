# Contributing to mind-nerve

Thank you for your interest in mind-nerve. This document describes the
build/test/PR flow for changes to the public repository at
<https://github.com/star-ga/mind-nerve>.

## Scope

mind-nerve is the public, Apache-2.0 portion of STARGA's intent-classification
preselector. The wheel additionally bundles `libmindnerve.so`, a separately
licensed native runtime component (see [`LICENSE.md`](LICENSE.md)). Pull
requests are welcome against the public surface; the bundled binary and any
private STARGA training corpora are out of scope for community PRs.

## Code of conduct

Be civil, technical, and specific. Discrimination, harassment, and personal
attacks are not tolerated. Maintainers may close, lock, or revert
contributions that violate this policy.

## Reporting bugs and proposing features

- **Bugs:** open a [bug report](.github/ISSUE_TEMPLATE/bug_report.yml).
  Include version (`mind-nerve --version`), platform, reproduction steps,
  observed vs expected behaviour, and full tracebacks.
- **Features:** open a [feature request](.github/ISSUE_TEMPLATE/feature_request.yml).
  Describe the use case, the proposed surface, and any alternatives you
  considered.
- **Security vulnerabilities:** do **not** open a public issue. See
  [`SECURITY.md`](SECURITY.md) for the disclosure channel.

## Development setup

mind-nerve targets Python ≥ 3.10 on Linux and macOS. The dev workflow is:

```bash
git clone https://github.com/star-ga/mind-nerve.git
cd mind-nerve

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev,mcp]"
```

This installs `pytest`, `ruff`, and `build` as dev dependencies and the
`mcp` Python package as the optional MCP extra.

## Running tests

The Python unit tests live under `tests/python/` and are the gate every
PR must clear:

```bash
pytest tests/python -q
```

The full lint + test loop a PR is expected to pass:

```bash
ruff check python/mind_nerve
ruff format --check python/mind_nerve
pytest tests/python -q
```

If you change the public surface, please add a regression test under
`tests/python/` that fails before your fix and passes after.

## Style

- **PEP 8** with `ruff` (config in [`pyproject.toml`](pyproject.toml)).
- Type-annotate every function signature on the public surface.
- Prefer many small files over few large files. Functions under ~50 lines
  and modules under ~800 lines is the default expectation.
- No mutation of caller-owned data structures. Return new objects.
- No `print()` in library code; raise or return structured results.
- Public functions, classes, and CLI flags must have docstrings or
  `argparse` `help=` strings.

## Commit messages

Conventional Commits format:

```
<type>(<scope>): <summary>

<body — what changed and why, not how>
```

Types: `feat`, `fix`, `docs`, `test`, `chore`, `refactor`, `perf`, `ci`,
`build`.

All commits must be authored by `STARGA Inc <noreply@star.ga>`. Do not add
`Co-Authored-By` lines or AI-tool attributions — STARGA Inc. is the single
author of every commit in this repository.

## Pull request flow

1. Fork the repository and create a topic branch from `main`.
2. Make your change in focused commits.
3. Run the full lint + test loop locally:
   ```bash
   ruff check python/mind_nerve
   ruff format --check python/mind_nerve
   pytest tests/python -q
   ```
4. Open a PR against `main`. Fill in the
   [PR template](.github/pull_request_template.md): describe what changed,
   how it was tested, and which audit finding (if any) it maps to.
5. CI must be green before review. The maintainer may request changes,
   tests, or a rebase before merge. Squash-merge is the default.

## Architecture changes

The frozen design is **drop-the-decoder + sliding-window encoder
(window 256, stride 192) + direct scoring head**, documented in
[`spec/architecture.md`](spec/architecture.md). Material deviations from
that spec require:

1. An RFC under [`RFCs/`](RFCs/) describing motivation, proposed change,
   and migration path.
2. Maintainer sign-off on the RFC.
3. A coordinated update to `spec/`, README, `ROADMAP.md`, and tests.

Small bug fixes, performance improvements that do not change the surface,
and integration shims do **not** require an RFC.

## Releases

Maintainers cut releases. Contributors should not bump
`python/mind_nerve/__version__` in a PR — that lands in the release commit.

## Licensing of contributions

By submitting a contribution, you license it under the Apache-2.0 license
that governs this repository. See [`LICENSE`](LICENSE) and
[`LICENSE.md`](LICENSE.md) for the full terms. You confirm you have the
right to license the contribution.
