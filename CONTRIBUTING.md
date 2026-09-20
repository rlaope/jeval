# Contributing to jeval

Thanks for considering a contribution. This document is short on purpose: the rules that matter
are the ones that keep the measurements trustworthy.

## Development environment

Python 3.10 or newer, and [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/rlaope/jeval
cd jeval
uv sync --all-groups
```

Run the full local gate before opening a pull request:

```sh
uv run ruff format --check .
uv run ruff check .
uv run mypy jeval
uv run pytest
uv run jeval demo --out-dir /tmp/jeval-demo
```

`pre-commit` is configured with the same checks:

```sh
uv run pre-commit install
```

## The one rule that matters

**The statistics are the product.** A calibration tool that is itself miscalibrated has no
reason to exist, so changes to `jeval/calibration.py`, `jeval/evaluate.py`, or
`jeval/synth.py` must be driven by a synthetic test:

1. Use `jeval/synth.py` to generate a log with a *known* miscalibration.
2. Show the test failing without your change.
3. Make the smallest change that passes it.
4. Do not widen tolerances to make a change pass. Widening a tolerance asserts that jeval no
   longer detects something it used to detect.

## What belongs here

Good contributions: a metric implemented correctly with intervals, a bug in binning, a report
rendering fix, a clearer diagnostic line, better synthetic coverage, doc corrections.

Out of scope, and closed on sight: model gateways and routers, provider adapters (before v0.3),
hosting, dashboards, accounts, databases, plotting libraries, vendor-specific behavior,
localization, and any change that folds `score` records into binary accuracy.

## Commit and pull request style

- [Conventional Commits](https://www.conventionalcommits.org/), English, imperative mood:
  `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`.
- One logical change per commit; a refactor and a behavior change do not share a commit.
- Pull requests state what changed, how it was verified, and which acceptance criterion it
  addresses. Paste the command you ran and its result.
- Keep the diff focused. Unrelated cleanups belong in their own pull request.

## Language

All artifacts are English: code, identifiers, comments, docstrings, commit messages, issues,
pull request text, tests, and documentation. Do not add translations or i18n.

## Reporting bugs

Use the issue template. A calibration bug report is far more actionable with a minimal synthetic
reproduction — a small JSONL snippet plus the command and the numbers you expected.

## Security

Do not open a public issue for a vulnerability. See `SECURITY.md`.

## License

By contributing you agree that your contribution is licensed under Apache-2.0.
