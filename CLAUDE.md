# CLAUDE.md

Guidance for coding agents working in this repository. Read this before changing code.

## What this project is

jeval is a provider-neutral CLI that measures the calibration of probabilistic AI classifiers
and, from a cost matrix, derives the confidence threshold at which a decision should be
automated or handed to a human. Everything is driven by labeled decision records stored as
JSONL on the filesystem; `jeval report` emits a single dependency-free HTML file, and
`jeval drift` (milestone M2) is meant to make model change a CI failure instead of a
three-week-late discovery.

## Non-goals — do not build these

Scope creep kills this project faster than bugs do. The following are explicitly out of scope,
and a pull request adding them will be closed:

- Model gateways, routers, proxies, or provider adapters (adapters: not before v0.3)
- Model hosting, prompt optimization, fine-tuning
- SaaS dashboards, servers, background daemons, account systems
- Databases, ORMs, migrations — storage is JSONL files
- Plotting libraries: the report is hand-written inline SVG
- Any chart that carries meaning in colour alone, or any chart without `<title>`/`<desc>`
- A report that writes configuration (`thresholds.yaml` belongs to `jeval threshold`), fetches
  anything at runtime, or posts to a PR (jeval never holds a token)
- Vendor-specific behavior of any kind: jeval must work for anything that returns a probability
- Non-English artifacts: no i18n, no translated docs, no localized CLI strings
- `score`-type records folded into binary accuracy (they are excluded and counted)

## Architecture map

| Path | Responsibility |
| --- | --- |
| `jeval/schema.py` | `DecisionRecord` and the per-`question_type` normalization rules |
| `jeval/store.py` | JSONL read/write; the only module that knows about `.jeval/` |
| `jeval/config.py` | `config.yaml` defaults and the ingest field mapping |
| `jeval/ingest.py` | raw JSONL/CSV rows to decision records, one record per question |
| `jeval/calibration.py` | binning, ECE/MCE/Brier, Wilson and bootstrap intervals — pure functions |
| `jeval/evaluate.py` | record set to report-ready structures; gold/silver separation |
| `jeval/synth.py` | synthetic logs with known miscalibration; shared by tests and `jeval demo` |
| `jeval/costs.py` | cost matrix, threshold sweep, the bootstrap interval on the threshold, impact table |
| `jeval/drift.py` | slices by model version or period, model-change detection, `--fail-on` checks, CI text block |
| `jeval/baseline.py` | baseline snapshots: measurements only, never records |
| `jeval/report/model.py` | the frozen data contract every chart and the template render from |
| `jeval/report/svg.py` | pure SVG primitives: scales, ticks, axes, legends, error bars |
| `jeval/report/charts/` | one module per chart: reliability, cost, segments, drift |
| `jeval/report/verdict.py` | verdict branching and the markdown summary |
| `jeval/report/assets.py` | inline CSS and JavaScript |
| `jeval/report/template.py` | section assembly in the fixed reading order |
| `jeval/cli.py` | the Typer command surface; thin, no statistics logic |

Statistics belong in `calibration.py`. If a metric needs to be computed anywhere else, it
needs to move there instead.

## Development commands

```sh
uv sync --all-groups
uv run pytest
uv run ruff format .
uv run ruff check .
uv run mypy jeval
uv run jeval demo --out-dir /tmp/jeval-demo
```

## Code rules

- **English only** in code, identifiers, comments, docstrings, error messages, CLI help, docs,
  and commit messages.
- New runtime dependencies require discussion before the change, not after. The runtime set is
  `numpy`, `pydantic`, `pyyaml`, `typer` and should be treated as closed by default.
- No database, no server, no network calls in any command.
- Public functions carry type hints and are checked by mypy; `jeval/calibration.py` is under
  strict typing because its correctness is the product.
- User-facing messages explain what was measured and what is missing, never oversell. If a
  number cannot be supported by the sample, say so.
- Never write a stub that looks implemented. A command that is not implemented does not appear
  in the CLI, and the README status table says so.

## Changing the statistics

The measurement code is the product, so it is gated:

1. Write or extend a synthetic test **first**, using `jeval/synth.py` to inject a known
   miscalibration.
2. The test must fail before the change and pass after.
3. `tests/test_synth.py` must keep restoring: calibrated data reports ECE near zero within its
   bootstrap interval, inflated data reports the inflation direction, and
   accuracy-95%-with-confidence-0.99 is flagged as overconfident.
4. Never widen a test tolerance to make a change pass. A widened tolerance is a claim that the
   tool no longer detects what it used to detect.

## Harness and scratch files

Never commit agent scratch, harness output, or local plans. `.gitignore` already excludes
`.omh/`, `.omc/`, `.omo/`, `.scratch/`, `.claude/settings.local.json` and local plan files;
keep it that way, and do not add new tracked paths for tool state.

User data is equally off-limits: `.jeval/`, `*.jsonl`, `costs.yaml`, `thresholds.yaml`, and
generated reports are never committed. Synthesized sample data belongs under `examples/` and
must be clearly labeled as synthetic.

## Commit rules

Conventional Commits, English, imperative mood, one logical change per commit:

```
feat: add quantile binning with automatic bin reduction
fix: keep score records out of binary accuracy
docs: document the free-label harvest path
test: cover inflated-confidence recovery
chore: pin ruff in the dev group
```

Do not mix refactoring with behavior change in one commit. Do not commit generated reports or
data files.
