<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo-dark.png" />
    <img src="docs/logo-light.png" alt="jeval" width="280" />
  </picture>
</p>

**Measure what your classifier's confidence is really worth, and set the human hand-off line from
what a mistake costs.**

<div align="center">

[![CI](https://github.com/rlaope/jeval/actions/workflows/ci.yml/badge.svg)](https://github.com/rlaope/jeval/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/rlaope/jeval?color=blue)](https://github.com/rlaope/jeval/releases)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![PyPI](https://img.shields.io/badge/PyPI-not%20published-lightgrey)](#install)

</div>

<p align="center">
  <img src="docs/report-verdict.png" width="49%" alt="Report verdict: 'Your threshold is too low', with stat cards and the reliability curve plotted against the perfect-calibration diagonal">
  <img src="docs/report-cost.png" width="49%" alt="Cost curve with its minimum and flat region, the impact table comparing 0.60 to 0.75, and the threshold slider">
</p>

Your classifier answers with a label and a confidence. jeval answers the two questions that follow:
*when it says 0.9, how often is it actually right?* and *given what a mistake costs, where should the
line sit?*

Both answers come out of **one self-contained HTML file** and **one YAML file your application
reads**. Everything is computed from labeled decision records on disk: **no server, no database, no
network call, no account, no token.**

It is **provider-neutral by design**. Anything that returns a probability works — a managed API, a
gateway, a local model, a logistic regression, a rules engine with a score. jeval verifies no
vendor's calibration claim and is tied to none: the name came from one model family, the tool sits
above all of them.

---

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/rlaope/jeval/main/install.sh | sh
jeval demo
```

That is the whole installation: a `jeval` command on your PATH. No uv, no pipx, no root, and nothing
to do with PyPI. The installer puts a private virtual environment under `~/.local/share/jeval`, links
`jeval` into `~/.local/bin`, and prints the one line to add if that directory is not on your PATH yet
— or pass `--modify-path` and it edits your shell rc for you. Re-run it to upgrade;
`sh install.sh uninstall` removes everything it made.

Prefer to install nothing at all?

```sh
uvx --from git+https://github.com/rlaope/jeval jeval demo            # current main
uvx --from git+https://github.com/rlaope/jeval@v0.1.7 jeval demo     # pinned tag
pip install https://github.com/rlaope/jeval/releases/download/v0.1.7/jeval_cli-0.1.7-py3-none-any.whl
```

`pip install jeval` installs an **unrelated project**: that name on PyPI belongs to someone else,
and this tool is not published to PyPI at all. Every tag is built by the release workflow, which
attaches the wheel and the sdist to the GitHub release and reads the asset list back.

Python 3.10 or newer · runtime dependencies `numpy`, `pydantic`, `pyyaml`, `typer` · from a checkout:
`uv sync --all-groups`.

---

## Hand it to an agent

Nobody wants a command tour. Give an agent one sentence and take the artifact back:

> Install jeval (`uvx --from git+https://github.com/rlaope/jeval jeval --help`), find where my
> classifier's decisions are logged, describe that shape in `.jeval/ingest-map.yaml`, run
> `jeval report`, and show me the report file.

[`llms.txt`](llms.txt) is the machine-readable entry point and
[`docs/agent-setup.md`](docs/agent-setup.md) is the playbook it follows — including the two places
every attempt stalls: nothing logged yet, and no labels yet.

---

## What comes out

| Artifact | What it answers | Who reads it |
| --- | --- | --- |
| `report.html` — one self-contained file | Are the confidences trustworthy, where do they break, and what does the threshold in use cost? | the human |
| `thresholds.yaml` | The threshold to deploy, with a bootstrap interval, and whether splitting by segment pays | your application |
| `labels.csv` | Which decisions, labeled next, buy the most certainty | whoever has the answers |
| `calibration-*.yaml` — optional | A correction map your application applies, exported only when the gain is real | your application |

---

## Instrumenting the service you already run

Two lines: one where the classifier is called, one where the human answer lands.

```python
from jeval import collect

client = collect.track(
    TypeSafeClient(),  # your SDK, not jeval's
    method_names=("system_one",),  # the method that answers questions
    source_key=lambda **kw: kw["trace_id"],  # what a human answer is joined back on
    segment=lambda **kw: {"lang": kw.get("lang")},  # request attributes become segment axes
)
```

Then check `collect.stats()["calls"]` is not zero after your first request: a wrapper that found no
method to patch is counted in `no_method_found`, because a silent no-op is how an integration looks
installed while collecting nothing. Point it at your own directory with
`JEVAL_ROOT=/var/lib/jeval`, which writes the same `.jeval/records.jsonl` that
`jeval report --root /var/lib/jeval` opens. Collection is off with `JEVAL_COLLECT=0`.

The wrapper is a **recorder, not a proxy**: it observes a call your code already makes, never calls a
model, never chooses one, never retries, never blocks, and never raises — a failed write is counted.
It imports no vendor SDK; a preset, not a branch, is where a product's field names live.

The end-to-end walkthrough, with the output of every step, is in
[`docs/instrumenting-a-service.md`](docs/instrumenting-a-service.md).

### A log you already have

Skip the wrapper and describe the shape instead — `field_map`, `questions_field`, `defaults`,
`label_from`; a flat column can serve as a segment axis:

```sh
jeval init --root ~/myproject
jeval ingest ~/myproject/decisions.jsonl --root ~/myproject
jeval ingest ~/myproject/decisions.jsonl --labels ~/myproject/resolutions.jsonl \
  --label-field final_department --label-source human_override \
  --join-on ticket_id --label-question department --root ~/myproject
```

A product's own response shape may already be a preset:

```sh
$ jeval ingest --preset jev-native api-decisions.jsonl
preset: jev-native (response at 'response', answers under 'answers', join key from 'request_id')
read 4 rows
wrote 8 records to .jeval/records.jsonl
```

`jeval ingest --list-presets` shows what is available; `--response-field` and `--source-key-field`
override where it looks.

---

## Where the labels come from

The single biggest reason people bounce off calibration tooling is thinking they need a labeling
project. You are already producing labels — labeling cost is a mapping exercise:

| You already have | Where the label is |
| --- | --- |
| Cases a human reviewed after escalation | the human's final answer — ground truth for the model's answer |
| Auto-processed cases later reversed | the reversal — the model was wrong |
| Refund approvals and rejections | the outcome — a real, dated answer |

```yaml
label_from:
  field: resolution.final_department   # dotted paths work
  source: human_override               # human_review | human_override | silver
  join_on: ticket_id                   # the key shared by your log and the resolution log
  question: department                 # the question this column answers
```

`question:` is not decoration. A join key is shared by every question of a request, so without it a
department answer would also be written as the *intent* answer. Two guards stand between your records
and a wrong label: a harvest never overwrites an existing label unless you pass `--overwrite`, and it
refuses a label the record's own question could not have produced — refusing is counted and named,
because a wrong label is worse than a missing one. The harvest rewrites `.jeval/records.jsonl` in
place and atomically, touching only the label fields.

**Without labels, jeval measures nothing.** It says so and stops rather than producing a number.

---

## What the numbers mean

A confidence is a claim. Five of the ten rows of the reliability table, copied out of the report in
[`examples/report-example.html`](examples/report-example.html) — a real generated report, over
synthetic data, committed so every claim in this README can be checked against the artifact itself:

```
Confidence bin      n   Stated   Observed   Wilson 95%      Gap
0.60-0.69          26     65%       46%     [29%, 65%]   -0.188
0.74-0.77          26     76%       69%     [50%, 83%]   -0.064
0.82-0.85          26     83%       69%     [50%, 83%]   -0.140
0.91-0.95          25     93%       96%     [80%, 99%]   +0.033
0.97-1.00          26     98%      100%     [87%, 100%]  +0.016
```

Read the third row: the model said 83% and was right 69% of the time, and the interval around that
number spans 50% to 83% — which is what a 26-record bin can actually support.

That report's own verdict, for the threshold the demo treats as deployed:

> **Your threshold is too low.** Band 0.03-0.44 measures 68.6% accuracy on 70 decisions (of 696 labels); the threshold belongs at 0.75, above the 0.60 in use.

| | now | recommended | change |
| --- | --- | --- | --- |
| confidence threshold | 0.60 | 0.75 | +0.15 |
| auto rate | 33% | 30% | -2.9 pt |
| accuracy (auto) | 85% | 91% | +5.4 pt |
| cost per case | KRW 1,926.23 | KRW 1,737.70 | -9.8% |
| monthly cost | KRW 38,524,590.16 | KRW 34,754,098.36 | -9.8% |

No invented numbers, because the artifact is in the repository. Open
[`examples/report-example.html`](examples/report-example.html) in a browser (one 249 KB file, no
network, no server), or rebuild it byte-for-byte:

```sh
uv run jeval demo --out-dir examples/report-example --seed 11 --scale 0.5
cp examples/report-example/report.html examples/report-example.html
```

### The report is an argument, not a dashboard

It reads top to bottom as one case, in a fixed order: ① verdict, ② reliability with Wilson intervals
and a density strip, ③ cost curve with its minimum and flat region marked, ④ impact of moving,
⑤ segments worst-first, ⑥ drift before/after, ⑦ data quality. Every chart is inline SVG built by
jeval — no chart library, no external request, and the file opens offline.

### You could write the measurement yourself with 30 lines of pandas

Yes. You should, once. Then the number will still disagree with `jeval report`, because of one line:
`pd.cut` bins by equal width while jeval bins by quantile. On the demo records that single choice is
the difference between ECE 0.113 and ECE 0.076 — equal-width binning put 38 labels in one bin while
the others held 17 and 18, so one bin carried 40% of the weight. Which number you ship is a decision,
not a detail.

---

## `jeval drift` is the part a notebook cannot replace

A notebook measures once. Drift detection is what happens when the thing you measured changes:

```
$ jeval drift --root /tmp/jeval-drift --fail-on ece-increase=0.05
costs: /tmp/jeval-drift/costs.yaml
model changed: jev-1.13.0 -> jev-1.14.0 (Sep 16)
  question    ECE before  ECE after   delta
  department       0.028      0.141  +0.113   FAIL
recommended threshold (department): 0.96 -> 0.98
  at the current 0.96: auto-rate 2% -> 5%
$ echo $?
1
```

That is a captured run, not a mock-up, and you can rebuild the log it ran on — 1,800 synthetic
decisions where the newer model version is deliberately overconfident:

```sh
uv run python examples/make-drift-log.py /tmp/jeval-drift
uv run jeval drift --root /tmp/jeval-drift --fail-on ece-increase=0.05
```

The threshold line appears only when a cost matrix is present; with no costs the block says
`recommended threshold: not available (no cost matrix was applied)` rather than inventing a number.
`--save-baseline .jeval/baseline.json` compares against the last measurement you accepted, which
matters when the model string never changes but the behaviour does — the snapshot holds measurements,
never records, so it is safe to commit. [`examples/ci/drift.yml`](examples/ci/drift.yml) is the
copy-paste CI starting point: it runs the check, prints the markdown summary, and comments it on the
pull request, because jeval itself never holds a token.

---

## Which labels next, and how many

`jeval plan` projects how many additional labels each question needs to tighten the interval, and
refuses to guess below 200 labels:

```
$ jeval plan --target-ci 0.05
scope      key                       n     ECE      CI  needed
question   department              511   0.062   0.059  0.015: 6,346 · 0.030: 1,204 · 0.050: 91
```

`jeval label` ranks what to label instead of asking for a labeling project: it exports a CSV sheet
of the decisions that sit on the decision line, and applies your answers back with
`jeval label --apply labels.csv`. It is a queue and a sheet, not a full-screen TUI.

`jeval calibrate` fits a correction — temperature scaling or isotonic regression — and exports it as
a YAML map your application applies, measured **cross-validated**, never on the records it was fitted
on. When the gain does not clear the sampling noise of your own log, it exports nothing and says so.
jeval never applies the map: it writes a file, your application reads it.

`jeval threshold --by lang` asks the follow-up question in money — does giving a segment its own
threshold pay? A split is recommended only when the segment's optimum moves by more than one sweep
step **and** adopting it changes cost per case by more than 2%; otherwise the output says "splitting
does not pay" and names the clause that failed.

---

## Honest limits

* **No labels, no measurement.** Without a human's answer on a row, jeval cannot tell whether a
  confident prediction was right, and it stops instead of inventing a number.
* **The costs are yours.** Every recommended threshold is a direct consequence of the figures in
  `costs.yaml`. Wrong costs give wrong thresholds, and jeval cannot know that a refund costs more
  than a support hour at your company.
* **A wide interval is a label problem, not an analysis problem.** `jeval plan` says how many more
  labels a target needs; nothing in the output warns you if the sample cannot support the answer.
* **Correctness is only defined for `choice`.** `score` questions get MAE, RMSE and rank agreement,
  are never folded into binary accuracy, and the excluded count is printed.
* **Results from silver labels are an agreement rate**, not an accuracy, and the report says so
  loudly. A label with no `label_source` counts as silver, never as gold.
* **The threshold "in use" is only known when you say it.** Pass `--current`, or keep a
  `thresholds.yaml`; with nothing deployed the report says so instead of comparing the recommendation
  with itself.
* **`--bins 1` is refused.** One bin averages every decision together, so ECE collapses toward zero
  and the report reads as "confidence is trustworthy" whatever the data says.
* **A projection is an estimate.** The label projection fits the width's scaling exponent from your
  own subsamples and prints the fit with its residual, and says when it fell back to 1/sqrt(n).
* **The demo is synthetic.** `jeval demo` shows what the tool computes, not what a real model does —
  the numbers in this README come from that demo, and the artifact says so itself.
* **No gateway, no router, no hosting, no prompt optimization, no fine-tuning, no dashboard, no
  accounts.** Adapters and request-path libraries stay out of scope: jeval writes files, your
  application writes the request path.

---

## Command surface

`tests/test_documented_features.py` fails if this table and the real CLI disagree in either
direction, and nothing here is a stub that only looks implemented.

| Command | Purpose | Status |
| --- | --- | --- |
| `jeval init` | scaffold `.jeval/` config and ingest map | implemented |
| `jeval ingest` | JSONL/CSV logs to decision records; `--preset jev-native` reads a decision API's own response log; `--labels` harvests human answers | implemented |
| `jeval report` | the argument document: verdict, reliability, cost, impact, segments, score questions, labels-and-correction, drift, data quality — one HTML file, or `--format md` for a paste-ready summary | implemented |
| `jeval threshold` | cost matrix to per-action threshold with a bootstrap interval, written to `thresholds.yaml`; `--by <segment>` answers whether splitting pays | implemented |
| `jeval drift` | model-version and period comparison, baseline snapshots, `--fail-on` exit codes, and the cost-driven threshold movement per slice | implemented |
| `jeval label` | active-learning labeling queue with a CSV sheet to fill in | implemented |
| `jeval plan` | additional labels needed for a tighter interval, per question and segment | implemented |
| `jeval calibrate` | temperature / isotonic correction map, cross-validated, exported as YAML for your app | implemented |
| `jeval demo` | synthetic log with known miscalibration, rendered through the same report path | implemented |

`jeval report` writes one self-contained HTML file and the terminal summary; the report never writes
configuration, never fetches anything at runtime, and never posts to a pull request.

---

## Data model

Everything is a decision record, one question per record, in `.jeval/records.jsonl` so it diffs,
streams and survives whatever tool you read it with:

```jsonc
{
  "id": "rec_1f4c2a...",
  "ts": "2026-09-20T10:31:02Z",
  "model": "jev-1.13.0",          // the model string from the response; the drift anchor
  "question_key": "department",
  "question_type": "choice",     // choice | score | noul
  "prediction": "billing",
  "confidence": 0.91,            // normalized top-1 probability
  "probabilities": {"billing": 0.91, "technical": 0.06, "other": 0.03},
  "label": "billing",            // null until something labels it
  "label_source": "human_override",
  "segment": {"lang": "ko", "tier": "pro"},
  "state_tokens": 1840,
  "latency_ms": 210,
  "cost_usd": 0.00008
}
```

Three properties carry most of the value:

* **One record per question.** A request that answers "is this a refund?" and "how annoyed is the
  customer?" can be reliable on one and useless on the other; pooled metrics hide exactly that.
* **`model` is preserved verbatim.** `jev-latest` points at different models over time, and a
  threshold tuned against it is tuned against something that no longer exists.
* **`state_tokens` is preserved.** Accuracy tends to fall as the state grows, so
  `--by state_tokens` is one of the more useful reports you can run.

---

## Docs

* [`llms.txt`](llms.txt) — machine-readable entry point for an agent
* [`docs/agent-setup.md`](docs/agent-setup.md) — the setup playbook, including the no-labels path
* [`docs/instrumenting-a-service.md`](docs/instrumenting-a-service.md) — instrumenting a running
  service end to end, with the output of every step
* [`examples/report-example.html`](examples/report-example.html) — a real generated report
* [`examples/ci/drift.yml`](examples/ci/drift.yml) — the CI starting point
* [`CHANGELOG.md`](CHANGELOG.md) — what changed, and why

## Development

```sh
uv sync --all-groups
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run mypy jeval
uv run jeval demo --out-dir /tmp/jeval-demo
```

The statistics are the product. `tests/test_synth.py` generates decision logs with a *known*
miscalibration and asserts the measurement code recovers it: a calibrated sample must report ECE
near zero, an inflated one must report the inflation, and a 95%-accurate sample that always claims
0.99 must be caught as overconfident. If a change to the statistics cannot pass those tests, the
change is wrong, not the tests. See `CONTRIBUTING.md` for the commit convention and `CLAUDE.md` for
the rules that apply to automated contributors.

## License

Apache-2.0. See `LICENSE`.
