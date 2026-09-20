# jeval

**jeval measures what your Jev classifier's confidence is really worth, and sets the human
hand-off line from what a mistake costs.**

Your classifier answers with a label and a confidence. jeval answers the two questions that
follow: *when it says 0.9, how often is it actually right?* and *given what a mistake costs,
where should the line sit?*

## What it looks like

<p align="center">
  <img src="docs/report-verdict.png" width="49%" alt="Report verdict: 'Your threshold is too low', with stat cards and the reliability curve plotted against the perfect-calibration diagonal">
  <img src="docs/report-cost.png" width="49%" alt="Cost curve with its minimum and flat region, the impact table comparing 0.60 to 0.97, and the threshold slider">
</p>

Left: the verdict and the reliability curve, with Wilson intervals on every bin. Right: the cost
curve, the impact table, and the slider that recomputes it. Both are screenshots of the report
committed to this repository — open
[`examples/report-example.html`](examples/report-example.html) in a browser (one 246 KB file, no
network, no server), or rebuild it byte-for-byte with:

```sh
uv run jeval demo --out-dir examples/report-example --seed 11 --scale 0.5
```

Synthetic data and synthetic costs, seeded, so the example is reproducible rather than a
one-off screenshot anyone could have made up.

It is **provider-neutral by design**. Anything that returns a probability works — a vendor API,
a local model, a logistic regression, a rules engine with a score. jeval does not verify any
vendor's calibration claim and is not tied to one: the name came from one model family, the
tool sits above all of them.

---

## What the number actually means

A confidence is a claim. Here is the curve's own table, five of its ten rows copied out of the
report in [`examples/report-example.html`](examples/report-example.html) — a real generated
report, over synthetic data, committed so you can check every claim in this README against the
artifact itself:

```
Confidence bin      n   Stated   Observed   Wilson 95%      Gap
0.61-0.70          27     66%       56%     [37%, 72%]   -0.104
0.74-0.77          27     76%       74%     [55%, 87%]   -0.016
0.82-0.84          26     83%       77%     [58%, 89%]   -0.064
0.90-0.94          27     92%       85%     [68%, 94%]   -0.070
0.97-1.00          27     98%      100%     [88%, 100%]  +0.017
```

Read the third row: the model said 83% and was right 77% of the time, and the interval around
that number spans 58% to 89% — which is what a 26-record bin can actually support.

That report's own verdict, for the threshold the demo treats as deployed:

> **Your threshold is too low.** Band 0.03-0.45 measures 69.4% accuracy on 72 decisions (of 713 labels); the threshold belongs at 0.97, above the 0.60 in use.

| | now | recommended | change |
| --- | --- | --- | --- |
| confidence threshold | 0.60 | 0.97 | +0.37 |
| auto rate | 100% | 55% | -45.2 pt |
| accuracy (auto) | 84% | 96% | +11.5 pt |
| cost per case | KRW 7,936.51 | KRW 2,095.24 | -73.6% |
| monthly cost | KRW 158,730,158.73 | KRW 41,904,761.90 | -73.6% |

No screenshot, because a table is easier to check — and no invented numbers, because the artifact
is right there in the repository.

## You could write this yourself with 30 lines of pandas

Yes. You should, once. Here is the whole idea:

```python
import pandas as pd

df = pd.read_json("decisions.jsonl", lines=True)
df = df[df.label.notna()].copy()  # only labeled decisions count
df["correct"] = df.prediction == df.label
df["bin"] = pd.cut(df.confidence, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

table = df.groupby("bin", observed=True).agg(
    n=("correct", "size"),
    claimed=("confidence", "mean"),
    actual=("correct", "mean"),
)
table["gap"] = table.actual - table.claimed
ece = (table.n / table.n.sum()).dot(table.gap.abs())

print(table)
print(f"ECE = {ece:.3f}")
```

That is the measurement, and it is correct. Run it on the same records as `jeval report` and the
number will still disagree, because of one line: `pd.cut` bins by equal width while jeval bins by
quantile. On the 91 labeled records in `examples/`, that single choice is the difference between
ECE 0.113 and ECE 0.076 — equal-width binning put 38 of those labels in one bin while the others
held 17 and 18, so one bin carried 40% of the weight.

Which number you ship is a real decision, not a detail. jeval is what you need *after* you have
written the 30 lines once and realised the hard part is everything around them.

## The hard part is that this is a standing problem, not an analysis

- It has to run **every week, per question, per segment** — not once on a laptop.
- **Sampling error is the whole ballgame.** Nobody hand-rolls Wilson or bootstrap intervals,
  so people read a 12-record bin as if it were the truth and move a production threshold on it.
- Notebooks live on **one person's machine**. When that person is on holiday, nobody can
  re-run the number, and the number still governs production.
- **The model changes under you.** A pinned tag or an auto-updating alias swaps the model
  behind a threshold you tuned last quarter, and you find out three weeks later.

jeval turns all four into one command with an exit code.

## `jeval drift` is the part a notebook cannot replace

A notebook measures once. Drift detection is what happens when the thing you measured changes:

```
$ jeval drift --fail-on ece-increase=0.05
model changed: jev-1.13.0 -> jev-1.14.0 (Sep 17)
  question    ECE before  ECE after   delta
  department       0.018      0.144  +0.126   FAIL
recommended threshold (department): 0.94 -> 1.00
  at the current 0.94: auto-rate 11% -> 58%
$ echo $?
1
```

That is a captured run, not a mock-up: 1,800 synthetic decisions where the newer model version is
deliberately overconfident. The threshold line only appears when a cost matrix is present — with
no costs, the block says `recommended threshold: not available (no cost matrix was applied)` rather
than inventing a number. Wire the same command into CI and a model swap cannot silently
degrade a production decision boundary. There is a copy-paste starting point in
[`examples/ci/drift.yml`](examples/ci/drift.yml): it runs `drift --fail-on`, prints the markdown
summary, and comments it on the pull request. jeval prints that summary; the workflow posts it,
because jeval never holds a token.

`--baseline .jeval/baseline.json` compares against the last measurement you accepted rather than
only model-to-model, which matters when the model string never changes but the behaviour does. The
snapshot holds measurements, never records, so it is safe to commit.

## Try it in five minutes

```sh
uvx jeval demo
```

Until the first release is on PyPI, run the same thing from a checkout
(`uv run jeval demo`). Either way it generates a synthetic decision log whose miscalibration is
known, reports on it, and shows you the reliability curve and ECE. Nothing is downloaded from your side and no data
leaves the machine; the demo data is generated locally and is explicitly labeled as synthetic.

## What jeval does not do

- It does **not** validate a vendor's calibration claim in general. It measures the model you
  actually ran, on the records you actually supplied.
- Threshold recommendations depend **entirely on the cost numbers you type in**. If your costs
  are wrong, the recommended threshold is wrong, and nothing in the output will warn you.
- Results built from **silver labels** (labeled by another model) are an *agreement rate*, not
  an accuracy, and the report says so loudly.
- **Without labels, jeval measures nothing.** Read "Getting labels for free" below before
  concluding the tool is unusable — you are probably already producing labels without
  noticing.
- `jeval calibrate` exports a correction map; **jeval never applies it**. Adapters, gateways,
  routers and request-path libraries stay out of scope — that is your application's job.
- No gateway, no router, no hosting, no prompt optimization, no fine-tuning, no dashboard,
  no accounts.

## Getting labels for free

The single biggest reason people bounce off calibration tooling is that they think they need a
labeling project. You usually do not. You are already producing labels:

| You already have | Where the label is |
| --- | --- |
| Cases a human reviewed after escalation | the human's final answer — that is ground truth for the model's answer |
| Auto-processed cases that were later reversed | the reversal — the model was wrong |
| Refund approvals and rejections | the outcome — a real, dated answer |

Point `.jeval/ingest-map.yaml` at those fields and labeling cost drops to a mapping exercise:

```yaml
label_from:
  field: resolution.final_department   # dotted paths work
  source: human_override               # human_review | human_override | silver
  join_on: ticket_id                   # the key shared by your log and the resolution log
  question: department                 # the question this column answers
```

```sh
$ jeval ingest log.jsonl                       # records carry source_key from `join_on`
$ jeval ingest log.jsonl --labels resolutions.jsonl
applied 412 labels to 1,240 records to the 'department' question
left 828 record(s) of other questions alone: 'resolution.final_department' answers one question,
  and a request's other questions share the same key
```

`question:` is not decoration. A join key is shared by every question of a request, so without it a
department answer would also be written as the *intent* answer. Two guards stand between your
records and a wrong label: a harvest never overwrites an existing label unless you pass
`--overwrite`, and it refuses a label the record's own question could not have produced (a
`choice` label outside that record's `probabilities`, a `noul` answer that is not yes/no) instead
of writing it — refusing is counted and named in the output, because a wrong label is worse than a
missing one. Pass `--allow-unlisted-labels` if your `probabilities` map lists top candidates only.

The harvest rewrites `.jeval/records.jsonl` in place and atomically, touching only the label
fields — every other byte of your log is preserved.

## How many more labels do you need?

The interval around your ECE is the honest limit of what your sample can say. `jeval plan` projects
how many additional labels each question needs to tighten it — and refuses to guess below 200
labels:

```
$ jeval plan --target-ci 0.05
scope      key                       n     ECE      CI  needed
question   department              511   0.062   0.059  0.015: 6,346 · 0.030: 1,204 · 0.050: 91
question   intent                  490   0.124   0.064  0.016: 6,967 · 0.032: 1,375 · 0.050: 276
question   satisfaction              0     nan     nan  only 0 gold-labeled records; at least 200
  are needed before the k/sqrt(n) scaling can be fitted
```

The projection assumes the interval width scales as `k/sqrt(n)` with `k` fitted from your own data
at `n` and `n/2`. It is an estimate, and the output says so.

## If you want to fix the confidence, not just measure it

`jeval calibrate` fits a correction — temperature scaling or isotonic regression — and exports it
as a YAML map your application applies:

```
$ jeval calibrate --method both
temperature: ECE 0.144 -> 0.061 (cross-validated, n=1,240)
  wrote /your/project/.jeval/calibration-temperature.yaml
isotonic: ECE 0.144 -> 0.072 (cross-validated, n=1,240)
  wrote /your/project/.jeval/calibration-isotonic.yaml
```

Two things keep this honest. The gain is measured **cross-validated**, never on the records the map
was fitted on, and a map ships only if that gain beats the sampling noise of your own log. When it
does not, `calibrate` exports nothing and says so:

```
$ jeval calibrate --method isotonic
isotonic: ECE 0.041 -> 0.041 (cross-validated, n=1,240)
  No correction retained: the best map moves cross-validated ECE by +0.003, which does not clear
  the 0.022 floor. Apply() is the identity here: this log does not need this correction.
  no correction exported: shipping an unearned map would make it worse
```

jeval never applies the map. It writes a file; your application reads it. There is no request path
where jeval sits between you and your model.

## Segments: does one threshold fit everyone?

The report already shows that your segments are not calibrated alike. `jeval threshold --by lang`
asks the follow-up question in money: does giving a segment its own threshold pay?

```
$ jeval threshold --by lang
auto_route by lang: global threshold 0.84
  segment            threshold  cost/case  vs global      n  verdict
  lang = en               0.81   1,306.30     -42.52    254  split
  lang = ko               0.84   1,159.92       0.00    257  splitting does not pay: its optimum 0.84
    is within one sweep step (0.01) of the global 0.84 and the cost change of 0 per case is
    under the 2% bar.
auto_escalate_urgent by lang: global threshold 0.87
  segment            threshold  cost/case  vs global      n  verdict
  lang = en               0.70     740.74    -119.05    189  split
  lang = ko               0.89     912.09       0.00    182  splitting does not pay: its optimum 0.89
    differs from the global 0.87 and the cost change of 0 per case is under the 2% bar.
```

That is the demo dataset with the demo cost matrix. A split is recommended only when the segment's
optimum moves by more than one sweep step **and** adopting it changes cost per case by more than
2%; otherwise the output says "splitting does not pay" and names the clause that failed. Note the
answer is not uniform — `en` pays to split on both actions, `ko` never does. A one-line answer to a
question people usually settle by intuition.

## Status

v0.1 is milestones M0–M4 plus the report extension (R0–R4). Anything not listed as implemented
below is not implemented; the repository does not ship stubs that look finished. `tests/test_documented_features.py`
fails if this table and the command surface disagree in either direction.

| Command | Purpose | Status |
| --- | --- | --- |
| `jeval init` | scaffold `.jeval/` config and ingest map | implemented (M0) |
| `jeval ingest` | JSONL/CSV logs to decision records | implemented (M0) |
| `jeval report` | the argument document: verdict, reliability, cost, impact, segments, score questions, labels-and-correction, drift, data quality — one HTML file, or `--format md` for a paste-ready summary | implemented |
| `jeval demo` | synthetic log with known miscalibration, rendered through the same report path | implemented |
| `jeval threshold` | cost matrix to per-action threshold with a bootstrap interval, written to `thresholds.yaml`; `--by <segment>` answers whether splitting pays | implemented |
| `jeval drift` | model-version and period comparison, baseline snapshots, `--fail-on` exit codes, and the cost-driven threshold movement per slice when a cost matrix is present | implemented |
| `jeval ingest --labels` | free-label harvest: human answers from a resolution log, joined on your own key (M3) | implemented |
| `jeval label` | active-learning labeling queue with a CSV sheet to fill in (M4) | implemented |
| `jeval plan` | additional labels needed for a tighter interval, per question and segment | implemented |
| `jeval calibrate` | temperature / isotonic correction map, cross-validated, exported as YAML for your app | implemented |

`jeval label` is a queue and a sheet, not a full-screen TUI: it ranks what to label, exports the
sheet, and applies your answers back with `jeval label --apply labels.csv`.

## The report is an argument, not a dashboard

A terminal summary cannot show the shape of a curve, and it cannot be pasted into a thread when
you are trying to move a threshold. So `jeval report` writes one self-contained HTML file that
reads top to bottom as a single case:

| Section | What it settles |
| --- | --- |
| ① Verdict | the conclusion in one sentence, with a Copy summary button for the thread |
| ② Reliability | the curve, with dot size by sample count, Wilson intervals, and a density strip showing where your traffic sits |
| ③ Cost | expected cost per case for every candidate threshold, with the minimum and the flat region marked |
| ④ Impact | what changes if you move: auto rate, accuracy of what stays automated, cost per case, monthly |
| ⑤ Segments | which slice is misfiring, worst first, small segments greyed rather than dropped |
| ⑥ Drift | before/after curves, model-change markers, and the threshold staircase (when there is something to compare) |
| ⑦ Data quality | labeled vs unlabeled, label sources, bin counts, and the limitations in plain words |

No chart library, no server, no external request of any kind: every chart is inline SVG built by
jeval, and the file opens offline. The threshold slider in ④ is exploration only — the report
never writes `thresholds.yaml`, because a browser should not be editing your configuration.

```sh
jeval report                          # report.html
jeval report -o out/week38.html       # somewhere specific
jeval report --by lang --by tier      # segment breakdown, two axes gives a grid
jeval report --question intent        # one question only
jeval report --compare .jeval/baseline.json   # activate ⑥ Drift
jeval report --format md              # verdict + impact as markdown, for a PR comment
jeval report --open                   # open it when it is done
```

`--format md` prints only the verdict and the impact table. That is deliberate: it is what a CI
job pastes into a pull request when `jeval drift` fails, so a reviewer learns what happened
without opening a file. jeval itself never posts it — that needs a token, and jeval does not
handle tokens.

`jeval threshold` writes the file your application reads:

```yaml
generated_at: 2026-09-20T11:00:00Z
model: jev-1.13.0
n_records: 1240
actions:
  auto_refund:
    threshold: 0.91
    expected_cost_per_case: 1830
    auto_rate: 0.62
    ci: {low: 0.87, high: 0.94}
```

Wide interval? That is a label problem, not an analysis problem, and the command says so.

## Install

```sh
uvx jeval demo       # try it without installing
uv tool install jeval
```

Neither of those works yet: the first PyPI release is gated on the M0-M2 milestone set
completing. From a checkout:

```sh
git clone https://github.com/rlaope/jeval && cd jeval
uv sync --all-groups
uv run jeval demo --out-dir /tmp/jeval-demo
```

Python 3.10 or newer. Runtime dependencies: `numpy`, `pydantic`, `pyyaml`, `typer`. The report
contains no JavaScript and no external assets — one HTML file you can email.

## Data model

Everything is a decision record, one question per record. Records live in `.jeval/records.jsonl`
so they diff, stream and survive whatever tool you read them with.

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

Three properties of this schema carry most of the value:

- **One record per question.** A single request that answers "is this a refund?" and "how
  annoyed is the customer?" can be reliable on one and useless on the other. Pooled metrics
  hide exactly that.
- **`model` is preserved verbatim.** `jev-latest` points at different models over time; a
  threshold tuned against it is a threshold tuned against something that no longer exists.
- **`state_tokens` is preserved.** Accuracy tends to fall as the state grows, so
  `--by state_tokens` is one of the more useful reports you can run.

`score` records are deliberately not folded into accuracy: a score is near or far, not right or
wrong, and the report says how many were excluded instead of pretending they were counted.

## Development

```sh
uv sync --all-groups
uv run pytest                 # includes the synthetic-data restoration tests
uv run ruff format --check .
uv run ruff check .
uv run mypy jeval
uv run jeval demo --out-dir /tmp/jeval-demo
```

The statistics are the product. `tests/test_synth.py` generates decision logs with a *known*
miscalibration and asserts that the measurement code recovers it — a calibrated sample must
report ECE near zero, an inflated one must report the inflation, and a 95%-accurate sample that
always claims 0.99 must be caught as overconfident. If a change to the statistics cannot pass
those tests, the change is wrong, not the tests.

See `CONTRIBUTING.md` for the commit convention and `CLAUDE.md` for the rules that apply to
automated contributors.

## License

Apache-2.0. See `LICENSE`.
