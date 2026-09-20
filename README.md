# jeval

**jeval measures the calibration of probabilistic AI classifiers and puts the human/AI
hand-off line where the cost says it belongs.**

Your classifier answers with a label and a confidence. jeval answers the two questions that
follow: *when it says 0.9, how often is it actually right?* and *given what a mistake costs,
where should the line sit?*

It is **provider-neutral by design**. Anything that returns a probability works — a vendor API,
a local model, a logistic regression, a rules engine with a score. jeval does not verify any
vendor's calibration claim and is not tied to one: the name came from one model family, the
tool sits above all of them.

---

## What the number actually means

A confidence is a claim. Here is the shape of the report — illustrative numbers, so you can
see what the output looks like before installing anything:

```
confidence   actual accuracy   n
0.5-0.6          48%          120
0.7-0.8          71%          298
0.8-0.9          77%          354   <-- you were treating this as "80%+"
0.9-1.0          93%          395
```

Nothing here is a screenshot, because a table is more honest and easier to check. The claim
"0.8-0.9 means 80%+" was wrong by three points in the band where most of your volume sits.

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
$ jeval drift --baseline .jeval/baseline.json --fail-on ece-increase=0.05
model changed: jev-1.13.0 -> jev-1.14.0 (Sep 18)
  question      ECE before   ECE after   delta
  department        0.061       0.142   +0.081   FAIL
  intent            0.044       0.049   +0.005   ok
recommended threshold (auto_refund): 0.89 -> 0.82
  at the current 0.89: auto-rate 58% -> 41%
exit 1
```

Wire that into CI and a model swap cannot silently degrade a production decision boundary.
**`jeval drift` is not implemented yet** — it is milestone M2 (see Status), and the output
above is its specified behavior, not a captured run.

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
- No gateway, no router, no hosting, no prompt optimization, no dashboard, no accounts.

## Getting labels for free

The single biggest reason people bounce off calibration tooling is that they think they need a
labeling project. You usually do not. You are already producing labels:

| You already have | Where the label is |
| --- | --- |
| Cases a human reviewed after escalation | the human's final answer — that is ground truth for the model's answer |
| Auto-processed cases that were later reversed | the reversal — the model was wrong |
| Refund approvals and rejections | the outcome — a real, dated answer |

Map those fields in `.jeval/ingest-map.yaml` and labeling cost drops to a mapping exercise.

## Status

v0.1 is milestones M0–M2. Anything not listed as implemented below is not implemented; the
repository does not ship stubs that look finished.

| Command | Purpose | Status |
| --- | --- | --- |
| `jeval init` | scaffold `.jeval/` config and ingest map | implemented (M0) |
| `jeval ingest` | JSONL/CSV logs to decision records | implemented (M0) |
| `jeval report` | reliability curve, ECE/MCE/Brier, Wilson and bootstrap intervals, one HTML file | implemented (M0) |
| `jeval demo` | synthetic log with known miscalibration | implemented (M0) |
| `jeval threshold` | cost matrix to per-action optimal threshold with a CI | milestone M1 |
| `jeval drift` | model-version and period comparison with CI exit codes | milestone M2 |
| `jeval label` | active-learning labeling queue | milestone M4 |

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
