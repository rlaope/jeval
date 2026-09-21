# Setting jeval up on a real system

For the person driving an agent, and for the agent doing the work. Every command here was run
before it was written down. If a step cannot work for this user yet, it says so instead of
producing a number.

The whole task: **turn the classifier's own log into one artifact someone can act on.**

## 0. Confirm the tool works before touching anyone's data

```sh
uvx --from git+https://github.com/rlaope/jeval jeval demo --out-dir /tmp/jeval-demo
```

Synthetic data with a known miscalibration, so you can see what the output looks like with nothing
at stake. It writes three things: `report.html`, `costs.yaml`, `.jeval/records.jsonl`. Open the HTML
once — the rest of this page is about producing that file from real records.

## 1. Find the decisions

You need rows where a model answered a question: the prediction, the confidence it claimed, and a
way to join the human's eventual answer back onto that row later.

| Where decisions usually are | What to do |
| --- | --- |
| The app logs each model response (JSONL, CSV, or a table export) | Describe it in the ingest map (step 2) |
| A gateway or observability tool logs them (LiteLLM, Langfuse, Braintrust, …) | Export that view to JSONL, then step 2 |
| Nothing is logged yet | Wrap the client the app already calls, in one line (step 2b) |
| The vendor is Jev | Its managed API returns responses but keeps no log; same as "nothing is logged yet" |

One row per **question**, not per request: a request that answers three questions becomes three
records. Two things decide almost everything downstream — the confidence value, and the join key
(the trace/request id) that lets a human answer be matched back.

## 2a. Rows already exist: describe their shape

```sh
jeval init --root ~/myproject
```

Then edit `~/myproject/.jeval/ingest-map.yaml`. It is data, not code, and jeval refuses to guess:

```yaml
field_map:            # your column -> the record field it means
  ts: created_at
  model: model_name
  question_key: task
  prediction: label
  confidence: score_0_to_1
questions_field: questions      # the array holding per-question answers, if there is one
defaults:                       # values every record in this file shares
  model: jev-1.13.0
label_from:                     # where the human's answer lives, if the log already has it
  field: final_department
  question: department          # only rows for this question take that answer
```

A vendor shape may already be a preset — check before writing a map by hand:

```sh
jeval ingest --list-presets
jeval ingest --preset jev-native api-log.jsonl --root ~/myproject
```

Then load it and **read the count**:

```sh
jeval ingest ~/myproject/decisions.jsonl --root ~/myproject
```

It prints `wrote N records` and names every skipped row with its reason and its line number. If N
is 0, or rows are skipped in bulk, the map is wrong. Do not rewrite the user's file to fit the tool.

## 2b. Nothing is logged: one line in the app

```python
from jeval import collect

client = collect.track(YourClient(), source_key=lambda **kw: kw["trace_id"])
```

Every call the app already makes is recorded where it was made, including the model string and the
join key. It makes no calls of its own, never blocks, and never raises: a failed write is counted,
not thrown. Off with `JEVAL_COLLECT=0`, elsewhere with `JEVAL_COLLECT=/var/log/decisions.jsonl`.

This is the only path that works when the vendor keeps no log — and the only one that gives you a
join key for free.

## 3. Labels are the gate

Without a human's answer on a row, jeval cannot tell whether a confident prediction was right. The
command says `no labeled records` and means it. Three sources, in the order worth trying:

```sh
# (a) it is already in the log — the eventual outcome, the corrected department, the ticket that
#     was reopened. Map it with `label_from` (step 2a) or point at a separate file:
jeval ingest --labels resolutions.jsonl --root ~/myproject

# (b) harvest from a second file when the answer lives somewhere else
#     (same command, same flags; rows are matched on the join key and counted)

# (c) ask a person to fill in the ones that matter most, as a CSV sheet
jeval label --root ~/myproject --limit 20 --out labels.csv
jeval label --root ~/myproject --apply labels.csv
```

Before asking for hundreds of labels, check what they buy:

```sh
jeval plan --root ~/myproject --target-ci 0.05
```

It projects the interval width a given number of labels would give, fits the scaling from the data
in hand, and refuses below 200 labels rather than extrapolating from nothing.

## 4. Produce the artifact

```sh
jeval report --root ~/myproject \
  --current 0.7 \                 # the threshold in production right now, if there is one
  --costs ~/myproject/costs.yaml \
  --out ~/myproject/report.html
```

- `--current` is what makes the verdict usable: without it the report says it does not know what is
  deployed instead of inventing a baseline.
- `--costs` is the user's numbers, never yours. Copy the scaffold first:
  `cp ~/myproject/.jeval/examples/costs.example.yaml ~/myproject/costs.yaml`.
- `--by lang` splits the measurement per segment and says whether a per-segment threshold pays.
- `--format md` prints a paste-ready summary for a ticket or a PR.

Then hand the human the file path and three numbers: ECE, the accuracy-versus-claimed-confidence gap
in the worst bin, and the auto-rate at the threshold in use.

## 5. The artifact the application reads

```sh
jeval threshold --root ~/myproject            # writes thresholds.yaml
jeval threshold --root ~/myproject --by lang  # per-segment, when the split pays for itself
```

`thresholds.yaml` carries the threshold, its bootstrap interval, the expected cost per case and the
resulting auto-rate. A wide interval here is a label problem, not an analysis problem, and
`jeval plan` says how many more would narrow it.

If the confidences are systematically inflated, a correction map can be exported and applied by the
application:

```sh
jeval calibrate --root ~/myproject --method both
```

It declines to write a map when the gain is inside the noise, and says so.

## 6. Keep it from regressing

```sh
jeval drift --root ~/myproject --save-baseline ~/myproject/.jeval/baseline.json
jeval drift --root ~/myproject --fail-on ece-increase=0.05     # exit 1 when it degrades
```

`examples/ci/drift.yml` is a workflow that runs this on every change and posts the markdown summary.
The snapshot holds measurements, never records.

## What to show the user

1. The path to `report.html` and what it says in one sentence.
2. ECE and the worst bin, both read from the report — never estimated.
3. The threshold and its interval from `thresholds.yaml`, or an honest "not enough labels yet".
4. What one more labeling round would change, from `jeval plan`.

## Things that are not true, so do not say them

- `pip install jeval` — there is no PyPI release yet; use the wheel URL or `uvx --from git+…`.
- "The model is fine now" from a demo run — the demo data is synthetic.
- Any cost number the user did not give you.
- A single accuracy figure for `score` questions — those are MAE/RMSE/rank correlation, and they are
  never folded into binary accuracy.
