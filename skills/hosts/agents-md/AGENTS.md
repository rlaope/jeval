# jeval

How to measure a classifier's confidence and set the human/AI hand-off line with jeval
(0.2.0). Generated from the canonical skills in the jeval repository; each section
names the artifact it produces. Run `jeval <command> --help` for the flags of any step,
and `jeval report` once a step has produced records.

Read every number as an estimate with an interval. jeval measures what records support
and refuses to answer when they do not: no labels means no measurement, and it says so.

## jeval-calibration-audit

Use when a classifier returns a confidence and nobody has checked what it is worth. Measures calibration from labeled records and names the confidence bands where the model is wrong about itself.

1. Confirm the records carry labels. A record without one cannot be judged:

```sh
jeval report --root . --format md
```

If this answers `no labeled records: nothing to measure.`, load `jeval-labels-harvest` and come back.

2. Find out what threshold is in use today, and pass it. Without it the report cannot compare:

```sh
jeval report --root . --current 0.88 --format md
```

3. Read four things out of the output, in this order. Do not paraphrase them into adjectives:

- the labeled count, and how many records were excluded for having no label;
- `ECE` with its confidence interval — the average gap between the confidence claimed and the
  frequency observed, where 0 would be perfect;
- the worst band, printed as `Overconfidence in 0.62-0.88: claimed 0.76, observed 0.14`;
- the label projection, printed as `176 more labels would roughly halve the ECE interval`.

4. Look for a segment that is worse than the average, and only then by the segment the product
   actually cares about (language, tenant, tier, state length):

```sh
jeval report --root . --current 0.88 --by lang
```

5. Open the HTML artifact and confirm the same numbers are in it, then hand over the file. The report
   is the deliverable; your summary is a courtesy copy of its top section.

**Verify**

The terminal summary and the file agree: the same labeled count, the same ECE and interval, the same
verdict line. `report.html` exists, opens offline, and carries a `generated_at` timestamp. If the
numbers differ between the summary and the artifact, stop and report the disagreement rather than
choosing one.

**Do not claim**

- Do not claim the tool proved the model is miscalibrated: ECE is an estimate with an interval, and
  the interval is part of the answer.
- Do not claim a fix. `jeval calibrate` exports a correction map for the application to apply, is
  measured on held-out records, and exports nothing when the gain does not clear the noise.
- Do not claim accuracy for `score` questions: they are measured as MAE, RMSE and rank agreement, and
  are never folded into binary accuracy.

## jeval-drift-gate

Use when a model can change without the code changing. Compares slices, fails the build on the degradation you name, and reports where the recommended threshold moved.

1. Establish the reference. Either name the slices explicitly on one run:

```sh
jeval drift --root . --baseline .jeval/baseline.json
```

or save the measurement you accept as the baseline, which matters when the model string never changes
but its behaviour does:

```sh
jeval drift --root . --save-baseline .jeval/baseline.json
```

2. Put the check in the pipeline with the degradation you are willing to accept, in the units the
   report already uses:

```sh
jeval drift --root . --fail-on ece-increase=0.05
```

3. Read the block as a comparison, not a verdict on the model. A captured run:

```
model changed: jev-1.13.0 -> jev-1.14.0 (Sep 16)
  question    ECE before  ECE after   delta
  department       0.028      0.141  +0.113   FAIL
recommended threshold (department): 0.96 -> 0.98
  at the current 0.96: auto-rate 2% -> 5%
failed ece-increase: department: ECE 0.028 -> 0.141 (+0.113), limit 0.050
```

4. If a cost matrix is in the project, the movement of the recommended line comes with the check,
   and `--fail-on threshold-shift=0.05` fails the build when that line moves by more than 0.05 in
   either direction: the line a reviewer approved is the one the application runs.
   Use `examples/ci/drift.yml` as the starting point for the workflow; jeval itself holds no token
   and never posts to a pull request.

**Verify**

The exit code is the assertion, and it must be the one you asked for:

```sh
jeval drift --root . --fail-on ece-increase=0.05; echo "exit=$?"
```

`exit=1` with a `FAIL` line is a build that caught a real change; `exit=0` with the deltas printed is
a build that looked and found nothing past the line. Capture which of the two you got, and the deltas
beside it, before saying the gate works.

**Do not claim**

- Do not claim the new model is worse. The check reports a measured movement against a threshold you
  chose; naming the cause needs evidence outside it.
- Do not claim the gate ran. A workflow that was configured is not a workflow that failed a build —
  report the observed exit code and the deltas.
- Do not claim a baseline is a contract: it holds measurements, never records, and is safe to commit
  precisely because it contains no user data.

## jeval-handoff

Use when someone wants jeval's job done by an agent instead of learning the CLI. Turns one sentence into a report file, a threshold file and a CI gate, and stops when the files exist.

1. Install, and confirm the command exists. One line, no package manager, no root:

```sh
curl -fsSL https://raw.githubusercontent.com/rlaope/jeval/main/install.sh | sh
jeval --version
```

2. Find where decisions already land. Ask for the file, or the table, that holds the model's answer.
   Do not invent a shape: read three rows and write down what you see. Say which of these you have:

| You have | The skill to load |
| --- | --- |
| A service calling the model now | `jeval-instrument-service` |
| A log file of past decisions | `jeval-labels-harvest`, then `jeval-calibration-audit` |
| Labels somewhere else (tickets, outcomes) | `jeval-labels-harvest` |
| Nothing measured yet and no labels | stop and report that, do not guess |

3. Describe the layout in `.jeval/ingest-map.yaml` and load the records:

```sh
jeval init --root .
jeval ingest decisions.jsonl --root .
jeval report --root . --current 0.8
```

4. Read the report and answer the person's two questions in plain words: how often the model is right
   when it claims a given confidence, and where the line belongs given what a mistake costs. Then
   hand them the files, not a summary of the files.

5. Stop. The job ends when the artifacts exist and their numbers have been read back, not when a
   plan has been described. The artifact each step produces:

| Step | Artifact |
| --- | --- |
| instrument | `.jeval/records.jsonl` growing as the service runs |
| harvest | the same file, with labels attached |
| audit | `report.html` |
| threshold | `thresholds.yaml` |
| gate | a CI job that exits non-zero on drift |

**Verify**

`jeval report --root . --format md` prints the same numbers as the HTML artifact: a labeled-record
count, an ECE with its interval, and the verdict line. Quote those numbers from the output, not from
this skill. Confirm the artifacts exist on disk: `report.html`, and after the threshold step,
`thresholds.yaml`.

**Do not claim**

- Do not claim the model is calibrated, miscalibrated, or fixed. jeval measures what the records
  support and names the uncertainty; the decision is the owner's.
- Do not claim a threshold is deployed. jeval writes a YAML file; the application reads it.
- Do not claim results from `jeval demo`: that log is synthetic and the report says so itself.
- Do not claim a measurement happened when no labeled record was read.

## jeval-instrument-service

Use when a running service has to start producing decision records. Wraps the classifier call your code already makes and appends the human answer when the case closes.

1. Wrap the client where it is built. This does not change the request path: the wrapper observes a
   call the code already makes and appends a line.

```python
from jeval import collect

client = collect.track(
    TicketClassifier(),  # your client, unchanged
    method_names=("classify",),  # the method that answers questions
    source_key=lambda **kw: kw["ticket_id"],  # what a human answer is joined back on
    segment=lambda **kw: {"lang": kw["lang"]},  # request fields you want to compare later
)
```

2. Choose where the records live, in the service's environment:

```sh
JEVAL_ROOT=/var/lib/jeval
```

3. Append the human answer where the case closes, one line, in a file you can ship or copy:

```python
with open("/var/lib/jeval/resolutions.jsonl", "a", encoding="utf-8") as sink:
    sink.write(
        json.dumps(
            {
                "ticket_id": ticket.id,
                "final_department": ticket.final_department,
                "handled_by": "human_review",
            }
        )
        + "\n"
    )
```

4. After real traffic, attach the answers and measure — see `jeval-labels-harvest` for the join:

```sh
jeval ingest /var/lib/jeval/.jeval/records.jsonl --labels /var/lib/jeval/resolutions.jsonl \
  --label-field final_department --label-source human_override \
  --join-on ticket_id --label-question department --root /var/lib/jeval
jeval report --root /var/lib/jeval --current 0.88
```

**Verify**

Do this before trusting any report: after the first request, the counters must show the collector
actually attached.

```python
print(collect.stats())
# {'written': 240, 'dropped': 0, 'unsupported': 0, 'calls': 240, ..., 'no_method_found': 0}
```

`calls` above zero and `no_method_found` at zero is the proof. Then confirm the record file exists at
`$JEVAL_ROOT/.jeval/records.jsonl` and that `jeval report` reads it. Until the first harvest, that
report answers `no labeled records` — expected at this step.

**Do not claim**

- Do not claim jeval sits in the request path, routes, or changed any model's behaviour.
- Do not claim the records are complete: the counters are the only statement about what was written.
- Do not claim segment comparisons are meaningful before enough labeled records exist for the segment
  — the report greys out the ones it cannot support.

## jeval-labels-harvest

Use when calibration cannot be measured because the records carry no labels. Finds the answers the product already produces, joins them on your own key, and keeps silver apart from gold.

1. Ask for the file where a human's answer lands. The usual three, in order of quality:

| The product already has | Where the label is |
| --- | --- |
| Cases a human reviewed after escalation | the human's final answer — gold |
| Auto-handled cases later reversed | the reversal — the model was wrong |
| Refund approvals, rejections, closed tickets | the outcome — a real answer with a date |

2. Write the harvest into the ingest map, with the key both files share:

```yaml
label_from:
  field: resolution.final_department   # dotted paths work
  source: human_override               # human_review | human_override | silver
  join_on: ticket_id                   # the key present in both files
  question: department                 # which question this column answers
```

3. Run the harvest. It rewrites the record file in place and atomically, touching only label fields:

```sh
jeval ingest .jeval/records.jsonl --labels resolutions.jsonl \
  --label-field final_department --label-source human_override \
  --join-on ticket_id --label-question department --root .
```

4. Re-run the audit and check the counts moved the way you expected:

```sh
jeval report --root . --format md
```

**Verify**

The harvest prints both counts, and the record count must not change — a join labels records, it does
not duplicate them:

```
read 151 label rows from resolutions.jsonl
applied 151 labels to 240 records to the 'department' question
```

The report then separates the two kinds: `labeled decisions (gold): 151` and, when some rows came
from a rule rather than a person, `silver labels, kept separate: 13`. If the harvest applied far
fewer labels than the resolution file has rows, the join key is the likely cause.

**Do not claim**

- Do not claim the labels are correct because the join succeeded: matching on a key is not evidence
  that the human's answer judges the model's answer.
- Do not claim a score question was measured: score-type records are excluded from binary accuracy
  and the excluded count is printed.
- Do not claim the measurement is final. Report the labeled count the report gives you.

## jeval-threshold-from-costs

Use when a mistake has a price and the automation line has to be chosen. Turns a cost matrix into thresholds.yaml with an interval, and says when a segment does not deserve its own line.

1. Get the four costs per action from the person who owns the decision — a wrong automation, sending
   it to a human anyway, and a human rejecting something the model got right. Do not invent them:
   every number in the output follows from these, and wrong costs give wrong thresholds.

```yaml
actions:
  - name: auto_billing
    question: department
    when: billing
    cost_false_accept: 45000
    cost_escalate: 3000
    cost_false_reject: 1500
```

2. Measure first. The threshold sweep needs labeled records, not the raw log:

```sh
jeval report --root . --format md
jeval threshold --root . --costs costs.yaml
```

3. Read the output as three claims about one line: the threshold, the interval around it, and what it
   buys. Real output looks like this:

```
auto_billing:   threshold 0.88 (95% CI 0.88-0.88, width 0.00) · auto 38% · 1,877 per case
auto_technical: threshold 0.88 (95% CI 0.88-0.89, width 0.01) · auto 23% · 2,450 per case
wrote thresholds.yaml
```

4. Ask whether one line fits every slice, and accept either answer:

```sh
jeval threshold --root . --costs costs.yaml --by lang
```

5. Hand the file to the application owner. `thresholds.yaml` is the interface; say plainly that the
   application reads it and jeval stays out of the request path.

**Verify**

`thresholds.yaml` exists and names each action with its threshold, and the CLI printed the line
`wrote thresholds.yaml`. Re-run the same command: the numbers must be identical, because the sweep
is deterministic on the same records. If an action is missing from the file, it had too few labeled
records and the run said so — read that line before reporting success.

**Do not claim**

- Do not claim the threshold is deployed. jeval never sits in the request path and holds no token.
- Do not claim a cost saving. The output is a cost per case under stated assumptions, not a forecast.
- Do not claim the costs are right. They are the owner's input, and the recommendation is only as
  sound as they are.
