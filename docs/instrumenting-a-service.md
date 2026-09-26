# Instrumenting a service

How to get a real service's decisions into jeval, attach the human answers that arrive later, and
read the result. Every step below was run; the output shown is the output it produced.

The model here is a stand-in — nothing calls a real vendor. Everything else is real: the wrapper,
the call path, the files and the command line. The whole service is
[`examples/service-quickstart.py`](../examples/service-quickstart.py), so every output below can be
reproduced with `JEVAL_ROOT=/tmp/jeval-service uv run python examples/service-quickstart.py`.
[`docs/library.md`](library.md) is the reference for every argument used here.

## What you need

* A service that calls a classifier and gets back a label with a confidence.
* A place where a human's eventual answer already arrives: a resolved ticket, an approved refund, a
  corrected routing, a closed case.
* Python 3.10 or newer in the service, and nothing else. jeval is not on the request path.

## 1. Install

In the service's environment, and wherever you will run the command line:

```sh
pip install "jeval_cli @ git+https://github.com/rlaope/jeval"
```

That gives the service `from jeval import collect` and the host a `jeval` command. It needs no
server, no account and no network access at run time. (`collect.resolve` and `jeval status` are on
`main` ahead of the next tagged release.)

## 2. Record what the model answered

One import, one line where the client is built. `method_names` names the method on your client that
answers questions — `system_one` on the Jev SDK, `classify` on the stand-in.

```python
from jeval import collect

client = collect.track(
    TicketClassifier(),  # your client, unchanged
    method_names=("classify",),  # the method that answers questions
    source_key=lambda **kw: kw["ticket_id"],  # what a human answer is joined back on
    segment=lambda **kw: {"lang": kw["lang"]},  # request attributes become segment axes
)

answer = client.classify(ticket_id=ticket_id, text=text, lang=lang)  # unchanged call
```

The request path does not change. The wrapper makes no calls of its own, never chooses a model,
never retries, never blocks and never raises: it observes the call your code already makes and
appends a line.

Each record keeps three things. The third one matters most:

| Recorded | Why |
| --- | --- |
| The answer with its probabilities | the measurement itself |
| The model that actually answered (`jev-1.14.0`, not the `jev-latest` you asked for) | the drift anchor |
| The join key (`ticket_id`) | without it, nothing can attach a human answer to this decision later |

## 3. Record what a human settled on

This is the second and last integration point, in the handler your service already has for closing
a ticket:

```python
collect.resolve(source_key=ticket.id, question="department", answer=ticket.final_department)
```

`source=` defaults to `human_review`; pass `human_override` when the person changed the model's
answer, or `silver` for agreement with another model, which is kept apart from accuracy. The answer
goes to `labels.jsonl` beside the records, and every command joins it on read — there is no export
and no ingest step.

If the closing handler lives in a system that cannot import jeval, write the answers to any JSONL
file there and attach them with `jeval ingest --labels` instead (see "When the answers live
elsewhere" below).

## 4. Decide where the records live

```sh
JEVAL_ROOT=/var/lib/jeval      # in the service's environment
```

Records then go to `/var/lib/jeval/.jeval/records.jsonl` and answers to
`/var/lib/jeval/.jeval/labels.jsonl` — the same files `jeval report --root /var/lib/jeval` reads.

| Variable | Effect |
| --- | --- |
| `JEVAL_COLLECT=0` | collection off, immediately, including when a call site passed a path |
| `JEVAL_COLLECT=/var/log/jeval/decisions.jsonl` | a different records file, when the project tree is not writable; answers go beside it |
| `JEVAL_MODEL=jev-1.14.0` | the model string to record when the response does not carry one |

Nothing is written outside that path, and nothing leaves the machine.

## 5. Verify both sides — do not skip this

The default method names belong to the Jev SDK. If your client answers through a method with a
different name, everything looks fine and nothing is recorded: no error, no warning, an empty file.
So check once, in the service:

```python
print(collect.stats())
```

```
{'written': 600, 'dropped': 0, 'unsupported': 0, 'calls': 600, 'already_tracked': 0, 'install_failed': 0, 'unknown_flag_value': 0, 'retrack_ignored': 0, 'sink_not_a_file': 0, 'invalid_value': 0, 'no_method_found': 0, 'labels_written': 480}
```

| Counter | What to do when it is not what you expect |
| --- | --- |
| `calls` is 0 | `no_method_found` will be 1: your client's answering method is not in `method_names` |
| `unsupported` | the response shape was not recognised: check `container`/`keys` |
| `dropped` | a line failed to write; check the sink path and permissions |
| `invalid_value` | a value could not be used, for example an empty answer passed to `resolve` |
| `labels_written` | how many human answers `resolve` recorded |

And on the command line, what reached the files from every worker:

```
$ jeval status --root /tmp/jeval-service
labels: 480 answers from labels.jsonl, 480 applied
decisions: /tmp/jeval-service/.jeval/records.jsonl (600 decisions, last 2026-09-26 01:26Z)
answers:   /tmp/jeval-service/.jeval/labels.jsonl (480 answers)

  model              question             decisions   gold  silver
  jev-1.14.0         department                 600    480       0

ready: 480 gold-labeled decisions; `jeval report` can measure them.
no cost matrix: add costs.yaml to get a threshold; calibration works without one.
```

Until at least 100 gold-labeled decisions have arrived, the last lines say how many more the
verdict needs instead of `ready`. That is the gate, not a failure: a confidence can only be judged
against what actually happened.

## 6. Read the result

```
$ jeval report --root /tmp/jeval-service --current 0.8 --by lang
labels: 480 answers from labels.jsonl, 480 applied

labeled decisions (gold): 480
unlabeled, excluded: 120
ECE 0.083 (95% CI 0.064-0.126)  MCE 0.172  Brier 0.210  n=480
Overconfidence in 0.45-0.52: claimed 0.48, observed 0.31 (95% CI 0.20-0.45, n=48); ECE 0.083.
257 more labels would roughly halve the ECE interval (currently +/-0.031).
report: /tmp/jeval-service/report.html
```

* `--current 0.8` is the threshold your service actually uses. Without it the report says it does not
  know what is deployed rather than inventing a baseline.
* `480` of `600` decisions have a human answer: the rest are excluded and counted, never folded in.
* The diagnosis names the range, both numbers, and the interval around what was observed.
* The last line is the honest limit: 257 more labels would halve the interval.
* `--by lang` breaks the report down by the axis you passed to `segment=`.

## 7. Produce the file your application reads

Put your own numbers in `costs.yaml` first — every recommendation is a consequence of them:

```yaml
actions:
  - name: auto_billing
    question: department
    when: billing
    cost_false_accept: 45000   # auto-routed and wrong
    cost_escalate: 3000        # a human triaged it
    cost_false_reject: 1500    # right, but a human had to see it first
```

```
$ jeval threshold --root /tmp/jeval-service --costs /tmp/jeval-service/costs.yaml --currency KRW
labels: 480 answers from labels.jsonl, 480 applied
costs: /tmp/jeval-service/costs.yaml
auto_billing: threshold 0.94 (95% CI 0.79-0.98, width 0.19) · auto 3% · KRW 3,453 per case
wrote /tmp/jeval-service/thresholds.yaml
your application reads this file; jeval never sits in the request path.
```

With a wrong auto-route costing fifteen times a human triage, the line sits high and only 3% of
billing tickets run alone. The interval, 0.79 to 0.98, is the part to read before shipping it.

Ask whether one line fits everyone:

```
$ jeval threshold --root /tmp/jeval-service --costs /tmp/jeval-service/costs.yaml --currency KRW --by lang
...
auto_billing by lang: global threshold 0.94
  segment            threshold    cost/case (KRW)  vs global      n  verdict
  lang = en               0.92              3,251        -61    221  splitting does not pay: its optimum 0.92 differs from the global 0.94 and the cost change of 61.09 per case is under the 2% bar.
  lang = ko               0.97              3,463       -110    259  split
```

Korean tickets are the ones the stand-in overclaims on, and their own line sits higher; English
tickets would move too, but not by enough to pay for a second line.

## 8. Keep it running

```sh
# whenever you want a fresh picture: the answers are joined on read
jeval status --root /var/lib/jeval
jeval report --root /var/lib/jeval --current 0.94 --costs costs.yaml --currency KRW --format md

# in the deploy pipeline: the model swap that changes calibration, or moves the line, fails the build
jeval drift --root /var/lib/jeval --save-baseline /var/lib/jeval/.jeval/baseline.json
jeval drift --root /var/lib/jeval --fail-on ece-increase=0.05 --fail-on threshold-shift=0.05
```

`jeval drift` exits `1` when a check fails, names the check and its numbers, and prints the movement
of the recommended line beside it. `examples/ci/drift.yml` is a copy-paste workflow; jeval prints the
summary and the workflow posts it, because jeval never holds a token.

## When the answers live elsewhere

If the human answers are stored by another system — a ticketing tool's export, a warehouse table —
write them to a JSONL file and attach them once:

```sh
jeval ingest /var/lib/jeval/.jeval/records.jsonl \
  --labels /var/lib/jeval/resolutions.jsonl \
  --label-field final_department --label-source human_override \
  --join-on ticket_id --label-question department \
  --root /var/lib/jeval
```

This rewrites `records.jsonl` with the labels in place. `--label-question` is not decoration: a join
key is shared by every question of a request, so without it a department answer would also be
written as the *intent* answer.

## When nothing shows up

| Symptom | Cause | Fix |
| --- | --- | --- |
| `jeval status` says nothing was collected | collection is off, or `JEVAL_ROOT` differs between the service and the command | check `collect.target_path()` in the service and `--root` on the command |
| `calls` is 0 after traffic | the wrapper attached to nothing | pass `method_names=(...)` with your client's method |
| `unsupported` is growing | the response shape was not understood | set `container=`/`keys=`, or check the client's return type |
| `labels: ... matched no decision` | the answer's key differs from the decision's `source_key`, or the decision was recorded without one | pass the same key to `track(source_key=...)` and `resolve(source_key=...)` |
| `labels: ... not an answer the question can give` | the answer is outside the record's own probability map | check the question's candidate set |
| `labels: ... kept an existing label` | the decision already had a label; answers never overwrite one | expected when two sources answer the same case |

## What this walkthrough does not do

* It does not sit between your service and the model. Nothing here routes, retries, caches or
  proxies; a failed write is counted and dropped.
* It does not decide the threshold for you. It reports what your numbers imply, with the range the
  sample can support.
* It does not fix the confidence. `jeval calibrate` exports a correction map for your application to
  apply, and declines to export one when the gain does not clear the noise.
* It does not upload anything. The records stay on the machine you put them on.
