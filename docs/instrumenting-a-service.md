# Instrumenting a service

How to get a real service's decisions into jeval, attach the human answers that arrive later, and
read the result. Every step below was run; the output shown is the output it produced.

The model here is a stand-in — nothing calls a real vendor. Everything else is real: the wrapper,
the call path, the files, the command line, and the version of jeval the installer gave you.

## What you need

* A service that calls a classifier and gets back a label with a confidence.
* A place where a human's eventual answer already arrives: a resolved ticket, an approved refund, a
  corrected routing, a closed case.
* Python 3.10 or newer in the service, and nothing else. jeval is not on the request path.

## 1. Install the command

```sh
curl -fsSL https://raw.githubusercontent.com/rlaope/jeval/main/install.sh | sh
```

That puts `jeval` on your PATH without uv, pipx or root. It needs no PyPI release, no server, no
account and no network access at run time. Re-running the same command upgrades it;
`sh install.sh uninstall` removes it.

## 2. Add the wrapper where the client is built

One import, one line. `method_names` names the method on your client that answers questions —
`system_one` on the Jev SDK, something else on your own wrapper around a gateway or a local model.

```python
# service.py
from jeval import collect

client = collect.track(
    TicketClassifier(),  # your client, unchanged
    method_names=("classify",),  # the method that answers questions
    source_key=lambda **kw: kw["ticket_id"],  # what a human answer is joined back on
    segment=lambda **kw: {"lang": kw["lang"]},  # request attributes become segment axes
)


def handle_ticket(ticket_id: str, text: str, **request) -> str:
    answer = client.classify(ticket_id=ticket_id, text=text, **request)  # unchanged call
    return answer["answers"]["department"]["choice"]
```

The request path does not change. The wrapper makes no calls of its own, never chooses a model,
never retries, never blocks and never raises: it observes the call your code already makes and
appends a line.

Each record keeps three things. The third one matters most:

| Recorded | Why |
| --- | --- |
| The answer with its probabilities | the measurement itself |
| The model that actually answered (`jev-1.13.0`, not the `jev-latest` you asked for) | the drift anchor |
| The join key (`ticket_id`) | without it, nothing can attach a human answer to this decision later |

## 3. Add the outcome line where the ticket closes

This is the second and last integration point. It is the event your service already has.

```python
# the ticket-closing handler
import json

with open("/var/lib/jeval/resolutions.jsonl", "a", encoding="utf-8") as sink:
    sink.write(
        json.dumps(
            {
                "ticket_id": ticket.id,
                "final_department": ticket.final_department,  # the human's answer
                "handled_by": "human_review",
            }
        )
        + "\n"
    )
```

## 4. Decide where the records live

```sh
JEVAL_ROOT=/var/lib/jeval      # in the service's environment
```

That writes `/var/lib/jeval/.jeval/records.jsonl` — the same file `jeval report --root /var/lib/jeval`
opens, so there is no export step and no second copy to keep in sync. Two more environment variables
are worth knowing:

| Variable | Effect |
| --- | --- |
| `JEVAL_COLLECT=0` | collection off, immediately, including when a call site passed a path |
| `JEVAL_COLLECT=/var/log/jeval/decisions.jsonl` | a different sink, when the project tree is not writable |
| `JEVAL_MODEL=jev-1.13.0` | the model string to record when the response does not carry one |

Nothing is written outside that path, and nothing leaves the machine.

## 5. Verify the wrapper attached — do not skip this

The default method names belong to the Jev SDK. If your client answers through a method with a
different name, everything looks fine and nothing is recorded: no error, no warning, an empty file.
That is what happened on the first run of this walkthrough — `written: 0, calls: 0`.

```python
print(collect.stats())
```

```
{'written': 240, 'dropped': 0, 'unsupported': 0, 'calls': 240, 'already_tracked': 0,
 'install_failed': 0, 'unknown_flag_value': 0, 'retrack_ignored': 0, 'sink_not_a_file': 0,
 'invalid_value': 0, 'no_method_found': 0}
```

| Counter | What to do when it is not zero |
| --- | --- |
| `no_method_found` | your client's answering method is not in `method_names`; name it |
| `unsupported` | the response shape was not recognised: check `container`/`keys`, or print `collect.dump(payloads)` |
| `dropped` | a record failed to write; check the sink path and permissions |
| `invalid_value` | a value could not be used (for example a segment that is not a mapping) |
| `sink_not_a_file` | the sink is not a regular file (`/dev/null`, a directory) |
| `already_tracked` | a client was wrapped twice; the first configuration wins |

With `JEVAL_COLLECT=0`, `calls` stays at `0` and `collect.target_path()` returns `None` — no file is
created at all.

## 6. The first report measures nothing, and says so

```sh
jeval report --root /var/lib/jeval
```

```
no labeled records: nothing to measure. Labels first, thresholds later.
```

This is the gate, not a failure. A confidence can only be judged against what actually happened, and
until a human's answer is attached, no row can be judged.

## 7. Attach the human answers

The decisions came from the service; the answers come from the ticketing tool. Join them on the key
you recorded:

```sh
jeval ingest /var/lib/jeval/.jeval/records.jsonl \
  --labels /var/lib/jeval/resolutions.jsonl \
  --label-field final_department --label-source human_override \
  --join-on ticket_id --label-question department \
  --root /var/lib/jeval
```

```
read 151 label rows from resolutions.jsonl
applied 151 labels to 240 records to the 'department' question
rewrote /var/lib/jeval/.jeval/records.jsonl in place
```

Two details worth noticing. The record count stays at 240 — the join labels the records, it does not
duplicate them. And `--label-question` is not decoration: a join key is shared by every question of a
request, so without it a department answer would also be written as the *intent* answer. A label the
record's own question could not have produced is refused, counted and named, because a wrong label is
worse than a missing one.

## 8. Read the result

```sh
jeval report --root /var/lib/jeval --current 0.8 --by lang
```

```
labeled decisions (gold): 151
unlabeled, excluded: 89
ECE 0.142 (95% CI 0.107-0.180)  MCE 0.622  Brier 0.074  n=151
Overconfidence in 0.62-0.88: claimed 0.76, observed 0.14 (95% CI 0.05-0.33, n=22); ECE 0.142.
176 more labels would roughly halve the ECE interval (currently +/-0.037).
report: /var/lib/jeval/report.html
```

* `--current 0.8` is the threshold your service actually uses. Without it the report says it does not
  know what is deployed rather than inventing a baseline.
* `151` of `240` rows are labeled: the rest are excluded and counted, never folded in.
* The diagnosis names the bin, both numbers, and the interval. That interval is wide because 22
  records is 22 records.
* The last line is the honest limit: 176 more labels would halve the interval.

The artifact's segments section carries the axes you passed to `segment=`:

```
ECE by segment · bar scale 0 to 0.13 · grey = too few samples
  lang = ko  0.13  n=101
  lang = en  0.09  n=50
```

## 9. Produce the file your application reads

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

```sh
jeval threshold --root /var/lib/jeval --costs costs.yaml
```

```
auto_billing:   threshold 0.88 (95% CI 0.88-0.88, width 0.00) · auto 38% · 1,877 per case
auto_technical: threshold 0.88 (95% CI 0.88-0.89, width 0.01) · auto 23% · 2,450 per case
auto_other:     threshold 0.89 (95% CI 0.88-0.89, width 0.01) · auto 25% · 2,311 per case
wrote /var/lib/jeval/thresholds.yaml
your application reads this file; jeval never sits in the request path.
```

Ask whether one line fits everyone:

```sh
jeval threshold --root /var/lib/jeval --costs costs.yaml --by lang
```

```
auto_billing by lang: global threshold 0.88
  segment            threshold  cost/case  vs global      n  verdict
  lang = en                  —          —          —     50  only 50 gold records, below the 100 a
                                                             segment needs before a threshold is
                                                             recommended for it
  lang = ko               0.88   1,975.25       0.00    101  splitting does not pay: its optimum 0.88
                                                             is within one sweep step (0.01) of the
                                                             global 0.88
```

Both answers are refusals, and both are useful: one segment does not have enough labeled records for
a line of its own (50 of the 100 it needs), and the other does not need one — `lang = ko` stays on
the global 0.88. The verdict column in that block runs past the page width, so the numbers it
carries are repeated here.

## 10. Keep it running

```sh
# daily: attach the answers that arrived, refresh the report
jeval ingest /var/lib/jeval/.jeval/records.jsonl \
  --labels /var/lib/jeval/resolutions.jsonl \
  --label-field final_department --label-source human_override \
  --join-on ticket_id --label-question department --root /var/lib/jeval
jeval report --root /var/lib/jeval --current 0.88 --costs costs.yaml --format md

# in the deploy pipeline: the model swap that changes calibration is a failed build
jeval drift --root /var/lib/jeval --save-baseline /var/lib/jeval/.jeval/baseline.json
jeval drift --root /var/lib/jeval --fail-on ece-increase=0.05
```

`jeval drift` exits `1` when the calibration degrades past the threshold you set, and prints the
movement of the recommended line beside it, so the failure says what to do and not merely that
something changed. `examples/ci/drift.yml` is a copy-paste workflow; jeval prints the summary and the
workflow posts it, because jeval never holds a token.

## When nothing shows up

| Symptom | Cause | Fix |
| --- | --- | --- |
| `records.jsonl` is missing | collection is off, or the sink path is not writable | check `collect.target_path()` and `collect.stats()` |
| `calls` is 0 after traffic | the wrapper attached to nothing | pass `method_names=(...)` with your client's method |
| `unsupported` is growing | the response shape was not understood | set `container=`/`keys=`, or check the client's return type |
| `no labeled records` | no human answers attached yet | run the join in step 7 |
| the join applies few labels | the join key differs between the two files | `--join-on` must name a field present in both |
| a label was refused | it is not an answer that record could have produced | check the question's candidate set, or `--allow-unlisted-labels` |

## What this walkthrough does not do

* It does not sit between your service and the model. Nothing here routes, retries, caches or
  proxies; a failed write is counted and dropped.
* It does not decide the threshold for you. It reports what your numbers imply, with the range the
  sample can support.
* It does not fix the confidence. `jeval calibrate` exports a correction map for your application to
  apply, and declines to export one when the gain does not clear the noise.
* It does not upload anything. The records stay on the machine you put them on.
