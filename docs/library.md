# Using jeval as a library

jeval has two halves that share one directory:

1. **In your service**, the `jeval` Python package records what the model answered and, later,
   what a human settled on. Two lines of code, no pipeline.
2. **On the command line**, `jeval report`, `threshold`, `drift` and the rest read what was
   recorded and measure it.

The library writes `.jeval/records.jsonl` and `.jeval/labels.jsonl` under one root; the command line
reads the same two files from the same root. There is no export step, no server between them, and
nothing leaves the machine.

```
your service                              the command line
─────────────────────────────             ─────────────────────────────
collect.track(client)  ──► records.jsonl ─┐
collect.resolve(...)   ──► labels.jsonl  ─┴─► jeval status / report / threshold / drift
            both under $JEVAL_ROOT/.jeval/
```

Everything on this page was run; the output shown is the output it produced. The `last` time in `jeval status` is when the script ran. The model in the
example is a stand-in — [`examples/service-quickstart.py`](../examples/service-quickstart.py) is the
whole script, with synthetic answers and a synthetic human.

---

## Install

In the service's environment:

```sh
pip install \
  https://github.com/rlaope/jeval/releases/download/v0.2.0/jeval_cli-0.2.0-py3-none-any.whl
```

`collect.resolve` and `jeval status` arrived in v0.2.0; for what is on `main` since, install from the
repository instead: `pip install "jeval_cli @ git+https://github.com/rlaope/jeval"`, pinned with
`@<sha>` for a reproducible deploy. The distribution is `jeval_cli` and the import is `jeval`.
Never `pip install jeval`: that name on PyPI belongs to an unrelated project. The same install puts
the `jeval` command on the PATH, so the service host can also run the command line.

## 1. Record what the model answered: `collect.track`

Wrap the client where it is built. The call sites do not change.

```python
from jeval import collect

client = collect.track(
    TicketClassifier(),  # your client, unchanged
    method_names=("classify",),  # the method that answers questions
    source_key=lambda **kw: kw["ticket_id"],  # what a human answer is joined back on
    segment=lambda **kw: {"lang": kw["lang"]},  # request fields to compare later
)

answer = client.classify(ticket_id="T-1042", text="...", lang="ko")  # unchanged call
```

Every call now appends one record per question to `$JEVAL_ROOT/.jeval/records.jsonl`: the answer,
its probabilities, the model that actually answered, the join key, and the segment.

| Argument | Default | What it is for |
| --- | --- | --- |
| `client` | — | the object whose method answers questions; patched in place and returned |
| `method_names` | `("system_one", "systemOne", "systemone")` | the method(s) to wrap. The defaults are the Jev SDK's; name yours if it differs |
| `source_key` | `None` | a literal, or a function of the call's keyword arguments, returning the key a human answer will carry (ticket id, trace id) |
| `segment` | `None` | a mapping, or a function of the call's keyword arguments, returning request attributes to break results down by |
| `container` | `"answers"` | the response field holding typed answers keyed by question name |
| `keys` | `None` | renames when the response spells fields differently, e.g. `{"question_type": "kind"}` |
| `path` | `None` | a records file to write instead of the default |

The response is expected to carry typed answers keyed by question name — for each question a
`type` (`choice`, `noul` or `score`), the answer, and its `probabilities`. A response it cannot read
is counted in `stats()["unsupported"]` and skipped, never guessed at.

Without a `source_key`, decisions are still recorded, but no human answer can be attached to them.

### Record by hand instead: `collect.record`

When there is no client object to wrap — an HTTP call, a batch job — record the decision directly:

```python
collect.record(
    question_key="department",
    prediction="billing",
    probabilities={"billing": 0.82, "technical": 0.18},
    model="jev-1.14.0",
    source_key="T-1042",
    segment={"lang": "ko"},
)
```

It takes the same fields a record has and returns whether the line was written.

## 2. Record what a human settled on: `collect.resolve`

Call it where the truth arrives — an agent picks the final department for an escalated ticket, a
refund is approved or reversed, an auto-handled case is reopened:

```python
collect.resolve(source_key=ticket.id, question="department", answer=ticket.final_department)
```

| Argument | Default | What it is for |
| --- | --- | --- |
| `source_key` | — | the key the decision was recorded with |
| `question` | — | the question this answers; one request can answer several |
| `answer` | — | the answer a human settled on |
| `source` | `"human_review"` | `human_review` (a person checked it), `human_override` (a person changed it), or `silver` (agreement with another model, kept apart from accuracy) |
| `path` | `None` | a records file whose directory the answer goes beside |

The answer lands in `labels.jsonl` beside the records file. Every command joins it onto the matching
decision when it reads the records, by `(source_key, question)`:

- a decision that already carries a label keeps it — a later `silver` answer never replaces a human
  review;
- an answer the question could not have produced (a department that is not in the record's own
  probability map, a yes/no answer that is neither) is refused;
- when one case is answered twice, the later line in the file wins — except that a `silver` answer
  never replaces a human one. Later means appended later; with several workers that is the order
  the lines reached the file;
- nothing is rewritten on disk, so reading twice gives the same result. Even `jeval label --apply`,
  which writes records, writes them without the joined answers.

The commands print what happened to every answer on stderr, for example
`labels: 480 answers from labels.jsonl, 480 applied`.

## 3. Choose where it goes

| Variable | Effect |
| --- | --- |
| `JEVAL_ROOT=/var/lib/jeval` | records and answers go to `/var/lib/jeval/.jeval/`; `jeval report --root /var/lib/jeval` reads them |
| `JEVAL_COLLECT=0` | both functions write nothing, immediately — even when a call passed `path` |
| `JEVAL_COLLECT=/var/log/jeval/records.jsonl` | a different records file, with answers beside it as `labels.jsonl`. The command line joins answers only from `<root>/.jeval/labels.jsonl`, so use this when the records are shipped elsewhere and ingested; for the two-line loop, use `JEVAL_ROOT` |
| `JEVAL_MODEL=jev-1.14.0` | the model string to record when a response does not carry one |

## 4. Check it is working: `collect.stats()` and `jeval status`

The first run of a new integration is where silent failures hide, so check both sides once.

In the service, the counters for this process:

```python
print(collect.stats())
```

```
{'written': 600, 'dropped': 0, 'unsupported': 0, 'calls': 600, 'already_tracked': 0, 'install_failed': 0, 'unknown_flag_value': 0, 'retrack_ignored': 0, 'sink_not_a_file': 0, 'invalid_value': 0, 'no_method_found': 0, 'labels_written': 480}
```

| Counter | What to do when it is not what you expect |
| --- | --- |
| `calls` is 0 | `track` attached to nothing: `no_method_found` will be 1; name the method in `method_names` |
| `unsupported` | the response shape was not recognised: check `container` and `keys` |
| `dropped` | a line failed to write: check the path and permissions |
| `invalid_value` | a value could not be used: a negative latency, an empty or over-long (512 characters) answer passed to `resolve`, a `ts` that is not a datetime, an unknown `source` |
| `labels_written` | how many human answers `resolve` recorded |

On the command line, what has accumulated across every process:

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

Below 100 gold-labeled decisions it says how many more the verdict needs instead of `ready`.

## 5. Measure it

```
$ jeval report --root /tmp/jeval-service
labels: 480 answers from labels.jsonl, 480 applied

labeled decisions (gold): 480
unlabeled, excluded: 120
ECE 0.083 (95% CI 0.064-0.126)  MCE 0.172  Brier 0.210  n=480
Overconfidence in 0.45-0.52: claimed 0.48, observed 0.31 (95% CI 0.20-0.45, n=48); ECE 0.083.
257 more labels would roughly halve the ECE interval (currently +/-0.031).
report: /tmp/jeval-service/report.html
```

From here the rest of the README applies unchanged: add `costs.yaml` for a threshold, run
`jeval drift --fail-on ...` in CI when the model changes.

---

## What the library promises

- **It never breaks the service.** Neither function raises. A failure is counted in `stats()` and
  the call returns `False`; a wrapped client that cannot be patched comes back working and
  unwrapped.
- **It never calls a model.** `track` observes a call your code already makes; it never picks a
  model, retries or blocks. It imports no vendor SDK.
- **It never sends anything anywhere.** It appends lines to local files. There is no network call,
  no background thread and no account.
- **It is off with one variable.** `JEVAL_COLLECT=0`, including for call sites that passed a path.

## Limits worth knowing

- **Several worker processes** writing to the same files: each record is one short line (a few
  hundred bytes) appended with a single write, which in practice keeps lines whole on a local
  filesystem. It is not a lock, and a network filesystem makes no such promise. If a torn line ever
  appears, `jeval` refuses the file and names the line rather than reading around it. When in doubt,
  give each worker or host its own `JEVAL_ROOT` and concatenate the files before reading.
- **The files grow without bound.** Rotate them the way you rotate logs; `jeval` reads whatever is
  in `records.jsonl` and `labels.jsonl` when it runs.
- **`collect.stats()` is per process.** It counts what this process did. `jeval status` counts what
  reached the files.
- **Answers for decisions recorded without a `source_key`** cannot be joined; they are counted as
  `matched no decision`.
- **A log you already have** does not need the library: `jeval ingest <log>` reads it, and
  `jeval ingest --labels <resolutions> --join-on <key>` attaches answers you stored elsewhere.
