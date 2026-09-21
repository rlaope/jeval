---
name: jeval-instrument-service
description: Use when a running service has to start producing decision records. Wraps the classifier call your code already makes and appends the human answer when the case closes.
metadata:
  hermes:
    tags: [jeval, calibration, thresholds]
    category: evaluation
---

# Instrument a service

## Use when

A service calls a classifier today and nothing about those calls is recorded, or the recording is a
log line nobody can measure. Two integration points are needed: one where the classifier answers, one
where the human's answer eventually arrives.

## What to do

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

## Verify

Do this before trusting any report: after the first request, the counters must show the collector
actually attached.

```python
print(collect.stats())
# {'written': 240, 'dropped': 0, 'unsupported': 0, 'calls': 240, ..., 'no_method_found': 0}
```

`calls` above zero and `no_method_found` at zero is the proof. Then confirm the record file exists at
`$JEVAL_ROOT/.jeval/records.jsonl` and that `jeval report` reads it. Until the first harvest, that
report answers `no labeled records` — expected at this step.

## Pitfalls

1. **A wrapper that attached to nothing.** Its default method names belong to the Jev SDK. A client
   whose answering method has any other name is wrapped to zero calls with no error anywhere: the
   file stays empty and everything looks installed. `collect.stats()["calls"]` and the
   `no_method_found` counter are the signal; `method_names=(...)` is the fix.
2. **An unrecognised response shape.** Calls are counted in `unsupported` rather than written. Check
   what the client returns; the container and keys can be named explicitly.
3. **A sink that is not a file.** A directory or `/dev/null` shows up as `sink_not_a_file`; nothing
   is written and nothing raises.
4. **Turning collection off by accident.** `JEVAL_COLLECT=0` disables it immediately and
   `collect.target_path()` then returns nothing. That is the kill switch, not a configuration.
5. **Wrapping a client twice.** The second wrap is counted in `already_tracked` and ignored; the
   first configuration wins.
6. **Expecting the wrapper to fail a request.** It never raises, never retries, never blocks and never
   calls a model: a failed write is counted in `dropped`. Nothing here protects the service's traffic.

## Do not claim

- Do not claim jeval sits in the request path, routes, or changed any model's behaviour.
- Do not claim the records are complete: the counters are the only statement about what was written.
- Do not claim segment comparisons are meaningful before enough labeled records exist for the segment
  — the report greys out the ones it cannot support.
