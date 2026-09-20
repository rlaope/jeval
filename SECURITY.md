# Security Policy

## Reporting a vulnerability

Do not open a public issue. Use GitHub's private reporting flow:
**Security → Report a vulnerability** on
`https://github.com/rlaope/jeval/security/advisories/new`.

Please include:

- the affected version or commit,
- a minimal reproduction (a JSONL snippet plus the command is usually enough),
- the impact you believe it has,
- whether the issue can be triggered by untrusted input.

You can expect an acknowledgement within seven days. If a fix is warranted, it is released as a
patch version with an advisory that credits you unless you prefer otherwise.

## What jeval does with your data

jeval is a local CLI. It makes no network calls of its own, has no telemetry, no accounts, and
no server component. Everything it reads and writes stays on the machine that runs it:

- `.jeval/records.jsonl` holds decision records — treat it as production data, because it often
  contains user text, labels, and model identifiers.
- `costs.yaml` and `thresholds.yaml` hold business figures: what a wrong automation costs, what
  an escalation costs.
- `report.html` is a self-contained file with no external assets, but it *does* contain your
  data. It is ignored by `.gitignore` for that reason.

`.gitignore` excludes all of these paths by default. If you find a path through which user
records, cost figures, or generated reports can be committed to a repository, that is a security
report — it is exactly the class of leak this project must not have.

## Scope

In scope: data leaks from the paths above, unsafe deserialization of ingest input, resource
exhaustion from a crafted record file, incorrect handling of untrusted field values in the HTML
report.

Out of scope: the accuracy, calibration, or honesty of any model that jeval measures; results
produced from deliberately falsified input data; and anything requiring an attacker to already
control the machine running jeval.
