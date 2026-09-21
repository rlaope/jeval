---
name: jeval-handoff
description: Use when someone wants jeval's job done by an agent instead of learning the CLI. Turns one sentence into a report file, a threshold file and a CI gate, and stops when the files exist.
metadata:
  hermes:
    tags: [jeval, calibration, thresholds]
    category: evaluation
---

# Hand jeval to an agent

## Use when

The person has a classifier that returns a label with a confidence, wants to know whether that
confidence is worth anything and where the human line belongs, and does not want to learn a CLI.
This is the entry skill: read it first, then load the skill for the step you are on.

## What to do

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

## Verify

`jeval report --root . --format md` prints the same numbers as the HTML artifact: a labeled-record
count, an ECE with its interval, and the verdict line. Quote those numbers from the output, not from
this skill. Confirm the artifacts exist on disk: `report.html`, and after the threshold step,
`thresholds.yaml`.

## Pitfalls

1. **Measuring before labeling.** With no labels the report prints `no labeled records: nothing to
   measure. Labels first, thresholds later.` That is the tool refusing to invent a number, not a
   broken install. Go to `jeval-labels-harvest`.
2. **Comparing against nothing.** Without `--current`, or a `thresholds.yaml` in the project, the
   report says it does not know what is deployed instead of comparing the recommendation with
   itself. Ask the person what threshold the product uses today.
3. **Quoting ECE without its interval.** `ECE 0.142 (95% CI 0.107-0.180)` is the honest form; `0.14`
   alone claims a precision the sample does not have.
4. **Handing back a summary.** The deliverable is a file the person can open. A chat message that
   describes what the report would say is not the artifact.

## Do not claim

- Do not claim the model is calibrated, miscalibrated, or fixed. jeval measures what the records
  support and names the uncertainty; the decision is the owner's.
- Do not claim a threshold is deployed. jeval writes a YAML file; the application reads it.
- Do not claim results from `jeval demo`: that log is synthetic and the report says so itself.
- Do not claim a measurement happened when no labeled record was read.
