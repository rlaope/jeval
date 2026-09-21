---
name: jeval-threshold-from-costs
description: Use when a mistake has a price and the automation line has to be chosen. Turns a cost matrix into thresholds.yaml with an interval, and says when a segment does not deserve its own line.
metadata:
  hermes:
    tags: [jeval, calibration, thresholds]
    category: evaluation
---

# Turn costs into a threshold

## Use when

The product automates a decision above some confidence, and nobody can say where that line came
from. The input is a cost matrix written by the owner; the output is a YAML file the application
reads, with the threshold, the share of traffic it automates, and the interval the sample supports.

## What to do

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

## Verify

`thresholds.yaml` exists and names each action with its threshold, and the CLI printed the line
`wrote thresholds.yaml`. Re-run the same command: the numbers must be identical, because the sweep
is deterministic on the same records. If an action is missing from the file, it had too few labeled
records and the run said so — read that line before reporting success.

## Pitfalls

1. **A segment without enough records.** Below the `--min-records` floor (100 by default) the tool
   prints `only 50 gold records, below the 100 a segment needs before a threshold is recommended for
   it` and recommends nothing. That refusal is the correct answer for that segment.
2. **A split that does not pay.** A segment keeps the global line unless its optimum moves more than
   one sweep step **and** adopting it changes cost per case by more than 2%; otherwise the output
   says `splitting does not pay` and names the clause that failed. Do not read that as a failure.
3. **A wide interval read as a precise line.** `(95% CI 0.88-0.89, width 0.01)` on a few hundred
   labels is a narrow claim only because the sweep is discrete; quote the width.
4. **Editing `thresholds.yaml` by hand.** It is generated. Change `costs.yaml`, re-run, and let the
   file follow from the costs.
5. **Treating the threshold as a product decision.** It is arithmetic on the costs the owner gave
   you. If the resulting auto rate shocks them, the costs are what to argue about.

## Do not claim

- Do not claim the threshold is deployed. jeval never sits in the request path and holds no token.
- Do not claim a cost saving. The output is a cost per case under stated assumptions, not a forecast.
- Do not claim the costs are right. They are the owner's input, and the recommendation is only as
  sound as they are.
