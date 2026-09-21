---
name: jeval-labels-harvest
description: Use when calibration cannot be measured because the records carry no labels. Finds the answers the product already produces, joins them on your own key, and keeps silver apart from gold.
---

# Harvest the labels you already have

## Use when

The decision log exists but every row is unjudged, so `jeval report` answers `no labeled records:
nothing to measure.` That is the gate, not a bug: a confidence can only be judged against what
actually happened, and the answers usually already exist somewhere in the product.

## What to do

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

## Verify

The harvest prints both counts, and the record count must not change — a join labels records, it does
not duplicate them:

```
read 151 label rows from resolutions.jsonl
applied 151 labels to 240 records to the 'department' question
```

The report then separates the two kinds: `labeled decisions (gold): 151` and, when some rows came
from a rule rather than a person, `silver labels, kept separate: 13`. If the harvest applied far
fewer labels than the resolution file has rows, the join key is the likely cause.

## Pitfalls

1. **Leaving out `--label-question`.** Every question of a request shares the same join key, so
   without it a department answer is also written as the *intent* answer. This is the difference
   between a label and a wrong label.
2. **A join key that is not in both files.** The command reports how many rows it read and how many
   it applied. A small second number against a large first one means the keys do not match.
3. **Assuming a refusal is a bug.** A label the record's own question could not have produced is
   refused, counted and named: a wrong label is worse than a missing one. Read the refusal line
   before reaching for `--overwrite`.
4. **Overwriting good labels.** A harvest never replaces an existing label unless `--overwrite` is
   passed. On a second run with corrected resolutions that flag is deliberate, not routine.
5. **Counting silver as accuracy.** A label with no `label_source` counts as silver, and results from
   silver labels are an agreement rate, not an accuracy. The report says so; your summary must too.
6. **A column that serves two fields.** One raw column may feed only one ingest field. An application
   that writes its prediction into a column named `label` must not have that column read as ground
   truth as well — check the map if a question suddenly looks perfectly accurate.

## Do not claim

- Do not claim the labels are correct because the join succeeded: matching on a key is not evidence
  that the human's answer judges the model's answer.
- Do not claim a score question was measured: score-type records are excluded from binary accuracy
  and the excluded count is printed.
- Do not claim the measurement is final. Report the labeled count the report gives you.
