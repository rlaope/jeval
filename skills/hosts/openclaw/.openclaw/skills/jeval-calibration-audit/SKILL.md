---
name: jeval-calibration-audit
description: Use when a classifier returns a confidence and nobody has checked what it is worth.
---

# Audit the calibration

## Use when

A model answers with a probability and the product treats that probability as a fact: routing on
`confidence > 0.8`, showing "97% sure" to a user, or automating everything above a line nobody has
measured. The input is a log of decisions; the output is one HTML file that says where the
confidence can be taken at face value and where it cannot.

## What to do

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

## Verify

The terminal summary and the file agree: the same labeled count, the same ECE and interval, the same
verdict line. `report.html` exists, opens offline, and carries a `generated_at` timestamp. If the
numbers differ between the summary and the artifact, stop and report the disagreement rather than
choosing one.

## Pitfalls

1. **Recomputing ECE yourself and comparing the two numbers.** Equal-width binning (`pandas.cut`)
   and jeval's equal-size binning disagree: on the demo records, ECE 0.113 against 0.076, because
   equal-width binning put 38 labels in one bin while the others held 17 and 18. If you compute your
   own, say which binning produced it.
2. **`--bins 1`.** Refused on purpose: one bin averages every decision together, so ECE collapses
   toward zero and the report reads as "confidence is trustworthy" whatever the data says.
3. **Reading a wide interval as a small number.** `ECE 0.142 (95% CI 0.107-0.180)` on 151 labeled
   records is a wide claim. Say the width, or say the sample cannot support a narrow one.
4. **Pooling questions.** A request that answers "is this a refund?" and "how annoyed is this
   customer?" can be reliable on one and useless on the other; the report separates them per question
   and per segment. Do not average them into a single figure.
5. **Treating silver labels as truth.** A harvest from an auto-processed case is an agreement rate,
   not an accuracy. The report counts silver separately and so must your summary.

## Do not claim

- Do not claim the tool proved the model is miscalibrated: ECE is an estimate with an interval, and
  the interval is part of the answer.
- Do not claim a fix. `jeval calibrate` exports a correction map for the application to apply, is
  measured on held-out records, and exports nothing when the gain does not clear the noise.
- Do not claim accuracy for `score` questions: they are measured as MAE, RMSE and rank agreement, and
  are never folded into binary accuracy.
