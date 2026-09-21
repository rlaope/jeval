---
name: jeval-drift-gate
description: Use when a model can change without the code changing. Compares slices, fails the build on the degradation you name, and reports where the recommended threshold moved.
---

# Gate a model change

## Use when

The provider swaps the model behind an endpoint, two model versions are being compared, or a period
before a change has to be held against a period after it. A notebook measured once; this is what
notices that the thing measured is no longer the thing running.

## What to do

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
```

4. If a cost matrix is in the project, the movement of the recommended line comes with the check.
   Use `examples/ci/drift.yml` as the starting point for the workflow; jeval itself holds no token
   and never posts to a pull request.

## Verify

The exit code is the assertion, and it must be the one you asked for:

```sh
jeval drift --root . --fail-on ece-increase=0.05; echo "exit=$?"
```

`exit=1` with a `FAIL` line is a build that caught a real change; `exit=0` with the deltas printed is
a build that looked and found nothing past the line. Capture which of the two you got, and the deltas
beside it, before saying the gate works.

## Pitfalls

1. **Swallowing the failure.** The gate is only real if the step fails: `|| true`, a non-failing shell
   wrapper, or a step marked `continue-on-error` turns the check into decoration.
2. **Comparing unrelated slices.** Two periods with different mixes of questions or traffic produce a
   delta caused by the mix, not the model. The drift block separates a calibration change, which
   moves the line, from a class-mix change, which moves the share of traffic that gets automated.
3. **A slice with too few records.** A comparison over a handful of labeled records is noise; the
   tool reports the counts it used. Read them before reporting a regression.
4. **Waiting for the model string to change.** `jev-latest` and friends point at different models over
   time. A saved baseline catches the change that the version string hides.
5. **Assuming costs exist.** Without a cost matrix the block says `recommended threshold: not
   available (no cost matrix was applied)` instead of inventing a number. That is not a failure.

## Do not claim

- Do not claim the new model is worse. The check reports a measured movement against a threshold you
  chose; naming the cause needs evidence outside it.
- Do not claim the gate ran. A workflow that was configured is not a workflow that failed a build —
  report the observed exit code and the deltas.
- Do not claim a baseline is a contract: it holds measurements, never records, and is safe to commit
  precisely because it contains no user data.
