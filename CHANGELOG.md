# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- A chart's shaded band was a fixed light colour, so in a dark report it stayed near-white:
  the region below the reliability diagonal printed as a glowing block against the page. The
  shade is a theme variable now, with its own value for light, dark and print, and the
  reliability chart no longer carries a private one. Two golden SVGs changed by one attribute
  each (the shaded rectangle gained its class).

### Added
- An agent skill pack. Six skills — hand the job to an agent, audit the calibration, turn a cost
  matrix into a threshold, harvest the labels you already have, instrument a running service, gate a
  model change in CI — live as canonical markdown in `skills/`, and `tools/export_skills.py` renders
  them into the layout each host reads: `skills/hosts/claude-code`, `hermes`, `cursor`, and one
  `AGENTS.md` digest for hosts that read a single instruction file. The generated tree is committed
  and a test fails when it stops matching its sources.
- `install-skills.sh`: installs the pack with `--host`, `--project` or `--global`, and writes a
  marker into every file it creates, so `--uninstall` removes exactly what it added. An existing
  `AGENTS.md` is appended to, never replaced, unless `--force` says otherwise.
- `tests/test_skills_pack.py`: the pack's three laws — one canonical source whose generated copies
  are byte-identical, every `jeval` command inside a skill checked against the real CLI, and an
  install that is reversible without touching a file that is not ours. The release workflow verifies
  the pack is current at tag time and attaches `jeval-skills.tar.gz`, whose stable asset name is
  what makes `releases/latest/download/jeval-skills.tar.gz` resolve.
- `jeval ingest --labels`: free-label harvest. Human answers from a resolution log are joined onto
  existing records by your own key, rewritten atomically, and never overwrite an existing label
  unless `--overwrite` is passed. A label the record's own question cannot produce is refused and
  named rather than written.
- `jeval plan`: additional labels needed per question and segment for a target interval width,
  with an explicit refusal below 200 labels and the scaling assumption printed with every number.
- `jeval label`: active-learning queue (threshold band, flat probability distributions, sparse
  confidence bins) with a CSV sheet to fill in and `--apply` to write answers back.
- `jeval calibrate`: temperature scaling and isotonic regression fitted on your log, exported as a
  YAML map for your application to apply. The reported gain is cross-validated and a map is only
  exported when it beats the sampling noise of the log; otherwise the command exports nothing and
  says why.
- `jeval threshold --by <segment>`: per-segment threshold sweep that answers whether splitting pays,
  with the reason when it does not.
- `jeval drift` reports the cost-driven threshold movement of each compared slice when a cost matrix
  is present (`recommended threshold (department): 0.94 -> 1.00`, `auto-rate 11% -> 58%`).
- `jeval report`: a score-type section (MAE, RMSE, rank correlation, level table) — score records
  are measured as error and rank agreement, never as binary accuracy.
- `tests/test_documented_features.py`: the README and the command surface are now checked against
  each other in both directions, so a documented-but-unimplemented feature fails the suite.

### Added
- `jeval.collect`: a provider-neutral recorder. `track(client, source_key=...)` wraps the client you
  already use so every answered call is written to `.jeval/records.jsonl` with its distribution, the
  model that actually answered, the join key, the token count and the latency. Best effort by
  design: nothing raises, `JEVAL_COLLECT=0` switches it off, `JEVAL_COLLECT=<path>` moves the file.
- `jeval ingest --preset jev-native`: reads a decision API's own response log without reshaping it —
  answers keyed by question name, `choice`/`noul`/`score`, probabilities, confidence and usage.
  `--list-presets`, `--response-field` and `--source-key-field` cover a log that nests differently.
- `questions_field` accepts a container keyed by question name and a dotted path, which is the shape
  a decision API returns (`{"response": {"answers": {...}}}`) rather than a list of questions.

### Removed

- Publishing to PyPI. The name `jeval` on PyPI belongs to an unrelated project, and the free
  alternative `jeval-cli` is not one anyone would guess, so the package index bought nothing that a
  release asset does not. Releases are GitHub release assets: a tag builds the wheel and the sdist,
  attaches them, and reads the asset list back. Install with `uvx --from git+<repo>` or the pinned
  wheel URL. The distribution keeps the name `jeval-cli` because the wheel and its metadata need
  one, and the reported version is read from it.

### Fixed — second pass over the remaining QA findings

Observability and degenerate inputs:

- A token count that arrives as a string (`"input_tokens": "312"`) no longer vanishes: numeric
  strings are accepted, and a value that is not a count is counted as dropped.
- `record()` refuses a negative latency or cost and a boolean confidence instead of storing them,
  and a sink that is not a regular file (`/dev/null`) is no longer counted as a written record.
- A call that raises is counted as an attempt: the counter moved to the wrapper, before the call
  it wraps can raise, so `stats()` can tell "raised" from "never called".
- A second `track()` with a path or key is counted in `retrack_ignored`, because the first install
  wins and a silently dropped path looks like a log that went missing.
- Each preset skip cause now has its own message; one sentence previously told six different
  problems that the response object was missing, which was false for five of them.
- Harvest output reports label rows that carried no join key, and no longer claims the records
  file was rewritten when nothing was applied.

Degenerate inputs and honest defaults:

- Every distinct preset skip reason is reported (missing response, missing container, non-object
  container, empty container, unusable answers).
- `--by` on an axis the log does not carry is named instead of rendering a segment called
  `<axis> = unknown`, which invented a breakdown that was never measured. Duplicate axes are
  deduplicated.
- A blank `--currency` means USD rather than a leading space before every amount.

Engine edges:

- The label projection now fits the scaling **exponent** as well as the constant, from the observed
  interval width at n, n/2 and n/4, and reports the worst deviation of that fit. A fixed `a = 0.5`
  was optimistic: measured k drifted 1.32 -> 1.52 between n=200 and n=3200, so projections
  under-counted the labels a target needs.
- A flat cost curve (every threshold costs the same) reports no interval instead of `[1.0, 1.0]`,
  and the CLI says the sweep cannot recommend a threshold instead of quoting a tie-break as a
  finding. A narrow-but-real interval is left alone: every resample agreeing is stability, not a
  claim of exactness.
- `sweep_by_segment(min_records=0)` clamps to the module floor as its docstring promises, instead of
  raising.
- Two score pairs are no longer a measurement: a rank correlation from two points is always ±1.

Report and CLI:

- Every string in the report escapes through one boundary that also replaces invisible direction
  controls with a visible marker, so a label cannot reverse the text a reader sees.
- The report prints when it was generated; it printed an empty timestamp field.
- `--monthly -5` and `--current 1.5` are refused with the reason instead of rendering negative costs
  and quoting an impossible threshold as the deployed one.
- The segment table uses the same money format as the impact table.
- Dead markup is gone: unused nav CSS, an unused payload builder, and the heatmap moved inside its
  figure.
- `jeval label` on a project with no records says to ingest first instead of claiming every record
  already has a label; every command now loads records through one helper, so a missing records file
  is one readable line rather than an uncaught `FileNotFoundError`.
- `jeval init` over a file named `.jeval` explains the conflict instead of raising
  `FileExistsError`.

### Fixed — third pass

- A flat column can be a segment. A log that carries `lang: ko` as a column had no way to reach the
  documented `--by lang` sweep: the mapping renamed fields but could not build the object the schema
  wants, so every row was skipped. A non-object value mapped onto `segment` now becomes
  `{column_name: value}`, and an object column is still mapped as an object.
- One column is not two fields. A field the map does not name fell back to a raw column of the same
  name, so an application that logs its prediction in a column called `label` fed that column to the
  schema's `label` as well: a ground truth equal to the prediction, every record correct, silver
  labeled, out of nothing. The report refused to measure it, but the records were poisoned and the
  silver-slice sections counted them. A column already assigned to another field is no longer read
  implicitly; an explicit mapping is unaffected.
- A skipped row says what to change. The message was a pydantic dump
  (`2 validation errors for DecisionRecord ... visit https://errors.pydantic.dev/...`) that named
  neither the field nor the file, and a missing `model` or a text confidence read as "jeval is
  broken". Each cause now names the field, what arrived, and whether to fix `field_map` or
  `defaults`.

### Fixed
Four adversarial QA passes over the whole tool found and closed the following. Release-blocking
first.

Silent data loss and host-service risk:

- A decision-API response without a `model` field dropped **every** answer (the record failed
  validation and was counted as dropped). The model now falls back to `JEVAL_MODEL`, then `unknown`.
- `track()` raised out of the module for clients whose attributes cannot be set (pydantic models,
  `__slots__`, frozen dataclasses, raising properties). It now leaves the client working, counts the
  refusal in `stats()["install_failed"]`, and verifies the install by readback.
- `JEVAL_COLLECT=0` did not stop a call site that passed `path=`, and an unrecognised value such as
  `JEVAL_COLLECT=enabled` created a file named `enabled` in the working directory. The off switch now
  wins over everything, and only a value that looks like a path is used as one.
- `track()` called twice recorded every call twice, doubling the weight of the entire log. It is now
  idempotent.

Numbers a record was not allowed to claim:

- A `choice` record stored the top-1 mass as its confidence even when the stored prediction was a
  different class (0.85 confidence for a class the map gave 0.08). The confidence is now the
  probability of the prediction, and a prediction absent from its own distribution is refused.
- A `noul` probability of 1.5 was stored verbatim, and a NaN became a maximally confident answer.
  Values are clamped to [0, 1] and non-finite input is refused.
- Reductions over finite score inputs could overflow to `inf`, which the report printed as `MAE inf`.
  A non-finite result is now reported as unmeasurable.
- A `CostAction` with a NaN cost produced an invented threshold, because every NaN comparison is
  False and the first grid point "won". Costs must be finite and non-negative at construction.

Numbers the data could not support:

- The ECE bootstrap interval could exclude the ECE printed beside it, and excluded zero for a
  calibrated log. A percentile interval on a convex statistic sits above its own point estimate, so
  the interval is now widened to contain it, and a degenerate resample distribution reports no
  interval instead of `[x, x]`.
- `diagnose()` claimed over- or under-confidence from a bin whose own Wilson interval contained the
  confidence it contradicted: a calibrated log was flagged in 20 of 20 seeds at three sample sizes.
  A direction is now claimed only when the bin's interval excludes the claim.
- `jeval plan --target-ci 1e-160` crashed with `OverflowError`; projections are now computed in log
  space and refused past 1e9 labels.
- A labelled record with no `label_source` counted as gold for ECE while costs, drift, baseline and
  the segment bars excluded it — two definitions of one word in one report. Anything that is not a
  gold source now counts as silver everywhere.

Claims the report made without evidence:

- The threshold in use was substituted with the recommendation when nothing was deployed, so the
  report said "the 0.99 threshold in use" and compared it with itself. With nothing deployed the
  report now says so, and no impact table is built.
- Every action's cost section rendered the **first** action's impact table (wrong threshold, wrong
  cost, wrong auto-rate under another action's heading), and three sections shared one DOM id. Each
  action now gets its own table and its own control ids.
- A period comparison printed `model changed: 2026-W36 -> 2026-W37`, and comparing a slice with
  itself printed `model changed: new -> new` with a +0.000 delta. The block now names a period as a
  period and reports that there is nothing to compare.
- The drift note and failure detail were the only unescaped sinks in the report: a question key or
  model name containing `<script>` executed. Both are escaped, and the embedded JSON payload escapes
  every `<` so a `<!--<script` value can no longer swallow the report's own script and kill every
  interactive control.

Corrupting operations on the user's own files:

- Several questions of one request shared a single `id`, so one harvested answer was written onto
  every question of that request and the labeling sheet applied an answer to an arbitrary sibling.
  Each record now gets its own id.
- Harvesting one label silently rewrote every CRLF line ending in `records.jsonl`, contradicting the
  promise that only label fields change. The rewrite now reads bytes.
- `jeval ingest log.jsonl --labels resolutions.jsonl` — the README's own example — ingested nothing
  and said nothing. Both steps now run, in order; `--preset` together with `--labels` is refused
  with the reason instead of silently skipping the harvest.
- A malformed log line surfaced as a rich traceback; it is now one message with `file:line`.
- A `label` inside a native answer object was dropped with no counter; the preset keeps it.

- A label harvest joined on a shared key no longer writes one question's answer onto another
  question's records (found by running the documented flow against a two-question log).

### Added

- `jeval init` scaffolds `.jeval/config.yaml`, an ingest map, and a cost-matrix template.
- `jeval ingest` reads JSONL/CSV logs into decision records, one record per question, with a
  configurable field mapping and explicit counts for unlabeled, skipped, and duplicate rows.
- `jeval report` writes a single dependency-free HTML file containing the reliability curve,
  ECE, MCE, Brier score, per-bin Wilson intervals, a bootstrapped ECE interval, per-question
  breakdowns, and optional segment breakdowns.
- `jeval demo` generates a synthetic decision log with known miscalibration and reports on it.
- Quantile binning by default, with equal-width binning available via `--bins-equal-width` and
  automatic bin reduction on small samples.
- Gold and silver labels are aggregated separately; a silver-only dataset is reported with a
  warning rather than a headline number.
- Synthetic-data restoration tests: calibrated data must report ECE near zero, inflated data
  must report the inflation, and high confidence with low accuracy must be flagged as
  overconfident.

- `jeval threshold` turns a cost matrix into a per-action threshold with a bootstrap interval and
  writes `thresholds.yaml`; a wide interval is reported as a label problem rather than hidden.
- `jeval drift` compares model versions or periods, supports baseline snapshots
  (`--save-baseline` / `--baseline`), and fails a build with `--fail-on ece-increase=0.05`.
- `jeval report` now writes an argument document rather than a metrics dump, in a fixed reading
  order: verdict, reliability, cost, impact, segments, drift, data quality.
- Charts are hand-written inline SVG with no plotting dependency, colour-blind-safe and never
  colour-only: reliability curves with sample-sized dots and Wilson intervals, a cost curve with
  its minimum and flat region, segment bars with a per-segment curve on click, and a drift view
  with model-change markers.
- The report embeds only aggregates (never raw records), stays under 1 MB, works offline, prints
  to a single A4 page, and follows `prefers-color-scheme` for dark mode.
- `jeval report --format md` prints the verdict and impact table as markdown for a PR comment.
- Accessibility: `<title>`/`<desc>` on every chart, collapsible data tables under each chart, and
  a no-JavaScript reading path.

### Notes

- `score`-type records are excluded from binary accuracy and counted separately in the report.
- The report explores thresholds with a slider but never writes `thresholds.yaml`; only
  `jeval threshold` writes configuration.
- jeval prints markdown for CI but never posts it: it does not handle tokens.
- `jeval label` (M4) is not implemented yet.

[Unreleased]: https://github.com/rlaope/jeval/commits/main
