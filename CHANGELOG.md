# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

### Fixed
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
