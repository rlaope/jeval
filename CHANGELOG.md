# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Notes

- `score`-type records are excluded from binary accuracy and counted separately in the report.
- `jeval threshold` (M1), `jeval drift` (M2), and `jeval label` (M4) are not implemented yet.

[Unreleased]: https://github.com/rlaope/jeval/commits/main
