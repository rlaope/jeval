"""The jeval command line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from jeval import __version__
from jeval.calibration import DEFAULT_N_BINS
from jeval.config import load_config, load_ingest_map, write_default_config
from jeval.evaluate import DatasetReport, evaluate
from jeval.ingest import ingest_files
from jeval.report.html import write_report
from jeval.store import DATA_DIR_NAME, load_records, records_path, write_records
from jeval.synth import demo_dataset

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Measure the calibration of probabilistic classifiers and put the human/AI line where "
    "the cost says it belongs.",
)

INGEST_SCAFFOLD = """# jeval ingest map
# Rename the fields of your logs onto the decision-record schema.
version: 1

# raw field name -> jeval field
field_map:
  model: model
  question_key: question
  question_type: question_type
  prediction: prediction
  confidence: confidence
  probabilities: probabilities
  label: label
  label_source: label_source
  segment: segment
  state_tokens: state_tokens
  latency_ms: latency_ms
  cost_usd: cost_usd
  ts: ts

# Fields applied when a row does not carry them.
defaults:
  question_type: choice

# A row may carry several questions under this key; each becomes one record.
questions_field: questions

# Free labels: the human answer that already exists in your own logs.
# Enabling this turns labeling cost into a mapping exercise.
label_from: {}
  # field: resolution.final_department
  # source: human_override
  # join_on: ticket_id
"""


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"jeval {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: Annotated[
        bool, typer.Option("--version", callback=_version_callback, is_eager=True)
    ] = False,
) -> None:
    """jeval: confidence, measured."""


@app.command()
def init(
    root: Annotated[Path, typer.Option("--root", help="Project root to scaffold.")] = Path("."),
    force: Annotated[bool, typer.Option("--force", help="Overwrite existing config.")] = False,
) -> None:
    """Create the .jeval working directory and its config scaffolding."""
    config_file = write_default_config(root, force=force)
    directory = Path(root) / DATA_DIR_NAME
    ingest_map = directory / "ingest-map.yaml"
    if force or not ingest_map.exists():
        ingest_map.write_text(INGEST_SCAFFOLD, encoding="utf-8")
    examples = directory / "examples"
    examples.mkdir(parents=True, exist_ok=True)
    costs_template = examples / "costs.example.yaml"
    if force or not costs_template.exists():
        costs_template.write_text(COSTS_SCAFFOLD, encoding="utf-8")
    typer.echo(f"initialized {directory}/")
    typer.echo(f"  config:    {config_file}")
    typer.echo(f"  ingest:    {ingest_map}")
    typer.echo(f"  costs:     {costs_template}")
    typer.echo("next: jeval ingest <your-log.jsonl>, then jeval report")


COSTS_SCAFFOLD = """# jeval cost matrix template
# Copy to costs.yaml and replace the numbers with YOUR costs: every threshold jeval
# recommends is a direct consequence of these figures. Wrong costs give wrong thresholds.
actions:
  - name: auto_refund
    question: intent
    when: refund_request
    cost_false_accept: 50000   # you ran it automatically and it was wrong
    cost_escalate: 2000        # you sent it to a human
    cost_false_reject: 0       # it was right and you still sent it to a human
"""


@app.command()
def ingest(
    files: Annotated[list[Path], typer.Argument(help="JSONL or CSV files to ingest.")],
    root: Annotated[Path, typer.Option("--root", help="Project root holding .jeval/.")] = Path("."),
    mapping: Annotated[
        Path | None,
        typer.Option("--mapping", help="Ingest map file (default .jeval/ingest-map.yaml)."),
    ] = None,
    append: Annotated[bool, typer.Option("--append", help="Append instead of replacing.")] = False,
) -> None:
    """Turn raw logs into decision records."""
    config = load_config(root)
    map_path = mapping or (Path(root) / DATA_DIR_NAME / config.ingest_map)
    ingest_map = load_ingest_map(map_path)
    out_path = records_path(root)
    result = ingest_files(files, out_path, ingest_map, append=append)
    typer.echo(f"read {result.n_rows} rows from {len(files)} file(s)")
    typer.echo(f"wrote {result.n_records} records to {out_path}")
    if result.per_question:
        typer.echo("questions:")
        for key in sorted(result.per_question):
            typer.echo(f"  {key:<24} {result.per_question[key]}")
    if result.n_unlabeled:
        typer.echo(
            f"{result.n_unlabeled} records have no label yet and will be excluded from metrics."
        )
        typer.echo(
            "free labels: the human answer for every escalated case is already in your logs. "
            "Map it under label_from in your ingest map."
        )
    if result.n_skipped:
        typer.echo(f"skipped {result.n_skipped} rows")
        for error in result.errors:
            typer.echo(f"  {error}")
    if not result.ok:
        raise typer.Exit(code=1)


@app.command()
def report(
    root: Annotated[Path, typer.Option("--root", help="Project root holding .jeval/.")] = Path("."),
    out: Annotated[Path | None, typer.Option("--out", help="Output HTML path.")] = None,
    bins: Annotated[int | None, typer.Option("--bins", help="Requested bin count.")] = None,
    equal_width: Annotated[
        bool, typer.Option("--bins-equal-width", help="Use equal-width bins instead of quantiles.")
    ] = False,
    by: Annotated[
        list[str] | None, typer.Option("--by", help="Segment key to break down (repeatable).")
    ] = None,
    open_browser: Annotated[
        bool, typer.Option("--open", help="Open the report in a browser.")
    ] = False,
) -> None:
    """Build the reliability report as a single HTML file."""
    config = load_config(root)
    try:
        records = load_records(root)
    except FileNotFoundError:
        typer.echo(f"no decision records under {root}: run `jeval ingest <file>` first")
        raise typer.Exit(code=1) from None
    if not records:
        typer.echo(f"no records in {records_path(root)}: run `jeval ingest <file>` first")
        raise typer.Exit(code=1)
    breakdown = tuple(by) if by else config.by
    dataset = evaluate(
        records,
        n_bins=bins if bins is not None else config.bins,
        equal_width=equal_width or config.bins_equal_width,
        alpha=config.alpha,
        n_boot=config.bootstrap_samples,
        by=breakdown,
        min_segment_size=config.min_segment_size,
    )
    target = out or (Path(root) / "report.html")
    written = write_report(dataset, target, source_note=f"source: {records_path(root)}")
    _print_summary(dataset)
    typer.echo(f"report: {Path(written)}")
    if open_browser:
        typer.launch(str(Path(written).resolve()))


@app.command()
def demo(
    out_dir: Annotated[
        Path, typer.Option("--out-dir", help="Where to write the demo project.")
    ] = Path(".jeval-demo"),
    seed: Annotated[int, typer.Option("--seed")] = 11,
    scale: Annotated[float, typer.Option("--scale", help="Scale the demo sample sizes.")] = 1.0,
) -> None:
    """Generate a synthetic log with known miscalibration and report on it."""
    dataset = demo_dataset(seed=seed, scale=scale)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_file = write_records(dataset.records, records_path(out_dir))
    typer.echo(f"generated {len(dataset.records)} synthetic records with seed {seed}")
    typer.echo(f"wrote {records_file}")
    for line in _demo_question_lines(dataset):
        typer.echo(f"  {line}")
    write_default_config(out_dir, force=True)
    evaluation = evaluate(
        dataset.records,
        n_bins=DEFAULT_N_BINS,
        by=("lang", "tier"),
        seed=seed,
    )
    target = write_report(
        evaluation,
        out_dir / "report.html",
        source_note=f"synthetic demo data (seed {seed})",
        demo=dataset,
    )
    _print_summary(evaluation)
    typer.echo(f"report: {target}")
    typer.echo("these numbers are the output of a known generator, not a real model's record")
    typer.echo(f"try it yourself: jeval report --root {out_dir}")


def _demo_question_lines(dataset: object) -> list[str]:
    from jeval.schema import DecisionRecord
    from jeval.synth import DemoDataset, accuracy_of

    if not isinstance(dataset, DemoDataset):
        return []
    grouped: dict[str, list[DecisionRecord]] = {}
    for record in dataset.records:
        grouped.setdefault(record.question_key, []).append(record)
    lines = []
    for key in sorted(grouped):
        group = grouped[key]
        accuracy = accuracy_of(group)
        lines.append(f"{key:<14} n={len(group):<5} observed accuracy {accuracy:.3f}")
    return lines


def _print_summary(dataset: DatasetReport) -> None:
    overall = dataset.overall
    typer.echo("")
    typer.echo(f"labeled decisions (gold): {dataset.n_labeled_gold}")
    if dataset.n_labeled_silver:
        typer.echo(f"silver labels, kept separate: {dataset.n_labeled_silver}")
    if dataset.n_unlabeled:
        typer.echo(f"unlabeled, excluded: {dataset.n_unlabeled}")
    if dataset.n_score_excluded:
        typer.echo(f"score-type records excluded from accuracy: {dataset.n_score_excluded}")
    if overall.n == 0:
        typer.echo("no labeled records: nothing to measure. Labels first, thresholds later.")
        return
    typer.echo(
        f"ECE {overall.ece:.3f} (95% CI {overall.ece_ci_low:.3f}-{overall.ece_ci_high:.3f})  "
        f"MCE {overall.mce:.3f}  Brier {overall.brier:.3f}  n={overall.n}"
    )
    typer.echo(dataset.overall_diagnosis)
    needed = _labels_for_tighter_interval(overall.n, overall.ece_ci_span)
    if needed:
        typer.echo(
            f"{needed} more labels would roughly halve the ECE interval "
            f"(currently +/-{overall.ece_ci_span / 2:.3f})."
        )


def _labels_for_tighter_interval(n: int, span: float, target_span: float = 0.05) -> int:
    """Rough label count to shrink the interval, using the 1/sqrt(n) sampling rate."""
    if n <= 0 or span != span or span <= target_span:
        return 0
    factor = (span / target_span) ** 2
    return max(0, int(n * (factor - 1)))


def main() -> None:
    """Console-script entry point (``jeval``)."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
