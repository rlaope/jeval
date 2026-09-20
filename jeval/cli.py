"""The jeval command line interface."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import typer

from jeval import __version__
from jeval.calibration import compute_calibration
from jeval.config import load_config, load_ingest_map, write_default_config
from jeval.evaluate import DatasetReport, evaluate
from jeval.ingest import ingest_files
from jeval.report import template
from jeval.report.charts import segments as segment_charts
from jeval.report.model import (
    DataQuality,
    HeatmapCell,
    ImpactTable,
    ReportModel,
    SegmentView,
    ThresholdResult,
)
from jeval.report.verdict import build_verdict
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

COSTS_SCAFFOLD = """# jeval cost matrix template
# Copy to costs.yaml and replace the numbers with YOUR costs: every threshold jeval
# recommends is a direct consequence of these figures. Wrong costs give wrong thresholds.
actions:
  - name: auto_refund
    question: intent
    when: refund_request
    cost_false_accept: 50000   # it ran automatically and it was wrong
    cost_escalate: 2000        # it went to a human
    cost_false_reject: 0       # it was right and still went to a human
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


def resolve_cost_actions(costs: Path | None, root: Path) -> tuple[list[Any], str]:
    """Load a cost matrix if one exists. Returns the actions and a note for the report."""
    from jeval.costs import load_cost_actions

    candidates = (
        [Path(costs)]
        if costs
        else [Path(root) / "costs.yaml", Path(root) / DATA_DIR_NAME / "costs.yaml"]
    )
    for candidate in candidates:
        if candidate.exists():
            return list(load_cost_actions(candidate)), f"costs: {candidate}"
    return [], ""


def deployed_threshold(root: Path, explicit: float | None) -> float | None:
    """The threshold currently in production, if the project can tell us.

    Read from ``thresholds.yaml`` because that file is what the application consumes; the report
    can then say what the deployed line costs, instead of comparing the recommendation to itself.
    """
    if explicit is not None:
        return explicit
    import yaml

    path = Path(root) / "thresholds.yaml"
    if not path.exists():
        return None
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return None
    actions = loaded.get("actions") if isinstance(loaded, dict) else None
    if not isinstance(actions, dict):
        return None
    for spec in actions.values():
        if isinstance(spec, dict):
            value = spec.get("threshold")
            if isinstance(value, (int, float)):
                return float(value)
    return None


def sweep_actions(
    actions: Sequence[Any],
    records: Sequence[Any],
    *,
    steps: int = 101,
    n_boot: int = 200,
    alpha: float = 0.05,
    monthly_volume: float | None = None,
    currency: str = "USD",
    current_threshold: float | None = None,
) -> tuple[tuple[ThresholdResult, ...], ImpactTable | None]:
    """Sweep every action's threshold and build the impact table for the first one."""
    from jeval.costs import build_impact, sweep

    results: list[ThresholdResult] = []
    impact: ImpactTable | None = None
    for action in actions:
        result = sweep(action, records, steps=steps, n_boot=n_boot, alpha=alpha)
        results.append(result)
        if impact is None:
            current = current_threshold if current_threshold is not None else result.threshold
            impact = build_impact(
                result,
                current_threshold=current,
                monthly_volume=monthly_volume,
                currency=currency,
            )
    return tuple(results), impact


def _metrics_for(group: Sequence[Any], *, alpha: float, n_boot: int) -> Any:
    pairs = [point for point in (record.calibration_point() for record in group) if point]
    return compute_calibration(
        [point[0] for point in pairs],
        [point[1] for point in pairs],
        alpha=alpha,
        n_boot=n_boot,
    )


def segment_views(
    records: Sequence[Any],
    by: Sequence[str],
    *,
    alpha: float = 0.05,
    n_boot: int = 200,
    min_samples: int = 30,
) -> tuple[SegmentView | None, dict[str, Any]]:
    """Segment bars, a two-axis grid when asked for two axes, and the metrics behind them."""
    if not by:
        return None, {}
    rows: list[tuple[str, str, float, int]] = []
    metrics_by_slug: dict[str, Any] = {}
    for axis in by:
        grouped: dict[str, list[Any]] = {}
        for record in records:
            grouped.setdefault(record.segment.get(axis, "unknown"), []).append(record)
        for value in sorted(grouped):
            group = [record for record in grouped[value] if record.is_gold]
            metrics = _metrics_for(group, alpha=alpha, n_boot=n_boot)
            rows.append((axis, value, metrics.ece, metrics.n))
            metrics_by_slug[segment_charts.segment_slug(axis, value)] = metrics

    cells: list[HeatmapCell] = []
    if len(by) == 2:
        pairs: dict[tuple[str, str], list[Any]] = {}
        for record in records:
            if not record.is_gold:
                continue
            key = (record.segment.get(by[0], "unknown"), record.segment.get(by[1], "unknown"))
            pairs.setdefault(key, []).append(record)
        for (x_value, y_value), group in sorted(pairs.items()):
            metrics = _metrics_for(group, alpha=alpha, n_boot=n_boot)
            if metrics.n:
                cells.append(HeatmapCell(x_value, y_value, metrics.ece, metrics.n))

    return (
        SegmentView(
            bars=segment_charts.bars_from_ece(tuple(rows)),
            heatmap=tuple(cells),
            x_axis=by[0],
            y_axis=by[1] if len(by) > 1 else "",
            min_samples=min_samples,
        ),
        metrics_by_slug,
    )


def drift_view(
    records: Sequence[Any],
    *,
    baseline: Path | None = None,
    by_period: str | None = None,
    min_slice: int = 30,
    alpha: float = 0.05,
    n_boot: int = 200,
) -> Any:
    """Build the drift view, or ``None`` when there is nothing to compare."""
    from typing import Literal, cast

    from jeval import baseline as baseline_engine
    from jeval import drift as drift_engine

    if by_period:
        period = by_period.strip().upper()
        if period not in ("W", "M"):
            raise typer.BadParameter("--by-period takes W (week) or M (month)")
        return drift_engine.compare(
            records,
            by_period=cast('Literal["W", "M"]', period),
            min_slice=min_slice,
            alpha=alpha,
            n_boot=n_boot,
        )
    models = sorted({record.model for record in records})
    if baseline is not None:
        payload = json.loads(Path(baseline).read_text(encoding="utf-8"))
        return baseline_engine.view_from_snapshot(
            payload, records, min_slice=min_slice, alpha=alpha, n_boot=n_boot
        )
    if len(models) < 2:
        return None
    return drift_engine.compare(records, min_slice=min_slice, alpha=alpha, n_boot=n_boot)


def _write_demo_costs(out_dir: Path) -> Path:
    """Synthetic cost figures for the demo, labelled as such and never a default for anyone."""
    target = out_dir / "costs.yaml"
    target.write_text(
        "# SYNTHETIC cost figures for the demo. Replace them with your own before believing\n"
        "# any threshold jeval recommends: wrong costs give wrong thresholds.\n"
        "actions:\n"
        "  - name: auto_refund\n"
        "    question: intent\n"
        "    when: refund_request\n"
        "    cost_false_accept: 50000\n"
        "    cost_escalate: 2000\n"
        "    cost_false_reject: 0\n"
        "  - name: auto_route\n"
        "    question: department\n"
        "    when: billing\n"
        "    cost_false_accept: 8000\n"
        "    cost_escalate: 1500\n"
        "    cost_false_reject: 200\n"
        "  - name: auto_escalate_urgent\n"
        "    question: is_urgent\n"
        '    when: "yes"\n'
        "    cost_false_accept: 30000\n"
        "    cost_escalate: 500\n"
        "    cost_false_reject: 1000\n",
        encoding="utf-8",
    )
    return target


def build_and_write_report(
    records: Sequence[Any],
    *,
    root: Path,
    out: Path,
    bins: int | None,
    equal_width: bool,
    by: Sequence[str],
    costs: Path | None,
    monthly: float | None,
    currency: str,
    compare: Path | None = None,
    current: float | None = None,
    alpha: float = 0.05,
    bootstrap: int = 200,
    min_segment_size: int = 30,
    generated_at: str = "",
    demo_note: str = "",
) -> tuple[DatasetReport, tuple[ThresholdResult, ...], ImpactTable | None, Path]:
    """Evaluate, sweep, assemble and write one report. Shared by `report` and `demo`."""
    dataset = evaluate(
        records,
        n_bins=bins if bins is not None else 10,
        equal_width=equal_width,
        alpha=alpha,
        n_boot=bootstrap,
        by=tuple(by),
        min_segment_size=min_segment_size,
    )
    actions, costs_note = resolve_cost_actions(costs, root)
    thresholds: tuple[ThresholdResult, ...] = ()
    impact: ImpactTable | None = None
    if actions:
        thresholds, impact = sweep_actions(
            actions,
            records,
            n_boot=bootstrap,
            alpha=alpha,
            monthly_volume=monthly,
            currency=currency,
            current_threshold=deployed_threshold(root, current),
        )
    segments_view, segment_metrics = segment_views(
        records, by, alpha=alpha, n_boot=bootstrap, min_samples=min_segment_size
    )
    compare_view = drift_view(
        records, baseline=compare, min_slice=min_segment_size, alpha=alpha, n_boot=bootstrap
    )
    verdict = build_verdict(
        dataset.overall,
        threshold=thresholds[0] if thresholds else None,
        current_threshold=impact.current_threshold if impact else None,
        recommended_threshold=thresholds[0].threshold if thresholds else None,
    )
    quality: DataQuality = template.data_quality_from(dataset)  # type: ignore[assignment]
    model = ReportModel(
        verdict=verdict,
        thresholds=thresholds,
        impact=impact,
        segments=segments_view,
        drift=compare_view,
        data_quality=quality,
        generated_at=generated_at,
        source_note=f"source: {records_path(root)}" + (f" · {costs_note}" if costs_note else ""),
        demo_note=demo_note,
    )
    threshold_by_question = {
        result.question: result.threshold for result in thresholds if result.curve
    }
    blocks = template.build_blocks(dataset, thresholds=threshold_by_question)
    html = template.render_document(model, blocks, segment_metrics=segment_metrics)
    template.write_report(html, out)
    return dataset, thresholds, impact, out


@app.command()
def report(
    root: Annotated[Path, typer.Option("--root", help="Project root holding .jeval/.")] = Path("."),
    out: Annotated[Path | None, typer.Option("-o", "--out", help="Output file.")] = None,
    bins: Annotated[int | None, typer.Option("--bins", help="Requested bin count.")] = None,
    equal_width: Annotated[
        bool, typer.Option("--bins-equal-width", help="Use equal-width bins instead of quantiles.")
    ] = False,
    by: Annotated[
        list[str] | None, typer.Option("--by", help="Segment axis to break down (repeatable).")
    ] = None,
    question: Annotated[
        str | None, typer.Option("--question", help="Report a single question only.")
    ] = None,
    compare: Annotated[
        Path | None,
        typer.Option("--compare", help="Baseline snapshot to compare against (enables Drift)."),
    ] = None,
    format_: Annotated[
        str, typer.Option("--format", help="html (default) or md for a paste-ready summary.")
    ] = "html",
    costs: Annotated[
        Path | None, typer.Option("--costs", help="Cost matrix (default costs.yaml if present).")
    ] = None,
    monthly: Annotated[
        float | None, typer.Option("--monthly", help="Monthly case volume for the impact table.")
    ] = None,
    current: Annotated[
        float | None,
        typer.Option(
            "--current",
            help="Threshold currently in production (default: read thresholds.yaml).",
        ),
    ] = None,
    currency: Annotated[
        str, typer.Option("--currency", help="Currency code for money labels.")
    ] = "USD",
    open_browser: Annotated[
        bool, typer.Option("--open", help="Open the report when written.")
    ] = False,
) -> None:
    """Build the report: reliability, cost, impact, segments, drift, data quality."""
    config = load_config(root)
    try:
        records = load_records(root)
    except FileNotFoundError:
        typer.echo(f"no decision records under {root}: run `jeval ingest <file>` first")
        raise typer.Exit(code=1) from None
    if not records:
        typer.echo(f"no records in {records_path(root)}: run `jeval ingest <file>` first")
        raise typer.Exit(code=1)
    if question:
        records = [record for record in records if record.question_key == question]
        if not records:
            typer.echo(f"no records for question {question!r}")
            raise typer.Exit(code=1)

    breakdown = tuple(by) if by else config.by
    bootstrap = min(200, config.bootstrap_samples)
    if format_ == "md":
        dataset = evaluate(
            records,
            n_bins=bins if bins is not None else config.bins,
            equal_width=equal_width or config.bins_equal_width,
            alpha=config.alpha,
            n_boot=bootstrap,
        )
        actions, _ = resolve_cost_actions(costs, root)
        thresholds, impact = (
            sweep_actions(
                actions,
                records,
                n_boot=bootstrap,
                alpha=config.alpha,
                monthly_volume=monthly,
                currency=currency,
                current_threshold=deployed_threshold(root, current),
            )
            if actions
            else ((), None)
        )
        verdict = build_verdict(
            dataset.overall,
            threshold=thresholds[0] if thresholds else None,
            current_threshold=impact.current_threshold if impact else None,
            recommended_threshold=thresholds[0].threshold if thresholds else None,
        )
        typer.echo(
            template.markdown_summary(
                ReportModel(
                    verdict=verdict,
                    thresholds=thresholds,
                    impact=impact,
                    data_quality=template.data_quality_from(dataset),  # type: ignore[arg-type]
                )
            ),
            nl=False,
        )
        raise typer.Exit(code=0)
    if format_ != "html":
        typer.echo(f"unknown --format {format_!r}: expected html or md")
        raise typer.Exit(code=1)

    target = out or (Path(root) / "report.html")
    dataset, thresholds, impact, written = build_and_write_report(
        records,
        root=root,
        out=target,
        bins=bins if bins is not None else config.bins,
        equal_width=equal_width or config.bins_equal_width,
        by=breakdown,
        costs=costs,
        monthly=monthly,
        currency=currency,
        compare=compare,
        current=current,
        alpha=config.alpha,
        bootstrap=bootstrap,
        min_segment_size=config.min_segment_size,
    )
    print_report_summary(dataset)
    for result in thresholds:
        if result.curve:
            typer.echo(
                f"threshold {result.action}: {result.threshold:.2f} "
                f"(95% CI {result.ci_low:.2f}-{result.ci_high:.2f}) · "
                f"{result.expected_cost_per_case:,.0f} cost/case · auto {result.auto_rate:.0%}"
            )
    typer.echo(f"report: {Path(written)}")
    if open_browser:
        typer.launch(str(Path(written).resolve()))


@app.command()
def threshold(
    root: Annotated[Path, typer.Option("--root", help="Project root holding .jeval/.")] = Path("."),
    costs: Annotated[
        Path | None, typer.Option("--costs", help="Cost matrix (default costs.yaml if present).")
    ] = None,
    out: Annotated[
        Path | None, typer.Option("-o", "--out", help="Output path (default thresholds.yaml).")
    ] = None,
    steps: Annotated[int, typer.Option("--steps", help="Sweep resolution.")] = 101,
    bootstrap: Annotated[
        int, typer.Option("--bootstrap", help="Bootstrap resamples for the interval.")
    ] = 200,
) -> None:
    """Turn a cost matrix into per-action thresholds, with confidence intervals."""
    from jeval.costs import write_thresholds_yaml

    config = load_config(root)
    records = load_records(root)
    actions, costs_note = resolve_cost_actions(costs, root)
    if not actions:
        typer.echo(
            "no cost matrix found: copy .jeval/examples/costs.example.yaml to costs.yaml and "
            "replace the numbers with your own"
        )
        raise typer.Exit(code=1)
    if costs_note:
        typer.echo(costs_note)
    results, _ = sweep_actions(actions, records, steps=steps, n_boot=bootstrap, alpha=config.alpha)
    target = out or (Path(root) / "thresholds.yaml")
    write_thresholds_yaml(results, target, generated_at=template._stamp(None))
    for result in results:
        if not result.curve or result.ci_low != result.ci_low:
            typer.echo(
                f"{result.action}: not enough labeled decisions ({result.n_records}) to recommend "
                "a threshold. Collect labels first."
            )
            continue
        span = result.ci_high - result.ci_low
        typer.echo(
            f"{result.action}: threshold {result.threshold:.2f} "
            f"(95% CI {result.ci_low:.2f}-{result.ci_high:.2f}, width {span:.2f}) · "
            f"auto {result.auto_rate:.0%} · {result.expected_cost_per_case:,.0f} per case"
        )
        if span > 0.2:
            typer.echo(
                "  wide interval: more labels, not more analysis, would sharpen this threshold."
            )
    typer.echo(f"wrote {target}")
    typer.echo("your application reads this file; jeval never sits in the request path.")


@app.command()
def drift(
    root: Annotated[Path, typer.Option("--root", help="Project root holding .jeval/.")] = Path("."),
    baseline: Annotated[
        Path | None,
        typer.Option("--baseline", help="Snapshot written by --save-baseline, to compare against."),
    ] = None,
    save_baseline: Annotated[
        Path | None, typer.Option("--save-baseline", help="Write a snapshot for later runs.")
    ] = None,
    fail_on: Annotated[
        list[str] | None,
        typer.Option("--fail-on", help="Check that fails the run, e.g. ece-increase=0.05."),
    ] = None,
    by_period: Annotated[
        str | None, typer.Option("--by-period", help="Split by W (week) or M (month).")
    ] = None,
) -> None:
    """Compare model versions or periods, and fail the build when calibration degrades."""
    from jeval import baseline as baseline_engine
    from jeval import drift as drift_engine

    records = load_records(root)
    models = sorted({record.model for record in records})
    if save_baseline is not None:
        baseline_engine.write_snapshot(baseline_engine.snapshot(records), save_baseline)
        typer.echo(f"saved baseline to {save_baseline} (models: {', '.join(models) or 'none'})")
        if baseline is None:
            return
    view = drift_view(
        records,
        baseline=baseline,
        by_period=by_period,
        min_slice=30,
        alpha=0.05,
        n_boot=200,
    )
    if view is None or not view.slices:
        typer.echo("no drift comparison available: need two model versions or --by-period")
        raise typer.Exit(code=0)
    checks = drift_engine.parse_fail_on(fail_on or [])
    failures = drift_engine.run_checks(view, checks) if checks else ()
    typer.echo(drift_engine.format_ci_block(view, failures), nl=False)
    if failures:
        raise typer.Exit(code=1)


@app.command()
def demo(
    out_dir: Annotated[
        Path, typer.Option("--out-dir", help="Where to write the demo project.")
    ] = Path(".jeval-demo"),
    seed: Annotated[int, typer.Option("--seed")] = 11,
    scale: Annotated[float, typer.Option("--scale", help="Scale the demo sample sizes.")] = 1.0,
    open_browser: Annotated[
        bool, typer.Option("--open", help="Open the report when written.")
    ] = False,
) -> None:
    """Generate a synthetic log with known miscalibration and report on it."""
    dataset = demo_dataset(seed=seed, scale=scale)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_file = write_records(dataset.records, records_path(out_dir))
    typer.echo(f"generated {len(dataset.records)} synthetic records with seed {seed}")
    typer.echo(f"wrote {records_file}")
    write_default_config(out_dir, force=True)
    costs_path = _write_demo_costs(out_dir)
    for line in demo_question_lines(dataset):
        typer.echo(f"  {line}")
    result, thresholds, _, written = build_and_write_report(
        dataset.records,
        root=out_dir,
        out=out_dir / "report.html",
        bins=10,
        equal_width=False,
        by=("lang", "tier"),
        costs=costs_path,
        monthly=20_000.0,
        currency="KRW",
        current=0.60,
        demo_note=dataset.description,
    )
    print_report_summary(result)
    for threshold_result in thresholds:
        if threshold_result.curve:
            typer.echo(
                f"threshold {threshold_result.action}: {threshold_result.threshold:.2f} "
                f"(95% CI {threshold_result.ci_low:.2f}-{threshold_result.ci_high:.2f}) · "
                f"auto {threshold_result.auto_rate:.0%}"
            )
    typer.echo(f"report: {Path(written)}")
    typer.echo("")
    typer.echo("the demo treats 0.60 as the threshold currently in production, so the impact")
    typer.echo("section has something real to compare against.")
    typer.echo("the costs in this demo are synthetic, and so is the data:")
    typer.echo("these numbers show what the tool computes, not what a real model does")
    if open_browser:
        typer.launch(str(Path(written).resolve()))


def demo_question_lines(dataset: Any) -> list[str]:
    from jeval.schema import DecisionRecord
    from jeval.synth import DemoDataset, accuracy_of

    if not isinstance(dataset, DemoDataset):
        return []
    grouped: dict[str, list[DecisionRecord]] = {}
    for record in dataset.records:
        grouped.setdefault(record.question_key, []).append(record)
    return [
        f"{key:<14} n={len(group):<5} observed accuracy {accuracy_of(group):.3f}"
        for key, group in sorted(grouped.items())
    ]


def print_report_summary(dataset: DatasetReport) -> None:
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
    needed = labels_for_tighter_interval(overall.n, overall.ece_ci_span)
    if needed:
        typer.echo(
            f"{needed} more labels would roughly halve the ECE interval "
            f"(currently +/-{overall.ece_ci_span / 2:.3f})."
        )


def labels_for_tighter_interval(n: int, span: float, target_span: float = 0.05) -> int:
    """Rough label count to shrink the interval, using the 1/sqrt(n) sampling rate."""
    if n <= 0 or span != span or span <= target_span:
        return 0
    return max(0, int(n * ((span / target_span) ** 2 - 1)))


def main() -> None:
    """Console-script entry point (``jeval``)."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
