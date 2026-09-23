"""The jeval command line interface."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any

import typer

from jeval import __version__
from jeval.calibration import compute_calibration
from jeval.config import load_config, load_ingest_map, write_default_config
from jeval.costs import CostAction
from jeval.evaluate import DatasetReport, evaluate
from jeval.ingest import ingest_files
from jeval.report import template
from jeval.report.charts import segments as segment_charts
from jeval.report.model import (
    DataQuality,
    HeatmapCell,
    ImpactTable,
    LabelPlanRow,
    RecalibrationView,
    ReportModel,
    ScoreLevelRow,
    ScoreView,
    SegmentThresholdRow,
    SegmentView,
    ThresholdResult,
)
from jeval.report.verdict import build_verdict
from jeval.schema import DecisionRecord
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
    directory = Path(root) / DATA_DIR_NAME
    if directory.exists() and not directory.is_dir():
        # A file named .jeval used to surface as a FileExistsError traceback.
        typer.echo(f"{directory} exists and is not a directory: move it aside and run init again")
        raise typer.Exit(code=1)
    config_file = write_default_config(root, force=force)
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
    files: Annotated[
        list[Path] | None, typer.Argument(help="JSONL or CSV files to ingest.")
    ] = None,
    root: Annotated[Path, typer.Option("--root", help="Project root holding .jeval/.")] = Path("."),
    mapping: Annotated[
        Path | None,
        typer.Option("--mapping", help="Ingest map file (default .jeval/ingest-map.yaml)."),
    ] = None,
    append: Annotated[bool, typer.Option("--append", help="Append instead of replacing.")] = False,
    labels: Annotated[
        Path | None,
        typer.Option(
            "--labels",
            help="Resolution log to harvest human labels from (the free labels you already have).",
        ),
    ] = None,
    label_field: Annotated[
        str | None,
        typer.Option("--label-field", help="Field in the resolution log holding the answer."),
    ] = None,
    label_source: Annotated[
        str | None,
        typer.Option("--label-source", help="human_review | human_override | silver."),
    ] = None,
    join_on: Annotated[
        str | None,
        typer.Option("--join-on", help="Join key shared by the log and the resolution log."),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Let harvested labels replace existing ones."),
    ] = False,
    preset: Annotated[
        str | None,
        typer.Option(
            "--preset",
            help="Ingest a log a product already writes, e.g. jev-native (native decision-API "
            "responses). List them with --list-presets.",
        ),
    ] = None,
    response_field: Annotated[
        str | None,
        typer.Option("--response-field", help="Where the response object sits in each log line."),
    ] = None,
    source_key_field: Annotated[
        str | None,
        typer.Option(
            "--source-key-field", help="Field carrying the join key (ticket id, trace id)."
        ),
    ] = None,
    list_presets: Annotated[
        bool, typer.Option("--list-presets", help="Show the available ingest presets and exit.")
    ] = False,
    label_question: Annotated[
        str | None,
        typer.Option(
            "--label-question",
            help="Restrict the harvest to this question (one join key is usually shared by all "
            "of a request's questions).",
        ),
    ] = None,
    allow_unlisted: Annotated[
        bool,
        typer.Option(
            "--allow-unlisted-labels",
            help="Accept a choice label that is not among the record's own probabilities.",
        ),
    ] = False,
) -> None:
    """Turn raw logs into decision records, and optionally harvest labels you already have."""
    files = list(files or [])
    if list_presets:
        from jeval import presets as preset_module

        for name in preset_module.available():
            item = preset_module.get(name)
            typer.echo(f"{name}\n  {item.description}\n  {preset_module.describe(item)}")
        return
    if preset is not None:
        if not files:
            typer.echo(f"--preset {preset} needs at least one log file")
            raise typer.Exit(code=1)
        if labels is not None:
            typer.echo(
                "--preset reads a log and --labels harvests human answers onto records: run them "
                "as two commands, because the harvest needs the records the preset just wrote"
            )
            raise typer.Exit(code=1)
        _preset_command(
            root=root,
            inputs=files,
            name=preset,
            appendix=append,
            response_field=response_field,
            source_key_field=source_key_field,
        )
        return
    if labels is not None:
        if files:
            # `jeval ingest log.jsonl --labels resolutions.jsonl` reads as "ingest the log and
            # harvest the answers". Silently ignoring the log would leave the user believing their
            # records were loaded, so both steps run, in the order that makes them work.
            ingest_map_for_files = load_ingest_map(
                mapping or (Path(root) / DATA_DIR_NAME / load_config(root).ingest_map)
            )
            try:
                loaded = ingest_files(
                    files, records_path(root), ingest_map_for_files, append=append
                )
            except (ValueError, FileNotFoundError) as exc:
                typer.echo(str(exc))
                raise typer.Exit(code=1) from None
            _report_ingest(loaded)
        _harvest_command(
            root=root,
            labels=labels,
            label_field=label_field,
            label_source=label_source,
            join_on=join_on,
            overwrite=overwrite,
            label_question=label_question,
            allow_unlisted=allow_unlisted,
        )
        return
    if not files:
        typer.echo("nothing to ingest: pass a JSONL/CSV file, or use --preset / --labels with one")
        raise typer.Exit(code=1)
    config = load_config(root)
    map_path = mapping or (Path(root) / DATA_DIR_NAME / config.ingest_map)
    ingest_map = load_ingest_map(map_path)
    out_path = records_path(root)
    try:
        result = ingest_files(files, out_path, ingest_map, append=append)
    except (ValueError, FileNotFoundError) as exc:
        # One malformed line used to abort the command as a rich traceback panel.
        typer.echo(str(exc))
        raise typer.Exit(code=1) from None
    _report_ingest(result)


def _fmt_target(width: float) -> str:
    """A target width the reader can actually read: 1e-06 printed as 0.000 is not a target."""
    return f"{width:.3f}" if width >= 0.0005 else f"{width:.2g}"


def _resolve_axes(records: Sequence[DecisionRecord], requested: Sequence[str]) -> tuple[str, ...]:
    """Requested segment axes, deduped, with the ones the log does not carry named out loud.

    An axis nobody recorded used to render as a segment called "<axis> = unknown", which invents a
    finding: the report looked like it had measured a breakdown that does not exist.
    """
    present: set[str] = set()
    for record in records:
        present.update(record.segment)
    used: list[str] = []
    for axis in requested:
        if axis in used:
            continue
        if axis in present or not records:
            used.append(axis)
            continue
        typer.echo(f"note: no record carries segment {axis!r}, so it is not broken down")
    return tuple(used)


MIN_BINS = 2
"""One bin aggregates every decision into a single average: ECE collapses toward zero and the
report reads as 'confidence is trustworthy' whatever the data says. Two is the least that can
show a direction, so two is the floor."""


def _load_records(root: Path) -> list[DecisionRecord]:
    """Read a project's records, or exit with one readable line.

    Six commands read records directly, so a missing file surfaced as whatever each of them happened
    to do — one of them as an uncaught FileNotFoundError.
    """
    try:
        return load_records(root)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None


def _report_ingest(result: Any) -> None:
    """Print one ingest run's numbers, including every reason a row was skipped."""
    typer.echo(f"read {result.n_rows} rows")
    typer.echo(f"wrote {result.n_records} records to {result.out_path}")
    if result.per_question:
        typer.echo("questions:")
        for key in sorted(result.per_question):
            typer.echo(f"  {key:<24} {result.per_question[key]}")
    if result.n_impossible_labels:
        named = ", ".join(result.impossible_labels[:3])
        typer.echo(
            f"dropped {result.n_impossible_labels} label(s) the record's own question cannot "
            f"produce ({named}) — an impossible label counts as a wrong answer forever, so the "
            "prediction is kept and the label is not"
        )
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
) -> tuple[tuple[ThresholdResult, ...], dict[str, ImpactTable]]:
    """Sweep every action's threshold, and build one impact table per action.

    Per action, because a single table shared across the report showed the first action's
    numbers beside every other action's curve — the wrong threshold, the wrong cost and the
    wrong auto-rate under a heading naming a different decision.

    With no deployed threshold there is no table at all: substituting the recommendation would
    make the report claim a threshold is in use when nobody said so, and every delta would be
    zero by construction.
    """
    from jeval.costs import build_impact, sweep

    results: list[ThresholdResult] = []
    impacts: dict[str, ImpactTable] = {}
    for action in actions:
        result = sweep(action, records, steps=steps, n_boot=n_boot, alpha=alpha)
        results.append(result)
        if current_threshold is not None:
            impacts[result.action] = build_impact(
                result,
                current_threshold=current_threshold,
                monthly_volume=monthly_volume,
                currency=currency,
            )
    return tuple(results), impacts


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
        # 6:1 rather than 25:1: at 25:1 this demo model's best bucket sits exactly at break-even, so
        # the honest answer is the top of the grid (automate nothing) and the reader learns nothing
        # about the curve. The demo exists to show a minimum inside the range.
        "  - name: auto_refund\n"
        "    question: intent\n"
        "    when: refund_request\n"
        "    cost_false_accept: 12000\n"
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


def _score_view(records: Sequence[DecisionRecord]) -> ScoreView | None:
    """Score metrics for the report, or nothing at all when the log has no score records."""
    from jeval.score import measure_score

    metrics = measure_score(records)
    if metrics.n_records == metrics.n_other_type:
        return None  # a binary-only log gets no score section rather than an empty one
    return ScoreView(
        n=metrics.n,
        mae=metrics.mae,
        rmse=metrics.rmse,
        spearman_rho=metrics.spearman_rho,
        levels=tuple(
            ScoreLevelRow(
                lo=level.lo,
                hi=level.hi,
                n=level.n,
                mean_predicted=level.mean_predicted,
                mean_actual=level.mean_actual,
            )
            for level in metrics.levels
        ),
        n_other_type=metrics.n_other_type,
        n_unlabeled=metrics.n_unlabeled,
        n_unparseable=metrics.n_unparseable,
    )


def _recalibration_view(
    records: Sequence[DecisionRecord], *, alpha: float
) -> RecalibrationView | None:
    """One cheap temperature fit for the report; `jeval calibrate` does the thorough version."""
    from jeval import recalibrate

    if sum(1 for record in records if record.label is not None) < 30:
        return None
    fit = recalibrate.fit_temperature(
        records, grid=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0), folds=3, repeats=2
    )
    return RecalibrationView(
        method=fit.method,
        before_ece=fit.before_ece,
        after_ece=fit.after_ece,
        n=fit.n,
        helps=fit.helps,
        note=fit.note,
    )


def _label_plan_rows(
    records: Sequence[DecisionRecord], *, by: Sequence[str], alpha: float
) -> tuple[LabelPlanRow, ...]:
    """Project the labels needed for the tightest useful interval, per question."""
    from jeval.planning import plan_labels

    rows: list[LabelPlanRow] = []
    for item in plan_labels(records, target_ci=0.05, by=(), alpha=alpha):
        needed = (
            " · ".join(
                f"{_fmt_target(target)}: {count:,}" for target, count in item.labels_for_target
            )
            if item.labels_for_target
            else ""
        )
        rows.append(
            LabelPlanRow(
                scope=item.scope,
                key=item.key,
                n_now=item.n_now,
                ece=item.ece,
                ci_width=item.ci_width,
                needed=needed,
                reason="" if needed else item.reason,
            )
        )
    return tuple(rows)


def _segment_threshold_rows(
    actions: Sequence[CostAction],
    records: Sequence[DecisionRecord],
    *,
    by: Sequence[str],
    steps: int,
    min_records: int,
) -> tuple[SegmentThresholdRow, ...]:
    """Per-segment optima for the first cost axis, so the report can say if splitting pays."""
    from jeval.costs import sweep_by_segment

    if not by:
        return ()
    axis = by[0]
    rows: list[SegmentThresholdRow] = []
    for action in actions:
        for segment in sweep_by_segment(
            action, records, segment_key=axis, min_records=min_records, steps=steps
        ):
            rows.append(
                SegmentThresholdRow(
                    label=f"{action.name} {segment.label}",
                    threshold=segment.threshold,
                    cost_per_case=segment.expected_cost_per_case,
                    delta=segment.cost_delta_vs_global,
                    n=segment.n_records,
                    worth_splitting=segment.worth_splitting,
                    reason=segment.reason,
                )
            )
    return tuple(rows)


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
    impacts: dict[str, ImpactTable] = {}
    if actions:
        thresholds, impacts = sweep_actions(
            actions,
            records,
            n_boot=bootstrap,
            alpha=alpha,
            monthly_volume=monthly,
            currency=currency,
            current_threshold=deployed_threshold(root, current),
        )
        impact = next(iter(impacts.values()), None)
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
        impacts=impacts,
        score=_score_view(records),
        recalibration=_recalibration_view(records, alpha=alpha),
        label_plan=_label_plan_rows(records, by=by, alpha=alpha),
        segment_thresholds=(
            _segment_threshold_rows(
                actions, records, by=by, steps=101, min_records=min_segment_size
            )
            if actions and thresholds and by
            else ()
        ),
        data_quality=quality,
        generated_at=generated_at,
        source_note=f"source: {records_path(root)}" + (f" · {costs_note}" if costs_note else ""),
        demo_note=demo_note,
    )
    threshold_by_question = {
        result.question: result.threshold for result in thresholds if result.curve
    }
    action_by_question = {result.question: result.action for result in thresholds if result.curve}
    blocks = template.build_blocks(
        dataset, thresholds=threshold_by_question, actions=action_by_question
    )
    html = template.render_document(model, blocks, segment_metrics=segment_metrics)
    template.write_report(html, out)
    return dataset, thresholds, impact, out


@app.command()
def report(
    root: Annotated[Path, typer.Option("--root", help="Project root holding .jeval/.")] = Path("."),
    out: Annotated[Path | None, typer.Option("-o", "--out", help="Output file.")] = None,
    bins: Annotated[
        int | None,
        typer.Option("--bins", help=f"Requested bin count (at least {MIN_BINS})."),
    ] = None,
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
    records = _load_records(root)  # the shared loader owns the missing-file message
    if not records:
        typer.echo(f"no records in {records_path(root)}: run `jeval ingest <file>` first")
        raise typer.Exit(code=1)
    currency = currency.strip() or "USD"  # a blank code left a leading space before every amount
    if monthly is not None and monthly < 0:
        typer.echo(f"--monthly {monthly} is not a volume: monthly cases cannot be negative")
        raise typer.Exit(code=1)
    if current is not None and not 0.0 <= current <= 1.0:
        typer.echo(
            f"--current {current} is not a confidence: a deployed threshold is a probability "
            "between 0 and 1"
        )
        raise typer.Exit(code=1)
    if question:
        records = [record for record in records if record.question_key == question]
        if not records:
            typer.echo(f"no records for question {question!r}")
            raise typer.Exit(code=1)

    if bins is not None and bins < MIN_BINS:
        # One bin aggregates everything, so ECE collapses toward zero and the report reads as
        # "confidence is trustworthy" whatever the data says.
        typer.echo(
            f"--bins {bins} cannot support a calibration claim: one bin averages every decision "
            f"together, so the curve disappears. Use at least {MIN_BINS}."
        )
        raise typer.Exit(code=1)
    breakdown = _resolve_axes(records, tuple(by) if by else config.by)
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
        thresholds, impacts = (
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
            else ((), {})
        )
        impact = next(iter(impacts.values()), None)
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
    generated_at = template._stamp(None)
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
        generated_at=generated_at,
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
    by: Annotated[
        str | None,
        typer.Option("--by", help="Segment axis: does one threshold fit everyone?"),
    ] = None,
    min_records: Annotated[
        int, typer.Option("--min-records", help="Smallest segment worth sweeping.")
    ] = 100,
) -> None:
    """Turn a cost matrix into per-action thresholds, with confidence intervals."""
    from jeval.costs import write_thresholds_yaml

    config = load_config(root)
    records = _load_records(root)
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
        if (
            result.flat_region is not None
            and result.flat_region[0] <= 0.0
            and result.flat_region[1] >= 1.0
        ):
            typer.echo(
                f"{result.action}: every threshold costs the same on this data, so the sweep "
                f"cannot recommend one. {result.n_records} labeled decisions, "
                f"{result.auto_rate:.0%} would be automated at any line."
            )
            continue
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
    if by:
        typer.echo("")
        _print_segment_sweep(actions, results, records, by, min_records=min_records, steps=steps)


def _print_segment_sweep(
    actions: Sequence[CostAction],
    results: Sequence[ThresholdResult],
    records: Sequence[DecisionRecord],
    axis: str,
    *,
    min_records: int,
    steps: int,
) -> None:
    """Print whether giving each segment its own threshold pays for itself."""
    from jeval.costs import sweep_by_segment

    swept = {result.action: result for result in results}
    stated = 0
    for action in actions:
        result = swept.get(action.name)
        if result is None or not result.curve:
            continue
        segments = sweep_by_segment(
            action, records, segment_key=axis, min_records=min_records, steps=steps
        )
        if not segments:
            continue
        if stated:
            typer.echo("")
        stated += 1
        typer.echo(f"{result.action} by {axis}: global threshold {result.threshold:.2f}")
        typer.echo(
            f"  {'segment':<18} {'threshold':>9} {'cost/case':>10} "
            f"{'vs global':>10} {'n':>6}  verdict"
        )
        for segment in segments:
            if segment.threshold != segment.threshold:  # NaN: never swept
                typer.echo(
                    f"  {segment.label:<18} {'—':>9} {'—':>10} {'—':>10} {segment.n_records:>6}  "
                    f"{segment.reason}"
                )
                continue
            verdict = "split" if segment.worth_splitting else segment.reason
            typer.echo(
                f"  {segment.label:<18} {segment.threshold:>9.2f} "
                f"{segment.expected_cost_per_case:>10,.2f} {segment.cost_delta_vs_global:>10,.2f} "
                f"{segment.n_records:>6}  {verdict}"
            )
    if not stated:
        typer.echo("")
        typer.echo("no action had enough labeled decisions for a segment sweep.")


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

    records = _load_records(root)
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
    # Costs turn the block from "something changed" into "the line you should draw moved" — the
    # line a reviewer actually acts on.
    actions, costs_note = resolve_cost_actions(None, root)
    if actions:
        view = drift_engine.attach_thresholds(view, records, actions)
        if costs_note:
            typer.echo(costs_note)
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


def _preset_command(
    *,
    root: Path,
    inputs: Sequence[Path],
    name: str,
    appendix: bool,
    response_field: str | None,
    source_key_field: str | None,
) -> None:
    """Ingest a log a product already writes, through a preset."""
    from jeval import presets as preset_module
    from jeval.ingest import ingest_preset_files

    try:
        item = preset_module.get(name)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from None
    if response_field:
        item = replace(item, response_field=response_field)
    if source_key_field:
        item = replace(item, source_key_field=source_key_field)
    report = ingest_preset_files(
        list(inputs),
        records_path(root),
        item,
        append=appendix,
        source_key_field=source_key_field,
    )
    typer.echo(f"preset: {name} ({preset_module.describe(item)})")
    if report.n_records == 0:
        typer.echo(f"wrote 0 records from {report.n_rows} row(s)")
        for error in report.errors[:5]:
            typer.echo(f"  {error}")
        return
    _report_ingest(report)


def _harvest_command(
    *,
    root: Path,
    labels: Path,
    label_field: str | None,
    label_source: str | None,
    join_on: str | None,
    overwrite: bool,
    label_question: str | None = None,
    allow_unlisted: bool = False,
) -> None:
    """Apply an external resolution log onto existing records, atomically."""
    from jeval.ingest import harvest_file, iter_rows

    config = load_config(root)
    mapping = load_ingest_map(Path(root) / DATA_DIR_NAME / config.ingest_map)
    spec = None
    try:
        spec = mapping.label_spec()
    except ValueError as exc:
        if label_field is None or label_source is None or join_on is None:
            typer.echo(f"ingest map: {exc}")
            raise typer.Exit(code=1) from None
    field = label_field or (spec.field if spec else None)
    source = label_source or (spec.source if spec else None)
    key = join_on or (spec.join_on if spec else None)
    scope = label_question or (spec.question if spec else None)
    if not (field and source and key):
        typer.echo(
            "harvesting needs --label-field, --label-source and --join-on, or a complete "
            "label_from block in .jeval/ingest-map.yaml"
        )
        raise typer.Exit(code=1)
    records_file = records_path(root)
    if not records_file.exists():
        typer.echo(f"no decision records at {records_file}: run `jeval ingest <file>` first")
        raise typer.Exit(code=1)
    try:
        report = harvest_file(
            records_file,
            iter_rows(labels),
            field=field,
            source=source,  # type: ignore[arg-type]
            join_on=key,
            overwrite=overwrite,
            question=scope,
            allow_unlisted=allow_unlisted,
        )
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from None
    typer.echo(f"read {report.n_label_rows} label rows from {labels}")
    where = f" to the {scope!r} question" if scope else ""
    typer.echo(f"applied {report.n_applied} labels to {report.n_records} records{where}")
    if report.n_other_question:
        typer.echo(
            f"left {report.n_other_question} record(s) of other questions alone: {field!r} answers "
            "one question, and a request's other questions share the same key"
            + ("" if scope else " — set label_from.question (or --label-question) to name it")
        )
    if report.n_unlisted_label:
        named = ", ".join(report.unlisted_labels[:3])
        typer.echo(
            f"refused {report.n_unlisted_label} label(s) that the record's own question cannot "
            f"produce ({named}) — a wrong label is worse than a missing one; pass "
            "--allow-unlisted-labels only if your probabilities list top candidates only"
        )
    if report.n_kept_existing:
        typer.echo(
            f"kept {report.n_kept_existing} existing label(s) — a harvested label never replaces "
            "one that is already there unless you pass --overwrite"
        )
    if report.n_unmatched:
        named = ", ".join(report.unmatched_keys[:5])
        typer.echo(f"{report.n_unmatched} label row(s) matched no record: {named}")
    if report.n_skipped_without_key:
        typer.echo(
            f"{report.n_skipped_without_key} record(s) carry no {key!r}, so nothing could be "
            "joined to them"
        )
    if report.n_rows_without_label:
        typer.echo(f"{report.n_rows_without_label} label row(s) had no value in {field!r}")
    if report.n_rows_without_key:
        typer.echo(
            f"{report.n_rows_without_key} label row(s) carried no {key!r}, so there was nothing "
            "to join on"
        )
    if report.applications:
        typer.echo(f"rewrote {records_file} in place")
    else:
        typer.echo(f"left {records_file} untouched: nothing applied")
    if not report.ok:
        raise typer.Exit(code=1)


@app.command()
def plan(
    root: Annotated[Path, typer.Option("--root", help="Project root holding .jeval/.")] = Path("."),
    target_ci: Annotated[
        float | None,
        typer.Option("--target-ci", help="Interval width you want (0.05 means +/-0.025)."),
    ] = None,
    by: Annotated[
        list[str] | None,
        typer.Option("--by", help="Break the plan down by a segment axis (repeatable)."),
    ] = None,
) -> None:
    """Say how many more labels each question needs for a tighter interval."""
    from jeval.planning import plan_labels

    config = load_config(root)
    records = _load_records(root)
    plans = plan_labels(
        records,
        target_ci=target_ci if target_ci is not None else 0.05,
        by=tuple(by) if by else config.by,
        alpha=config.alpha,
    )
    if not plans:
        typer.echo("no labeled decisions to plan from: labels first, then a plan")
        raise typer.Exit(code=1)
    # The scope column is printed only when the rows disagree about it: repeating "question" on
    # every line spends 11 columns of a table that already runs wide.
    show_scope = len({item.scope for item in plans}) > 1
    head = f"{'scope':<10} " if show_scope else ""
    typer.echo(f"{head}{'key':<20} {'n':>6} {'ECE':>7} {'CI':>7}  needed")
    for item in plans:
        lead = f"{item.scope:<10} " if show_scope else ""
        if item.labels_for_target is None:
            typer.echo(
                f"{lead}{item.key:<20} {item.n_now:>6} "
                f"{item.ece:>7.3f} {item.ci_width:>7.3f}  {item.reason}"
            )
            continue
        wanted = " · ".join(
            f"{_fmt_target(target)}: {count:,}" for target, count in item.labels_for_target
        )
        typer.echo(
            f"{lead}{item.key:<20} {item.n_now:>6} {item.ece:>7.3f} {item.ci_width:>7.3f}  {wanted}"
        )
    from jeval.planning import _assumption_note

    typer.echo("")
    typer.echo(_assumption_note(plans[0]))
    typer.echo("this is a projection from your own data, not a measurement")


@app.command()
def label(
    root: Annotated[Path, typer.Option("--root", help="Project root holding .jeval/.")] = Path("."),
    costs: Annotated[
        Path | None, typer.Option("--costs", help="Cost matrix, so the queue knows the band.")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Records to queue.")] = 20,
    out: Annotated[
        Path | None, typer.Option("-o", "--out", help="Where to write the labeling sheet.")
    ] = None,
    apply_from: Annotated[
        Path | None,
        typer.Option("--apply", help="Apply a filled-in labeling sheet you exported earlier."),
    ] = None,
    source: Annotated[
        str, typer.Option("--source", help="Label source for applied answers.")
    ] = "human_review",
) -> None:
    """Queue the records whose labels would teach the tool the most."""
    from jeval import active

    records = _load_records(root)
    if apply_from is not None:
        with Path(apply_from).open(encoding="utf-8", newline="") as handle:
            applied = active.apply_labels(
                records,
                csv.DictReader(handle),
                key=active.KEY_COLUMN,
                label=active.LABEL_COLUMN,
                source=source,  # type: ignore[arg-type]
            )
        write_records(records, records_path(root))
        typer.echo(f"applied {applied} label(s) from {apply_from}")
        if applied == 0:
            typer.echo("no rows carried a label: fill the label column and try again")
            raise typer.Exit(code=1)
        return

    actions, _ = resolve_cost_actions(costs, root)
    thresholds: dict[str, ThresholdResult] = {}
    if actions:
        swept, _ = sweep_actions(actions, records, n_boot=60, alpha=load_config(root).alpha)
        thresholds = {result.question: result for result in swept if result.curve}
    queue = active.build_queue(records, thresholds=thresholds, limit=limit)
    typer.echo(active.format_queue(queue))
    breakdown = active.priority_breakdown(queue)
    if breakdown:
        typer.echo("")
        typer.echo("what this queue buys:")
        for name, count in breakdown.items():
            typer.echo(f"  {name:<24} {count}")
    sheet = out or (Path(root) / "labels.csv")
    if queue:
        active.export_session(queue, sheet)
        typer.echo("")
        typer.echo(f"labeling sheet: {sheet}")
        typer.echo(f"fill the label column, then: jeval label --apply {sheet}")
    else:
        typer.echo("")
        if not records:
            typer.echo("no decision records yet: run `jeval ingest <file>` first")
            raise typer.Exit(code=1)
        typer.echo("nothing to queue: every record already has a label")


@app.command()
def calibrate(
    root: Annotated[Path, typer.Option("--root", help="Project root holding .jeval/.")] = Path("."),
    method: Annotated[str, typer.Option("--method", help="temperature | isotonic | both")] = "both",
    out: Annotated[
        Path | None, typer.Option("-o", "--out", help="Where to write the correction map.")
    ] = None,
) -> None:
    """Fit a confidence correction and export it, or say plainly that none is needed."""
    from jeval import recalibrate

    records = _load_records(root)
    methods = ["temperature", "isotonic"] if method == "both" else [method]
    if any(name not in ("temperature", "isotonic") for name in methods):
        typer.echo(f"unknown --method {method!r}: expected temperature, isotonic or both")
        raise typer.Exit(code=1)
    for name in methods:
        fit = (
            recalibrate.fit_temperature(records)
            if name == "temperature"
            else recalibrate.fit_isotonic(records)
        )
        typer.echo(
            f"{fit.method}: ECE {fit.before_ece:.3f} -> {fit.after_ece:.3f} "
            f"(cross-validated, n={fit.n})"
        )
        if not fit.helps:
            typer.echo(f"  {fit.note}")
            typer.echo("  no correction exported: shipping an unearned map would make it worse")
            continue
        target = out or (Path(root) / f"calibration-{fit.method}.yaml")
        written = recalibrate.export_yaml(fit, target)
        typer.echo(f"  wrote {written}")
        typer.echo("  your application applies this map; jeval stays out of the request path")


def main() -> None:
    """Console-script entry point (``jeval``)."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
