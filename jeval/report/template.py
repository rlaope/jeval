"""Assemble the single-file HTML report.

Reading order is the specification: verdict, reliability, cost, impact, segments, drift, data
quality. A reader who stops after the first screen should already know the conclusion and what
it rests on; a reader who reaches the bottom should know exactly how thin the evidence is.

The document is a pure function of its inputs. No network, no files, no clock beyond the stamp
the caller passes in, which is what makes the golden-file tests possible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from jeval.calibration import CalibrationMetrics
from jeval.evaluate import DatasetReport
from jeval.report import svg as S
from jeval.report.charts import cost as cost_charts
from jeval.report.charts import drift as drift_charts
from jeval.report.charts import reliability as reliability_charts
from jeval.report.charts import segments as segment_charts
from jeval.report.model import (
    DEFAULT_LIMITATIONS,
    ReportModel,
    ThresholdResult,
    dumps,
)
from jeval.report.svg import details_table


@dataclass(frozen=True)
class ReliabilityBlock:
    """One question's reliability section."""

    question_key: str
    metrics: CalibrationMetrics
    diagnosis: str
    threshold: float | None = None
    subtitle: str = ""
    silver_note: str = ""


def markdown_summary(model: ReportModel) -> str:
    """The paste-into-Slack summary, formatted in exactly one place.

    Delegates to :func:`jeval.report.verdict.markdown_summary`, because the copy button, the
    embedded payload and ``--format md`` must all hand out the same text: a second formatter here
    would quietly drift from the one that has the tests.
    """
    from jeval.report.verdict import markdown_summary as _verdict_summary

    return _verdict_summary(model)


def _stamp(value: datetime | None) -> str:
    return (value or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M:%SZ")


def render_document(
    model: ReportModel,
    blocks: Sequence[ReliabilityBlock],
    *,
    segment_metrics: Mapping[str, CalibrationMetrics] | None = None,
    baseline_metrics: CalibrationMetrics | None = None,
    current_metrics: CalibrationMetrics | None = None,
) -> str:
    """Render the whole report as one self-contained HTML document."""
    from jeval.report.assets import REPORT_CSS, REPORT_JS, minify

    segment_metrics = dict(segment_metrics or {})
    thresholds = list(model.thresholds)
    current_thresholds = {
        result.action: result.point_at(result.threshold).threshold  # type: ignore[union-attr]
        for result in thresholds
        if result.curve
    }
    summary = markdown_summary(model)
    payload = dumps(
        {
            "actions": {
                result.action: {
                    "question": result.question,
                    "when": result.when,
                    "threshold": round(result.threshold, 4),
                    "n": result.n_records,
                    "currency": model.impact.currency if model.impact else "USD",
                    "monthly_volume": model.impact.monthly_volume if model.impact else None,
                    "current_threshold": current_thresholds.get(result.action),
                    "ci": [result.ci_low, result.ci_high],
                    "flat_region": list(result.flat_region) if result.flat_region else None,
                    "curve": [
                        {
                            "t": round(point.threshold, 4),
                            "cost": round(point.expected_cost, 6),
                            "auto": round(point.auto_rate, 6),
                            "acc": round(point.accuracy_auto, 6),
                        }
                        for point in result.curve
                    ],
                }
                for result in thresholds
            },
            "summary_markdown": summary,
        }
    )

    parts: list[str] = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>jeval report</title>",
        f"<style>{minify(REPORT_CSS)}</style>",
        "</head>",
        "<body>",
        '<main class="report">',
    ]
    parts.append(_header(model))
    if model.data_quality.silver_only:
        parts.append(
            '<div class="warn" role="alert"><strong>Every label here is silver.</strong> Silver '
            "labels record agreement with a model, not whether the answer was right. Read the "
            "numbers below as an agreement rate and collect human labels before acting.</div>"
        )
    if model.data_quality.no_gold_labels:
        parts.append(
            '<div class="warn" role="alert"><strong>No gold labels.</strong> jeval measures '
            "nothing until humans label decisions. The cheapest source is the human answer for "
            "every case that was escalated.</div>"
        )

    parts.append(_verdict_section(model, summary))
    parts.append(_reliability_section(blocks))
    parts.append(_cost_section(model, current_thresholds, thresholds))
    parts.append(_segments_section(model, segment_metrics))
    if model.drift is not None and model.drift.slices:
        parts.append(
            '<section id="drift"><h2>Drift</h2>'
            + drift_charts.render_drift_section(
                model.drift,
                baseline_metrics=baseline_metrics,
                current_metrics=current_metrics,
                threshold=thresholds[0].threshold if thresholds else None,
            )
            + "</section>"
        )
    parts.append(_data_quality_section(model))
    parts.append(
        "<footer><p>jeval · single file, no server, no external assets · the numbers come from "
        "the decision records on the machine that generated this file.</p>"
        '<p class="note">Thresholds are exploratory here. Confirming a value is the job of '
        "<code>jeval threshold</code>, which writes <code>thresholds.yaml</code>.</p></footer>"
    )
    parts.append("</main>")
    parts.append(S.embed_json(payload, element_id="jeval-data"))
    parts.append(f"<script>{minify(REPORT_JS)}</script>")
    parts.append("</body></html>")
    return "".join(parts)


def _header(model: ReportModel) -> str:
    meta = [f"Generated {escape(model.generated_at)}"]
    if model.source_note:
        meta.append(escape(model.source_note))
    if model.demo_note:
        meta.append(f"synthetic demo data: {escape(model.demo_note)}")
    return (
        '<header class="report-header"><h1>jeval report</h1>'
        f'<p class="meta">{" · ".join(meta)}</p></header>'
    )


def _verdict_section(model: ReportModel, summary: str) -> str:
    verdict = model.verdict
    cards = "".join(
        '<div class="stat"><div class="k">'
        + escape(stat.label)
        + '</div><div class="v">'
        + escape(stat.value)
        + "</div>"
        + (f'<div class="s">{escape(stat.sub)}</div>' if stat.sub else "")
        + "</div>"
        for stat in verdict.stats
    )
    return (
        '<section id="verdict" class="verdict" data-status="'
        + escape(verdict.status)
        + '">'
        + f'<p class="headline">{escape(verdict.headline)}</p>'
        + f'<p class="detail">{escape(verdict.detail)}</p>'
        + (f'<div class="stats">{cards}</div>' if cards else "")
        + '<button type="button" id="copy-summary" class="copy" data-jeval-copy-summary '
        + 'data-copy-target="jeval-data">Copy summary</button>'
        + "</section>"
    )


def _reliability_section(blocks: Sequence[ReliabilityBlock]) -> str:
    if not blocks:
        return '<section id="reliability"><h2>Reliability</h2><p class="note">No questions found.</p></section>'
    single = len(blocks) == 1
    tabs = ""
    if not single:
        tabs = (
            '<div class="tabs" role="tablist">'
            + "".join(
                f'<button type="button" role="tab" class="tab{" is-active" if index == 0 else ""}" '
                f'data-jeval-tab="{escape(block.question_key)}" '
                f'aria-selected="{"true" if index == 0 else "false"}">{escape(block.question_key)}</button>'
                for index, block in enumerate(blocks)
            )
            + "</div>"
        )
    figures = []
    for index, block in enumerate(blocks):
        body = [f"<h3>{escape(block.question_key)}</h3>"]
        if block.subtitle:
            body.append(f'<p class="note">{escape(block.subtitle)}</p>')
        body.append(f'<p class="diag">{escape(block.diagnosis)}</p>')
        body.append(
            reliability_charts.render_reliability_section(
                block.metrics,
                threshold=block.threshold,
                title=f"Reliability · {block.question_key}",
            )
        )
        if block.silver_note:
            body.append(f'<p class="note">{escape(block.silver_note)}</p>')
        figures.append(
            f'<div class="question-block{" is-active" if index == 0 else ""}" '
            f'data-jeval-question="{escape(block.question_key)}">' + "".join(body) + "</div>"
        )
    return (
        '<section id="reliability"><h2>Reliability</h2>'
        "<p>Stated confidence against what actually happened. Points below the diagonal mean the "
        "model was more confident than it was right.</p>" + tabs + "".join(figures) + "</section>"
    )


def _cost_section(
    model: ReportModel,
    current_thresholds: Mapping[str, float],
    thresholds: Sequence[ThresholdResult],
) -> str:
    if not thresholds:
        return (
            '<section id="cost"><h2>Cost</h2><p class="note">No cost matrix was supplied, so no '
            "threshold was computed. Add <code>costs.yaml</code> (see "
            "<code>costs.example.yaml</code>) and run <code>jeval threshold</code>.</p></section>"
        )
    body = "".join(
        f"<h3>{escape(result.action)}</h3>"
        + cost_charts.render_cost_section(
            result,
            current_threshold=current_thresholds.get(result.action),
            impact=model.impact,
        )
        for result in thresholds
    )
    return (
        '<section id="cost"><h2>Cost</h2>'
        "<p>Expected cost per case for every candidate threshold. The minimum is the "
        "recommendation; the flat region is where the data cannot tell neighbouring thresholds "
        "apart.</p>" + body + "</section>"
    )


def _segments_section(model: ReportModel, segment_metrics: Mapping[str, CalibrationMetrics]) -> str:
    view = model.segments
    if view is None or not view.bars:
        return (
            '<section id="segments"><h2>Segments</h2><p class="note">No segment breakdown was '
            "requested. Pass <code>--by lang</code> (and others) to break the numbers down.</p></section>"
        )
    figures = ""
    for bar in view.bars:
        if bar.too_few_samples:
            continue
        slug = segment_charts.segment_slug(bar.key, bar.value)
        metrics = segment_metrics.get(slug)
        if metrics is None:
            continue
        figures += (
            f'<figure class="seg-figure" id="seg-fig-{escape(slug)}" '
            f'data-jeval-segment-chart="{escape(slug)}" hidden>'
            f"{reliability_charts.render_reliability(metrics, title=f'{bar.label}')}"
            "</figure>"
        )
    return (
        '<section id="segments"><h2>Segments</h2>'
        "<p>Where the miscalibration actually lives. Bars are sorted worst-first; click one to "
        "see that segment's own reliability curve.</p>"
        + segment_charts.render_segments_section(view)
        + figures
        + "</section>"
    )


def _data_quality_section(model: ReportModel) -> str:
    quality = model.data_quality
    rows = list(quality.rows)
    table = details_table(
        ("Field", "Value"), rows, summary="Sample and label state", numeric_from=1
    )
    if quality.sparse_bins:
        sparse = (
            f'<p class="warn"><strong>{quality.sparse_bins} of {quality.total_bins} confidence '
            "bins hold fewer than 30 labeled decisions.</strong> Read those bins as a direction, "
            "not a number.</p>"
        )
    else:
        sparse = (
            f'<p class="note">No confidence bin holds fewer than 30 labeled decisions '
            f"({quality.total_bins} bins).</p>"
        )
    limitations = quality.limitations or DEFAULT_LIMITATIONS
    items = "".join(f"<li>{escape(item)}</li>" for item in limitations)
    return (
        '<section id="data-quality"><h2>Data quality</h2>'
        f"{table}{sparse}"
        f'<h3>Limitations</h3><ul class="limits">{items}</ul>'
        "</section>"
    )


def write_report(
    html: str,
    path: Path | str,
) -> Path:
    """Write the document to disk. Callers keep the path they pass in."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")
    return target


def build_blocks(
    dataset: DatasetReport, *, thresholds: Mapping[str, float] | None = None
) -> list[ReliabilityBlock]:
    """One block per question, carrying the threshold that applies to it."""
    thresholds = dict(thresholds or {})
    blocks: list[ReliabilityBlock] = []
    for question in dataset.questions:
        block_threshold = thresholds.get(question.question_key)
        silver_note = ""
        if question.silver_metrics is not None and question.silver_metrics.n > 0:
            silver_note = (
                f"Silver labels are measured separately: n={question.silver_metrics.n}, "
                f"ECE {S.fmt(question.silver_metrics.ece, 3)}. Agreement with a model is not accuracy."
            )
        blocks.append(
            ReliabilityBlock(
                question_key=question.question_key,
                metrics=question.metrics,
                diagnosis=question.diagnosis,
                threshold=block_threshold,
                subtitle=f"{question.question_type} · {question.n_records} records · "
                f"{question.n_unlabeled} unlabeled",
                silver_note=silver_note,
            )
        )
    return blocks


def data_quality_from(dataset: DatasetReport, *, sparse_threshold: int = 30) -> object:
    """Assemble the data-quality rows from an evaluated dataset."""
    from jeval.report.model import DataQuality

    sources: dict[str, int] = {}
    for question in dataset.questions:
        if question.silver_metrics is not None:
            sources["silver"] = sources.get("silver", 0) + question.silver_metrics.n
    gold = dataset.n_labeled_gold - sources.get("silver", 0)
    sparse = sum(1 for b in dataset.overall.bins if b.n < sparse_threshold)
    rows = (
        ("Records", f"{dataset.n_records:,}"),
        ("Labeled (gold)", f"{gold:,} of {dataset.n_records:,}"),
        ("Labeled (silver)", f"{dataset.n_labeled_silver:,}"),
        ("Unlabeled", f"{dataset.n_unlabeled:,}"),
        ("Excluded score records", f"{dataset.n_score_excluded:,}"),
        ("Questions", f"{dataset.n_questions}"),
        ("Models seen", ", ".join(dataset.models) or "unknown"),
        (
            "Binning",
            f"{dataset.overall.binning}, {len(dataset.overall.bins)} bins "
            f"({dataset.overall.bins_requested} requested)",
        ),
        ("Bins under 30 labeled", f"{sparse} of {len(dataset.overall.bins)}"),
    )
    return DataQuality(
        rows=rows,
        limitations=DEFAULT_LIMITATIONS,
        sparse_bins=sparse,
        total_bins=len(dataset.overall.bins),
        silver_only=dataset.silver_only,
        no_gold_labels=dataset.n_labeled_gold == 0,
    )
