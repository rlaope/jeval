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
from pathlib import Path

from jeval.calibration import CalibrationMetrics
from jeval.currency import format_amount, format_delta
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
from jeval.report.svg import escape


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
    # "now" means what is deployed, never the recommendation: marking the recommended threshold
    # as the current one would show the gap as zero and quietly flatter the report.
    current_thresholds: dict[str, float] = {}
    if thresholds and model.impact is not None and model.impact.current_threshold:
        current_thresholds[thresholds[0].action] = model.impact.current_threshold
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

    sections = _section_numbers(model, thresholds)
    parts: list[str] = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<meta name="color-scheme" content="light dark">',
        "<title>jeval report</title>",
        f"<style>{minify(REPORT_CSS)}</style>",
        "</head>",
        "<body>",
        '<main class="report">',
    ]
    parts.append(_header(model))
    parts.append(_toc(sections))
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

    parts.append(_verdict_section(model, summary, thresholds, current_thresholds))
    parts.append(_reliability_section(blocks, sections))
    parts.append(_cost_section(model, current_thresholds, thresholds, sections))
    parts.append(_segments_section(model, segment_metrics, sections))
    if model.score is not None:
        parts.append(_score_section(model, sections))
    if model.label_plan or model.recalibration is not None:
        parts.append(_label_plan_section(model, sections))
    if model.drift is not None and model.drift.slices:
        parts.append(
            '<section id="drift">'
            + _h2("drift", sections)
            + drift_charts.render_drift_section(
                model.drift,
                baseline_metrics=baseline_metrics,
                current_metrics=current_metrics,
                threshold=thresholds[0].threshold if thresholds else None,
            )
            + "</section>"
        )
    parts.append(_data_quality_section(model, sections))
    parts.append(
        "<footer><p>Generated by <strong>jeval</strong>. One file: no server, no external "
        "assets, no network request. Every number comes from the decision records on the machine "
        "that wrote it.</p>"
        "<p>Thresholds here are for exploration. <code>jeval threshold</code> confirms a value "
        "and writes <code>thresholds.yaml</code>.</p></footer>"
    )
    parts.append("</main>")
    parts.append(S.embed_json(payload, element_id="jeval-data"))
    parts.append(f"<script>{minify(REPORT_JS)}</script>")
    parts.append("</body></html>")
    return "".join(parts)


# Section id -> heading, in reading order. The verdict has no heading of its own: it is the
# answer, and the headline inside it is the largest type on the page after the title.
_SECTION_TITLES: tuple[tuple[str, str], ...] = (
    ("reliability", "Reliability"),
    ("cost", "Cost and threshold"),
    ("segments", "Segments"),
    ("score", "Score questions"),
    ("labels", "Labels and correction"),
    ("drift", "Drift"),
    ("data-quality", "Data quality"),
)


def _section_numbers(
    model: ReportModel, thresholds: Sequence[ThresholdResult]
) -> dict[str, tuple[int, str]]:
    """Number the sections this report actually has, so the index never skips a digit."""
    present = {
        "reliability": True,
        "cost": True,
        "segments": True,
        "score": model.score is not None,
        "labels": bool(model.label_plan) or model.recalibration is not None,
        "drift": model.drift is not None and bool(model.drift.slices),
        "data-quality": True,
    }
    numbered: dict[str, tuple[int, str]] = {}
    for key, title in _SECTION_TITLES:
        if present[key]:
            numbered[key] = (len(numbered) + 1, title)
    return numbered


def _h2(key: str, sections: Mapping[str, tuple[int, str]]) -> str:
    number, title = sections[key]
    return f'<h2><span class="idx">{number:02d}</span>{escape(title)}</h2>'


def _toc(sections: Mapping[str, tuple[int, str]]) -> str:
    links = "".join(
        f'<a href="#{key}"><span class="idx">{number:02d}</span>{escape(title)}</a>'
        for key, (number, title) in sections.items()
    )
    return f'<nav class="toc" aria-label="Sections">{links}</nav>'


def _provenance_rows(model: ReportModel) -> list[tuple[str, str, bool]]:
    """``(label, value, is_path)`` for the masthead, from the model's provenance notes.

    ``source_note`` arrives as ``"source: <path> · costs: <path>"``; each part becomes its own row,
    so a long path wraps inside its own cell instead of dragging the next fact onto a new line.
    """
    rows: list[tuple[str, str, bool]] = []
    if model.generated_at:
        rows.append(("Generated", model.generated_at, False))
    for part in (piece.strip() for piece in model.source_note.split(" · ")):
        if not part:
            continue
        label, sep, value = part.partition(": ")
        if sep and label and " " not in label:
            rows.append((label.capitalize(), value, "/" in value or "\\" in value))
        else:
            rows.append(("Source", part, False))
    if model.demo_note:
        rows.append(("Data", f"Synthetic demo data. {model.demo_note}", False))
    return rows


def _header(model: ReportModel) -> str:
    """Title, one line of intent, and the facts a reader needs to judge the sample.

    The provenance is a definition list rather than one ``·``-separated line: the demo note is a
    paragraph's worth of text, and a single run-on line is read as boilerplate and skipped.
    """
    path_class = ' class="path"'
    rows = "".join(
        f"<dt>{escape(label)}</dt><dd{path_class if is_path else ''}>{escape(value)}</dd>"
        for label, value, is_path in _provenance_rows(model)
    )
    return (
        '<header class="masthead">'
        '<p class="brand"><span class="mark">jeval</span>calibration and threshold report</p>'
        "<h1>jeval report</h1>"
        '<p class="lede">What this classifier&rsquo;s confidence is worth, and where the line '
        "between the machine deciding and a human deciding belongs.</p>"
        + (f'<dl class="provenance">{rows}</dl>' if rows else "")
        + "</header>"
    )


# One word for the status, so the verdict never says "too low" in colour alone.
_STATUS_LABELS: dict[str, str] = {
    "clear": "No change needed",
    "too_low": "Threshold too low",
    "too_high": "Threshold too high",
    "well_calibrated": "Well calibrated",
    "insufficient_data": "Not enough labels",
}


def _stat_swatch(label: str) -> str:
    lowered = label.lower()
    if "current" in lowered or "in use" in lowered:
        return '<span class="swatch current" aria-hidden="true"></span>'
    if "recommended" in lowered:
        return '<span class="swatch recommended" aria-hidden="true"></span>'
    return ""


def _verdict_section(
    model: ReportModel,
    summary: str,
    thresholds: Sequence[ThresholdResult] = (),
    current_thresholds: Mapping[str, float] | None = None,
) -> str:
    verdict = model.verdict
    cards = "".join(
        '<div class="stat"><div class="k">'
        + _stat_swatch(stat.label)
        + escape(stat.label)
        + '</div><div class="v num">'
        + escape(stat.value)
        + "</div>"
        + (f'<div class="s">{escape(stat.sub)}</div>' if stat.sub else "")
        + "</div>"
        for stat in verdict.stats
    )
    ruler = ""
    if thresholds and thresholds[0].curve:
        first = thresholds[0]
        current = (current_thresholds or {}).get(first.action)
        # Two drawings of one ruler: a chart scaled down to a phone is a chart whose labels are
        # five pixels tall, so the narrow screen gets a narrow drawing instead.
        wide = cost_charts.render_threshold_ruler(first, current_threshold=current)
        narrow = cost_charts.render_threshold_ruler(
            first, current_threshold=current, width=340.0, cls="ruler-svg narrow"
        )
        if wide:
            ruler = f'<figure class="ruler">{wide}{narrow}</figure>'
    status = _STATUS_LABELS.get(verdict.status, verdict.status.replace("_", " "))
    return (
        '<section id="verdict" class="verdict" data-status="'
        + escape(verdict.status)
        + '">'
        + f'<p class="status">{escape(status)}</p>'
        + f'<p class="headline">{escape(verdict.headline)}</p>'
        + f'<p class="detail">{escape(verdict.detail)}</p>'
        + ruler
        + (f'<div class="stats">{cards}</div>' if cards else "")
        + '<div class="verdict-actions">'
        + '<button type="button" id="copy-summary" class="copy" data-jeval-copy-summary '
        + 'data-copy-target="jeval-data">Copy summary</button>'
        + "</div>"
        + "</section>"
    )


def _reliability_section(
    blocks: Sequence[ReliabilityBlock], sections: Mapping[str, tuple[int, str]]
) -> str:
    head = '<section id="reliability">' + _h2("reliability", sections)
    if not blocks:
        return head + '<p class="note">No questions found.</p></section>'
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
        body = [
            '<div class="block-head">'
            f"<h3>{escape(block.question_key)}</h3>"
            + (f'<p class="block-sub">{escape(block.subtitle)}</p>' if block.subtitle else "")
            + "</div>",
            f'<p class="diag">{escape(block.diagnosis)}</p>',
            reliability_charts.render_reliability_section(
                block.metrics,
                threshold=block.threshold,
                title=f"Reliability · {block.question_key}",
            ),
        ]
        if block.silver_note:
            body.append(f'<p class="note">{escape(block.silver_note)}</p>')
        figures.append(
            f'<div class="question-block{" is-active" if index == 0 else ""}" '
            f'data-jeval-question="{escape(block.question_key)}">' + "".join(body) + "</div>"
        )
    return (
        head
        + '<p class="intro">Stated confidence against what actually happened. Points below the '
        "diagonal mean the model was more confident than it was right.</p>"
        + tabs
        + "".join(figures)
        + "</section>"
    )


def _num(value: float, text: str) -> str:
    return "–" if value != value else text


def _cost_section(
    model: ReportModel,
    current_thresholds: Mapping[str, float],
    thresholds: Sequence[ThresholdResult],
    sections: Mapping[str, tuple[int, str]],
) -> str:
    head = '<section id="cost">' + _h2("cost", sections)
    if not thresholds:
        return (
            head + '<p class="note">No cost matrix was supplied, so no threshold was computed. '
            "Add <code>costs.yaml</code> (see <code>costs.example.yaml</code>) and run "
            "<code>jeval threshold</code>.</p></section>"
        )
    body = "".join(
        '<h3 class="action-head">'
        f'<span class="ident">{escape(result.action)}</span>'
        + (
            f' <span class="block-sub">{escape(result.question)} = {escape(result.when)} · '
            f"{result.n_records:,} labeled decisions</span>"
            if result.question
            else ""
        )
        + "</h3>"
        + cost_charts.render_cost_section(
            result,
            current_threshold=current_thresholds.get(result.action),
            impact=model.impacts.get(result.action) or model.impact,
            slider_id=f"threshold-slider-{_slug(result.action)}",
        )
        for result in thresholds
    )
    splits = ""
    currency = model.impact.currency if model.impact is not None else "USD"
    if model.segment_thresholds:
        rows = "".join(
            f'<tr><td class="ident">{escape(row.label)}</td>'
            f'<td class="num">{_num(row.threshold, f"{row.threshold:.2f}")}</td>'
            f'<td class="num">{_num(row.cost_per_case, format_amount(row.cost_per_case, currency, unit=False))}</td>'
            f'<td class="num">{_num(row.delta, format_delta(row.delta, currency, unit=False))}</td>'
            f'<td class="num">{row.n:,}</td>'
            + (
                '<td><span class="tag yes">split</span></td></tr>'
                if row.worth_splitting
                else f'<td class="prose">{escape(row.reason or "splitting does not pay")}</td></tr>'
            )
            for row in model.segment_thresholds
        )
        splits = (
            "<h3>Does one threshold fit every segment?</h3>"
            '<p class="intro">A split is recommended only when a segment&rsquo;s own optimum moves '
            "by more than one sweep step <em>and</em> adopting it changes cost per case by more "
            "than 2%. Everything else is the same threshold with extra machinery, and this table "
            "says so.</p>"
            '<div class="table-wrap"><table class="details">'
            "<caption>Segment optimum against the global optimum</caption>"
            '<thead><tr><th>Segment</th><th class="num">Threshold</th>'
            f'<th class="num">Cost / case ({escape(currency)})</th>'
            f'<th class="num">vs global</th><th class="num">n</th><th>Verdict</th></tr></thead>'
            f"<tbody>{rows}</tbody></table></div>"
        )
    return (
        head
        + '<p class="intro">Expected cost per case for every candidate threshold. The minimum is '
        "the recommendation; the flat region is where the data cannot tell neighbouring "
        "thresholds apart.</p>" + body + splits + "</section>"
    )


def _segments_section(
    model: ReportModel,
    segment_metrics: Mapping[str, CalibrationMetrics],
    sections: Mapping[str, tuple[int, str]],
) -> str:
    head = '<section id="segments">' + _h2("segments", sections)
    view = model.segments
    if view is None or not view.bars:
        return (
            head + '<p class="note">No segment breakdown was requested. Pass '
            "<code>--by lang</code> (and others) to break the numbers down.</p></section>"
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
            f'<figure class="seg-figure plate" id="seg-fig-{escape(slug)}" '
            f'data-jeval-segment-chart="{escape(slug)}" hidden>'
            f"{reliability_charts.render_reliability(metrics, title=f'{bar.label}')}"
            "</figure>"
        )
    return (
        head
        + '<p class="intro">Where the miscalibration actually lives. Bars are sorted worst-first; '
        "click one to see that segment&rsquo;s own reliability curve.</p>"
        + segment_charts.render_segments_section(view)
        + figures
        + "</section>"
    )


def _score_section(model: ReportModel, sections: Mapping[str, tuple[int, str]]) -> str:
    """Score-type questions: error and rank agreement, deliberately not accuracy."""
    view = model.score
    if view is None:  # pragma: no cover - guarded by the caller
        return ""
    head = '<section id="score">' + _h2("score", sections)
    if view.n < 2:
        return (
            head + f'<p class="note">This log has {view.n} usable numeric score decision(s): too '
            "few to measure error or rank agreement. Score answers are never folded into binary "
            "accuracy, so they are counted and set aside instead.</p></section>"
        )
    rows = "".join(
        f'<tr><td class="num l">{level.lo:.2f}–{level.hi:.2f}</td><td class="num">{level.n}</td>'
        f'<td class="num">{level.mean_predicted:.3f}</td>'
        f'<td class="num">{level.mean_actual:.3f}</td>'
        f'<td class="num">{level.mean_actual - level.mean_predicted:+.3f}</td></tr>'
        for level in view.levels
    )
    table = (
        '<table class="details"><caption>Predicted band, and the actual values inside it</caption>'
        '<thead><tr><th>Predicted</th><th class="num">n</th>'
        '<th class="num">Mean predicted</th><th class="num">Mean actual</th>'
        f'<th class="num">Gap</th></tr></thead><tbody>{rows}</tbody></table>'
    )
    counted = (
        f" Counted and excluded from this section: {view.n_other_type} non-score, "
        f"{view.n_unlabeled} unlabeled, {view.n_unparseable} unparseable."
        if (view.n_other_type or view.n_unlabeled or view.n_unparseable)
        else ""
    )
    rho = "–" if view.spearman_rho != view.spearman_rho else f"{view.spearman_rho:.3f}"
    note = f'<p class="note">{view.n} usable numeric decision(s).{counted}</p></section>'
    return (
        head + '<p class="intro">A numeric answer is measured as <strong>error</strong> and '
        "<strong>rank agreement</strong>, never as right-or-wrong: a model can be monotone and "
        "still be off by a constant, and binary accuracy would call that a failure at every level "
        "while rank correlation calls it a success. Read the gap column: a systematically "
        "non-zero gap is bias, not noise.</p>"
        '<div class="stats small">'
        f'<div class="stat"><div class="k">MAE</div><div class="v num">{view.mae:.3f}</div></div>'
        f'<div class="stat"><div class="k">RMSE</div><div class="v num">{view.rmse:.3f}</div></div>'
        '<div class="stat"><div class="k">Spearman rho</div>'
        f'<div class="v num">{rho}</div></div>'
        '<div class="stat"><div class="k">Decisions</div>'
        f'<div class="v num">{view.n}</div></div>'
        "</div>"
        f"{table}{note}"
    )


def _label_plan_section(model: ReportModel, sections: Mapping[str, tuple[int, str]]) -> str:
    """What more labels would buy, and what a correction would (or would not) do."""
    parts = ['<section id="labels">', _h2("labels", sections)]
    if model.label_plan:
        rows = "".join(
            f'<tr><td>{escape(row.scope)}</td><td class="ident">{escape(row.key)}</td>'
            f'<td class="num">{row.n_now:,}</td>'
            f'<td class="num">{_num(row.ece, f"{row.ece:.3f}")}</td>'
            f'<td class="num">{_num(row.ci_width, f"{row.ci_width:.3f}")}</td>'
            f'<td class="prose">{escape(row.needed or row.reason)}</td></tr>'
            for row in model.label_plan
        )
        parts.append(
            '<p class="intro">An interval is the honest limit of what this sample can say. More '
            "labels are the only way to narrow it, and the projection below is an estimate from "
            "your own data, not a measurement.</p>"
            '<div class="table-wrap"><table class="details">'
            "<caption>Additional labels needed for a tighter interval</caption>"
            '<thead><tr><th>Scope</th><th>Key</th><th class="num">n now</th>'
            '<th class="num">ECE</th><th class="num">CI width</th>'
            f"<th>Needed (target width: labels)</th></tr></thead><tbody>{rows}</tbody></table></div>"
        )
    if model.recalibration is not None:
        view = model.recalibration
        if view.helps:
            verdict = (
                f"<strong>A correction is available:</strong> {escape(view.method)}, ECE "
                f"{view.before_ece:.3f} → {view.after_ece:.3f} cross-validated on {view.n:,} "
                "labeled decisions. jeval exports the map; your application applies it."
            )
        else:
            verdict = (
                f"<strong>No correction is warranted.</strong> {escape(view.method)} would move "
                f"ECE {view.before_ece:.3f} → {view.after_ece:.3f} cross-validated, which is not "
                f"a gain worth shipping. {escape(view.note)}"
            )
        parts.append(f'<h3>Recalibration</h3><p class="diag">{verdict}</p>')
    parts.append("</section>")
    return "".join(parts)


def _slug(value: str) -> str:
    """A DOM-safe id fragment for one action's controls, so two actions cannot share an id."""
    return "".join(character if character.isalnum() else "-" for character in value).strip("-")


def _data_quality_section(model: ReportModel, sections: Mapping[str, tuple[int, str]]) -> str:
    quality = model.data_quality
    rows = "".join(
        f'<tr><td>{escape(label)}</td><td class="num">{escape(value)}</td></tr>'
        for label, value in quality.rows
    )
    table = (
        f'<table class="kv"><caption>Sample and label state</caption><tbody>{rows}</tbody></table>'
        if rows
        else ""
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
        '<section id="data-quality">'
        + _h2("data-quality", sections)
        + f"{table}{sparse}"
        + f'<h3>Limitations</h3><ul class="limits">{items}</ul>'
        + "</section>"
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
    dataset: DatasetReport,
    *,
    thresholds: Mapping[str, float] | None = None,
    actions: Mapping[str, str] | None = None,
) -> list[ReliabilityBlock]:
    """One block per question, carrying the threshold that applies to it.

    ``actions`` names the action each question's line belongs to. Every question has its own line
    and several questions can share an action, so the number on the chart is not always the number
    in the verdict: without the action's name beside it, a reader sees two thresholds and assumes
    one of them is wrong.
    """
    thresholds = dict(thresholds or {})
    actions = dict(actions or {})
    blocks: list[ReliabilityBlock] = []
    for question in dataset.questions:
        block_threshold = thresholds.get(question.question_key)
        line_note = ""
        if block_threshold is not None:
            line_note = f" · line in use {S.fmt(block_threshold)}"
            if actions.get(question.question_key):
                line_note += f" ({actions[question.question_key]})"
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
                subtitle=(
                    f"{question.question_type} · {question.n_records} records · "
                    f"{question.n_unlabeled} unlabeled{line_note}"
                ),
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
