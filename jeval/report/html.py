"""Single-file HTML report. No server, no CDN, no external assets."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path

from jeval.calibration import CalibrationMetrics
from jeval.evaluate import DatasetReport, QuestionReport, SegmentReport
from jeval.report.svg import histogram, metric_bar, reliability_diagram
from jeval.synth import DemoDataset

STYLE = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; padding: 40px 24px; background: #fff; color: #111;
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width: 960px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 6px; letter-spacing: -0.01em; }
h2 { font-size: 16px; margin: 44px 0 12px; padding-bottom: 6px; border-bottom: 1px solid #e6e6e6; }
h3 { font-size: 14px; margin: 28px 0 10px; }
p, li { margin: 0 0 8px; }
.meta { color: #555; font-size: 13px; margin-bottom: 22px; }
.meta code { color: #111; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin: 16px 0 8px; }
.card { border: 1px solid #e6e6e6; padding: 14px 16px; }
.card .k { font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.04em; }
.card .v { font: 600 22px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace; margin-top: 6px; }
.card .s { font-size: 12px; color: #666; margin-top: 4px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0 18px; font-size: 13px; }
th, td { border-bottom: 1px solid #eee; padding: 7px 10px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { color: #555; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.03em; }
td.num, th.num { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.diag { background: #fafafa; border-left: 3px solid #111; padding: 10px 14px; margin: 10px 0 18px; font-size: 14px; }
.warn { background: #fff8e6; border-left: 3px solid #b58900; padding: 12px 14px; margin: 14px 0 18px; font-size: 14px; }
.note { color: #666; font-size: 13px; }
figure { margin: 12px 0 20px; }
figcaption { color: #666; font-size: 12px; margin-top: 6px; }
footer { margin-top: 56px; border-top: 1px solid #e6e6e6; padding-top: 18px; color: #555; font-size: 13px; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }
"""


def _pct(value: float, digits: int = 1) -> str:
    if value != value:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def _num(value: float, digits: int = 3) -> str:
    if value != value:
        return "n/a"
    return f"{value:.{digits}f}"


def _metrics_cards(metrics: CalibrationMetrics, *, labeled: int, unlabeled: int) -> str:
    cards = [
        (
            "ECE",
            _num(metrics.ece),
            f"95% CI {_num(metrics.ece_ci_low)}-{_num(metrics.ece_ci_high)}",
        ),
        ("MCE", _num(metrics.mce), "largest single-bin gap"),
        ("Brier", _num(metrics.brier), "mean squared confidence error"),
        ("Labeled", f"{labeled}", f"{unlabeled} unlabeled, excluded"),
        (
            "Bins",
            f"{len(metrics.bins)}",
            f"{metrics.binning} binning, {metrics.bins_requested} requested",
        ),
        ("Accuracy", _pct(_overall_accuracy(metrics)), "over labeled records in scope"),
    ]
    items = "".join(
        f'<div class="card"><div class="k">{escape(k)}</div><div class="v">{escape(v)}</div>'
        f'<div class="s">{escape(s)}</div></div>'
        for k, v, s in cards
    )
    return f'<div class="grid">{items}</div>'


def _overall_accuracy(metrics: CalibrationMetrics) -> float:
    total = sum(b.n for b in metrics.bins)
    if total == 0:
        return float("nan")
    return sum(b.accuracy * b.n for b in metrics.bins) / total


def _bin_table(metrics: CalibrationMetrics) -> str:
    if not metrics.bins:
        return '<p class="note">No labeled records: no bins to show. Ingest labeled decisions first.</p>'
    rows = []
    for cal_bin in metrics.bins:
        direction = (
            "overconfident"
            if cal_bin.gap < -0.02
            else ("underconfident" if cal_bin.gap > 0.02 else "ok")
        )
        rows.append(
            "<tr>"
            f'<td class="num">{escape(cal_bin.label)}</td>'
            f'<td class="num">{cal_bin.n}</td>'
            f'<td class="num">{_num(cal_bin.mean_confidence)}</td>'
            f'<td class="num">{_pct(cal_bin.accuracy)}</td>'
            f'<td class="num">[{_pct(cal_bin.ci_low, 0)}, {_pct(cal_bin.ci_high, 0)}]</td>'
            f'<td class="num">{cal_bin.gap:+.3f}</td>'
            f"<td>{escape(direction)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        '<th>Confidence bin</th><th class="num">n</th><th class="num">Mean confidence</th>'
        '<th class="num">Observed accuracy</th><th class="num">Wilson 95%</th>'
        '<th class="num">Gap</th><th>Reading</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _question_section(question: QuestionReport) -> str:
    parts = [
        f"<h3>{escape(question.question_key)} <span class='note'>({escape(question.question_type)})</span></h3>",
        f'<div class="diag">{escape(question.diagnosis)}</div>',
    ]
    if question.metrics.n == 0:
        parts.append(
            f'<p class="note">No labeled decisions for this question '
            f"({question.n_records} records, {question.n_unlabeled} unlabeled). "
            f"Labels are what this report runs on.</p>"
        )
    else:
        parts.append(
            f"<figure>{reliability_diagram(question.metrics, title=question.question_key)}"
            "<figcaption>Points below the diagonal are overconfidence. "
            "Vertical lines are Wilson 95% intervals.</figcaption></figure>"
        )
        parts.append(_bin_table(question.metrics))
    if question.n_score_excluded:
        parts.append(
            f'<p class="note">{question.n_score_excluded} score-type records excluded from '
            "binary accuracy: a score is not right or wrong, it is near or far.</p>"
        )
    if question.silver_metrics is not None and question.silver_metrics.n > 0:
        parts.append(
            f'<p class="note">Silver labels kept separate: n={question.silver_metrics.n}, '
            f"ECE {_num(question.silver_metrics.ece)}. Agreement with a model is not accuracy.</p>"
        )
    return "".join(parts)


def _segment_table(segments: list[SegmentReport]) -> str:
    if not segments:
        return (
            '<p class="note">No segment breakdown requested, or no segment had enough labels.</p>'
        )
    rows = []
    for segment in segments:
        rows.append(
            "<tr>"
            f"<td>{escape(segment.key)}</td>"
            f"<td>{escape(segment.value)}</td>"
            f'<td class="num">{segment.metrics.n}</td>'
            f'<td class="num">{_num(segment.metrics.ece)}</td>'
            f'<td class="num">[{_num(segment.metrics.ece_ci_low)}, {_num(segment.metrics.ece_ci_high)}]</td>'
            f'<td class="num">{_num(segment.metrics.ece_ci_span)}</td>'
            f'<td class="num">{_pct(_overall_accuracy(segment.metrics))}</td>'
            "</tr>"
        )
    return (
        '<table><thead><tr><th>Segment key</th><th>Value</th><th class="num">n</th>'
        '<th class="num">ECE</th><th class="num">ECE 95% CI</th>'
        '<th class="num">CI width</th><th class="num">Accuracy</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table>"
    )


def render_report(
    report: DatasetReport,
    *,
    generated_at: datetime | None = None,
    source_note: str = "",
    demo_description: str = "",
) -> str:
    """Render the whole report as one dependency-free HTML document."""
    stamp = (generated_at or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M:%SZ")
    models = ", ".join(report.models) if report.models else "unknown"
    parts: list[str] = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>jeval report</title>",
        f"<style>{STYLE}</style>",
        "</head><body><main>",
        "<h1>jeval report</h1>",
        f'<p class="meta">Generated {escape(stamp)} &middot; models <code>{escape(models)}</code> '
        f"&middot; {report.n_records} records &middot; {report.n_questions} questions</p>",
    ]
    if source_note:
        parts.append(f'<p class="meta">{escape(source_note)}</p>')
    if demo_description:
        parts.append(f'<p class="meta">Synthetic demo data: {escape(demo_description)}</p>')

    if report.silver_only:
        parts.append(
            '<div class="warn"><strong>Every label here is silver.</strong> Silver labels record '
            "agreement with a model, not whether the answer was right. Read the numbers below as "
            "an agreement rate, and collect human labels before acting on them.</div>"
        )
    if report.n_labeled_gold == 0:
        parts.append(
            '<div class="warn"><strong>No gold labels.</strong> This tool measures nothing until '
            "humans label decisions. A free source of labels is the human answer for every case "
            "that was escalated.</div>"
        )

    parts.append("<h2>Headline</h2>")
    parts.append(
        _metrics_cards(report.overall, labeled=report.n_labeled_gold, unlabeled=report.n_unlabeled)
    )
    parts.append(f'<div class="diag">{escape(report.overall_diagnosis)}</div>')
    if report.overall.n:
        parts.append(
            f'<figure><div style="display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start">'
            f"{reliability_diagram(report.overall, title='all questions, gold labels')}"
            f"<div>{metric_bar('ECE', report.overall.ece, (report.overall.ece_ci_low, report.overall.ece_ci_high))}"
            f'<p class="note">ECE {_num(report.overall.ece)} (95% CI '
            f"{_num(report.overall.ece_ci_low)}-{_num(report.overall.ece_ci_high)}). "
            "The CI is bootstrapped: a wide interval means the sample cannot yet support a firm "
            "claim, however confident the point estimate looks.</p></div></div>"
            "<figcaption>All gold-labeled decisions pooled. Per-question views follow.</figcaption></figure>"
        )
    parts.append(_bin_table(report.overall))

    parts.append("<h2>By question</h2>")
    if not report.questions:
        parts.append('<p class="note">No questions found.</p>')
    for question in report.questions:
        parts.append(_question_section(question))

    if report.segment_keys:
        parts.append("<h2>By segment</h2>")
        parts.append(_segment_table(report.segments))

    parts.append("<h2>Counts</h2>")
    parts.append(
        "<table><thead><tr><th>Bucket</th><th class='num'>n</th><th>Meaning</th></tr></thead><tbody>"
        f"<tr><td>Records</td><td class='num'>{report.n_records}</td><td>every ingested decision</td></tr>"
        f"<tr><td>Gold labels</td><td class='num'>{report.n_labeled_gold}</td>"
        "<td>human review or human override</td></tr>"
        f"<tr><td>In binary metrics</td><td class='num'>{report.overall.n}</td>"
        "<td>gold labels on choice and noul questions; score types are excluded</td></tr>"
        f"<tr><td>Silver labels</td><td class='num'>{report.n_labeled_silver}</td>"
        "<td>model-generated, reported separately, never pooled</td></tr>"
        f"<tr><td>Unlabeled</td><td class='num'>{report.n_unlabeled}</td>"
        "<td>no ground truth available; excluded from metrics</td></tr>"
        f"<tr><td>Score excluded</td><td class='num'>{report.n_score_excluded}</td>"
        "<td>score-type records, not scored as right or wrong</td></tr>"
        "</tbody></table>"
    )
    if report.silver_metrics is not None and report.silver_metrics.n > 0:
        parts.append("<h2>Silver labels (separate)</h2>")
        parts.append(
            "<p>Measured apart from the gold population on purpose. A silver label answers "
            '"did another model agree", which is a different question from "was this right".</p>'
        )
        parts.append(_bin_table(report.silver_metrics))

    parts.append("<h2>How to read this</h2>")
    parts.append(
        "<ul>"
        "<li>ECE is the sample-weighted average gap between claimed confidence and observed "
        "accuracy. Zero is perfect; lower is better.</li>"
        "<li>Bin intervals are Wilson 95% intervals. A bin with 12 records cannot support a "
        "threshold move on its own.</li>"
        "<li>Confidence is what the model said. Accuracy is what happened. This report never "
        "assumes they match.</li>"
        "<li>Nothing here validates a vendor's calibration claim in general; it measures the "
        "model on the records you supplied.</li>"
        "</ul>"
    )
    parts.append(
        "<footer>jeval 'measure the line, do not guess it' &middot; single-file report, no "
        "server, no external assets &middot; generated from decision records on this machine.</footer>"
    )
    parts.append("</main></body></html>")
    return "".join(parts)


def write_report(
    report: DatasetReport,
    out_path: Path | str,
    *,
    generated_at: datetime | None = None,
    source_note: str = "",
    demo: DemoDataset | None = None,
) -> Path:
    """Write the report to a single HTML file."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    html = render_report(
        report,
        generated_at=generated_at,
        source_note=source_note,
        demo_description=demo.description if demo else "",
    )
    target.write_text(html, encoding="utf-8")
    return target


def score_section(values: list[float]) -> str:
    """Score-type records get a distribution view, never a right/wrong accuracy."""
    return (
        "<h3>Score distribution</h3>"
        '<p class="note">Score-type records are shown as a distribution. Calibration for a score '
        "is the level-by-level comparison of predicted and observed distributions, which arrives "
        "with the threshold milestone; right/wrong accuracy is the wrong frame here.</p>"
        + histogram(values)
    )
