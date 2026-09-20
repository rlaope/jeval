"""Wiring tests: the report's JavaScript and the template must agree.

The bug this file exists to prevent: the chart and template modules were written in parallel with
the inline JavaScript, and the script reads ``data-jeval-*`` hooks that the template never emitted.
Nothing failed — the document built, the tests passed, and every interactive control was silently
inert. Markup that only matters when JavaScript runs needs a test that treats the attribute names
as a contract.
"""

from __future__ import annotations

import re
from pathlib import Path

from jeval import report as report_package
from jeval.evaluate import evaluate
from jeval.report import template
from jeval.report.assets import REPORT_CSS
from jeval.report.charts import segments as segment_charts
from jeval.report.model import (
    CostPoint,
    ImpactRow,
    ImpactTable,
    SegmentView,
    ThresholdResult,
    Verdict,
    VerdictStat,
)
from jeval.synth import SynthSpec, generate

# Hooks the script reads from the document. Every one of these must exist in a rendered report.
REQUIRED_HOOKS: frozenset[str] = frozenset(
    {
        "data-jeval-tab",
        "data-jeval-question",
        "data-jeval-action",
        "data-jeval-threshold",
        "data-jeval-value",
        "data-jeval-copy-summary",
        "data-jeval-segment",
        "data-jeval-segment-chart",
    }
)

# Hooks the script writes back onto elements it already owns, so the template does not emit them.
JS_INTERNAL_HOOKS: frozenset[str] = frozenset({"data-jeval-label", "data-jeval-active"})

FIGURE_KEYS: frozenset[str] = frozenset(
    {
        "auto_rate",
        "accuracy_auto",
        "cost_per_case",
        "cost_per_month",
        "auto_per_month",
        "escalations_per_month",
    }
)


def _threshold_result() -> ThresholdResult:
    points = tuple(
        CostPoint(
            threshold=index / 20.0,
            expected_cost=1000.0 - 200.0 * (index / 20.0) + 40.0 * (index / 20.0) ** 2,
            auto_rate=max(0.0, 1.0 - index / 20.0),
            accuracy_auto=0.7 + 0.005 * index,
            accuracy_escalated=0.8,
            accept_cost=10.0 * index,
            escalate_cost=5.0 * index,
            n_auto=index,
            n_escalate=20 - index,
            n_wrong_auto=index // 3,
            n_wrong_escalate=1,
        )
        for index in range(21)
    )
    return ThresholdResult(
        action="auto_refund",
        question="intent",
        when="refund_request",
        threshold=0.68,
        expected_cost_per_case=880.0,
        auto_rate=0.32,
        accuracy_auto=0.94,
        ci_low=0.6,
        ci_high=0.75,
        curve=points,
        flat_region=(0.6, 0.72),
        n_records=500,
        models=("jev-1.13.0",),
    )


def _impact() -> ImpactTable:
    return ImpactTable(
        rows=(
            ImpactRow("confidence threshold", "0.60", "0.68", "+0.08"),
            ImpactRow("auto rate", "100%", "32%", "-68 pt"),
            ImpactRow("accuracy (auto)", "79%", "94%", "+15 pt"),
            ImpactRow("cost per case", "1,500", "880", "-41%"),
        ),
        monthly_volume=20_000.0,
        currency="KRW",
        current_threshold=0.60,
        recommended_threshold=0.68,
    )


def _document() -> str:
    """A document with every interactive feature present: tabs, slider, segments."""
    records = generate(SynthSpec(n=200, mode="calibrated", seed=1, question_key="department")) + (
        generate(SynthSpec(n=200, mode="inflated", inflation=1.2, seed=2, question_key="intent"))
    )
    for index, record in enumerate(records):
        record.segment = {"lang": "ko" if index % 2 else "en"}
    dataset = evaluate(records, n_boot=40, by=("lang",))
    metrics_by_slug = {
        segment_charts.segment_slug("lang", segment.value): segment.metrics
        for segment in dataset.segments
    }
    view = SegmentView(
        bars=segment_charts.bars_from_ece(
            tuple(
                ("lang", segment.value, segment.metrics.ece, segment.metrics.n)
                for segment in dataset.segments
            )
        ),
        x_axis="lang",
    )
    model = template.ReportModel(
        verdict=Verdict(
            status="too_low",
            headline="Your threshold is too low.",
            detail="detail",
            stats=(VerdictStat("current", "0.60"),),
        ),
        thresholds=(_threshold_result(),),
        impact=_impact(),
        segments=view,
        data_quality=template.data_quality_from(dataset),  # type: ignore[arg-type]
        generated_at="2026-09-20 12:00:00Z",
    )
    blocks = template.build_blocks(dataset)
    assert len(blocks) == 2, "the fixture needs two questions to exercise tabs"
    return template.render_document(model, blocks, segment_metrics=metrics_by_slug)


def test_every_hook_the_javascript_reads_exists_in_the_document() -> None:
    document = _document()
    missing = sorted(hook for hook in REQUIRED_HOOKS if hook not in document)
    assert missing == [], f"the report script reads hooks the template never emits: {missing}"


def test_the_hook_inventory_is_fully_accounted_for() -> None:
    """A new hook in the script must come with a template change, or this test fails."""
    source = Path(report_package.__file__).parent.joinpath("assets.py").read_text(encoding="utf-8")
    declared = set(re.findall(r"data-jeval-[a-z-]+", source))
    assert declared == REQUIRED_HOOKS | JS_INTERNAL_HOOKS, (
        "assets.py changed its hook inventory: "
        f"unexpected {sorted(declared - REQUIRED_HOOKS - JS_INTERNAL_HOOKS)}, "
        f"unused {sorted(REQUIRED_HOOKS | JS_INTERNAL_HOOKS - declared)}"
    )


def test_slider_exposes_every_figure_key_the_script_recomputes() -> None:
    document = _document()
    keys = set(re.findall(r'data-jeval-value="([a-z_]+)"', document))
    assert keys >= FIGURE_KEYS, f"missing figure keys: {sorted(FIGURE_KEYS - keys)}"
    assert 'data-jeval-action="auto_refund"' in document
    assert "data-jeval-threshold" in document


def test_tabs_and_panels_use_the_same_question_keys() -> None:
    document = _document()
    tabs = set(re.findall(r'data-jeval-tab="([^"]+)"', document))
    panels = set(re.findall(r'data-jeval-question="([^"]+)"', document))
    assert tabs == panels == {"department", "intent"}


def test_segment_bars_point_at_a_chart_that_exists() -> None:
    document = _document()
    bars = set(re.findall(r'data-jeval-segment="([^"]+)"', document))
    charts = set(re.findall(r'data-jeval-segment-chart="([^"]+)"', document))
    assert bars, "no clickable segment bars in the fixture document"
    assert bars <= charts, f"bars without a chart to reveal: {sorted(bars - charts)}"


def test_copy_button_is_wired_to_the_embedded_payload() -> None:
    document = _document()
    assert "data-jeval-copy-summary" in document
    assert 'data-copy-target="jeval-data"' in document
    assert 'id="jeval-data"' in document


def test_inline_assets_are_minified() -> None:
    document = _document()
    style = re.search(r"<style>(.*?)</style>", document, re.DOTALL)
    script = re.search(r"<script>(.*?)</script>", document, re.DOTALL)
    assert style is not None and script is not None
    assert len(style.group(1)) < len(REPORT_CSS), "the stylesheet shipped unminified"
    assert "\n    " not in script.group(1), "the script shipped unminified"
    assert "/*" not in style.group(1), "comments survived minification"


def test_the_summary_is_the_single_formatter_from_verdict() -> None:
    """The copy button's text, the payload and --format md must come from one place."""
    from jeval.report.verdict import markdown_summary as verdict_summary

    document = _document()
    model = template.ReportModel(
        verdict=Verdict(status="clear", headline="Headline.", detail="Detail."),
        impact=_impact(),
    )
    expected = verdict_summary(model)
    assert template.markdown_summary(model) == expected
    assert 'summary_markdown":' in document.replace(" ", "")
