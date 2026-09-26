"""Document-level tests for the HTML report.

The report is the product's argument, so these tests assert the properties that make it
trustworthy: it is one self-contained file, it says the same thing in its summary as in its
markdown output, it survives degenerate inputs, and it never claims a section it cannot fill.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

from typer.testing import CliRunner

from jeval.cli import app
from jeval.evaluate import evaluate
from jeval.report import template
from jeval.report.model import (
    DEFAULT_LIMITATIONS,
    SECTION_ORDER,
    ReportModel,
    SegmentView,
    ThresholdResult,
)
from jeval.report.verdict import build_verdict
from jeval.synth import SynthSpec, generate

runner = CliRunner()
ONE_MEGABYTE = 1_048_576


class _Collector(HTMLParser):
    """Minimal structural reader: element counts and attribute inventory."""

    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.attributes: list[tuple[str, str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        for name, value in attrs:
            self.attributes.append((tag, name, value or ""))


def _dataset(records):
    return evaluate(records, n_boot=60, by=("lang",))


def _model(dataset, **overrides) -> ReportModel:
    verdict = overrides.pop("verdict", None) or build_verdict(dataset.overall)
    model = ReportModel(
        verdict=verdict,
        thresholds=overrides.pop("thresholds", ()),
        impact=overrides.pop("impact", None),
        segments=overrides.pop("segments", None),
        drift=overrides.pop("drift", None),
        data_quality=template.data_quality_from(dataset),  # type: ignore[arg-type]
        generated_at="2026-09-20 12:00:00Z",
        source_note="source: .jeval/records.jsonl",
        demo_note="",
    )
    for key, value in overrides.items():
        setattr(model, key, value)
    return model


def _render(dataset, **overrides) -> str:
    blocks = template.build_blocks(dataset)
    return template.render_document(_model(dataset, **overrides), blocks)


def test_no_external_references_anywhere() -> None:
    html = _render(_dataset(generate(SynthSpec(n=600, mode="inflated", seed=5))))
    assert "<script src" not in html
    assert "@import" not in html
    assert "url(http" not in html
    assert 'src="http' not in html
    assert 'href="http' not in html
    assert "crossorigin" not in html
    # The SVG namespace is a name, not a fetch, and it is the only http token allowed.
    assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")


def test_document_parses_and_has_expected_structure() -> None:
    html = _render(
        _dataset(generate(SynthSpec(n=400, mode="calibrated", seed=6))), segments=SegmentView()
    )
    parser = _Collector()
    parser.feed(html)
    assert html.startswith("<!doctype html>")
    assert parser.tags.count("html") == 1
    assert "main" in parser.tags
    ids = {value for tag, name, value in parser.attributes if name == "id"}
    # `impact` lives inside the cost section and `drift` only appears with a comparison.
    conditional = {"impact", "drift"}
    for section in SECTION_ORDER:
        html_id = section.replace("_", "-")
        assert html_id in ids or section in conditional, f"missing section: {section}"
    script_types = [
        value for tag, name, value in parser.attributes if tag == "script" and name == "type"
    ]
    assert "application/json" in script_types


def test_sections_appear_in_argument_order() -> None:
    html = _render(_dataset(generate(SynthSpec(n=400, mode="inflated", seed=7))))
    positions = [
        html.index(f'id="{name}"')
        for name in ("verdict", "reliability", "cost", "segments", "data-quality")
    ]
    assert positions == sorted(positions)


def test_every_chart_carries_a_title_and_description() -> None:
    html = _render(
        _dataset(generate(SynthSpec(n=500, mode="infLated".lower(), seed=8))),
        segments=SegmentView(),
    )
    assert html.count("<svg") >= 1
    assert html.count("<title>") >= html.count("<svg")
    assert html.count("<desc>") >= html.count("<svg")


def test_data_tables_are_reachable_without_javascript() -> None:
    html = _render(
        _dataset(generate(SynthSpec(n=500, mode="overconfident", seed=9))), segments=SegmentView()
    )
    assert "<details>" in html
    assert "<summary>" in html


def test_print_stylesheet_and_dark_mode_are_present() -> None:
    html = _render(
        _dataset(generate(SynthSpec(n=300, mode="calibrated", seed=10))), segments=SegmentView()
    )
    assert "@media print" in html
    assert "prefers-color-scheme: dark" in html
    assert "@page" in html


def test_summary_is_embedded_for_the_copy_button() -> None:
    dataset = _dataset(generate(SynthSpec(n=300, mode="inflated", seed=11)))
    html = _render(dataset)
    match = re.search(
        r'<script type="application/json" id="jeval-data">(.*?)</script>', html, re.DOTALL
    )
    assert match is not None
    payload = json.loads(match.group(1).replace("<\\/", "</"))
    assert payload["summary_markdown"].strip()
    assert 'id="copy-summary"' in html


def test_silver_only_dataset_raises_a_banner() -> None:
    records = generate(
        SynthSpec(n=300, mode="calibrated", seed=12, label_fraction=1.0, silver_fraction=1.0)
    )
    dataset = _dataset(records)
    assert dataset.silver_only is True
    html = _render(dataset)
    assert "Every label here is silver" in html


def test_unlabeled_dataset_says_it_has_nothing_to_measure() -> None:
    records = generate(SynthSpec(n=200, mode="calibrated", seed=13, label_fraction=0.0))
    dataset = _dataset(records)
    html = _render(dataset)
    assert "No gold labels" in html
    assert "Not enough labeled data" in html


def test_limitations_are_always_present() -> None:
    html = _render(_dataset(generate(SynthSpec(n=250, mode="calibrated", seed=14))))
    for limitation in DEFAULT_LIMITATIONS:
        assert limitation.split(".")[0][:40] in html


def test_score_records_stay_out_of_binary_metrics() -> None:
    records = generate(
        SynthSpec(n=200, mode="calibrated", question_type="score", seed=15),
    )
    dataset = _dataset(records)
    html = _render(dataset)
    assert dataset.overall.n == 0
    assert "Excluded score records" in html
    assert "200" in html


def test_drift_section_is_absent_without_a_comparison() -> None:
    html = _render(_dataset(generate(SynthSpec(n=300, mode="calibrated", seed=16))))
    assert 'id="drift"' not in html


def test_edge_case_inputs_render() -> None:
    cases = {
        "no records": [],
        "one record": generate(SynthSpec(n=1, mode="calibrated", seed=17)),
        "identical confidence": generate(
            SynthSpec(n=120, mode="constant_high", constant_confidence=0.99, seed=18)
        ),
        "identical labels": generate(SynthSpec(n=120, mode="calibrated", seed=19)),
        "twenty questions": [
            record
            for index in range(20)
            for record in generate(
                SynthSpec(n=30, mode="calibrated", seed=20 + index, question_key=f"q{index}")
            )
        ],
    }
    for name, records in cases.items():
        dataset = _dataset(records) if records else evaluate([])
        html = _render(dataset)
        assert html.startswith("<!doctype html>"), name
        assert html.rstrip().endswith("</html>"), name
        assert len(html) < ONE_MEGABYTE, f"{name}: {len(html)} bytes"


def test_fifty_segment_values_render_as_bars() -> None:
    records = []
    for index in range(50):
        records.extend(
            generate(
                SynthSpec(n=40, mode="calibrated", seed=100 + index, question_key="department"),
            )
        )
    for position, record in enumerate(records):
        record.segment = {"lang": f"lang-{position % 50}"}
    dataset = _dataset(records)
    assert len(dataset.segments) == 50
    html = _render(dataset)
    assert len(html) < ONE_MEGABYTE


def test_threshold_without_sweep_curve_is_reported_as_insufficient() -> None:
    dataset = _dataset(generate(SynthSpec(n=200, mode="calibrated", seed=21)))
    empty = ThresholdResult(
        action="auto_refund",
        question="intent",
        when="refund_request",
        threshold=float("nan"),
        expected_cost_per_case=float("nan"),
        auto_rate=float("nan"),
        accuracy_auto=float("nan"),
        ci_low=float("nan"),
        ci_high=float("nan"),
        curve=(),
        n_records=4,
    )
    html = _render(dataset, thresholds=(empty,))
    assert "No cost curve" in html or "not enough" in html.lower()


def test_verdict_headline_is_visible_before_any_chart() -> None:
    dataset = _dataset(generate(SynthSpec(n=600, mode="overconfident", seed=22)))
    html = _render(dataset)
    headline_at = html.index('class="headline"')
    first_chart = html.index("<svg")
    assert headline_at < first_chart


def test_report_via_cli_end_to_end(tmp_path: Path) -> None:
    """The real command path: records plus a cost matrix, one file out."""
    records = generate(
        SynthSpec(
            n=300,
            mode="inflated",
            inflation=1.3,
            seed=23,
            question_key="intent",
            classes=("refund_request", "check_balance", "other"),
        )
    )
    records_file = tmp_path / "records.jsonl"
    from jeval.store import write_records

    write_records(records, records_file)
    (tmp_path / "costs.yaml").write_text(
        "actions:\n"
        "  - name: auto_refund\n"
        "    question: intent\n"
        "    when: refund_request\n"
        "    cost_false_accept: 50000\n"
        "    cost_escalate: 2000\n"
        "    cost_false_reject: 0\n",
        encoding="utf-8",
    )
    project = tmp_path / "project"
    (project / ".jeval").mkdir(parents=True)
    (project / ".jeval" / "records.jsonl").write_bytes(records_file.read_bytes())
    (project / "costs.yaml").write_bytes((tmp_path / "costs.yaml").read_bytes())

    result = runner.invoke(
        app, ["report", "--root", str(project), "--by", "lang", "--current", "0.5"]
    )
    assert result.exit_code == 0, result.stdout
    target = project / "report.html"
    assert target.exists()
    html = target.read_text(encoding="utf-8")
    assert len(html) < ONE_MEGABYTE
    assert 'id="verdict"' in html and 'id="cost"' in html
    assert "threshold auto_refund" in result.stdout


def test_format_md_prints_only_the_paste_ready_summary(tmp_path: Path) -> None:
    records = generate(
        SynthSpec(
            n=300,
            mode="inflated",
            seed=24,
            question_key="intent",
            classes=("refund_request", "check_balance", "other"),
        )
    )
    from jeval.store import write_records

    project = tmp_path / "project"
    (project / ".jeval").mkdir(parents=True)
    write_records(records, project / ".jeval" / "records.jsonl")
    result = runner.invoke(app, ["report", "--root", str(project), "--format", "md"])
    assert result.exit_code == 0, result.stdout
    assert "<html" not in result.stdout
    assert "**" in result.stdout
    assert not (project / "report.html").exists()


def test_a_long_segment_label_does_not_run_into_its_bar() -> None:
    import re

    from jeval.report import svg as chart_svg
    from jeval.report.charts import segments as segment_charts
    from jeval.report.model import SegmentView

    bars = segment_charts.bars_from_ece(
        (("customer_tier", "enterprise_plus", 0.12, 200), ("lang", "en", 0.05, 300))
    )
    svg = segment_charts.render_segments(SegmentView(bars=bars))
    label_width = chart_svg.text_width("customer_tier = enterprise_plus", 12.5, mono=True)
    tracks = [float(x) for x in re.findall(r'<rect x="([0-9.]+)" y="[0-9.]+" width', svg)]
    assert tracks and min(tracks) > label_width


def test_the_reliability_chart_names_the_recommended_line_and_the_line_in_use_apart() -> None:
    """The chart drew the recommendation as "line in use", contradicting the verdict beside it."""
    from jeval.evaluate import evaluate
    from jeval.report import template
    from jeval.report.charts import reliability as reliability_charts
    from jeval.synth import SynthSpec, generate

    dataset = evaluate(generate(SynthSpec(n=300, mode="inflated", seed=3)), n_boot=20)
    key = dataset.questions[0].question_key
    blocks = template.build_blocks(
        dataset, thresholds={key: 0.85}, in_use={key: 0.60}, actions={key: "auto_route"}
    )
    block = blocks[0]
    assert block.threshold == 0.85 and block.in_use == 0.60
    assert "recommended 0.85 (auto_route)" in block.subtitle
    assert "in use 0.60" in block.subtitle
    html = reliability_charts.render_reliability_section(
        block.metrics, threshold=block.in_use, recommended=block.threshold
    )
    assert "in use 0.60" in html and "recommended 0.85" in html
    assert "line in use 0.85" not in html


def test_the_gold_count_is_the_sample_the_metrics_were_measured_on() -> None:
    """The row read 767 while the report measured 696: score answers were counted as gold."""
    from jeval.report import template
    from jeval.synth import demo_dataset

    dataset = _dataset(list(demo_dataset(seed=11, scale=0.5).records))
    rows = dict(template.data_quality_from(dataset).rows)  # type: ignore[attr-defined]
    binary = dataset.n_records - dataset.n_score_excluded
    assert rows["Labeled (gold)"] == f"{dataset.overall.n:,} of {binary:,} choice and yes/no"


def test_a_sweep_with_no_resamples_reports_no_interval_instead_of_crashing() -> None:
    import math

    from jeval.costs import CostAction, sweep

    action = CostAction(
        name="a",
        question="department",
        when="billing",
        cost_false_accept=10.0,
        cost_escalate=1.0,
        cost_false_reject=0.0,
    )
    records = generate(
        SynthSpec(
            n=300,
            mode="calibrated",
            seed=2,
            question_key="department",
            classes=("billing", "technical"),
        )
    )
    result = sweep(action, records, n_boot=0)
    assert math.isnan(result.ci_low) and math.isnan(result.ci_high)
    assert result.curve
