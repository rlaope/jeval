"""Report wiring for discrimination and classwise calibration.

The statistics are tested in ``test_discrimination.py`` and ``test_classwise.py``; this file checks
that the report shows them where a reader looks for them, with the accessible names every chart
owes its reader, and without a section it cannot fill.
"""

from __future__ import annotations

import re

from jeval.evaluate import evaluate
from jeval.report import template
from jeval.report.model import ReportModel
from jeval.report.verdict import build_verdict
from jeval.synth import SynthSpec, generate


def _records():
    return [
        *generate(
            SynthSpec(
                n=900,
                mode="class_inflated",
                inflation=1.3,
                inflated_class="refund_request",
                classes=("refund_request", "check_balance", "other"),
                question_key="intent",
                seed=31,
            )
        ),
        *generate(
            SynthSpec(n=500, mode="calibrated", question_key="is_urgent", question_type="noul")
        ),
        *generate(
            SynthSpec(n=200, mode="inflated", question_key="satisfaction", question_type="score")
        ),
    ]


def _render(records, **build) -> str:
    dataset = evaluate(records, n_boot=60)
    model = ReportModel(
        verdict=build_verdict(dataset.overall),
        data_quality=template.data_quality_from(dataset),  # type: ignore[arg-type]
        generated_at="2026-09-20 12:00:00Z",
    )
    return template.render_document(model, template.build_blocks(dataset, **build))


def _section(html: str, section_id: str) -> str:
    return html.split(f'<section id="{section_id}"', 1)[1].split("</section>", 1)[0]


def test_discrimination_section_follows_reliability_with_accessible_charts() -> None:
    html = _render(_records())
    assert html.index('id="reliability"') < html.index('id="discrimination"')
    assert html.index('id="discrimination"') < html.index('id="cost"')
    assert '<span class="idx">02</span>Discrimination' in html
    section = _section(html, "discrimination")
    for key in ("intent", "is_urgent"):
        match = re.search(
            rf'<svg[^>]*aria-label="Risk and coverage · {key}"[^>]*>'
            rf"<title>Risk and coverage · {key}</title><desc>([^<]+)</desc>",
            section,
        )
        assert match, key
        assert "Risk-coverage curve over" in match.group(1)
    assert "coverage: share of decisions automated" in section
    assert "risk: error rate among them" in section
    assert "AUROC" in section and "AURC" in section
    # Each question's panel follows the same tabs as the reliability charts.
    assert 'data-jeval-question="intent"' in section
    # A score question has no ranking to draw, and the section says so instead of drawing one.
    assert 'data-jeval-question="satisfaction"' in section
    assert "measured as error, not as right or wrong" in section


def test_recommended_line_is_marked_with_the_accent_only() -> None:
    html = _render(_records(), thresholds={"intent": 0.7}, actions={"intent": "auto_refund"})
    panel = _section(html, "discrimination").split('data-jeval-question="is_urgent"', 1)[0]
    assert "recommended 0.70 (auto_refund)" in panel
    assert 'class="accent-dot"' in panel and 'class="accent-stroke"' in panel
    assert "alert-" not in panel
    assert "At the recommended line" in panel


def test_choice_question_carries_a_per_class_table() -> None:
    html = _render(
        _records(),
        cost_classes={"intent": {"refund_request": ["auto_refund"]}},
    )
    reliability = _section(html, "reliability")
    intent = reliability.split('data-jeval-question="intent"', 1)[1].split(
        'data-jeval-question="is_urgent"', 1
    )[0]
    assert "Calibration per class" in intent
    for name in ("refund_request", "check_balance", "other"):
        assert f'<td class="ident">{name}</td>' in intent
    assert "auto_refund fires on it" in intent
    assert "Worst class: refund_request" in intent
    assert "It is the class auto_refund fires on." in intent
    # A yes/no question has no class map worth splitting: no table there.
    urgent = reliability.split('data-jeval-question="is_urgent"', 1)[1]
    assert "Calibration per class" not in urgent


def test_truncated_maps_are_counted_in_the_report() -> None:
    records = _records()
    for record in records[:40]:
        record.probabilities = {record.prediction: record.confidence}
    html = _render(records)
    assert "40 labeled decision(s) set aside" in _section(html, "reliability")


def test_no_discrimination_section_without_a_rankable_question() -> None:
    records = generate(
        SynthSpec(n=200, mode="inflated", question_key="satisfaction", question_type="score")
    )
    html = _render(records)
    assert 'id="discrimination"' not in html
    assert "Discrimination</a>" not in html
