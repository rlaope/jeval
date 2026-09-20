"""Golden-file snapshots for the chart builders.

A rendering change that nobody intended is a regression with no error message: the report still
builds, the numbers still compute, and the picture says something different. These snapshots make
that visible.

Regenerate deliberately with ``JEJVAL_UPDATE_GOLDEN=1 uv run pytest tests/test_golden_charts.py``
and read the diff before committing it.
"""

from __future__ import annotations

import os
from pathlib import Path

from jeval.evaluate import evaluate
from jeval.report.charts import cost as cost_charts
from jeval.report.charts import drift as drift_charts
from jeval.report.charts import reliability as reliability_charts
from jeval.report.charts import segments as segment_charts
from jeval.report.model import (
    CostPoint,
    DriftSlice,
    DriftView,
    HeatmapCell,
    ModelChange,
    SegmentView,
    ThresholdResult,
)
from jeval.synth import SynthSpec, generate

GOLDEN_DIR = Path(__file__).parent / "golden"


def _update_requested() -> bool:
    return os.environ.get("JEJVAL_UPDATE_GOLDEN") == "1"


def _assert_golden(name: str, produced: str) -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    target = GOLDEN_DIR / name
    if _update_requested() or not target.exists():
        target.write_text(produced, encoding="utf-8")
        if _update_requested():
            return
    expected = target.read_text(encoding="utf-8")
    assert produced == expected, (
        f"{name} changed. If the change is intended, regenerate with "
        f"JEJVAL_UPDATE_GOLDEN=1 and review the diff in tests/golden/."
    )


def _reliability_fixture():
    records = generate(
        SynthSpec(n=1200, mode="inflated", inflation=1.25, question_key="department", seed=42)
    )
    return evaluate(records, n_bins=6, n_boot=120, seed=42).overall


def _cost_fixture() -> ThresholdResult:
    points = []
    for index in range(21):
        threshold = index / 20.0
        auto = max(0.0, 1.0 - threshold)
        accuracy = 0.62 + 0.33 * threshold
        accept = (1.0 - accuracy) * auto * 5000.0
        escalate = 300.0 * (1.0 - auto)
        points.append(
            CostPoint(
                threshold=threshold,
                expected_cost=accept + escalate,
                auto_rate=auto,
                accuracy_auto=accuracy,
                accuracy_escalated=0.7,
                accept_cost=accept * 100.0,
                escalate_cost=escalate * 100.0,
                n_auto=int(auto * 100),
                n_escalate=int((1.0 - auto) * 100),
                n_wrong_auto=int((1.0 - accuracy) * auto * 100),
                n_wrong_escalate=1,
            )
        )
    best = min(points, key=lambda point: point.expected_cost)
    return ThresholdResult(
        action="auto_refund",
        question="intent",
        when="refund_request",
        threshold=best.threshold,
        expected_cost_per_case=best.expected_cost,
        auto_rate=best.auto_rate,
        accuracy_auto=best.accuracy_auto,
        ci_low=0.55,
        ci_high=0.72,
        curve=tuple(points),
        flat_region=(0.55, 0.8),
        n_records=100,
        models=("jev-1.13.0",),
        cost_false_accept=5000.0,
        cost_escalate=300.0,
        cost_false_reject=0.0,
    )


def test_reliability_chart_matches_golden() -> None:
    metrics = _reliability_fixture()
    produced = reliability_charts.render_reliability(
        metrics, threshold=0.8, title="Reliability · department"
    )
    _assert_golden("reliability-department.svg", produced)


def test_cost_chart_matches_golden() -> None:
    produced = cost_charts.render_cost_curve(_cost_fixture(), current_threshold=0.35)
    _assert_golden("cost-auto-refund.svg", produced)


def test_segments_chart_matches_golden() -> None:
    view = SegmentView(
        bars=segment_charts.bars_from_ece(
            (
                ("lang", "en", 0.14, 203),
                ("lang", "ko", 0.05, 412),
                ("tier", "pro", 0.09, 12),
            ),
            min_samples=30,
        ),
        heatmap=(HeatmapCell("ko", "free", 0.07, 180), HeatmapCell("en", "pro", 0.16, 140)),
        x_axis="lang",
        y_axis="tier",
    )
    _assert_golden("segments.svg", segment_charts.render_segments(view))


def test_drift_chart_matches_golden() -> None:
    view = DriftView(
        baseline_label="jev-1.13.0",
        current_label="jev-1.14.0",
        slices=(
            DriftSlice(
                "2026-09-01", "jev-1.13.0", "2026-09-01", "2026-09-07", 400, 0.061, 0.89, 0.58
            ),
            DriftSlice(
                "2026-09-08", "jev-1.14.0", "2026-09-08", "2026-09-14", 420, 0.142, 0.82, 0.41
            ),
        ),
        changes=(ModelChange("jev-1.14.0", "2026-09-08", "jev-1.13.0 -> jev-1.14.0"),),
    )
    _assert_golden("drift-ece-series.svg", drift_charts.render_ece_series(view))


def test_charts_are_deterministic() -> None:
    """Two builds of the same data must produce byte-identical output."""
    metrics = _reliability_fixture()
    first = reliability_charts.render_reliability(metrics, threshold=0.8)
    second = reliability_charts.render_reliability(metrics, threshold=0.8)
    assert first == second


def test_coordinates_are_rounded_to_two_decimals() -> None:
    """Long floats are a file-size tax with no visual benefit."""
    import re

    produced = cost_charts.render_cost_curve(_cost_fixture())
    offenders = [
        value
        for value in re.findall(
            r'(?:x|y|cx|cy|x1|y1|x2|y2|width|height)="(-?\d+\.\d{3,})"', produced
        )
    ]
    assert offenders == [], f"unrounded coordinates: {offenders[:5]}"
