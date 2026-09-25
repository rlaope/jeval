"""The verdict: the one-line answer, plus the summary that gets pasted into Slack.

This module turns calibration metrics and an optional cost result into the conclusion a reader
came for -- are the confidences trustworthy, and is the threshold in the right place -- and
into the markdown the report's copy button hands to Slack or an issue.

The threshold input is duck-typed on purpose. It arrives as ``jeval.costs.ThresholdResult``, as
an ``ImpactTable``, or as ``None``, and this module reads only the few numbers it needs. That
keeps the verdict free of a hard dependency on whichever cost module produced those numbers.

Branch order, which is the whole specification:

1. fewer than ``min_labels`` gold-labeled decisions -> :data:`INSUFFICIENT_DATA`,
2. ECE and MCE both inside ``tolerance`` -> :data:`CLEAR`, confidences are trustworthy,
3. a recommended threshold more than one sweep step above the one in use -> :data:`TOO_LOW`,
4. more than one sweep step below it -> :data:`TOO_HIGH`,
5. otherwise the thresholds agree -> :data:`CLEAR`, the threshold is already at the minimum.

Rule 5 needs a threshold to say anything about, so a miscalibrated report with no threshold at
all gets its own honest headline instead of a claim about a cost curve it never saw.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise

from jeval.calibration import CalibrationMetrics
from jeval.report.model import (
    CLEAR,
    INSUFFICIENT_DATA,
    TOO_HIGH,
    TOO_LOW,
    ImpactRow,
    ImpactTable,
    Verdict,
    VerdictStat,
)
from jeval.report.svg import fmt, pct

__all__ = [
    "HEADLINE_CLEAR",
    "HEADLINE_INSUFFICIENT_DATA",
    "HEADLINE_NEAR_MINIMUM",
    "HEADLINE_NO_THRESHOLD",
    "HEADLINE_TOO_HIGH",
    "HEADLINE_TOO_LOW",
    "build_verdict",
    "markdown_summary",
]

DEFAULT_MIN_LABELS = 100
DEFAULT_TOLERANCE = 0.02

HEADLINE_INSUFFICIENT_DATA = "Not enough labeled data to decide."
HEADLINE_CLEAR = "Confidence is trustworthy at the volumes you have."
HEADLINE_TOO_LOW = "Your threshold is too low."
HEADLINE_TOO_HIGH = "Your threshold is too high."
HEADLINE_NEAR_MINIMUM = "Your current threshold is already near the cost minimum."
HEADLINE_NO_THRESHOLD = "Calibration is off, but there is no threshold to move on this data."

# Guards the comparison against float noise when a curve supplied no usable sweep step.
_EPSILON = 1e-12


def build_verdict(
    metrics: CalibrationMetrics,
    *,
    threshold: object | None = None,
    current_threshold: float | None = None,
    recommended_threshold: float | None = None,
    min_labels: int = DEFAULT_MIN_LABELS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Verdict:
    """Decide the status and write the sentence that supports it.

    ``threshold`` is deliberately typed ``object | None``: it is read with ``getattr`` so a
    ``ThresholdResult``, an ``ImpactTable``, or anything else shaped like them all work, and so
    a missing attribute degrades to "no threshold evidence" instead of an ``AttributeError``.

    When only a recommendation is available the current threshold defaults to it -- one
    threshold and nothing to compare against is agreement, not evidence of a problem.
    """
    recommended = _resolve_recommended(threshold, recommended_threshold)
    current = _resolve_current(threshold, current_threshold, recommended)
    accuracy, accuracy_sub = _measured_accuracy(threshold, metrics, recommended)
    stats = _stats(metrics, current, recommended, accuracy, accuracy_sub, min_labels)

    if metrics.n < min_labels:
        return Verdict(
            status=INSUFFICIENT_DATA,
            headline=HEADLINE_INSUFFICIENT_DATA,
            detail=_insufficient_detail(metrics.n, min_labels),
            stats=stats,
        )
    if _within(metrics.ece, tolerance) and _within(metrics.mce, tolerance):
        return Verdict(
            status=CLEAR,
            headline=HEADLINE_CLEAR,
            detail=_calibrated_detail(metrics, tolerance),
            stats=stats,
        )
    if current is not None and recommended is not None:
        step = _sweep_step(threshold)
        gap = recommended - current
        if gap > step + _EPSILON:
            return Verdict(
                status=TOO_LOW,
                headline=HEADLINE_TOO_LOW,
                detail=_move_detail(metrics, current, recommended),
                stats=stats,
            )
        if gap < -(step + _EPSILON):
            return Verdict(
                status=TOO_HIGH,
                headline=HEADLINE_TOO_HIGH,
                detail=_move_detail(metrics, current, recommended),
                stats=stats,
            )
        return Verdict(
            status=CLEAR,
            headline=HEADLINE_NEAR_MINIMUM,
            detail=_near_minimum_detail(metrics, current, recommended),
            stats=stats,
        )
    return Verdict(
        status=CLEAR,
        headline=HEADLINE_NO_THRESHOLD,
        detail=_no_threshold_detail(metrics, tolerance),
        stats=stats,
    )


def markdown_summary(model: object) -> str:
    """The paste-into-Slack text: verdict, stat line, impact table.

    Accepts a :class:`~jeval.report.model.Verdict` or anything carrying one as ``.verdict``
    (a ``ReportModel``), plus an optional ``.impact`` table. Plain markdown only -- no HTML, no
    ANSI -- and exactly one trailing newline, because the report's copy button copies this
    string verbatim and it has to look the same everywhere it lands.
    """
    verdict = model if isinstance(model, Verdict) else _attribute(model, "verdict")
    if not isinstance(verdict, Verdict):
        raise TypeError("markdown_summary expects a Verdict or an object with a .verdict field")

    lines = [f"> **{verdict.headline}** {verdict.detail}".rstrip()]
    if verdict.stats:
        stats = " | ".join(f"{stat.label} {stat.value}" for stat in verdict.stats)
        lines.extend(["", stats])

    rows = _impact_rows(model)
    if rows:
        lines.extend(
            [
                "",
                "| Change | Now | Recommended | Delta |",
                "| --- | --- | --- | --- |",
            ]
        )
        for row in rows:
            cells = (row.label, row.current, row.recommended, row.change)
            lines.append("| " + " | ".join(_cell(cell) for cell in cells) + " |")
    return "\n".join(lines).rstrip("\n") + "\n"


def _attribute(source: object | None, name: str) -> object:
    """Read one field defensively; anything missing is ``None``, never an exception."""
    if source is None:
        return None
    value: object = getattr(source, name, None)
    return value


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _resolve_recommended(
    threshold: object | None, recommended_threshold: float | None
) -> float | None:
    explicit = _as_float(recommended_threshold)
    if explicit is not None:
        return explicit
    for name in ("recommended_threshold", "threshold"):
        found = _as_float(_attribute(threshold, name))
        if found is not None:
            return found
    return None


def _resolve_current(
    threshold: object | None, current_threshold: float | None, recommended: float | None
) -> float | None:
    explicit = _as_float(current_threshold)
    if explicit is not None:
        return explicit
    found = _as_float(_attribute(threshold, "current_threshold"))
    if found is not None:
        return found
    # Deliberately no fallback to `recommended`: substituting it made the report claim a threshold
    # was in use and compare the recommendation with itself. With nothing deployed, the verdict
    # says no threshold was supplied, which is the true statement.
    return None


def _sweep_step(threshold: object | None) -> float:
    """One sweep step, taken as the narrowest gap between curve thresholds.

    The smallest gap is the conservative reading of "more than one sweep step": a threshold
    that moved anywhere past a single step is worth flagging. No usable curve means no known
    step, so any difference at all counts.
    """
    curve = _attribute(threshold, "curve")
    if not isinstance(curve, Sequence) or isinstance(curve, (str, bytes)):
        return 0.0
    values: list[float] = []
    for point in curve:
        value = _as_float(_attribute(point, "threshold"))
        if value is not None:
            values.append(value)
    unique = sorted(set(values))
    gaps = [right - left for left, right in pairwise(unique) if right > left]
    return min(gaps) if gaps else 0.0


def _within(value: float, tolerance: float) -> bool:
    return math.isfinite(value) and value <= tolerance


def _observed_accuracy(metrics: CalibrationMetrics) -> float:
    total = sum(bin_.n for bin_ in metrics.bins)
    if total == 0:
        return float("nan")
    return sum(bin_.accuracy * bin_.n for bin_ in metrics.bins) / total


def _measured_accuracy(
    threshold: object | None,
    metrics: CalibrationMetrics,
    recommended: float | None = None,
) -> tuple[str, str]:
    """The middle headline figure: the accuracy the numbers actually support.

    The caption names the line the figure belongs to. "Measured accuracy 95.7%" on its own reads
    like the model's overall accuracy, when it is really the accuracy of what stays automated once
    the recommended threshold is in place — a difference of eleven points in the example report.
    """
    auto = _as_float(_attribute(threshold, "accuracy_auto"))
    if auto is not None:
        # Captions stay under 24 characters (a HUD-width rule the tests enforce), so the
        # threshold is named compactly rather than dropped.
        caption = (
            f"auto decisions at {fmt(recommended, 2)}"
            if recommended is not None
            else "of auto decisions"
        )
        return (pct(auto), caption)
    return (pct(_observed_accuracy(metrics)), f"of {metrics.n:,} labels")


def _stats(
    metrics: CalibrationMetrics,
    current: float | None,
    recommended: float | None,
    accuracy: str,
    accuracy_sub: str,
    min_labels: int,
) -> tuple[VerdictStat, ...]:
    stats: list[VerdictStat] = []
    if current is not None:
        stats.append(
            VerdictStat(label="Current threshold", value=fmt(current, 2), sub="auto at or above it")
        )
    if metrics.n > 0:
        stats.append(VerdictStat(label="Measured accuracy", value=accuracy, sub=accuracy_sub))
    if recommended is not None:
        stats.append(
            VerdictStat(
                label="Recommended threshold", value=fmt(recommended, 2), sub="cost minimum"
            )
        )
    else:
        stats.append(
            VerdictStat(
                label="Labeled decisions", value=f"{metrics.n:,}", sub=f"need {min_labels:,}"
            )
        )
    return tuple(stats[:3])


def _band_fragment(metrics: CalibrationMetrics) -> str:
    """Worst band, its measured accuracy and the label count: the numbers every detail carries."""
    worst = metrics.worst_bin
    if worst is None:
        return f"No confidence band holds samples (of {metrics.n:,} labels)"
    return (
        f"Band {worst.label} measures {pct(worst.accuracy)} accuracy on {worst.n:,} decisions"
        f" (of {metrics.n:,} labels)"
    )


def _insufficient_detail(n: int, min_labels: int) -> str:
    return (
        f"{n:,} labeled decisions exist; {min_labels:,} are needed before this report can "
        "decide anything, so more labels, not more analysis, are the blocker."
    )


def _calibrated_detail(metrics: CalibrationMetrics, tolerance: float) -> str:
    return (
        f"{_band_fragment(metrics)}; ECE {fmt(metrics.ece, 3)} and MCE {fmt(metrics.mce, 3)} "
        f"both sit inside the {fmt(tolerance, 2)} tolerance."
    )


def _move_detail(metrics: CalibrationMetrics, current: float, recommended: float) -> str:
    direction = "above" if recommended > current else "below"
    return (
        f"{_band_fragment(metrics)}; the threshold belongs at {fmt(recommended, 2)}, "
        f"{direction} the {fmt(current, 2)} in use."
    )


def _near_minimum_detail(metrics: CalibrationMetrics, current: float, recommended: float) -> str:
    return (
        f"{_band_fragment(metrics)}; the {fmt(current, 2)} threshold in use is within one "
        f"sweep step of the {fmt(recommended, 2)} cost minimum."
    )


def _no_threshold_detail(metrics: CalibrationMetrics, tolerance: float) -> str:
    return (
        f"{_band_fragment(metrics)}; ECE {fmt(metrics.ece, 3)} and MCE {fmt(metrics.mce, 3)} "
        f"sit outside the {fmt(tolerance, 2)} tolerance and no threshold was supplied to move."
    )


def _impact_rows(model: object) -> list[ImpactRow]:
    if isinstance(model, ImpactTable):
        return list(model.rows)
    impact = _attribute(model, "impact")
    if isinstance(impact, ImpactTable):
        return list(impact.rows)
    rows = _attribute(impact, "rows")
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        return [row for row in rows if isinstance(row, ImpactRow)]
    return []


def _cell(value: str) -> str:
    """One markdown table cell: no stray pipes, no newlines, no collapsed empty cells."""
    return " ".join(value.split()).replace("|", "\\|")
