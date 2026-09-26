"""Risk-coverage chart: does confidence rank the model's own errors?

Reliability says whether confidence is honest on average. This chart says whether it is useful for
a threshold: automate the most confident answers first and watch the error rate among them. A curve
that falls to the left means escalating the least confident answers removes errors; a curve that
sits on the flat line means it removes volume and nothing else.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from jeval.calibration import MIN_BIN_SIZE, DiscriminationMetrics
from jeval.report import svg as S

WIDTH = 660.0
HEIGHT = 420.0
LEFT = 66.0
RIGHT = 18.0
TOP = 48.0
BOTTOM = 86.0

# A rate over the first handful of automated answers is an anecdote: the first answer alone puts
# the curve at 0% or 100%. The curve starts where a calibration bin would.
MIN_POINT_COUNT = MIN_BIN_SIZE

COVERAGE_ROWS: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)


def _risk_ticks(peak: float) -> list[float]:
    """Round ticks from zero to just above the highest risk on the chart."""
    ceiling = max(peak, 0.01) * 1.08
    ticks = S.nice_ticks(0.0, ceiling, target=5)
    if ticks[-1] < ceiling:
        step = ticks[1] - ticks[0] if len(ticks) > 1 else ceiling
        ticks.append(round(ticks[-1] + step, 10))
    if ticks[0] > 0.0:
        ticks.insert(0, 0.0)
    return ticks


def _percent_format(ticks: list[float]) -> Callable[[float], str]:
    base = S.tick_format([tick * 100.0 for tick in ticks])

    def render(value: float) -> str:
        return f"{base(value * 100.0)}%"

    return render


def render_risk_coverage(
    metrics: DiscriminationMetrics,
    *,
    threshold: float | None = None,
    action: str = "",
    title: str = "Risk and coverage",
    width: float = WIDTH,
    height: float = HEIGHT,
) -> str:
    """The error rate among automated answers at every coverage, most confident first."""
    if metrics.n == 0 or not metrics.coverage:
        return _empty(width, title)
    plot_left, plot_right = LEFT, width - RIGHT
    plot_top, plot_bottom = TOP, height - BOTTOM
    start = next(
        (
            index
            for index, covered in enumerate(metrics.coverage)
            if covered * metrics.n >= MIN_POINT_COUNT - 1e-9
        ),
        len(metrics.coverage) - 1,
    )
    shown = list(zip(metrics.coverage[start:], metrics.risk[start:], strict=True))
    base_rate = metrics.full_coverage_risk
    peak = max([risk for _, risk in shown] + [base_rate])
    y_ticks = _risk_ticks(peak)
    x = S.lin_scale((0.0, 1.0), (plot_left, plot_right))
    y = S.lin_scale((0.0, y_ticks[-1]), (plot_bottom, plot_top))
    x_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    desc = (
        f"Risk-coverage curve over {metrics.n} labeled decisions, most confident answers "
        f"automated first. At full coverage {S.pct(base_rate, 1)} of answers are wrong. "
        f"AUROC {S.fmt(metrics.auroc)}, AURC {S.fmt(metrics.aurc, 3)}. A curve below the flat "
        "dashed line means escalating the least confident answers removes errors; a curve on "
        "it means confidence ranks nothing."
    )
    rec: tuple[float, float] | None = None
    if threshold is not None and math.isfinite(threshold):
        covered, risk = metrics.at_threshold(threshold)
        if covered > 0.0 and math.isfinite(risk):
            rec = (covered, risk)
            desc += (
                f" At the recommended line {S.fmt(threshold)}"
                + (f" ({action})" if action else "")
                + f", {S.pct(covered, 0)} of this question's answers are stated at or above "
                f"it and {S.pct(risk, 1)} of those are wrong. The action itself runs only on the "
                "class it names, so its own automation rate can be lower."
            )
    parts: list[str] = [S.svg_open(width, height, title=title, desc=desc, cls="chart")]
    parts.append(S.grid_x(x, x_ticks, y0=plot_top, y1=plot_bottom))
    parts.append(S.grid_y(y, y_ticks, x0=plot_left, x1=plot_right))

    # No ranking: automating a random subset leaves the error rate where it is at full coverage.
    parts.append(
        S.line(plot_left, y(base_rate), plot_right, y(base_rate), stroke=S.DIAGONAL, dash="5 4")
    )
    # Perfect ranking: every right answer before every wrong one.
    accuracy = 1.0 - base_rate
    perfect: list[tuple[float, float]] = [(x(0.0), y(0.0))]
    if accuracy > 0.0:
        perfect.append((x(accuracy), y(0.0)))
        for step in range(1, 41):
            covered = accuracy + (1.0 - accuracy) * step / 40.0
            perfect.append((x(covered), y(max(0.0, 1.0 - accuracy / covered))))
    else:
        perfect.append((x(1.0), y(1.0)))
    parts.append(S.polyline(perfect, stroke=S.SOFT, width=1.2))
    if 0.12 < accuracy < 0.97:
        parts.append(
            S.text(
                x(accuracy) - 6,
                y(0.0) - 7,
                "perfect ranking",
                anchor="end",
                size=11,
                fill=S.MUTED,
                halo=True,
            )
        )

    # The model, thinned to one point per pixel and a half: a thousand-point path draws the same
    # line as a four-hundred-point one and costs the file twice as much.
    curve: list[tuple[float, float]] = []
    for covered, risk in shown:
        px, py = x(covered), y(risk)
        if curve and px - curve[-1][0] < 1.5 and (covered, risk) != shown[-1]:
            continue
        curve.append((px, py))
    if len(curve) > 1:
        parts.append(S.polyline(curve, stroke=S.INK, width=1.8))
    for px, py in curve[:1] + curve[-1:]:
        parts.append(S.dot(px, py, 2.6, fill=S.INK))

    if rec is not None:
        covered, risk = rec
        rx = x(covered)
        parts.append(S.line(rx, plot_top, rx, plot_bottom, stroke=S.ACCENT, cls="accent-stroke"))
        parts.append(
            S.dot(
                rx,
                y(risk),
                4.6,
                fill=S.ACCENT,
                cls="accent-dot",
                tooltip=(
                    f"recommended {S.fmt(threshold or 0.0)} · {S.pct(covered, 0)} of answers "
                    f"at or above it · {S.pct(risk, 1)} of them wrong"
                ),
                extra=' style="stroke: var(--panel); stroke-width: 2px"',
            )
        )
        # Two short lines rather than one long one: a single line runs off the plot whenever the
        # recommended coverage sits mid-chart, which is where it usually sits.
        label = f"recommended {S.fmt(threshold or 0.0)}" + (f" ({action})" if action else "")
        detail = f"{S.pct(covered, 0)} of answers at or above, {S.pct(risk, 1)} wrong"
        label_width = max(S.text_width(label, 12.0), S.text_width(detail, 11.5))
        anchor = "start" if rx + 6 + label_width <= plot_right else "end"
        if anchor == "end" and rx - 6 - label_width < plot_left:
            anchor = "start"
        label_x = rx + (6.0 if anchor == "start" else -6.0)
        parts.append(
            S.text(
                label_x,
                plot_top - 20,
                label,
                anchor=anchor,
                size=12,
                weight=600,
                fill=S.ACCENT,
                cls="accent-ink",
                halo=True,
            )
        )
        parts.append(
            S.text(label_x, plot_top - 6, detail, anchor=anchor, size=11.5, fill=S.MUTED, halo=True)
        )

    parts.append(
        S.axis_x(
            x,
            y=plot_bottom,
            tick_values=x_ticks,
            format_=lambda value: f"{value * 100:.0f}%",
            title="coverage: share of decisions automated",
        )
    )
    parts.append(
        S.axis_y(
            y,
            x=plot_left,
            tick_values=y_ticks,
            format_=_percent_format(y_ticks),
            title="risk: error rate among them",
        )
    )
    parts.append(
        S.legend(
            [
                ("this model", S.INK, "line"),
                ("no ranking", S.DIAGONAL, "dash"),
                ("perfect ranking", S.SOFT, "line"),
            ],
            x=plot_left,
            y=height - 8,
        )
    )
    parts.append(S.svg_close())
    return "".join(parts)


def _empty(width: float, title: str) -> str:
    height = 120.0
    return (
        S.svg_open(width, height, title=title, desc="No labeled decisions to rank.", cls="chart")
        + S.text(
            S.MARGIN,
            height / 2 + 4,
            "No labeled decisions: a ranking needs right and wrong answers to compare.",
            size=12,
            fill=S.MUTED,
        )
        + S.svg_close()
    )


def key_figures(
    metrics: DiscriminationMetrics, *, threshold: float | None = None, action: str = ""
) -> str:
    """AUROC with its interval, AURC against its two bounds, and the error rate to beat."""

    def interval(low: float, high: float, digits: int) -> str:
        if low != low or high != high:
            return "no interval: too few right or wrong answers"
        return f"95% CI {S.fmt(low, digits)}–{S.fmt(high, digits)}"

    rows: list[tuple[str, str, str]] = [
        (
            "AUROC",
            S.fmt(metrics.auroc),
            interval(metrics.auroc_ci_low, metrics.auroc_ci_high, 2) + " · 0.50 is a coin flip",
        ),
        (
            "AURC",
            S.fmt(metrics.aurc, 3),
            interval(metrics.aurc_ci_low, metrics.aurc_ci_high, 3)
            + f" · no ranking {S.fmt(metrics.full_coverage_risk, 3)}, "
            f"perfect {S.fmt(metrics.perfect_aurc, 3)}",
        ),
        (
            "Error rate, all automated",
            S.pct(metrics.full_coverage_risk, 1),
            f"{metrics.n_wrong:,} of {metrics.n:,} labeled answers wrong",
        ),
    ]
    if threshold is not None and math.isfinite(threshold):
        covered, risk = metrics.at_threshold(threshold)
        if covered > 0.0 and math.isfinite(risk):
            rows.append(
                (
                    "At the recommended line",
                    f"{S.pct(risk, 1)} wrong",
                    f"{S.pct(covered, 0)} of answers stated at or above {S.fmt(threshold)}"
                    + (f" ({action})" if action else ""),
                )
            )
    items = "".join(
        f"<div><dt>{S.escape(label)}</dt><dd>{S.escape(value)}"
        f"<small>{S.escape(sub)}</small></dd></div>"
        for label, value, sub in rows
    )
    return f'<dl class="keyfigs">{items}</dl>'


def coverage_rows(metrics: DiscriminationMetrics) -> list[list[str]]:
    """The curve read at round coverages, for the table under the chart."""
    rows: list[list[str]] = []
    seen: set[float] = set()
    for share in COVERAGE_ROWS:
        level, covered, risk = metrics.at_coverage(share)
        if level != level or covered in seen:
            continue
        seen.add(covered)
        rows.append(
            [
                S.pct(share, 0),
                S.fmt(level),
                f"{round(covered * metrics.n):,}",
                S.pct(covered, 1),
                S.pct(risk, 1),
            ]
        )
    return rows


COVERAGE_HEADERS: tuple[str, ...] = (
    "Target coverage",
    "Line at or above",
    "Automated",
    "Coverage",
    "Error rate",
)


def render_discrimination_section(
    metrics: DiscriminationMetrics,
    *,
    threshold: float | None = None,
    action: str = "",
    title: str = "Risk and coverage",
) -> str:
    """Chart beside its key figures, then the curve read at round coverages."""
    chart = render_risk_coverage(metrics, threshold=threshold, action=action, title=title)
    caption = (
        "<figcaption>Answers are automated most confident first. The solid line is the error rate "
        "among the automated answers; the dashed line is where it would stay if confidence ranked "
        "nothing, and the grey curve is the best any ranking could do. The curve starts once "
        f"{MIN_POINT_COUNT} decisions are automated: before that, a rate is an anecdote. The "
        "recommended line, when there is one, is marked where it cuts this curve: every answer to "
        "this question stated at or above it, whichever class it names. The action runs only on "
        "its own class, so the cost section's automation rate can be lower.</figcaption>"
    )
    if metrics.n == 0:
        return f'<div class="plate"><figure>{chart}{caption}</figure></div>'
    table = S.details_table(
        COVERAGE_HEADERS,
        coverage_rows(metrics),
        summary="Error rate at round coverages",
    )
    return (
        f'<div class="plate"><figure><div class="chart-row">{chart}'
        f"{key_figures(metrics, threshold=threshold, action=action)}</div>"
        f"{caption}</figure></div>{table}"
    )
