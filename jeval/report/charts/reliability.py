"""Reliability diagram: stated confidence against observed accuracy.

The centrepiece of the report. Everything about it exists to stop a reader from trusting a
point that has three records behind it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from jeval.calibration import CalibrationMetrics
from jeval.report import svg as S

WIDTH = 660.0
HEIGHT = 470.0
LEFT = 62.0
RIGHT = 18.0
TOP = 34.0
BOTTOM = 136.0
DENSITY_HEIGHT = 34.0
MARGIN = S.MARGIN


def _radius(n: int, n_max: int) -> float:
    """Dot size scales with the sample behind it: 12 records must not look like 395."""
    if n_max <= 0:
        return 3.0
    ratio = float(n) / float(n_max)
    return 2.2 + 4.6 * math.sqrt(ratio)


def render_reliability(
    metrics: CalibrationMetrics,
    *,
    threshold: float | None = None,
    title: str = "Reliability",
    width: float = WIDTH,
    height: float = HEIGHT,
    subtitle: str = "",
) -> str:
    """Draw the reliability curve with Wilson intervals, density strip and threshold marker."""
    if metrics.n == 0:
        return _empty(width, title)

    plot_left, plot_right = LEFT, width - RIGHT
    plot_top, plot_bottom = TOP, height - BOTTOM
    x = S.lin_scale((0.0, 1.0), (plot_left, plot_right))
    y = S.lin_scale((0.0, 1.0), (plot_bottom, plot_top))

    desc = (
        f"Reliability diagram over {metrics.n} labeled decisions. "
        f"ECE {S.fmt(metrics.ece, 3)}, MCE {S.fmt(metrics.mce, 3)}. "
        f"Points below the diagonal mean the model claimed more confidence than its accuracy "
        f"earned. Vertical bars are Wilson 95% intervals."
    )
    parts: list[str] = [S.svg_open(width, height, title=title, desc=desc, cls="chart")]
    # The overconfident half is a triangle below the diagonal, not a band across the plot: the
    # shape is the meaning, and a full-width rectangle would dominate the curve it is explaining.
    parts.append(
        S.polygon(
            [
                (x(0.0), y(0.0)),
                (x(1.0), y(1.0)),
                (x(1.0), y(0.0)),
            ]
        )
    )
    parts.append(
        S.text(
            x(0.68),
            y(0.22),
            "overconfident",
            anchor="middle",
            size=10.5,
            fill=S.MUTED,
            halo=True,
        )
    )
    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    parts.append(S.grid_x(x, ticks, y0=plot_top, y1=plot_bottom))
    parts.append(S.grid_y(y, ticks, x0=plot_left, x1=plot_right))
    parts.append(S.line(x(0.0), y(0.0), x(1.0), y(1.0), stroke=S.DIAGONAL, width=1.2, dash="5 4"))

    n_max = max((cal_bin.n for cal_bin in metrics.bins), default=0)
    curve: list[tuple[float, float]] = []
    for cal_bin in metrics.bins:
        px = x(min(max(cal_bin.mean_confidence, 0.0), 1.0))
        py = y(min(max(cal_bin.accuracy, 0.0), 1.0))
        curve.append((px, py))
        parts.append(S.error_bar(px, cal_bin.ci_low, cal_bin.ci_high, y, stroke=S.SOFT))
    parts.append(S.polyline(curve, stroke=S.INK, width=1.8))
    for cal_bin, (px, py) in zip(metrics.bins, curve, strict=True):
        tooltip = (
            f"n={cal_bin.n} · stated {S.fmt(cal_bin.mean_confidence)} · "
            f"actual {S.fmt(cal_bin.accuracy)} "
            f"[{S.fmt(cal_bin.ci_low)}, {S.fmt(cal_bin.ci_high)}]"
        )
        parts.append(S.dot(px, py, _radius(cal_bin.n, n_max), fill=S.INK, tooltip=tooltip))

    if threshold is not None and 0.0 <= threshold <= 1.0:
        parts.append(
            S.marker_line(
                x(threshold),
                y0=plot_top,
                y1=plot_bottom,
                label=f"line in use {S.fmt(threshold)}",
                colour=S.ALERT,
                label_y=plot_top - 8,
                anchor="end" if threshold > 0.78 else "start",
            )
        )

    # Budget for the space under the plot, top to bottom: tick labels, the axis title, the density
    # strip and its own label, then the legend. Overlapping these is what makes a chart look
    # assembled rather than drawn, so the strip starts clear of the axis title.
    strip_top = plot_bottom + 56
    parts.append(_density_strip(metrics, x, strip_top, DENSITY_HEIGHT))
    parts.append(S.axis_x(x, y=plot_bottom, tick_values=ticks, title="stated confidence"))
    parts.append(S.axis_y(y, x=plot_left, tick_values=ticks, title="observed accuracy"))
    parts.append(
        S.legend(
            [
                ("stated = actual", S.DIAGONAL, "dash"),
                ("observed", S.INK, "dot"),
                ("95% interval", S.SOFT, "line"),
            ],
            x=plot_left,
            y=height - 20,
        )
    )
    if subtitle:
        parts.append(S.text(plot_left, 20, subtitle, size=11.5, fill=S.MUTED))
    parts.append(S.svg_close())
    return "".join(parts)


def _density_strip(metrics: CalibrationMetrics, x: S.Scale, top: float, height: float) -> str:
    """Where the cases actually pile up, on the same x axis as the curve.

    Quantile calibration bins are equal-sized by construction, so they cannot show volume. This
    strip uses equal-width buckets for exactly that reason, and it is how a reader knows which
    part of the curve carries the traffic.
    """
    buckets = metrics.confidence_buckets
    parts = [S.text(x.range[0], top - 6, "where the decisions are", size=10.5, fill=S.MUTED)]
    if not buckets:
        return "".join(parts)
    peak = max(count for _, _, count in buckets) or 1
    for lo, hi, count in buckets:
        bar_h = height * (count / peak)
        parts.append(
            S.rect(
                x(lo) + 0.5,
                top + height - bar_h,
                max(1.0, x(hi) - x(lo) - 1.0),
                bar_h,
                # Kept out of the palette: the strip is a backdrop, and an inline fill keeps it
                # one shade of the page's own ink in either theme.
                fill=S.DIAGONAL,
                extra=' style="fill: var(--ink); fill-opacity: 0.14"',
            )
        )
    parts.append(
        S.line(x.range[0], top + height, x.range[1], top + height, stroke=S.GRID, width=1.0)
    )
    busiest = max(buckets, key=lambda bucket: bucket[2])
    parts.append(
        S.text(
            x.range[1],
            top - 6,
            f"peak {S.fmt(busiest[0])}-{S.fmt(busiest[1])} · n={busiest[2]}",
            anchor="end",
            size=10.5,
            fill=S.MUTED,
        )
    )
    return "".join(parts)


def _empty(width: float, title: str) -> str:
    height = 120.0
    return (
        S.svg_open(width, height, title=title, desc="No labeled decisions to plot.", cls="chart")
        + S.text(
            S.MARGIN,
            height / 2 + 4,
            "No labeled decisions: a curve here would be a drawing, not a measurement.",
            size=12,
            fill=S.MUTED,
        )
        + S.svg_close()
    )


def reliability_table_rows(metrics: CalibrationMetrics) -> list[list[str]]:
    """Rows for the collapsible data table under the chart."""
    rows: list[list[str]] = []
    for cal_bin in metrics.bins:
        rows.append(
            [
                cal_bin.label,
                f"{cal_bin.n}",
                S.pct(cal_bin.mean_confidence, 0),
                S.pct(cal_bin.accuracy, 0),
                f"[{S.pct(cal_bin.ci_low, 0)}, {S.pct(cal_bin.ci_high, 0)}]",
                f"{cal_bin.gap:+.3f}",
            ]
        )
    return rows


RELIABILITY_HEADERS: tuple[str, ...] = (
    "Confidence bin",
    "n",
    "Stated",
    "Observed",
    "Wilson 95%",
    "Gap",
)


def render_reliability_section(
    metrics: CalibrationMetrics,
    *,
    threshold: float | None = None,
    title: str = "Reliability",
    subtitle: str = "",
) -> str:
    """Chart plus its accessible data table, as one block."""
    chart = render_reliability(metrics, threshold=threshold, title=title, subtitle=subtitle)
    caption = (
        "<figcaption>Points below the dashed line are overconfident: the model claimed more "
        "than it earned. The bars are 95% Wilson intervals, and the strip underneath shows "
        "where the decisions pile up.</figcaption>"
    )
    if metrics.n == 0:
        return f"<figure>{chart}{caption}</figure>"
    table = S.details_table(
        RELIABILITY_HEADERS,
        reliability_table_rows(metrics),
        summary="Data behind this curve",
    )
    return f"<figure>{chart}{caption}{table}</figure>"


def ci_span_note(metrics: CalibrationMetrics) -> str:
    """One honest sentence about how much the sample can support."""
    if metrics.n == 0:
        return "No labeled decisions: nothing to measure yet."
    span = metrics.ece_ci_span
    width = "narrow" if span < 0.05 else ("moderate" if span < 0.12 else "wide")
    return (
        f"ECE {S.fmt(metrics.ece, 3)} with a {width} bootstrap interval "
        f"({S.fmt(metrics.ece_ci_low, 3)}-{S.fmt(metrics.ece_ci_high, 3)}) over {metrics.n} "
        f"labeled decisions."
    )


def density_summary(metrics: CalibrationMetrics, top: int = 3) -> str:
    """Where the volume sits, e.g. for the verdict's supporting line."""
    buckets = sorted(metrics.confidence_buckets, key=lambda bucket: -bucket[2])[:top]
    populated = [b for b in buckets if b[2] > 0]
    if not populated:
        return ""
    share = sum(b[2] for b in populated) / max(1, sum(b[2] for b in metrics.confidence_buckets))
    spans: Sequence[str] = [f"{S.fmt(b[0])}-{S.fmt(b[1])}" for b in populated]
    return f"{S.pct(share, 0)} of decisions land in {', '.join(spans)}"
