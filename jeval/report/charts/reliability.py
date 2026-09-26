"""Reliability diagram: stated confidence against observed accuracy.

The centrepiece of the report. Everything about it exists to stop a reader from trusting a
point that has three records behind it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from jeval.calibration import CalibrationMetrics
from jeval.report import svg as S
from jeval.report.charts.cost import _lane_labels

WIDTH = 660.0
HEIGHT = 480.0
LEFT = 66.0
RIGHT = 18.0
TOP = 34.0
BOTTOM = 142.0
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
    recommended: float | None = None,
    title: str = "Reliability",
    width: float = WIDTH,
    height: float = HEIGHT,
    subtitle: str = "",
) -> str:
    """Draw the reliability curve with Wilson intervals, density strip and both threshold lines.

    ``threshold`` is the line in use, drawn dashed in the alert colour; ``recommended`` is the line
    the cost minimum points at, drawn solid in the accent. Each is named in words, so the two can
    never be read as one another.
    """
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
            size=11.5,
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
        parts.append(
            S.dot(
                px,
                py,
                _radius(cal_bin.n, n_max),
                fill=S.INK,
                tooltip=tooltip,
                extra=' style="stroke: var(--panel); stroke-width: 1.5px"',
            )
        )

    marks: list[tuple[float, str, str, str]] = []
    in_use = threshold if threshold is not None and 0.0 <= threshold <= 1.0 else None
    rec = recommended if recommended is not None and 0.0 <= recommended <= 1.0 else None
    same = in_use is not None and rec is not None and abs(in_use - rec) < 1e-9
    if in_use is not None and not same:
        parts.append(
            S.line(
                x(in_use),
                plot_top,
                x(in_use),
                plot_bottom,
                stroke=S.ALERT,
                width=1.5,
                dash="4 3",
                cls="alert-stroke",
            )
        )
        marks.append((x(in_use), f"in use {S.fmt(in_use)}", S.ALERT, "alert-ink"))
    if rec is not None:
        parts.append(
            S.line(
                x(rec),
                plot_top,
                x(rec),
                plot_bottom,
                stroke=S.ACCENT,
                width=1.5,
                cls="accent-stroke",
            )
        )
        label = f"{'in use = recommended' if same else 'recommended'} {S.fmt(rec)}"
        marks.append((x(rec), label, S.ACCENT, "accent-ink"))
    if marks:
        parts.append(_lane_labels(marks, y=plot_top - 8, left=4.0, right=width - 4.0))

    # Budget for the space under the plot, top to bottom: tick labels, the axis title, the density
    # strip and its own label, then the legend. Overlapping these is what makes a chart look
    # assembled rather than drawn, so the strip starts clear of the axis title.
    strip_top = plot_bottom + 62
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
            y=height - 14,
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
    parts = [S.text(x.range[0], top - 7, "where the decisions are", size=11.5, fill=S.MUTED)]
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
            top - 7,
            f"peak {S.fmt(busiest[0])}–{S.fmt(busiest[1])} · n={busiest[2]}",
            anchor="end",
            size=11.5,
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


def key_figures(metrics: CalibrationMetrics) -> str:
    """The numbers a reader quotes from this chart, beside it rather than under it."""
    rows: list[tuple[str, str, str]] = [
        (
            "ECE",
            S.fmt(metrics.ece, 3),
            f"95% CI {S.fmt(metrics.ece_ci_low, 3)}–{S.fmt(metrics.ece_ci_high, 3)}",
        ),
        ("MCE", S.fmt(metrics.mce, 3), "largest bin gap"),
        ("Brier score", S.fmt(metrics.brier, 3), "lower is better"),
        ("Labeled decisions", f"{metrics.n:,}", f"{len(metrics.bins)} bins, {metrics.binning}"),
    ]
    worst = metrics.worst_bin
    if worst is not None:
        rows.append(
            (
                "Widest gap",
                f"{worst.gap:+.2f}",
                f"{worst.label}: stated {S.pct(worst.mean_confidence, 0)}, "
                f"observed {S.pct(worst.accuracy, 0)}",
            )
        )
    items = "".join(
        f"<div><dt>{S.escape(label)}</dt><dd>{S.escape(value)}"
        f"<small>{S.escape(sub)}</small></dd></div>"
        for label, value, sub in rows
    )
    return f'<dl class="keyfigs">{items}</dl>'


#: A bin whose observed accuracy is this far from what it claimed is named as a finding.
CLAIM_GAP = 0.05


def claim_rows_html(metrics: CalibrationMetrics, question: str = "") -> str:
    """The ranges where what the model said and how often it was right part ways, one row each.

    Each row reads left to right as a sentence — "says 65%, right 46%, overclaims by 19 pt" — with a
    drawing between the two numbers: a tick at what it said, a dot at what happened, the line
    between them the gap, and the band the 95% interval on what happened. Ranges within
    :data:`CLAIM_GAP` are counted in the heading and left out, because the chart above already
    shows them sitting on the diagonal.
    """
    scored = [cal_bin for cal_bin in metrics.bins if cal_bin.n > 0]
    flagged = [cal_bin for cal_bin in scored if abs(cal_bin.gap) > CLAIM_GAP]
    if not flagged:
        return ""
    over = sum(1 for cal_bin in flagged if cal_bin.gap < 0)
    under = len(flagged) - over
    who = f'<span class="ident">{S.escape(question)}</span>' if question else "The model"
    if over and not under:
        claim = f"{who} is right less often than it says in {over} of {len(scored)} ranges"
    elif under and not over:
        claim = f"{who} is right more often than it says in {under} of {len(scored)} ranges"
    else:
        claim = f"{who} overclaims in {over} and underclaims in {under} of {len(scored)} ranges"
    floor = min(min(b.mean_confidence for b in flagged), min(b.ci_low for b in flagged))
    lo = max(0.0, math.floor(floor * 10) / 10)
    width, height = 480.0, 34.0
    x = S.lin_scale((lo, 1.0), (10.0, width - 10.0))
    rows: list[str] = []
    for cal_bin in flagged:
        gap_pt = abs(cal_bin.gap) * 100
        kind = "over" if cal_bin.gap < 0 else "under"
        words = f"{'overclaims' if kind == 'over' else 'underclaims'} by {gap_pt:.0f} pt"
        band_lo, band_hi = x(max(lo, cal_bin.ci_low)), x(min(1.0, cal_bin.ci_high))
        said, was = x(cal_bin.mean_confidence), x(cal_bin.accuracy)
        drawing = (
            f'<svg class="claim-svg" viewBox="0 0 {width:.0f} {height:.0f}" width="{width:.0f}" '
            f'height="{height:.0f}" role="img" aria-label="said {cal_bin.mean_confidence:.0%}, '
            f"right {cal_bin.accuracy:.0%}, 95% interval {cal_bin.ci_low:.0%} to "
            f'{cal_bin.ci_high:.0%}"><title>{S.escape(cal_bin.label)}: {S.escape(words)}</title>'
            f"<desc>Confidence range {S.escape(cal_bin.label)}: stated "
            f"{cal_bin.mean_confidence:.0%}, observed {cal_bin.accuracy:.0%}, 95% interval "
            f"{cal_bin.ci_low:.0%} to {cal_bin.ci_high:.0%}, n={cal_bin.n}.</desc>"
            + S.rect(x(lo), 16, x(1.0) - x(lo), 2, fill=S.GRID, rx=1)
            + S.rect(band_lo, 9, max(2.0, band_hi - band_lo), 16, fill=S.SHADE, cls="shade", rx=3)
            + S.line(said, 17, was, 17, stroke=S.INK, width=2.5)
            + S.rect(said - 1.5, 5, 3, 24, fill=S.SOFT, rx=1)
            + S.dot(
                was, 17, 7, fill=S.INK, extra=' style="stroke: var(--panel); stroke-width: 2px"'
            )
            + "</svg>"
        )
        rows.append(
            f'<div class="claim-row {kind}"><div class="cr-said">says '
            f'<b class="num">{cal_bin.mean_confidence:.0%}</b></div>{drawing}'
            f'<div class="cr-was">right <b class="num">{cal_bin.accuracy:.0%}</b></div>'
            f'<div class="cr-flag">{S.escape(words)}<span>{S.escape(cal_bin.label)} · '
            f"n={cal_bin.n}</span></div></div>"
        )
    ticks = "".join(
        f'<span style="left:{(value - lo) / (1.0 - lo) * 100:.1f}%">{value:.0%}</span>'
        for value in (lo, (lo + 1.0) / 2, 1.0)
    )
    return (
        f'<div class="finding"><p class="claim small">{claim}</p>'
        '<div class="claims-key"><span><i class="k-said"></i>what it said</span>'
        '<span><i class="k-was"></i>how often it was right</span>'
        '<span><i class="k-band"></i>95% interval</span></div>'
        f'<div class="claims">{"".join(rows)}'
        f'<div class="claim-row axis"><div></div><div class="claim-ticks">{ticks}</div></div>'
        "</div></div>"
    )


def render_reliability_section(
    metrics: CalibrationMetrics,
    *,
    threshold: float | None = None,
    recommended: float | None = None,
    title: str = "Reliability",
    subtitle: str = "",
    question: str = "",
) -> str:
    """Chart beside its key figures, the ranges that miss named as rows, then the data table."""
    chart = render_reliability(
        metrics, threshold=threshold, recommended=recommended, title=title, subtitle=subtitle
    )
    caption = (
        "<figcaption>Points below the dashed line are overconfident: the model claimed more "
        "than it earned. Dot size follows the number of decisions in the bin, the bars are 95% "
        "Wilson intervals, and the strip underneath shows where the decisions pile up."
        "</figcaption>"
    )
    if metrics.n == 0:
        return f'<div class="plate"><figure>{chart}{caption}</figure></div>'
    table = S.details_table(
        RELIABILITY_HEADERS,
        reliability_table_rows(metrics),
        summary="The numbers behind this curve, as a table",
    )
    return (
        f'<div class="plate"><figure><div class="chart-row">{chart}{key_figures(metrics)}</div>'
        f"{caption}</figure></div>{claim_rows_html(metrics, question)}{table}"
    )


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
