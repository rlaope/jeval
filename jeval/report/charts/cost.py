"""Cost curve, threshold ruler and impact table.

The cost curve answers the only question that matters for a threshold: what does each candidate
line cost per case, and how much worse is the line you are on now. The flat region matters as
much as the minimum — it says out loud when the data cannot distinguish neighbouring thresholds.

Colour carries two meanings in this module and no others: the accent is the recommendation, the
alert is the line in use. The curve's own components are drawn in greys, so a reader who learns
"blue is the proposal" once never has to unlearn it on the next chart.
"""

from __future__ import annotations

import math
from itertools import pairwise

from jeval.currency import (
    format_amount,
    format_compact,
    format_delta,
    format_percent,
    minor_units,
)
from jeval.report import svg as S
from jeval.report.model import CostPoint, ImpactRow, ImpactTable, ThresholdResult
from jeval.report.svg import escape

WIDTH = 880.0
HEIGHT = 440.0
LEFT = 84.0
RIGHT = 28.0
TOP = 52.0
BOTTOM = 126.0
CURRENT_COLOUR = S.ALERT
# The minimum is the mark the report is arguing for, so it wears the accent, not the ink: one
# colour in this document means "this is the proposal", the other means "this is what you run".
MIN_COLOUR = S.ACCENT

# Impact-row label -> the key the report's JavaScript recomputes for that figure.
FIGURE_KEYS: dict[str, str] = {
    "auto rate": "auto_rate",
    "accuracy (auto)": "accuracy_auto",
    "cost per case": "cost_per_case",
    "monthly cost": "cost_per_month",
    "cost per month": "cost_per_month",
}

# Which way is better for each impact row. A row that is neither (the threshold itself, the auto
# rate) gets a direction mark and no judgement: escalating more is a cost or a saving depending on
# the matrix, and the cost rows already say which.
_BETTER_WHEN: dict[str, int] = {
    "accuracy (auto)": +1,
    "cost per case": -1,
    "monthly cost": -1,
    "cost per month": -1,
}


def _finite(points: list[CostPoint]) -> list[CostPoint]:
    return [point for point in points if point.expected_cost == point.expected_cost]


def _lane_labels(
    marks: list[tuple[float, str, str, str]], *, y: float, left: float, right: float
) -> str:
    """Labels for vertical marks, set in a lane above the plot so they never sit on the curve.

    ``marks`` is ``(x, text, colour, css class)``. Two marks share the lane by facing away from each
    other: the left one reads leftward from its line, the right one rightward. A label that would
    run off the plot is turned to face inward, and one that would then overlap its neighbour moves
    up one row instead of printing on top of it.
    """
    ordered = sorted(marks, key=lambda mark: mark[0])
    size = 13.0
    row_gap = 15.0
    placed: list[tuple[float, float, float]] = []  # (start, end, row y)
    parts: list[str] = []
    for index, (x, label, colour, cls) in enumerate(ordered):
        width = S.text_width(label, size)
        anchor = "end" if (len(ordered) > 1 and index == 0) else "start"
        if anchor == "start" and x + 6 + width > right:
            anchor = "end"
        if anchor == "end" and x - 6 - width < left:
            anchor = "start"
        offset = 6.0 if anchor == "start" else -6.0
        start = x + offset if anchor == "start" else x + offset - width
        end = start + width
        row = y
        while any(
            start < p_end + 8 and end > p_start - 8 and p_row == row
            for p_start, p_end, p_row in placed
        ):
            row -= row_gap
        placed.append((start, end, row))
        parts.append(
            S.text(
                x + offset, row, label, anchor=anchor, size=size, weight=600, fill=colour, cls=cls
            )
        )
    return "".join(parts)


def _place_label(
    label: str,
    *,
    at: tuple[float, float],
    size: float,
    curve: list[tuple[float, float]],
    bounds: tuple[float, float, float, float],
    prefer: str,
) -> tuple[float, float, str]:
    """Where a point's label sits: beside the point, on whichever side the curve leaves clear.

    Four candidates -- left or right of the point, above or below it -- are scored by how close the
    curve comes to the label's box; the first candidate with clearance wins, in an order that starts
    on the ``prefer`` side, and the roomiest one wins when none is clear. A fixed offset put the
    label on the curve whenever the curve ran the other way.
    """
    px, py = at
    left, top, right, bottom = bounds
    width = S.text_width(label, size)
    sides = ("end", "start") if prefer == "end" else ("start", "end")
    candidates: list[tuple[float, float, str]] = []
    for anchor in sides:
        for dy in (-12.0, size + 12.0):
            tx = px - 10 if anchor == "end" else px + 10
            candidates.append((tx, min(bottom - 4, max(top + size, py + dy)), anchor))

    def clearance(candidate: tuple[float, float, str]) -> float:
        tx, ty, anchor = candidate
        x0 = tx - width if anchor == "end" else tx
        x1 = x0 + width
        if x0 < left or x1 > right:
            return -1.0
        y0, y1 = ty - size, ty + 3
        lo_x, hi_x = x0 - 2, x1 + 2
        best = float(size)
        # Segments, not vertices: a steep step between two sweep points crosses the box even when
        # neither end lies inside it.
        for (ax, ay), (bx, by) in pairwise(curve):
            if bx < lo_x or ax > hi_x or bx == ax:
                continue
            ends = []
            for cx in (max(ax, lo_x), min(bx, hi_x)):
                ends.append(ay + (by - ay) * (cx - ax) / (bx - ax))
            seg_lo, seg_hi = min(ends), max(ends)
            if seg_hi >= y0 and seg_lo <= y1:
                return 0.0
            best = min(best, y0 - seg_hi if seg_hi < y0 else seg_lo - y1)
        return best

    scored = [(clearance(candidate), candidate) for candidate in candidates]
    for score, candidate in scored:
        if score >= 4.0:
            return candidate
    return max(scored, key=lambda item: item[0])[1]


def render_cost_curve(
    result: ThresholdResult,
    *,
    current_threshold: float | None = None,
    width: float = WIDTH,
    height: float = HEIGHT,
    title: str | None = None,
    currency: str = "",
) -> str:
    """Expected cost per case against threshold, with its components and both lines named."""
    points = _finite(list(result.curve))
    heading = title or f"Expected cost per case · {result.action}"
    if not points:
        return _empty(width, heading, "No cost curve: this action has too few labeled decisions.")

    code = currency or None

    def amount(value: float) -> str:
        return format_amount(value, code) if code else format_compact(value)

    thresholds = [point.threshold for point in points]
    costs = [point.expected_cost for point in points]
    lo, hi = min(costs), max(costs)
    pad = max(1e-9, (hi - lo) * 0.14)
    plot_left, plot_right = LEFT, width - RIGHT
    plot_top, plot_bottom = TOP, height - BOTTOM
    x = S.lin_scale((min(thresholds), max(thresholds)), (plot_left, plot_right))
    y = S.lin_scale((max(0.0, lo - pad), hi + pad), (plot_bottom, plot_top))
    minimum = min(points, key=lambda point: point.expected_cost)

    desc = (
        f"Expected cost per case across {len(points)} candidate thresholds for the action "
        f"{result.action}. Minimum at {S.fmt(result.threshold)} with "
        f"{amount(result.expected_cost_per_case)} per case over {result.n_records} labeled "
        f"decisions."
    )
    parts: list[str] = [S.svg_open(width, height, title=heading, desc=desc, cls="chart")]
    # The component lines are clipped to the plot: the escalation share keeps rising after the
    # total has turned back down, and an unclipped line would draw over the labels above it.
    clip_id = f"cost-clip-{S.slug(result.action)}"
    parts.append(
        S.defs(
            S.clip_rect(
                clip_id, plot_left, plot_top, plot_right - plot_left, plot_bottom - plot_top
            )
        )
    )

    legend: list[tuple[str, str] | tuple[str, str, str]] = [
        ("total expected cost", S.INK, "line"),
        ("escalation share", S.SOFT, "line"),
    ]
    if result.flat_region is not None:
        flat_lo, flat_hi = result.flat_region
        parts.append(
            S.shaded_region(x(flat_lo), x(flat_hi), y0=plot_top, y1=plot_bottom, fill=S.SHADE)
        )

    x_ticks = S.nice_ticks(min(thresholds), max(thresholds), target=5)
    y_ticks = S.nice_ticks(max(0.0, lo - pad), hi + pad, target=4)
    parts.append(S.grid_y(y, y_ticks, x0=plot_left, x1=plot_right))

    denominator = max(1, result.n_records)
    escalate = [(x(point.threshold), y(point.escalate_cost / denominator)) for point in points]
    total = [(x(point.threshold), y(point.expected_cost)) for point in points]
    stacked = [
        (x(point.threshold), y((point.escalate_cost + point.accept_cost) / denominator))
        for point in points
    ]
    # The stacked line earns its ink only when it is not the total line again. When a wrong
    # auto-accept is the only cost, the two coincide, and a dashed line drawn under a solid one
    # is noise pretending to be information.
    stacked_adds = any(
        abs(point.expected_cost - (point.escalate_cost + point.accept_cost) / denominator)
        > 1e-9 * max(1.0, abs(point.expected_cost))
        for point in points
    )
    components = ""
    if stacked_adds:
        components += S.polyline(stacked, stroke=S.SOFT, width=1.2, dash="3 3")
        legend.append(("accept + escalate", S.SOFT, "dash"))
    components += S.polyline(escalate, stroke=S.SOFT, width=1.4)
    parts.append(S.clipped(clip_id, components))
    if result.flat_region is not None:
        legend.append(("flat region: the sample cannot separate these", S.SHADE, "band"))

    # The recommendation, as a line across the plot: the dot says where the minimum is, the line
    # says which threshold that is, so a reader never has to drop a perpendicular by eye.
    marks: list[tuple[float, str, str, str]] = []
    rec_x = x(minimum.threshold)
    parts.append(
        S.line(
            rec_x, plot_top, rec_x, plot_bottom, stroke=MIN_COLOUR, width=1.5, cls="accent-stroke"
        )
    )
    current = result.point_at(current_threshold) if current_threshold is not None else None
    same_line = current is not None and abs(current.threshold - minimum.threshold) < 1e-9
    marks.append(
        (
            rec_x,
            f"{'in use = recommended' if same_line else 'recommended'} {S.fmt(minimum.threshold)}",
            S.ACCENT,
            "accent-ink",
        )
    )
    if current is not None and not same_line:
        cur_x = x(current.threshold)
        parts.append(
            S.line(
                cur_x,
                plot_top,
                cur_x,
                plot_bottom,
                stroke=CURRENT_COLOUR,
                width=1.5,
                dash="5 4",
                cls="alert-stroke",
            )
        )
        marks.append((cur_x, f"in use {S.fmt(current.threshold)}", CURRENT_COLOUR, "alert-ink"))
    parts.append(_lane_labels(marks, y=plot_top - 14, left=4.0, right=width - 4.0))

    parts.append(S.polyline(total, stroke=S.INK, width=2.0))

    # Point labels face away from the other line, so the two annotations never meet in the middle.
    min_label = f"{amount(minimum.expected_cost)} / case"
    min_y = y(minimum.expected_cost)
    faces_right = current is None or same_line or current.threshold <= minimum.threshold
    bounds = (plot_left, plot_top, plot_right, plot_bottom)
    tx, ty, min_anchor = _place_label(
        min_label,
        at=(rec_x, min_y),
        size=12.5,
        curve=total,
        bounds=bounds,
        prefer="start" if faces_right else "end",
    )
    parts.append(S.text(tx, ty, min_label, anchor=min_anchor, size=12.5, weight=600, halo=True))
    parts.append(
        S.dot(
            rec_x,
            min_y,
            5.0,
            fill=MIN_COLOUR,
            cls="accent-dot",
            tooltip=(
                f"minimum · threshold {S.fmt(minimum.threshold)} · "
                f"{amount(minimum.expected_cost)}/case · auto {S.pct(minimum.auto_rate, 0)}"
            ),
            extra=' style="stroke: var(--panel); stroke-width: 2px"',
        )
    )

    if current is not None and not same_line:
        cur_x = x(current.threshold)
        cur_y = y(current.expected_cost)
        gap = current.expected_cost - minimum.expected_cost
        gap_text = format_delta(gap, code) if code else f"+{S.money(gap)}"
        gap_label = f"{gap_text} / case vs the minimum"
        gx, gy, anchor = _place_label(
            gap_label,
            at=(cur_x, cur_y),
            size=11.5,
            curve=total,
            bounds=bounds,
            prefer="end" if current.threshold < minimum.threshold else "start",
        )
        parts.append(
            S.text(
                gx,
                gy,
                gap_label,
                anchor=anchor,
                size=11.5,
                weight=600,
                fill=CURRENT_COLOUR,
                halo=True,
                cls="alert-ink",
            )
        )
        parts.append(
            S.dot(
                cur_x,
                cur_y,
                4.6,
                fill=CURRENT_COLOUR,
                cls="alert-dot",
                tooltip=(
                    f"in use · threshold {S.fmt(current.threshold)} · "
                    f"{amount(current.expected_cost)}/case · auto {S.pct(current.auto_rate, 0)}"
                ),
                extra=' style="stroke: var(--panel); stroke-width: 2px"',
            )
        )

    parts.append(S.axis_x(x, y=plot_bottom, tick_values=x_ticks, title="confidence threshold"))
    parts.append(
        S.axis_y(
            y,
            x=plot_left,
            tick_values=y_ticks,
            format_=S.tick_format(y_ticks),
            title=f"cost per case ({code})" if code else "cost per case",
        )
    )
    parts.append(S.legend(legend, x=plot_left, y=height - 44))
    parts.append(
        S.text(
            plot_left,
            height - 24,
            (
                f"{result.n_records:,} labeled decisions · 95% CI on the threshold "
                f"{S.fmt(result.ci_low)}–{S.fmt(result.ci_high)}"
            ),
            size=11.5,
            fill=S.MUTED,
        )
    )
    parts.append(
        S.text(
            plot_left,
            height - 7,
            (
                f"a wrong auto-accept costs {amount(result.cost_false_accept)} · a review "
                f"{amount(result.cost_escalate)} · a missed automation "
                f"{amount(result.cost_false_reject)}"
            ),
            size=11.5,
            fill=S.MUTED,
        )
    )
    parts.append(S.svg_close())
    return "".join(parts)


def render_threshold_ruler(
    result: ThresholdResult,
    *,
    current_threshold: float | None,
    width: float = 880.0,
    cls: str = "ruler-svg",
) -> str:
    """The verdict in one picture: the line in use and the recommended line on the 0-1 scale.

    The recommended threshold's bootstrap interval is drawn as a band, so "0.75" is never read as
    more exact than the sample allows, and the gap between the two lines is the move the verdict is
    asking for.
    """
    if not math.isfinite(result.threshold):
        return ""
    height = 84.0
    left, right = 10.0, width - 10.0
    track_y = 50.0
    x = S.lin_scale((0.0, 1.0), (left, right))
    has_current = current_threshold is not None and math.isfinite(current_threshold)
    desc = (
        f"Confidence threshold for {result.action} on a scale from 0 to 1. Recommended "
        f"{S.fmt(result.threshold)} with a 95% interval of {S.fmt(result.ci_low)} to "
        f"{S.fmt(result.ci_high)}"
        + (f"; the line in use is {S.fmt(current_threshold or 0.0)}." if has_current else ".")
        + " Cases at or above the line run automatically; below it they go to a human."
    )
    parts = [S.svg_open(width, height, title=f"Threshold · {result.action}", desc=desc, cls=cls)]
    parts.append(S.rect(left, track_y - 3, right - left, 6, fill=S.GRID, rx=3.0))
    if math.isfinite(result.ci_low) and math.isfinite(result.ci_high):
        parts.append(
            S.rect(
                x(result.ci_low),
                track_y - 9,
                max(2.0, x(result.ci_high) - x(result.ci_low)),
                18,
                fill=S.ACCENT,
                cls="accent-band",
                rx=3.0,
            )
        )
    ticks = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) if width >= 560 else (0.0, 0.5, 1.0)
    for tick in ticks:
        anchor = "start" if tick == 0.0 else ("end" if tick == 1.0 else "middle")
        parts.append(
            S.text(x(tick), track_y + 28, S.fmt(tick, 1), anchor=anchor, size=11.5, fill=S.MUTED)
        )
    rec_x = x(result.threshold)
    marks: list[tuple[float, str, str, str]] = []
    same = has_current and abs((current_threshold or 0.0) - result.threshold) < 1e-9
    if has_current and not same:
        cur_x = x(current_threshold or 0.0)
        parts.append(
            S.line(
                cur_x,
                track_y - 16,
                cur_x,
                track_y + 16,
                stroke=S.ALERT,
                width=2.0,
                dash="4 3",
                cls="alert-stroke",
            )
        )
        marks.append((cur_x, f"in use {S.fmt(current_threshold or 0.0)}", S.ALERT, "alert-ink"))
        # The move, as an arrow along the track from the line in use to the recommended one.
        direction = 1.0 if rec_x > cur_x else -1.0
        if abs(rec_x - cur_x) > 18:
            parts.append(
                S.line(
                    cur_x + 5 * direction,
                    track_y,
                    rec_x - 9 * direction,
                    track_y,
                    stroke=S.INK,
                    width=1.5,
                )
            )
            tip = rec_x - 8 * direction
            parts.append(
                S.polygon(
                    [(tip, track_y - 4), (tip + 6 * direction, track_y), (tip, track_y + 4)],
                    fill=S.INK,
                    cls="",
                )
            )
    parts.append(
        S.line(
            rec_x,
            track_y - 16,
            rec_x,
            track_y + 16,
            stroke=S.ACCENT,
            width=2.5,
            cls="accent-stroke",
        )
    )
    marks.append(
        (
            rec_x,
            f"{'in use = recommended' if same else 'recommended'} {S.fmt(result.threshold)}",
            S.ACCENT,
            "accent-ink",
        )
    )
    parts.append(_lane_labels(marks, y=track_y - 24, left=left, right=right))
    parts.append(S.svg_close())
    return "".join(parts)


def cost_table_rows(result: ThresholdResult, currency: str = "") -> list[list[str]]:
    code = currency or None
    return [
        [
            S.fmt(point.threshold),
            format_amount(point.expected_cost, code, unit=False)
            if code
            else S.money(point.expected_cost),
            S.pct(point.auto_rate, 0),
            S.pct(point.accuracy_auto, 0),
            f"{point.n_wrong_auto}",
            f"{point.n_wrong_escalate}",
        ]
        for point in result.curve
    ]


COST_HEADERS: tuple[str, ...] = (
    "Threshold",
    "Cost / case",
    "Auto rate",
    "Accuracy (auto)",
    "Wrong (auto)",
    "Wrong (escalated)",
)


def _slider_figures(
    result: ThresholdResult | None, impact: ImpactTable
) -> tuple[tuple[str, str, str], ...]:
    """The figures the report's slider recomputes, as ``(label, key, initial value)``.

    Driven by the threshold result rather than by whichever impact rows happen to exist: the
    script updates every ``data-jeval-value`` it finds, so a figure that is never emitted is a
    control that silently does nothing.

    ``tests/test_report_wiring.py`` and the inline script in :mod:`jeval.report.assets` must agree
    on both the keys and the formatting: the readout sits directly under the impact table, and two
    spellings of the same number read as two different numbers. Both sides format through the rules
    in :mod:`jeval.currency`.
    """
    volume = impact.monthly_volume
    code = impact.currency or None
    figures: list[tuple[str, str, str]] = []
    if result is not None and result.curve:
        per_case = result.expected_cost_per_case
        figures.append(("auto rate", "auto_rate", format_percent(result.auto_rate)))
        figures.append(("accuracy (auto)", "accuracy_auto", format_percent(result.accuracy_auto)))
        figures.append(("cost per case", "cost_per_case", format_amount(per_case, code)))
        if volume:
            figures.append(
                ("cost per month", "cost_per_month", format_compact(per_case * volume, code))
            )
            figures.append(("auto per month", "auto_per_month", S.money(result.auto_rate * volume)))
            figures.append(
                (
                    "escalations per month",
                    "escalations_per_month",
                    S.money((1 - result.auto_rate) * volume),
                )
            )
    if figures:
        return tuple(figures)
    return tuple(
        (row.label, key, row.recommended)
        for row, key in ((row, FIGURE_KEYS.get(row.label.strip().lower())) for row in impact.rows)
        if key
    )


def _delta_cell(row: ImpactRow) -> str:
    """The change column: the figure as written, plus which way it moved and whether that helps.

    The arrow and the judgement are presentation only; the text inside is the impact row's own
    string, so a copied value is exactly the number the table computed.
    """
    text = row.change.strip()
    if not text or text == "n/a":
        return escape(row.change)
    direction = "up" if text.startswith("+") else ("down" if text.startswith(("-", "−")) else "")
    if not direction:
        return escape(row.change)
    better = _BETTER_WHEN.get(row.label.strip().lower(), 0)
    judgement = ""
    if better:
        judgement = " better" if (better > 0) == (direction == "up") else " worse"
    # The colour is backed by a word: hovering (or a screen reader) says "better" or "worse".
    label = (
        f' title="{judgement.strip()}" aria-label="{escape(row.change)}, {judgement.strip()}"'
        if judgement
        else ""
    )
    return f'<span class="delta {direction}{judgement}"{label}>{escape(row.change)}</span>'


def impact_table_html(
    impact: ImpactTable,
    *,
    action: str = "",
    result: ThresholdResult | None = None,
    slider_id: str = "threshold-slider",
) -> str:
    """The persuasion table plus the slider scaffolding the inline JavaScript drives.

    The markup is the contract: ``data-jeval-action`` marks the box, ``[data-jeval-threshold]``
    is the slider's own label, and each figure carries ``data-jeval-value`` with the key the
    script recomputes. Renaming any of them silently disables the interaction, which is why
    ``tests/test_report_wiring.py`` asserts the hooks exist in a rendered document.
    """
    head = (
        '<tr><th>Metric</th><th class="num">In use</th>'
        '<th class="num">Recommended</th><th class="num">Change</th></tr>'
    )
    body = "".join(
        "<tr"
        + (' class="impact-monthly"' if row.label.lower().startswith("monthly") else "")
        + f'><td>{escape(row.label)}</td><td class="num">{escape(row.current)}</td>'
        + f'<td class="num">{escape(row.recommended)}</td>'
        + f'<td class="num">{_delta_cell(row)}</td></tr>'
        for row in impact.rows
    )
    figures_markup = "".join(
        '<div class="figure"><span class="k">'
        + escape(label)
        + '</span><span class="v num" data-jeval-value="'
        + key
        + '">'
        + escape(value)
        + "</span></div>"
        for label, key, value in _slider_figures(result, impact)
    )
    note = f'<p class="note">{escape(impact.paradox_note)}</p>' if impact.paradox_note else ""
    volume = (
        '<label class="volume">monthly cases '
        f'<input type="number" min="0" step="100" value="{int(impact.monthly_volume or 0)}" '
        f'id="{escape(slider_id)}-volume"></label>'
        if impact.monthly_volume is not None
        else ""
    )
    slider = (
        f'<div class="slider" data-jeval-action="{escape(action)}" '
        f'data-currency="{escape(impact.currency)}" '
        f'data-currency-digits="{minor_units(impact.currency)}">'
        '<div class="slider-head">'
        f'<label for="{escape(slider_id)}">Try another threshold</label>'
        f'<output data-jeval-threshold for="{escape(slider_id)}" class="num">'
        f"{S.fmt(impact.recommended_threshold)}</output>"
        "</div>"
        f'<input type="range" id="{escape(slider_id)}" min="0" max="1" step="0.01" '
        f'value="{impact.recommended_threshold:.2f}" data-action="threshold">'
        '<div class="scale"><span>0.00</span><span>1.00</span></div>'
        + volume
        + (f'<div class="figures">{figures_markup}</div>' if figures_markup else "")
        + '<p class="note">Exploration only: the report never writes thresholds.yaml. '
        "Confirm a value with <code>jeval threshold</code>.</p></div>"
    )
    table = f'<table class="impact"><thead>{head}</thead><tbody>{body}</tbody></table>'
    return _IMPACT_PARTS_SEPARATOR.join((table, note, slider))


# impact_table_html's three parts, joined so the historical one-string contract still holds and
# split again by render_cost_section, which places them apart. The marker is an HTML comment, so a
# caller that concatenates the whole string still renders exactly what it did.
_IMPACT_PARTS_SEPARATOR = "<!--jeval:impact-part-->"


def _signed_share(after: float, before: float) -> float:
    return math.nan if not before else after / before - 1.0


def impact_figures_html(
    impact: ImpactTable, result: ThresholdResult, *, current_threshold: float | None
) -> str:
    """The finding as a sentence, then the four figures it rests on, each set large.

    Read by someone deciding, not auditing: the sentence says what moving the line buys, in money
    when there is a monthly volume, and each card shows the figure after the move with the figure
    before it struck through. The table with every cell stays one click away.
    """
    if current_threshold is None or not result.curve or not math.isfinite(result.threshold):
        return ""
    before = result.point_at(current_threshold)
    after = result.point_at(result.threshold)
    if before is None or after is None:
        return ""
    code = impact.currency or None
    volume = impact.monthly_volume
    move = (
        f'Moving the line from <b class="cur">{S.fmt(before.threshold)}</b> to '
        f'<b class="rec">{S.fmt(after.threshold)}</b>'
    )
    saving = before.expected_cost - after.expected_cost
    if abs(after.threshold - before.threshold) < 1e-9:
        claim = (
            f'The line in use, <b class="cur">{S.fmt(before.threshold)}</b>, is already at the '
            "cost minimum"
        )
    elif volume:
        verb = "saves" if saving > 0 else "costs"
        claim = f"{move} {verb} <b>{escape(format_amount(abs(saving) * volume, code))}</b> a month"
    else:
        verb = "cuts" if saving > 0 else "raises"
        claim = f"{move} {verb} cost per case by <b>{escape(format_amount(abs(saving), code))}</b>"
    context = (
        f"at {volume:,.0f} cases a month · " if volume else ""
    ) + f"95% interval on the recommended line {S.fmt(result.ci_low)}–{S.fmt(result.ci_high)}"

    def card(label: str, was: str, now: str, change: str, judgement: str, unit: str = "") -> str:
        arrow = "▼" if change.startswith("-") else ("▲" if change.startswith("+") else "")
        verdict = {"better": " · better", "worse": " · worse"}.get(judgement, "")
        return (
            f'<div class="figcard"><div class="fc-label">{escape(label)}</div>'
            f'<div class="fc-now num">'
            + (f'<span class="fc-unit">{escape(unit)}</span>' if unit else "")
            + f'{escape(now)}</div><div class="fc-was">from <s class="num">{escape(was)}</s></div>'
            f'<div class="fc-change {judgement}">{arrow} {escape(change)}{verdict}</div></div>'
        )

    def money(value: float) -> tuple[str, str]:
        text = format_amount(value, code) if code else S.money(value)
        if code and text.startswith(f"{code} "):
            return code, text[len(code) + 1 :]
        return "", text

    cost_share = _signed_share(after.expected_cost, before.expected_cost)
    cost_judgement = "better" if saving > 0 else ("worse" if saving < 0 else "")
    unit, now_case = money(after.expected_cost)
    _, was_case = money(before.expected_cost)
    cards = [card("Cost per case", was_case, now_case, f"{cost_share:+.1%}", cost_judgement, unit)]
    if volume:
        cards.append(
            card(
                "Monthly cost",
                format_compact(before.expected_cost * volume, None),
                format_compact(after.expected_cost * volume, None),
                f"{cost_share:+.1%}",
                cost_judgement,
                unit,
            )
        )
    accuracy_move = (after.accuracy_auto - before.accuracy_auto) * 100
    cards.append(
        card(
            "Right when automated",
            format_percent(before.accuracy_auto),
            format_percent(after.accuracy_auto),
            f"{accuracy_move:+.1f} pt",
            "better" if accuracy_move > 0 else ("worse" if accuracy_move < 0 else ""),
        )
    )
    cards.append(
        card(
            "Share automated",
            format_percent(before.auto_rate),
            format_percent(after.auto_rate),
            f"{(after.auto_rate - before.auto_rate) * 100:+.1f} pt",
            "",
        )
    )
    return (
        f'<div class="finding"><p class="claim">{claim}</p><p class="claim-sub">{context}</p>'
        f'<div class="figcards">{"".join(cards)}</div></div>'
    )


def render_cost_section(
    result: ThresholdResult,
    *,
    current_threshold: float | None = None,
    impact: ImpactTable | None = None,
    slider_id: str = "threshold-slider",
) -> str:
    """Curve, its data table, and the impact table when one exists."""
    currency = impact.currency if impact is not None else ""
    chart = render_cost_curve(result, current_threshold=current_threshold, currency=currency)
    caption = (
        "<figcaption>The lowest point of the curve is the recommendation. The shaded band is "
        "the flat region — thresholds inside it cost the same within what this sample can "
        "resolve.</figcaption>"
    )
    if not _finite(list(result.curve)):
        return f'<div class="plate"><figure>{chart}{caption}</figure></div>'
    headers = COST_HEADERS
    if currency:
        headers = (COST_HEADERS[0], f"Cost / case ({currency})", *COST_HEADERS[2:])
    table = S.details_table(
        headers, cost_table_rows(result, currency), summary="Every threshold in the sweep"
    )
    ci_note = (
        f'<p class="note">Recommended threshold {S.fmt(result.threshold)} '
        f"(95% CI {S.fmt(result.ci_low)}–{S.fmt(result.ci_high)}). "
        "A wide interval means more labels, not more analysis, would sharpen this."
        "</p>"
    )
    plate = f'<div class="plate"><figure>{chart}{caption}</figure></div>'
    if impact is None:
        return f"{plate}{ci_note}{table}"
    impact_table, paradox, slider = impact_table_html(
        impact, action=result.action, result=result, slider_id=slider_id
    ).split(_IMPACT_PARTS_SEPARATOR)
    figures = impact_figures_html(impact, result, current_threshold=current_threshold)
    raw = (
        '<details class="raw"><summary>The numbers behind this, as a table</summary>'
        f"{impact_table}{paradox}</details>"
    )
    return f"{figures}{plate}{ci_note}{slider}{raw}{table}"


def _empty(width: float, title: str, message: str) -> str:
    height = 120.0
    return (
        S.svg_open(width, height, title=title, desc=message, cls="chart")
        + S.text(S.MARGIN, height / 2 + 4, message, size=12.5, fill=S.MUTED)
        + S.svg_close()
    )
