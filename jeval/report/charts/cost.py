"""Cost curve and impact table.

The cost curve answers the only question that matters for a threshold: what does each candidate
line cost per case, and how much worse is the line you are on now. The flat region matters as
much as the minimum — it says out loud when the data cannot distinguish neighbouring thresholds.
"""

from __future__ import annotations

from jeval.report import svg as S
from jeval.report.model import CostPoint, ImpactTable, ThresholdResult
from jeval.report.svg import escape

WIDTH = 660.0
HEIGHT = 430.0
LEFT = 74.0
RIGHT = 150.0
TOP = 40.0
BOTTOM = 96.0
CURRENT_COLOUR = S.ALERT

# Impact-row label -> the key the report's JavaScript recomputes for that figure.
FIGURE_KEYS: dict[str, str] = {
    "auto rate": "auto_rate",
    "accuracy (auto)": "accuracy_auto",
    "cost per case": "cost_per_case",
    "monthly cost": "cost_per_month",
    "cost per month": "cost_per_month",
}
# The minimum is the mark the report is arguing for, so it wears the accent, not the ink: one
# colour in this document means "this is the proposal", the other means "this is what you run".
MIN_COLOUR = S.ACCENT


def _finite(points: list[CostPoint]) -> list[CostPoint]:
    return [point for point in points if point.expected_cost == point.expected_cost]


def render_cost_curve(
    result: ThresholdResult,
    *,
    current_threshold: float | None = None,
    width: float = WIDTH,
    height: float = HEIGHT,
    title: str | None = None,
    currency: str = "",
) -> str:
    """Expected cost per case against threshold, with its components stacked."""
    points = _finite(list(result.curve))
    heading = title or f"Expected cost per case · {result.action}"
    if not points:
        return _empty(width, heading, "No cost curve: this action has too few labeled decisions.")

    def amount(value: float) -> str:
        compact = S.money(value)
        return f"{currency} {compact}" if currency else compact

    thresholds = [point.threshold for point in points]
    costs = [point.expected_cost for point in points]
    lo, hi = min(costs), max(costs)
    pad = max(1e-9, (hi - lo) * 0.12)
    x = S.lin_scale((min(thresholds), max(thresholds)), (LEFT, width - RIGHT))
    y = S.lin_scale((max(0.0, lo - pad), hi + pad), (height - BOTTOM, TOP))

    desc = (
        f"Expected cost per case across {len(points)} candidate thresholds for the action "
        f"{result.action}. Minimum at {S.fmt(result.threshold)} with "
        f"{amount(result.expected_cost_per_case)} per case over {result.n_records} labeled "
        f"decisions."
    )
    parts: list[str] = [S.svg_open(width, height, title=heading, desc=desc, cls="chart")]
    # The component lines are clipped to the plot: the escalation share keeps rising after the
    # total has turned back down, and an unclipped line would draw over the flat-region label.
    clip_id = f"cost-clip-{S.slug(result.action)}"
    parts.append(
        S.defs(S.clip_rect(clip_id, LEFT, TOP, width - RIGHT - LEFT, height - BOTTOM - TOP))
    )

    if result.flat_region is not None:
        flat_lo, flat_hi = result.flat_region
        # Keep the label inside the plot: a flat region near either edge would otherwise push its
        # caption past the axis, where it collides with the marker or gets clipped.
        centre = (x(flat_lo) + x(flat_hi)) / 2.0
        caption_x = min(max(centre, LEFT + 105), width - RIGHT - 105)
        parts.append(
            S.shaded_region(
                x(flat_lo),
                x(flat_hi),
                y0=TOP,
                y1=height - BOTTOM,
                fill=S.SHADE,
                label="flat region — anything here is fine",
                label_x=caption_x,
                label_y=TOP + 26,
            )
        )

    x_ticks = S.nice_ticks(min(thresholds), max(thresholds), target=5)
    y_ticks = S.nice_ticks(max(0.0, lo - pad), hi + pad, target=4)
    parts.append(S.grid_x(x, x_ticks, y0=TOP, y1=height - BOTTOM))
    parts.append(S.grid_y(y, y_ticks, x0=LEFT, x1=width - RIGHT))

    escalate = [
        (x(point.threshold), y(point.escalate_cost / max(1, result.n_records))) for point in points
    ]
    total = [(x(point.threshold), y(point.expected_cost)) for point in points]
    stacked = [
        (
            x(point.threshold),
            y((point.escalate_cost + point.accept_cost) / max(1, result.n_records)),
        )
        for point in points
    ]
    # The stacked line earns its ink only when it is not the total line again. When a wrong
    # auto-accept is the only cost, the two coincide, and a dashed line drawn under a solid one
    # is noise pretending to be information.
    stacked_adds = any(
        abs(
            point.expected_cost
            - (point.escalate_cost + point.accept_cost) / max(1, result.n_records)
        )
        > 1e-9 * max(1.0, abs(point.expected_cost))
        for point in points
    )
    legend: list[tuple[str, str] | tuple[str, str, str]] = [
        ("total expected cost", S.INK, "line"),
        ("escalation cost", S.ACCENT, "line"),
    ]
    components = ""
    if stacked_adds:
        components += S.polyline(stacked, stroke=S.SOFT, width=1.2, dash="3 3")
        legend.append(("accept + escalate", S.SOFT, "dash"))
    components += S.polyline(escalate, stroke=S.ACCENT, width=1.4, cls="accent-stroke")
    parts.append(S.clipped(clip_id, components))
    parts.append(S.polyline(total, stroke=S.INK, width=2.0))

    minimum = min(points, key=lambda point: point.expected_cost)
    parts.append(
        S.dot(
            x(minimum.threshold),
            y(minimum.expected_cost),
            4.6,
            fill=MIN_COLOUR,
            cls="accent-dot",
            tooltip=(
                f"minimum · threshold {S.fmt(minimum.threshold)} · "
                f"{amount(minimum.expected_cost)}/case · auto {S.pct(minimum.auto_rate, 0)}"
            ),
        )
    )
    parts.append(
        S.text(
            x(minimum.threshold) + 8,
            y(minimum.expected_cost) - 10,
            f"{amount(minimum.expected_cost)}/case at {S.fmt(minimum.threshold)}",
            size=11,
            weight=600,
            halo=True,
        )
    )

    if current_threshold is not None:
        current = result.point_at(current_threshold)
        if current is not None:
            parts.append(
                S.marker_line(
                    x(current.threshold),
                    y0=TOP,
                    y1=height - BOTTOM,
                    label=f"now {S.fmt(current.threshold)}",
                    colour=CURRENT_COLOUR,
                    dash="4 3",
                )
            )
            parts.append(
                S.dot(
                    x(current.threshold),
                    y(current.expected_cost),
                    4.0,
                    fill=CURRENT_COLOUR,
                    cls="alert-dot",
                    tooltip=(
                        f"current · threshold {S.fmt(current.threshold)} · "
                        f"{amount(current.expected_cost)}/case · auto {S.pct(current.auto_rate, 0)}"
                    ),
                )
            )
            gap = amount(current.expected_cost - minimum.expected_cost)
            parts.append(
                S.text(
                    x(current.threshold) + 8,
                    y(current.expected_cost) + 18,
                    f"{gap}/case worse than the minimum",
                    size=10.5,
                    fill=CURRENT_COLOUR,
                    halo=True,
                )
            )

    parts.append(S.legend(legend, x=LEFT, y=height - 30))
    parts.append(
        S.text(
            LEFT,
            height - 12,
            (
                f"{result.n_records} labeled decisions · "
                f"CI {S.fmt(result.ci_low)}-{S.fmt(result.ci_high)} · "
                + (
                    "costs: accept "
                    f"{amount(result.cost_false_accept)}, escalate "
                    f"{amount(result.cost_escalate)}, miss {amount(result.cost_false_reject)}"
                )
            ),
            size=10.5,
            fill=S.MUTED,
        )
    )
    parts.append(S.axis_x(x, y=height - BOTTOM, tick_values=x_ticks, title="confidence threshold"))
    parts.append(
        S.axis_y(
            y,
            x=LEFT,
            tick_values=y_ticks,
            format_=S.money,
            title=f"cost per case ({currency})" if currency else "cost per case",
        )
    )
    parts.append(S.svg_close())
    return "".join(parts)


def cost_table_rows(result: ThresholdResult) -> list[list[str]]:
    return [
        [
            S.fmt(point.threshold),
            S.money(point.expected_cost),
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
    spellings of the same number read as two different numbers.
    """
    volume = impact.monthly_volume
    currency = f"{impact.currency} " if impact.currency else ""
    figures: list[tuple[str, str, str]] = []
    if result is not None and result.curve:
        per_case = result.expected_cost_per_case
        figures.append(("auto rate", "auto_rate", f"{result.auto_rate:.0%}"))
        figures.append(("accuracy (auto)", "accuracy_auto", f"{result.accuracy_auto:.0%}"))
        figures.append(("cost per case", "cost_per_case", f"{currency}{per_case:,.2f}"))
        if volume:
            figures.append(
                ("cost per month", "cost_per_month", f"{currency}{S.money(per_case * volume)}")
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
        '<tr><th>Metric</th><th class="num">current</th>'
        '<th class="num">recommended</th><th class="num">change</th></tr>'
    )
    body = "".join(
        "<tr"
        + (' class="impact-monthly"' if row.label.lower().startswith("monthly") else "")
        + f'><td>{escape(row.label)}</td><td class="num">{escape(row.current)}</td>'
        + f'<td class="num">{escape(row.recommended)}</td>'
        + f'<td class="num">{escape(row.change)}</td></tr>'
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
        f'data-currency="{escape(impact.currency)}">'
        '<div class="slider-head">'
        f'<label for="{escape(slider_id)}">try another threshold</label>'
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
    return f'<table class="impact"><thead>{head}</thead><tbody>{body}</tbody></table>{note}{slider}'


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
        return f"<figure>{chart}{caption}</figure>"
    table = S.details_table(
        COST_HEADERS, cost_table_rows(result), summary="Every threshold in the sweep"
    )
    impact_html = (
        impact_table_html(impact, action=result.action, result=result, slider_id=slider_id)
        if impact is not None
        else ""
    )
    ci_note = (
        f'<p class="note">Recommended threshold {S.fmt(result.threshold)} '
        f"(95% CI {S.fmt(result.ci_low)}-{S.fmt(result.ci_high)}). "
        "A wide interval means more labels, not more analysis, would sharpen this."
        "</p>"
    )
    return f"<figure>{chart}{table}</figure>{ci_note}{impact_html}"


def _empty(width: float, title: str, message: str) -> str:
    height = 120.0
    return (
        S.svg_open(width, height, title=title, desc=message, cls="chart")
        + S.text(S.MARGIN, height / 2 + 4, message, size=12, fill=S.MUTED)
        + S.svg_close()
    )
