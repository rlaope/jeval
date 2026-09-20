"""Cost curve and impact table.

The cost curve answers the only question that matters for a threshold: what does each candidate
line cost per case, and how much worse is the line you are on now. The flat region matters as
much as the minimum — it says out loud when the data cannot distinguish neighbouring thresholds.
"""

from __future__ import annotations

import json
from html import escape

from jeval.report import svg as S
from jeval.report.model import CostPoint, ImpactTable, ThresholdResult

WIDTH = 660.0
HEIGHT = 400.0
LEFT = 74.0
RIGHT = 150.0
TOP = 40.0
BOTTOM = 66.0
CURRENT_COLOUR = "#B00020"

# Impact-row label -> the key the report's JavaScript recomputes for that figure.
FIGURE_KEYS: dict[str, str] = {
    "auto rate": "auto_rate",
    "accuracy (auto)": "accuracy_auto",
    "cost per case": "cost_per_case",
    "monthly cost": "cost_per_month",
    "cost per month": "cost_per_month",
}
MIN_COLOUR = "#111111"


def _finite(points: list[CostPoint]) -> list[CostPoint]:
    return [point for point in points if point.expected_cost == point.expected_cost]


def render_cost_curve(
    result: ThresholdResult,
    *,
    current_threshold: float | None = None,
    width: float = WIDTH,
    height: float = HEIGHT,
    title: str | None = None,
) -> str:
    """Expected cost per case against threshold, with its components stacked."""
    points = _finite(list(result.curve))
    heading = title or f"Expected cost per case · {result.action}"
    if not points:
        return _empty(width, heading, "No cost curve: this action has too few labeled decisions.")
    thresholds = [point.threshold for point in points]
    costs = [point.expected_cost for point in points]
    lo, hi = min(costs), max(costs)
    pad = max(1e-9, (hi - lo) * 0.12)
    x = S.lin_scale((min(thresholds), max(thresholds)), (LEFT, width - RIGHT))
    y = S.lin_scale((max(0.0, lo - pad), hi + pad), (height - BOTTOM, TOP))

    desc = (
        f"Expected cost per case across {len(points)} candidate thresholds for the action "
        f"{result.action}. Minimum at {S.fmt(result.threshold)} with {S.money(result.expected_cost_per_case)} "
        f"per case over {result.n_records} labeled decisions."
    )
    parts: list[str] = [S.svg_open(width, height, title=heading, desc=desc, cls="chart")]
    parts.append(S.text(LEFT, 20, heading, size=12.5, weight=600))

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
                fill="#f4f4f4",
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
    stacked = [
        (
            x(point.threshold),
            y((point.escalate_cost + point.accept_cost) / max(1, result.n_records)),
        )
        for point in points
    ]
    parts.append(S.polyline(stacked, stroke="#8a8a8a", width=1.2, dash="3 3"))
    parts.append(S.polyline(escalate, stroke=S.PALETTE[1], width=1.4))
    parts.append(
        S.polyline([(x(p.threshold), y(p.expected_cost)) for p in points], stroke=S.INK, width=2.0)
    )

    minimum = min(points, key=lambda point: point.expected_cost)
    parts.append(
        S.dot(
            x(minimum.threshold),
            y(minimum.expected_cost),
            4.6,
            fill=MIN_COLOUR,
            tooltip=(
                f"minimum · threshold {S.fmt(minimum.threshold)} · "
                f"{S.money(minimum.expected_cost)}/case · auto {S.pct(minimum.auto_rate, 0)}"
            ),
        )
    )
    parts.append(
        S.text(
            x(minimum.threshold) + 8,
            y(minimum.expected_cost) - 8,
            f"{S.money(minimum.expected_cost)}/case at {S.fmt(minimum.threshold)}",
            size=11,
            weight=600,
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
                    tooltip=(
                        f"current · threshold {S.fmt(current.threshold)} · "
                        f"{S.money(current.expected_cost)}/case · auto {S.pct(current.auto_rate, 0)}"
                    ),
                )
            )
            gap = f"{S.money(current.expected_cost - minimum.expected_cost)}/case"
            parts.append(
                S.text(
                    x(current.threshold) + 8,
                    y(current.expected_cost) + 16,
                    f"{gap} worse than the minimum",
                    size=10.5,
                    fill=CURRENT_COLOUR,
                )
            )

    parts.append(
        S.legend(
            [
                ("total expected cost", S.INK),
                ("escalation share", S.PALETTE[1]),
                ("accept + escalate", "#8a8a8a"),
            ],
            x=LEFT,
            y=height - 34,
        )
    )
    parts.append(
        S.text(
            LEFT,
            height - 16,
            (
                f"{result.n_records} labeled decisions · "
                f"CI {S.fmt(result.ci_low)}-{S.fmt(result.ci_high)} · "
                + (
                    "costs: accept "
                    f"{S.money(result.cost_false_accept)}, escalate {S.money(result.cost_escalate)}, "
                    f"miss {S.money(result.cost_false_reject)}"
                )
            ),
            size=10.5,
            fill=S.MUTED,
        )
    )
    parts.append(S.axis_x(x, y=height - BOTTOM, tick_values=x_ticks, title="confidence threshold"))
    parts.append(S.axis_y(y, x=LEFT, tick_values=y_ticks, format_=S.money, title="cost per case"))
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
    """
    volume = impact.monthly_volume
    figures: list[tuple[str, str, str]] = []
    if result is not None and result.curve:
        per_case = result.expected_cost_per_case
        figures.append(("auto rate", "auto_rate", S.pct(result.auto_rate, 0)))
        figures.append(("accuracy (auto)", "accuracy_auto", S.pct(result.accuracy_auto, 0)))
        figures.append(("cost per case", "cost_per_case", S.money(per_case)))
        if volume:
            figures.append(("cost per month", "cost_per_month", S.money(per_case * volume)))
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
        f'id="{escape(slider_id)}-volume" data-currency="{escape(impact.currency)}"></label>'
        if impact.monthly_volume is not None
        else ""
    )
    slider = (
        f'<div class="slider-block" data-jeval-action="{escape(action)}">'
        f'<label for="{escape(slider_id)}">try another threshold: '
        f'<output data-jeval-threshold for="{escape(slider_id)}" class="num">'
        f"{S.fmt(impact.recommended_threshold)}</output></label>"
        f'<input type="range" id="{escape(slider_id)}" min="0" max="1" step="0.01" '
        f'value="{impact.recommended_threshold:.2f}" data-action="threshold">'
        + (f'<div class="figures">{figures_markup}</div>' if figures_markup else "")
        + '<p class="note">Exploration only: the report never writes thresholds.yaml. '
        "Confirm a value with <code>jeval threshold</code>.</p></div>"
    )
    return (
        f'<table class="impact"><thead>{head}</thead><tbody>{body}</tbody></table>'
        f"{note}{volume}{slider}"
    )


def embed_cost_payload(
    result: ThresholdResult, *, impact: ImpactTable | None = None, summary_markdown: str = ""
) -> str:
    """Aggregated, slider-ready payload: the sweep, never the raw records."""
    payload = {
        "actions": {
            result.action: {
                "question": result.question,
                "when": result.when,
                "threshold": result.threshold,
                "curve": [
                    {
                        "t": round(point.threshold, 4),
                        "cost": round(point.expected_cost, 6),
                        "auto": round(point.auto_rate, 6),
                        "acc": round(point.accuracy_auto, 6),
                        "accept": round(point.accept_cost, 6),
                        "escalate": round(point.escalate_cost, 6),
                    }
                    for point in result.curve
                ],
                "flat_region": list(result.flat_region) if result.flat_region else None,
                "ci": [result.ci_low, result.ci_high],
                "n": result.n_records,
            }
        },
        "impact": (
            {
                "rows": [
                    {"label": row.label, "current": row.current, "recommended": row.recommended}
                    for row in impact.rows
                ],
                "monthly_volume": impact.monthly_volume,
                "currency": impact.currency,
                "current_threshold": impact.current_threshold,
                "recommended_threshold": impact.recommended_threshold,
            }
            if impact is not None
            else None
        ),
        "summary_markdown": summary_markdown,
    }
    return json.dumps(payload, separators=(",", ":"))


def render_cost_section(
    result: ThresholdResult,
    *,
    current_threshold: float | None = None,
    impact: ImpactTable | None = None,
    slider_id: str = "threshold-slider",
) -> str:
    """Curve, its data table, and the impact table when one exists."""
    chart = render_cost_curve(result, current_threshold=current_threshold)
    if not _finite(list(result.curve)):
        return f"<figure>{chart}</figure>"
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
