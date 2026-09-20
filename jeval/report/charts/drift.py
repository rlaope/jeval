"""Drift charts: ECE over time, a before/after curve overlay, and threshold movement.

Drift is the one thing a notebook cannot do, so these charts are built to make the change
visible rather than stated: two curves pulling apart is the evidence, and the numbers are in the
collapsible table underneath.
"""

from __future__ import annotations

from jeval.calibration import CalibrationMetrics
from jeval.report import svg as S
from jeval.report.model import DriftView
from jeval.report.svg import escape

WIDTH = 660.0
LEFT = 62.0
RIGHT = 24.0
BOTTOM = 70.0
CHANGE_COLOUR = "#D55E00"


def render_ece_series(view: DriftView, *, width: float = WIDTH, height: float = 320.0) -> str:
    """ECE per slice with a vertical marker wherever the serving model changed."""
    slices = view.slices
    if not slices:
        return ""
    values = [s.ece for s in slices if s.ece == s.ece]
    if not values:
        return ""
    top = 44.0
    lo, hi = min(values), max(values)
    pad = max(0.005, (hi - lo) * 0.15)
    plot_top, plot_bottom = top, height - BOTTOM
    x = S.lin_scale((0.0, max(1.0, len(slices) - 1.0)), (LEFT, width - RIGHT))
    y = S.lin_scale((0.0, hi + pad), (plot_bottom, plot_top))

    desc = (
        f"Expected calibration error across {len(slices)} slices from {view.baseline_label} to "
        f"{view.current_label}. Highest {S.fmt(max(values), 3)}, lowest {S.fmt(min(values), 3)}."
    )
    parts: list[str] = [
        S.svg_open(width, height, title="ECE over time", desc=desc, cls="chart"),
        S.text(LEFT, 22, "ECE over time", size=12.5, weight=600),
        S.text(LEFT, 38, f"{view.baseline_label} → {view.current_label}", size=11, fill=S.MUTED),
    ]
    y_ticks = S.nice_ticks(0.0, hi + pad, target=4)
    parts.append(S.grid_y(y, y_ticks, x0=LEFT, x1=width - RIGHT))

    changed: dict[str, int] = {}
    for change in view.changes:
        for index, sl in enumerate(slices):
            if sl.model == change.model and sl.model not in changed:
                changed[sl.model] = index
    for model, index in changed.items():
        if index == 0:
            continue
        parts.append(
            S.marker_line(
                x(float(index)),
                y0=plot_top,
                y1=plot_bottom,
                label=model,
                colour=CHANGE_COLOUR,
                dash="4 3",
                label_y=plot_top - 8,
            )
        )

    points = [(x(float(i)), y(sl.ece)) for i, sl in enumerate(slices) if sl.ece == sl.ece]
    parts.append(S.polyline(points, stroke=S.INK, width=2.0, smooth=True))
    for index, sl in enumerate(slices):
        if sl.ece != sl.ece:
            continue
        parts.append(
            S.dot(
                x(float(index)),
                y(sl.ece),
                4.0,
                fill=S.INK,
                tooltip=f"{sl.label} · {sl.model} · ECE {S.fmt(sl.ece, 3)} · n={sl.n}",
            )
        )
        parts.append(
            S.text(
                x(float(index)),
                plot_bottom + 32,
                sl.label,
                anchor="middle",
                size=10.0,
                fill=S.MUTED,
            )
        )
    parts.append(
        S.axis_y(y, x=LEFT, tick_values=y_ticks, format_=lambda v: S.fmt(v, 2), title="ECE")
    )
    parts.append(
        S.legend(
            [("ECE by slice", S.INK), ("model changed", CHANGE_COLOUR)],
            x=LEFT,
            y=height - 10,
        )
    )
    parts.append(S.svg_close())
    return "".join(parts)


def render_overlay(
    baseline: CalibrationMetrics,
    current: CalibrationMetrics,
    *,
    baseline_label: str = "before",
    current_label: str = "after",
    width: float = WIDTH,
    height: float = 340.0,
    threshold: float | None = None,
) -> str:
    """Reliability curves before and after, drawn on one pair of axes."""
    if baseline.n == 0 and current.n == 0:
        return ""
    top = 46.0
    plot_top, plot_bottom = top, height - BOTTOM
    x = S.lin_scale((0.0, 1.0), (LEFT, width - RIGHT))
    y = S.lin_scale((0.0, 1.0), (plot_bottom, plot_top))
    desc = (
        f"Reliability before and after: {baseline_label} ECE {S.fmt(baseline.ece, 3)} over "
        f"{baseline.n} decisions, {current_label} ECE {S.fmt(current.ece, 3)} over {current.n}. "
        "Two curves pulling apart is the drift signal."
    )
    parts: list[str] = [
        S.svg_open(width, height, title="Reliability before and after", desc=desc, cls="chart"),
        S.text(LEFT, 22, "Reliability before and after", size=12.5, weight=600),
    ]
    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    parts.append(S.grid_x(x, ticks, y0=plot_top, y1=plot_bottom))
    parts.append(S.grid_y(y, ticks, x0=LEFT, x1=width - RIGHT))
    parts.append(S.line(x(0.0), y(0.0), x(1.0), y(1.0), stroke=S.DIAGONAL, width=1.2, dash="5 4"))
    for metrics, colour, opacity in ((baseline, S.PALETTE[0], 0.45), (current, S.INK, 1.0)):
        curve = [
            (x(min(max(b.mean_confidence, 0.0), 1.0)), y(min(max(b.accuracy, 0.0), 1.0)))
            for b in metrics.bins
        ]
        parts.append(
            S.polyline(curve, stroke=colour, width=2.0 if opacity == 1.0 else 1.4, opacity=opacity)
        )
        for cal_bin, (px, py) in zip(metrics.bins, curve, strict=True):
            parts.append(
                S.dot(
                    px,
                    py,
                    3.2,
                    fill=colour,
                    opacity=opacity,
                    tooltip=f"{cal_bin.label} · n={cal_bin.n} · actual {S.fmt(cal_bin.accuracy)}",
                )
            )
    if threshold is not None:
        parts.append(
            S.marker_line(
                x(threshold),
                y0=plot_top,
                y1=plot_bottom,
                label=f"threshold {S.fmt(threshold)}",
                colour="#B00020",
                label_y=plot_top - 6,
            )
        )
    parts.append(
        S.legend(
            [
                (f"{baseline_label} (ECE {S.fmt(baseline.ece, 3)}, n={baseline.n})", S.PALETTE[0]),
                (f"{current_label} (ECE {S.fmt(current.ece, 3)}, n={current.n})", S.INK),
            ],
            x=LEFT,
            y=height - 12,
        )
    )
    parts.append(S.axis_x(x, y=plot_bottom, tick_values=ticks, title="stated confidence"))
    parts.append(S.axis_y(y, x=LEFT, tick_values=ticks, title="observed accuracy"))
    parts.append(S.svg_close())
    return "".join(parts)


def render_threshold_steps(view: DriftView, *, width: float = WIDTH, height: float = 260.0) -> str:
    """Recommended threshold per slice, as the staircase it is."""
    slices = [sl for sl in view.slices if sl.threshold is not None]
    if len(slices) < 2:
        return ""
    top = 40.0
    plot_top, plot_bottom = top, height - BOTTOM
    x = S.lin_scale((0.0, max(1.0, len(slices) - 1.0)), (LEFT, width - RIGHT))
    values = [float(sl.threshold) for sl in slices if sl.threshold is not None]
    lo, hi = min(values), max(values)
    pad = max(0.01, (hi - lo) * 0.2)
    y = S.lin_scale((max(0.0, lo - pad), min(1.0, hi + pad)), (plot_bottom, plot_top))
    steps: list[str] = []
    for index in range(len(slices) - 1):
        left_value = float(slices[index].threshold)  # type: ignore[arg-type]
        right_value = float(slices[index + 1].threshold)  # type: ignore[arg-type]
        steps.append(
            S.line(
                x(float(index)),
                y(right_value),
                x(float(index + 1)),
                y(right_value),
                stroke=S.INK,
                width=2.0,
            )
        )
        steps.append(
            S.line(
                x(float(index + 1)),
                y(left_value),
                x(float(index + 1)),
                y(right_value),
                stroke=S.INK,
                width=1.0,
                dash="3 3",
            )
        )
    head = (
        S.svg_open(
            width,
            height,
            title="Recommended threshold over time",
            desc="Recommended confidence threshold per slice; a step means the cost-optimal line moved.",
            cls="chart",
        )
        + S.text(LEFT, 22, "Recommended threshold over time", size=12.5, weight=600)
        + S.grid_y(
            y,
            S.nice_ticks(max(0.0, lo - pad), min(1.0, hi + pad), target=4),
            x0=LEFT,
            x1=width - RIGHT,
        )
    )
    tail = (
        S.axis_x(
            x,
            y=plot_bottom,
            tick_values=[float(i) for i in range(len(slices))],
            format_=lambda v: slices[int(v)].label,
            title="slice",
        )
        + S.axis_y(y, x=LEFT, title="threshold")
        + S.svg_close()
    )
    return head + "".join(steps) + tail


def render_drift_section(
    view: DriftView,
    *,
    baseline_metrics: CalibrationMetrics | None = None,
    current_metrics: CalibrationMetrics | None = None,
    threshold: float | None = None,
) -> str:
    """The whole drift section, or an explanation of why there is none."""
    if not view.slices:
        return f'<p class="note">{view.note or "No drift comparison available."}</p>'
    parts = [f'<p class="note">{escape(view.note)}</p>' if view.note else ""]
    if view.failures:
        parts.append(
            '<div class="warn"><strong>Drift checks failed.</strong><ul>'
            + "".join(
                f"<li><code>{escape(failure.check)}</code> {escape(failure.detail)} "
                f"(value {S.fmt(failure.value, 3)} against limit {S.fmt(failure.limit, 3)})</li>"
                for failure in view.failures
            )
            + "</ul></div>"
        )
    parts.append(f"<figure>{render_ece_series(view)}</figure>")
    if baseline_metrics is not None and current_metrics is not None:
        parts.append(
            f"<figure>{render_overlay(baseline_metrics, current_metrics, baseline_label=view.baseline_label, current_label=view.current_label, threshold=threshold)}</figure>"
        )
    steps = render_threshold_steps(view)
    if steps:
        parts.append(f"<figure>{steps}</figure>")
    parts.append(
        S.details_table(
            ("Slice", "Model", "Start", "End", "n", "ECE", "Threshold", "Auto rate"),
            [
                [
                    sl.label,
                    sl.model,
                    sl.start,
                    sl.end,
                    str(sl.n),
                    S.fmt(sl.ece, 3),
                    S.fmt(sl.threshold) if sl.threshold is not None else "n/a",
                    S.pct(sl.auto_rate, 0) if sl.auto_rate is not None else "n/a",
                ]
                for sl in view.slices
            ],
            summary="Data behind the drift view",
        )
    )
    return "".join(parts)
