"""Inline SVG rendering. No plotting dependency, no external assets."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape

from jeval.calibration import CalibrationMetrics

PLOT_MARGIN_LEFT = 56
PLOT_MARGIN_BOTTOM = 46
PLOT_MARGIN_TOP = 34
PLOT_MARGIN_RIGHT = 18


def _fmt(value: float, digits: int = 2) -> str:
    if value != value:  # NaN
        return "n/a"
    return f"{value:.{digits}f}"


def reliability_diagram(
    metrics: CalibrationMetrics,
    *,
    width: int = 520,
    height: int = 380,
    title: str = "",
) -> str:
    """Reliability diagram: predicted confidence against observed accuracy.

    The diagonal is perfect calibration. Points below it mean the model claimed more
    confidence than its accuracy earned; the vertical whiskers are Wilson intervals.
    """
    plot_w = width - PLOT_MARGIN_LEFT - PLOT_MARGIN_RIGHT
    plot_h = height - PLOT_MARGIN_TOP - PLOT_MARGIN_BOTTOM
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" role="img" aria-label="{escape(title or "reliability diagram")}">',
        "<style>"
        ".jw{stroke:#111;stroke-width:1}"
        ".jg{stroke:#d8d8d8;stroke-width:1}"
        ".jd{stroke:#999;stroke-width:1;stroke-dasharray:4 3}"
        ".jl{stroke:#111;stroke-width:1.6;fill:none}"
        ".jp{fill:#111}"
        ".jt{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;fill:#111}"
        ".jb{fill:#bbb}"
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>',
    ]

    def px(value: float) -> float:
        return PLOT_MARGIN_LEFT + value * plot_w

    def py(value: float) -> float:
        return PLOT_MARGIN_TOP + (1.0 - value) * plot_h

    if title:
        parts.append(
            f'<text class="jt" x="{PLOT_MARGIN_LEFT}" y="20" '
            f'font-weight="600">{escape(title)}</text>'
        )

    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = px(tick)
        y = py(tick)
        parts.append(
            f'<line class="jg" x1="{x:.2f}" y1="{py(0.0):.2f}" x2="{x:.2f}" y2="{py(1.0):.2f}"/>'
        )
        parts.append(
            f'<line class="jg" x1="{px(0.0):.2f}" y1="{y:.2f}" x2="{px(1.0):.2f}" y2="{y:.2f}"/>'
        )
        parts.append(
            f'<text class="jt" x="{px(0.0) - 8:.2f}" y="{y + 4:.2f}" '
            f'text-anchor="end">{tick:.2f}</text>'
        )
        parts.append(
            f'<text class="jt" x="{x:.2f}" y="{py(0.0) + 16:.2f}" '
            f'text-anchor="middle">{tick:.2f}</text>'
        )

    parts.append(
        f'<line class="jd" x1="{px(0.0):.2f}" y1="{py(0.0):.2f}" x2="{px(1.0):.2f}" y2="{py(1.0):.2f}"/>'
    )
    parts.append(
        f'<line class="jw" x1="{px(0.0):.2f}" y1="{py(0.0):.2f}" x2="{px(0.0):.2f}" y2="{py(1.0):.2f}"/>'
    )
    parts.append(
        f'<line class="jw" x1="{px(0.0):.2f}" y1="{py(1.0):.2f}" x2="{px(1.0):.2f}" y2="{py(1.0):.2f}"/>'
    )

    points: list[tuple[float, float]] = []
    for cal_bin in metrics.bins:
        x = px(min(max(cal_bin.mean_confidence, 0.0), 1.0))
        y = py(min(max(cal_bin.accuracy, 0.0), 1.0))
        points.append((x, y))
        if cal_bin.ci_low == cal_bin.ci_low:
            parts.append(
                f'<line x1="{x:.2f}" y1="{py(max(0.0, cal_bin.ci_low)):.2f}" '
                f'x2="{x:.2f}" y2="{py(min(1.0, cal_bin.ci_high)):.2f}" '
                'stroke="#777" stroke-width="1.4"/>'
            )
            parts.append(
                f'<line x1="{x - 3:.2f}" y1="{py(max(0.0, cal_bin.ci_low)):.2f}" '
                f'x2="{x + 3:.2f}" y2="{py(max(0.0, cal_bin.ci_low)):.2f}" stroke="#777" stroke-width="1.4"/>'
            )
            parts.append(
                f'<line x1="{x - 3:.2f}" y1="{py(min(1.0, cal_bin.ci_high)):.2f}" '
                f'x2="{x + 3:.2f}" y2="{py(min(1.0, cal_bin.ci_high)):.2f}" stroke="#777" stroke-width="1.4"/>'
            )
    if len(points) > 1:
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{x:.2f},{y:.2f}" for i, (x, y) in enumerate(points)
        )
        parts.append(f'<path class="jl" d="{path}"/>')
    for x, y in points:
        parts.append(f'<circle class="jp" cx="{x:.2f}" cy="{y:.2f}" r="3.4"/>')

    # Sample-count strip under the axis: a tall bin is a claim worth trusting.
    total = sum(b.n for b in metrics.bins) or 1
    strip_y = py(0.0) + 22
    for cal_bin in metrics.bins:
        x = px(min(max(cal_bin.mean_confidence, 0.0), 1.0))
        bar_h = 16 * (cal_bin.n / max(b.n for b in metrics.bins))
        parts.append(
            f'<rect class="jb" x="{x - 3:.2f}" y="{strip_y + (16 - bar_h):.2f}" '
            f'width="6" height="{bar_h:.2f}"/>'
        )
    parts.append(
        f'<text class="jt" x="{px(1.0):.2f}" y="{strip_y + 14:.2f}" text-anchor="end">'
        f"n={total} per bin \u2193 tallest</text>"
    )
    parts.append(
        f'<text class="jt" x="{px(0.0) - 8:.2f}" y="{py(0.5):.2f}" text-anchor="end" '
        f'transform="rotate(-90 {px(0.0) - 34:.2f} {py(0.5):.2f})">observed accuracy</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def metric_bar(label: str, value: float, ci: tuple[float, float] | None = None) -> str:
    """A one-line horizontal bar used for ECE-style magnitudes."""
    width = 220.0
    filled = 0.0 if value != value else max(0.0, min(1.0, value / 0.25)) * width
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 18" width="300" height="18">',
        '<rect x="0" y="4" width="220" height="10" fill="#eee"/>',
        f'<rect x="0" y="4" width="{filled:.2f}" height="10" fill="#111"/>',
    ]
    if ci is not None and ci[0] == ci[0]:
        low = max(0.0, min(1.0, ci[0] / 0.25)) * width
        high = max(0.0, min(1.0, ci[1] / 0.25)) * width
        parts.append(
            f'<line x1="{low:.2f}" y1="1" x2="{high:.2f}" y2="17" stroke="#777" stroke-width="2"/>'
        )
    parts.append(
        f'<text x="228" y="14" font-family="ui-monospace,Menlo,monospace" font-size="11" '
        f'fill="#111">{_fmt(value, 3)}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def histogram(
    values: Sequence[float], *, width: int = 300, height: int = 90, bins: int = 12
) -> str:
    """Plain histogram used for the score-type level distribution."""
    if not values:
        return '<p class="note">No values.</p>'
    lo = min(values)
    hi = max(values)
    span = (hi - lo) or 1.0
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, int((value - lo) / span * bins))
        counts[index] += 1
    peak = max(counts) or 1
    bar_w = width / bins
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%">'
    ]
    for index, count in enumerate(counts):
        bar_h = (count / peak) * (height - 12)
        parts.append(
            f'<rect x="{index * bar_w + 1:.2f}" y="{height - 10 - bar_h:.2f}" '
            f'width="{bar_w - 2:.2f}" height="{bar_h:.2f}" fill="#111"/>'
        )
    parts.append(
        f'<text x="0" y="{height}" font-family="ui-monospace,Menlo,monospace" font-size="10" '
        f'fill="#555">{_fmt(lo)} .. {_fmt(hi)}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)
