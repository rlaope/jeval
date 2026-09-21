"""Pure SVG primitives.

Every function here maps data to an SVG string and nothing else: no state, no I/O, no
knowledge of jeval's domain types. Charts are built from these, so a rendering bug is fixed
once instead of in four chart modules.

Conventions that matter for output size and accessibility:

- coordinates are rounded to two decimals (visibly smaller files, no visual difference),
- every chart carries a ``<title>`` and ``<desc>``,
- colour never carries meaning alone: the colour-blind-safe palette is paired with labels
  and positions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from html import escape as _html_escape

# Okabe-Ito: distinguishable under the common colour-vision deficiencies. Never rely on a
# single channel to say good or bad; position and label carry that.
PALETTE: tuple[str, ...] = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#8C6D31",
    "#444444",
)

INK = "#111111"
MUTED = "#666666"
GRID = "#dddddd"
DIAGONAL = "#999999"
SHADE = "#f2f2f2"

FONT_STACK = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', ui-sans-serif, Helvetica, Arial, sans-serif"
)
MONO_STACK = "ui-monospace, SFMono-Regular, Menlo, monospace"

MARGIN = 58.0  # left gutter used by every chart's y axis


def r2(value: float) -> str:
    """Format a coordinate to two decimals; the file-size saving is real, the loss is not."""
    return f"{value:.2f}"


def fmt(value: float, digits: int = 2) -> str:
    """Human-readable number that never prints ``nan`` as a number."""
    if value != value:  # NaN
        return "n/a"
    return f"{value:.{digits}f}"


# Bidirectional and invisible controls let a string render as something other than what it is:
# a label or model name carrying U+202E can reverse the text a reader sees. They are replaced with
# a visible marker rather than dropped, so the reader can see that the value contained one.
_BIDI_CONTROLS = {
    "\u202a",
    "\u202b",
    "\u202c",
    "\u202d",
    "\u202e",
    "\u2066",
    "\u2067",
    "\u2068",
    "\u2069",
    "\u200e",
    "\u200f",
    "\u061c",
}


def escape(value: object) -> str:
    """Escape for HTML, with invisible direction controls made visible.

    Every module in the report escapes through here, so a value cannot spoof the text around it in
    one sink while being neutralised in another.
    """
    text = "" if value is None else str(value)
    for control in _BIDI_CONTROLS:
        if control in text:
            text = text.replace(control, f"<U+{ord(control):04X}>")
    return _html_escape(text)


def pct(value: float, digits: int = 1) -> str:
    if value != value:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def money(value: float) -> str:
    """Compact currency for labels: 1.2M, 45.3k, 890."""
    magnitude = abs(value)
    if magnitude >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if magnitude >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:,.0f}"


def nice_ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    """Tick positions on the 1 / 2 / 5 x 10^n rule, so labels stay round numbers."""
    if hi <= lo:
        return [lo]
    step = (hi - lo) / max(1, target)
    exponent = 0
    while step >= 10:
        step /= 10
        exponent += 1
    while step < 1:
        step *= 10
        exponent -= 1
    for candidate in (1.0, 2.0, 5.0, 10.0):
        if step <= candidate:
            step = candidate
            break
    step *= 10**exponent
    if step <= 0:
        return [lo, hi]
    ticks: list[float] = []
    value = (lo // step) * step
    if value < lo:
        value += step
    while value <= hi + step * 1e-9:
        ticks.append(round(value, 10))
        value += step
    return ticks or [lo, hi]


@dataclass(frozen=True)
class Scale:
    """A linear mapping from a data domain onto a pixel range."""

    domain: tuple[float, float]
    range: tuple[float, float]

    def __call__(self, value: float) -> float:
        lo, hi = self.domain
        r_lo, r_hi = self.range
        if hi == lo:
            return (r_lo + r_hi) / 2.0
        return r_lo + (value - lo) / (hi - lo) * (r_hi - r_lo)

    def invert(self, pixel: float) -> float:
        lo, hi = self.domain
        r_lo, r_hi = self.range
        if r_hi == r_lo:
            return lo
        return lo + (pixel - r_lo) / (r_hi - r_lo) * (hi - lo)


def lin_scale(domain: tuple[float, float], range_: tuple[float, float]) -> Scale:
    return Scale(domain=domain, range=range_)


def svg_open(width: float, height: float, *, title: str, desc: str, cls: str = "") -> str:
    """Open a responsive SVG with an accessible name and description."""
    class_attr = f' class="{escape(cls)}"' if cls else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {r2(width)} {r2(height)}"'
        f'{class_attr} role="img" aria-label="{escape(title)}" preserveAspectRatio="xMidYMid meet">'
        f"<title>{escape(title)}</title><desc>{escape(desc)}</desc>"
    )


def svg_close() -> str:
    return "</svg>"


def rect(x: float, y: float, w: float, h: float, *, fill: str = "none", cls: str = "") -> str:
    class_attr = f' class="{cls}"' if cls else ""
    return (
        f'<rect x="{r2(x)}" y="{r2(y)}" width="{r2(max(0.0, w))}" '
        f'height="{r2(max(0.0, h))}"{class_attr} fill="{fill}"/>'
    )


def vband(
    x: float,
    *,
    width: float,
    y0: float,
    y1: float,
    fill: str = SHADE,
    opacity: float = 0.75,
) -> str:
    """Vertical shaded region, used for expected-overconfidence zones and flat cost regions."""
    return rect(x, y0, width, y1 - y0, fill=fill, cls="") + ""
    # opacity handled by the caller's palette choice; kept simple on purpose


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = INK,
    width: float = 1.0,
    dash: str = "",
    opacity: float = 1.0,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    opacity_attr = "" if opacity >= 1.0 else f' stroke-opacity="{opacity:.2f}"'
    return (
        f'<line x1="{r2(x1)}" y1="{r2(y1)}" x2="{r2(x2)}" y2="{r2(y2)}" '
        f'stroke="{stroke}" stroke-width="{width:.2f}"{dash_attr}{opacity_attr}/>'
    )


def polyline(
    points: Sequence[tuple[float, float]],
    *,
    stroke: str = INK,
    width: float = 1.8,
    dash: str = "",
    opacity: float = 1.0,
    smooth: bool = False,
) -> str:
    if not points:
        return ""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    opacity_attr = "" if opacity >= 1.0 else f' stroke-opacity="{opacity:.2f}"'
    if smooth and len(points) > 2:
        d = _smooth_path(points)
    else:
        d = " ".join(f"{'M' if i == 0 else 'L'}{r2(x)},{r2(y)}" for i, (x, y) in enumerate(points))
    return (
        f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{width:.2f}"'
        f"{dash_attr}{opacity_attr}/>"
    )


def _smooth_path(points: Sequence[tuple[float, float]]) -> str:
    """Horizontal-tangent smoothing, used only for presentation curves."""
    d = f"M{r2(points[0][0])},{r2(points[0][1])}"
    for index in range(len(points) - 1):
        x0, y0 = points[index]
        x1, y1 = points[index + 1]
        cx = (x0 + x1) / 2.0
        d += f"C{r2(cx)},{r2(y0)} {r2(cx)},{r2(y1)} {r2(x1)},{r2(y1)}"
    return d


def dot(
    x: float,
    y: float,
    radius: float,
    *,
    fill: str = INK,
    tooltip: str = "",
    cls: str = "",
    opacity: float = 1.0,
    extra: str = "",
) -> str:
    class_attr = f' class="{cls}"' if cls else ""
    opacity_attr = "" if opacity >= 1.0 else f' fill-opacity="{opacity:.2f}"'
    inner = f"<title>{escape(tooltip)}</title>" if tooltip else ""
    return (
        f'<circle cx="{r2(x)}" cy="{r2(y)}" r="{r2(radius)}" fill="{fill}"'
        f"{opacity_attr}{class_attr}{extra}>{inner}</circle>"
    )


def error_bar(
    x: float,
    low: float,
    high: float,
    y_of: Callable[[float], float],
    *,
    stroke: str = MUTED,
    width: float = 1.3,
    cap: float = 3.0,
) -> str:
    """Vertical interval with caps; the visible alternative to pretending a bin is precise."""
    if low != low or high != high:
        return ""
    top = y_of(high)
    bottom = y_of(low)
    return "".join(
        [
            line(x, top, x, bottom, stroke=stroke, width=width),
            line(x - cap, top, x + cap, top, stroke=stroke, width=width),
            line(x - cap, bottom, x + cap, bottom, stroke=stroke, width=width),
        ]
    )


def shaded_region(
    x0: float,
    x1: float,
    *,
    y0: float,
    y1: float,
    fill: str = SHADE,
    cls: str = "shade",
    label: str = "",
    label_x: float | None = None,
    label_y: float | None = None,
) -> str:
    # `fill` stays as the fallback for a viewer that ignores CSS; the class themes it.
    body = rect(x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0), fill=fill, cls=cls)
    if not label:
        return body
    lx = label_x if label_x is not None else (x0 + x1) / 2.0
    ly = label_y if label_y is not None else (y0 + y1) / 2.0
    return body + text(lx, ly, label, anchor="middle", size=10.5, fill=MUTED)


def text(
    x: float,
    y: float,
    value: str,
    *,
    anchor: str = "start",
    size: float = 11.5,
    fill: str = INK,
    weight: int = 400,
    mono: bool = False,
    rotate: float | None = None,
    rotate_at: tuple[float, float] | None = None,
    opacity: float = 1.0,
    cls: str = "",
) -> str:
    family = MONO_STACK if mono else FONT_STACK
    transform = ""
    if rotate is not None:
        cx, cy = rotate_at if rotate_at else (x, y)
        transform = f' transform="rotate({rotate:.1f} {r2(cx)} {r2(cy)})"'
    opacity_attr = "" if opacity >= 1.0 else f' fill-opacity="{opacity:.2f}"'
    class_attr = f' class="{cls}"' if cls else ""
    return (
        f'<text x="{r2(x)}" y="{r2(y)}" text-anchor="{anchor}" font-family="{family}" '
        f'font-size="{size:.1f}" font-weight="{weight}" fill="{fill}"{opacity_attr}{class_attr}'
        f"{transform}>{escape(value)}</text>"
    )


def axis_x(
    scale: Scale,
    *,
    y: float,
    tick_values: Sequence[float] | None = None,
    format_: Callable[[float], str] = lambda v: fmt(v),
    title: str = "",
) -> str:
    values = list(tick_values) if tick_values is not None else nice_ticks(*scale.domain)
    parts = [line(scale.range[0], y, scale.range[1], y, stroke=INK, width=1.0)]
    for value in values:
        x = scale(value)
        parts.append(line(x, y, x, y + 4, stroke=INK, width=1.0))
        parts.append(text(x, y + 16, format_(value), anchor="middle", size=11, fill=MUTED))
    if title:
        parts.append(
            text(
                (scale.range[0] + scale.range[1]) / 2.0,
                y + 34,
                title,
                anchor="middle",
                size=11.5,
                weight=500,
            )
        )
    return "".join(parts)


def axis_y(
    scale: Scale,
    *,
    x: float,
    tick_values: Sequence[float] | None = None,
    format_: Callable[[float], str] = lambda v: fmt(v),
    title: str = "",
) -> str:
    values = list(tick_values) if tick_values is not None else nice_ticks(*scale.domain)
    parts = [line(x, scale.range[0], x, scale.range[1], stroke=INK, width=1.0)]
    for value in values:
        y = scale(value)
        parts.append(line(x - 4, y, x, y, stroke=INK, width=1.0))
        parts.append(text(x - 8, y + 4, format_(value), anchor="end", size=11, fill=MUTED))
    if title:
        centre = (scale.range[0] + scale.range[1]) / 2.0
        parts.append(
            text(
                x - 38,
                centre,
                title,
                anchor="middle",
                size=11.5,
                weight=500,
                rotate=-90,
                rotate_at=(x - 38, centre),
            )
        )
    return "".join(parts)


def grid_x(scale: Scale, tick_values: Sequence[float], *, y0: float, y1: float) -> str:
    return "".join(line(scale(v), y0, scale(v), y1, stroke=GRID, width=1.0) for v in tick_values)


def grid_y(scale: Scale, tick_values: Sequence[float], *, x0: float, x1: float) -> str:
    return "".join(line(x0, scale(v), x1, scale(v), stroke=GRID, width=1.0) for v in tick_values)


def legend(entries: Sequence[tuple[str, str]], *, x: float, y: float, size: float = 11.0) -> str:
    """Legend as swatch + label pairs; never colour alone."""
    parts: list[str] = []
    cursor = x
    for label, colour in entries:
        parts.append(rect(cursor, y - 9, 10, 10, fill=colour))
        parts.append(text(cursor + 15, y - 1, label, size=size, fill=MUTED))
        cursor += 15 + len(label) * size * 0.56 + 18
    return "".join(parts)


def marker_line(
    x: float,
    *,
    y0: float,
    y1: float,
    label: str,
    colour: str = INK,
    dash: str = "4 3",
    label_y: float | None = None,
) -> str:
    """A vertical rule with a label, used for the current threshold and model changes."""
    ty = label_y if label_y is not None else y0 + 11
    return line(x, y0, x, y1, stroke=colour, width=1.4, dash=dash) + text(
        x + 4, ty, label, size=10.5, fill=colour, weight=600
    )


def details_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[str]],
    *,
    summary: str = "Data behind this chart",
    numeric_from: int = 1,
) -> str:
    """Collapsible table of the underlying numbers.

    Accessibility and honesty in one control: a screen reader can read the table, and anyone
    quoting a figure can copy it instead of eyeballing a pixel.
    """
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="{"num" if i >= numeric_from else ""}">{escape(str(cell))}</td>'
            for i, cell in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    return (
        f"<details><summary>{escape(summary)}</summary>"
        f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></details>"
    )


def embed_json(payload: str, *, element_id: str) -> str:
    """Embed aggregates as data, never as executable code.

    The ``application/json`` type means the browser will not execute it, and the payload holds
    aggregated values only — raw decision records never enter the report.
    """
    # Escaping every `<` (not just `</`) also closes the HTML script-data double-escape hole: a
    # `<!--<script` inside a value would otherwise leave the comment open and swallow the report's
    # own script, killing every interactive control.
    safe = payload.replace("<", "\\u003c")
    return f'<script type="application/json" id="{escape(element_id)}">{safe}</script>'


def mapping_legend(items: Mapping[str, str], *, x: float, y: float) -> str:
    return legend([(key, value) for key, value in items.items()], x=x, y=y)
