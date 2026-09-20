"""Segment breakdown: which slice of traffic is misfiring.

Ordered worst-first, because a segment table sorted by name is a table nobody reads. Segments
that cannot support a claim are shown greyed with the reason, not dropped silently.
"""

from __future__ import annotations

from html import escape

from jeval.report import svg as S
from jeval.report.model import HeatmapCell, SegmentBar, SegmentView

WIDTH = 660.0
ROW_HEIGHT = 30.0
BAR_X = 250.0
BAR_WIDTH = 300.0
TOP = 34.0
BOTTOM = 42.0
HEATMAP_MAX_CELLS = 20


def render_segments(view: SegmentView, *, title: str = "Segments", width: float = WIDTH) -> str:
    """Horizontal ECE bars, sorted worst-first, with sample-size honesty."""
    bars = view.bars
    if not bars:
        return _empty(width, title, "No segment had enough labeled decisions to break down.")
    height = TOP + ROW_HEIGHT * len(bars) + BOTTOM
    strongest = max((bar.ece for bar in bars if not bar.too_few_samples), default=0.0)
    scale = max(strongest, 0.02)
    desc = (
        f"Expected calibration error for {len(bars)} segments, worst first. "
        f"Bars are scaled to the largest reliable value, {S.fmt(scale, 3)}."
    )
    parts: list[str] = [S.svg_open(width, height, title=title, desc=desc, cls="chart")]
    parts.append(
        S.text(
            0.0,
            20.0,
            f"ECE by segment · bar scale 0 to {S.fmt(scale, 2)} · grey = too few samples",
            size=11.5,
            fill=S.MUTED,
        )
    )
    for index, bar in enumerate(bars):
        y = TOP + index * ROW_HEIGHT
        fraction = 0.0 if bar.too_few_samples else min(1.0, bar.ece / scale)
        width_px = max(2.0, fraction * BAR_WIDTH)
        colour = S.GRID if bar.too_few_samples else S.INK
        parts.append(
            S.text(
                0.0, y + 14.0, bar.label, size=11.5, fill=S.MUTED if bar.too_few_samples else S.INK
            )
        )
        if bar.too_few_samples:
            parts.append(S.rect(BAR_X, y + 3, width_px, 14, fill=colour))
        else:
            target = segment_target(bar)
            parts.append(
                f'<g class="seg-bar" role="button" tabindex="0" data-target="{escape(target)}" '
                f'aria-label="show the reliability curve for {escape(bar.label)}">'
                f"{S.rect(BAR_X, y + 3, width_px, 14, fill=colour)}</g>"
            )
        value_label = "n/a" if bar.too_few_samples else S.fmt(bar.ece, 2)
        parts.append(S.text(BAR_X + width_px + 8, y + 14.0, value_label, size=11.0, mono=True))
        note = "too few samples" if bar.too_few_samples else f"n={bar.n}"
        parts.append(S.text(BAR_X + BAR_WIDTH + 46, y + 14.0, note, size=10.5, fill=S.MUTED))
    parts.append(
        S.text(
            0.0,
            height - 14.0,
            "Lower is better. A greyed segment needs more labels before it means anything.",
            size=10.5,
            fill=S.MUTED,
        )
    )
    parts.append(S.svg_close())
    return "".join(parts)


def render_heatmap(
    cells: tuple[HeatmapCell, ...],
    *,
    x_values: tuple[str, ...],
    y_values: tuple[str, ...],
    x_axis: str,
    y_axis: str,
    title: str = "Segment grid",
    cell: float = 78.0,
    width: float | None = None,
) -> str:
    """ECE grid for two segment axes, or a table when the grid would be unreadable."""
    if not cells or len(cells) > HEATMAP_MAX_CELLS:
        return ""
    left = 120.0
    top = 56.0
    total_width = width or (left + cell * len(x_values) + 24.0)
    height = top + cell * len(y_values) + 40.0
    peak = max((c.ece for c in cells), default=0.02) or 0.02
    lookup = {(c.x, c.y): c for c in cells}
    desc = (
        f"Expected calibration error for {len(cells)} combinations of {x_axis} and {y_axis}. "
        f"Darker means worse calibrated. Values are printed in every cell."
    )
    parts: list[str] = [S.svg_open(total_width, height, title=title, desc=desc, cls="chart")]
    parts.append(S.text(0.0, 20.0, f"ECE · {y_axis} × {x_axis}", size=11.5, fill=S.MUTED))
    for column, xv in enumerate(x_values):
        parts.append(
            S.text(
                left + cell * column + cell / 2.0,
                top - 10,
                xv,
                anchor="middle",
                size=10.5,
                fill=S.MUTED,
            )
        )
    for row, yv in enumerate(y_values):
        parts.append(
            S.text(
                left - 10,
                top + cell * row + cell / 2.0 + 4,
                yv,
                anchor="end",
                size=10.5,
                fill=S.MUTED,
            )
        )
        for column, xv in enumerate(x_values):
            found = lookup.get((xv, yv))
            shade = 0.0 if found is None else min(1.0, found.ece / peak)
            grey = int(255 - 150 * shade)
            parts.append(
                S.rect(
                    left + cell * column,
                    top + cell * row,
                    cell - 2,
                    cell - 2,
                    fill=f"rgb({grey},{grey},{grey})",
                )
            )
            if found is not None:
                ink = "#ffffff" if shade > 0.55 else S.INK
                parts.append(
                    S.text(
                        left + cell * column + cell / 2.0,
                        top + cell * row + cell / 2.0,
                        S.fmt(found.ece, 2),
                        anchor="middle",
                        size=11.0,
                        mono=True,
                        fill=ink,
                    )
                )
                parts.append(
                    S.text(
                        left + cell * column + cell / 2.0,
                        top + cell * row + cell / 2.0 + 14,
                        f"n={found.n}",
                        anchor="middle",
                        size=9.5,
                        fill=ink,
                    )
                )
    parts.append(S.svg_close())
    return "".join(parts)


def render_segments_section(view: SegmentView, *, title: str = "Segments") -> str:
    """Bars, optional grid, and the numbers behind both."""
    if not view.bars:
        return (
            '<p class="note">No segment breakdown requested, or no segment had enough labels.</p>'
        )
    chart = render_segments(view, title=title)
    rows = [
        [
            bar.label,
            "n/a" if bar.too_few_samples else S.fmt(bar.ece),
            str(bar.n),
            "yes" if bar.too_few_samples else "",
        ]
        for bar in view.bars
    ]
    table = S.details_table(
        ("Segment", "ECE", "n", "Below sample floor"), rows, summary="Data behind these bars"
    )
    heat = ""
    if view.heatmap:
        x_values = tuple(sorted({cell.x for cell in view.heatmap}))
        y_values = tuple(sorted({cell.y for cell in view.heatmap}))
        heat = render_heatmap(
            view.heatmap,
            x_values=x_values,
            y_values=y_values,
            x_axis=view.x_axis or "x",
            y_axis=view.y_axis or "y",
        )
        if not heat:
            heat = S.details_table(
                ("x", "y", "ECE", "n"),
                [[cell.x, cell.y, S.fmt(cell.ece), str(cell.n)] for cell in view.heatmap],
                summary="Too many cells for a grid: the numbers instead",
            )
    return f"<figure>{chart}{table}</figure>{heat}"


def _empty(width: float, title: str, message: str) -> str:
    height = 110.0
    return (
        S.svg_open(width, height, title=title, desc=message, cls="chart")
        + S.text(S.MARGIN, height / 2 + 4, message, size=12, fill=S.MUTED)
        + S.svg_close()
    )


def segment_slug(key: str, value: str) -> str:
    """Stable id fragment for a segment, safe in an HTML id and a data attribute."""
    raw = f"{key}-{value}".lower()
    return "".join(character if character.isalnum() else "-" for character in raw).strip("-")


def segment_target(bar: SegmentBar) -> str:
    """Id of the server-rendered figure this bar reveals.

    The per-segment curve is rendered at build time rather than drawn in the browser: the report
    must stay readable with JavaScript off, and a precomputed figure keeps the file honest.
    """
    return f"seg-fig-{segment_slug(bar.key, bar.value)}"


def bars_from_ece(
    rows: tuple[tuple[str, str, float, int], ...],
    *,
    min_samples: int = 30,
) -> tuple[SegmentBar, ...]:
    """Build bars from ``(key, value, ece, n)`` rows, worst first, flagging small samples."""
    bars = [
        SegmentBar(key=key, value=value, ece=ece, n=n, too_few_samples=n < min_samples)
        for key, value, ece, n in rows
    ]
    return tuple(sorted(bars, key=lambda bar: (bar.too_few_samples, -bar.ece)))
