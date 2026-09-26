"""Inline CSS and JavaScript for the single-file report.

One HTML file, no sibling assets: the stylesheet and the behaviour are module-level strings
that the template inlines. Nothing here reads the filesystem, and nothing it emits reaches the
network -- no ``@import``, no ``url()``, no web font, no CDN script.

All colour lives in custom properties, so a single ``@media (prefers-color-scheme: dark)``
block restyles the whole report, and the ``@media print`` block prints a three-page A4 summary
that says it is one. The one thing custom properties cannot reach is a chart's own presentation
attributes, so the dark block also re-maps the chart layer's greys from the same constants
that layer paints with.

What the template has to emit. Everything is optional: a report without tabs, sliders, segment
charts or a copy button is still a readable static document.

``<script type="application/json" id="jeval-data">``
    Aggregates only; raw decision records never enter the report. Keys:

    ``summary_markdown`` (str)
        Exactly the string ``jeval.report.verdict.markdown_summary`` returns. The copy button
        writes it to the clipboard, so it must stay stable.
    ``active_question`` (str, optional)
        Question key open on load; defaults to the first tab.
    ``actions`` (object, optional)
        Keyed by action name. Each entry:

        ``threshold`` (float) -- recommended threshold, the slider's start value,
        ``monthly_volume`` (float, optional) -- decisions per month, for the monthly figures,
        ``curve`` (array, optional) -- one point per sweep step; the slider snaps to the
        nearest point, matching ``ThresholdResult.point_at``.

    A curve point is ``[threshold, cost_per_case, auto_rate, accuracy_auto]``, or an object using
    those names. ``jeval.report.template`` emits ``{"t": ..., "cost": ..., "auto": ..., "acc"}``,
    and the short names are read too.

Elements the script looks for, all optional:

``[data-jeval-tab="<question>"]``
    Tab button; ``aria-selected`` and ``tabindex`` are maintained.
``[data-jeval-question="<question>"]``
    Panel -- a reliability figure or a data-quality block -- shown only for the active tab.
``[data-jeval-action="<action>"]``
    Slider box. Contains ``input[type="range"]``, ``[data-jeval-threshold]`` for the slider's
    own label, one ``[data-jeval-value="<key>"]`` per displayed figure, and optionally a
    ``data-currency`` attribute (e.g. ``KRW``) with its ``data-currency-digits`` (``0`` for KRW, from
    :func:`jeval.currency.minor_units`), and an ``input[type="number"]`` for the monthly
    volume. Keys: ``auto_rate``, ``accuracy_auto``, ``measured_accuracy``, ``cost_per_case``,
    ``cost_per_month``, ``auto_per_month``, ``escalations_per_month``.
``[data-jeval-copy-summary]``
    Copy-summary button; copies ``summary_markdown``. ``[data-copy-target]`` is accepted as an
    alias because the template already ships it.
``[data-jeval-segment="<key>"]`` / ``[data-jeval-segment-chart="<key>"]``
    Clickable segment bar and the per-segment chart it toggles into view.
    ``[data-target="<element id>"]`` is accepted as an alias: the bar toggles the element whose
    ``id`` it names.
"""

from __future__ import annotations

from jeval.report import svg as chart_svg

__all__ = ["REPORT_CSS", "REPORT_JS", "minify"]

# --------------------------------------------------------------------------------------
# Colour
# --------------------------------------------------------------------------------------
# Charts paint literal greys: an SVG presentation attribute cannot read a custom property, and a
# report whose curves go invisible on a reader's dark-mode laptop is a report that lied. Rather
# than keep two palettes in step by hand, the dark block's grey re-map is generated here from the
# chart layer's own constants -- and it re-maps greys only. The Okabe-Ito hues that carry meaning
# are never touched.
_GREY_RULES: tuple[tuple[str, str, str, str], ...] = (
    ('text[fill="{v}"]', "fill", "ink", chart_svg.INK),
    ('circle[fill="{v}"]', "fill", "ink", chart_svg.INK),
    ('polygon[fill="{v}"]', "fill", "ink", chart_svg.INK),
    ('line[stroke="{v}"]', "stroke", "ink", chart_svg.INK),
    ('path[stroke="{v}"]', "stroke", "ink", chart_svg.INK),
    ('text[fill="{v}"]', "fill", "muted", chart_svg.MUTED),
    ('line[stroke="{v}"]', "stroke", "muted", chart_svg.SOFT),
    ('path[stroke="{v}"]', "stroke", "muted", chart_svg.SOFT),
    ('circle[fill="{v}"]', "fill", "muted", chart_svg.SOFT),
    ('line[stroke="{v}"]', "stroke", "rule", chart_svg.GRID),
    ('line[stroke="{v}"]', "stroke", "muted", chart_svg.DIAGONAL),
    ('rect[fill="{v}"]', "fill", "rule", chart_svg.GRID),
    ('rect[fill="{v}"]', "fill", "shade", chart_svg.SHADE),
    ('polygon[fill="{v}"]', "fill", "shade", chart_svg.SHADE),
    ('rect[fill="{v}"]', "fill", "shade", chart_svg.DIAGONAL),
)

_GREY_REMAP = "".join(
    "  figure svg " + selector.format(v=value) + " { " + prop + ": var(--" + token + "); }\n"
    for selector, prop, token, value in _GREY_RULES
)

# The placeholder lives in the dark-mode block; replacing it keeps the stylesheet readable as one
# document while the values it swaps in come from the chart layer. Resolved below, once the
# stylesheet itself is defined.

_CSS = """\
:root {
  color-scheme: light dark;
  /* Surfaces: warm paper, one step of lift for panels, one step of recess for tracks. */
  --bg: #f7f6f2;
  --panel: #ffffff;
  --panel-alt: #f0eee8;
  --shade: #f2efe9;
  /* Ink: three steps, all at or above 4.5:1 on --bg. */
  --ink: #1b1a17;
  --ink-2: #3f3d38;
  --muted: #6f6b63;
  --rule: #e4e0d8;
  --rule-strong: #cfcac0;
  /* The two semantic hues. Text in either wears the darker -ink step so it clears 4.5:1. */
  --accent: #1f5fbf;
  --accent-ink: #1a4f9e;
  --accent-soft: #e9f0fa;
  --alert: #d1491f;
  --alert-ink: #ad3a14;
  --alert-soft: #fbece5;
  --good: #1d7a4b;
  --warn: #a06000;
  --bad: #b3261e;
  --shadow: 0 1px 0 rgba(27, 26, 23, 0.04), 0 1px 3px rgba(27, 26, 23, 0.06);
  --radius: 6px;
  /* No web font: the report never fetches anything. Each stack names the best face a reader is
     likely to have, then falls through to the platform's own. */
  --font: Inter, "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI Variable Text",
    "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --serif: "Iowan Old Style", Charter, "Bitstream Charter", "Sitka Heading", "Source Serif Pro",
    Cambria, Georgia, serif;
  --mono: "JetBrains Mono", "SF Mono", SFMono-Regular, ui-monospace, "Cascadia Mono",
    "Roboto Mono", Menlo, Consolas, monospace;
  /* Type scale: 12 / 13 / 14 / 15 (body) / 17 / 22 / 26 / 34. */
  --fs-xs: 12px;
  --fs-sm: 13px;
  --fs-md: 14px;
  --fs-body: 15px;
  --fs-lg: 17px;
  --fs-h2: 22px;
  --fs-head: 26px;
  --fs-h1: 34px;
}
/* A shaded band in a chart is a tint, not a colour of its own: as a variable it follows the
   theme instead of staying near-white in a dark report. */
.shade { fill: var(--shade); }
* { box-sizing: border-box; }
[hidden] { display: none !important; }
.print-only { display: none; }
html { -webkit-text-size-adjust: 100%; text-size-adjust: 100%; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: var(--fs-body)/1.65 var(--font);
  font-feature-settings: "cv11", "ss01";
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
/* One column. 960px holds a 660px chart beside its key figures, and keeps prose inside 72ch. */
main { max-width: 1008px; margin: 0 auto; padding: 48px 24px 112px; }
h1, h2, .verdict .headline { font-family: var(--serif); font-weight: 600; color: var(--ink); }
h1 { font-size: var(--fs-h1); line-height: 1.12; letter-spacing: -0.015em; margin: 0 0 12px; }
h2 {
  display: flex; align-items: baseline; gap: 14px;
  font-size: var(--fs-h2); line-height: 1.25; letter-spacing: -0.01em;
  margin: 72px 0 8px; padding-top: 20px; border-top: 1px solid var(--rule-strong);
}
h2 .idx {
  font-family: var(--mono); font-size: var(--fs-xs); font-weight: 500; letter-spacing: 0.04em;
  color: var(--muted); min-width: 22px;
}
h3 { font-size: var(--fs-lg); line-height: 1.4; font-weight: 600; margin: 40px 0 6px; letter-spacing: -0.005em; }
h4 { font-size: var(--fs-md); font-weight: 600; margin: 24px 0 6px; }
p { margin: 0 0 14px; }
ul { margin: 0 0 14px; padding-left: 20px; }
li { margin: 0 0 6px; }
li::marker { color: var(--muted); }
a { color: var(--accent-ink); text-decoration-thickness: 1px; text-underline-offset: 3px; }
section { margin: 0; }
code, kbd, samp {
  font-family: var(--mono); font-size: 0.86em;
  background: var(--panel-alt); border-radius: 3px; padding: 1px 5px;
}
strong { font-weight: 600; }
.num { font-family: var(--font); font-variant-numeric: tabular-nums lining-nums; white-space: nowrap; }
.note { color: var(--muted); font-size: var(--fs-sm); line-height: 1.6; max-width: 80ch; }
.intro { color: var(--ink-2); font-size: var(--fs-body); max-width: 70ch; margin-bottom: 20px; }
.ident { font-family: var(--mono); font-size: 0.9em; }
/* The template emits sections in SECTION_ORDER, so the verdict is already first in the
   document; the rule below keeps it first if a container is ever flexed or reordered. */
#verdict { order: -1; }

/* Masthead: the name, what the page is for, and where its numbers came from. */
.masthead { margin: 0 0 28px; }
.brand {
  display: flex; align-items: center; gap: 10px; margin: 0 0 22px;
  font-size: var(--fs-xs); letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted);
}
.brand .mark {
  font-family: var(--mono); font-weight: 700; letter-spacing: 0; text-transform: none;
  font-size: var(--fs-sm); color: var(--panel); background: var(--ink);
  border-radius: 3px; padding: 2px 7px;
}
.lede { font-size: var(--fs-lg); line-height: 1.55; color: var(--ink-2); max-width: 60ch; margin: 0 0 24px; }
/* Provenance as a definition list: one item per row, label left, value right. A single run-on
   line is read as boilerplate and skipped, which is exactly the part a reader needs. */
.provenance {
  display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 6px 20px;
  margin: 0; padding: 14px 0 0; border-top: 1px solid var(--rule);
  font-size: var(--fs-sm); line-height: 1.55;
}
.provenance dt { color: var(--muted); }
.provenance dd { margin: 0; color: var(--ink-2); overflow-wrap: anywhere; max-width: 88ch; }
.provenance dd.path { font-family: var(--mono); font-size: var(--fs-xs); padding-top: 1px; }

/* Section index: the reading order, as links. */
.toc {
  display: flex; flex-wrap: wrap; gap: 4px 18px; margin: 0 0 32px; padding: 10px 0;
  border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule);
  font-size: var(--fs-sm);
}
.toc a { color: var(--muted); text-decoration: none; }
.toc a:hover { color: var(--ink); }
.toc .idx { font-family: var(--mono); font-size: var(--fs-xs); margin-right: 6px; opacity: 0.6; }

/* The verdict. A status line with a word and a mark, the headline, what it rests on, then the
   line in use and the recommended line drawn on one ruler, and the three figures. */
.verdict {
  background: var(--panel); border: 1px solid var(--rule); border-radius: var(--radius);
  box-shadow: var(--shadow); padding: 26px 30px 24px; position: relative; overflow: hidden;
}
.verdict::before {
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--accent);
}
#verdict[data-status="too_low"].verdict::before, #verdict[data-status="too_high"].verdict::before { background: var(--alert); }
#verdict[data-status="insufficient_data"].verdict::before { background: var(--muted); }
.status {
  display: inline-flex; align-items: center; gap: 8px; margin: 0 0 12px;
  font-size: var(--fs-xs); font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--accent-ink);
}
.status::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
#verdict[data-status="too_low"] .status, #verdict[data-status="too_high"] .status { color: var(--alert-ink); }
#verdict[data-status="insufficient_data"] .status { color: var(--muted); }
.verdict .headline { font-size: var(--fs-head); line-height: 1.22; letter-spacing: -0.012em; margin: 0 0 10px; }
.verdict .detail { margin: 0; font-size: var(--fs-body); line-height: 1.65; color: var(--ink-2); max-width: 70ch; }
.ruler { margin: 22px 0 0; }
.ruler svg { display: block; max-width: 100%; height: auto; }
.ruler .narrow { display: none; }
.stats {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 18px 32px; margin: 22px 0 0; padding-top: 20px; border-top: 1px solid var(--rule);
}
.stat .k { display: flex; align-items: center; gap: 8px; font-size: var(--fs-xs); color: var(--muted); letter-spacing: 0.02em; }
.stat .v {
  font-size: 30px; line-height: 1.15; font-weight: 600; letter-spacing: -0.02em; margin-top: 6px;
  font-variant-numeric: tabular-nums lining-nums;
}
.stat .s { font-size: var(--fs-sm); line-height: 1.5; color: var(--muted); margin-top: 3px; }
/* A key swatch that matches the ruler's mark, so the figure and the line are one thing. */
.swatch { display: inline-block; width: 14px; height: 0; border-top: 2px solid var(--muted); }
.swatch.current { border-top: 2px dashed var(--alert); }
.swatch.recommended { border-top-color: var(--accent); }
.stats.small { gap: 14px 28px; margin-top: 12px; }
.stats.small .stat .v { font-size: 22px; }
.verdict-actions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 22px; }

/* Tabs as one segmented control: a row of loose labels reads as text, not as a control. */
.tabs {
  display: inline-flex; flex-wrap: wrap; gap: 2px; padding: 3px;
  margin: 8px 0 4px; border: 1px solid var(--rule); border-radius: 8px;
  background: var(--panel-alt);
}
.tab {
  -webkit-appearance: none; appearance: none; cursor: pointer;
  border: 0; background: none; color: var(--muted);
  font: inherit; font-family: var(--mono); font-size: var(--fs-sm); padding: 6px 14px; border-radius: 6px;
}
.tab:hover { color: var(--ink); }
.tab[aria-selected="true"], .tab.is-active {
  color: var(--ink); background: var(--panel); font-weight: 600; box-shadow: var(--shadow);
}
.tab:focus-visible, button:focus-visible, input:focus-visible, a:focus-visible,
[data-jeval-segment]:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.block-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 14px; margin: 28px 0 6px; }
.block-head h3 { margin: 0; font-family: var(--mono); font-size: var(--fs-lg); font-weight: 600; }
.block-sub { margin: 0; font-size: var(--fs-sm); font-weight: 400; color: var(--muted); }
.action-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 12px; }
.action-head .ident { font-size: var(--fs-lg); font-weight: 600; }
/* Per-class calibration sits under the reliability data table, as its own short argument. */
.classwise { margin-top: 22px; }

/* Callouts. The left edge carries the kind, the tint only separates them from the page. */
.diag {
  border-left: 3px solid var(--accent); border-radius: 0 var(--radius) var(--radius) 0;
  background: var(--accent-soft); padding: 12px 16px; margin: 10px 0 20px;
  font-size: var(--fs-md); line-height: 1.6; color: var(--ink); max-width: 80ch;
}
.warn {
  border-left: 3px solid var(--alert); border-radius: 0 var(--radius) var(--radius) 0;
  background: var(--alert-soft); padding: 12px 16px; margin: 16px 0 20px;
  font-size: var(--fs-md); line-height: 1.6; color: var(--ink);
}
.warn ul { margin: 8px 0 0; }
.good { color: var(--good); }
.bad { color: var(--bad); }

figure { margin: 18px 0 28px; }
figure svg { display: block; max-width: 100%; height: auto; }
figcaption { margin-top: 10px; color: var(--muted); font-size: var(--fs-sm); line-height: 1.6; max-width: 76ch; }
svg text { font-variant-numeric: tabular-nums; }
/* A chart beside the figures that summarise it; stacks on a narrow screen. */
.chart-row { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 20px 36px; }
.chart-row > svg { flex: 0 1 auto; }
.keyfigs {
  flex: 1 1 200px; margin: 34px 0 0; display: grid; gap: 16px;
  border-left: 1px solid var(--rule); padding-left: 24px;
}
.keyfigs div { display: flex; flex-direction: column; gap: 2px; }
.keyfigs dt { font-size: var(--fs-xs); color: var(--muted); letter-spacing: 0.02em; }
.keyfigs dd { margin: 0; font-size: 20px; font-weight: 600; letter-spacing: -0.01em; font-variant-numeric: tabular-nums; }
.keyfigs dd small { display: block; font-size: var(--fs-xs); font-weight: 400; color: var(--muted); letter-spacing: 0; }

/* Tables. Text reads left, numbers right in tabular figures, a heavier rule under the header and
   hairlines between rows; long prose in a cell wraps at a measure instead of stretching a row. */
table { border-collapse: collapse; width: 100%; margin: 14px 0 22px; font-size: var(--fs-md); line-height: 1.5; }
caption { caption-side: top; text-align: left; color: var(--muted); font-size: var(--fs-sm); padding: 0 0 8px; }
th, td { padding: 10px 14px; border-bottom: 1px solid var(--rule); text-align: left; vertical-align: top; }
th:first-child, td:first-child { padding-left: 0; }
th:last-child, td:last-child { padding-right: 0; }
thead th {
  color: var(--muted); font-weight: 600; font-size: var(--fs-xs); letter-spacing: 0.02em;
  border-bottom: 1px solid var(--rule-strong); padding-top: 6px; padding-bottom: 8px; white-space: nowrap;
}
th.num, td.num { text-align: right; }
td.num.l { text-align: left; }
td.ident { font-family: var(--mono); font-size: var(--fs-sm); white-space: nowrap; }
table.kv { max-width: 620px; }
table.kv td:first-child { color: var(--ink-2); }
td.prose { color: var(--ink-2); font-size: var(--fs-sm); min-width: 22ch; max-width: 46ch; }
td.label { font-weight: 500; }
tbody tr:hover td { background: color-mix(in srgb, var(--panel-alt) 55%, transparent); }
tbody tr:last-child td { border-bottom: 0; }
.tag {
  display: inline-block; font-size: var(--fs-xs); font-weight: 600; line-height: 1.5;
  padding: 0 7px; border-radius: 3px; background: var(--panel-alt); color: var(--ink-2); white-space: nowrap;
}
.tag.yes { background: var(--accent-soft); color: var(--accent-ink); }

/* The impact table: the proposal this document argues for is the recommended column, so it is
   the one with a ground and a heavier figure; the change column says which way things moved. */
table.impact { margin-top: 18px; }
table.impact th:nth-child(3), table.impact td:nth-child(3) {
  background: var(--accent-soft); padding-left: 16px; padding-right: 16px;
}
table.impact thead th:nth-child(3) { color: var(--accent-ink); box-shadow: inset 0 2px 0 var(--accent); }
table.impact td:nth-child(3) { font-weight: 600; }
table.impact td:first-child { color: var(--ink-2); }
table.impact tr.impact-monthly td { border-top: 1px solid var(--rule-strong); }
.delta { display: inline-flex; align-items: baseline; gap: 5px; }
.delta::before { font-size: 0.72em; color: var(--muted); }
.delta.up::before { content: "\\25B2"; }
.delta.down::before { content: "\\25BC"; }
.delta.better { color: var(--good); }
.delta.worse { color: var(--bad); }
.delta.better::before, .delta.worse::before { color: inherit; }

/* Collapsible data tables: closed by default, obviously openable, and print when open. */
details { border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule); margin: 16px 0 22px; }
details + details { margin-top: -23px; }
details > summary {
  cursor: pointer; list-style: none; padding: 10px 0; font-size: var(--fs-sm); color: var(--muted);
  display: flex; align-items: center; gap: 8px;
}
details > summary:hover { color: var(--ink); }
details > summary::-webkit-details-marker { display: none; }
details > summary::before {
  content: ""; width: 6px; height: 6px; border-right: 1.5px solid currentColor; border-bottom: 1.5px solid currentColor;
  transform: rotate(-45deg); transition: transform 0.15s ease; margin: 0 4px 0 2px;
}
details[open] > summary::before { transform: rotate(45deg); }
details[open] > summary { color: var(--ink); }
details table { margin: 0 0 12px; font-size: var(--fs-sm); }
.table-wrap { overflow-x: auto; }

button, .button {
  -webkit-appearance: none; appearance: none; cursor: pointer;
  border: 1px solid var(--rule-strong); border-radius: 6px; background: var(--panel);
  color: var(--ink); font: inherit; font-size: var(--fs-sm); font-weight: 500; padding: 7px 14px;
}
button:hover, .button:hover { border-color: var(--ink); }
.copy-summary { white-space: nowrap; }

/* The threshold explorer. Everything the slider rewrites sits inside this panel, so a reader can
   see at a glance which numbers are live and which ones are the report's own measurement. */
.controls, .actions { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin: 12px 0; }
.slider {
  border: 1px solid var(--rule); border-radius: var(--radius); background: var(--panel);
  padding: 18px 22px 16px; margin: 18px 0 24px;
}
.slider-head { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 10px 16px; }
.slider-head label { font-size: var(--fs-sm); font-weight: 600; color: var(--ink); }
.slider [data-jeval-threshold], [data-jeval-threshold] {
  font-family: var(--font); font-size: 24px; font-weight: 600; letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums; color: var(--accent-ink);
}
.volume { display: inline-flex; align-items: center; gap: 8px; font-size: var(--fs-sm); color: var(--muted); margin-top: 10px; }
input[type="number"] {
  font: inherit; font-size: var(--fs-sm); font-variant-numeric: tabular-nums; width: 110px;
  padding: 5px 8px; border: 1px solid var(--rule-strong); border-radius: 5px;
  background: var(--bg); color: var(--ink);
}
input[type="range"] {
  -webkit-appearance: none; appearance: none; display: block;
  width: 100%; height: 24px; margin: 10px 0 0; background: transparent; cursor: ew-resize;
}
input[type="range"]::-webkit-slider-runnable-track {
  height: 4px; border-radius: 999px; background: var(--rule-strong);
}
input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none; appearance: none; width: 18px; height: 18px;
  margin-top: -7px; border: 3px solid var(--panel); border-radius: 50%;
  background: var(--accent); box-shadow: 0 0 0 1px var(--accent);
}
input[type="range"]::-moz-range-track { height: 4px; border-radius: 999px; background: var(--rule-strong); }
input[type="range"]::-moz-range-thumb {
  width: 14px; height: 14px; border: 3px solid var(--panel); border-radius: 50%; background: var(--accent);
  box-shadow: 0 0 0 1px var(--accent);
}
.scale { display: flex; justify-content: space-between; font-size: var(--fs-xs); color: var(--muted); font-variant-numeric: tabular-nums; }
.figures {
  display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px 28px; margin: 18px 0 6px; padding-top: 16px; border-top: 1px solid var(--rule);
}
.figures .figure { display: flex; flex-direction: column; gap: 2px; }
.figures .k { color: var(--muted); font-size: var(--fs-xs); }
.figures .v { font-size: var(--fs-lg); font-weight: 600; }

/* Segment bars: the track behind them is what makes two lengths comparable. */
.seg-bar { cursor: pointer; }
.seg-bar:hover .seg-fill, .seg-bar.is-open .seg-fill { opacity: 0.8; }
.seg-fill { fill: var(--ink-2); }
.accent-stroke { stroke: var(--accent); }
.accent-fill { fill: var(--accent); }
.accent-ink { fill: var(--accent-ink); }
.accent-dot { fill: var(--accent); }
.accent-band { fill: var(--accent); fill-opacity: 0.13; }
/* The line in use, and a model change: the same warning in two charts, one value per theme. */
.alert-stroke { stroke: var(--alert); }
.alert-ink { fill: var(--alert-ink); }
.alert-dot { fill: var(--alert); }
/* An annotation drawn on top of a gridline needs its own ground; a box would be louder. */
.halo { paint-order: stroke; stroke: var(--panel); stroke-width: 4px; stroke-linejoin: round; }
[data-jeval-segment] { cursor: pointer; }
[data-jeval-segment][aria-expanded="true"] { font-weight: 600; }
[data-jeval-segment-chart] { margin-top: 12px; }

/* Charts sit on the panel colour, so the halo, the page and the plot agree. */
.plate {
  background: var(--panel); border: 1px solid var(--rule); border-radius: var(--radius);
  padding: 20px 22px 16px; margin: 16px 0 24px;
}
.plate > figure { margin: 0; }

.limits li { color: var(--ink-2); }
footer {
  margin-top: 80px; border-top: 1px solid var(--rule-strong); padding-top: 20px;
  color: var(--muted); font-size: var(--fs-sm);
  display: flex; flex-wrap: wrap; justify-content: space-between; gap: 6px 24px;
}
footer p { margin: 0; max-width: 72ch; }
@media screen and (max-width: 720px) {
  main { padding: 28px 16px 72px; }
  h1 { font-size: 28px; }
  h2 { margin-top: 52px; font-size: 20px; }
  .verdict { padding: 20px 18px 18px; }
  .verdict .headline { font-size: 22px; }
  .stat .v { font-size: 24px; }
  .stats, .figures { grid-template-columns: 1fr 1fr; }
  .keyfigs { border-left: 0; padding-left: 0; margin-top: 0; grid-template-columns: 1fr 1fr; }
  .plate { padding: 14px 12px 10px; overflow-x: auto; }
  .ruler .ruler-svg { display: none; }
  .ruler .narrow { display: block; width: 100%; }
  /* A wide chart keeps a readable size and scrolls sideways inside its own plate, rather than
     shrinking every label on it below what a phone can render. */
  figure { overflow-x: auto; }
  svg.chart[width="660.00"], svg.chart[width="780.00"], svg.chart[width="880.00"] {
    max-width: none; width: 480px; height: auto; flex-shrink: 0;
  }
  .provenance { grid-template-columns: 1fr; gap: 0; }
  .provenance dd { margin-bottom: 8px; }
  /* A wide table takes its own horizontal scroll instead of dragging the page sideways. */
  table { display: block; max-width: 100%; overflow-x: auto; font-size: var(--fs-sm); }
  th, td { padding: 8px 10px; }
  td.prose { min-width: 26ch; }
}
/* One block, custom properties only: every colour in the report is a var(), so dark mode is
   a palette swap and nothing else has to know it happened. */
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --bg: #141412;
    --panel: #1b1b18;
    --panel-alt: #25241f;
    --shade: #23221e;
    --ink: #eeece6;
    --ink-2: #cfccc3;
    --muted: #a19d93;
    --rule: #2e2d28;
    --rule-strong: #45433c;
    --accent: #4f8be8;
    --accent-ink: #8fb6f5;
    --accent-soft: #1b2535;
    --alert: #e8683a;
    --alert-ink: #f59a74;
    --alert-soft: #2f1f18;
    --good: #5cc98f;
    --warn: #e0b357;
    --bad: #ff8f85;
    --shadow: 0 1px 0 rgba(0, 0, 0, 0.3);
  }
  /* Charts are painted with fixed grey values, because SVG has no custom properties of its own
     for a presentation attribute. The greys are re-mapped by attribute selector rather than
     removed from the chart layer: the same hex that is ink on white is ink on near-black, and
     the Okabe-Ito hues (which carry meaning) are never touched. The rules below are generated
     from jeval.report.svg, so a new grey cannot be forgotten here. */
/*__GREY_REMAP__*/
}
/* Text inside a dark heatmap cell, in both themes: a cell painted at high opacity is the one cell
   whose label needs the page's own background. This rule sits after the dark-mode block on
   purpose -- it has to beat the grey remap above it. */
.heat-strong { fill: var(--panel); }
@page { size: A4 portrait; margin: 12mm; }
/* One page: verdict, reliability chart and impact table. Everything interactive or repeated
   is chrome and goes; collapsed details stay collapsed; nothing splits across a page. */
@media print {
  :root {
    color-scheme: light;
    --bg: #ffffff;
    --panel: #ffffff;
    --panel-alt: #f5f5f5;
    --shade: #f3f3f3;
    --ink: #000000;
    --ink-2: #222222;
    --muted: #454545;
    --rule: #c8c8c8;
    --rule-strong: #888888;
    --accent: #1a4f9e;
    --accent-ink: #1a4f9e;
    --accent-soft: #eef2f8;
    --alert: #a8360f;
    --alert-ink: #a8360f;
    --shadow: none;
  }
  body { background: #fff; color: #000; font-size: 10pt; line-height: 1.45; }
  main { max-width: none; margin: 0; padding: 0; }
  /* A printed report is a summary, and says so: the verdict, the reliability chart open on
     screen, and the first action's cost and impact. Everything interactive, repeated or
     exploratory stays in the file. */
  nav, .toc, .tabs, .tab, .controls, .actions, .verdict-actions, button, .button,
  .copy-summary, .no-print, footer, .slider, #discrimination, #segments, #score, #labels, #drift,
  #data-quality, .splits, .action-block ~ .action-block, .question-block > .note,
  #cost > .intro, #reliability > .intro { display: none !important; }
  .print-only { display: block !important; font-size: 8.5pt; color: var(--muted);
    border-top: 0.5pt solid var(--rule); border-bottom: 0.5pt solid var(--rule); padding: 1.5mm 0; }
  .masthead { margin-bottom: 4mm; }
  .brand { margin-bottom: 3mm; }
  .lede { font-size: 10pt; margin-bottom: 3mm; }
  .provenance { font-size: 7.5pt; padding-top: 2mm; }
  #verdict { order: -1; break-after: auto; }
  .verdict { padding: 5mm 6mm; margin-bottom: 4mm; }
  details:not([open]) { display: none !important; }
  details > summary { display: none !important; }
  details, .verdict, figure, .diag, .warn, .plate { break-inside: avoid; page-break-inside: avoid; }
  .verdict, .plate { box-shadow: none; }
  .plate { padding: 3mm 4mm; margin: 2mm 0 4mm; }
  table { break-inside: avoid; page-break-inside: avoid; font-size: 9pt; }
  tr, th, td { break-inside: avoid; page-break-inside: avoid; }
  th, td { padding: 1.6mm 2.5mm; }
  thead { display: table-header-group; }
  /* Charts print at their drawn proportions, only ever shrunk to the page width: a chart capped
     by height is a chart with five-point labels. */
  figure { margin: 0; }
  figure svg { max-width: 100%; height: auto; }
  .ruler .narrow { display: none !important; }
  .chart-row { display: block; }
  .keyfigs { display: grid; grid-template-columns: repeat(5, 1fr); gap: 2mm; margin: 3mm 0 0;
    border-left: 0; padding: 2mm 0 0; border-top: 0.5pt solid var(--rule); }
  .keyfigs dd { font-size: 12pt; }
  h1 { font-size: 20pt; margin-bottom: 1mm; }
  h2 { font-size: 13pt; margin: 5mm 0 2mm; padding-top: 2mm; break-after: avoid; }
  h3, .block-head, .action-head { break-after: avoid; margin-top: 3mm; }
  .verdict .headline { font-size: 15pt; }
  .stats { grid-template-columns: repeat(3, 1fr); gap: 2mm; margin-top: 3mm; padding-top: 3mm; }
  .stat .v { font-size: 15pt; }
  figcaption { font-size: 8pt; margin-top: 1.5mm; }
  .diag { font-size: 9pt; padding: 2mm 3mm; margin: 2mm 0; }
  #cost { break-before: page; }
}
"""

REPORT_CSS = _CSS.replace("/*__GREY_REMAP__*/", _GREY_REMAP.rstrip("\n"))

REPORT_JS = """\
(function () {
  "use strict";

  var doc = document;
  var DATA_ID = "jeval-data";
  var FLASH_MS = 1400;

  function all(selector, root) {
    return Array.prototype.slice.call((root || doc).querySelectorAll(selector));
  }

  function first(source, names) {
    if (!source) { return undefined; }
    for (var i = 0; i < names.length; i += 1) {
      var value = source[names[i]];
      if (value !== undefined && value !== null) { return value; }
    }
    return undefined;
  }

  function finite(value) {
    var number = Number(value);
    return isFinite(number) ? number : null;
  }

  function pct(value) {
    var number = finite(value);
    if (number === null) { return "n/a"; }
    // Whole percent, matching how the impact table and the charts write a rate: a live readout
    // that disagrees with the table above it by half a point reads as a second, wrong number.
    // toFixed on the magnitude rounds a tie away from zero, which is jeval.currency.format_percent.
    return noNegativeZero((number < 0 ? "-" : "") + Math.abs(number * 100).toFixed(0)) + "%";
  }

  function amount(value, digits) {
    var number = finite(value);
    if (number === null) { return "n/a"; }
    var parts = Math.abs(number).toFixed(digits === undefined ? 2 : digits).split(".");
    var whole = parts[0];
    var out = "";
    var count = 0;
    for (var i = whole.length - 1; i >= 0; i -= 1) {
      out = whole.charAt(i) + out;
      count += 1;
      if (count % 3 === 0 && i > 0) { out = "," + out; }
    }
    if (parts.length > 1) { out = out + "." + parts[1]; }
    return (number < 0 ? "-" : "") + out;
  }

  // Mirrors jeval.currency: the currency's own decimals, widened only for an average smaller than
  // one minor unit, and three significant figures once a suffix is needed.
  function amountDigits(value, base) {
    var magnitude = Math.abs(value);
    if (magnitude === 0 || magnitude >= Math.pow(10, -base)) { return base; }
    var needed = 1 - Math.floor(Math.log(magnitude) / Math.LN10);
    return Math.min(base + 2, needed);
  }

  function withCurrency(text, currency) {
    if (text === "n/a") { return text; }
    return currency ? currency + " " + text : text;
  }

  function noNegativeZero(text) {
    if (text.charAt(0) === "-" && Number(text.replace(/,/g, "")) === 0) { return text.slice(1); }
    return text;
  }

  function formatAmount(value, currency, base) {
    var number = finite(value);
    if (number === null) { return "n/a"; }
    var digits = amountDigits(number, base);
    var text = noNegativeZero(amount(number, digits));
    if (digits > base && Number(text.replace(/,/g, "")) === 0) { text = amount(0, base); }
    return withCurrency(text, currency);
  }

  var LADDER = [[1e3, "k"], [1e6, "M"], [1e9, "B"], [1e12, "T"]];

  function scaledText(scaled) {
    var size = Math.abs(scaled);
    var text = noNegativeZero(amount(scaled, size >= 100 ? 0 : (size >= 10 ? 1 : 2)));
    if (text.indexOf(".") >= 0) {
      text = text.replace(/0+$/, "");
      if (text.charAt(text.length - 1) === ".") { text = text.slice(0, -1); }
    }
    return text.replace(/,/g, "");
  }

  function compact(value, currency, base) {
    var number = finite(value);
    if (number === null) { return "n/a"; }
    var magnitude = Math.abs(number);
    for (var i = 0; i < LADDER.length; i += 1) {
      var fits = magnitude >= LADDER[i][0] && (i + 1 >= LADDER.length || magnitude < LADDER[i + 1][0]);
      if (!fits) { continue; }
      var step = LADDER[i];
      var text = scaledText(number / step[0]);
      if (Math.abs(Number(text)) >= 1000 && i + 1 < LADDER.length) {
        step = LADDER[i + 1];
        text = scaledText(number / step[0]);
      }
      return withCurrency(text + step[1], currency);
    }
    if (currency) { return formatAmount(number, currency, base); }
    return noNegativeZero(amount(number, magnitude >= 10 || magnitude === 0 ? 0 : 2));
  }

  function setText(node, value) {
    if (node && node.textContent !== value) { node.textContent = value; }
  }

  function show(node, visible) {
    if (!node) { return; }
    if (visible) {
      node.removeAttribute("hidden");
    } else {
      node.setAttribute("hidden", "hidden");
    }
  }

  function closest(node, selector) {
    if (node && node.closest) { return node.closest(selector); }
    return null;
  }

  function readData() {
    var node = doc.getElementById(DATA_ID);
    if (!node) { return {}; }
    try {
      var parsed = JSON.parse(node.textContent || "{}");
      if (parsed && typeof parsed === "object") { return parsed; }
      return {};
    } catch (error) {
      return {};
    }
  }

  var data = readData();
  var actions = first(data, ["actions", "thresholds"]) || {};

  function normalise(raw) {
    var threshold;
    var cost;
    var autoRate;
    var accuracy;
    if (Array.isArray(raw)) {
      threshold = finite(raw[0]);
      cost = finite(raw[1]);
      autoRate = finite(raw[2]);
      accuracy = finite(raw[3]);
    } else if (raw && typeof raw === "object") {
      threshold = finite(first(raw, ["threshold", "t"]));
      cost = finite(first(raw, ["expected_cost", "expected_cost_per_case"]));
      if (cost === null) { cost = finite(first(raw, ["cost", "cost_per_case"])); }
      autoRate = finite(first(raw, ["auto_rate", "auto"]));
      accuracy = finite(first(raw, ["accuracy_auto", "accuracy", "acc"]));
    } else {
      return null;
    }
    if (threshold === null) { return null; }
    return {
      "threshold": threshold,
      "cost": cost,
      "auto_rate": autoRate,
      "accuracy_auto": accuracy
    };
  }

  function pointAt(curve, threshold) {
    var best = null;
    var bestGap = Infinity;
    for (var i = 0; i < curve.length; i += 1) {
      var point = normalise(curve[i]);
      if (!point) { continue; }
      var gap = Math.abs(point.threshold - threshold);
      if (gap < bestGap) { bestGap = gap; best = point; }
    }
    return best;
  }

  function renderValue(box, key, value) {
    var nodes = all('[data-jeval-value="' + key + '"]', box);
    for (var i = 0; i < nodes.length; i += 1) { setText(nodes[i], value); }
  }

  function initSlider(box) {
    var slider = box.querySelector('input[type="range"]');
    if (!slider) { return; }
    var action = box.getAttribute("data-jeval-action") || "";
    var spec = first(actions, [action]);
    if (!spec) { return; }
    var currency = box.getAttribute("data-currency") || "";
    var digits = finite(box.getAttribute("data-currency-digits"));
    if (digits === null) { digits = 2; }
    var volumeInput = box.querySelector('input[type="number"]');
    var curve = first(spec, ["curve", "points"]);
    if (!Array.isArray(curve)) { curve = []; }
    var volume = finite(first(spec, ["monthly_volume", "volume"]));

    function render() {
      var threshold = finite(slider.value);
      if (threshold === null) { return; }
      setText(box.querySelector("[data-jeval-threshold]"), threshold.toFixed(2));
      slider.setAttribute("aria-valuetext", threshold.toFixed(2));
      // A typed volume wins over the embedded one, so the monthly figures answer the question the
      // reader just asked instead of the one the report was built with.
      var typed = volumeInput ? finite(volumeInput.value) : null;
      var live = typed === null ? volume : typed;
      var point = pointAt(curve, threshold);
      if (!point) { return; }
      if (point.auto_rate !== null) {
        renderValue(box, "auto_rate", pct(point.auto_rate));
      }
      if (point.accuracy_auto !== null) {
        renderValue(box, "accuracy_auto", pct(point.accuracy_auto));
        renderValue(box, "measured_accuracy", pct(point.accuracy_auto));
      }
      if (point.cost !== null) {
        renderValue(box, "cost_per_case", formatAmount(point.cost, currency, digits));
        if (live !== null) {
          renderValue(box, "cost_per_month", compact(point.cost * live, currency, digits));
        }
      }
      if (point.auto_rate !== null && live !== null) {
        renderValue(box, "auto_per_month", compact(point.auto_rate * live, "", 0));
        renderValue(box, "escalations_per_month", compact((1 - point.auto_rate) * live, "", 0));
      }
    }

    slider.addEventListener("input", render);
    slider.addEventListener("change", render);
    if (volumeInput) {
      volumeInput.addEventListener("input", render);
      volumeInput.addEventListener("change", render);
    }
    render();
  }

  var activateTab = function () {};

  function initTabs() {
    var buttons = all("[data-jeval-tab]");
    if (!buttons.length) { return; }
    var panels = all("[data-jeval-question]");

    activateTab = function (key) {
      for (var i = 0; i < buttons.length; i += 1) {
        var on = buttons[i].getAttribute("data-jeval-tab") === key;
        buttons[i].setAttribute("aria-selected", on ? "true" : "false");
        buttons[i].setAttribute("tabindex", on ? "0" : "-1");
        // The server-rendered first tab carries `is-active` for a reader without JavaScript; left
        // in place, it stays highlighted beside the tab that was actually chosen.
        buttons[i].classList.toggle("is-active", on);
      }
      for (var j = 0; j < panels.length; j += 1) {
        show(panels[j], panels[j].getAttribute("data-jeval-question") === key);
      }
      doc.documentElement.setAttribute("data-jeval-active", key);
    };

    var initial = first(data, ["active_question", "question"]);
    if (typeof initial !== "string" || !initial) {
      initial = buttons[0].getAttribute("data-jeval-tab") || "";
    }
    activateTab(initial);
  }

  function initArrowKeys() {
    doc.addEventListener("keydown", function (event) {
      var tab = closest(event.target, "[data-jeval-tab]");
      if (!tab) { return; }
      var step = event.key === "ArrowRight" ? 1 : (event.key === "ArrowLeft" ? -1 : 0);
      if (!step) { return; }
      var buttons = all("[data-jeval-tab]");
      for (var i = 0; i < buttons.length; i += 1) {
        if (buttons[i] !== tab) { continue; }
        var next = buttons[(i + step + buttons.length) % buttons.length];
        event.preventDefault();
        activateTab(next.getAttribute("data-jeval-tab") || "");
        next.focus();
        return;
      }
    });
  }

  function flash(button, message) {
    if (!button.getAttribute("data-jeval-label")) {
      button.setAttribute("data-jeval-label", button.textContent || "");
    }
    setText(button, message);
    window.setTimeout(function () {
      setText(button, button.getAttribute("data-jeval-label") || "");
    }, FLASH_MS);
  }

  function legacyCopy(text, done, failed) {
    var area = doc.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "readonly");
    area.style.position = "fixed";
    area.style.top = "-1000px";
    doc.body.appendChild(area);
    area.select();
    var ok = false;
    try {
      ok = doc.execCommand("copy");
    } catch (error) {
      ok = false;
    }
    doc.body.removeChild(area);
    if (ok) { done(); } else { failed(); }
  }

  function copySummary(button) {
    var markdown = first(data, ["summary_markdown", "summary"]);
    if (typeof markdown !== "string" || !markdown) { return; }
    function done() { flash(button, "Copied"); }
    function failed() { flash(button, "Copy failed"); }
    var clipboard = navigator.clipboard;
    if (clipboard && clipboard.writeText) {
      var promise = clipboard.writeText(markdown);
      promise.then(done, function () { legacyCopy(markdown, done, failed); });
      return;
    }
    legacyCopy(markdown, done, failed);
  }

  function findChart(bar) {
    var key = bar.getAttribute("data-jeval-segment");
    if (key !== null) {
      var charts = all("[data-jeval-segment-chart]");
      for (var i = 0; i < charts.length; i += 1) {
        if (charts[i].getAttribute("data-jeval-segment-chart") === key) { return charts[i]; }
      }
      return null;
    }
    var target = bar.getAttribute("data-target");
    if (target) { return doc.getElementById(target); }
    return null;
  }

  function toggleSegment(bar) {
    var chart = findChart(bar);
    if (!chart) { return; }
    var opening = chart.hasAttribute("hidden");
    show(chart, opening);
    bar.setAttribute("aria-expanded", opening ? "true" : "false");
    if (bar.classList) { bar.classList.toggle("is-open", opening); }
  }

  function init() {
    initTabs();
    initArrowKeys();
    var boxes = all("[data-jeval-action]");
    for (var i = 0; i < boxes.length; i += 1) { initSlider(boxes[i]); }
  }

  doc.addEventListener("click", function (event) {
    var node = event.target;
    var copy = closest(node, "[data-jeval-copy-summary], [data-copy-target]");
    if (copy) { copySummary(copy); return; }
    var tab = closest(node, "[data-jeval-tab]");
    if (tab) { activateTab(tab.getAttribute("data-jeval-tab") || ""); return; }
    var bar = closest(node, "[data-jeval-segment], [data-target]");
    if (bar) { toggleSegment(bar); }
  });

  if (doc.readyState === "loading") {
    doc.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
"""

# Characters that never need surrounding whitespace once comments are gone.
_TIGHT = "{};,"
_QUOTES = "\"'`"


def minify(source: str) -> str:
    """Strip comments and collapse whitespace runs outside string literals.

    Report size is a hard requirement, so this runs on :data:`REPORT_CSS` and :data:`REPORT_JS`
    before inlining. Text inside quotes is copied byte for byte: collapsing whitespace inside a
    CSS ``content`` value or a JavaScript string would change what the report renders.

    ``//`` is only treated as a line comment when it does not follow a colon, so a URL that
    survives in a string or a plain text node is not truncated. JavaScript written for this
    module therefore terminates every statement with a semicolon: with whitespace collapsed,
    anything relying on automatic semicolon insertion would stop parsing.
    """
    out: list[str] = []
    index = 0
    length = len(source)
    quote = ""
    previous = ""
    pending_space = False

    while index < length:
        char = source[index]

        if quote:
            out.append(char)
            if char == "\\" and index + 1 < length:
                out.append(source[index + 1])
                index += 2
                continue
            if char == quote:
                quote = ""
                previous = char
            index += 1
            continue

        if char in _QUOTES:
            if pending_space and _needs_space(previous, char):
                out.append(" ")
            pending_space = False
            quote = char
            previous = char
            out.append(char)
            index += 1
            continue

        if char == "/" and index + 1 < length:
            following = source[index + 1]
            if following == "*":
                end = source.find("*/", index + 2)
                index = length if end < 0 else end + 2
                continue
            if following == "/" and previous != ":":
                newline = source.find("\n", index + 2)
                index = length if newline < 0 else newline
                continue

        if char.isspace():
            pending_space = True
            index += 1
            continue

        if pending_space and _needs_space(previous, char):
            out.append(" ")
        pending_space = False
        out.append(char)
        previous = char
        index += 1

    return "".join(out)


def _needs_space(previous: str, char: str) -> bool:
    if not previous:
        return False
    return previous not in _TIGHT and char not in _TIGHT
