"""Inline CSS and JavaScript for the single-file report.

One HTML file, no sibling assets: the stylesheet and the behaviour are module-level strings
that the template inlines. Nothing here reads the filesystem, and nothing it emits reaches the
network -- no ``@import``, no ``url()``, no web font, no CDN script.

All colour lives in custom properties, so a single ``@media (prefers-color-scheme: dark)``
block restyles the whole report, and the ``@media print`` block lays the argument out for one
A4 portrait page. The one thing custom properties cannot reach is a chart's own presentation
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
    ``data-currency`` attribute (e.g. ``KRW``) and an ``input[type="number"]`` for the monthly
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
    ('line[stroke="{v}"]', "stroke", "ink", chart_svg.INK),
    ('path[stroke="{v}"]', "stroke", "ink", chart_svg.INK),
    ('text[fill="{v}"]', "fill", "muted", chart_svg.MUTED),
    ('line[stroke="{v}"]', "stroke", "muted", chart_svg.SOFT),
    ('path[stroke="{v}"]', "stroke", "muted", chart_svg.SOFT),
    ('circle[fill="{v}"]', "fill", "muted", chart_svg.SOFT),
    ('line[stroke="{v}"]', "stroke", "rule", chart_svg.GRID),
    ('line[stroke="{v}"]', "stroke", "rule", chart_svg.DIAGONAL),
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
  --bg: #fbfbfc;
  --panel: #ffffff;
  --panel-alt: #f4f6f9;
  --ink: #16181d;
  --muted: #5f6773;
  --rule: #e6e8ec;
  --accent: #2f5fd0;
  --accent-soft: #eef3fd;
  --good: #10795c;
  --warn: #95590a;
  --bad: #b3261e;
  --alert: #b3261e;
  --shade: #f2f4f7;
  --shadow: 0 1px 2px rgba(22, 24, 29, 0.08);
  --radius: 10px;
  --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
/* A shaded band in a chart is a tint, not a colour of its own: as a variable it follows the
   theme instead of staying near-white in a dark report. */
.shade { fill: var(--shade); }
* { box-sizing: border-box; }
[hidden] { display: none !important; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 15px/1.6 var(--font);
  -webkit-font-smoothing: antialiased;
}
/* One column, wide enough for a 660pt chart to scale up and short enough to read: 880px is the
   measure this report is typeset for. */
main { max-width: 880px; margin: 0 auto; padding: 44px 24px 96px; }
h1 { font-size: 28px; line-height: 1.2; letter-spacing: -0.02em; margin: 0 0 10px; }
.lede { font-size: 16px; line-height: 1.55; color: var(--muted); max-width: 58ch; margin: 0 0 20px; }
h2 {
  font-size: 20px; line-height: 1.3; letter-spacing: -0.01em;
  margin: 46px 0 14px; padding-bottom: 8px; border-bottom: 1px solid var(--rule);
}
h3 { font-size: 15px; line-height: 1.45; margin: 26px 0 6px; }
p { margin: 0 0 12px; }
ul { margin: 0 0 12px; padding-left: 22px; }
li { margin: 0 0 6px; }
a { color: var(--accent); }
section { margin: 0 0 10px; }
code, kbd, samp { font-family: var(--mono); font-size: 0.94em; }
.num { font-family: var(--mono); font-variant-numeric: tabular-nums; }
.note { color: var(--muted); font-size: 12.5px; }
/* The template emits sections in SECTION_ORDER, so the verdict is already first in the
   document; the rule below keeps it first if a container is ever flexed or reordered. */
#verdict { order: -1; margin: 0 0 10px; }

/* Provenance: one item per line. As a single `·`-separated run it reads as boilerplate and gets
   skipped, which is exactly the part a reader needs to judge the sample. */
.meta { display: flex; flex-direction: column; gap: 4px; margin: 0; color: var(--muted); font-size: 12.5px; line-height: 1.5; }
.meta-item { display: block; max-width: 86ch; }

/* The verdict. One card, one coloured edge for the status, and the three figures that carry it
   underneath — no shadow, no second frame. */
.verdict {
  border: 1px solid var(--rule);
  border-left: 3px solid var(--accent);
  border-radius: var(--radius);
  background: var(--panel);
  padding: 20px 24px;
}
#verdict[data-status="too_low"].verdict, #verdict[data-status="too_high"].verdict { border-left-color: var(--warn); }
#verdict[data-status="insufficient_data"].verdict { border-left-color: var(--muted); }
.verdict .headline { font-size: 21px; line-height: 1.32; letter-spacing: -0.01em; font-weight: 600; margin: 0 0 8px; }
.verdict .detail { margin: 0; font-size: 14.5px; line-height: 1.6; color: var(--muted); max-width: 68ch; }
.stats {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 18px 28px; margin: 20px 0 0; padding-top: 18px; border-top: 1px solid var(--rule);
}
.stat .k { font-size: 12px; color: var(--muted); }
.stat .v { font-size: 22px; line-height: 1.25; font-weight: 600; letter-spacing: -0.01em; margin-top: 4px; }
.stat .s { font-size: 12.5px; line-height: 1.5; color: var(--muted); margin-top: 2px; }
.stats.small { gap: 14px 24px; margin-top: 16px; }
.stats.small .stat .v { font-size: 17px; }
.verdict-actions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }

/* Tabs as one segmented control: a row of loose labels reads as text, not as a control. */
.tabs {
  display: inline-flex; flex-wrap: wrap; gap: 2px; padding: 3px;
  margin: 16px 0; border: 1px solid var(--rule); border-radius: var(--radius);
  background: var(--panel-alt);
}
.tab {
  -webkit-appearance: none; appearance: none; cursor: pointer;
  border: 0; background: none; color: var(--muted);
  font: inherit; font-size: 13px; padding: 6px 12px; border-radius: 7px;
}
.tab:hover { color: var(--ink); background: var(--panel); }
.tab[aria-selected="true"], .tab.is-active {
  color: var(--ink); background: var(--panel); font-weight: 600; box-shadow: var(--shadow);
}
.tab:focus-visible, button:focus-visible, input:focus-visible,
[data-jeval-segment]:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.block-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 12px; margin: 26px 0 4px; }
.block-head h3 { margin: 0; }
.block-sub { margin: 0; font-size: 12.5px; color: var(--muted); }

/* Callouts. The left edge carries the status, the tint only separates them from the page. */
.diag {
  border: 1px solid var(--rule); border-left: 3px solid var(--accent); border-radius: 0 8px 8px 0;
  background: var(--accent-soft); padding: 12px 16px; margin: 12px 0 18px;
  font-size: 14.5px; line-height: 1.6;
}
.warn {
  border: 1px solid var(--rule); border-left: 3px solid var(--warn); border-radius: 0 8px 8px 0;
  background: var(--panel-alt); padding: 12px 16px; margin: 16px 0 20px;
  font-size: 14.5px; line-height: 1.6;
}
.warn ul { margin: 8px 0 0; }
.good { color: var(--good); }
.bad { color: var(--bad); }

figure { margin: 18px 0 24px; }
figure svg { display: block; width: 100%; height: auto; }
figcaption { margin-top: 10px; color: var(--muted); font-size: 12.5px; line-height: 1.55; max-width: 74ch; }

table { border-collapse: collapse; width: 100%; margin: 14px 0 20px; font-size: 13.5px; }
caption { caption-side: top; text-align: left; color: var(--muted); font-size: 12.5px; padding: 0 0 8px; }
th, td { padding: 9px 12px; border-bottom: 1px solid var(--rule); text-align: right; vertical-align: baseline; }
th:first-child, td:first-child { text-align: left; }
th {
  color: var(--muted); font-weight: 600; font-size: 11.5px;
  text-transform: uppercase; letter-spacing: 0.06em;
}
th.num, td.num { text-align: right; }
tbody tr:hover td { background: var(--panel-alt); }
tbody tr:last-child td { border-bottom: 0; }
/* The recommended column is the proposal this whole document argues for; a tint says so
   without a second colour legend. */
table.impact th:nth-child(3), table.impact td:nth-child(3) { background: var(--accent-soft); }
table.impact td:first-child { font-weight: 500; }

/* Collapsible data tables: closed by default, obviously openable, and print when open. */
details {
  border: 1px solid var(--rule); border-radius: var(--radius); background: var(--panel);
  margin: 14px 0 20px;
}
details > summary { cursor: pointer; list-style: none; padding: 11px 16px; font-size: 13px; color: var(--muted); }
details > summary:hover { color: var(--ink); }
details > summary::-webkit-details-marker { display: none; }
details > summary::before { content: "▸"; display: inline-block; width: 14px; color: var(--muted); }
details[open] > summary { border-bottom: 1px solid var(--rule); }
details[open] > summary::before { content: "▾"; }
details table { margin: 0; }
details th:first-child, details td:first-child { padding-left: 16px; }
details th:last-child, details td:last-child { padding-right: 16px; }

button, .button {
  -webkit-appearance: none; appearance: none; cursor: pointer;
  border: 1px solid var(--rule); border-radius: 8px; background: var(--panel);
  color: var(--ink); font: inherit; font-size: 13px; padding: 7px 14px;
}
button:hover, .button:hover { border-color: var(--accent); color: var(--accent); }
.copy-summary { white-space: nowrap; }

/* The threshold explorer. Everything the slider rewrites sits below the rule, so a reader can
   see at a glance which numbers are live and which ones are the report's own measurement. */
.controls, .actions { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin: 12px 0; }
.slider {
  border: 1px solid var(--rule); border-radius: var(--radius); background: var(--panel);
  padding: 16px 18px; margin: 16px 0 20px;
}
.slider-head { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 10px 16px; }
.slider-title { display: inline-flex; align-items: baseline; gap: 10px; }
.slider-title label { font-size: 13px; color: var(--muted); }
.slider [data-jeval-threshold], [data-jeval-threshold] {
  font-family: var(--mono); font-size: 18px; font-weight: 600; font-variant-numeric: tabular-nums;
}
.volume { display: inline-flex; align-items: center; gap: 8px; font-size: 12.5px; color: var(--muted); }
input[type="number"] {
  font: inherit; font-family: var(--mono); font-size: 13px; width: 96px;
  padding: 5px 8px; border: 1px solid var(--rule); border-radius: 8px;
  background: var(--bg); color: var(--ink);
}
input[type="range"] {
  -webkit-appearance: none; appearance: none; display: block;
  width: 100%; height: 24px; margin: 8px 0 0; background: transparent; cursor: ew-resize;
}
input[type="range"]::-webkit-slider-runnable-track {
  height: 6px; border-radius: 999px; background: var(--rule);
}
input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none; appearance: none; width: 18px; height: 18px;
  margin-top: -6px; border: 2px solid var(--panel); border-radius: 50%;
  background: var(--accent); box-shadow: var(--shadow);
}
input[type="range"]::-moz-range-track { height: 6px; border-radius: 999px; background: var(--rule); }
input[type="range"]::-moz-range-thumb {
  width: 18px; height: 18px; border: 2px solid var(--panel); border-radius: 50%; background: var(--accent);
}
.scale { display: flex; justify-content: space-between; font-family: var(--mono); font-size: 11.5px; color: var(--muted); }
.figures {
  display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px 24px; margin: 16px 0 4px; padding-top: 16px; border-top: 1px solid var(--rule);
}
.figures .figure { display: flex; flex-direction: column; gap: 2px; }
.figures .k { color: var(--muted); font-size: 12px; }
.figures .v { font-size: 15px; font-weight: 600; font-variant-numeric: tabular-nums; }

/* Segment bars: the track behind them is what makes two lengths comparable. */
.seg-bar { cursor: pointer; }
.seg-bar:hover .seg-fill, .seg-bar.is-open .seg-fill { opacity: 0.82; }
.seg-fill { fill: var(--accent); }
.accent-stroke { stroke: var(--accent); }
.accent-dot { fill: var(--accent); }
/* The line in use, and a model change: the same warning in two charts, one value per theme. */
.alert-stroke { stroke: var(--alert); }
.alert-ink { fill: var(--alert); }
.alert-dot { fill: var(--alert); }
/* An annotation drawn on top of a gridline needs its own ground; a box would be louder. */
.halo { paint-order: stroke; stroke: var(--bg); stroke-width: 3px; stroke-linejoin: round; }
[data-jeval-segment] { cursor: pointer; }
[data-jeval-segment][aria-expanded="true"] { font-weight: 600; }
[data-jeval-segment-chart] { margin-top: 12px; }

footer {
  margin-top: 56px; border-top: 1px solid var(--rule); padding-top: 18px;
  color: var(--muted); font-size: 12.5px;
}
footer p { margin: 0 0 6px; }
@media (max-width: 640px) {
  main { padding: 28px 16px 64px; }
  h1 { font-size: 24px; }
  h2 { margin-top: 34px; }
  .stats, .figures { grid-template-columns: 1fr 1fr; }
  table { font-size: 12.5px; }
}
/* One block, custom properties only: every colour in the report is a var(), so dark mode is
   a palette swap and nothing else has to know it happened. */
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --bg: #12141a;
    --panel: #181b22;
    --panel-alt: #1f232b;
    --shade: #22262f;
    --ink: #e7eaef;
    --muted: #9aa3b1;
    --rule: #2a2f39;
    --accent: #7ea6ff;
    --accent-soft: #1b2436;
    --good: #4fc99f;
    --warn: #e0b357;
    --bad: #ff8f85;
    --alert: #ff9d92;
    --shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
  }
  /* Charts are painted with fixed grey values, because SVG has no custom properties of its own
     for a presentation attribute. The greys are re-mapped by attribute selector rather than
     removed from the chart layer: the same hex that is ink on white is ink on near-black, and
     the Okabe-Ito hues (which carry meaning) are never touched. The rules below are generated
     from jeval.report.svg, so a new grey cannot be forgotten here. */
/*__GREY_REMAP__*/
}
/* Text inside a dark heatmap cell, in both themes: a cell painted with the page's own ink at
   high opacity is the one cell whose label needs the page's own background. This rule sits
   after the dark-mode block on purpose -- it has to beat the grey remap above it. */
.heat-strong { fill: var(--bg); }
@page { size: A4 portrait; margin: 12mm; }
/* One page: verdict, reliability chart and impact table. Everything interactive or repeated
   is chrome and goes; collapsed details stay collapsed; nothing splits across a page. */
@media print {
  :root {
    color-scheme: light;
    --bg: #ffffff;
    --panel: #ffffff;
    --panel-alt: #f5f5f5;
    --shade: #f5f5f5;
    --ink: #000000;
    --muted: #333333;
    --rule: #b8b8b8;
    --accent: #1a3f8f;
    --accent-soft: #f0f0f0;
    --alert: #a01b14;
    --shadow: none;
  }
  body { background: #fff; color: #000; font-size: 10.5pt; line-height: 1.45; }
  main { max-width: none; margin: 0; padding: 0; }
  nav, .report-nav, .tabs, .tab, .controls, .actions, .verdict-actions, button, .button,
  .copy-summary, .no-print, footer, .slider { display: none !important; }
  .lede { font-size: 10pt; margin-bottom: 3mm; }
  .meta { font-size: 8pt; }
  #verdict { order: -1; break-after: avoid; page-break-after: avoid; }
  section { break-inside: avoid; page-break-inside: avoid; margin: 0 0 5mm; }
  details:not([open]) { display: none !important; }
  details > summary { display: none !important; }
  details, .verdict, figure, .diag, .warn { break-inside: avoid; page-break-inside: avoid; }
  table { break-inside: avoid; page-break-inside: avoid; font-size: 9.5pt; }
  tr, th, td { break-inside: avoid; page-break-inside: avoid; }
  thead { display: table-header-group; }
  figure svg { max-height: 56mm; width: auto; margin: 0 auto; }
  h1 { font-size: 16pt; margin-bottom: 1mm; }
  h2 { font-size: 12pt; margin: 4mm 0 1.5mm; }
  .verdict .headline { font-size: 14pt; }
  .stats { grid-template-columns: repeat(3, 1fr); gap: 2mm; }
  .stat .v { font-size: 13pt; }
  figcaption { font-size: 8pt; margin-top: 1.5mm; }
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
    return Math.round(number * 100) + "%";
  }

  function grouped(value) {
    var number = finite(value);
    if (number === null) { return "n/a"; }
    var text = String(Math.round(number));
    var sign = text.charAt(0) === "-" ? "-" : "";
    if (sign) { text = text.slice(1); }
    var out = "";
    var count = 0;
    for (var i = text.length - 1; i >= 0; i -= 1) {
      out = text.charAt(i) + out;
      count += 1;
      if (count % 3 === 0 && i > 0) { out = "," + out; }
    }
    return sign + out;
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

  function withCurrency(text, currency) {
    if (text === "n/a") { return text; }
    return currency ? currency + " " + text : text;
  }

  function money(value) {
    var number = finite(value);
    if (number === null) { return "n/a"; }
    var magnitude = Math.abs(number);
    if (magnitude >= 1e9) { return (number / 1e9).toFixed(1) + "B"; }
    if (magnitude >= 1e6) { return (number / 1e6).toFixed(1) + "M"; }
    if (magnitude >= 1e3) { return (number / 1e3).toFixed(1) + "k"; }
    return grouped(number);
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
        renderValue(box, "cost_per_case", withCurrency(amount(point.cost, 2), currency));
        if (live !== null) {
          renderValue(box, "cost_per_month", withCurrency(money(point.cost * live), currency));
        }
      }
      if (point.auto_rate !== null && live !== null) {
        renderValue(box, "auto_per_month", money(point.auto_rate * live));
        renderValue(box, "escalations_per_month", money((1 - point.auto_rate) * live));
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
