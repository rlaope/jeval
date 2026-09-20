"""Inline CSS and JavaScript for the single-file report.

One HTML file, no sibling assets: the stylesheet and the behaviour are module-level strings
that the template inlines. Nothing here reads the filesystem, and nothing it emits reaches the
network -- no ``@import``, no ``url()``, no web font, no CDN script.

All colour lives in custom properties, so a single ``@media (prefers-color-scheme: dark)``
block restyles the whole report, and the ``@media print`` block lays the argument out for one
A4 portrait page.

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
    own label, and one ``[data-jeval-value="<key>"]`` per displayed figure. Keys: ``auto_rate``,
    ``accuracy_auto``, ``measured_accuracy``, ``cost_per_case``, ``cost_per_month``,
    ``auto_per_month``, ``escalations_per_month``.
``[data-jeval-copy-summary]``
    Copy-summary button; copies ``summary_markdown``. ``[data-copy-target]`` is accepted as an
    alias because the template already ships it.
``[data-jeval-segment="<key>"]`` / ``[data-jeval-segment-chart="<key>"]``
    Clickable segment bar and the per-segment chart it toggles into view.
    ``[data-target="<element id>"]`` is accepted as an alias: the bar toggles the element whose
    ``id`` it names.
"""

from __future__ import annotations

__all__ = ["REPORT_CSS", "REPORT_JS", "minify"]

REPORT_CSS = """\
:root {
  color-scheme: light dark;
  --bg: #ffffff;
  --panel: #f6f7f9;
  --panel-alt: #eceff3;
  --ink: #14161a;
  --muted: #5b6472;
  --rule: #e2e5ea;
  --accent: #0072b2;
  --accent-soft: #e8f1f9;
  --good: #009e73;
  --warn: #b58900;
  --bad: #d55e00;
  --shadow: 0 1px 2px rgba(20, 22, 26, 0.07);
  --radius: 8px;
  --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
* { box-sizing: border-box; }
[hidden] { display: none !important; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0;
  padding: 0;
  background: var(--bg);
  color: var(--ink);
  font: 15px/1.55 var(--font);
}
main { max-width: 940px; margin: 0 auto; padding: 28px 20px 72px; }
h1 { font-size: 22px; line-height: 1.25; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 {
  font-size: 17px; margin: 34px 0 12px; padding-bottom: 6px;
  border-bottom: 1px solid var(--rule);
}
h3 { font-size: 14px; margin: 22px 0 8px; }
p, li { margin: 0 0 8px; }
ul { margin: 0 0 8px; padding-left: 20px; }
a { color: var(--accent); }
.meta { color: var(--muted); font-size: 12.5px; margin: 0 0 20px; }
.note, figcaption, .card .s { color: var(--muted); font-size: 12.5px; }
.num { font-family: var(--mono); font-variant-numeric: tabular-nums; }
code, kbd, samp { font-family: var(--mono); font-size: 0.94em; }
nav.report-nav {
  display: flex; flex-wrap: wrap; gap: 14px;
  margin: 0 0 18px; font-size: 13px;
}
nav.report-nav a { color: var(--muted); text-decoration: none; }
nav.report-nav a:hover { color: var(--accent); }
section { margin: 0 0 26px; }
/* The template emits sections in SECTION_ORDER, so the verdict is already first in the
   document; the rule below keeps it first if a container is ever flexed or reordered. */
#verdict { order: -1; margin-top: 8px; }
.verdict {
  border: 1px solid var(--rule);
  border-left: 4px solid var(--accent);
  border-radius: var(--radius);
  background: var(--panel);
  padding: 16px 18px;
}
#verdict[data-status="too_low"] .verdict, #verdict[data-status="too_high"] .verdict {
  border-left-color: var(--warn);
}
#verdict[data-status="insufficient_data"] .verdict { border-left-color: var(--muted); }
.verdict .headline { font-size: 19px; font-weight: 600; margin: 0 0 6px; }
.verdict .detail { margin: 0; font-size: 14px; color: var(--muted); }
.stats {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px; margin: 16px 0 4px;
}
.card {
  border: 1px solid var(--rule); border-radius: var(--radius); background: var(--bg);
  padding: 12px 14px; box-shadow: var(--shadow);
}
.card .k {
  font-size: 11.5px; letter-spacing: 0.05em; text-transform: uppercase; color: var(--muted);
}
.card .v { font: 600 20px/1.25 var(--mono); margin-top: 5px; }
.card .s { margin-top: 3px; }
.tabs {
  display: flex; flex-wrap: wrap; gap: 6px; margin: 14px 0 12px;
  border-bottom: 1px solid var(--rule);
}
.tab {
  -webkit-appearance: none; appearance: none; background: none; color: var(--muted);
  border: 1px solid transparent; border-bottom: 2px solid transparent;
  font: inherit; font-size: 13.5px; padding: 7px 11px; border-radius: 6px 6px 0 0;
  cursor: pointer;
}
.tab:hover { background: var(--panel); color: var(--ink); }
.tab[aria-selected="true"], .tab.is-active {
  color: var(--ink); background: var(--panel); border-bottom-color: var(--accent);
  font-weight: 600;
}
.tab:focus-visible, button:focus-visible, input[type="range"]:focus-visible,
[data-jeval-segment]:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
figure { margin: 10px 0 16px; }
figure svg { display: block; width: 100%; height: auto; }
figcaption { margin-top: 6px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0 16px; font-size: 13px; }
th, td {
  border-bottom: 1px solid var(--rule); padding: 7px 10px;
  text-align: right; vertical-align: top;
}
th:first-child, td:first-child { text-align: left; }
th {
  color: var(--muted); font-weight: 600; font-size: 11.5px;
  text-transform: uppercase; letter-spacing: 0.04em;
}
tbody tr:last-child td { border-bottom: 0; }
.diag {
  border-left: 3px solid var(--accent); background: var(--panel);
  padding: 10px 14px; margin: 10px 0 16px; font-size: 14px;
}
.warn {
  border-left: 3px solid var(--warn); background: var(--panel-alt);
  padding: 10px 14px; margin: 12px 0 16px; font-size: 14px;
}
.good { color: var(--good); }
.bad { color: var(--bad); }
details {
  border: 1px solid var(--rule); border-radius: var(--radius); background: var(--panel);
  padding: 8px 12px; margin: 10px 0 16px;
}
details > summary { cursor: pointer; font-size: 13px; color: var(--muted); }
details[open] > summary { margin-bottom: 8px; }
details table { margin: 4px 0 6px; }
.controls, .actions {
  display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin: 12px 0;
}
button, .button {
  -webkit-appearance: none; appearance: none; cursor: pointer;
  border: 1px solid var(--rule); border-radius: 6px; background: var(--panel);
  color: var(--ink); font: inherit; font-size: 13px; padding: 6px 12px;
}
button:hover, .button:hover { border-color: var(--accent); color: var(--accent); }
.copy-summary { white-space: nowrap; }
.slider {
  border: 1px solid var(--rule); border-radius: var(--radius); background: var(--panel);
  padding: 12px 14px; margin: 12px 0 16px;
}
.slider .row { display: flex; flex-wrap: wrap; align-items: baseline; gap: 12px;
  justify-content: space-between; }
.slider .label { font-size: 13px; color: var(--muted); }
.slider [data-jeval-threshold] { font-family: var(--mono); font-size: 15px; font-weight: 600; }
.slider .readout { font-family: var(--mono); }
input[type="range"] {
  -webkit-appearance: none; appearance: none; display: block;
  width: 100%; height: 22px; margin: 6px 0 2px; background: transparent; cursor: ew-resize;
}
input[type="range"]::-webkit-slider-runnable-track {
  height: 4px; border-radius: 2px; background: var(--rule);
}
input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none; appearance: none; width: 16px; height: 16px;
  margin-top: -6px; border: 0; border-radius: 50%; background: var(--accent);
}
input[type="range"]::-moz-range-track { height: 4px; border-radius: 2px; background: var(--rule); }
input[type="range"]::-moz-range-thumb {
  width: 16px; height: 16px; border: 0; border-radius: 50%; background: var(--accent);
}
[data-jeval-segment] { cursor: pointer; }
[data-jeval-segment][aria-expanded="true"] { font-weight: 600; }
[data-jeval-segment-chart] { margin-top: 10px; }
footer {
  margin-top: 46px; border-top: 1px solid var(--rule); padding-top: 16px;
  color: var(--muted); font-size: 12.5px;
}
@media (max-width: 640px) {
  main { padding: 20px 14px 48px; }
  .stats { grid-template-columns: 1fr; }
}
/* One block, custom properties only: every colour in the report is a var(), so dark mode is
   a palette swap and nothing else has to know it happened. */
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --bg: #0f1115;
    --panel: #171b22;
    --panel-alt: #1e232c;
    --ink: #eef1f5;
    --muted: #a5aeba;
    --rule: #2a303a;
    --accent: #56b4e9;
    --accent-soft: #12304a;
    --good: #4cc79a;
    --warn: #e0b64a;
    --bad: #f08a54;
    --shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
  }
  /* Charts are painted with fixed ink values; remap only the greys the palette keeps as
     text and grid, never the Okabe-Ito hues that carry meaning. */
  figure svg text[fill="#111111"], figure svg circle[fill="#111111"] { fill: var(--ink); }
  figure svg text[fill="#666666"] { fill: var(--muted); }
  figure svg line[stroke="#111111"] { stroke: var(--ink); }
  figure svg line[stroke="#dddddd"], figure svg line[stroke="#999999"] { stroke: var(--rule); }
  figure svg rect[fill="#f2f2f2"] { fill: var(--panel-alt); }
}
@page { size: A4 portrait; margin: 12mm; }
/* One page: verdict, reliability chart and impact table. Everything interactive or repeated
   is chrome and goes; collapsed details stay collapsed; nothing splits across a page. */
@media print {
  :root {
    color-scheme: light;
    --bg: #ffffff;
    --panel: #ffffff;
    --panel-alt: #f5f5f5;
    --ink: #000000;
    --muted: #333333;
    --rule: #b8b8b8;
    --shadow: none;
  }
  body { background: #fff; color: #000; font-size: 10.5pt; }
  main { max-width: none; margin: 0; padding: 0; }
  nav, .report-nav, .tabs, .tab, .controls, .actions, button, .button, .copy-summary,
  .no-print, footer, .slider { display: none !important; }
  #verdict { order: -1; break-after: avoid; page-break-after: avoid; }
  section { break-inside: avoid; page-break-inside: avoid; margin: 0 0 5mm; }
  details:not([open]) { display: none !important; }
  details > summary { display: none !important; }
  details, .card, .verdict, figure, .diag, .warn {
    break-inside: avoid; page-break-inside: avoid;
  }
  table { break-inside: avoid; page-break-inside: avoid; font-size: 9.5pt; }
  tr, th, td { break-inside: avoid; page-break-inside: avoid; }
  thead { display: table-header-group; }
  figure svg { max-height: 56mm; width: auto; margin: 0 auto; }
  h1 { font-size: 16pt; margin-bottom: 1mm; }
  h2 { font-size: 12pt; margin: 4mm 0 1.5mm; }
  .verdict .headline { font-size: 14pt; }
  .stats { grid-template-columns: repeat(3, 1fr); gap: 2mm; }
  .card .v { font-size: 13pt; }
}
"""

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

  function pct(value, digits) {
    var number = finite(value);
    if (number === null) { return "n/a"; }
    return (number * 100).toFixed(digits === undefined ? 1 : digits) + "%";
  }

  function fixed(value, digits) {
    var number = finite(value);
    if (number === null) { return "n/a"; }
    return number.toFixed(digits === undefined ? 3 : digits);
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
    var curve = first(spec, ["curve", "points"]);
    if (!Array.isArray(curve)) { curve = []; }
    var volume = finite(first(spec, ["monthly_volume", "volume"]));

    function render() {
      var threshold = finite(slider.value);
      if (threshold === null) { return; }
      setText(box.querySelector("[data-jeval-threshold]"), threshold.toFixed(2));
      slider.setAttribute("aria-valuetext", threshold.toFixed(2));
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
        renderValue(box, "cost_per_case", fixed(point.cost, 3));
        if (volume !== null) {
          renderValue(box, "cost_per_month", money(point.cost * volume));
        }
      }
      if (point.auto_rate !== null && volume !== null) {
        renderValue(box, "auto_per_month", grouped(point.auto_rate * volume));
        renderValue(box, "escalations_per_month", grouped((1 - point.auto_rate) * volume));
      }
    }

    slider.addEventListener("input", render);
    slider.addEventListener("change", render);
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
