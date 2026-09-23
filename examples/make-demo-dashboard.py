"""Build the single-file demo dashboard used for the walkthrough video.

This is not a jeval command and not a product surface: the product writes one report, and a
dashboard is explicitly out of scope. This is a *presentation* of the same analysis, built from the
same code, so a recorded walkthrough can move through the argument in five steps instead of fifteen
scrolls.

    uv run python examples/make-demo-dashboard.py examples/demo-dashboard.html

One HTML file, no server, no network, no chart library — and every figure is computed by the library
the report uses (``evaluate``, ``sweep_actions``, ``segment_views``, ``drift_view``,
``build_verdict``, ``build_impact``). Nothing on the page is typed in by hand, which is what makes
it safe to put on screen.

The synthetic log is two model versions of the same four questions. The older version is roughly
honest on the question the cost matrix fires on; the newer one is degraded, which is the change the
drift step exists to catch.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jeval.cli import resolve_cost_actions, segment_views, sweep_actions
from jeval.drift import DriftView, parse_fail_on, run_checks
from jeval.evaluate import DatasetReport, evaluate
from jeval.report import svg as S
from jeval.report import template
from jeval.report.assets import REPORT_CSS, minify
from jeval.report.charts import cost as cost_charts
from jeval.report.charts import drift as drift_charts
from jeval.report.charts import reliability as reliability_charts
from jeval.report.charts import segments as segment_charts
from jeval.report.model import ImpactTable
from jeval.report.verdict import build_verdict
from jeval.schema import DecisionRecord
from jeval.synth import SynthSpec, demo_dataset, generate

NEWER_START = datetime(2026, 9, 17, tzinfo=timezone.utc)

CURRENCY = "KRW"
MONTHLY_VOLUME = 20_000.0
CURRENT_THRESHOLD = 0.60
FAIL_ON = "ece-increase=0.05"

# The cost matrix the dashboard runs on -- the same synthetic one `jeval demo` writes, so the numbers
# on the first screen can be checked against `examples/report-example.html` in this repository.
COSTS_YAML = """\
# SYNTHETIC cost figures. Replace them with your own before believing any threshold that comes
# out of this file.
actions:
  - name: auto_refund
    question: intent
    when: refund_request
    cost_false_accept: 12000
    cost_escalate: 2000
    cost_false_reject: 0
  - name: auto_route
    question: department
    when: billing
    cost_false_accept: 8000
    cost_escalate: 1500
    cost_false_reject: 200
  - name: auto_escalate_urgent
    question: is_urgent
    when: "yes"
    cost_false_accept: 30000
    cost_escalate: 500
    cost_false_reject: 1000
"""

# (title, the claim the screen makes, the line read out while it is on screen)
SCENES: tuple[tuple[str, str, str], ...] = (
    (
        "The line is in the wrong place",
        "One number decides whether a case is handled by the model or by a person. jeval measures "
        "what that line is worth and where it belongs.",
        "This is the whole argument in one screen: the line in use, the accuracy it actually "
        "buys, and the line the cost minimum points at.",
    ),
    (
        "Move the line and watch the cost",
        "Every threshold has a price. Drag the line and the cost per case, the auto rate and the "
        "monthly bill follow — this is the decision, not a chart of one.",
        "Watch the two ends of the curve. Below the minimum, the cost is wrong decisions the "
        "model made on its own; above it, the cost is humans looking at cases the model could "
        "have handled.",
    ),
    (
        "Four questions, four different answers",
        "A request answers several questions at once. Pooled metrics hide the one that is lying, "
        "so jeval measures each question on its own.",
        "Same model, same day, four questions, and only some of them are overconfident. Pooled "
        "into a single number, the bad one disappears.",
    ),
    (
        "Find the segment that is worse",
        "The average hides who is being failed. Every segment is measured on its own, worst "
        "first, and a segment without enough labels says so instead of guessing.",
        "Sorted worst first. Click a bar and that segment gets its own reliability curve — the "
        "miscalibration usually lives in one slice, not everywhere.",
    ),
    (
        "The model changed and nobody noticed",
        "A notebook measured once. jeval keeps the measurement, compares it with the model that "
        "is serving now, and fails the build when the difference is real.",
        "Same question, same cost matrix, two model versions. The new one is worse, and this is "
        "the check that turns that into a red build instead of a support ticket.",
    ),
)


@dataclass
class Analysis:
    """Everything the five screens show, computed once."""

    dataset: DatasetReport
    blocks: list[template.ReliabilityBlock]
    result: Any
    impact: Any
    segments: Any
    segment_metrics: dict[str, Any]
    drift: DriftView | None
    failures: list[Any] = field(default_factory=list)
    before: Any = None
    after: Any = None
    # Per question, per model version: (question, ECE before, ECE after, n before, n after). Built
    # here rather than in the renderer because the drift engine compares slices, and a slice is not
    # a question: on a log with two versions of four questions the same question appears twice.
    comparison: tuple[tuple[str, float, float, int, int], ...] = ()
    records: Sequence[DecisionRecord] = field(default_factory=tuple)


def build_records() -> tuple[list[DecisionRecord], list[DecisionRecord]]:
    """The version in production, and the candidate that has to be judged against it.

    The production half is `jeval demo`'s own log at the same seed, so the first screen's numbers can
    be checked against the committed report in this repository. The candidate is the same four
    questions generated by a degraded model: overconfident on the question the cost matrix fires on,
    which is the change the drift check exists to catch.
    """
    serving = list(demo_dataset(seed=11, scale=0.5).records)
    degraded = (
        SynthSpec(
            n=310,
            mode="overconfident",
            # Strong enough that the drift gate fires with room to spare.
            exponent=0.35,
            question_key="department",
            model="jev-1.14.0",
            start=NEWER_START,
            seed=21,
            languages=("ko", "en"),
            tiers=("free", "pro"),
            label_fraction=0.86,
        ),
        SynthSpec(
            n=270,
            mode="inflated",
            inflation=1.3,
            question_key="intent",
            classes=("refund_request", "check_balance", "other"),
            model="jev-1.14.0",
            start=NEWER_START,
            seed=22,
            languages=("ko", "en"),
            label_fraction=0.9,
        ),
        SynthSpec(
            n=215,
            mode="calibrated",
            question_key="is_urgent",
            question_type="noul",
            model="jev-1.14.0",
            start=NEWER_START,
            seed=23,
            languages=("ko", "en"),
            label_fraction=0.88,
        ),
        SynthSpec(
            n=110,
            mode="inflated",
            question_key="satisfaction",
            question_type="score",
            score_bias=0.18,
            model="jev-1.14.0",
            start=NEWER_START,
            seed=24,
            label_fraction=0.8,
        ),
    )
    candidate = [record for spec in degraded for record in generate(spec)]
    return serving, candidate


def analyse(
    serving: Sequence[DecisionRecord], candidate: Sequence[DecisionRecord], costs_path: Path
) -> Analysis:
    """Run the same analysis `jeval report` runs, and keep every piece the screens need.

    Every screen but the last measures the version in production: a rollout decision is "here is
    what is live, here is what is proposed", and mixing the candidate into the reliability numbers
    would hide which of the two is miscalibrated.
    """
    records = list(serving)
    dataset = evaluate(
        records,
        n_bins=10,
        alpha=0.05,
        n_boot=200,
        by=("lang", "tier"),
        min_segment_size=30,
    )
    actions, _note = resolve_cost_actions(costs_path, costs_path.parent)
    thresholds, impacts = sweep_actions(
        actions,
        records,
        n_boot=200,
        alpha=0.05,
        monthly_volume=MONTHLY_VOLUME,
        currency=CURRENCY,
        current_threshold=CURRENT_THRESHOLD,
    )
    result = next((item for item in thresholds if item.curve), None)
    assert result is not None, "the demo cost matrix must produce a sweep"
    impact = impacts[result.action]
    segments, segment_metrics = segment_views(records, ("lang", "tier"), alpha=0.05, n_boot=200)
    assert segments is not None
    drift = drift_view_of([*records, *candidate])
    failures = list(run_checks(drift, parse_fail_on((FAIL_ON,)))) if drift is not None else []
    before = after = None
    comparison: list[tuple[str, float, float, int, int]] = []
    if drift is not None:
        before = _version_metrics([*records, *candidate], result.question, drift.baseline_label)
        after = _version_metrics([*records, *candidate], result.question, drift.current_label)
        for question in dataset.questions:
            before_metrics = _version_metrics(
                [*records, *candidate], question.question_key, drift.baseline_label
            )
            after_metrics = _version_metrics(
                [*records, *candidate], question.question_key, drift.current_label
            )
            if before_metrics.n and after_metrics.n:
                comparison.append(
                    (
                        question.question_key,
                        before_metrics.ece,
                        after_metrics.ece,
                        before_metrics.n,
                        after_metrics.n,
                    )
                )
    return Analysis(
        dataset=dataset,
        blocks=template.build_blocks(
            dataset,
            thresholds={item.question: item.threshold for item in thresholds},
            actions={item.question: item.action for item in thresholds},
        ),
        result=result,
        impact=impact,
        segments=segments,
        segment_metrics=segment_metrics,
        drift=drift,
        failures=failures,
        before=before,
        after=after,
        comparison=tuple(comparison),
        records=records,
    )


def drift_view_of(records: Sequence[DecisionRecord], *, min_slice: int = 30) -> DriftView | None:
    """Compare model versions when the log holds more than one."""
    if len({record.model for record in records}) < 2:
        return None
    from jeval import drift as drift_engine

    return drift_engine.compare(records, min_slice=min_slice, alpha=0.05, n_boot=200)


def _version_metrics(records: Sequence[DecisionRecord], question: str, model: str) -> Any:
    """One question's calibration, for a single model version."""
    subset = [
        record for record in records if record.question_key == question and record.model == model
    ]
    return evaluate(subset, n_bins=10, alpha=0.05, n_boot=200).overall


# --------------------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------------------
# Everything below is presentation. It reuses the report's own stylesheet as its base, so the
# dashboard cannot drift away from the artefact it is demonstrating, and adds only the shell: the
# sticky step rail, the fixed narration bar, and the live controls.
DASHBOARD_CSS = """\
body.dashboard main { max-width: 1180px; padding: 0 24px 150px; }
.dash-top {
  position: sticky; top: 0; z-index: 5; display: flex; flex-wrap: wrap; align-items: center;
  gap: 10px 16px; padding: 12px 24px; background: var(--panel); border-bottom: 1px solid var(--rule);
}
.dash-brand { font-size: 15px; font-weight: 700; letter-spacing: -0.02em; }
.dash-tag { font-size: 12px; color: var(--muted); }
.dash-steps {
  margin-left: auto; display: inline-flex; flex-wrap: wrap; gap: 2px; padding: 3px;
  border: 1px solid var(--rule); border-radius: 10px; background: var(--panel-alt);
}
.dash-step {
  -webkit-appearance: none; appearance: none; border: 0; background: none; cursor: pointer;
  color: var(--muted); font: inherit; font-size: 12.5px; padding: 6px 12px; border-radius: 7px;
}
.dash-step:hover { color: var(--ink); background: var(--panel); }
.dash-step[aria-selected="true"] { background: var(--panel); color: var(--ink); font-weight: 600; box-shadow: var(--shadow); }
.scene { padding: 20px 0 4px; }
.scene-head h1 { font-size: 30px; line-height: 1.2; letter-spacing: -0.02em; margin: 0 0 8px; }
.scene-head p { font-size: 16px; line-height: 1.6; color: var(--muted); margin: 0; max-width: 70ch; }
.scene h2 { margin-top: 30px; }
.panel { border: 1px solid var(--rule); border-radius: 12px; background: var(--panel); padding: 18px 20px; margin: 16px 0; }
.stat-row { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 22px; }
.stat-big .v { font-family: var(--mono); font-size: 26px; line-height: 1.15; font-weight: 600; margin-top: 6px; }
.live-row { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px 16px; margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--rule); }
.live-row .k { font-size: 12px; color: var(--muted); }
.live-row .v { font-family: var(--mono); font-size: 17px; font-weight: 600; margin-top: 4px; }
.control { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 10px 16px; }
.control .hint { font-size: 13px; color: var(--muted); }
.control .value { font-family: var(--mono); font-size: 22px; font-weight: 600; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 12.5px; font-weight: 600; }
.badge.fail { background: var(--alert); color: var(--bg); }
.badge.pass { background: var(--accent-soft); color: var(--accent); }
.facts { display: grid; gap: 8px; margin: 14px 0 0; padding: 0; list-style: none; font-size: 13.5px; color: var(--muted); }
.facts li { display: flex; align-items: baseline; gap: 10px; }
.facts b { color: var(--ink); font-family: var(--mono); font-weight: 600; }
.dash-caption {
  position: fixed; left: 0; right: 0; bottom: 0; z-index: 6; display: flex; align-items: baseline;
  gap: 18px; padding: 16px 24px; background: var(--ink); color: var(--bg);
  font-size: 14.5px; line-height: 1.5;
}
.dash-caption .count { font-family: var(--mono); font-size: 12.5px; opacity: 0.65; white-space: nowrap; }
.dash-caption p { margin: 0; max-width: 100ch; }
body:not(.stepper) .dash-caption { display: none; }
/* The bar inverts the page in light mode, which is the loudest thing on screen and exactly right
   for the line being read aloud. Inverted in dark mode it would be a white slab on a dark page, so
   there it becomes the page's own raised surface instead. */
@media (prefers-color-scheme: dark) {
  .dash-caption { background: var(--panel-alt); color: var(--ink); border-top: 1px solid var(--rule); }
}
.sequences { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 22px; margin-top: 6px; }
body.dashboard figure { max-width: 800px; }
/* The per-question screen carries a tab row, a diagnosis and the measured interval as well as the
   curve, so its chart is sized to leave room for the text that explains it. */
#scene-3 figure { max-width: 600px; }
.split { display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 28px; align-items: start; }
.split > figure { max-width: none; }
.split .panel { margin-top: 0; position: sticky; top: 84px; }
@media (max-width: 900px) {
  .split { grid-template-columns: 1fr; }
  .split .panel { position: static; }
}
@media (max-width: 760px) {
  .stat-row, .live-row { grid-template-columns: 1fr 1fr; }
  .scene-head h1 { font-size: 24px; }
  .dash-caption { position: static; display: block; color: var(--muted); background: none; padding: 16px 0; }
}
"""

# The stepper, the live readout and the click-to-reveal behaviour. Kept deliberately small: every
# chart is server-rendered, so nothing here draws. `amount`, `money` and `pct` are the report's own
# readout formats, restated here because the script is the presenter, not the report.
DASHBOARD_JS = """\
(function () {
  "use strict";

  var doc = document;
  var body = doc.body;

  function all(selector, root) {
    return Array.prototype.slice.call((root || doc).querySelectorAll(selector));
  }

  function closest(node, selector) {
    if (node && node.closest) { return node.closest(selector); }
    return null;
  }

  function finite(value) {
    var number = Number(value);
    return isFinite(number) ? number : null;
  }

  function setText(node, text) {
    if (node && node.textContent !== text) { node.textContent = text; }
  }

  function grouped(value) {
    var text = String(Math.round(value));
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
    var parts = Math.abs(value).toFixed(digits === undefined ? 2 : digits).split(".");
    var whole = parts[0];
    var out = "";
    var count = 0;
    for (var i = whole.length - 1; i >= 0; i -= 1) {
      out = whole.charAt(i) + out;
      count += 1;
      if (count % 3 === 0 && i > 0) { out = "," + out; }
    }
    if (parts.length > 1) { out = out + "." + parts[1]; }
    return (value < 0 ? "-" : "") + out;
  }

  function money(value) {
    var magnitude = Math.abs(value);
    if (magnitude >= 1e9) { return (value / 1e9).toFixed(1) + "B"; }
    if (magnitude >= 1e6) { return (value / 1e6).toFixed(1) + "M"; }
    if (magnitude >= 1e3) { return (value / 1e3).toFixed(1) + "k"; }
    return grouped(value);
  }

  function pct(value) {
    return Math.round(value * 100) + "%";
  }

  function withCurrency(text, currency) {
    return currency ? currency + " " + text : text;
  }

  function readData() {
    var node = doc.getElementById("jeval-demo-data");
    if (!node) { return {}; }
    try {
      return JSON.parse(node.textContent || "{}") || {};
    } catch (error) {
      return {};
    }
  }

  var data = readData();

  // --- scenes -------------------------------------------------------------------------
  var scenes = all("[data-scene]");
  var steps = all("[data-step]");
  var caption = doc.querySelector("[data-caption]");
  var counter = doc.querySelector("[data-count]");
  var current = 0;

  function show(index) {
    if (!scenes.length) { return; }
    current = (index + scenes.length) % scenes.length;
    body.classList.add("stepper");
    for (var i = 0; i < scenes.length; i += 1) {
      var on = i === current;
      scenes[i].hidden = !on;
      scenes[i].classList.toggle("is-active", on);
    }
    for (var j = 0; j < steps.length; j += 1) {
      var selected = j === current;
      steps[j].setAttribute("aria-selected", selected ? "true" : "false");
      steps[j].setAttribute("tabindex", selected ? "0" : "-1");
    }
    setText(caption, scenes[current].getAttribute("data-narration") || "");
    setText(counter, (current + 1) + " / " + scenes.length);
    if (doc.documentElement) {
      history.replaceState(null, "", "#s" + (current + 1));
    }
    window.scrollTo(0, 0);
  }

  var initial = (window.location.hash || "").match(/^#s([0-9]+)$/);
  show(initial ? Number(initial[1]) - 1 : 0);

  doc.addEventListener("keydown", function (event) {
    if (event.key === "ArrowRight") { event.preventDefault(); show(current + 1); }
    if (event.key === "ArrowLeft") { event.preventDefault(); show(current - 1); }
  });

  doc.addEventListener("click", function (event) {
    var step = closest(event.target, "[data-step]");
    if (step) { show(Number(step.getAttribute("data-step")) - 1); return; }
    var tab = closest(event.target, "[data-question-tab]");
    if (tab) { activateQuestion(tab.getAttribute("data-question-tab")); return; }
    var bar = closest(event.target, "[data-target]");
    if (bar) { toggleTarget(bar); }
  });

  // --- per-question panels ------------------------------------------------------------
  function activateQuestion(key) {
    var tabs = all("[data-question-tab]");
    var panels = all("[data-question-panel]");
    for (var i = 0; i < tabs.length; i += 1) {
      tabs[i].setAttribute("aria-selected", tabs[i].getAttribute("data-question-tab") === key ? "true" : "false");
    }
    for (var j = 0; j < panels.length; j += 1) {
      panels[j].hidden = panels[j].getAttribute("data-question-panel") !== key;
    }
  }

  function toggleTarget(bar) {
    var target = doc.getElementById(bar.getAttribute("data-target") || "");
    if (!target) { return; }
    var opening = target.hidden;
    target.hidden = !opening;
    bar.setAttribute("aria-expanded", opening ? "true" : "false");
  }

  var firstTab = doc.querySelector("[data-question-tab]");
  if (firstTab) { activateQuestion(firstTab.getAttribute("data-question-tab")); }

  // --- the live threshold -------------------------------------------------------------
  function initSlider() {
    var box = doc.querySelector("[data-live]");
    if (!box) { return; }
    var slider = box.querySelector('input[type="range"]');
    var cursor = doc.getElementById("cost-cursor");
    var points = data.points || [];
    var anchors = data.cursor || [];
    if (!slider || !points.length || !anchors.length) { return; }
    var currency = data.currency || "";
    var volume = data.volume || 0;

    function nearest(value) {
      var best = 0;
      var gap = Infinity;
      for (var i = 0; i < points.length; i += 1) {
        var distance = Math.abs(points[i].t - value);
        if (distance < gap) { gap = distance; best = i; }
      }
      return best;
    }

    function render() {
      var threshold = finite(slider.value);
      if (threshold === null) { return; }
      setText(box.querySelector('[data-value="threshold"]'), threshold.toFixed(2));
      var index = nearest(threshold);
      var point = points[index];
      setText(box.querySelector('[data-value="cost_per_case"]'), withCurrency(amount(point.cost, 2), currency));
      setText(box.querySelector('[data-value="cost_per_month"]'), withCurrency(money(point.cost * volume), currency));
      setText(box.querySelector('[data-value="auto_rate"]'), point.auto === null ? "n/a" : pct(point.auto));
      setText(box.querySelector('[data-value="accuracy_auto"]'), point.acc === null ? "n/a" : pct(point.acc));
      setText(
        box.querySelector('[data-value="escalations_per_month"]'),
        point.auto === null ? "n/a" : money((1 - point.auto) * volume)
      );
      if (cursor && anchors[index]) {
        cursor.setAttribute(
          "transform",
          "translate(" + (anchors[index].x - anchors[0].x) + " " + (anchors[index].y - anchors[0].y) + ")"
        );
      }
    }

    slider.addEventListener("input", render);
    slider.addEventListener("change", render);
    render();
  }

  initSlider();
})();
"""


def _stat(label: str, value: str, sub: str = "") -> str:
    body = f'<div class="stat stat-big"><div class="k">{S.escape(label)}</div>'
    body += f'<div class="v">{S.escape(value)}</div>'
    if sub:
        body += f'<div class="s">{S.escape(sub)}</div>'
    return body + "</div>"


def _impact_table(impact: ImpactTable) -> str:
    head = (
        '<tr><th>metric</th><th class="num">current</th>'
        '<th class="num">recommended</th><th class="num">change</th></tr>'
    )
    rows = "".join(
        f'<tr><td>{S.escape(row.label)}</td><td class="num">{S.escape(row.current)}</td>'
        f'<td class="num">{S.escape(row.recommended)}</td>'
        f'<td class="num">{S.escape(row.change)}</td></tr>'
        for row in impact.rows
    )
    return f'<table class="impact"><thead>{head}</thead><tbody>{rows}</tbody></table>'


def _cost_cursor(analysis: Analysis) -> tuple[str, list[dict[str, float]], list[dict[str, float]]]:
    """The draggable marker, and the two coordinate lists the script needs to move it.

    The scale is recomputed here from the chart's own constants: the cursor has to sit exactly on
    the curve, so it uses the same domain and the same padding the chart drew with rather than an
    approximation of it.
    """
    points = [
        point for point in analysis.result.curve if point.expected_cost == point.expected_cost
    ]

    def number(value: float) -> float | None:
        """JSON has no NaN, and a JavaScript parser rejects one outright.

        At threshold 1.00 nothing is auto-accepted, so the auto-branch accuracy is undefined. It is
        emitted as null and read as "n/a", exactly as the report's own readout does.
        """
        return round(value, 6) if value == value else None

    costs = [point.expected_cost for point in points]
    lo, hi = min(costs), max(costs)
    pad = max(1e-9, (hi - lo) * 0.12)
    y_lo, y_hi = max(0.0, lo - pad), hi + pad
    t_lo = min(point.threshold for point in points)
    t_hi = max(point.threshold for point in points)
    x0, x1 = cost_charts.LEFT, cost_charts.WIDTH - cost_charts.RIGHT
    y_bottom, y_top = cost_charts.HEIGHT - cost_charts.BOTTOM, cost_charts.TOP

    def px(value: float) -> float:
        return x0 + (value - t_lo) / (t_hi - t_lo) * (x1 - x0) if t_hi > t_lo else x0

    def py(value: float) -> float:
        return y_bottom + (value - y_lo) / (y_hi - y_lo) * (y_top - y_bottom)

    anchors = [
        {
            "t": round(point.threshold, 6),
            "x": round(px(point.threshold), 2),
            "y": round(py(point.expected_cost), 2),
        }
        for point in points
    ]
    live = [
        {
            "t": round(point.threshold, 6),
            "cost": number(point.expected_cost),
            "auto": number(point.auto_rate),
            "acc": number(point.accuracy_auto),
        }
        for point in points
    ]
    start = next(
        (anchor for anchor in anchors if abs(anchor["t"] - analysis.result.threshold) < 1e-9),
        anchors[0],
    )
    cursor = (
        '<g id="cost-cursor">'
        f'<line x1="{start["x"]:.2f}" y1="{y_top:.2f}" x2="{start["x"]:.2f}" y2="{y_bottom:.2f}" '
        'stroke="#2f5fd0" stroke-width="1.2" stroke-dasharray="3 3" class="accent-stroke"/>'
        f'<circle cx="{start["x"]:.2f}" cy="{start["y"]:.2f}" r="4.6" fill="#2f5fd0" '
        'class="accent-dot"/>'
        "</g>"
    )
    return cursor, anchors, live


def _scene_verdict(analysis: Analysis) -> str:
    verdict = build_verdict(
        analysis.dataset.overall,
        threshold=analysis.result,
        current_threshold=CURRENT_THRESHOLD,
        recommended_threshold=analysis.result.threshold,
    )
    stats = "".join(_stat(stat.label, stat.value, stat.sub) for stat in verdict.stats)
    return (
        '<div class="scene-head">'
        f"<h1>{S.escape(SCENES[0][0])}</h1><p>{S.escape(SCENES[0][1])}</p></div>"
        f'<div class="panel verdict" id="verdict" data-status="{S.escape(verdict.status)}">'
        f'<p class="headline">{S.escape(verdict.headline)}</p>'
        f'<p class="detail">{S.escape(verdict.detail)}</p>'
        f'<div class="stat-row">{stats}</div>'
        "</div>"
        '<div class="panel"><h3>What changes if you move the line</h3>'
        f"{_impact_table(analysis.impact)}"
        '<p class="note">Measured on '
        f"{analysis.result.n_records:,} labeled decisions for "
        f"<code>{S.escape(analysis.result.action)}</code>, at "
        f"{CURRENCY} {MONTHLY_VOLUME:,.0f} decisions a month.</p></div>"
    )


def _scene_cost(analysis: Analysis) -> str:
    result = analysis.result
    chart = cost_charts.render_cost_curve(
        result, current_threshold=CURRENT_THRESHOLD, currency=CURRENCY
    )
    cursor, _anchors, _live = _cost_cursor(analysis)
    chart = chart.replace("</svg>", cursor + "</svg>")
    per_case = result.expected_cost_per_case
    rows = (
        ("cost per case", "cost_per_case", f"{CURRENCY} {per_case:,.2f}"),
        ("cost per month", "cost_per_month", f"{CURRENCY} {S.money(per_case * MONTHLY_VOLUME)}"),
        ("auto rate", "auto_rate", f"{result.auto_rate:.0%}"),
        ("accuracy (auto)", "accuracy_auto", f"{result.accuracy_auto:.0%}"),
        (
            "escalations per month",
            "escalations_per_month",
            S.money((1 - result.auto_rate) * MONTHLY_VOLUME),
        ),
    )
    live = "".join(
        f'<div><div class="k">{S.escape(label)}</div>'
        f'<div class="v" data-value="{key}">{S.escape(value)}</div></div>'
        for label, key, value in rows
    )
    return (
        '<div class="scene-head">'
        f"<h1>{S.escape(SCENES[1][0])}</h1><p>{S.escape(SCENES[1][1])}</p></div>"
        '<div class="split">'
        f"<figure>{chart}</figure>"
        f'<div class="panel" data-live data-currency="{S.escape(CURRENCY)}">'
        '<div class="control"><label for="threshold-live">try another threshold</label>'
        f'<span class="value" data-value="threshold">{result.threshold:.2f}</span></div>'
        '<input type="range" id="threshold-live" min="0" max="1" step="0.01" '
        f'value="{result.threshold:.2f}">'
        '<div class="scale"><span>0.00</span><span>1.00</span></div>'
        f'<div class="live-row">{live}</div>'
        '<p class="note">The report never writes <code>thresholds.yaml</code>: the slider is '
        "exploration, and confirming a value is the job of <code>jeval threshold</code>.</p></div>"
        "</div>"
    )


def _scene_reliability(analysis: Analysis) -> str:
    tabs = "".join(
        '<button type="button" role="tab" class="tab" data-question-tab="'
        + S.escape(block.question_key)
        + '" aria-selected="'
        + ("true" if index == 0 else "false")
        + f'">{S.escape(block.question_key)}</button>'
        for index, block in enumerate(analysis.blocks)
    )
    panels = "".join(
        f'<div class="question-block{" is-active" if index == 0 else ""}" '
        f'data-question-panel="{S.escape(block.question_key)}">'
        f'<div class="block-head"><h3>{S.escape(block.question_key)}</h3>'
        f'<p class="block-sub">{S.escape(block.subtitle)}</p></div>'
        f'<p class="diag">{S.escape(block.diagnosis)}</p>'
        f"{reliability_charts.render_reliability_section(block.metrics, threshold=block.threshold)}"
        f'<ul class="facts"><li><span>ECE</span><b>{S.fmt(block.metrics.ece, 3)}</b>'
        f"<span>95% interval {S.fmt(block.metrics.ece_ci_low, 3)}–{S.fmt(block.metrics.ece_ci_high, 3)}</span>"
        f"<span>n={block.metrics.n:,} labeled</span></li></ul>"
        "</div>"
        for index, block in enumerate(analysis.blocks)
    )
    return (
        '<div class="scene-head">'
        f"<h1>{S.escape(SCENES[2][0])}</h1><p>{S.escape(SCENES[2][1])}</p></div>"
        f'<div class="tabs" role="tablist">{tabs}</div>{panels}'
    )


def _scene_segments(analysis: Analysis) -> str:
    view = analysis.segments
    chart = segment_charts.render_segments(view)
    figures = ""
    for bar in view.bars:
        if bar.too_few_samples:
            continue
        slug = segment_charts.segment_slug(bar.key, bar.value)
        metrics = analysis.segment_metrics.get(slug)
        if metrics is None:
            continue
        figures += (
            f'<figure id="{S.escape(segment_charts.segment_target(bar))}" hidden>'
            f"{reliability_charts.render_reliability(metrics, title=bar.label)}"
            f"<figcaption>{S.escape(bar.label)} on its own: ECE "
            f"{S.fmt(metrics.ece, 3)} over {metrics.n:,} labeled decisions.</figcaption>"
            "</figure>"
        )
    return (
        '<div class="scene-head">'
        f"<h1>{S.escape(SCENES[3][0])}</h1><p>{S.escape(SCENES[3][1])}</p></div>"
        f"<figure>{chart}</figure>{figures}"
    )


def _scene_drift(analysis: Analysis) -> str:
    view = analysis.drift
    if view is None:
        return (
            '<div class="scene-head">'
            f"<h1>{S.escape(SCENES[4][0])}</h1><p>{S.escape(SCENES[4][1])}</p></div>"
            '<p class="note">This log holds a single model version, so there is nothing to '
            "compare.</p>"
        )
    limit = analysis.failures[0].limit if analysis.failures else 0.05
    checks = sorted({failure.check for failure in analysis.failures})
    failed = {failure.detail.split(":", 1)[0] for failure in analysis.failures}
    badge = (
        f'<span class="badge fail">FAIL · {S.escape(", ".join(checks))} over {S.fmt(limit, 3)}</span>'
        if analysis.failures
        else '<span class="badge pass">within the drift limit</span>'
    )
    changed = (
        ", ".join(f"{change.label} on {change.at}" for change in view.changes)
        or "no model change detected"
    )
    # One row per question. A slice table would list the same question twice, once per model
    # version, which is exactly the reading this screen exists to make unnecessary.
    rows = "".join(
        f"<li><span>{S.escape(question)}</span>"
        f"<b>{S.fmt(before_ece, 3)} → {S.fmt(after_ece, 3)}</b>"
        f"<span>{after_ece - before_ece:+.3f}</span>"
        f"<span>n {n_before}/{n_after}</span>"
        f'<span class="badge {"fail" if question in failed else "pass"}">'
        f"{'FAIL' if question in failed else 'within budget'}</span></li>"
        for question, before_ece, after_ece, n_before, n_after in analysis.comparison
    )
    overlay = (
        drift_charts.render_overlay(
            analysis.before,
            analysis.after,
            baseline_label=view.baseline_label,
            current_label=view.current_label,
            threshold=analysis.result.threshold,
        )
        if analysis.before is not None and analysis.after is not None
        else drift_charts.render_ece_series(view)
    )
    return (
        '<div class="scene-head">'
        f"<h1>{S.escape(SCENES[4][0])}</h1><p>{S.escape(SCENES[4][1])}</p></div>"
        '<div class="split">'
        f"<figure>{overlay}<figcaption>Stated confidence against observed accuracy, before and "
        f"after the change, for {S.escape(analysis.result.question)}. Two curves pulling apart is "
        "the drift; the gap between them is what the check measures.</figcaption></figure>"
        f'<div class="panel">{badge}'
        f'<p class="note">{S.escape(changed)}. Every number below comes from the same measurement: '
        "ECE before and after, per question, on the labeled records of each version.</p>"
        f'<ul class="facts">{rows}</ul>'
        '<p class="note">The gate measures the <em>change</em> between versions. '
        "<code>is_urgent</code> is far worse than the others in both of them, which is a different "
        "problem — that is what the <code>ece-above</code> level check is for.</p>"
        "</div></div>"
    )


def render_dashboard(analysis: Analysis, *, provenance: str) -> str:
    """Render the whole dashboard as one self-contained document."""
    _cursor, anchors, live = _cost_cursor(analysis)
    payload = json.dumps(
        {
            "action": analysis.result.action,
            "currency": CURRENCY,
            "volume": MONTHLY_VOLUME,
            "threshold": round(analysis.result.threshold, 6),
            "current": CURRENT_THRESHOLD,
            "points": live,
            "cursor": anchors,
        },
        allow_nan=False,
    ).replace("<", "\\u003c")
    steps = "".join(
        f'<button type="button" role="tab" class="dash-step" data-step="{index + 1}" '
        f'aria-selected="{"true" if index == 0 else "false"}">{index + 1}. {S.escape(title)}</button>'
        for index, (title, _claim, _narration) in enumerate(SCENES)
    )
    scenes = "".join(
        f'<section class="scene" id="scene-{index + 1}" data-scene="{index + 1}" '
        f'data-narration="{S.escape(SCENES[index][2])}">{body}</section>'
        for index, body in enumerate(
            (
                _scene_verdict(analysis),
                _scene_cost(analysis),
                _scene_reliability(analysis),
                _scene_segments(analysis),
                _scene_drift(analysis),
            )
        )
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>jeval · demo dashboard</title>"
        f"<style>{minify(REPORT_CSS + DASHBOARD_CSS)}</style></head>"
        '<body class="dashboard">'
        '<header class="dash-top"><span class="dash-brand">jeval</span>'
        f'<span class="dash-tag">{S.escape(provenance)}</span>'
        f'<nav class="dash-steps" role="tablist">{steps}</nav></header>'
        f"<main>{scenes}</main>"
        '<footer class="dash-caption"><span class="count" data-count>1 / '
        f"{len(SCENES)}</span><p data-caption>{S.escape(SCENES[0][2])}</p></footer>"
        f'<script type="application/json" id="jeval-demo-data">{payload}</script>'
        f"<script>{minify(DASHBOARD_JS)}</script>"
        "</body></html>"
    )


def build_document() -> tuple[str, Analysis]:
    """The whole dashboard, plus the analysis behind it.

    ``main`` writes what this returns, and ``tests/test_demo_dashboard.py`` rebuilds it to check the
    committed file still matches its inputs. The build is deterministic — seeded synthesis, no clock
    in the document — which is what makes that check an equality rather than a property.
    """
    serving, candidate = build_records()
    # The cost matrix lives in this file, not beside the artefact: the script is the provenance, and
    # a stray YAML in examples/ would look like a runnable project instead of one demo's inputs.
    with tempfile.TemporaryDirectory(prefix="jeval-dashboard-") as work:
        costs_path = Path(work) / "costs.yaml"
        costs_path.write_text(COSTS_YAML, encoding="utf-8")
        analysis = analyse(serving, candidate, costs_path)
    models = sorted({record.model for record in [*serving, *candidate]})
    provenance = (
        f"synthetic demo · {len(serving):,} decisions in production "
        f"({analysis.dataset.n_labeled_gold:,} labeled, {len(analysis.dataset.questions)} questions) · "
        f"candidate {models[-1]}"
    )
    return render_dashboard(analysis, provenance=provenance), analysis


def main(argv: list[str]) -> int:
    out = Path(argv[1]) if len(argv) > 1 else Path("examples/demo-dashboard.html")
    html, analysis = build_document()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    print(
        f"recommended threshold {analysis.result.threshold:.2f} "
        f"(95% CI {analysis.result.ci_low:.2f}-{analysis.result.ci_high:.2f}), "
        f"current {CURRENT_THRESHOLD:.2f}"
    )
    for failure in analysis.failures:
        print(f"drift failure: {failure.check} {failure.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
