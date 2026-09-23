"""The one-screen dashboard, and the recording it exists for.

It is not a product surface -- the report is the product, and a dashboard is explicitly out of scope
-- but it is a committed artefact that puts a dozen numbers on screen, so it is held to the standards
of one: one self-contained file, nothing from the machine that built it, and every figure computed
rather than typed. It can be held to one more thing the report cannot, because nothing in it depends
on the clock or on an unseeded draw: the committed file is byte-identical to a fresh build, so what
is on screen in a recording cannot quietly stop matching the script.

The other thing worth guarding is the argument. One screen makes one claim, and the claim has to
survive the data: the screen draws the line that is in use and the line the cost minimum points at,
says both in words, and puts the answer in the largest type on the page. Those are the properties a
"calmer design" would quietly lose.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "examples" / "make-demo-dashboard.py"
ARTIFACT = REPO / "examples" / "demo-dashboard.html"


def _script() -> ModuleType:
    """Load the example script the way a reader runs it: as a file, not as an installed module."""
    spec = importlib.util.spec_from_file_location("make_demo_dashboard", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _artifact() -> str:
    assert ARTIFACT.exists(), f"the dashboard is missing: {ARTIFACT.relative_to(REPO)}"
    return ARTIFACT.read_text(encoding="utf-8")


def test_the_committed_dashboard_is_what_the_script_builds() -> None:
    html, _screen = _script().build_document()
    assert html == _artifact(), (
        "examples/demo-dashboard.html no longer matches examples/make-demo-dashboard.py. "
        "Rebuild it: uv run python examples/make-demo-dashboard.py examples/demo-dashboard.html"
    )


def test_the_dashboard_is_one_self_contained_file() -> None:
    html = _artifact()
    assert html.startswith("<!doctype html>")
    assert len(html.encode("utf-8")) < 1_048_576
    for forbidden in ("<script", 'src="http', 'href="http', "@import", "<link", "url(http"):
        assert forbidden not in html, f"the dashboard reaches outside itself: {forbidden}"


def test_the_dashboard_carries_nothing_from_this_machine() -> None:
    html = _artifact()
    for token in ("/Users/", "khope", "@sionic", "Desktop", "/home/", "TemporaryDirectory"):
        assert token not in html, f"the dashboard leaks {token!r}"


def test_one_screen_makes_one_argument() -> None:
    """A single claim, a single page, and no navigation to lose the reader in."""
    html = _artifact()
    assert html.count("<h1>") == 1, "one screen carries one headline"
    for removed in ("data-scene", "data-step", "data-narration", "data-caption"):
        assert removed not in html, f"the walkthrough's {removed!r} is back"


def test_both_thresholds_are_drawn_and_named() -> None:
    """The claim is about a line, so the line has to be on the chart -- and labelled in words."""
    html, screen = _script().build_document()
    current = screen.impact.current_threshold
    belongs = screen.result.threshold
    assert current != belongs, "the demo is pointless if the two thresholds agree"
    for label, text in (
        ("in use", f"in use {current:.2f}"),
        ("where it belongs", f"where it belongs {belongs:.2f}"),
    ):
        assert text in html, f"the chart does not say {label} ({text})"
    # Two vertical lines and a labelled minimum: one per threshold, one at the cost minimum.
    assert html.count('class="inuse-line"') == 1
    assert html.count('class="belongs-line"') == 1
    assert html.count('class="min-dot"') == 1
    # The confidence distribution is what makes the line concrete.
    assert 'class="dist"' in html


def test_the_answer_is_the_loudest_number_on_the_page() -> None:
    """Hierarchy, guarded: the recommendation is not one of three equal statistics."""
    html, screen = _script().build_document()
    assert f'<div class="big">{screen.result.threshold:.2f}</div>' in html
    assert 'class="eyebrow">where the line belongs' in html
    # ... and the line in use is deliberately quieter, in the supporting table and the chip.
    assert 'class="chip">in use' in html


def test_the_numbers_on_screen_are_the_computed_ones() -> None:
    html, screen = _script().build_document()
    assert f"{screen.result.expected_cost_per_case:,.2f}" in html
    assert f"n={screen.result.n_records:,}" in html
    assert f"{screen.metrics.n:,} labeled decisions" in html
    for row in screen.impact.rows:
        assert row.recommended in html, f"the impact table lost {row.label!r}"
    for failure in screen.failures:
        # The detail is escaped in the markup, so the check name and the limit are what to assert.
        assert "fails the drift gate" in html, "the drift caveat is not on the page"
        assert failure.check in html, "the drift gate's failing check is not on the page"
        assert f"over the {failure.limit:.3f} limit" in html
