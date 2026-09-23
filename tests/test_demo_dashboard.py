"""The demo dashboard, and the walkthrough it exists for.

It is not a product surface -- the report is the product, and a dashboard is explicitly out of scope
-- but it is a committed artefact that puts a dozen numbers on screen, so it is held to the standards
of one: one self-contained file, nothing from the machine that built it, and every figure computed
rather than typed. It can be held to one more thing the report cannot, because nothing in it depends
on the clock or on an unseeded draw: the committed file is byte-identical to a fresh build, so the
screenshots in the README cannot quietly stop matching the script.
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
    html, _analysis = _script().build_document()
    assert html == _artifact(), (
        "examples/demo-dashboard.html no longer matches examples/make-demo-dashboard.py. "
        "Rebuild it: uv run python examples/make-demo-dashboard.py examples/demo-dashboard.html"
    )


def test_the_dashboard_is_one_self_contained_file() -> None:
    html = _artifact()
    assert html.startswith("<!doctype html>")
    assert len(html.encode("utf-8")) < 1_048_576
    for forbidden in ("<script src", 'src="http', 'href="http', "@import", "<link"):
        assert forbidden not in html, f"the dashboard reaches outside itself: {forbidden}"
    # A clip path is the one url() reference, and it is local to the document.
    assert 'clip-path="url(#' in html
    assert "url(#" in html and "url(http" not in html


def test_the_dashboard_carries_nothing_from_this_machine() -> None:
    html = _artifact()
    for token in ("/Users/", "khope", "@sionic", "Desktop", "/home/", "TemporaryDirectory"):
        assert token not in html, f"the dashboard leaks {token!r}"


def test_every_screenshot_in_the_walkthrough_has_its_words() -> None:
    """Five steps, five surfaces: the script's narration is what the video is recorded to."""
    html = _artifact()
    assert html.count('data-scene="') == 5
    for index in range(1, 6):
        assert f'id="scene-{index}"' in html
    assert html.count("<p data-caption>") == 1
    assert html.count('data-narration="') == 5
    # Every step's control that the recording touches is present.
    for control in (
        "threshold-live",
        "cost-cursor",
        "data-question-tab",
        "data-target",
        "data-step",
    ):
        assert control in html, f"the walkthrough lost its {control!r} control"


def test_the_numbers_on_screen_are_the_computed_ones() -> None:
    """Spot-check the three numbers the narration points at, against a fresh analysis."""
    html, analysis = _script().build_document()
    assert f"{analysis.result.threshold:.2f}" in html
    assert f"{analysis.result.expected_cost_per_case:,.2f}" in html
    for failure in analysis.failures:
        question = failure.detail.split(":", 1)[0]
        row = [line for line in html.split("<li>") if line.startswith(f"<span>{question}</span>")]
        assert row, f"the drift panel does not show the question the gate failed on: {question}"
        assert "FAIL" in row[0]
