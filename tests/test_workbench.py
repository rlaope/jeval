"""The workbench: the one-argument screen and the full report on one page.

It is not a product surface -- the report is the product, and both a dashboard and a second
presentation of it are out of scope for the tool -- but it is a committed artefact that puts a
dozen computed numbers on screen, so it is held to the standards of one: one self-contained file,
nothing from the machine that built it, and byte-identical to a fresh build, so what is on screen
cannot quietly stop matching the two surfaces it claims to present.

What makes the workbench worth guarding beyond the screen it borrows is the claim it makes about
itself: that it *contains* both surfaces rather than pointing at them. The screen is rendered by
``examples/make-demo-dashboard.py`` at build time, and the report is
``examples/report-example.html`` embedded through ``srcdoc``. If either were replaced by a link, a
copy, or a stale paste, the page would still open and still look right -- so the tests below prove
the embedding by looking for text that exists only inside the report, prove the screen's argument
survives the framing, and prove the extra chrome the workbench adds stays small enough to keep the
argument in the first viewport. Pixel heights belong to a browser check, not to pytest; here the
guard is that the stylesheet is the screen's own, and that the report is folded away below it.
"""

from __future__ import annotations

import importlib.util
import sys
from html import escape
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "examples" / "make-workbench.py"
ARTIFACT = REPO / "examples" / "workbench.html"
REPORT = REPO / "examples" / "report-example.html"
SCREEN_SCRIPT = REPO / "examples" / "make-demo-dashboard.py"

# A phrase that occurs in the report and in neither the screen nor the workbench's own chrome. Its
# presence, exactly once, is the proof that the report body itself was embedded.
REPORT_ONLY_SIGNATURE = "recommended 0.85 (auto_route) · in use 0.60"


def _load(name: str, path: Path) -> ModuleType:
    """Load an example script the way a reader runs it: as a file, not as an installed module."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _script() -> ModuleType:
    return _load("make_workbench", SCRIPT)


def _artifact() -> str:
    assert ARTIFACT.exists(), f"the workbench is missing: {ARTIFACT.relative_to(REPO)}"
    return ARTIFACT.read_text(encoding="utf-8")


def test_the_committed_workbench_is_what_the_script_builds() -> None:
    assert _script().build_document() == _artifact(), (
        "examples/workbench.html no longer matches examples/make-workbench.py (or one of the two "
        "surfaces it embeds). Rebuild it: "
        "uv run python examples/make-workbench.py examples/workbench.html"
    )


def test_the_workbench_is_one_self_contained_file() -> None:
    html = _artifact()
    assert html.startswith("<!doctype html>")
    assert len(html.encode("utf-8")) < 1_572_864, "the workbench has grown past 1.5 MB"
    for forbidden in ("<script", 'src="http', 'href="http', "@import", "<link", "url(http"):
        assert forbidden not in html, f"the workbench reaches outside itself: {forbidden}"


def test_the_workbench_carries_nothing_from_this_machine() -> None:
    html = _artifact()
    for token in ("/Users/", "khope", "@sionic", "Desktop", "/home/", "TemporaryDirectory"):
        assert token not in html, f"the workbench leaks {token!r}"


def test_the_report_is_embedded_not_referenced() -> None:
    """The report is inside the page, as the committed file, not a link and not a stale copy."""
    html = _artifact()
    assert html.count("srcdoc=") == 1, "the report is framed exactly once, through srcdoc"
    # The signature is the report's alone: absent from the screen and from the workbench's chrome,
    # so finding it once means the report body is really here, and only here.
    report = REPORT.read_text(encoding="utf-8")
    assert report.count(REPORT_ONLY_SIGNATURE) == 1, "the signature is no longer report-only"
    screen_html, _screen = _load("make_demo_dashboard", SCREEN_SCRIPT).build_document()
    assert REPORT_ONLY_SIGNATURE not in screen_html, "the signature is no longer report-only"
    assert html.count(REPORT_ONLY_SIGNATURE) == 1, "the report body is not embedded in the page"
    # The embedding is the committed file, whole, in its escaped form -- not a paste of parts.
    assert escape(report) in html, "the framed report is not examples/report-example.html"
    # The frame never fetches: an inline document has no src to fetch.
    assert "<iframe src=" not in html
    assert " src=" not in html.split("<iframe", 1)[1].split(">", 1)[0]


def test_both_surfaces_are_present_and_navigable() -> None:
    """An index at the top, one anchor per surface, and both anchors resolve."""
    html = _artifact()
    assert html.count('<nav class="index"') == 1
    assert html.count('href="#screen"') == 1
    assert html.count('href="#report"') == 1
    assert html.count('id="screen"') == 1
    assert html.count('id="report"') == 1
    # The index precedes both surfaces; the screen precedes the report.
    index_at = html.index('<nav class="index"')
    screen_at = html.index('<section id="screen">')
    report_at = html.index('<details class="report" id="report">')
    assert index_at < screen_at < report_at


def test_the_screen_still_makes_its_one_argument() -> None:
    """Framing the screen must not blur the claim: one headline, both lines drawn and named."""
    html = _artifact()
    assert html.count("<h1>") == 1, "one page carries one headline; the report's are in the frame"
    _screen_html, screen = _load("make_demo_dashboard", SCREEN_SCRIPT).build_document()
    current = screen.impact.current_threshold
    belongs = screen.result.threshold
    assert current != belongs, "the demo is pointless if the two thresholds agree"
    assert (
        f'<h1>The line belongs at <span class="answer">{belongs:.2f}</span>, not {current:.2f}.</h1>'
        in html
    )
    for label, text in (
        ("in use", f"in use {current:.2f}"),
        ("where it belongs", f"where it belongs {belongs:.2f}"),
    ):
        assert text in html, f"the chart does not say {label} ({text})"
    assert html.count('class="inuse-line"') == 1
    assert html.count('class="belongs-line"') == 1
    # The drift gate is answered by a word in a badge, never by colour alone.
    assert 'class="badge fail">FAIL</span>' in html or 'class="badge pass">PASS</span>' in html


def test_the_workbench_keeps_the_screen_in_the_first_viewport() -> None:
    """The workbench adds chrome above the screen and a frame below it, nothing in between.

    The chart's place in the flow is the screen's own: the workbench ships the screen's stylesheet
    unchanged and appends only rules for the index and the frame, so the layout that keeps the
    argument on one screen is the one guarded by the screen's own tests. Above the screen there is
    exactly one kicker and one two-line index. Below it, the report is folded in a closed <details>
    so its 900px frame is never in the flow until the reader opens it.
    """
    html = _artifact()
    workbench = _script()
    dashboard = workbench.load_screen_script()
    assert f"<style>{dashboard.CSS}{workbench.EXTRA_CSS}</style>" in html, (
        "the workbench's stylesheet is no longer the screen's stylesheet plus its own additions"
    )
    assert "figure svg { display: block; width: 100%; height: auto; }" in html
    # Everything between <main> and the screen: one kicker and one index, in that order.
    above = html.split("<main>", 1)[1].split('<section id="screen">', 1)[0]
    assert above.startswith('<p class="kicker">')
    assert above.count("<p ") + above.count("<p>") == 3, "one kicker line and two index lines"
    assert above.count("<nav") == 1
    assert above.endswith("</nav>")
    assert "<h1" not in above and "<figure" not in above and "<table" not in above
    # The frame is below the screen, and closed by default.
    below = html.split("</section>", 1)[1]
    assert below.startswith('<details class="report" id="report">')
    assert "<details open" not in html, "the framed report opens itself and pushes the screen down"
    assert "<iframe" not in html.split("</section>", 1)[0], "the frame has moved above the screen"
    assert f"height: {workbench.FRAME_HEIGHT}px" in html
