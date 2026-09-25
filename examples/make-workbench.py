"""Build the workbench: the one-argument screen and the full report on one page.

This is not a jeval command and not a product surface. The product writes one report, and a
dashboard is explicitly out of scope. This is a *presentation* that puts the two committed surfaces
in this directory on a single page, so one file answers "show me everything":

    uv run python examples/make-workbench.py examples/workbench.html

Nothing on the page is typed in by hand. The screen is computed and rendered by
``examples/make-demo-dashboard.py``, imported as a file and called the way its tests call it. The
report is ``examples/report-example.html`` read at build time and placed in an inline frame through
``srcdoc``, so it keeps its own stylesheet instead of colliding with the screen's, and so a change to
the report can never leave a stale copy here. One file, no server, no network, no JavaScript, and a
second build produces byte-identical output.
"""

from __future__ import annotations

import html
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
SCREEN_SCRIPT = HERE / "make-demo-dashboard.py"
REPORT = HERE / "report-example.html"

# The framed report gets one laptop screen of height and scrolls inside itself, so the reader is
# never thrown to the bottom of the page by a 260 KB document.
FRAME_HEIGHT = 900

# The workbench adds three things above the screen -- a kicker, two index lines and the padding
# around them -- and nothing else, so the screen still ends inside the first 900px of the page. Its
# stylesheet is the screen's; these rules only cover what the screen does not have.
EXTRA_CSS = """\
/* The workbench frames the screen; it borrows the screen's stylesheet and adds only the index and
   the report frame. Type sizes, the column and the colours are the screen's. */
main { padding-top: 4px; }
main > .kicker { margin: 0; }
/* The index has to read as an index and not as three more lines of front matter, without taking a
   pixel from the screen below it: the separation is a hairline, and the links carry the weight. */
.index { margin: 0; font-size: 12px; color: var(--muted); border-top: 1px solid var(--rule); padding-top: 4px; }
.index p { margin: 0; }
.index a { color: var(--ink); font-weight: 600; text-decoration: underline; text-decoration-color: var(--rule-strong); text-decoration-thickness: 1px; text-underline-offset: 3px; }
.index a:hover { text-decoration-color: var(--ink); }
.index code, details.report code { font-size: 11.5px; background: none; padding: 0; color: var(--ink-2); }
#screen h1 { margin-top: 8px; }
details.report > summary { margin-top: 8px; color: var(--ink); }
details.report .frame { padding: 0 0 10px; }
details.report iframe { display: block; width: 100%; height: FRAME_HEIGHTpx; border: 1px solid var(--rule); background: transparent; }
details.report .note { margin: 4px 0 0; font-size: 12px; color: var(--muted); }
""".replace("FRAME_HEIGHTpx", f"{FRAME_HEIGHT}px")


def load_screen_script() -> ModuleType:
    """Load the screen's script the way its tests do: as a file, not as an installed module."""
    spec = importlib.util.spec_from_file_location("make_demo_dashboard", SCREEN_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _index(dashboard: ModuleType, screen: Any, report_kb: int) -> str:
    """Two lines, one per part, each saying what the part is and linking to it."""
    escape = dashboard.S.escape
    return (
        '<nav class="index" aria-label="Contents">'
        '<p><a href="#screen">The argument</a> · one screen: the line in use against the line the '
        f"cost minimum points at, on <code>{escape(screen.question)}</code>, with the drift "
        "gate&rsquo;s verdict.</p>"
        '<p><a href="#report">The report</a> · what <code>jeval report</code> writes on the same '
        f"synthetic log, unchanged, framed below (<code>examples/report-example.html</code>, "
        f"<code>{report_kb} KB</code>).</p>"
        "</nav>"
    )


def _report_details(report_html: str, report_kb: int) -> str:
    """The full report, folded, as an inline frame whose document is the committed file.

    ``srcdoc`` takes the whole document HTML-escaped, so the report's own ``<style>`` applies inside
    the frame and nothing of it leaks into the page. The frame has a fixed height and scrolls inside
    itself.
    """
    return (
        '<details class="report" id="report">'
        "<summary>The full report: what <code>jeval report</code> writes on this log "
        f"(<code>examples/report-example.html</code>, <code>{report_kb} KB</code>)</summary>"
        '<div class="frame">'
        '<p class="note">The document below is <code>examples/report-example.html</code> as '
        "committed, placed in an inline frame at build time; it scrolls on its own and follows the "
        "same light or dark scheme as this page.</p>"
        f'<iframe title="jeval report on the synthetic log" srcdoc="{html.escape(report_html)}">'
        "</iframe>"
        "</div></details>"
    )


def build_document() -> str:
    """The whole page: kicker, index, the screen, then the report in a frame."""
    dashboard = load_screen_script()
    screen, provenance, models = dashboard.build_screen()
    report_html = REPORT.read_text(encoding="utf-8")
    report_kb = len(report_html.encode("utf-8")) // 1024
    kicker = dashboard.kicker_line(
        provenance=provenance, models=models, product="jeval · workbench"
    )
    body = dashboard.render_body(screen, kicker="")
    escape = dashboard.S.escape
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>jeval · workbench · {escape(screen.question)} · the screen and the report</title>"
        f"<style>{dashboard.CSS}{EXTRA_CSS}</style></head><body><main>"
        f"{kicker}"
        f"{_index(dashboard, screen, report_kb)}"
        f'<section id="screen">{body}</section>'
        f"{_report_details(report_html, report_kb)}"
        "</main></body></html>\n"
    )


def main(argv: list[str]) -> int:
    out = Path(argv[1]) if len(argv) > 1 else HERE / "workbench.html"
    document = build_document()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")
    print(f"wrote {out} ({len(document.encode('utf-8')) // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
