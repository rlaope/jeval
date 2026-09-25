"""The report's inline stylesheet and behaviour, and the minifier that shrinks them.

These are string-level assertions on purpose: the CSS and JS are shipped inside the HTML, so
what matters is the contract they promise the template -- a restyleable palette, a one-page
print layout, offline-only behaviour, and every interactive hook degrading to a static
document when the template does not emit it.
"""

from __future__ import annotations

from jeval.report import svg as chart_svg
from jeval.report.assets import REPORT_CSS, REPORT_JS, minify

# --------------------------------------------------------------------------------------
# Stylesheet
# --------------------------------------------------------------------------------------


def test_css_colour_is_custom_properties_only() -> None:
    assert ":root {" in REPORT_CSS
    assert REPORT_CSS.count("--") >= 15
    # A restyleable report: colours are declared once and consumed through var().
    assert "var(--ink)" in REPORT_CSS
    assert "var(--bg)" in REPORT_CSS
    assert "color-scheme: light dark" in REPORT_CSS


def test_css_has_exactly_one_dark_mode_block() -> None:
    assert REPORT_CSS.count("@media (prefers-color-scheme: dark)") == 1
    dark = REPORT_CSS.split("@media (prefers-color-scheme: dark)", 1)[1]
    # The dark block overrides the palette; it must not be a second stylesheet.
    for variable in ("--bg:", "--ink:", "--muted:", "--rule:", "--accent:"):
        assert variable in dark


def test_css_remaps_every_grey_the_chart_layer_paints() -> None:
    """Dark mode is a palette swap, and a chart's presentation attribute is the one thing a
    custom property cannot reach.

    The dark block re-maps those greys by attribute selector, generated from the chart layer's
    own constants. A grey that exists in only one of the two places is a curve that disappears on
    a dark-mode reader's laptop, and nothing else in the suite would notice.
    """
    dark = REPORT_CSS.split("@media (prefers-color-scheme: dark)", 1)[1]
    for grey in (
        chart_svg.INK,
        chart_svg.MUTED,
        chart_svg.GRID,
        chart_svg.DIAGONAL,
        chart_svg.SHADE,
    ):
        assert grey in dark, f"{grey} is painted by the charts but not re-mapped for dark mode"
    assert f'text[fill="{chart_svg.MUTED}"]' in dark
    assert "figure svg" in dark
    # Okabe-Ito carries meaning, so it is deliberately never re-mapped.
    for hue in chart_svg.PALETTE:
        assert hue not in dark, f"{hue} carries meaning and must not be re-mapped"


def test_css_is_self_contained() -> None:
    for forbidden in ("@import", "url(", "http://", "https://", "<link", "fonts.googleapis"):
        assert forbidden not in REPORT_CSS


def test_css_has_a_one_page_print_stylesheet() -> None:
    assert "@page" in REPORT_CSS
    assert "size: A4 portrait" in REPORT_CSS
    assert "margin: 12mm" in REPORT_CSS
    assert "@media print" in REPORT_CSS
    print_block = REPORT_CSS.split("@media print", 1)[1]
    # Verdict first, and nothing interactive or repeated left on the page.
    assert "#verdict" in print_block
    assert "order: -1" in print_block
    for chrome in ("nav", ".tabs", ".tab", "footer", ".copy-summary", "button"):
        assert chrome in print_block
    assert "display: none !important" in print_block
    # Collapsed details are chrome; open tables print whole.
    assert "details:not([open])" in print_block
    assert "break-inside: avoid" in print_block
    assert "table" in print_block
    assert "tr, th, td" in print_block
    # A chart capped by height printed with five-point labels; it only ever shrinks to the width.
    assert "figure svg { max-width: 100%; height: auto; }" in print_block
    assert "max-height" not in print_block
    # The printout says it is a summary, and names what it left in the file.
    assert ".print-only { display: block !important;" in print_block


def test_css_styles_tabs_sliders_and_numbers() -> None:
    assert ".tabs {" in REPORT_CSS
    assert ".tab {" in REPORT_CSS
    assert '[aria-selected="true"]' in REPORT_CSS
    assert ".slider {" in REPORT_CSS
    assert 'input[type="range"]' in REPORT_CSS
    assert "::-webkit-slider-thumb" in REPORT_CSS
    assert "::-moz-range-thumb" in REPORT_CSS
    assert "[data-jeval-threshold]" in REPORT_CSS
    assert "font-variant-numeric: tabular-nums" in REPORT_CSS
    assert ".num { font-family: var(--font)" in REPORT_CSS


def test_css_referenced_attributes_match_the_javascript_contract() -> None:
    # The stylesheet hooks the JS toggles, and the figure cells it writes into.
    for hook in ("data-jeval-threshold", "data-jeval-segment", "data-jeval-segment-chart"):
        assert hook in REPORT_CSS
        assert hook in REPORT_JS
    assert "data-jeval-value" in REPORT_JS


# --------------------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------------------


def test_js_is_one_iife_and_loads_nothing() -> None:
    stripped = REPORT_JS.strip()
    assert stripped.startswith("(function () {")
    assert stripped.endswith("})();")
    assert '"use strict";' in REPORT_JS
    for forbidden in ("fetch(", "XMLHttpRequest", "WebSocket", "import(", "require("):
        assert forbidden not in REPORT_JS
    # No external asset is pulled in from the behaviour either.
    assert "http://" not in REPORT_JS
    assert "https://" not in REPORT_JS


def test_js_wires_tabs_sliders_copy_and_segments() -> None:
    # Tabs switch which reliability figure and data-quality block is visible.
    assert "data-jeval-tab" in REPORT_JS
    assert "data-jeval-question" in REPORT_JS
    # The slider recomputes every displayed figure from the embedded JSON.
    assert "jeval-data" in REPORT_JS
    assert "JSON.parse" in REPORT_JS
    assert "data-jeval-action" in REPORT_JS
    assert 'input[type="range"]' in REPORT_JS
    for key in (
        "auto_rate",
        "accuracy_auto",
        "cost_per_case",
        "cost_per_month",
        "auto_per_month",
        "escalations_per_month",
    ):
        assert key in REPORT_JS
    assert "monthly_volume" in REPORT_JS
    # The copy button copies the markdown summary out of the same embedded JSON.
    assert "data-jeval-copy-summary" in REPORT_JS
    assert "summary_markdown" in REPORT_JS
    assert "clipboard" in REPORT_JS
    # Segment bars toggle a per-segment chart.
    assert "data-jeval-segment-chart" in REPORT_JS
    assert "data-jeval-segment" in REPORT_JS


def test_js_checks_every_lookup_before_use() -> None:
    # Every element lookup is guarded, so a report without tabs, sliders or segments is still
    # a readable static document instead of a JavaScript error.
    for lookup, guard in (
        ("var node = doc.getElementById(DATA_ID);", "if (!node)"),
        ("var slider = box.querySelector('input[type=\"range\"]');", "if (!slider)"),
        ("var spec = first(actions, [action]);", "if (!spec)"),
        ("var chart = findChart(bar);", "if (!chart)"),
        ("var point = pointAt(curve, threshold);", "if (!point)"),
    ):
        assert lookup in REPORT_JS, f"missing lookup: {lookup}"
        remainder = REPORT_JS.split(lookup, 1)[1].lstrip()
        assert remainder.startswith(guard), f"{lookup} is not guarded with {guard}"
    assert "return {};" in REPORT_JS  # unreadable/absent JSON degrades to empty data
    assert "catch (error)" in REPORT_JS


def test_js_reads_the_curve_shape_the_template_emits() -> None:
    # jeval.report.template emits {"t", "cost", "auto", "acc"}; the documented names are read
    # too, so neither lane has to change for the slider to move.
    for alias in ('"threshold", "t"', '"auto_rate", "auto"', '"acc"', '"cost", "cost_per_case"'):
        assert alias in REPORT_JS
    assert '"auto_rate"' in REPORT_JS
    assert '"accuracy_auto"' in REPORT_JS


def test_js_accepts_the_hooks_the_template_already_ships() -> None:
    # data-copy-target and data-target are in the tree today; the documented data-jeval-*
    # hooks stay the contract and are tried first.
    assert "[data-jeval-copy-summary], [data-copy-target]" in REPORT_JS
    assert "[data-jeval-segment], [data-target]" in REPORT_JS
    assert "doc.getElementById(target)" in REPORT_JS


def test_js_formats_money_the_way_the_python_helpers_do() -> None:
    # The slider's figures follow jeval.currency: the same suffixes, three significant figures,
    # and the currency's own decimals read from data-currency-digits rather than a fixed two.
    for suffix in ('"T"', '"B"', '"M"', '"k"'):
        assert suffix in REPORT_JS
    assert "size >= 100 ? 0 : (size >= 10 ? 1 : 2)" in REPORT_JS
    assert 'getAttribute("data-currency-digits")' in REPORT_JS
    assert "amount(point.cost, 2)" not in REPORT_JS, "a fixed two decimals prints KRW with cents"


# --------------------------------------------------------------------------------------
# minify
# --------------------------------------------------------------------------------------


def test_minify_strips_block_comments() -> None:
    assert minify("a /* note */ b") == "a b"
    assert minify("a/* note */b") == "ab"
    assert minify("/* whole thing */x") == "x"


def test_minify_strips_line_comments_without_eating_urls() -> None:
    assert minify("a // note\nb") == "a b"
    assert minify("a // note") == "a"
    # A colon before the slashes means it is a URL, not a comment.
    assert minify("see https://example.test/a") == "see https://example.test/a"


def test_minify_collapses_whitespace_tightly() -> None:
    assert minify("a  \n\t b") == "a b"
    assert minify("a{b: 1;\n  c: 2}") == "a{b: 1;c: 2}"
    assert minify("   ") == ""
    assert minify("") == ""


def test_minify_never_touches_string_literals() -> None:
    css = 'content: "two  spaces";'
    assert minify(css) == 'content: "two  spaces";'
    js = 'x = "a  //b  /*c*/";'
    assert minify(js) == 'x = "a  //b  /*c*/";'


def test_minify_handles_escaped_quotes_inside_strings() -> None:
    assert minify("var a = 'it\\'s  fine';") == "var a = 'it\\'s  fine';"


def test_minify_is_idempotent_and_shrinks_the_shipped_assets() -> None:
    for source in (REPORT_CSS, REPORT_JS):
        once = minify(source)
        assert len(once) < len(source)
        assert minify(once) == once


def test_minified_css_is_still_valid_shape() -> None:
    css = minify(REPORT_CSS)
    assert css.count("{") == css.count("}")
    assert css.count("(") == css.count(")")
    assert "/*" not in css and "*/" not in css
    assert "@media print" in css
    assert "prefers-color-scheme: dark" in css
    assert ":root{" in css


def test_minified_js_is_still_one_iife() -> None:
    js = minify(REPORT_JS)
    assert js.startswith("(function (){")
    assert js.endswith("})();")
    assert "//" not in js
    assert "jeval-data" in js
    assert "summary_markdown" in js


def test_dark_mode_remaps_the_ink_polygons_too() -> None:
    # The verdict ruler's arrowhead is an ink polygon; unmapped, it is a black triangle on a dark
    # panel while its shaft turns light.
    dark = REPORT_CSS.split("@media (prefers-color-scheme: dark)", 1)[1]
    assert f'polygon[fill="{chart_svg.INK}"]' in dark


def test_one_axis_keeps_its_decimals_at_billion_scale() -> None:
    ticks = [0.0, 5e8, 1e9, 1.5e9]
    render = chart_svg.tick_format(ticks)
    assert [render(value) for value in ticks] == ["0", "0.5B", "1.0B", "1.5B"]


def test_the_phone_layout_does_not_leak_into_print() -> None:
    assert "@media screen and (max-width: 720px)" in REPORT_CSS
    assert "@media (max-width: 720px)" not in REPORT_CSS
