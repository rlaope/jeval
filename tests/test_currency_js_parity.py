"""The report's slider formats money in the browser; it must print what the Python tables print.

The inline script is executed with node on the same inputs the Python helpers receive. Node is
present on the CI runners and on most development machines; where it is missing, this module is
skipped rather than silently passing, and ``tests/test_currency.py`` still pins the tie cases.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from jeval.currency import format_amount, format_compact, format_percent, minor_units
from jeval.report.assets import REPORT_JS

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

VALUES = (
    0.0,
    0.125,
    0.4321,
    0.04321,
    0.0049,
    1e-9,
    2.5,
    12.46,
    189.0,
    1_737.7,
    1_926.5,
    -12.5,
    1_125.0,
    12_250.0,
    34_754_098.36,
    38_524_590.16,
    999_499.0,
    999_999.5,
    999_500_000.0,
)
CODES = ("KRW", "USD", "KWD")


def _functions() -> str:
    start = REPORT_JS.index("  function finite(value) {")
    end = REPORT_JS.index("  function setText(node, value) {")
    return REPORT_JS[start:end]


def test_the_slider_prints_every_amount_the_way_the_table_does() -> None:
    cases = [(value, code, minor_units(code)) for value in VALUES for code in CODES]
    script = (
        _functions()
        + f"var cases = {json.dumps(cases)};"
        + "console.log(JSON.stringify(cases.map(function (c) {"
        + "return [formatAmount(c[0], c[1], c[2]), compact(c[0], c[1], c[2]), compact(c[0], '', 0)];"
        + "})));"
    )
    result = subprocess.run(
        [NODE or "node", "-e", script], capture_output=True, text=True, check=True, timeout=30
    )
    produced = json.loads(result.stdout)
    expected = [
        [format_amount(value, code), format_compact(value, code), format_compact(value, None)]
        for value, code, _ in cases
    ]
    mismatches = [
        (case, got, want)
        for case, got, want in zip(cases, produced, expected, strict=True)
        if got != want
    ]
    assert mismatches == []


def test_the_slider_prints_every_share_the_way_the_table_does() -> None:
    shares = [0.0, 0.005, 0.125, 0.3, 0.335, 0.5, 0.905, 0.995, 1.0, 17 / 136, 45 / 200]
    script = (
        _functions()
        + f"console.log(JSON.stringify({json.dumps(shares)}.map(function (v) {{ return pct(v); }})));"
    )
    result = subprocess.run(
        [NODE or "node", "-e", script], capture_output=True, text=True, check=True, timeout=30
    )
    assert json.loads(result.stdout) == [format_percent(value) for value in shares]
