"""The README quotes a real artifact, and this test keeps the two in agreement.

A README that shows numbers nobody can check is marketing. The example report is committed so the
claims can be verified, which only works if they stay in sync: every figure the README quotes must
exist in the artifact, the artifact must be reproducible from the command the README gives, and it
must not carry anything from the machine it was built on.
"""

from __future__ import annotations

import re
from html import unescape
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
ARTIFACT = REPO / "examples" / "report-example.html"
README = REPO / "README.md"

BIN_ROWS = (
    "0.61-0.70          27     66%       56%     [37%, 72%]   -0.104",
    "0.74-0.77          27     76%       74%     [55%, 87%]   -0.016",
    "0.82-0.84          26     83%       77%     [58%, 89%]   -0.064",
    "0.90-0.94          27     92%       85%     [68%, 94%]   -0.070",
    "0.97-1.00          27     98%      100%     [88%, 100%]  +0.017",
)

IMPACT_ROWS = (
    ("confidence threshold", "0.60", "0.97", "+0.37"),
    ("auto rate", "100%", "55%", "-45.2 pt"),
    ("accuracy (auto)", "84%", "96%", "+11.5 pt"),
    ("cost per case", "KRW 7,936.51", "KRW 2,095.24", "-73.6%"),
    ("monthly cost", "KRW 158,730,158.73", "KRW 41,904,761.90", "-73.6%"),
)


def _artifact() -> str:
    assert ARTIFACT.exists(), f"the example report is missing: {ARTIFACT.relative_to(REPO)}"
    return ARTIFACT.read_text(encoding="utf-8")


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def _text(fragment: str) -> str:
    return unescape(re.sub(r"<[^>]+>", " ", fragment)).strip()


def _table_rows(html: str, marker: str) -> list[list[str]]:
    section = html.split(marker, 1)[1]
    table = section.split("</table>", 1)[0]
    return [
        [_text(cell) for cell in re.findall(r"<td[^>]*>(.*?)</td>", row)]
        for row in re.findall(r"<tr>(.*?)</tr>", table)
    ]


def test_the_example_report_is_one_self_contained_file() -> None:
    html = _artifact()
    assert html.startswith("<!doctype html>")
    assert len(html.encode("utf-8")) < 1_048_576
    assert "<script src" not in html
    assert "@import" not in html
    assert 'src="http' not in html and 'href="http' not in html


def test_the_example_report_carries_nothing_from_this_machine() -> None:
    """A committed artifact must not leak the author's filesystem or account."""
    html = _artifact()
    for token in ("/Users/", "khope", "@sionic", "Desktop", "/home/"):
        assert token not in html, f"the example report leaks {token!r}"
    assert "source: examples/" in html, "the source note should be a repository-relative path"


def test_the_artifact_is_reproducible_from_the_documented_command() -> None:
    readme = _readme()
    assert "--out-dir examples/report-example --seed 11 --scale 0.5" in readme
    assert "seed 11" in _artifact(), "the artifact does not record the seed it was generated with"


def test_every_bin_row_the_readme_quotes_exists_in_the_artifact() -> None:
    rows = [_text(" ".join(row)) for row in _table_rows(_artifact(), 'id="reliability"')]
    for quoted in BIN_ROWS:
        # The README reproduces one row per line, so every token must appear in that row's text.
        matched = any(all(token in row for token in quoted.split()) for row in rows)
        assert matched, f"README quotes a bin row the artifact does not contain: {quoted!r}"


def test_every_impact_row_the_readme_quotes_exists_in_the_artifact() -> None:
    html = _artifact()
    table = html.split('<table class="impact">', 1)[1].split("</table>", 1)[0]
    rows = [
        tuple(_text(cell) for cell in cells)
        for cells in re.findall(
            r'<tr[^>]*><td>(.*?)</td><td class="num">(.*?)</td><td class="num">(.*?)</td>'
            r'<td class="num">(.*?)</td></tr>',
            table,
        )
    ]
    for quoted in IMPACT_ROWS:
        assert quoted in rows, (
            f"README quotes an impact row the artifact does not contain: {quoted}"
        )
        assert all(cell in _readme() for cell in quoted), f"README row missing a value: {quoted}"


def test_the_verdict_the_readme_quotes_is_the_artifact_verdict() -> None:
    html = _artifact()
    readme = _readme()
    headline = _text(re.search(r'<p class="headline">(.*?)</p>', html).group(1))
    detail = _text(re.search(r'<p class="detail">(.*?)</p>', html).group(1))
    assert headline in readme
    assert detail in readme, "the quoted verdict detail does not match the artifact"


def test_the_readme_size_claim_matches_the_file() -> None:
    size_kb = len(_artifact().encode("utf-8")) / 1024
    claimed = re.search(r"one (\d+) KB file", _readme())
    assert claimed is not None, "the README no longer states the example's file size"
    assert abs(float(claimed.group(1)) - size_kb) < 5, (
        f"claimed {claimed.group(1)} KB, actual {size_kb:.1f} KB"
    )


@pytest.mark.parametrize("token", ["Your threshold is too low.", "0.97"])
def test_representative_claims_are_present(token: str) -> None:
    assert token in _artifact()
