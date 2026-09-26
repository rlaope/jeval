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
    "0.60-0.69          26     65%       46%     [29%, 65%]   -0.188",
    "0.74-0.77          26     76%       69%     [50%, 83%]   -0.064",
    "0.82-0.85          26     83%       69%     [50%, 83%]   -0.140",
    "0.91-0.95          25     93%       96%     [80%, 99%]   +0.033",
    "0.97-1.00          26     98%      100%     [87%, 100%]  +0.016",
)

IMPACT_ROWS = (
    ("confidence threshold", "0.60", "0.75", "+0.15"),
    ("auto rate", "33%", "30%", "-2.9 pt"),
    ("accuracy (auto)", "85%", "91%", "+5.4 pt"),
    ("cost per case", "KRW 1,926", "KRW 1,738", "-9.8%"),
    ("monthly cost", "KRW 38,524,590", "KRW 34,754,098", "-9.8%"),
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
    assert '<dt>Source</dt><dd class="path">examples/' in html, (
        "the source note should be a repository-relative path"
    )


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


def test_every_image_the_readme_references_exists() -> None:
    readme = _readme()
    # Markdown images and the HTML <img> form used for the side-by-side pair.
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme) + re.findall(
        r'<img[^>]+src="([^"]+)"', readme
    )
    assert images, "the README should show the artifact it quotes"

    # Badges are remote by design and cannot be checked without a network call; a repository file
    # shown in the README has to exist, and has to be big enough to be a real screenshot rather
    # than a placeholder that renders as an empty box.
    remote = [image for image in images if re.match(r"https?://", image)]
    assert any("rlaope/jeval" in url for url in remote), "the badges should be this repository's"

    local = [image for image in images if not re.match(r"https?://", image)]
    assert len(local) >= 2, "the README pairs the verdict and cost screenshots"
    for image in local:
        target = REPO / image
        assert target.exists(), f"README references a missing image: {image}"
        assert target.stat().st_size > 20_000, f"{image} looks too small to be a real screenshot"


@pytest.mark.parametrize("token", ["Your threshold is too low.", "0.97"])
def test_representative_claims_are_present(token: str) -> None:
    assert token in _artifact()


def test_the_drift_log_the_readme_asks_a_reader_to_build_is_buildable() -> None:
    """The README quotes a drift run and hands the reader a script; the script must produce it."""
    import importlib.util

    script = REPO / "examples" / "make-drift-log.py"
    assert script.exists(), "the README tells readers to run examples/make-drift-log.py"
    spec = importlib.util.spec_from_file_location("make_drift_log", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    records = module.build()

    assert len(records) == 1800, "the README says the drift log holds 1,800 decisions"
    assert {record.model for record in records} == {"jev-1.13.0", "jev-1.14.0"}, (
        "the drift example needs two model versions to compare"
    )
    assert all(record.is_gold for record in records), "an unlabeled record cannot be compared"


def test_the_readme_paired_block_is_a_captured_run(tmp_path: Path) -> None:
    """The `--paired` block in the README is reproduced literally from the script it names."""
    import importlib.util

    from typer.testing import CliRunner

    from jeval.cli import app
    from jeval.store import records_path, write_records

    quoted = re.search(
        r"```\n\$ jeval drift --root /tmp/jeval-paired (?P<args>[^\n]*)\n(?P<out>.*?)\n```",
        _readme(),
        re.DOTALL,
    )
    assert quoted is not None, "the README no longer quotes a `jeval drift --paired` run"
    script = REPO / "examples" / "make-paired-log.py"
    spec = importlib.util.spec_from_file_location("make_paired_log", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    write_records(module.build(), records_path(tmp_path))

    args = ["drift", "--root", str(tmp_path), *quoted.group("args").split()]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert result.output.rstrip("\n") == quoted.group("out")
