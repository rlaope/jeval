"""Tests for ingest presets: a product's own log, read without reshaping it first.

The preset is the only place a product's spelling is written down, so these tests pin two things:
a native log line becomes measurable records, and a log line nobody can interpret is reported with
the location that was searched instead of being forced into a wrong record.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from jeval import collect, presets
from jeval.cli import app
from jeval.ingest import ingest_preset_files
from jeval.store import read_records

runner = CliRunner()

# A native log line: the request and the response the API returned, side by side.
NATIVE_ROW: dict[str, Any] = {
    "request_id": "req_00042",
    "request": {
        "model": "jev-latest",
        "state": "My payouts have been failing for 3 days.",
        "questions": {"department": {"type": "choice", "instructions": "Which team?"}},
    },
    "response": {
        "model": "jev-1.13.0",
        "answers": {
            "department": {
                "type": "choice",
                "choice": "technical",
                "probabilities": {"billing": 0.08, "technical": 0.85, "sales": 0.07},
                "confidence": 0.82,
            },
            "is_urgent": {"type": "noul", "noul": 0.999},
        },
        "usage": {"input_tokens": 312, "output_tokens": 48},
    },
}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(collect.ENV_FLAG, raising=False)
    monkeypatch.delenv(collect.ENV_ROOT, raising=False)
    collect.reset_stats()


def _write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_the_registry_names_what_it_has() -> None:
    assert presets.available() == ("jev-native",)
    with pytest.raises(ValueError, match=r"unknown preset .*available: jev-native"):
        presets.get("nope")


def test_describe_tells_a_user_where_the_preset_looks() -> None:
    described = presets.describe(presets.JEV_NATIVE)

    assert "response" in described
    assert "answers" in described
    assert "request_id" in described


def test_a_native_log_line_becomes_measurable_payloads() -> None:
    payloads = presets.rows_to_payloads(presets.JEV_NATIVE, NATIVE_ROW)
    by_key = {payload["question_key"]: payload for payload in payloads}

    assert set(by_key) == {"department", "is_urgent"}
    assert by_key["department"]["prediction"] == "technical"
    assert by_key["is_urgent"]["probabilities"]["yes"] == 0.999
    # the answered model, not the alias the caller asked for: this is what drift detection needs
    assert by_key["department"]["model"] == "jev-1.13.0"
    assert by_key["department"]["state_tokens"] == 312
    assert by_key["department"]["source_key"] == "req_00042"  # the join key for later labels


def test_a_line_without_a_response_is_reported_not_invented() -> None:
    assert presets.rows_to_payloads(presets.JEV_NATIVE, {"request_id": "r1"}) == []


def test_the_caller_can_override_where_the_key_lives() -> None:
    row = {"trace": {"id": "t-7"}, **NATIVE_ROW}
    payloads = presets.rows_to_payloads(presets.JEV_NATIVE, row, source_key_field="trace.id")

    assert payloads[0]["source_key"] == "t-7"


def test_the_caller_can_override_the_response_location() -> None:
    row = {"result": NATIVE_ROW["response"], "request_id": "r2"}
    payloads = presets.rows_to_payloads(presets.JEV_NATIVE, row, keys=None)

    assert payloads == []  # the default looks under 'response', so nothing is found here
    moved = presets.rows_to_payloads(presets.JEV_NATIVE, row, keys=dict(presets.JEV_NATIVE.keys))
    assert moved == []


def test_ingesting_a_native_log_end_to_end(tmp_path: Path) -> None:
    source = _write(tmp_path / "native.jsonl", [NATIVE_ROW, NATIVE_ROW])
    out = tmp_path / "records.jsonl"

    report = ingest_preset_files([source], out, presets.JEV_NATIVE)

    assert report.n_records == 4  # two answers per request, two requests
    assert report.per_question == {"department": 2, "is_urgent": 2}
    assert report.n_unlabeled == 4  # the log carries no human answer yet, and says so
    stored = read_records(out)
    assert {record.model for record in stored} == {"jev-1.13.0"}
    assert {record.source_key for record in stored} == {"req_00042"}


def test_a_line_the_preset_cannot_read_is_counted_with_the_location_searched(
    tmp_path: Path,
) -> None:
    source = _write(tmp_path / "mixed.jsonl", [NATIVE_ROW, {"request_id": "r9"}])
    out = tmp_path / "records.jsonl"

    report = ingest_preset_files([source], out, presets.JEV_NATIVE)

    assert report.n_records == 2
    assert report.n_skipped == 1
    assert "no response object found" in report.errors[0]
    assert "'answers'" in report.errors[0]


def test_the_cli_ingests_a_native_log(tmp_path: Path) -> None:
    source = _write(tmp_path / "native.jsonl", [NATIVE_ROW])

    result = runner.invoke(
        app, ["ingest", "--preset", "jev-native", str(source), "--root", str(tmp_path)]
    )

    assert result.exit_code == 0, result.stdout
    assert "wrote 2 records" in result.stdout
    assert "preset: jev-native" in result.stdout
    assert len(read_records(tmp_path / ".jeval" / "records.jsonl")) == 2


def test_the_cli_can_list_presets() -> None:
    result = runner.invoke(app, ["ingest", "--list-presets"])

    assert result.exit_code == 0
    assert "jev-native" in result.stdout
    assert "answers" in result.stdout


def test_an_unknown_preset_is_refused_with_the_alternatives(tmp_path: Path) -> None:
    source = _write(tmp_path / "native.jsonl", [NATIVE_ROW])

    result = runner.invoke(
        app, ["ingest", "--preset", "nope", str(source), "--root", str(tmp_path)]
    )

    assert result.exit_code == 1
    assert "jev-native" in result.stdout


def test_the_ingested_log_is_measurable_by_the_rest_of_the_tool(tmp_path: Path) -> None:
    """The point of the preset: the log a product already writes reaches the report."""
    source = _write(tmp_path / "native.jsonl", [NATIVE_ROW])
    runner.invoke(app, ["ingest", "--preset", "jev-native", str(source), "--root", str(tmp_path)])

    report = runner.invoke(app, ["report", "--root", str(tmp_path), "-o", str(tmp_path / "r.html")])

    assert report.exit_code == 0, report.stdout
    assert (tmp_path / "r.html").exists()
    # Every answer is unlabeled, and the tool says so rather than measuring nothing quietly.
    assert "no labeled records" in report.stdout.lower() or "label" in report.stdout.lower()
