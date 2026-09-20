"""Tests for ingest: raw rows to one record per question."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from jeval.config import load_ingest_map
from jeval.ingest import ingest_files, iter_rows
from jeval.store import read_records


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _mapping_file(tmp_path: Path, payload: dict) -> Path:
    target = tmp_path / "ingest-map.yaml"
    target.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return target


def test_one_request_with_several_questions_becomes_several_records(tmp_path: Path) -> None:
    source = _write_jsonl(
        tmp_path / "req.jsonl",
        [
            {
                "model": "jev-1.13.0",
                "segment": {"lang": "ko"},
                "questions": [
                    {
                        "question_key": "department",
                        "question_type": "choice",
                        "prediction": "billing",
                        "confidence": 0.91,
                        "label": "billing",
                        "label_source": "human_override",
                    },
                    {
                        "question_key": "intent",
                        "question_type": "choice",
                        "prediction": "refund_request",
                        "confidence": 0.58,
                        "label": None,
                    },
                ],
            }
        ],
    )
    out = tmp_path / "records.jsonl"
    report = ingest_files([source], out, load_ingest_map(tmp_path / "missing.yaml"))
    assert report.n_rows == 1
    assert report.n_records == 2
    assert report.per_question == {"department": 1, "intent": 1}
    assert report.n_unlabeled == 1
    records = read_records(out)
    assert {record.question_key for record in records} == {"department", "intent"}
    assert all(record.segment["lang"] == "ko" for record in records)
    assert all(record.model == "jev-1.13.0" for record in records)


def test_field_map_renames_production_columns(tmp_path: Path) -> None:
    mapping = load_ingest_map(
        _mapping_file(
            tmp_path,
            {
                "version": 1,
                "field_map": {
                    "model": "model_version",
                    "question_key": "question",
                    "prediction": "answer",
                    "confidence": "score",
                    "label": "final_answer",
                    "label_source": None,
                },
                "defaults": {"question_type": "choice", "label_source": "human_review"},
            },
        )
    )
    source = _write_jsonl(
        tmp_path / "prod.jsonl",
        [
            {
                "model_version": "jev-1.13.0",
                "question": "department",
                "answer": "billing",
                "score": 0.77,
                "final_answer": "technical",
            }
        ],
    )
    out = tmp_path / "records.jsonl"
    ingest_files([source], out, mapping)
    record = read_records(out)[0]
    assert record.model == "jev-1.13.0"
    assert record.question_key == "department"
    assert record.prediction == "billing"
    assert record.confidence == 0.77
    assert record.label == "technical"
    assert record.label_source == "human_review"
    assert record.is_correct is False


def test_csv_input_is_supported(tmp_path: Path) -> None:
    mapping = load_ingest_map(
        _mapping_file(tmp_path, {"version": 1, "field_map": {"question_key": "question"}})
    )
    source = tmp_path / "log.csv"
    source.write_text(
        "model,question,question_type,prediction,confidence,label,label_source\n"
        "jev-1.13.0,department,choice,billing,0.9,billing,human_override\n"
        "jev-1.13.0,department,choice,billing,0.6,technical,human_override\n",
        encoding="utf-8",
    )
    out = tmp_path / "records.jsonl"
    report = ingest_files([source], out, mapping)
    assert report.n_records == 2
    records = read_records(out)
    assert [record.is_correct for record in records] == [True, False]


def test_duplicate_question_keys_inside_one_row_are_reported_not_double_counted(
    tmp_path: Path,
) -> None:
    source = _write_jsonl(
        tmp_path / "dup.jsonl",
        [
            {
                "model": "m",
                "questions": [
                    {
                        "question_key": "department",
                        "question_type": "choice",
                        "prediction": "a",
                        "confidence": 0.8,
                    },
                    {
                        "question_key": "department",
                        "question_type": "choice",
                        "prediction": "b",
                        "confidence": 0.7,
                    },
                ],
            }
        ],
    )
    report = ingest_files(
        [source], tmp_path / "records.jsonl", load_ingest_map(tmp_path / "none.yaml")
    )
    assert report.n_records == 1
    assert report.n_duplicate_questions == 1


def test_bad_rows_are_skipped_with_a_readable_error(tmp_path: Path) -> None:
    source = _write_jsonl(
        tmp_path / "mixed.jsonl",
        [
            {
                "model": "m",
                "question_key": "department",
                "question_type": "choice",
                "prediction": "a",
                "confidence": 0.8,
            },
            {
                "model": "m",
                "question_key": "department",
                "question_type": "choice",
                "prediction": "a",
                "confidence": 4.2,
            },
        ],
    )
    report = ingest_files(
        [source], tmp_path / "records.jsonl", load_ingest_map(tmp_path / "none.yaml")
    )
    assert report.n_records == 1
    assert report.n_skipped == 1
    assert report.errors and "out of range" in report.errors[0]


def test_timestamp_epoch_seconds_are_accepted(tmp_path: Path) -> None:
    source = _write_jsonl(
        tmp_path / "epoch.jsonl",
        [
            {
                "model": "m",
                "question_key": "q",
                "question_type": "choice",
                "prediction": "a",
                "confidence": 0.9,
                "ts": 1795000000,
            }
        ],
    )
    ingest_files([source], tmp_path / "records.jsonl", load_ingest_map(tmp_path / "none.yaml"))
    record = read_records(tmp_path / "records.jsonl")[0]
    assert record.ts.year == 2026


def test_append_keeps_existing_records(tmp_path: Path) -> None:
    source = _write_jsonl(
        tmp_path / "one.jsonl",
        [
            {
                "model": "m",
                "question_key": "q",
                "question_type": "choice",
                "prediction": "a",
                "confidence": 0.9,
            }
        ],
    )
    out = tmp_path / "records.jsonl"
    ingest_files([source], out, load_ingest_map(tmp_path / "none.yaml"))
    ingest_files([source], out, load_ingest_map(tmp_path / "none.yaml"), append=True)
    assert len(read_records(out)) == 2


def test_iter_rows_rejects_a_non_object_json_line(tmp_path: Path) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text("[1, 2, 3]\n", encoding="utf-8")
    try:
        list(iter_rows(source))
    except ValueError as exc:
        assert "expected a JSON object" in str(exc)
    else:  # pragma: no cover - the assertion above is the contract
        raise AssertionError("expected a ValueError for a non-object JSON line")


def test_missing_input_file_is_reported(tmp_path: Path) -> None:
    try:
        ingest_files(
            [tmp_path / "nope.jsonl"],
            tmp_path / "records.jsonl",
            load_ingest_map(tmp_path / "none.yaml"),
        )
    except FileNotFoundError as exc:
        assert "nope.jsonl" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected FileNotFoundError")
