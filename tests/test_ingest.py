"""Tests for ingest: raw rows to one record per question."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from jeval.config import IngestMap, load_ingest_map
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


# --- the inline label path gets the same guard as the harvest path ---------------------------
# A label that arrives inside the log used to skip the check entirely, so a department answer on a
# noul record was stored as ground truth and counted as a wrong answer forever.


def test_an_inline_label_the_question_cannot_produce_is_dropped_and_counted(tmp_path: Path) -> None:
    raw = tmp_path / "log.jsonl"
    raw.write_text(
        json.dumps(
            {
                "model": "jev-1.13.0",
                "questions": [
                    {
                        "question_key": "is_urgent",
                        "question_type": "noul",
                        "prediction": "yes",
                        "confidence": 0.6,
                        "probabilities": {"yes": 0.8, "no": 0.2},
                        "label": "billing",  # a department answer on a yes/no question
                        "label_source": "human_override",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "records.jsonl"
    report = ingest_files([raw], out, IngestMap())

    assert report.n_impossible_labels == 1
    assert report.impossible_labels == ["is_urgent='billing'"]
    stored = read_records(out)
    assert len(stored) == 1
    assert stored[0].label is None  # the label is refused
    assert stored[0].label_source is None
    assert stored[0].prediction == "yes"  # the prediction is kept: it is evidence either way
    assert report.n_unlabeled == 1


def test_a_possible_inline_label_is_left_alone(tmp_path: Path) -> None:
    raw = tmp_path / "log.jsonl"
    raw.write_text(
        json.dumps(
            {
                "model": "jev-1.13.0",
                "questions": [
                    {
                        "question_key": "department",
                        "question_type": "choice",
                        "prediction": "billing",
                        "confidence": 0.91,
                        "probabilities": {"billing": 0.91, "technical": 0.06},
                        "label": "technical",
                        "label_source": "human_override",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "records.jsonl"
    report = ingest_files([raw], out, IngestMap())

    assert report.n_impossible_labels == 0
    stored = read_records(out)
    assert stored[0].label == "technical"
    assert stored[0].label_source == "human_override"


def test_a_flat_column_can_be_a_segment(tmp_path: Path) -> None:
    """`--by lang` was unreachable for any log that carries `lang` as a column, not an object.

    A mapping config renames fields; it could not build the object the schema wants, so the row was
    rejected and the documented per-segment sweep had no segment to sweep.
    """
    rows = [
        {"question": "department", "prediction": "billing", "confidence": 0.9, "lang": "ko"},
        {"question": "department", "prediction": "technical", "confidence": 0.8, "lang": "en"},
    ]
    source = _write_jsonl(tmp_path / "app.jsonl", rows)
    mapping = IngestMap(
        field_map={
            "question_key": "question",
            "prediction": "prediction",
            "confidence": "confidence",
            "segment": "lang",
        },
        defaults={"question_type": "choice", "model": "jev-1.13.0"},
    )

    report = ingest_files([source], tmp_path / "records.jsonl", mapping)

    assert report.n_skipped == 0
    records = read_records(tmp_path / "records.jsonl")
    assert [record.segment for record in records] == [{"lang": "ko"}, {"lang": "en"}]


def test_an_object_column_is_still_mapped_as_an_object(tmp_path: Path) -> None:
    """The flat-column convenience must not flatten a log that already carries segments."""
    rows = [
        {
            "question": "department",
            "prediction": "billing",
            "confidence": 0.9,
            "segment": {"lang": "ko", "channel": "email"},
        }
    ]
    source = _write_jsonl(tmp_path / "app.jsonl", rows)
    mapping = IngestMap(
        field_map={
            "question_key": "question",
            "prediction": "prediction",
            "confidence": "confidence",
            "segment": "segment",
        },
        defaults={"question_type": "choice", "model": "jev-1.13.0"},
    )

    ingest_files([source], tmp_path / "records.jsonl", mapping)

    assert read_records(tmp_path / "records.jsonl")[0].segment == {
        "lang": "ko",
        "channel": "email",
    }


def test_a_skipped_row_says_what_to_change(tmp_path: Path) -> None:
    """The message used to be a pydantic dump naming neither the field nor where to fix it."""
    source = _write_jsonl(
        tmp_path / "app.jsonl",
        [{"question": "department", "prediction": "billing", "confidence": 0.9}],
    )
    mapping = IngestMap(
        field_map={
            "question_key": "question",
            "prediction": "prediction",
            "confidence": "confidence",
        },
        defaults={"question_type": "choice"},
    )

    report = ingest_files([source], tmp_path / "records.jsonl", mapping)

    assert report.n_skipped == 1
    message = report.errors[0]
    assert "model" in message
    assert "field_map" in message and "defaults" in message
    for jargon in ("pydantic", "ValidationError", "input_value", "errors.pydantic.dev"):
        assert jargon not in message
    assert "\n" not in message  # one line per row, so a log of them stays readable


def test_a_numeric_column_is_a_usable_segment_axis(tmp_path: Path) -> None:
    """Tiers logged as numbers are a segment too: the column name is the key, the value a string."""
    source = _write_jsonl(
        tmp_path / "app.jsonl",
        [
            {
                "question": "department",
                "prediction": "billing",
                "confidence": 0.9,
                "model": "jev-1.13.0",
                "tier": 3,
            }
        ],
    )
    mapping = IngestMap(
        field_map={
            "question_key": "question",
            "prediction": "prediction",
            "confidence": "confidence",
            "segment": "tier",
        },
        defaults={"question_type": "choice"},
    )

    report = ingest_files([source], tmp_path / "records.jsonl", mapping)

    assert report.n_skipped == 0
    assert read_records(tmp_path / "records.jsonl")[0].segment == {"tier": "3"}


def test_a_wrong_shaped_value_names_the_field(tmp_path: Path) -> None:
    """Text where a number belongs: the message says which field and what arrived."""
    source = _write_jsonl(
        tmp_path / "app.jsonl",
        [
            {
                "question": "department",
                "prediction": "billing",
                "confidence": "high",
                "model": "jev-1.13.0",
            }
        ],
    )
    mapping = IngestMap(
        field_map={
            "question_key": "question",
            "prediction": "prediction",
            "confidence": "confidence",
        },
        defaults={"question_type": "choice"},
    )

    report = ingest_files([source], tmp_path / "records.jsonl", mapping)

    assert report.n_skipped == 1
    message = report.errors[0]
    assert "confidence" in message
    assert "'high'" in message
    assert "errors.pydantic.dev" not in message


def test_a_column_claimed_by_one_field_is_not_reused_for_another(tmp_path: Path) -> None:
    """The map said `prediction` comes from the column `label`; that column must not also become
    the answer.

    An application that logs its prediction in a column called `label` produced a ground truth equal
    to the prediction — every record "correct", silver-labeled, out of nothing. The report refused to
    measure it (gold and silver stay separate), but the records were poisoned and the silver-slice
    sections counted them.
    """
    source = _write_jsonl(
        tmp_path / "app.jsonl",
        [
            {"task": "department", "label": "billing", "score_0_to_1": 0.91},
            {"task": "department", "label": "technical", "score_0_to_1": 0.72},
        ],
    )
    mapping = IngestMap(
        field_map={
            "question_key": "task",
            "prediction": "label",
            "confidence": "score_0_to_1",
        },
        defaults={"question_type": "choice", "model": "jev-1.13.0"},
    )

    report = ingest_files([source], tmp_path / "records.jsonl", mapping)
    records = read_records(tmp_path / "records.jsonl")

    assert report.n_skipped == 0
    assert [record.prediction for record in records] == ["billing", "technical"]
    assert all(record.label is None for record in records), (
        "a column the map assigns to `prediction` was also read as the answer"
    )


def test_an_identity_mapping_still_reads_its_own_column(tmp_path: Path) -> None:
    """The convenience this guard must not break: `label: label` reads the label column."""
    source = _write_jsonl(
        tmp_path / "records.jsonl",
        [
            {
                "question_key": "department",
                "prediction": "billing",
                "confidence": 0.9,
                "label": "technical",
            }
        ],
    )
    mapping = IngestMap(
        field_map={"label": "label", "prediction": "prediction", "confidence": "confidence"},
        defaults={"question_type": "choice", "model": "jev-1.13.0"},
    )

    ingest_files([source], tmp_path / "out.jsonl", mapping)

    assert read_records(tmp_path / "out.jsonl")[0].label == "technical"
