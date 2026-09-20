"""Tests for the free-label harvest: external human answers onto existing records."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from jeval.config import LabelFromSpec, load_ingest_map
from jeval.ingest import (
    LabelApplication,
    harvest_file,
    harvest_labels,
    ingest_files,
    iter_rows,
    rewrite_labels_atomic,
)
from jeval.schema import DecisionRecord
from jeval.store import read_records, write_records


def _record(
    *,
    source_key: str | None,
    question: str = "department",
    **overrides: Any,
) -> DecisionRecord:
    payload: dict[str, Any] = {
        "model": "jev-1.13.0",
        "question_key": question,
        "question_type": "choice",
        "prediction": "billing",
        "confidence": 0.8,
        "source_key": source_key,
    }
    payload.update(overrides)
    return DecisionRecord.model_validate(payload)


def _resolution_rows(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    path = tmp_path / "resolution.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_rows_joined_by_key_get_the_label_and_its_source() -> None:
    records = [_record(source_key="t-1"), _record(source_key="t-2")]
    rows = [
        {"ticket_id": "t-1", "final_answer": "billing"},
        {"ticket_id": "t-2", "final_answer": "technical"},
    ]
    report = harvest_labels(
        records, rows, field="final_answer", source="human_override", join_on="ticket_id"
    )

    assert report.n_applied == 2
    assert report.n_records == 2
    assert report.applied_question_keys == ("department",)
    assert [record.label for record in records] == ["billing", "technical"]
    assert {record.label_source for record in records} == {"human_override"}
    assert [record.is_correct for record in records] == [True, False]
    assert all(record.is_gold for record in records)


def test_silver_source_is_recorded_as_silver_and_not_as_gold() -> None:
    records = [_record(source_key="t-9")]
    rows = [{"ticket_id": "t-9", "final_answer": "technical"}]

    report = harvest_labels(
        records, rows, field="final_answer", source="silver", join_on="ticket_id"
    )

    assert report.n_applied == 1
    assert report.source == "silver"
    assert records[0].is_silver is True
    assert records[0].is_gold is False
    assert records[0].is_correct is False


def test_noul_answers_are_normalized_the_way_the_schema_defines_it() -> None:
    records = [
        DecisionRecord.model_validate(
            {
                "model": "jev-1.13.0",
                "question_key": "is_urgent",
                "question_type": "noul",
                "prediction": "yes",
                "confidence": 0.6,
                "source_key": "t-1",
            }
        )
    ]
    rows = [{"ticket_id": "t-1", "resolution": {"final_urgent": False}}]

    report = harvest_labels(
        records,
        rows,
        field="resolution.final_urgent",
        source="human_override",
        join_on="ticket_id",
    )

    assert report.n_applied == 1
    assert records[0].label == "no"
    assert records[0].is_correct is False


def test_dotted_label_field_reads_the_resolution_block() -> None:
    records = [_record(source_key="t-1")]
    rows = [{"ticket_id": "t-1", "resolution": {"final_department": "technical"}}]

    report = harvest_labels(
        records,
        rows,
        field="resolution.final_department",
        source="human_override",
        join_on="ticket_id",
    )

    assert report.n_applied == 1
    assert records[0].label == "technical"


def test_unmatched_rows_are_counted_and_named_never_dropped() -> None:
    records = [_record(source_key="t-1")]
    rows = [
        {"ticket_id": "t-1", "final_answer": "billing"},
        {"ticket_id": "t-404", "final_answer": "technical"},
        {"ticket_id": "t-405", "final_answer": "sales"},
    ]

    report = harvest_labels(
        records, rows, field="final_answer", source="human_override", join_on="ticket_id"
    )

    assert report.n_label_rows == 3
    assert report.n_applied == 1
    assert report.n_unmatched == 2
    assert report.unmatched_keys == ("t-404", "t-405")
    assert records[0].label == "billing"


def test_rows_without_a_key_or_without_an_answer_are_counted() -> None:
    records = [_record(source_key="t-1"), _record(source_key="t-2")]
    rows = [
        {"ticket_id": "t-1", "final_answer": "billing"},
        {"ticket_id": "", "final_answer": "technical"},
        {"ticket_id": "t-2", "final_answer": "   "},
        {"final_answer": "sales"},
    ]

    report = harvest_labels(
        records, rows, field="final_answer", source="human_override", join_on="ticket_id"
    )

    assert report.n_rows_without_key == 2
    assert report.n_rows_without_label == 1
    assert report.n_applied == 1
    assert report.n_unmatched == 0
    assert [record.label for record in records] == ["billing", None]


def test_records_without_a_join_key_are_counted_not_guessed() -> None:
    records = [_record(source_key=None), _record(source_key="t-1")]
    rows = [{"ticket_id": "t-1", "final_answer": "billing"}]

    report = harvest_labels(
        records, rows, field="final_answer", source="human_override", join_on="ticket_id"
    )

    assert report.n_skipped_without_key == 1
    assert report.n_applied == 1
    assert records[0].label is None
    assert records[0].label_source is None


def test_existing_labels_are_kept_and_counted_separately() -> None:
    existing = _record(source_key="t-1", label="billing", label_source="human_review")
    records = [existing]
    rows = [{"ticket_id": "t-1", "final_answer": "technical"}]

    report = harvest_labels(
        records, rows, field="final_answer", source="silver", join_on="ticket_id"
    )

    assert report.n_applied == 0
    assert report.n_kept_existing == 1
    assert report.kept_existing_question_keys == ("department",)
    assert report.applications == ()
    assert records[0].label == "billing"
    assert records[0].label_source == "human_review"


def test_overwrite_replaces_an_existing_label_only_when_asked() -> None:
    records = [_record(source_key="t-1", label="billing", label_source="human_review")]
    rows = [{"ticket_id": "t-1", "final_answer": "technical"}]

    report = harvest_labels(
        records,
        rows,
        field="final_answer",
        source="human_override",
        join_on="ticket_id",
        overwrite=True,
    )

    assert report.n_applied == 1
    assert report.n_kept_existing == 0
    assert records[0].label == "technical"
    assert records[0].label_source == "human_override"


def test_missing_join_column_raises_a_readable_error() -> None:
    records = [_record(source_key="t-1")]
    rows = [{"case": "t-1", "final_answer": "billing"}]

    with pytest.raises(ValueError, match="ticket_id"):
        harvest_labels(
            records, rows, field="final_answer", source="human_override", join_on="ticket_id"
        )

    assert records[0].label is None


def test_missing_label_field_raises_a_readable_error() -> None:
    records = [_record(source_key="t-1")]
    rows = [{"ticket_id": "t-1", "resolution": "billing"}]

    with pytest.raises(ValueError, match=re.escape("resolution.final_department")):
        harvest_labels(
            records,
            rows,
            field="resolution.final_department",
            source="human_override",
            join_on="ticket_id",
        )

    assert records[0].label is None


def test_unknown_label_source_is_refused_before_anything_is_written() -> None:
    records = [_record(source_key="t-1")]
    rows = [{"ticket_id": "t-1", "final_answer": "billing"}]

    with pytest.raises(ValueError, match="not one of"):
        harvest_labels(
            records,
            rows,
            field="final_answer",
            source="model_vote",  # type: ignore[arg-type]
            join_on="ticket_id",
        )

    assert records[0].label is None


def test_rewrite_leaves_unrelated_fields_byte_identical(tmp_path: Path) -> None:
    records = [
        _record(source_key="t-1", question="intent", prediction="refund_request", confidence=0.62),
        _record(source_key="t-2", question="intent", prediction="refund_request", confidence=0.71),
        _record(source_key=None, question="intent", prediction="refund_request", confidence=0.44),
    ]
    path = tmp_path / "records.jsonl"
    write_records(records, path)
    before = path.read_text(encoding="utf-8").splitlines()
    assert len(before) == 3
    assert '"label":null' in before[0]

    rows = [{"ticket_id": "t-1", "final_answer": "refund_request"}]
    report = harvest_file(path, rows, field="final_answer", source="silver", join_on="ticket_id")

    assert report.n_applied == 1
    assert report.applications[0].question_key == "intent"
    after = path.read_text(encoding="utf-8").splitlines()
    assert len(after) == len(before)
    assert path.read_text(encoding="utf-8").endswith("\n")

    # The two records no row matched are the same bytes they were before the harvest.
    assert after[1] == before[1]
    assert after[2] == before[2]

    # The labeled record changed in exactly two places and nowhere else.
    expected_first = (
        before[0]
        .replace('"label":null', '"label":"refund_request"')
        .replace('"label_source":null', '"label_source":"silver"')
    )
    assert expected_first != before[0]
    assert path.read_text(encoding="utf-8") == "\n".join([expected_first, *before[1:]]) + "\n"

    parsed_before = json.loads(before[0])
    parsed_after = json.loads(after[0])
    assert set(parsed_after) == set(parsed_before)
    for key, value in parsed_before.items():
        if key in ("label", "label_source"):
            continue
        assert parsed_after[key] == value
    stored = read_records(path)[0]
    assert stored.is_silver is True
    assert stored.is_correct is True
    assert stored.confidence == 0.62


def test_rewrite_preserves_foreign_formatting_and_nested_fields(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    first = (
        '{"id": "rec_a", "question_key": "lang", "prediction": "ko", "confidence": 0.9, '
        '"label": null, "label_source": null, "segment": {"lang": "ko", "note": "라벨"}, '
        '"cost_usd": 0.0125}'
    )
    second = (
        '{"id": "rec_b", "question_key": "lang", "prediction": "en", "confidence": 0.4, '
        '"label": "ko", "label_source": "human_review"}'
    )
    path.write_text(f"{first}\n{second}", encoding="utf-8")

    changed = rewrite_labels_atomic(
        path,
        [LabelApplication("rec_a", "lang", "ko", "human_override")],
    )

    assert changed == 1
    expected_first = first.replace('"label": null', '"label": "ko"').replace(
        '"label_source": null', '"label_source": "human_override"'
    )
    # No trailing newline is added, the spacing survives, and the other line is untouched.
    assert path.read_text(encoding="utf-8") == f"{expected_first}\n{second}"
    # The temp file the rewrite used is gone: only the records file is left behind.
    assert [entry.name for entry in tmp_path.iterdir()] == ["records.jsonl"]


def test_rewrite_reports_nothing_changed_when_no_application_matches(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    write_records([_record(source_key="t-1", question="intent")], path)
    before = path.read_text(encoding="utf-8")

    changed = rewrite_labels_atomic(
        path, [LabelApplication("rec_missing", "intent", "billing", "human_override")]
    )

    assert changed == 0
    assert path.read_text(encoding="utf-8") == before


def test_second_harvest_keeps_the_labels_it_already_wrote(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    write_records(
        [
            _record(source_key="t-1", question="intent"),
            _record(source_key="t-2", question="intent"),
        ],
        path,
    )
    rows = [
        {"ticket_id": "t-1", "final_answer": "refund_request"},
        {"ticket_id": "t-2", "final_answer": "refund_request"},
    ]

    first = harvest_file(
        path, rows, field="final_answer", source="human_override", join_on="ticket_id"
    )
    after_first = path.read_text(encoding="utf-8")
    second = harvest_file(
        path, rows, field="final_answer", source="human_override", join_on="ticket_id"
    )

    assert first.n_applied == 2
    assert second.n_applied == 0
    assert second.n_kept_existing == 2
    assert second.kept_existing_question_keys == ("intent",)
    assert path.read_text(encoding="utf-8") == after_first
    assert [record.is_gold for record in read_records(path)] == [True, True]


def test_ingest_then_harvest_turns_a_resolution_log_into_ground_truth(tmp_path: Path) -> None:
    mapping_file = tmp_path / "ingest-map.yaml"
    mapping_file.write_text(
        "version: 1\n"
        "field_map:\n"
        "  question_key: question\n"
        "  prediction: answer\n"
        "defaults:\n"
        "  question_type: choice\n"
        "label_from:\n"
        "  field: final_department\n"
        "  source: human_override\n"
        "  join_on: ticket_id\n",
        encoding="utf-8",
    )
    raw = tmp_path / "requests.jsonl"
    raw.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {
                    "ticket_id": "t-1",
                    "model": "jev-1.13.0",
                    "question": "department",
                    "answer": "billing",
                    "confidence": 0.9,
                },
                {
                    "ticket_id": "t-2",
                    "model": "jev-1.13.0",
                    "question": "department",
                    "answer": "billing",
                    "confidence": 0.7,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    records_file = tmp_path / "records.jsonl"
    ingest_files([raw], records_file, load_ingest_map(mapping_file))

    spec = load_ingest_map(mapping_file).label_spec()
    assert spec is not None
    assert spec == LabelFromSpec(
        field="final_department", source="human_override", join_on="ticket_id"
    )
    assert [record.source_key for record in read_records(records_file)] == ["t-1", "t-2"]

    resolution = _resolution_rows(
        tmp_path,
        [
            {"ticket_id": "t-1", "final_department": "billing"},
            {"ticket_id": "t-2", "final_department": "technical"},
            {"ticket_id": "t-3", "final_department": "sales"},
        ],
    )
    report = harvest_file(
        records_file,
        iter_rows(resolution),
        field=spec.field,
        source=spec.source,  # type: ignore[arg-type]
        join_on=spec.join_on,
    )

    assert report.n_applied == 2
    assert report.n_unmatched == 1
    assert report.unmatched_keys == ("t-3",)
    assert report.applied_question_keys == ("department",)
    harvested = read_records(records_file)
    assert [record.label for record in harvested] == ["billing", "technical"]
    assert [record.is_correct for record in harvested] == [True, False]
    assert all(record.is_gold for record in harvested)
    assert [record.source_key for record in harvested] == ["t-1", "t-2"]


def test_source_key_can_be_renamed_like_any_other_field(tmp_path: Path) -> None:
    mapping_file = tmp_path / "ingest-map.yaml"
    mapping_file.write_text(
        "version: 1\n"
        "field_map:\n"
        "  question_key: question\n"
        "  prediction: answer\n"
        "  source_key: request_id\n"
        "defaults:\n"
        "  question_type: choice\n",
        encoding="utf-8",
    )
    raw = tmp_path / "requests.jsonl"
    raw.write_text(
        json.dumps(
            {
                "request_id": 4711,
                "model": "jev-1.13.0",
                "question": "department",
                "answer": "billing",
                "confidence": 0.9,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    records_file = tmp_path / "records.jsonl"

    ingest_files([raw], records_file, load_ingest_map(mapping_file))

    assert read_records(records_file)[0].source_key == "4711"


def test_half_filled_label_from_raises_instead_of_being_ignored(tmp_path: Path) -> None:
    path = tmp_path / "ingest-map.yaml"
    path.write_text("version: 1\nlabel_from:\n  field: final_department\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing source, join_on"):
        load_ingest_map(path).label_spec()


def test_label_from_with_an_unknown_source_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "ingest-map.yaml"
    path.write_text(
        "version: 1\n"
        "label_from:\n"
        "  field: final_department\n"
        "  source: model_vote\n"
        "  join_on: ticket_id\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected one of"):
        load_ingest_map(path).label_spec()


# --- guards added after a real fixture wrote a department answer onto intent records ---------
# A join key is shared by every question of a request, so an unscoped harvest matched the intent
# record too and wrote "billing" — a class the intent question cannot produce — reporting it as a
# successful apply. A label that is merely well-formed must never reach the label file.


def test_a_label_the_questions_own_classes_cannot_produce_is_refused() -> None:
    records = [
        _record(
            source_key="t-1",
            probabilities={"refund_request": 0.88, "check_balance": 0.08, "other": 0.04},
        )
    ]
    report = harvest_labels(
        records,
        [{"ticket_id": "t-1", "final_answer": "billing"}],
        field="final_answer",
        source="human_override",
        join_on="ticket_id",
    )

    assert report.n_applied == 0
    assert report.n_unlisted_label == 1
    assert records[0].label is None
    assert report.unlisted_labels == ("t-1:department=billing",)
    assert not report.ok


def test_question_scope_leaves_the_requests_other_questions_untouched() -> None:
    department = _record(source_key="t-1", question="department", probabilities={"billing": 0.9})
    intent = _record(source_key="t-1", question="intent", probabilities={"billing": 0.9})
    records = [department, intent]

    report = harvest_labels(
        records,
        [{"ticket_id": "t-1", "final_answer": "billing"}],
        field="final_answer",
        source="human_override",
        join_on="ticket_id",
        question="department",
    )

    assert report.n_applied == 1
    assert report.n_other_question == 1
    # The harvest replaces entries in the list, so read them back through it.
    assert records[0] is not department
    assert records[0].label == "billing"
    assert records[1] is intent
    assert records[1].label is None  # a fine label value, but not for this question
    assert report.applied_question_keys == ("department",)


def test_allow_unlisted_is_the_documented_escape_hatch() -> None:
    records = [_record(source_key="t-1", probabilities={"refund_request": 0.88, "other": 0.12})]
    report = harvest_labels(
        records,
        [{"ticket_id": "t-1", "final_answer": "billing"}],
        field="final_answer",
        source="human_override",
        join_on="ticket_id",
        allow_unlisted=True,
    )

    assert report.n_applied == 1
    assert report.n_unlisted_label == 0
    assert records[0].label == "billing"


def test_choice_without_a_probabilities_map_is_still_accepted() -> None:
    records = [_record(source_key="t-1")]
    report = harvest_labels(
        records,
        [{"ticket_id": "t-1", "final_answer": "billing"}],
        field="final_answer",
        source="human_override",
        join_on="ticket_id",
    )

    assert report.n_applied == 1  # nothing to check against, so the mapping is the contract


def test_noul_answer_outside_yes_no_is_refused() -> None:
    records = [
        _record(source_key="t-1", question="is_urgent", question_type="noul", prediction="yes")
    ]
    report = harvest_labels(
        records,
        [{"ticket_id": "t-1", "final_answer": "maybe"}],
        field="final_answer",
        source="human_override",
        join_on="ticket_id",
    )

    assert report.n_applied == 0
    assert report.n_unlisted_label == 1


def test_score_answer_that_is_not_a_number_is_refused() -> None:
    records = [
        _record(source_key="t-1", question="satisfaction", question_type="score", prediction="0.7")
    ]
    report = harvest_labels(
        records,
        [{"ticket_id": "t-1", "final_answer": "very happy"}],
        field="final_answer",
        source="human_override",
        join_on="ticket_id",
    )

    assert report.n_applied == 0
    assert report.n_unlisted_label == 1
