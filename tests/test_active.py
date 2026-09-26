"""Tests for the active-learning labeling queue.

The assertions here are about directions and counts the specification's priority order produces,
not about recorded output: a near-threshold record must come out before a far one, a flat
distribution must be described as a criteria problem rather than as model uncertainty, a sparse
confidence bin must be represented, and an empty, fully labeled or probability-free log must
produce a queue instead of an exception.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from jeval.active import (
    FLAT_ENTROPY_MIN,
    KEY_COLUMN,
    LABEL_COLUMN,
    MIN_LABELED_PER_BIN,
    PRIORITY_BAND,
    PRIORITY_FLAT,
    PRIORITY_NAMES,
    PRIORITY_REMAINDER,
    PRIORITY_SPARSE,
    SHEET_COLUMNS,
    apply_labels,
    build_queue,
    export_session,
    format_queue,
    priority_breakdown,
)
from jeval.calibration import DEFAULT_N_BINS
from jeval.report.model import ThresholdResult
from jeval.schema import DecisionRecord, LabelSource, QuestionType
from jeval.synth import SynthSpec, generate

MODEL = "jev-1.13.0"

# Skewed enough that the record is never mistaken for a flat distribution:
# normalized entropy 0.40 against the FLAT_ENTROPY_MIN floor of 0.90.
SKEWED = {"refund_request": 0.92, "other": 0.08}


def _skewed() -> dict[str, float]:
    """A fresh copy: records must not share one probability map."""
    return dict(SKEWED)


def _record(
    record_id: str,
    confidence: float,
    *,
    question_key: str = "intent",
    prediction: str = "refund_request",
    question_type: QuestionType = "choice",
    probabilities: dict[str, float] | None = None,
    label: str | None = None,
    label_source: LabelSource | None = None,
) -> DecisionRecord:
    """One decision record with an explicit id, so queue assertions can name records."""
    return DecisionRecord(
        id=record_id,
        model=MODEL,
        question_key=question_key,
        question_type=question_type,
        prediction=prediction,
        confidence=confidence,
        probabilities=probabilities,
        label=label,
        label_source=label_source,
    )


def _labeled(bin_index: int, count: int) -> list[DecisionRecord]:
    """``count`` labeled records packed into one confidence bin, so its occupancy is exact."""
    confidence = (bin_index + 0.5) / DEFAULT_N_BINS
    return [
        _record(
            f"lbl_{bin_index}_{position}",
            confidence,
            probabilities=_skewed(),
            label="refund_request",
            label_source="human_review",
        )
        for position in range(count)
    ]


def _dense_bins(*, except_bin: int | None = None, except_count: int = 3) -> list[DecisionRecord]:
    """A log with every confidence bin at the coverage floor, and one bin deliberately thin."""
    records: list[DecisionRecord] = []
    for index in range(DEFAULT_N_BINS):
        count = except_count if index == except_bin else MIN_LABELED_PER_BIN
        records.extend(_labeled(index, count))
    return records


def _four_class_log() -> list[DecisionRecord]:
    """One fixture with exactly one candidate in each priority class.

    Bins 0-8 hold the coverage floor, so ``rec_rest`` and ``rec_flat`` sit in covered regions and
    only bin 9 (three labeled records) is sparse. The candidate threshold 0.80 puts ``rec_band``
    inside the band and every other record well outside it.
    """
    return [
        *_dense_bins(except_bin=9, except_count=3),
        _record("rec_band", 0.82, probabilities=_skewed()),
        _record(
            "rec_flat",
            0.36,
            prediction="a",
            probabilities={"a": 0.36, "b": 0.32, "c": 0.32},
        ),
        _record("rec_sparse", 0.95, probabilities=_skewed()),
        _record("rec_rest", 0.65, probabilities=_skewed()),
    ]


def test_near_threshold_record_outranks_far_ones() -> None:
    """A label on the decision line is worth more than a label far from it."""
    records = [
        *_dense_bins(),
        _record("rec_far_low", 0.40, probabilities=_skewed()),
        _record("rec_near", 0.85, probabilities=_skewed()),
        _record("rec_far_high", 0.99, probabilities=_skewed()),
    ]
    queue = build_queue(records, thresholds={"intent": 0.86})

    assert queue[0].record_id == "rec_near"
    assert queue[0].priority == PRIORITY_BAND
    assert queue[0].confidence == pytest.approx(0.85)
    assert "0.86" in queue[0].reason
    assert "threshold" in queue[0].reason

    by_id = {item.record_id: item for item in queue}
    assert by_id["rec_far_low"].priority != PRIORITY_BAND
    assert by_id["rec_far_high"].priority != PRIORITY_BAND
    assert by_id["rec_far_low"].reason != queue[0].reason


def test_band_distance_breaks_ties_inside_the_band() -> None:
    """Within the band, the record closest to the candidate comes first."""
    records = [
        *_dense_bins(),
        _record("rec_edge", 0.80, probabilities=_skewed()),
        _record("rec_middle", 0.86, probabilities=_skewed()),
    ]
    queue = build_queue(records, thresholds={"intent": 0.81})

    assert [item.record_id for item in queue] == ["rec_edge", "rec_middle"]
    assert [item.priority for item in queue] == [PRIORITY_BAND, PRIORITY_BAND]


def test_priority_classes_come_out_in_specification_order() -> None:
    """Band, then flat, then sparse bin, then everything else."""
    queue = build_queue(_four_class_log(), thresholds={"intent": 0.80})

    assert [item.priority for item in queue] == [
        PRIORITY_BAND,
        PRIORITY_FLAT,
        PRIORITY_SPARSE,
        PRIORITY_REMAINDER,
    ]
    assert [item.record_id for item in queue] == [
        "rec_band",
        "rec_flat",
        "rec_sparse",
        "rec_rest",
    ]
    assert [item.priority_name for item in queue] == list(PRIORITY_NAMES.values())


def test_flat_distribution_is_reported_as_a_criteria_problem() -> None:
    """A flat answer is evidence about the question, not about the model's confidence."""
    records = [
        *_dense_bins(),
        _record(
            "rec_flat",
            0.36,
            prediction="a",
            probabilities={"a": 0.36, "b": 0.32, "c": 0.32},
        ),
        _record(
            "rec_peaked",
            0.36,
            prediction="a",
            probabilities={"a": 0.97, "b": 0.02, "c": 0.01},
        ),
    ]
    queue = build_queue(records)
    by_id = {item.record_id: item for item in queue}

    flat = by_id["rec_flat"]
    assert flat.priority == PRIORITY_FLAT
    assert "criteria" in flat.reason
    assert "question" in flat.reason
    assert "3 classes" in flat.reason
    # The reason must not be an uncertainty claim about the model.
    assert "uncertain" not in flat.reason.lower()

    # The same bin, the same confidence, a peaked distribution: an ordinary queue position.
    peaked = by_id["rec_peaked"]
    assert peaked.priority == PRIORITY_REMAINDER
    assert peaked.reason != flat.reason
    assert [item.record_id for item in queue].index("rec_flat") < [
        item.record_id for item in queue
    ].index("rec_peaked")


def test_flat_ordering_prefers_the_flattest_distribution() -> None:
    """Both records are flat; the more uniform one is the stronger signal about the criteria."""
    records = [
        *_dense_bins(),
        _record(
            "rec_nearly_uniform",
            0.35,
            prediction="a",
            probabilities={"a": 0.35, "b": 0.34, "c": 0.31},
        ),
        _record(
            "rec_less_uniform",
            0.37,
            prediction="a",
            probabilities={"a": 0.38, "b": 0.33, "c": 0.29},
        ),
    ]
    queue = build_queue(records)

    assert [item.priority for item in queue] == [PRIORITY_FLAT, PRIORITY_FLAT]
    assert [item.record_id for item in queue] == ["rec_nearly_uniform", "rec_less_uniform"]
    # Sanity: the fixture is only meaningful if both records clear the flat floor.
    assert FLAT_ENTROPY_MIN <= 0.999


def test_sparse_confidence_bins_are_represented() -> None:
    """A region with no labels is where nothing is measured, so it must reach the queue."""
    records = [
        *_labeled(8, MIN_LABELED_PER_BIN + 20),
        *_labeled(9, MIN_LABELED_PER_BIN + 20),
        *_labeled(4, 5),
        *_labeled(2, 1),
        _record("rec_gap", 0.65, probabilities=_skewed()),
        _record("rec_thin", 0.45, probabilities=_skewed()),
        _record("rec_almost_empty", 0.25, probabilities=_skewed()),
        _record("rec_covered", 0.95, probabilities=_skewed()),
    ]
    queue = build_queue(records)
    by_id = {item.record_id: item for item in queue}
    order = [item.record_id for item in queue]

    gap = by_id["rec_gap"]
    assert gap.priority == PRIORITY_SPARSE
    assert "0.60-0.70" in gap.reason
    assert "no labeled records" in gap.reason

    # Two sparse records: the emptier bin comes first.
    assert by_id["rec_thin"].priority == PRIORITY_SPARSE
    assert "5 labeled records" in by_id["rec_thin"].reason
    assert order.index("rec_gap") < order.index("rec_thin")

    # One record in a bin: singular, and still a gap, ranked ahead of the busier thin bin.
    almost_empty = by_id["rec_almost_empty"]
    assert almost_empty.priority == PRIORITY_SPARSE
    assert "1 labeled record:" in almost_empty.reason
    assert order.index("rec_almost_empty") < order.index("rec_thin")

    # A covered bin is queued only as a top-up, after every sparse-bin record.
    assert by_id["rec_covered"].priority == PRIORITY_REMAINDER
    assert order.index("rec_thin") < order.index("rec_covered")


def test_empty_and_degenerate_inputs_do_not_raise() -> None:
    """An empty queue is a legitimate answer, not an error."""
    assert build_queue([]) == ()
    assert build_queue([], thresholds={"intent": 0.5}, limit=0) == ()
    assert format_queue(()) == "labeling queue is empty: every record already has a label."
    assert set(priority_breakdown(()).values()) == {0}

    labeled_only = _dense_bins()
    assert all(record.is_labeled for record in labeled_only)
    assert build_queue(labeled_only) == ()

    # No probability map, a single-class map, an all-zero map and a negative map are all
    # "not flat" rather than a crash.
    no_distributions = [
        _record("rec_none", 0.44),
        _record("rec_single", 0.48, probabilities={"only": 1.0}),
        _record("rec_zero", 0.52, probabilities={"a": 0.0, "b": 0.0}),
        _record("rec_negative", 0.58, probabilities={"a": -1.0, "b": 2.0}),
    ]
    queue = build_queue(no_distributions)
    assert len(queue) == 4
    assert all(item.priority != PRIORITY_FLAT for item in queue)
    assert all(item.reason for item in queue)


def test_queue_is_deterministic_and_the_seed_only_rotates_ties() -> None:
    """Same input and seed, same queue; another seed reorders ties without losing a record."""
    records = [
        *_dense_bins(),
        *[_record(f"rec_rest_{index}", 0.65, probabilities=_skewed()) for index in range(8)],
    ]
    first = build_queue(records, seed=0)

    assert build_queue(records, seed=0) == first
    # The sort keys do not depend on input order, so neither may the queue.
    assert build_queue(list(reversed(records)), seed=0) == first

    other = build_queue(records, seed=1)
    # The fixtures are fixed, so this comparison is deterministic, not flaky: with all eight
    # records in one class, a different digest order has to show up somewhere.
    assert [item.record_id for item in other] != [item.record_id for item in first]
    assert {item.record_id for item in other} == {item.record_id for item in first}
    assert [item.priority for item in other] == [item.priority for item in first]


def test_limit_truncates_without_dropping_priority_classes() -> None:
    """``limit`` cuts the tail, never reshuffles the head."""
    records = _four_class_log()

    head = build_queue(records, thresholds={"intent": 0.80}, limit=2)
    assert [item.priority for item in head] == [PRIORITY_BAND, PRIORITY_FLAT]
    assert build_queue(records, thresholds={"intent": 0.80}, limit=0) == ()
    with pytest.raises(ValueError, match="limit"):
        build_queue(records, limit=-1)


def test_format_queue_prints_one_plain_line_per_record() -> None:
    """Plain text a terminal can render: no ANSI, one line per record, same facts as the sheet."""
    queue = build_queue(_four_class_log(), thresholds={"intent": 0.80})
    text = format_queue(queue)

    assert "\x1b" not in text
    lines = text.splitlines()
    assert len(lines) == len(queue)
    for item, line in zip(queue, lines, strict=True):
        assert item.record_id in line
        assert item.question_key in line
        assert f"{item.confidence:.2f}" in line
        assert item.prediction in line
        assert item.reason in line
        assert item.priority_name in line


def test_priority_breakdown_counts_every_class_in_queue_order() -> None:
    """The breakdown always names all four classes, so a partial queue cannot look complete."""
    queue = build_queue(_four_class_log(), thresholds={"intent": 0.80})

    breakdown = priority_breakdown(queue)
    assert list(breakdown) == list(PRIORITY_NAMES.values())
    assert breakdown == {
        "threshold band": 1,
        "flat distribution": 1,
        "sparse bin": 1,
        "remainder": 1,
    }
    assert priority_breakdown(()) == dict.fromkeys(PRIORITY_NAMES.values(), 0)


def test_thresholds_accept_a_sweep_result_and_reject_a_bad_one() -> None:
    """A candidate threshold is ranked against, never invented and never clamped."""
    records = [*_dense_bins(), _record("rec_near", 0.85, probabilities=_skewed())]
    result = ThresholdResult(
        action="auto_refund",
        question="intent",
        when="refund_request",
        threshold=0.86,
        expected_cost_per_case=12.4,
        auto_rate=0.58,
        accuracy_auto=0.94,
        ci_low=0.83,
        ci_high=0.93,
    )

    assert build_queue(records, thresholds={"intent": result})[0].record_id == "rec_near"
    assert build_queue(records, thresholds={"intent": 0.86})[0].record_id == "rec_near"

    # A withheld recommendation is a question with no candidate, not a band at nan.
    withheld = ThresholdResult(
        action="auto_refund",
        question="intent",
        when="refund_request",
        threshold=float("nan"),
        expected_cost_per_case=float("nan"),
        auto_rate=float("nan"),
        accuracy_auto=float("nan"),
        ci_low=float("nan"),
        ci_high=float("nan"),
    )
    no_band = build_queue(records, thresholds={"intent": withheld})
    assert all(item.priority != PRIORITY_BAND for item in no_band)

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        build_queue(records, thresholds={"intent": 1.5})
    with pytest.raises(ValueError, match="thresholds"):
        build_queue(records, thresholds={"intent": "high"})  # type: ignore[dict-item]


def test_export_session_and_apply_labels_round_trip(tmp_path: Path) -> None:
    """The sheet jeval writes is the sheet a reviewer fills in and jeval reads back."""
    records = [
        _record("rec_a", 0.85, probabilities=_skewed()),
        _record(
            "rec_b",
            0.70,
            prediction="x",
            probabilities={"x": 0.7, "y": 0.2, "z": 0.1},
        ),
        _record("rec_c", 0.70, probabilities=_skewed(), prediction="other"),
    ]
    queue = build_queue(records, thresholds={"intent": 0.86})

    sheet = export_session(queue, tmp_path / "session" / "labels.csv")
    assert sheet.exists()
    with sheet.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0].keys()) == list(SHEET_COLUMNS)
    assert [row["record_id"] for row in rows] == [item.record_id for item in queue]
    assert all(row["label"] == "" for row in rows)

    by_id = {row["record_id"]: row for row in rows}
    by_id["rec_a"]["label"] = "refund_request"
    by_id["rec_c"]["label"] = "other"
    # rec_b is left unanswered, and one row names a record that is not in the log.
    rows.append({"record_id": "rec_missing", "label": "other"})

    filled = tmp_path / "labels-filled.csv"
    with filled.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SHEET_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SHEET_COLUMNS})

    with filled.open(newline="", encoding="utf-8") as handle:
        applied = apply_labels(
            records,
            csv.DictReader(handle),
            key=KEY_COLUMN,
            label=LABEL_COLUMN,
            source="human_review",
        )

    assert applied == 2
    assert records[0].label == "refund_request"
    assert records[0].label_source == "human_review"
    assert records[0].is_correct is True
    assert records[2].label == "other"
    assert records[2].label_source == "human_review"
    assert records[1].is_labeled is False

    # The labels buy coverage: rec_b is now the only record still worth a label, and it sits in
    # a confidence bin where the two new labels left nothing behind.
    remaining = build_queue(records, thresholds={"intent": 0.86}, limit=10)
    assert [item.record_id for item in remaining] == ["rec_b"]
    assert remaining[0].priority == PRIORITY_SPARSE


def test_apply_labels_normalizes_a_binary_answer_to_the_schema_form() -> None:
    """A human answer reaches the record through the schema, so ``YES`` cannot mean ``yes``-only."""
    record = _record(
        "rec_noul",
        0.4,
        question_type="noul",
        prediction="yes",
        probabilities={"yes": 0.7, "no": 0.3},
    )

    applied = apply_labels(
        [record],
        [{"record_id": "rec_noul", "label": "YES"}],
        key=KEY_COLUMN,
        label=LABEL_COLUMN,
        source="human_review",
    )

    assert applied == 1
    assert record.label == "yes"
    assert record.is_correct is True


def test_apply_labels_counts_records_once_and_keeps_the_last_answer() -> None:
    """A repeated row is one labeled record, and the sheet's last word wins."""
    records = [
        _record(
            "rec_noul",
            0.4,
            question_type="noul",
            prediction="yes",
            probabilities={"yes": 0.7, "no": 0.3},
        ),
        _record("rec_other", 0.6, probabilities=_skewed()),
    ]

    applied = apply_labels(
        records,
        [
            {"record_id": "rec_noul", "label": "yes"},
            {"record_id": "rec_noul", "label": "no"},
            {"record_id": "rec_other", "label": "   "},
        ],
        key=KEY_COLUMN,
        label=LABEL_COLUMN,
        source="human_override",
    )

    assert applied == 1
    assert records[0].label == "no"
    assert records[0].label_source == "human_override"
    assert records[0].is_correct is False
    assert records[1].is_labeled is False


def test_apply_labels_refuses_a_sheet_with_the_wrong_columns(tmp_path: Path) -> None:
    """A column-name mistake must not be reported as "nothing to apply"."""
    records = [_record("rec_a", 0.5, probabilities=_skewed())]

    with pytest.raises(ValueError, match="columns present"):
        apply_labels(
            records,
            [{"id": "rec_a", "answer": "x"}],
            key=KEY_COLUMN,
            label=LABEL_COLUMN,
            source="human_review",
        )
    assert records[0].is_labeled is False

    with pytest.raises(ValueError, match="label source"):
        apply_labels(
            records,
            [{"record_id": "rec_a", "label": "x"}],
            key=KEY_COLUMN,
            label=LABEL_COLUMN,
            source="robot",  # type: ignore[arg-type]
        )
    assert records[0].is_labeled is False

    empty = export_session((), tmp_path / "empty.csv")
    with empty.open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == []
    assert apply_labels(records, [], key=KEY_COLUMN, label=LABEL_COLUMN, source="human_review") == 0
    assert records[0].is_labeled is False


def test_a_synthetic_log_queues_only_unlabeled_records_in_priority_order() -> None:
    """On a real-shaped log: nothing already labeled is queued, and the head is the band."""
    log = generate(SynthSpec(n=400, mode="calibrated", model=MODEL, seed=5))
    candidates = [
        record.model_copy(update={"id": f"rec_open_{index}", "label": None, "label_source": None})
        for index, record in enumerate(log[:200])
    ]
    records = [*log, *candidates]

    queue = build_queue(records, thresholds={"department": 0.90}, limit=10)

    assert len(queue) == 10
    priorities = [item.priority for item in queue]
    assert priorities == sorted(priorities)
    assert priorities[0] == PRIORITY_BAND
    assert {item.record_id for item in queue} <= {record.id for record in candidates}
    assert build_queue(records, thresholds={"department": 0.90}, limit=10) == queue


def test_a_yes_no_record_is_ranked_against_its_threshold_on_the_same_scale() -> None:
    """P(yes) = 0.8 is right with probability 0.8, so it sits in the band of a 0.8 line."""
    from jeval.active import PRIORITY_BAND, build_queue
    from jeval.schema import normalize_record

    record = normalize_record(
        {
            "model": "m",
            "question_key": "is_urgent",
            "question_type": "noul",
            "probability_positive": 0.8,
        }
    )
    queue = build_queue([record], thresholds={"is_urgent": 0.8})
    assert len(queue) == 1
    assert queue[0].priority == PRIORITY_BAND
    assert queue[0].confidence == pytest.approx(0.8)
