"""Tests for the per-segment cost sweep: does one threshold per segment pay?

The fixtures here are built so every figure can be checked with a calculator. Each record
predicts the action's class ``refund_request`` at confidence 0.60 or 0.95, and with
``cost_false_accept = 30``, ``cost_escalate = 10`` and ``cost_false_reject = 0`` a segment's cost
curve has exactly three levels:

- run everything: ``30 * (wrong) / n``
- run only the 0.95 band: ``(30 * wrong_high + 10 * low) / n``
- run nothing: ``10.0``

A record is "wrong" when its gold label is another class, "right" when it is the action's class.
"""

from __future__ import annotations

import math

import pytest

from jeval.costs import (
    DEFAULT_MIN_SEGMENT_RECORDS,
    MIN_GOLD_RECORDS,
    NOT_WORTH_SPLITTING,
    SEGMENT_UNKNOWN,
    SPLIT_COST_TOLERANCE,
    CostAction,
    SegmentThreshold,
    sweep,
    sweep_by_segment,
)
from jeval.schema import DecisionRecord

WHEN = "refund_request"
OTHER = "check_balance"
LOW = 0.60
HIGH = 0.95

ACTION = CostAction(
    name="auto_refund",
    question="intent",
    when=WHEN,
    cost_false_accept=30.0,
    cost_escalate=10.0,
    cost_false_reject=0.0,
)


def record(
    prediction: str,
    confidence: float,
    label: str | None,
    *,
    lang: str | None = None,
    question: str = "intent",
    label_source: str = "human_override",
) -> DecisionRecord:
    """One decision record with just enough fields for the cost engine."""
    return DecisionRecord.model_validate(
        {
            "model": "jev-1.13.0",
            "question_key": question,
            "question_type": "choice",
            "prediction": prediction,
            "confidence": confidence,
            "label": label,
            "label_source": None if label is None else label_source,
            "segment": {} if lang is None else {"lang": lang},
        }
    )


def block(
    lang: str | None,
    *,
    low_wrong: int,
    low_right: int,
    high_wrong: int,
    high_right: int,
) -> list[DecisionRecord]:
    """One segment: every record predicts ``refund_request`` at 0.60 or at 0.95.

    ``lang=None`` leaves the records without a segment key at all, which lands them under
    :data:`SEGMENT_UNKNOWN`.
    """
    return (
        [record(WHEN, LOW, OTHER, lang=lang) for _ in range(low_wrong)]
        + [record(WHEN, LOW, WHEN, lang=lang) for _ in range(low_right)]
        + [record(WHEN, HIGH, OTHER, lang=lang) for _ in range(high_wrong)]
        + [record(WHEN, HIGH, WHEN, lang=lang) for _ in range(high_right)]
    )


def by_value(rows: tuple[SegmentThreshold, ...]) -> dict[str, SegmentThreshold]:
    return {row.segment_value: row for row in rows}


def honest_en() -> list[DecisionRecord]:
    """100 records: a bad 0.60 band, then 70 cases at 0.95 that are right 67 times.

    Running the 0.95 band costs ``(30 * 3 + 10 * 30) / 100 = 3.90`` per case, against 6.90 for
    running everything and 10.00 for running nothing, so its optimum is 3.90 at 0.95.
    """
    return block("en", low_wrong=20, low_right=10, high_wrong=3, high_right=67)


def overconfident_ko() -> list[DecisionRecord]:
    """100 records: 0.95 confidence with a 25-in-70 error rate, so nothing is worth automating.

    The 0.95 band would cost ``(30 * 25 + 10 * 30) / 100 = 10.50`` per case, running everything
    15.00, and escalating all of it 10.00: its optimum is 10.00 at 1.0.
    """
    return block("ko", low_wrong=25, low_right=5, high_wrong=25, high_right=45)


def two_segment_log() -> list[DecisionRecord]:
    """The honest segment and the overconfident one, 200 gold records of ``intent``."""
    return honest_en() + overconfident_ko()


# --------------------------------------------------------------------------------------
# the global line these segments are compared against
# --------------------------------------------------------------------------------------


def test_the_global_sweep_on_the_fixture_is_the_hand_computed_line() -> None:
    # 73 of 200 records are wrong; the 0.95 band holds 28 of them and the 0.60 band 60 records.
    global_result = sweep(ACTION, two_segment_log(), seed=0)

    assert global_result.threshold == pytest.approx(0.95)
    assert global_result.expected_cost_per_case == pytest.approx(7.20)
    assert global_result.n_records == 200


# --------------------------------------------------------------------------------------
# worth splitting: a segment that pays and a segment that does not
# --------------------------------------------------------------------------------------


def test_an_overconfident_segment_gets_a_higher_threshold_and_is_worth_splitting() -> None:
    rows = by_value(sweep_by_segment(ACTION, two_segment_log(), segment_key="lang"))

    assert sorted(rows) == ["en", "ko"]
    overconfident = rows["ko"]
    assert overconfident.segment_key == "lang"
    assert overconfident.label == "lang = ko"
    assert overconfident.n_records == 100
    # Escalating every case is cheaper than automating a band that is wrong one time in three.
    assert overconfident.threshold == pytest.approx(1.0)
    assert overconfident.expected_cost_per_case == pytest.approx(10.00)
    assert overconfident.auto_rate == pytest.approx(0.0)
    assert overconfident.cost_delta_vs_global == pytest.approx(-0.50)
    assert overconfident.worth_splitting is True
    assert overconfident.reason == ""


def test_the_segment_that_agrees_with_the_global_line_is_not_worth_splitting() -> None:
    rows = by_value(sweep_by_segment(ACTION, two_segment_log(), segment_key="lang"))

    honest = rows["en"]
    assert honest.threshold == pytest.approx(0.95)
    assert honest.expected_cost_per_case == pytest.approx(3.90)
    assert honest.auto_rate == pytest.approx(0.70)
    assert honest.n_records == 100
    # Its own optimum is the global one, so adopting it would change nothing at all.
    assert honest.cost_delta_vs_global == pytest.approx(0.0)
    assert honest.worth_splitting is False
    assert honest.reason.startswith(NOT_WORTH_SPLITTING)
    assert "within one sweep step" in honest.reason
    assert "under the 2% bar" in honest.reason


def test_exactly_one_segment_of_the_fixture_pays_to_split() -> None:
    rows = sweep_by_segment(ACTION, two_segment_log(), segment_key="lang")

    assert [row.worth_splitting for row in rows] == [False, True]
    assert [row.threshold for row in rows] == [pytest.approx(0.95), pytest.approx(1.0)]
    # Every segment of the action's question is accounted for, and they add up to the global log.
    assert sum(row.n_records for row in rows) == sweep(ACTION, two_segment_log(), seed=0).n_records


# --------------------------------------------------------------------------------------
# splitting does not pay on homogeneous data
# --------------------------------------------------------------------------------------


def homogeneous_log() -> list[DecisionRecord]:
    """The same 100 records in both segments: identical distributions, so identical optima."""
    return honest_en() + block("ko", low_wrong=20, low_right=10, high_wrong=3, high_right=67)


def test_a_homogeneous_log_is_not_worth_splitting_anywhere() -> None:
    global_result = sweep(ACTION, homogeneous_log(), seed=0)
    rows = sweep_by_segment(ACTION, homogeneous_log(), segment_key="lang")

    assert global_result.threshold == pytest.approx(0.95)
    assert global_result.expected_cost_per_case == pytest.approx(3.90)
    assert [row.threshold for row in rows] == [pytest.approx(0.95), pytest.approx(0.95)]
    assert [row.worth_splitting for row in rows] == [False, False]
    for row in rows:
        assert row.cost_delta_vs_global == pytest.approx(0.0)
        assert row.reason.startswith(NOT_WORTH_SPLITTING)
        assert "splitting does not pay" in row.reason


# --------------------------------------------------------------------------------------
# the two clauses of the rule, one at a time
# --------------------------------------------------------------------------------------


def material_but_not_moved_log() -> list[DecisionRecord]:
    """A segment whose own optimum is far away but whose money barely moves.

    200 records at 0.95 with 67 wrong: automating them costs 10.05 per case against 10.00 for
    escalating every one, so the segment wants 1.00 while the global optimum of the combined
    300 records is 0.95 at 8.00 per case. The move is five steps; the saving is 0.05 per case,
    well inside 2% of 8.00.
    """
    return honest_en() + block("ko", low_wrong=0, low_right=0, high_wrong=67, high_right=133)


def test_a_segment_must_move_money_as_well_as_the_threshold() -> None:
    rows = by_value(sweep_by_segment(ACTION, material_but_not_moved_log(), segment_key="lang"))

    global_result = sweep(ACTION, material_but_not_moved_log(), seed=0)
    assert global_result.threshold == pytest.approx(0.95)
    assert global_result.expected_cost_per_case == pytest.approx(8.00)

    barely = rows["ko"]
    assert barely.threshold == pytest.approx(1.0)
    assert barely.cost_delta_vs_global == pytest.approx(-0.05)
    assert barely.worth_splitting is False
    assert "differs from the global" in barely.reason
    assert "under the 2% bar" in barely.reason
    # The cost change is real but tiny: 0.05 per case against a bar of 2% of 8.00.
    assert (
        abs(barely.cost_delta_vs_global)
        < SPLIT_COST_TOLERANCE * global_result.expected_cost_per_case
    )


def test_one_sweep_step_of_movement_is_grid_noise_not_a_reason_to_split() -> None:
    # On an 11-point grid one step is 0.10, so the overconfident segment's 1.00 sits exactly one
    # step above the global 0.90. The money moves (0.50 per case), the line does not.
    rows = by_value(sweep_by_segment(ACTION, two_segment_log(), segment_key="lang", steps=11))

    assert sweep(ACTION, two_segment_log(), steps=11, seed=0).threshold == pytest.approx(0.90)
    overconfident = rows["ko"]
    assert overconfident.threshold == pytest.approx(1.0)
    assert overconfident.cost_delta_vs_global == pytest.approx(-0.50)
    assert overconfident.worth_splitting is False
    assert "within one sweep step (0.10)" in overconfident.reason
    assert "more than the 2% bar on its own" in overconfident.reason


def test_a_coarser_grid_leaves_the_segment_that_agreed_still_agreeing() -> None:
    rows = by_value(sweep_by_segment(ACTION, two_segment_log(), segment_key="lang", steps=11))

    assert rows["en"].threshold == pytest.approx(0.90)
    assert rows["en"].worth_splitting is False
    assert rows["en"].cost_delta_vs_global == pytest.approx(0.0)


# --------------------------------------------------------------------------------------
# min_records is enforced, and thin segments are returned rather than dropped
# --------------------------------------------------------------------------------------


def thin_segment_log() -> list[DecisionRecord]:
    """The honest segment plus a 40-record one: 30 right and 10 wrong at 0.95.

    Automating those 40 cases costs 7.50 per case against 10.00 for escalating them.
    """
    return honest_en() + block("ja", low_wrong=0, low_right=0, high_wrong=10, high_right=30)


def test_a_segment_below_min_records_is_returned_with_its_reason() -> None:
    rows = by_value(sweep_by_segment(ACTION, thin_segment_log(), segment_key="lang"))

    assert sorted(rows) == ["en", "ja"]
    thin = rows["ja"]
    assert thin.n_records == 40
    assert math.isnan(thin.threshold)
    assert math.isnan(thin.expected_cost_per_case)
    assert math.isnan(thin.auto_rate)
    assert math.isnan(thin.cost_delta_vs_global)
    assert thin.worth_splitting is False
    assert "only 40 gold records" in thin.reason
    assert str(DEFAULT_MIN_SEGMENT_RECORDS) in thin.reason
    assert DEFAULT_MIN_SEGMENT_RECORDS == 100


def test_lowering_min_records_sweeps_the_same_segment() -> None:
    rows = by_value(
        sweep_by_segment(ACTION, thin_segment_log(), segment_key="lang", min_records=40)
    )

    swept = rows["ja"]
    assert len(rows) == 2
    assert swept.threshold == pytest.approx(0.95)
    assert swept.expected_cost_per_case == pytest.approx(7.50)
    assert swept.auto_rate == pytest.approx(1.0)
    assert swept.n_records == 40
    assert math.isfinite(swept.cost_delta_vs_global)


def test_min_records_never_falls_below_the_gold_floor() -> None:
    # A 20-record segment stays unswept even when the caller asks for a 10-record floor: the
    # module's own thirty-record rule is not a caller option.
    log = honest_en() + block("ja", low_wrong=0, low_right=0, high_wrong=5, high_right=15)

    thin = by_value(sweep_by_segment(ACTION, log, segment_key="lang", min_records=10))["ja"]

    assert thin.n_records == 20 < MIN_GOLD_RECORDS
    assert math.isnan(thin.threshold)
    assert f"below the {MIN_GOLD_RECORDS}" in thin.reason


# --------------------------------------------------------------------------------------
# segments that cannot be swept at all
# --------------------------------------------------------------------------------------


def test_a_segment_the_model_never_predicts_the_action_class_in_is_not_swept() -> None:
    records = [record(OTHER, 0.90, WHEN, lang="en") for _ in range(100)]

    rows = sweep_by_segment(ACTION, records, segment_key="lang")

    assert len(rows) == 1
    only = rows[0]
    assert math.isnan(only.threshold)
    assert only.n_records == 100
    assert only.worth_splitting is False
    assert "is predicted as the action's class" in only.reason


def test_a_segment_with_no_case_labeled_the_action_class_is_not_swept() -> None:
    records = [record(WHEN, 0.90, OTHER, lang="en") for _ in range(100)]

    rows = sweep_by_segment(ACTION, records, segment_key="lang")

    assert len(rows) == 1
    only = rows[0]
    assert math.isnan(only.threshold)
    assert math.isnan(only.cost_delta_vs_global)
    assert only.n_records == 100
    assert "is labeled as the action's class" in only.reason


# --------------------------------------------------------------------------------------
# which records make up a segment
# --------------------------------------------------------------------------------------


def test_records_without_the_segment_key_are_counted_under_unknown() -> None:
    log = honest_en() + [record(WHEN, LOW, OTHER) for _ in range(40)]

    rows = sweep_by_segment(ACTION, log, segment_key="lang")

    assert [row.segment_value for row in rows] == ["en", SEGMENT_UNKNOWN]
    unknown = rows[1]
    assert unknown.n_records == 40
    assert math.isnan(unknown.threshold)
    assert "only 40 gold records" in unknown.reason


def test_a_segment_without_the_key_is_still_swept_when_it_is_large_enough() -> None:
    rows = sweep_by_segment(
        ACTION,
        block(None, low_wrong=20, low_right=10, high_wrong=3, high_right=67),
        segment_key="lang",
    )

    assert len(rows) == 1
    assert rows[0].segment_value == SEGMENT_UNKNOWN
    assert rows[0].threshold == pytest.approx(0.95)
    assert rows[0].n_records == 100


def test_silver_labels_and_other_questions_make_no_segment() -> None:
    log = (
        honest_en()
        + [record(WHEN, 0.90, OTHER, lang="ja", label_source="silver") for _ in range(50)]
        + [record(WHEN, 0.90, OTHER, lang="ja", question="severity") for _ in range(50)]
    )

    rows = by_value(sweep_by_segment(ACTION, log, segment_key="lang"))

    assert sorted(rows) == ["en"]
    assert rows["en"].n_records == 100
    assert rows["en"].threshold == pytest.approx(0.95)


def test_every_segment_value_is_present_and_sorted() -> None:
    log = (
        honest_en()
        + overconfident_ko()
        + block("ja", low_wrong=0, low_right=0, high_wrong=10, high_right=30)
    )

    rows = sweep_by_segment(ACTION, log, segment_key="lang")

    assert [row.segment_value for row in rows] == ["en", "ja", "ko"]
    assert [row.worth_splitting for row in rows] == [False, False, True]


# --------------------------------------------------------------------------------------
# edges
# --------------------------------------------------------------------------------------


def test_no_records_gives_no_segments() -> None:
    assert sweep_by_segment(ACTION, [], segment_key="lang") == ()


def test_an_unswept_segment_reports_nan_rather_than_zero() -> None:
    rows = sweep_by_segment(ACTION, thin_segment_log(), segment_key="lang")

    assert [row.segment_value for row in rows] == ["en", "ja"]
    thin = rows[1]
    assert math.isnan(thin.expected_cost_per_case)
    # Absence of a measurement must never read as a measured zero.
    assert thin.expected_cost_per_case != 0.0
    assert thin.auto_rate != 0.0
    assert math.isnan(thin.auto_rate)
    assert math.isnan(thin.cost_delta_vs_global)
    assert thin.threshold != 0.0


def test_sweep_by_segment_rejects_an_unusable_segment_key() -> None:
    with pytest.raises(ValueError, match="segment_key must be a non-empty string"):
        sweep_by_segment(ACTION, two_segment_log(), segment_key="")
    with pytest.raises(ValueError, match="segment_key must be a non-empty string"):
        sweep_by_segment(ACTION, two_segment_log(), segment_key="   ")


def test_sweep_by_segment_rejects_unusable_parameters() -> None:
    with pytest.raises(ValueError, match="min_records must be >= 1"):
        sweep_by_segment(ACTION, two_segment_log(), segment_key="lang", min_records=0)
    with pytest.raises(ValueError, match="steps must be >= 2"):
        sweep_by_segment(ACTION, two_segment_log(), segment_key="lang", steps=1)
    with pytest.raises(ValueError, match="alpha must be in"):
        sweep_by_segment(ACTION, two_segment_log(), segment_key="lang", alpha=1.0)


def test_sweep_by_segment_is_deterministic() -> None:
    first = sweep_by_segment(ACTION, two_segment_log(), segment_key="lang")
    second = sweep_by_segment(ACTION, two_segment_log(), segment_key="lang")

    assert first == second
