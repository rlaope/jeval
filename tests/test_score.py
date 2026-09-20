"""Tests for score-type measurement: error scale, rank agreement, and the level view.

Fixtures are built by hand rather than through ``jeval.synth``: the synthetic generator labels
``score`` questions with class-name strings, which is a real case this module must count as
unparseable but cannot be used to exercise the numeric metrics the module is defined on.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import pytest

from jeval.schema import DecisionRecord, QuestionType
from jeval.score import (
    DEFAULT_LEVEL_BINS,
    MIN_USABLE_PAIRS,
    mean_absolute_error,
    measure_score,
    parse_score_value,
    root_mean_squared_error,
    score_levels,
    spearman_rho,
)


def record(
    prediction: str,
    label: str | None,
    *,
    question_type: QuestionType = "score",
    question_key: str = "satisfaction",
) -> DecisionRecord:
    """One decision record with the numeric strings a real score log carries."""
    return DecisionRecord(
        model="jev-1.13.0",
        question_key=question_key,
        question_type=question_type,
        prediction=prediction,
        confidence=0.8,
        label=label,
        label_source="human_override" if label is not None else None,
    )


def records_from(
    pairs: Iterable[tuple[float, float]],
    *,
    question_type: QuestionType = "score",
    question_key: str = "satisfaction",
) -> list[DecisionRecord]:
    """Records for ``(prediction, label)`` pairs, stringified the way the schema stores them."""
    return [
        record(str(pred), str(actual), question_type=question_type, question_key=question_key)
        for pred, actual in pairs
    ]


def test_perfect_predictions_have_zero_error_and_perfect_rank_agreement() -> None:
    records = records_from([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0), (5.0, 5.0)])

    metrics = measure_score(records)

    assert metrics.n == 5
    assert metrics.mae == 0.0
    assert metrics.rmse == 0.0
    assert metrics.bias == 0.0
    assert metrics.spearman_rho == pytest.approx(1.0, abs=1e-12)
    assert metrics.is_measurable


OFFSET = 2.5


def test_a_constant_offset_is_the_mae_and_leaves_rank_agreement_perfect() -> None:
    """A shifted model is measured perfectly by MAE and perfectly ranked at the same time."""
    actual = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    predicted = [value + OFFSET for value in actual]
    records = records_from(zip(predicted, actual, strict=True))

    metrics = measure_score(records)

    assert metrics.n == len(actual)
    assert metrics.mae == pytest.approx(OFFSET, abs=1e-12)
    assert metrics.rmse == pytest.approx(OFFSET, abs=1e-12)
    assert metrics.bias == pytest.approx(OFFSET, abs=1e-12)
    assert metrics.spearman_rho == pytest.approx(1.0, abs=1e-12)


def test_a_monotone_rescaling_survives_in_rho_but_not_in_mae() -> None:
    """The distinction this lane exists for: binary accuracy would call this 0% correct."""
    actual = [1.0, 2.0, 4.0, 8.0, 16.0]
    predicted = [2.0 * value - 5.0 for value in actual]
    records = records_from(zip(predicted, actual, strict=True))

    metrics = measure_score(records)

    assert sum(1 for item in records if item.is_correct) == 0
    assert metrics.mae == pytest.approx(4.4, abs=1e-12)  # mean |a - 5| over the labels
    assert metrics.bias == pytest.approx(1.2, abs=1e-12)  # mean 2a - 5 - a = mean a - 5
    assert metrics.spearman_rho == pytest.approx(1.0, abs=1e-12)


def test_a_constant_prediction_gives_nan_rho_without_raising() -> None:
    records = records_from([(5.0, 1.0), (5.0, 4.0), (5.0, 9.0)])

    metrics = measure_score(records)

    assert metrics.n == 3
    assert metrics.mae == pytest.approx(3.0)  # the error is still measurable
    assert math.isnan(metrics.spearman_rho)  # the ordering is not
    assert not math.isnan(metrics.rmse)


def test_a_constant_label_also_leaves_rho_undefined() -> None:
    metrics = measure_score(records_from([(1.0, 4.0), (2.0, 4.0), (3.0, 4.0)]))

    assert metrics.n == 3
    assert math.isnan(metrics.spearman_rho)
    assert metrics.mae == pytest.approx(2.0)


def test_fewer_than_three_usable_pairs_returns_nan_with_n_recorded() -> None:
    metrics = measure_score(records_from([(3.0, 4.0)]))

    assert MIN_USABLE_PAIRS == 3  # two pairs always give rho = +/-1, which is arithmetic
    assert metrics.n == 1
    assert metrics.levels == ()
    assert metrics.n_records == 1
    assert not metrics.is_measurable
    for value in (metrics.mae, metrics.rmse, metrics.bias, metrics.spearman_rho):
        assert math.isnan(value)


def test_a_log_with_no_usable_score_pairs_returns_nan_and_keeps_the_counts() -> None:
    records = [record("7", None), record("9", None), record("high", "5")]

    metrics = measure_score(records)

    assert metrics.n == 0
    assert metrics.n_unlabeled == 2
    assert metrics.n_unparseable == 1
    assert metrics.levels == ()
    assert math.isnan(metrics.mae)
    assert math.isnan(metrics.spearman_rho)


def test_level_bins_cover_the_predicted_range_and_their_counts_sum_to_n() -> None:
    pairs = [(value, value + 1.0) for value in range(100)]
    metrics = measure_score(records_from(pairs))

    assert len(metrics.levels) == DEFAULT_LEVEL_BINS
    assert [level.index for level in metrics.levels] == list(range(DEFAULT_LEVEL_BINS))
    assert metrics.levels[0].lo == 0.0
    assert metrics.levels[-1].hi == 99.0
    assert sum(level.n for level in metrics.levels) == metrics.n == 100
    for earlier, later in zip(metrics.levels[:-1], metrics.levels[1:], strict=True):
        assert earlier.hi == later.lo  # contiguous bands, no gaps and no overlap
    assert all(level.mean_error == pytest.approx(-1.0, abs=1e-9) for level in metrics.levels)


def test_the_level_view_locates_the_error_instead_of_averaging_it_away() -> None:
    """Errors confined to the top half must show up there and nowhere else."""
    pairs = [
        (float(value), float(value) if value < 50 else float(value) - 10.0) for value in range(100)
    ]

    metrics = measure_score(records_from(pairs))

    # The model runs high only in the top half: the level view says where, the bias says how big.
    assert metrics.bias == pytest.approx(5.0, abs=1e-9)
    assert metrics.levels[0].mean_error == pytest.approx(0.0, abs=1e-9)
    assert metrics.levels[1].mean_error == pytest.approx(0.0, abs=1e-9)
    assert metrics.levels[2].mean_error == pytest.approx(5.0, abs=1e-9)  # the band straddling 50
    assert metrics.levels[-2].mean_error == pytest.approx(10.0, abs=1e-9)
    assert metrics.levels[-1].mean_error == pytest.approx(10.0, abs=1e-9)
    assert sum(level.n for level in metrics.levels) == metrics.n


def test_level_bands_the_model_never_predicts_into_stay_visible() -> None:
    pairs = [(0.0, 1.0), (0.0, 2.0), (100.0, 99.0), (100.0, 98.0)]

    metrics = measure_score(records_from(pairs), n_bins=4)

    assert [level.n for level in metrics.levels] == [2, 0, 0, 2]
    assert sum(level.n for level in metrics.levels) == metrics.n == 4
    assert metrics.levels[0].mean_predicted == 0.0
    assert metrics.levels[0].mean_actual == pytest.approx(1.5)
    assert metrics.levels[-1].mean_predicted == 100.0
    assert metrics.levels[-1].mean_actual == pytest.approx(98.5)
    assert math.isnan(metrics.levels[1].mean_predicted)
    assert math.isnan(metrics.levels[2].mean_actual)


def test_a_single_distinct_prediction_collapses_to_one_band() -> None:
    metrics = measure_score(records_from([(7.0, 6.0), (7.0, 8.0), (7.0, 7.0)]), n_bins=5)

    assert len(metrics.levels) == 1
    level = metrics.levels[0]
    assert level.lo == level.hi == 7.0
    assert level.n == 3
    assert level.mean_predicted == 7.0
    assert level.mean_actual == pytest.approx(7.0)
    assert metrics.mae == pytest.approx(2.0 / 3.0)


def test_unparseable_and_unlabeled_records_are_counted_not_dropped() -> None:
    records = [
        record("7", "6"),
        record("8.5", "9"),
        record("high", "5"),
        record("4", "about five"),
        record("3", ""),
        record("2", None),
        record("nan", "1"),
        record("1", "inf"),
        record("5", "5", question_type="choice", question_key="intent"),
        record("yes", None, question_type="noul", question_key="is_urgent"),
    ]

    metrics = measure_score(records)

    assert metrics.n == 2
    assert metrics.n_unparseable == 4
    assert metrics.n_unlabeled == 2
    assert metrics.n_other_type == 2
    assert metrics.n_records == len(records)
    assert metrics.n_records == (
        metrics.n + metrics.n_unparseable + metrics.n_unlabeled + metrics.n_other_type
    )


def test_parseable_choice_records_are_still_not_score_measurements() -> None:
    records = [record("9", "9", question_type="choice"), record("4", "4")]

    metrics = measure_score(records)

    assert metrics.n == 1  # only the score record, even though both parse as numbers
    assert metrics.n_other_type == 1
    assert math.isnan(metrics.mae)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("7", 7.0),
        (" 7.5 ", 7.5),
        ("-2", -2.0),
        ("1e2", 100.0),
        (3, 3.0),
        ("", None),
        ("   ", None),
        ("seven", None),
        ("7 of 10", None),
        ("nan", None),
        ("inf", None),
        ("-inf", None),
        (None, None),
        (True, None),
    ],
)
def test_parse_score_value_accepts_only_finite_numbers(
    value: object, expected: float | None
) -> None:
    parsed = parse_score_value(value)

    assert (parsed is None) == (expected is None)
    if expected is not None:
        assert parsed == pytest.approx(expected)


def test_spearman_uses_average_ranks_for_ties() -> None:
    """Ranks 1.5/1.5/3.5/3.5 give 4/sqrt(20); breaking ties by position would give 1.0."""
    rho = spearman_rho([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 2.0, 2.0])

    assert rho == pytest.approx(4.0 / math.sqrt(20.0), abs=1e-12)


def test_spearman_detects_a_reversed_ordering() -> None:
    assert spearman_rho([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0, abs=1e-12)


def test_spearman_returns_nan_instead_of_raising_on_empty_or_mismatched_input() -> None:
    assert math.isnan(spearman_rho([], []))
    assert math.isnan(spearman_rho([1.0], [2.0]))
    with pytest.raises(ValueError):
        spearman_rho([1.0, 2.0], [1.0])


def test_error_helpers_agree_with_hand_computed_values() -> None:
    predicted = [1.0, 2.0, 3.0]
    actual = [2.0, 2.0, 5.0]

    assert mean_absolute_error(predicted, actual) == pytest.approx(1.0)
    assert root_mean_squared_error(predicted, actual) == pytest.approx(math.sqrt(5.0 / 3.0))
    assert math.isnan(mean_absolute_error([], []))
    assert math.isnan(root_mean_squared_error([], []))
    with pytest.raises(ValueError):
        mean_absolute_error([1.0], [1.0, 2.0])


def test_score_levels_rejects_a_useless_bin_count() -> None:
    with pytest.raises(ValueError):
        score_levels([1.0, 2.0], [1.0, 2.0], n_bins=0)
    with pytest.raises(ValueError):
        measure_score(records_from([(1.0, 1.0), (2.0, 2.0)]), n_bins=0)
    assert score_levels([], []) == ()
