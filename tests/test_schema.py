"""Tests for the decision-record schema and its per-type normalization rules."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jeval.schema import DecisionRecord, normalize_record


def test_choice_confidence_comes_from_the_top_probability() -> None:
    record = normalize_record(
        {
            "model": "jev-1.13.0",
            "question_key": "department",
            "question_type": "choice",
            "prediction": "billing",
            "probabilities": {"billing": 0.91, "technical": 0.06, "other": 0.03},
        }
    )
    assert record.confidence == pytest.approx(0.91)
    assert record.prediction == "billing"


def test_choice_prediction_defaults_to_the_argmax() -> None:
    record = normalize_record(
        {
            "model": "m",
            "question_key": "q",
            "question_type": "choice",
            "probabilities": {"a": 0.2, "b": 0.7},
        }
    )
    assert record.prediction == "b"
    assert record.confidence == pytest.approx(0.7)


def test_noul_confidence_is_normalized_to_distance_from_a_coin_flip() -> None:
    record = normalize_record(
        {
            "model": "m",
            "question_key": "is_urgent",
            "question_type": "noul",
            "probability_positive": 0.9,
            "label": True,
        }
    )
    assert record.confidence == pytest.approx(0.8)
    assert record.prediction == "yes"
    assert record.label == "yes"
    # Stored as the distance from a coin flip; measured as how likely the answer is to be right.
    assert record.stated_probability == pytest.approx(0.9)
    assert record.calibration_point() == (pytest.approx(0.9), True)


def test_noul_can_be_rebuilt_from_a_reported_confidence() -> None:
    record = normalize_record(
        {
            "model": "m",
            "question_key": "is_urgent",
            "question_type": "noul",
            "prediction": "no",
            "confidence": 0.4,
        }
    )
    assert record.confidence == pytest.approx(0.4)
    assert record.positive_probability == pytest.approx(0.3)


def test_noul_without_enough_information_is_rejected() -> None:
    with pytest.raises(ValueError, match="noul records need"):
        normalize_record({"model": "m", "question_key": "q", "question_type": "noul"})


def test_unsupported_question_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported question_type"):
        normalize_record({"model": "m", "question_key": "q", "question_type": "regression"})


def test_out_of_range_confidence_is_rejected() -> None:
    with pytest.raises(ValueError, match="out of range"):
        normalize_record(
            {
                "model": "m",
                "question_key": "q",
                "question_type": "choice",
                "prediction": "a",
                "confidence": 1.4,
            }
        )


def test_unlabeled_records_keep_no_label_source() -> None:
    record = normalize_record(
        {
            "model": "m",
            "question_key": "q",
            "question_type": "choice",
            "prediction": "a",
            "confidence": 0.7,
            "label": None,
            "label_source": "human_review",
        }
    )
    assert record.label is None
    assert record.label_source is None
    assert record.calibration_point() is None


def test_score_records_never_enter_binary_accuracy() -> None:
    record = normalize_record(
        {
            "model": "m",
            "question_key": "satisfaction",
            "question_type": "score",
            "prediction": "0.8",
            "confidence": 0.8,
            "label": "0.7",
            "label_source": "human_review",
        }
    )
    assert record.calibration_point() is None


def test_gold_and_silver_are_distinguishable() -> None:
    gold = DecisionRecord(
        model="m",
        question_key="q",
        question_type="choice",
        prediction="a",
        confidence=0.8,
        label="a",
        label_source="human_override",
    )
    silver = DecisionRecord(
        model="m",
        question_key="q",
        question_type="choice",
        prediction="a",
        confidence=0.8,
        label="a",
        label_source="silver",
    )
    assert gold.is_gold and not gold.is_silver
    assert silver.is_silver and not silver.is_gold
    assert gold.is_correct is True and silver.is_correct is True


def test_unknown_fields_are_rejected_instead_of_silently_dropped() -> None:
    with pytest.raises(ValidationError):
        DecisionRecord.model_validate(
            {
                "model": "m",
                "question_key": "q",
                "question_type": "choice",
                "prediction": "a",
                "confidence": 0.5,
                "confidence_score": 0.5,
            }
        )


def test_segment_values_are_stringified_for_stable_grouping() -> None:
    record = normalize_record(
        {
            "model": "m",
            "question_key": "q",
            "question_type": "choice",
            "prediction": "a",
            "confidence": 0.6,
            "segment": {"lang": "ko", "attempts": 3},
        }
    )
    assert record.segment == {"lang": "ko", "attempts": "3"}


def test_record_ids_are_unique() -> None:
    ids = {
        DecisionRecord(
            model="m", question_key="q", question_type="choice", prediction="a", confidence=0.5
        ).id
        for _ in range(200)
    }
    assert len(ids) == 200
