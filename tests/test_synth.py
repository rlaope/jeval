"""Synthetic-data restoration tests.

The generator injects a known miscalibration; these tests assert the measurement code recovers
it. If a change to the statistics breaks these, the change is wrong.
"""

from __future__ import annotations

import pytest

from jeval.calibration import compute_calibration, diagnose
from jeval.schema import DecisionRecord
from jeval.synth import SynthSpec, accuracy_of, generate


def _measure(records: list[DecisionRecord], n_boot: int = 300):
    pairs = [p for p in (record.calibration_point() for record in records) if p is not None]
    return compute_calibration(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        n_boot=n_boot,
        seed=5,
    )


def test_calibrated_ece_near_zero() -> None:
    records = generate(SynthSpec(n=4000, mode="calibrated", seed=3))
    metrics = _measure(records)
    assert metrics.n == 4000
    assert metrics.ece < 0.03, metrics.ece
    assert metrics.brier < 0.25
    assert metrics.ece_ci_low < 0.03


def test_constant_high_confidence_overconfidence() -> None:
    records = generate(
        SynthSpec(
            n=4000, mode="constant_high", accuracy_target=0.95, constant_confidence=0.99, seed=4
        )
    )
    metrics = _measure(records)
    worst = metrics.worst_bin
    assert worst is not None
    assert worst.gap < -0.02, worst.gap
    assert metrics.ece > 0.02, metrics.ece
    assert "Overconfidence" in diagnose(metrics)


def test_inflated_confidence_recovers() -> None:
    small = _measure(generate(SynthSpec(n=4000, mode="inflated", inflation=1.15, seed=6)))
    large = _measure(generate(SynthSpec(n=4000, mode="inflated", inflation=1.45, seed=6)))
    assert small.ece > 0.01, small.ece
    assert large.ece > small.ece, (small.ece, large.ece)
    assert small.worst_bin is not None and small.worst_bin.gap < 0


def test_underconfidence_detected_in_the_other_direction() -> None:
    records = generate(SynthSpec(n=4000, mode="underconfident", exponent=0.55, seed=8))
    metrics = _measure(records)
    worst = metrics.worst_bin
    assert worst is not None
    assert worst.gap > 0.02, worst.gap
    assert "Underconfidence" in diagnose(metrics)


def test_calibrated_beats_inflated_under_equal_sample_size() -> None:
    calibrated = _measure(generate(SynthSpec(n=2000, mode="calibrated", seed=9)))
    inflated = _measure(generate(SynthSpec(n=2000, mode="inflated", inflation=1.2, seed=9)))
    assert calibrated.ece < inflated.ece


def test_generator_reports_its_own_accuracy() -> None:
    records = generate(SynthSpec(n=2000, mode="calibrated", seed=10))
    measured = accuracy_of(records)
    true_accuracy = sum(record.is_correct for record in records if record.is_labeled) / sum(
        1 for record in records if record.is_labeled
    )
    assert measured == pytest.approx(true_accuracy)


def test_noul_records_carry_normalized_confidence() -> None:
    records = generate(
        SynthSpec(n=500, mode="calibrated", question_type="noul", question_key="is_urgent", seed=12)
    )
    for record in records:
        assert record.question_type == "noul"
        assert record.prediction in {"yes", "no"}
        assert 0.0 <= record.confidence <= 1.0
        assert record.probabilities is not None
        positive = record.probabilities["yes"]
        assert record.confidence == pytest.approx(abs(positive - 0.5) * 2)


def test_unlabeled_and_silver_fractions_are_respected() -> None:
    records = generate(
        SynthSpec(n=1000, mode="calibrated", label_fraction=0.5, silver_fraction=0.2, seed=13)
    )
    labeled = [record for record in records if record.is_labeled]
    silver = [record for record in records if record.is_silver]
    assert len(labeled) == pytest.approx(500, abs=80)
    assert len(silver) <= len(labeled)
    assert all(record.label_source == "silver" for record in silver)


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        generate(SynthSpec(n=10, mode="nonsense", seed=1))


# --- numeric score questions -----------------------------------------------------------------
# A score question fails by being *off*, not by naming the wrong class. The generator must be able
# to inject that known failure, or the score section has nothing true to be tested against.


def test_a_score_question_produces_numeric_predictions_and_labels() -> None:
    from jeval.score import measure_score

    records = generate(
        SynthSpec(
            n=400, mode="calibrated", question_key="satisfaction", question_type="score", seed=5
        )
    )
    metrics = measure_score(records)

    assert metrics.n == 400  # every record is a usable numeric pair, none unparseable
    assert metrics.n_unparseable == 0
    assert metrics.mae < 0.08  # calibrated: small noise, no offset
    assert abs(metrics.bias) < 0.02


def test_an_injected_score_bias_is_recovered_in_the_direction_it_was_injected() -> None:
    from jeval.score import measure_score

    records = generate(
        SynthSpec(
            n=800,
            mode="inflated",
            question_key="satisfaction",
            question_type="score",
            score_bias=0.18,
            seed=6,
        )
    )
    metrics = measure_score(records)

    # The whole point of measuring score questions this way: an offset barely touches rank
    # agreement and lands squarely in MAE and bias.
    assert metrics.spearman_rho > 0.95
    assert abs(metrics.mae - 0.18) < 0.03
    assert metrics.bias > 0.15
    assert metrics.levels  # the level view is populated, not empty
    assert sum(level.n for level in metrics.levels) == metrics.n


# --- yes/no questions ----------------------------------------------------------------------------
# A noul record stores its confidence as the distance from a coin flip, |p - 0.5| * 2. Calibration
# is a claim about how often the answer is right, which is max(p, 1 - p): measuring the stored
# distance against accuracy reported a calibrated yes/no question as ECE 0.23 and "underconfident".


def test_calibrated_yes_no_reports_ece_near_zero() -> None:
    records = generate(
        SynthSpec(n=4000, mode="calibrated", question_type="noul", question_key="q", seed=3)
    )
    metrics = _measure(records)
    assert metrics.n == 4000
    assert metrics.ece < 0.03, metrics.ece
    assert metrics.ece_ci_low < 0.03


def test_inflated_yes_no_is_flagged_overconfident() -> None:
    records = generate(
        SynthSpec(
            n=4000, mode="inflated", inflation=1.25, question_type="noul", question_key="q", seed=6
        )
    )
    metrics = _measure(records)
    assert metrics.worst_bin is not None and metrics.worst_bin.gap < -0.02, metrics.worst_bin
    assert "Overconfidence" in diagnose(metrics)


def test_a_yes_no_answer_is_measured_on_the_same_scale_as_a_choice_answer() -> None:
    choice = _measure(generate(SynthSpec(n=3000, mode="calibrated", seed=11)))
    noul = _measure(
        generate(
            SynthSpec(n=3000, mode="calibrated", question_type="noul", question_key="q", seed=11)
        )
    )
    assert abs(choice.ece - noul.ece) < 0.03, (choice.ece, noul.ece)
