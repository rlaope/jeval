"""Discrimination: does confidence rank the model's own errors?

ECE says whether confidence is honest on average. It cannot say whether a confident answer is more
likely to be right than an unconfident one, which is the only thing a threshold can exploit. These
tests pin the rank statistics on hand-computed cases and on synthetic logs whose answer is known.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from jeval.calibration import (
    aurc,
    auroc,
    compute_discrimination,
    describe_discrimination,
    risk_coverage,
)
from jeval.schema import DecisionRecord
from jeval.synth import SynthSpec, generate


def _pairs(records: list[DecisionRecord]) -> tuple[list[float], list[bool]]:
    points = [p for p in (record.calibration_point() for record in records) if p is not None]
    return [p[0] for p in points], [p[1] for p in points]


# --- hand-computed cases ---------------------------------------------------------------------


def test_auroc_perfect_and_inverted_ranking() -> None:
    assert auroc([0.9, 0.8, 0.7, 0.6], [True, True, False, False]) == pytest.approx(1.0)
    assert auroc([0.9, 0.8, 0.7, 0.6], [False, False, True, True]) == pytest.approx(0.0)


def test_auroc_counts_pairs() -> None:
    # Right answers at 0.9 and 0.7, wrong ones at 0.8 and 0.6. Of the four (right, wrong) pairs,
    # the right answer is more confident in three: 0.9>0.8, 0.9>0.6, 0.7>0.6. AUROC = 3/4.
    assert auroc([0.9, 0.8, 0.7, 0.6], [True, False, True, False]) == pytest.approx(0.75)


def test_auroc_counts_a_tie_as_half() -> None:
    # Right at 0.8; wrong at 0.8 (a tie, worth 1/2) and 0.6 (a win, worth 1). (1/2 + 1) / 2.
    assert auroc([0.8, 0.8, 0.6], [True, False, False]) == pytest.approx(0.75)


def test_auroc_is_undefined_when_one_class_is_empty() -> None:
    assert math.isnan(auroc([0.9, 0.8, 0.7], [True, True, True]))
    assert math.isnan(auroc([0.9, 0.8, 0.7], [False, False, False]))
    assert math.isnan(auroc([], []))


def test_risk_coverage_by_hand() -> None:
    # Most confident first: right, wrong, right, wrong.
    coverage, risk = risk_coverage([0.9, 0.8, 0.7, 0.6], [True, False, True, False])
    assert list(coverage) == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert list(risk) == pytest.approx([0.0, 0.5, 1 / 3, 0.5])
    # The area is the mean risk over the four coverage levels.
    assert aurc([0.9, 0.8, 0.7, 0.6], [True, False, True, False]) == pytest.approx(
        (0.0 + 0.5 + 1 / 3 + 0.5) / 4
    )


def test_risk_coverage_does_not_split_a_tie() -> None:
    # A threshold cannot automate one 0.8 and escalate the other, so the tied pair is one step:
    # coverage 2/3 at risk 1/2, then coverage 1 at risk 2/3. Area = 2/3 * 1/2 + 1/3 * 2/3 = 5/9.
    coverage, risk = risk_coverage([0.8, 0.8, 0.6], [True, False, False])
    assert list(coverage) == pytest.approx([2 / 3, 1.0])
    assert list(risk) == pytest.approx([0.5, 2 / 3])
    assert aurc([0.8, 0.8, 0.6], [True, False, False]) == pytest.approx(5 / 9)


def test_all_correct_has_zero_risk_and_undefined_auroc() -> None:
    metrics = compute_discrimination([0.9, 0.8, 0.7, 0.6] * 10, [True] * 40, n_boot=50)
    assert math.isnan(metrics.auroc)
    assert metrics.aurc == pytest.approx(0.0)
    assert metrics.full_coverage_risk == pytest.approx(0.0)
    assert "undefined" in describe_discrimination(metrics)


def test_threshold_lookup_reads_the_curve() -> None:
    metrics = compute_discrimination([0.9, 0.8, 0.7, 0.6], [True, False, True, False], n_boot=20)
    # At 0.75 the two answers at 0.9 and 0.8 run: half of them, one wrong.
    assert metrics.at_threshold(0.75) == pytest.approx((0.5, 0.5))
    assert metrics.at_threshold(0.6) == pytest.approx((1.0, 0.5))
    coverage, risk = metrics.at_threshold(0.95)
    assert coverage == 0.0 and math.isnan(risk)


# --- synthetic logs with a known answer ----------------------------------------------------


def test_constant_confidence_ranks_nothing() -> None:
    # Every decision claims 0.99, so every (right, wrong) pair is a tie and AUROC is 0.5 exactly,
    # whatever n is: the tolerance is float round-off, not sampling error.
    confidences, correct = _pairs(generate(SynthSpec(n=4000, mode="constant_high", seed=4)))
    metrics = compute_discrimination(confidences, correct, n_boot=200, seed=5)
    assert metrics.auroc == pytest.approx(0.5, abs=1e-12)
    # A flat risk-coverage curve: the area equals the error rate at full coverage.
    assert metrics.aurc == pytest.approx(metrics.full_coverage_risk, abs=1e-12)
    reading = describe_discrimination(metrics)
    assert "same confidence" in reading
    assert "0.50" in reading


def test_shuffled_confidence_is_indistinguishable_from_chance() -> None:
    # Calibrated confidences, outcomes shuffled against them: any ranking left is noise. Under the
    # null, the standard error of AUROC is sqrt((n1 + n0 + 1) / (12 n1 n0)); at n=4000 with ~79%
    # right that is ~0.011, so 0.045 is four standard errors.
    confidences, correct = _pairs(generate(SynthSpec(n=4000, mode="calibrated", seed=3)))
    shuffled = list(np.random.default_rng(1).permutation(np.asarray(correct)))
    metrics = compute_discrimination(confidences, shuffled, n_boot=300, seed=5)
    assert abs(metrics.auroc - 0.5) < 0.045, metrics.auroc
    assert metrics.auroc_ci_low <= 0.5 <= metrics.auroc_ci_high
    assert "not distinguishable" in describe_discrimination(metrics)


def test_calibrated_confidence_separates_right_from_wrong() -> None:
    # The generator draws the outcome with the stated probability, so a confident answer really is
    # likelier to be right. Its population AUROC is 0.717 (2M-draw Monte Carlo of the same
    # distribution); at n=4000 the standard error is ~0.01, so 0.045 is again four of them.
    confidences, correct = _pairs(generate(SynthSpec(n=4000, mode="calibrated", seed=3)))
    metrics = compute_discrimination(confidences, correct, n_boot=300, seed=5)
    assert metrics.auroc == pytest.approx(0.717, abs=0.045)
    assert metrics.auroc_ci_low > 0.6, metrics.auroc_ci_low
    assert metrics.auroc_ci_low <= metrics.auroc <= metrics.auroc_ci_high
    # Ranking pays: the area under the curve sits clearly below the error rate at full coverage.
    assert metrics.aurc < metrics.full_coverage_risk - 0.05
    assert metrics.aurc_ci_low <= metrics.aurc <= metrics.aurc_ci_high


def test_yes_no_answers_are_ranked_on_the_probability_of_being_right() -> None:
    confidences, correct = _pairs(
        generate(SynthSpec(n=3000, mode="calibrated", question_type="noul", seed=12))
    )
    assert min(confidences) >= 0.5
    metrics = compute_discrimination(confidences, correct, n_boot=200, seed=5)
    assert metrics.auroc_ci_low > 0.6, metrics.auroc


# --- verification findings ---------------------------------------------------------------------
# Under a null (confidence unrelated to correctness), a sample with two or three wrong answers put
# 0.5 outside its 95% AUROC interval up to 42% of the time, and the reading called it separation.


def test_a_null_with_few_wrong_answers_claims_no_interval_and_no_separation() -> None:
    import numpy as np

    from jeval.calibration import compute_discrimination, describe_discrimination

    rng = np.random.default_rng(1)
    confidences = rng.uniform(0.5, 1.0, 400)
    correct = np.ones(400, dtype=bool)
    correct[rng.choice(400, 3, replace=False)] = False
    metrics = compute_discrimination(confidences, correct, n_boot=200)
    assert metrics.auroc_ci_low != metrics.auroc_ci_low  # no interval
    reading = describe_discrimination(metrics)
    assert "3 wrong" in reading
    for claim in ("separation", "outranks", "buys"):
        assert claim not in reading


def test_the_interval_is_widened_to_contain_its_own_estimate() -> None:
    import numpy as np

    from jeval.calibration import _percentile_ci

    draws = np.linspace(0.70, 0.80, 200)
    assert _percentile_ci(draws, 0.95, 0.05)[1] == 0.95
    assert _percentile_ci(draws, 0.60, 0.05)[0] == 0.60


def test_discrimination_reads_gold_labels_only() -> None:
    from jeval.evaluate import evaluate
    from jeval.synth import SynthSpec, generate

    records = generate(SynthSpec(n=600, mode="calibrated", seed=4, silver_fraction=0.5))
    dataset = evaluate(records, n_boot=20)
    gold = sum(1 for record in records if record.is_gold)
    assert dataset.questions[0].discrimination is not None
    assert dataset.questions[0].discrimination.n == gold


def test_a_yes_no_question_is_ranked_on_the_probability_of_being_right() -> None:
    from jeval.evaluate import evaluate
    from jeval.synth import SynthSpec, generate

    records = generate(
        SynthSpec(n=600, mode="calibrated", seed=5, question_type="noul", question_key="q")
    )
    discrimination = evaluate(records, n_boot=20).questions[0].discrimination
    assert discrimination is not None
    assert min(discrimination.levels) >= 0.5, "a yes/no answer is right at least half the time"
