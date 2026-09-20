"""Tests for the pure statistics: binning, metrics, and intervals."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jeval.calibration import (
    bins_from_edges,
    brier_score,
    choose_n_bins,
    compute_calibration,
    diagnose,
    equal_width_edges,
    expected_calibration_error,
    maximum_calibration_error,
    quantile_edges,
    wilson_interval,
)


def test_wilson_matches_published_reference_values() -> None:
    low, high = wilson_interval(0, 10)
    assert low == pytest.approx(0.0, abs=1e-9)
    assert high == pytest.approx(0.2775, abs=1e-3)

    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-3)
    assert high == pytest.approx(0.7634, abs=1e-3)

    low, high = wilson_interval(10, 10)
    assert low == pytest.approx(0.7225, abs=1e-3)
    assert high == pytest.approx(1.0, abs=1e-9)


def test_wilson_is_not_the_normal_approximation_on_small_bins() -> None:
    """The normal approximation gives a zero-width interval at 0/10; Wilson does not."""
    _, high = wilson_interval(0, 10)
    assert high > 0.2
    low, _ = wilson_interval(10, 10)
    assert low < 0.8


def test_wilson_guards_its_inputs() -> None:
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_interval(1, 4, alpha=1.5)
    low, high = wilson_interval(0, 0)
    assert math.isnan(low) and math.isnan(high)


def test_ece_hand_computed_two_bins() -> None:
    confidences = np.array([0.10, 0.20])
    correct = np.array([True, False])
    edges = [0.10, 0.15, 0.20]
    bins = bins_from_edges(confidences, correct, edges)
    assert len(bins) == 2
    assert expected_calibration_error(bins, 2) == pytest.approx(0.55)


def test_ece_of_a_perfectly_accurate_high_confidence_set_is_zero() -> None:
    confidences = np.array([0.9, 0.9, 0.9])
    correct = np.array([True, True, True])
    bins = bins_from_edges(confidences, correct, [0.9 - 1e-9, 1.0])
    ece = expected_calibration_error(bins, 3)
    assert 0.0 <= ece < 0.11


def test_mce_takes_the_worst_bin_and_brier_is_a_mean_square() -> None:
    confidences = np.array([0.2, 0.2, 0.8, 0.8])
    correct = np.array([True, False, True, True])
    bins = bins_from_edges(confidences, correct, [0.0, 0.5, 1.0])
    assert len(bins) == 2
    assert maximum_calibration_error(bins) == pytest.approx(0.3)
    assert brier_score(confidences, correct) == pytest.approx(
        (0.8**2 + 0.2**2 + 0.2**2 + 0.2**2) / 4
    )


def test_quantile_bins_are_balanced_even_when_confidence_is_skewed() -> None:
    rng = np.random.default_rng(0)
    confidences = np.clip(0.85 + 0.14 * rng.beta(1.2, 3.0, size=1000), 0.0, 1.0)
    correct = rng.random(1000) < confidences
    metrics = compute_calibration(confidences, correct, n_bins=10, n_boot=50, seed=0)
    counts = [b.n for b in metrics.bins]
    assert len(counts) >= 5
    assert max(counts) < 3 * min(counts)


def test_equal_width_binning_starves_bins_where_quantiles_stay_balanced() -> None:
    """Confidence piles up at the top of a wide range: the case that motivates quantiles."""
    rng = np.random.default_rng(1)
    low = 0.05 + 0.10 * rng.random(200)
    high = 0.90 + 0.09 * rng.random(300)
    confidences = np.concatenate([low, high])
    correct = rng.random(500) < confidences
    quantile_bins = compute_calibration(confidences, correct, n_bins=10, n_boot=20, seed=0)
    width_bins = compute_calibration(
        confidences, correct, n_bins=10, equal_width=True, n_boot=20, seed=0
    )
    assert len(width_bins.bins) < len(quantile_bins.bins)
    quantile_counts = [b.n for b in quantile_bins.bins]
    assert max(quantile_counts) < 3 * min(quantile_counts)
    assert width_bins.ece >= 0.0


def test_bin_count_shrinks_automatically_on_small_samples() -> None:
    assert choose_n_bins(0, 10) == 0
    assert choose_n_bins(10, 10) == 1
    assert choose_n_bins(95, 10) == 4
    assert choose_n_bins(1240, 10) == 10
    assert choose_n_bins(50, 10) == 2


def test_degenerate_confidence_does_not_crash_and_keeps_one_bin() -> None:
    confidences = np.full(200, 0.99)
    correct = np.zeros(200, dtype=bool)
    metrics = compute_calibration(confidences, correct, n_bins=10, n_boot=20, seed=0)
    assert len(metrics.bins) == 1
    assert metrics.bins[0].n == 200
    assert metrics.ece == pytest.approx(0.99)


def test_edges_helpers_cover_the_observed_range() -> None:
    values = np.array([0.3, 0.5, 0.7, 0.9])
    assert quantile_edges(values, 2)[0] == pytest.approx(0.3)
    assert quantile_edges(values, 2)[-1] == pytest.approx(0.9)
    width = equal_width_edges(values, 4)
    assert width[0] == pytest.approx(0.3)
    assert width[-1] == pytest.approx(0.9)
    assert equal_width_edges(np.array([0.5, 0.5]), 4) == [0.5, 0.5]


def test_bootstrap_interval_widens_as_the_sample_shrinks() -> None:
    rng = np.random.default_rng(2)
    big = np.clip(rng.normal(0.85, 0.06, size=4000), 0.01, 0.99)
    small = big[:200]
    big_metrics = compute_calibration(big, rng.random(4000) < big, n_boot=200, seed=0)
    small_metrics = compute_calibration(small, rng.random(200) < small, n_boot=200, seed=0)
    assert small_metrics.ece_ci_span > big_metrics.ece_ci_span


def test_empty_input_returns_nan_metrics_not_an_exception() -> None:
    metrics = compute_calibration([], [])
    assert metrics.n == 0
    assert math.isnan(metrics.ece)
    assert metrics.bins == ()
    assert "No labeled decisions" in diagnose(metrics)


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_calibration([0.5, 0.6], [True])


def test_a_tiny_bin_does_not_hijack_the_one_line_diagnosis() -> None:
    """A 4-record bin with a huge gap must not outrank a large bin with a real gap."""
    confidences = np.concatenate([np.full(1000, 0.80), np.full(4, 0.20)])
    correct = np.concatenate([np.zeros(1000, dtype=bool), np.ones(4, dtype=bool)])
    metrics = compute_calibration(confidences, correct, n_bins=2, n_boot=20, seed=0)
    worst = metrics.worst_bin
    assert worst is not None
    assert worst.n >= 100, worst.n
    assert worst.gap < 0


def test_diagnose_names_the_direction_and_the_bin() -> None:
    confidences = np.concatenate([np.full(100, 0.85), np.full(100, 0.55)])
    correct = np.concatenate([np.zeros(100, dtype=bool), np.ones(100, dtype=bool)])
    metrics = compute_calibration(confidences, correct, n_bins=2, n_boot=20, seed=0)
    message = diagnose(metrics)
    assert "Overconfidence" in message
    assert "ECE" in message
