"""Classwise calibration for ``choice`` questions.

Top-1 calibration averages over whichever class was predicted. A class that is overconfident only
when it is predicted -- exactly the class a cost action fires on -- can hide behind classes that
err the other way. These tests inject that failure with ``jeval.synth`` and require the
per-class measurement to find it.
"""

from __future__ import annotations

import pytest

from jeval.calibration import (
    MAP_SUM_TOLERANCE,
    MIN_CLASS_SIZE,
    classwise_calibration,
    compute_calibration,
    describe_classwise,
)
from jeval.schema import DecisionRecord
from jeval.synth import SynthSpec, generate


def _classwise(records: list[DecisionRecord], n_boot: int = 300):
    return classwise_calibration(
        [record.probabilities for record in records],
        [record.label or "" for record in records],
        n_boot=n_boot,
        seed=5,
    )


def _top1(records: list[DecisionRecord]):
    points = [p for p in (record.calibration_point() for record in records) if p is not None]
    return compute_calibration([p[0] for p in points], [p[1] for p in points], n_boot=300, seed=5)


def test_one_class_inflated_is_found_and_hidden_from_top1() -> None:
    records = generate(
        SynthSpec(n=6000, mode="class_inflated", inflation=1.3, inflated_class="billing", seed=21)
    )
    top1 = _top1(records)
    classwise = _classwise(records)
    by_name = {item.name: item for item in classwise.classes}
    assert set(by_name) == {"billing", "technical", "other"}
    billing = by_name["billing"].metrics
    assert billing is not None

    # Top-1 reliability looks honest: the other classes err the other way and the average cancels.
    assert top1.ece_ci_low < 0.03, (top1.ece, top1.ece_ci_low)
    # The class the model inflates is the worst class, by a margin its interval can defend.
    assert classwise.worst is not None and classwise.worst.name == "billing"
    assert billing.ece > 2 * top1.ece, (billing.ece, top1.ece)
    for name in ("technical", "other"):
        other = by_name[name].metrics
        assert other is not None
        assert billing.ece_ci_low > other.ece_ci_high, (name, billing, other)
    reading = describe_classwise(classwise)
    assert "billing" in reading


def test_a_calibrated_map_is_calibrated_for_every_class() -> None:
    # inflation 1.0: the whole probability map is the distribution the label was drawn from.
    records = generate(SynthSpec(n=6000, mode="class_inflated", inflation=1.0, seed=22))
    classwise = _classwise(records)
    assert classwise.n_used == 6000
    for item in classwise.classes:
        assert item.metrics is not None
        assert item.metrics.ece < 0.03, (item.name, item.metrics.ece)
        assert item.metrics.ece_ci_low < 0.03
    assert classwise.macro_ece < 0.03


def test_truncated_and_missing_maps_are_refused_and_counted() -> None:
    records = generate(SynthSpec(n=600, mode="class_inflated", inflation=1.0, seed=23))
    maps = [dict(record.probabilities or {}) for record in records]
    # A top-1 map: the rest of the mass is missing, so the map cannot say P(class) for the others.
    for index in range(0, 100):
        prediction = records[index].prediction
        maps[index] = {prediction: maps[index][prediction]}
    # Guard the case: a top-1 map sums to its top probability, which the generator keeps < 0.98.
    assert all(abs(sum(m.values()) - 1.0) > MAP_SUM_TOLERANCE for m in maps[:100])
    labels = [record.label or "" for record in records]
    missing: list[dict[str, float] | None] = [*maps[:-7], *([None] * 7)]
    result = classwise_calibration(missing, labels, n_boot=50, seed=1)
    assert result.n_refused_map == 107
    assert result.n_used == 600 - 107


def test_a_class_below_the_minimum_is_refused_and_says_so() -> None:
    records = generate(SynthSpec(n=900, mode="class_inflated", inflation=1.0, seed=24))
    maps = [record.probabilities for record in records]
    labels = [record.label or "" for record in records]
    # Relabel all but a handful of "other" rows as "billing": "other" keeps too few labels.
    kept = 0
    for index, label in enumerate(labels):
        if label == "other":
            if kept < MIN_CLASS_SIZE - 1:
                kept += 1
            else:
                labels[index] = "billing"
    result = classwise_calibration(maps, labels, n_boot=50, seed=1)
    other = next(item for item in result.classes if item.name == "other")
    assert other.n == MIN_CLASS_SIZE - 1
    assert other.metrics is None
    assert str(MIN_CLASS_SIZE) in other.refused_reason
    assert all(item.metrics is not None for item in result.classes if item.name != "other")


def test_the_synth_option_leaves_existing_generators_alone() -> None:
    # The new mode is opt-in: the default spec must not grow the option's behaviour.
    assert SynthSpec().inflated_class == ""
    with pytest.raises(ValueError):
        generate(SynthSpec(n=10, mode="class_inflated", inflated_class="nope"))
