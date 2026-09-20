"""Label-planning tests.

The projection is an estimate under an assumed scaling law, so these tests pin the properties
that make it usable rather than the exact counts it produces: it is monotone in the target
width, it is refused (with a reason) when a scope is too small to fit, it covers one plan per
scope, and it stays within a sanity bound of the naive sample-size formula it replaces.
"""

from __future__ import annotations

import pytest

from jeval.planning import (
    MIN_LABELS_FOR_PROJECTION,
    LabelPlan,
    plan_labels,
)
from jeval.schema import DecisionRecord
from jeval.synth import SynthSpec, generate


def _plan(plans: tuple[LabelPlan, ...], scope: str, key: str, value: str) -> LabelPlan:
    matches = [p for p in plans if (p.scope, p.key, p.value) == (scope, key, value)]
    assert len(matches) == 1, f"expected exactly one {scope} plan for {key}={value}, got {matches}"
    return matches[0]


def _gold_count(records: list[DecisionRecord]) -> int:
    """Labels the planner is allowed to count: gold only, and measurable on the binary scale."""
    return sum(
        1
        for record in records
        if record.is_labeled and not record.is_silver and record.calibration_point() is not None
    )


def _naive_extra(plan: LabelPlan, target: float) -> int:
    """The one-liner jeval used before this lane: n * ((span / target) ** 2 - 1)."""
    assert plan.labels_for_target is not None
    assert plan.n_now > 0
    return max(0, int(plan.n_now * ((plan.ci_width / target) ** 2 - 1)))


def _two_scale_records() -> list[DecisionRecord]:
    """One question with plenty of labels and one too small to project from."""
    return generate(
        SynthSpec(n=900, mode="inflated", inflation=1.2, question_key="department", seed=21)
    ) + generate(
        SynthSpec(
            n=120,
            mode="calibrated",
            question_key="is_urgent",
            question_type="noul",
            seed=22,
        )
    )


def test_plan_measures_every_question_and_projects_only_where_it_can() -> None:
    records = _two_scale_records()
    plans = plan_labels(records)

    assert {p.key for p in plans} == {"department", "is_urgent"}
    assert all(p.scope == "question" for p in plans)

    big = _plan(plans, "question", "department", "choice")
    assert big.n_now == _gold_count([r for r in records if r.question_key == "department"]) == 900
    assert big.ece > 0.0
    assert big.ci_width > 0.0
    assert big.labels_for_target is not None
    assert big.reason == ""
    assert "n**a" in big.assumption

    small = _plan(plans, "question", "is_urgent", "noul")
    assert small.n_now == 120
    assert small.labels_for_target is None
    assert "120" in small.reason


def test_tighter_target_never_needs_fewer_labels() -> None:
    plans = plan_labels(generate(SynthSpec(n=900, mode="overconfident", seed=23)))
    plan = plans[0]
    assert plan.labels_for_target is not None

    targets = [target for target, _ in plan.labels_for_target]
    counts = [count for _, count in plan.labels_for_target]
    assert targets == sorted(targets), targets
    assert all(count >= 0 for count in counts), counts
    assert counts == sorted(counts, reverse=True), counts
    assert counts[0] > counts[-1], counts


def test_scope_below_the_floor_has_no_projection_and_says_why() -> None:
    starved = generate(SynthSpec(n=199, mode="calibrated", question_key="tiny", seed=24))
    just_enough = generate(SynthSpec(n=200, mode="calibrated", question_key="exact", seed=25))

    assert _gold_count(starved) == 199
    assert plan_labels(starved)[0].labels_for_target is None
    reason = plan_labels(starved)[0].reason
    assert reason, "a refused projection must carry a reason"
    assert str(MIN_LABELS_FOR_PROJECTION) in reason
    assert "199" in reason

    boundary = plan_labels(just_enough)[0]
    assert boundary.n_now == MIN_LABELS_FOR_PROJECTION
    assert boundary.labels_for_target is not None
    assert boundary.reason == ""


def test_by_axis_produces_one_plan_per_segment_above_the_floor() -> None:
    records = generate(
        SynthSpec(
            n=900,
            mode="inflated",
            inflation=1.2,
            question_key="department",
            seed=26,
            languages=("ko", "en"),
        )
    )
    plans = plan_labels(records, by=("lang",))
    segments = [p for p in plans if p.scope == "segment"]

    assert {(p.key, p.value) for p in segments} == {("lang", "ko"), ("lang", "en")}
    assert all(p.n_now == 450 for p in segments), [p.n_now for p in segments]
    assert all(p.labels_for_target is not None for p in segments)
    for segment in segments:
        assert segment.ece > 0.0
        assert segment.ci_width > 0.0

    # One question plan still comes back alongside the segment plans.
    assert [p.key for p in plans if p.scope == "question"] == ["department"]
    # A repeated axis must not duplicate plans.
    assert len(plan_labels(records, by=("lang", "lang"))) == len(plans)


def test_segment_below_the_floor_is_reported_without_a_projection() -> None:
    records = generate(
        SynthSpec(
            n=120, mode="calibrated", question_key="department", seed=27, languages=("ko", "en")
        )
    )
    segments = [p for p in plan_labels(records, by=("lang",)) if p.scope == "segment"]

    assert {p.value for p in segments} == {"ko", "en"}
    for segment in segments:
        assert segment.n_now == 60
        assert segment.labels_for_target is None
        assert "60" in segment.reason


def test_silver_labels_do_not_shrink_the_interval_or_count_as_labels() -> None:
    records = generate(
        SynthSpec(
            n=1200,
            mode="inflated",
            inflation=1.2,
            question_key="department",
            seed=28,
            silver_fraction=0.5,
        )
    )
    plan = plan_labels(records)[0]
    expected = _gold_count(records)

    assert plan.n_now == expected
    assert expected < len(records), "the fixture must actually separate silver from gold"


def test_halving_the_interval_stays_within_a_factor_of_three_of_the_naive_estimate() -> None:
    records = generate(
        SynthSpec(n=1200, mode="inflated", inflation=1.2, question_key="department", seed=3)
    )
    plan = plan_labels(records)[0]
    assert plan.labels_for_target is not None
    assert plan.ci_width > 0.0

    target = plan.ci_width / 2.0
    halving = [count for width, count in plan.labels_for_target if abs(width - target) < 1e-12]
    assert len(halving) == 1, plan.labels_for_target
    projected = halving[0]

    naive = _naive_extra(plan, target)
    assert naive > 0
    ratio = projected / naive
    assert 1 / 3 <= ratio <= 3, (projected, naive, ratio)


def test_projection_scales_with_the_sample_size_it_was_fitted_on() -> None:
    def fit_at(n: int, seed: int) -> tuple[LabelPlan, int]:
        plan = plan_labels(generate(SynthSpec(n=n, mode="inflated", inflation=1.2, seed=seed)))[0]
        assert plan.labels_for_target is not None
        target = plan.ci_width / 2.0
        held = [count for width, count in plan.labels_for_target if abs(width - target) < 1e-12]
        assert len(held) == 1, plan.labels_for_target
        return plan, held[0]

    fits = [fit_at(n, 29) for n in (400, 800, 1600)]
    projected = [count for _, count in fits]
    # Halving the interval needs roughly three times the current sample on top of it, so the
    # ask grows with the sample it was fitted on and stays near the naive formula at every size.
    assert projected[0] < projected[1] < projected[2], projected
    for plan, count in fits:
        naive = _naive_extra(plan, plan.ci_width / 2.0)
        assert naive > 0
        assert 1 / 3 <= count / naive <= 3, (plan.n_now, count, naive)


def test_invalid_settings_are_rejected() -> None:
    records = generate(SynthSpec(n=250, mode="calibrated", seed=30))
    with pytest.raises(ValueError, match="target_ci"):
        plan_labels(records, target_ci=0.0)
    with pytest.raises(ValueError, match="alpha"):
        plan_labels(records, alpha=1.0)
