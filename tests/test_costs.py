"""Tests for the cost-matrix threshold engine.

Every expected number here is hand-computed from the rule documented in ``jeval/costs.py``, not
recorded from an earlier run: each cost figure can be checked with a calculator.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from jeval.costs import (
    DEFAULT_STEPS,
    MIN_GOLD_RECORDS,
    CostAction,
    bootstrap_threshold_ci,
    build_impact,
    evaluate_point,
    flat_region,
    load_cost_actions,
    sweep,
    write_thresholds_yaml,
)
from jeval.report.model import CostPoint, ThresholdResult
from jeval.schema import DecisionRecord
from jeval.synth import SynthSpec, generate

REFUND = CostAction(
    name="auto_refund",
    question="intent",
    when="refund_request",
    cost_false_accept=100.0,
    cost_escalate=10.0,
    cost_false_reject=1.0,
)

GROUPED = CostAction(
    name="auto_refund",
    question="intent",
    when="refund_request",
    cost_false_accept=100.0,
    cost_escalate=5.0,
    cost_false_reject=2.0,
)

VALID_COSTS = """\
actions:
  - name: auto_refund
    question: intent
    when: refund_request
    cost_false_accept: 50000
    cost_escalate: 2000
    cost_false_reject: 0
  - name: auto_close
    question: severity
    when: low
    cost_false_accept: 250.5
    cost_escalate: 12
    cost_false_reject: 3.25
"""


def record(
    prediction: str,
    confidence: float,
    label: str | None,
    *,
    question: str = "intent",
    label_source: str = "human_override",
    model: str = "jev-1.13.0",
) -> DecisionRecord:
    """One decision record with just enough fields for the cost engine."""
    return DecisionRecord.model_validate(
        {
            "model": model,
            "question_key": question,
            "question_type": "choice",
            "prediction": prediction,
            "confidence": confidence,
            "label": label,
            "label_source": None if label is None else label_source,
        }
    )


def repeated(
    count: int,
    prediction: str,
    confidence: float,
    label: str | None,
    **kwargs: str,
) -> list[DecisionRecord]:
    return [record(prediction, confidence, label, **kwargs) for _ in range(count)]


def same(left: float, right: float) -> bool:
    """Equality that treats two ``nan`` values as equal, which ``==`` does not."""
    return (math.isnan(left) and math.isnan(right)) or left == right


# --------------------------------------------------------------------------------------
# load_cost_actions
# --------------------------------------------------------------------------------------


def test_load_cost_actions_reads_the_documented_schema(tmp_path: Path) -> None:
    path = tmp_path / "costs.yaml"
    path.write_text(VALID_COSTS, encoding="utf-8")

    actions = load_cost_actions(path)

    assert [action.name for action in actions] == ["auto_refund", "auto_close"]
    assert actions[0] == CostAction(
        name="auto_refund",
        question="intent",
        when="refund_request",
        cost_false_accept=50000.0,
        cost_escalate=2000.0,
        cost_false_reject=0.0,
    )
    assert actions[1].cost_false_accept == pytest.approx(250.5)
    assert actions[1].cost_false_reject == pytest.approx(3.25)
    assert actions[0].label == "auto_refund (intent == refund_request)"


def test_empty_action_list_is_a_valid_cost_matrix(tmp_path: Path) -> None:
    path = tmp_path / "costs.yaml"
    path.write_text("actions: []\n", encoding="utf-8")

    assert load_cost_actions(path) == []


def test_missing_cost_file_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"nope\.yaml"):
        load_cost_actions(tmp_path / "nope.yaml")


MALFORMED_COSTS: list[tuple[str, str]] = [
    ("", "empty cost matrix"),
    ("- name: auto_refund\n", "must be a YAML mapping"),
    ("costs:\n  - name: auto_refund\n", "must be a list"),
    ("actions: 3\n", "must be a list"),
    ("actions:\n  - just_a_string\n", "must be a mapping"),
    (
        "actions:\n  - name: auto_refund\n    question: intent\n    when: refund_request\n"
        "    cost_false_accept: 1\n    cost_escalate: 2\n",
        "missing cost field",
    ),
    (
        "actions:\n  - name: auto_refund\n    question: intent\n    when: refund_request\n"
        "    cost_false_accept: 1\n    cost_escalate: 2\n    cost_false_reject: 3\n"
        "    cost_false_escalate: 4\n",
        "unknown cost field",
    ),
    (
        VALID_COSTS.replace("cost_false_accept: 50000", "cost_false_accept: lots", 1),
        "must be a number",
    ),
    (
        VALID_COSTS.replace("cost_false_accept: 50000", "cost_false_accept: true", 1),
        "must be a number",
    ),
    (VALID_COSTS.replace("cost_escalate: 2000", "cost_escalate: -1", 1), "must not be negative"),
    (
        VALID_COSTS.replace("cost_false_accept: 50000", "cost_false_accept: .nan", 1),
        "must be a finite number",
    ),
    (VALID_COSTS.replace("name: auto_refund", "name: 7", 1), "must be a non-empty string"),
    (VALID_COSTS.replace("when: refund_request", 'when: "   "', 1), "must be a non-empty string"),
]


@pytest.mark.parametrize(("text", "message"), MALFORMED_COSTS)
def test_load_cost_actions_rejects_malformed_files(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "costs.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_cost_actions(path)


def test_load_cost_actions_rejects_duplicate_action_names(tmp_path: Path) -> None:
    path = tmp_path / "costs.yaml"
    path.write_text(
        "actions:\n"
        + "".join(
            f"  - name: auto_refund\n    question: {question}\n    when: refund_request\n"
            "    cost_false_accept: 1\n    cost_escalate: 1\n    cost_false_reject: 1\n"
            for question in ("intent", "severity")
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"duplicate action name 'auto_refund'.*actions\[0\]"):
        load_cost_actions(path)


def test_a_loaded_action_sweeps_like_a_constructed_one(tmp_path: Path) -> None:
    path = tmp_path / "costs.yaml"
    path.write_text(VALID_COSTS, encoding="utf-8")
    loaded = load_cost_actions(path)[0]
    records = repeated(40, "refund_request", 0.9, "refund_request") + repeated(
        10, "refund_request", 0.9, "check_balance"
    )

    from_file = sweep(loaded, records, seed=0)
    from_code = sweep(
        CostAction(
            name="auto_refund",
            question="intent",
            when="refund_request",
            cost_false_accept=50000.0,
            cost_escalate=2000.0,
            cost_false_reject=0.0,
        ),
        records,
        seed=0,
    )

    assert from_file.threshold == from_code.threshold
    assert from_file.expected_cost_per_case == from_code.expected_cost_per_case
    assert from_file.auto_rate == from_code.auto_rate
    assert from_file.n_records == from_code.n_records == 50


# --------------------------------------------------------------------------------------
# evaluate_point
# --------------------------------------------------------------------------------------


def four_records() -> list[DecisionRecord]:
    """Two decisions run, two escalate; one of each four is wrong."""
    return [
        record("refund_request", 0.90, "refund_request"),  # runs, correct: 0
        record("refund_request", 0.95, "check_balance"),  # runs, wrong: 100
        record("refund_request", 0.50, "refund_request"),  # escalated, a when case: 10 + 1
        record("check_balance", 0.99, "check_balance"),  # escalated, not a when case: 10
    ]


def test_evaluate_point_matches_the_hand_computed_cost_rule() -> None:
    point = evaluate_point(REFUND, four_records(), 0.90)

    assert point.threshold == pytest.approx(0.90)
    assert point.expected_cost == pytest.approx((0.0 + 100.0 + 11.0 + 10.0) / 4)
    assert point.auto_rate == pytest.approx(0.5)
    assert point.accuracy_auto == pytest.approx(0.5)
    assert point.accuracy_escalated == pytest.approx(1.0)
    assert point.accept_cost == pytest.approx(100.0)
    assert point.escalate_cost == pytest.approx(21.0)
    assert (point.n_auto, point.n_escalate) == (2, 2)
    assert (point.n_wrong_auto, point.n_wrong_escalate) == (1, 0)


def test_evaluate_point_escalates_everything_above_the_models_confidence() -> None:
    point = evaluate_point(REFUND, four_records(), 0.96)

    # Every record escalates: 10 + 1, 10, 10 + 1, 10.
    assert point.expected_cost == pytest.approx(42.0 / 4)
    assert point.auto_rate == 0.0
    assert math.isnan(point.accuracy_auto)
    assert point.accuracy_escalated == pytest.approx(3 / 4)
    assert (point.n_auto, point.n_escalate) == (0, 4)
    assert (point.n_wrong_auto, point.n_wrong_escalate) == (0, 1)
    assert point.accept_cost == 0.0
    assert point.escalate_cost == pytest.approx(42.0)


def test_evaluate_point_ignores_silver_and_unlabeled_records() -> None:
    contaminated = [
        *four_records(),
        record("refund_request", 0.99, "refund_request", label_source="silver"),
        record("refund_request", 0.99, "refund_request", label_source="silver"),
        record("check_balance", 0.99, None, label_source="silver"),
        record("refund_request", 0.99, None),
    ]

    assert evaluate_point(REFUND, contaminated, 0.90) == evaluate_point(
        REFUND, four_records(), 0.90
    )


def test_evaluate_point_returns_nan_when_nothing_is_gold() -> None:
    records = [
        record("refund_request", 0.90, "refund_request", label_source="silver"),
        record("refund_request", 0.90, None),
    ]

    point = evaluate_point(REFUND, records, 0.80)

    assert point.threshold == pytest.approx(0.80)
    assert math.isnan(point.expected_cost)
    assert math.isnan(point.auto_rate)
    assert math.isnan(point.accuracy_auto)
    assert math.isnan(point.accuracy_escalated)
    assert (point.n_auto, point.n_escalate) == (0, 0)
    assert (point.accept_cost, point.escalate_cost) == (0.0, 0.0)


def test_evaluate_point_ignores_records_of_other_questions() -> None:
    records = [
        record("refund_request", 0.90, "refund_request"),
        record("low", 0.90, "low", question="severity"),
    ]

    point = evaluate_point(REFUND, records, 0.90)

    assert (point.n_auto, point.n_escalate) == (1, 0)


def test_evaluate_point_rejects_a_threshold_outside_the_unit_interval() -> None:
    records = [record("refund_request", 0.90, "refund_request")]

    with pytest.raises(ValueError, match="threshold must be in"):
        evaluate_point(REFUND, records, 1.5)
    with pytest.raises(ValueError, match="threshold must be in"):
        evaluate_point(REFUND, records, -0.01)

    # The closed interval itself is valid at both ends.
    assert evaluate_point(REFUND, records, 0.0).threshold == 0.0
    assert evaluate_point(REFUND, records, 1.0).threshold == 1.0


# --------------------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------------------


def grouped_records() -> list[DecisionRecord]:
    """Two confidence groups: a low band that is usually wrong, a high band that is not.

    With cost_false_accept 100, cost_escalate 5 and cost_false_reject 2, the three options cost
    (30 + 2) * 100 / 100 = 32.00 (run everything), (2 * 100 + 40 * 5 + 10 * 2) / 100 = 4.20 (run
    only the high band) and (100 * 5 + 68 * 2) / 100 = 6.36 (run nothing) per case.
    """
    return (
        repeated(10, "refund_request", 0.60, "refund_request")
        + repeated(30, "refund_request", 0.60, "check_balance")
        + repeated(58, "refund_request", 0.95, "refund_request")
        + repeated(2, "refund_request", 0.95, "check_balance")
    )


def test_sweep_walks_the_unit_interval_and_agrees_with_evaluate_point() -> None:
    records = grouped_records()

    result = sweep(GROUPED, records, seed=0)

    assert len(result.curve) == DEFAULT_STEPS
    assert result.curve[0].threshold == 0.0
    assert result.curve[-1].threshold == 1.0
    assert result.curve[1].threshold == pytest.approx(0.01)
    assert result.curve[75].threshold == pytest.approx(0.75)
    for point in result.curve:
        single = evaluate_point(GROUPED, records, point.threshold)
        assert single.threshold == point.threshold
        assert single.expected_cost == point.expected_cost
        assert single.auto_rate == point.auto_rate
        assert same(single.accuracy_auto, point.accuracy_auto)
        assert same(single.accuracy_escalated, point.accuracy_escalated)
        assert single.accept_cost == point.accept_cost
        assert single.escalate_cost == point.escalate_cost
        assert (single.n_auto, single.n_escalate) == (point.n_auto, point.n_escalate)
        assert (single.n_wrong_auto, single.n_wrong_escalate) == (
            point.n_wrong_auto,
            point.n_wrong_escalate,
        )


def test_sweep_honours_a_coarser_grid() -> None:
    records = grouped_records()

    result = sweep(GROUPED, records, steps=11, seed=0)

    assert [point.threshold for point in result.curve] == pytest.approx(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    assert any(abs(result.threshold - index / 10) < 1e-9 for index in range(11))

    with pytest.raises(ValueError, match="steps must be >= 2"):
        sweep(GROUPED, records, steps=1, seed=0)


def test_sweep_matches_the_hand_computed_optimum() -> None:
    result = sweep(GROUPED, grouped_records(), seed=0)

    # The cheapest band is the plateau 0.61-0.95, and ties resolve to its upper end.
    assert result.threshold == pytest.approx(0.95)
    assert result.expected_cost_per_case == pytest.approx(4.2)
    assert result.auto_rate == pytest.approx(0.60)
    assert result.accuracy_auto == pytest.approx(58 / 60)
    assert result.flat_region == pytest.approx((0.61, 0.95))
    assert result.n_records == 100
    assert result.models == ("jev-1.13.0",)
    assert result.action == "auto_refund"
    assert result.question == "intent"
    assert result.when == "refund_request"
    assert result.cost_false_accept == 100.0
    assert result.cost_escalate == 5.0
    assert result.cost_false_reject == 2.0
    assert result.silver_only is False


def test_sweep_auto_rate_never_rises_with_the_threshold() -> None:
    specs = [
        SynthSpec(n=400, mode="calibrated", question_key="intent", seed=3),
        SynthSpec(n=400, mode="overconfident", exponent=0.7, question_key="intent", seed=4),
    ]
    for spec in specs:
        result = sweep(GROUPED, generate(spec), seed=0)
        rates = [point.auto_rate for point in result.curve]
        counts = [point.n_auto for point in result.curve]
        escalated = [point.n_escalate for point in result.curve]
        assert rates == sorted(rates, reverse=True), spec.mode
        assert counts == sorted(counts, reverse=True), spec.mode
        assert escalated == sorted(escalated), spec.mode


def test_sweep_takes_the_highest_threshold_of_a_cheapest_plateau() -> None:
    # Every record sits at confidence 0.50 and is correct, so cost is zero while anything runs at
    # all and the highest zero-cost threshold on the grid is 0.50.
    records = repeated(40, "refund_request", 0.50, "refund_request")
    action = CostAction("auto", "intent", "refund_request", 100.0, 5.0, 1.0)

    result = sweep(action, records, seed=0)

    assert result.threshold == pytest.approx(0.50)
    assert result.expected_cost_per_case == 0.0
    assert result.auto_rate == 1.0
    assert result.flat_region == pytest.approx((0.0, 0.50))


def test_sweep_breaks_a_global_tie_toward_the_higher_threshold() -> None:
    # With these costs, running everything and running nothing cost exactly the same, so every
    # threshold ties and the conservative end wins.
    action = CostAction("auto", "intent", "refund_request", 2.9, 0.2, 0.1)
    records = repeated(27, "refund_request", 0.50, "refund_request") + repeated(
        3, "refund_request", 0.50, "check_balance"
    )

    result = sweep(action, records, seed=0)

    assert {round(point.expected_cost, 12) for point in result.curve} == {0.29}
    assert result.threshold == 1.0
    assert result.auto_rate == 0.0


def constant_high_records(n: int = 300, accuracy: float = 0.9) -> list[DecisionRecord]:
    """A model that only ever answers `refund_request`, always claiming 0.99, right `accuracy` of the
    time: the classic overconfidence case.

    Built by hand rather than generated. The generator draws a balanced multiclass classifier whose
    predictions follow the sampled outcome, so a fixture that leaned on its distribution was really
    testing that distribution — when the generator stopped always predicting `classes[0]`, these
    tests failed for a reason that had nothing to do with the sweep.
    """
    correct = round(n * accuracy)
    return [
        DecisionRecord(
            ts=datetime(2026, 9, 1, tzinfo=timezone.utc) - timedelta(minutes=index),
            model="jev-test",
            question_key="intent",
            question_type="choice",
            prediction="refund_request",
            confidence=0.99,
            label="refund_request" if index < correct else "check_balance",
            label_source="human_override",
        )
        for index in range(n)
    ]


def test_sweep_escalates_a_constant_high_model_when_mistakes_are_expensive() -> None:
    records = constant_high_records()
    accuracy = sum(1 for item in records if item.is_correct) / len(records)
    action = CostAction("auto", "intent", "refund_request", 1000.0, 1.0, 1.0)

    result = sweep(action, records, seed=0)

    # No threshold beats escalating every case: any confidence >= threshold is a 0.99 claim that
    # is wrong one time in ten at a thousand-fold cost.
    assert result.threshold == 1.0
    assert result.auto_rate == 0.0
    assert result.expected_cost_per_case == pytest.approx(1.0 + accuracy * 1.0)


def test_sweep_keeps_automating_when_mistakes_are_cheap() -> None:
    records = constant_high_records()
    wrong_share = sum(1 for item in records if not item.is_correct) / len(records)
    action = CostAction("auto", "intent", "refund_request", 0.5, 1.0, 1.0)

    result = sweep(action, records, seed=0)

    # A wrong automatic decision costs less than a human, so the machine keeps every case it
    # answers with 0.99 confidence; the highest such threshold is that confidence itself.
    assert result.threshold == pytest.approx(0.99)
    assert result.auto_rate == 1.0
    assert result.expected_cost_per_case == pytest.approx(wrong_share * 0.5)


def test_sweep_withholds_a_threshold_below_the_gold_floor() -> None:
    short = repeated(MIN_GOLD_RECORDS - 1, "refund_request", 0.9, "refund_request")

    held = sweep(REFUND, short, seed=0)

    assert math.isnan(held.threshold)
    assert math.isnan(held.expected_cost_per_case)
    assert math.isnan(held.auto_rate)
    assert math.isnan(held.accuracy_auto)
    assert math.isnan(held.ci_low) and math.isnan(held.ci_high)
    assert held.curve == ()
    assert held.flat_region is None
    assert held.n_records == MIN_GOLD_RECORDS - 1
    assert held.models == ("jev-1.13.0",)
    assert held.cost_false_accept == 100.0

    # The floor is inclusive: one more gold record and a number exists.
    enough = repeated(MIN_GOLD_RECORDS, "refund_request", 0.9, "refund_request")
    at_floor = sweep(REFUND, enough, seed=0)
    assert at_floor.n_records == MIN_GOLD_RECORDS
    assert math.isfinite(at_floor.threshold)
    assert len(at_floor.curve) == DEFAULT_STEPS


def test_sweep_counts_only_gold_records_toward_the_floor() -> None:
    records = repeated(MIN_GOLD_RECORDS - 1, "refund_request", 0.9, "refund_request") + repeated(
        50, "refund_request", 0.9, "refund_request", label_source="silver"
    )

    result = sweep(REFUND, records, seed=0)

    assert result.n_records == MIN_GOLD_RECORDS - 1
    assert math.isnan(result.threshold)
    assert result.silver_only is False


def test_sweep_flags_a_question_that_has_only_silver_labels() -> None:
    records = repeated(80, "refund_request", 0.9, "refund_request", label_source="silver")

    result = sweep(REFUND, records, seed=0)

    assert result.silver_only is True
    assert result.n_records == 0
    assert math.isnan(result.threshold)
    assert result.models == ()


def test_sweep_reports_every_model_behind_a_recommendation() -> None:
    records = repeated(20, "refund_request", 0.9, "refund_request", model="jev-1.13.0") + repeated(
        20, "refund_request", 0.9, "refund_request", model="jev-1.14.0"
    )

    result = sweep(REFUND, records, seed=0)

    assert result.models == ("jev-1.13.0", "jev-1.14.0")
    assert result.n_records == 40


def test_sweep_rejects_an_alpha_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match="alpha must be in"):
        sweep(REFUND, grouped_records(), alpha=1.0, seed=0)


# --------------------------------------------------------------------------------------
# bootstrap_threshold_ci
# --------------------------------------------------------------------------------------


def test_bootstrap_interval_is_deterministic_and_on_the_sweep_grid() -> None:
    records = grouped_records()

    low, high = bootstrap_threshold_ci(GROUPED, records, seed=0)

    assert (low, high) == bootstrap_threshold_ci(GROUPED, records, seed=0)
    assert low <= high
    for endpoint in (low, high):
        assert 0.0 <= endpoint <= 1.0
        assert endpoint == pytest.approx(round(endpoint, 2))
    # The recommendation for this data sits high, and resampling never drags it back to the
    # "run everything" band that costs 32 per case against 4.20.
    assert low >= 0.6

    result = sweep(GROUPED, records, seed=0)
    assert (result.ci_low, result.ci_high) == (low, high)


def test_bootstrap_interval_is_withheld_below_the_gold_floor() -> None:
    short = repeated(MIN_GOLD_RECORDS - 1, "refund_request", 0.9, "refund_request")

    low, high = bootstrap_threshold_ci(REFUND, short, seed=0)
    assert math.isnan(low) and math.isnan(high)

    no_draws_low, no_draws_high = bootstrap_threshold_ci(REFUND, short, n_boot=0, seed=0)
    assert math.isnan(no_draws_low) and math.isnan(no_draws_high)


def test_bootstrap_interval_narrows_as_the_sample_grows() -> None:
    small = grouped_records()
    large = grouped_records() * 10

    small_width = sweep(GROUPED, small, seed=0).ci_width
    large_width = sweep(GROUPED, large, seed=0).ci_width

    assert small_width >= large_width
    assert small_width > 0.0


# --------------------------------------------------------------------------------------
# flat_region
# --------------------------------------------------------------------------------------


def curve(*costs: float, step: float = 0.25) -> tuple[CostPoint, ...]:
    """A hand-built cost curve, one point per cost, evenly spaced."""
    return tuple(
        CostPoint(
            threshold=index * step,
            expected_cost=cost,
            auto_rate=1.0,
            accuracy_auto=1.0,
            accuracy_escalated=1.0,
            accept_cost=0.0,
            escalate_cost=0.0,
            n_auto=0,
            n_escalate=0,
            n_wrong_auto=0,
            n_wrong_escalate=0,
        )
        for index, cost in enumerate(costs)
    )


def test_flat_region_covers_the_run_around_the_cheapest_threshold() -> None:
    assert flat_region(curve(5.0, 1.0, 1.0, 1.0, 3.0)) == pytest.approx((0.25, 0.75))


def test_flat_region_includes_costs_at_the_tolerance_boundary() -> None:
    assert flat_region(curve(1.0, 1.05)) == pytest.approx((0.0, 0.25))
    assert flat_region(curve(1.0, 1.06)) is None


def test_flat_region_is_none_when_it_is_narrower_than_one_step() -> None:
    assert flat_region(curve(5.0, 1.0, 4.0)) is None
    assert flat_region(curve(1.0)) is None


def test_flat_region_is_none_without_a_measurable_curve() -> None:
    assert flat_region(()) is None
    assert flat_region(curve(1.0, float("nan"))) is None

    with pytest.raises(ValueError, match="tolerance must be >= 0"):
        flat_region(curve(1.0, 1.0), tolerance=-0.1)


def test_flat_region_does_not_care_about_the_order_of_the_curve() -> None:
    ordered = curve(5.0, 1.0, 1.0, 1.0, 3.0)
    shuffled = (ordered[3], ordered[0], ordered[4], ordered[1], ordered[2])

    assert flat_region(shuffled) == flat_region(ordered)


def test_flat_region_can_span_the_whole_sweep() -> None:
    assert flat_region(curve(1.0, 1.0, 1.0, 1.0, 1.0)) == pytest.approx((0.0, 1.0))


# --------------------------------------------------------------------------------------
# build_impact
# --------------------------------------------------------------------------------------


def impact_result() -> ThresholdResult:
    return ThresholdResult(
        action="auto_refund",
        question="intent",
        when="refund_request",
        threshold=0.85,
        expected_cost_per_case=8.0,
        auto_rate=0.55,
        accuracy_auto=0.97,
        ci_low=0.80,
        ci_high=0.92,
        curve=(
            CostPoint(0.50, 20.0, 0.80, 0.90, 1.0, 0.0, 0.0, 0, 0, 0, 0),
            CostPoint(0.85, 8.0, 0.55, 0.97, 1.0, 0.0, 0.0, 0, 0, 0, 0),
        ),
        flat_region=(0.80, 0.90),
        n_records=400,
        models=("jev-1.13.0",),
        cost_false_accept=100.0,
        cost_escalate=5.0,
        cost_false_reject=2.0,
    )


def test_build_impact_formats_every_cell() -> None:
    table = build_impact(impact_result(), current_threshold=0.50)

    assert [(row.label, row.current, row.recommended, row.change) for row in table.rows] == [
        ("confidence threshold", "0.50", "0.85", "+0.35"),
        ("auto rate", "80%", "55%", "-25.0 pt"),
        ("accuracy (auto)", "90%", "97%", "+7.0 pt"),
        ("cost per case", "USD 20.00", "USD 8.00", "-60.0%"),
    ]
    assert table.current_threshold == 0.50
    assert table.recommended_threshold == 0.85
    assert table.monthly_volume is None
    assert table.currency == "USD"


def test_build_impact_adds_the_monthly_row_only_with_a_volume() -> None:
    without = build_impact(impact_result(), current_threshold=0.50)
    with_volume = build_impact(impact_result(), current_threshold=0.50, monthly_volume=20000)

    assert len(without.rows) == 4
    assert len(with_volume.rows) == 5
    monthly = with_volume.rows[-1]
    assert monthly.label == "monthly cost"
    assert monthly.current == "USD 400,000.00"
    assert monthly.recommended == "USD 160,000.00"
    assert monthly.change == "-60.0%"
    assert with_volume.monthly_volume == 20000


def test_build_impact_uses_the_requested_currency() -> None:
    table = build_impact(impact_result(), current_threshold=0.50, currency="KRW")

    cost_row = next(row for row in table.rows if row.label == "cost per case")
    assert cost_row.current == "KRW 20.00"
    assert cost_row.recommended == "KRW 8.00"
    assert table.currency == "KRW"


def test_build_impact_explains_falling_auto_rate_with_falling_cost() -> None:
    table = build_impact(impact_result(), current_threshold=0.50)

    assert table.paradox_note != ""
    assert "Auto rate falls from 80% to 55%" in table.paradox_note
    assert "cost per case falls from USD 20.00 to USD 8.00" in table.paradox_note
    assert table.paradox_note.endswith(".")
    assert table.paradox_note.count(". ") == 0


def test_build_impact_stays_quiet_when_the_auto_rate_does_not_fall() -> None:
    table = build_impact(impact_result(), current_threshold=0.85)

    assert table.paradox_note == ""
    assert table.rows[1].current == "55%"
    assert table.rows[1].change == "+0.0 pt"


def test_build_impact_uses_the_nearest_curve_point_for_an_off_grid_threshold() -> None:
    table = build_impact(impact_result(), current_threshold=0.82)

    assert table.rows[0].current == "0.82"
    # 0.82 is nearest to 0.85 in the curve, so the cost cell reads that point's cost.
    assert table.rows[3].current == "USD 8.00"


def test_build_impact_explains_the_paradox_on_real_sweep_output() -> None:
    result = sweep(GROUPED, grouped_records(), seed=0)

    table = build_impact(result, current_threshold=0.0, monthly_volume=1000)

    assert table.paradox_note != ""
    assert "Auto rate falls from 100% to 60%" in table.paradox_note
    assert table.rows[1].current == "100%"
    assert table.rows[1].recommended == "60%"
    assert table.rows[1].change == "-40.0 pt"
    assert table.rows[3].current == "USD 32.00"
    assert table.rows[3].recommended == "USD 4.20"
    assert table.rows[4].change == "-86.9%"


def test_build_impact_says_n_a_when_the_threshold_was_withheld() -> None:
    result = sweep(REFUND, repeated(20, "refund_request", 0.9, "refund_request"), seed=0)

    table = build_impact(result, current_threshold=0.89)

    assert len(table.rows) == 4
    assert table.rows[0].current == "0.89"
    for row in table.rows:
        assert row.recommended == "n/a"
        assert row.change == "n/a"
    assert table.paradox_note == ""
    assert math.isnan(table.recommended_threshold)


# --------------------------------------------------------------------------------------
# write_thresholds_yaml
# --------------------------------------------------------------------------------------


def test_write_thresholds_yaml_round_trips(tmp_path: Path) -> None:
    good = sweep(GROUPED, grouped_records(), seed=0)
    other = sweep(
        CostAction("auto_close", "severity", "low", 20.0, 3.0, 1.0),
        repeated(40, "low", 0.9, "low", question="severity"),
        seed=0,
    )
    target = tmp_path / "nested" / "thresholds.yaml"

    written = write_thresholds_yaml([good, other], target, generated_at="2026-09-20T12:00:00Z")

    assert written == target
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert list(payload) == ["generated_at", "model", "n_records", "actions"]
    assert payload["generated_at"] == "2026-09-20T12:00:00Z"
    assert payload["model"] == "jev-1.13.0"
    assert payload["n_records"] == good.n_records + other.n_records
    assert list(payload["actions"]) == ["auto_refund", "auto_close"]

    entry = payload["actions"]["auto_refund"]
    assert entry["question"] == "intent"
    assert entry["when"] == "refund_request"
    assert entry["threshold"] == pytest.approx(0.95)
    assert entry["expected_cost_per_case"] == pytest.approx(4.2)
    assert entry["auto_rate"] == pytest.approx(0.60)
    assert entry["accuracy_auto"] == pytest.approx(58 / 60, rel=1e-5)
    assert entry["ci"] == {"low": pytest.approx(0.95), "high": pytest.approx(1.0)}
    assert entry["flat_region"] == {"low": pytest.approx(0.61), "high": pytest.approx(0.95)}
    assert entry["models"] == ["jev-1.13.0"]
    assert entry["n_records"] == 100
    assert entry["insufficient_data"] is False

    assert payload["actions"]["auto_close"]["question"] == "severity"
    assert payload["actions"]["auto_close"]["threshold"] == pytest.approx(0.90)


def test_write_thresholds_yaml_writes_nulls_for_withheld_numbers(tmp_path: Path) -> None:
    withheld = sweep(REFUND, repeated(20, "refund_request", 0.9, "refund_request"), seed=0)
    target = tmp_path / "thresholds.yaml"

    write_thresholds_yaml([withheld], target, generated_at="2026-09-20T12:00:00Z")

    text = target.read_text(encoding="utf-8")
    assert "nan" not in text.lower()
    entry = yaml.safe_load(text)["actions"]["auto_refund"]
    assert entry["threshold"] is None
    assert entry["expected_cost_per_case"] is None
    assert entry["auto_rate"] is None
    assert entry["accuracy_auto"] is None
    assert entry["ci"] is None
    assert entry["flat_region"] is None
    assert entry["n_records"] == 20
    assert entry["insufficient_data"] is True
    assert entry["models"] == ["jev-1.13.0"]


def test_write_thresholds_yaml_rejects_a_repeated_action(tmp_path: Path) -> None:
    result = sweep(GROUPED, grouped_records(), seed=0)

    with pytest.raises(ValueError, match="duplicate action"):
        write_thresholds_yaml(
            [result, result], tmp_path / "thresholds.yaml", generated_at="2026-09-20T12:00:00Z"
        )


# --------------------------------------------------------------------------------------
# synthetic data, end to end
# --------------------------------------------------------------------------------------


def test_synthetic_overconfidence_moves_the_threshold_up() -> None:
    spec = SynthSpec(
        n=600,
        mode="overconfident",
        exponent=0.7,
        question_key="intent",
        classes=("refund_request", "check_balance", "other"),
        seed=9,
    )
    records = generate(spec)
    gold = [item for item in records if item.is_gold]
    action = CostAction("auto", "intent", "refund_request", 50.0, 5.0, 1.0)

    result = sweep(action, records, seed=0)

    assert result.n_records == len(gold)
    # Reported confidence is inflated above the probability the answer was actually drawn with,
    # so the cost-optimal line cannot sit at the model's own idea of "confident".
    assert result.threshold > 0.5
    assert result.ci_low <= result.ci_high
    assert result.flat_region is not None
    assert result.flat_region[0] <= result.threshold <= result.flat_region[1]
    assert 0.0 <= result.expected_cost_per_case <= 50.0


def test_a_when_class_the_model_never_predicts_is_never_automated() -> None:
    spec = SynthSpec(
        n=200,
        mode="calibrated",
        question_key="intent",
        classes=("refund_request", "check_balance", "other"),
        seed=13,
    )
    # The condition under test: the action's trigger class is never the model's answer. The
    # generator now answers every class it is given, so the condition is built here rather than
    # assumed from its distribution.
    records = [item for item in generate(spec) if item.prediction != "check_balance"]
    action = CostAction("auto", "intent", "check_balance", 50.0, 5.0, 1.0)

    result = sweep(action, records, seed=0)

    when_labels = sum(1 for item in records if item.is_gold and item.label == "check_balance")
    # Nothing can run automatically, so every case escalates: 5 per case, plus 1 for the cases
    # that really were the action's trigger class.
    assert {point.n_auto for point in result.curve} == {0}
    assert result.auto_rate == 0.0
    assert result.expected_cost_per_case == pytest.approx(
        (5.0 * result.n_records + when_labels) / result.n_records
    )
    assert result.expected_cost_per_case == pytest.approx(5.0 + when_labels / result.n_records)
    assert result.threshold == 1.0


def test_thresholds_file_reads_back_the_number_it_wrote(tmp_path: Path) -> None:
    spec = SynthSpec(n=500, mode="calibrated", question_key="intent", seed=21)
    records = generate(spec)
    action = CostAction("auto", "intent", "refund_request", 30.0, 4.0, 1.0)
    result = sweep(action, records, seed=0)
    target = tmp_path / "thresholds.yaml"

    write_thresholds_yaml([result], target, generated_at="2026-09-20T12:00:00Z")
    entry = yaml.safe_load(target.read_text(encoding="utf-8"))["actions"]["auto"]

    assert entry["threshold"] == pytest.approx(result.threshold)
    assert entry["expected_cost_per_case"] == pytest.approx(result.expected_cost_per_case)
    assert entry["auto_rate"] == pytest.approx(result.auto_rate)
    assert entry["n_records"] == result.n_records
    assert entry["insufficient_data"] is False
