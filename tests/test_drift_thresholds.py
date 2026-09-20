"""Cost-derived thresholds on drift slices, and what the CI block does with them.

A drift comparison on its own only says that calibration moved. What a reviewer acts on is the
line the cost matrix moves, so these tests are about the wiring between the two: a two-model log
whose timestamps are separated, the newer model miscalibrated on purpose, one cost matrix, and a
block that has to show both the threshold movement and the automation rate it implies — or say
plainly that it cannot.

Fixtures come from ``jeval.synth``. The models are given clearly separated start times so the
sorted series never interleaves them; without that, every neighbouring pair of records looks like
a model change and the comparison measures nothing.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import pytest

from jeval.costs import CostAction, sweep
from jeval.drift import attach_thresholds, compare, format_ci_block, parse_fail_on, run_checks
from jeval.report.model import DriftSlice, DriftView
from jeval.schema import DecisionRecord
from jeval.synth import SynthSpec, generate

OLDER = "jev-1.13.0"
NEWER = "jev-1.14.0"
INTENT_CLASSES = ("refund_request", "check_balance", "other")
# Noon so a batch of a few hundred one-minute decisions stays inside the day it starts on.
OLD_START = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
NEW_START = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)

AUTO_REFUND = CostAction(
    name="auto_refund",
    question="intent",
    when="refund_request",
    cost_false_accept=50_000.0,
    cost_escalate=2_000.0,
    cost_false_reject=0.0,
)
AUTO_RARE = CostAction(
    name="auto_route",
    question="rare_question",
    when="billing",
    cost_false_accept=8_000.0,
    cost_escalate=1_500.0,
    cost_false_reject=200.0,
)


def _side(
    n: int,
    *,
    mode: str,
    model: str,
    start: datetime,
    seed: int,
    question_key: str = "intent",
    classes: tuple[str, ...] = INTENT_CLASSES,
    inflation: float = 1.35,
    exponent: float = 0.6,
) -> list[DecisionRecord]:
    return generate(
        SynthSpec(
            n=n,
            mode=mode,
            model=model,
            start=start,
            seed=seed,
            question_key=question_key,
            classes=classes,
            inflation=inflation,
            exponent=exponent,
        )
    )


def _inflated_log(n: int = 300) -> list[DecisionRecord]:
    """A calibrated model replaced by an inflated one: confidence rises above the accuracy."""
    return [
        *_side(n, mode="calibrated", model=OLDER, start=OLD_START, seed=21),
        *_side(n, mode="inflated", model=NEWER, start=NEW_START, seed=22, inflation=1.35),
    ]


def _underconfident_log(n: int = 200) -> list[DecisionRecord]:
    """A calibrated model replaced by one shrunk toward 0.5: it escalates more for the same work."""
    return [
        *_side(n, mode="calibrated", model=OLDER, start=OLD_START, seed=21),
        *_side(n, mode="underconfident", model=NEWER, start=NEW_START, seed=22, exponent=0.6),
    ]


def _rare_question(n: int = 12) -> list[DecisionRecord]:
    """A question with too few gold records for a cost sweep, on both model versions."""
    return [
        *_side(
            n,
            mode="calibrated",
            model=OLDER,
            start=OLD_START,
            seed=31,
            question_key="rare_question",
            classes=("billing", "technical", "other"),
        ),
        *_side(
            n,
            mode="calibrated",
            model=NEWER,
            start=NEW_START,
            seed=32,
            question_key="rare_question",
            classes=("billing", "technical", "other"),
        ),
    ]


def _slice(view: DriftView, label: str, model: str) -> DriftSlice:
    matches = [s for s in view.slices if s.label == label and s.model == model]
    assert len(matches) == 1, [(s.label, s.model) for s in view.slices]
    return matches[0]


def _records_of(records: list[DecisionRecord], label: str, model: str) -> list[DecisionRecord]:
    return [r for r in records if r.question_key == label and r.model == model]


def test_each_side_is_swept_over_its_own_records() -> None:
    records = _inflated_log()
    view = attach_thresholds(compare(records), records, [AUTO_REFUND])
    before = _slice(view, "intent", OLDER)
    after = _slice(view, "intent", NEWER)
    before_threshold, after_threshold = before.threshold, after.threshold
    before_rate, after_rate = before.auto_rate, after.auto_rate

    assert before_threshold is not None and after_threshold is not None
    assert before_rate is not None and after_rate is not None

    # The numbers have to come from that side's own records: sweeping the pooled question instead
    # would hand both models the same threshold and hide the movement entirely.
    older = sweep(AUTO_REFUND, _records_of(records, "intent", OLDER))
    newer = sweep(AUTO_REFUND, _records_of(records, "intent", NEWER))
    assert before_threshold == pytest.approx(older.threshold)
    assert before_rate == pytest.approx(older.auto_rate)
    assert after_threshold == pytest.approx(newer.threshold)
    assert after_rate == pytest.approx(newer.auto_rate)

    # An inflated model has to clear a higher bar, and more of its inflated confidences clear it.
    assert after_threshold > before_threshold
    assert after_rate > before_rate


def test_the_block_reports_the_threshold_movement_and_the_auto_rate() -> None:
    records = _inflated_log()
    view = attach_thresholds(compare(records), records, [AUTO_REFUND])
    before = _slice(view, "intent", OLDER)
    after = _slice(view, "intent", NEWER)
    before_threshold, after_threshold = before.threshold, after.threshold
    before_rate, after_rate = before.auto_rate, after.auto_rate

    assert before_threshold is not None and after_threshold is not None
    assert before_rate is not None and after_rate is not None

    failures = run_checks(view, parse_fail_on(["ece-increase=0.05"]))
    block = format_ci_block(view, failures)

    # The degradation is real, so the movement cannot be an artefact of an unchanged population.
    assert [f.check for f in failures] == ["ece-increase"]
    assert "recommended threshold: not available" not in block
    assert (
        f"recommended threshold (intent): {before_threshold:.2f} -> {after_threshold:.2f}" in block
    )
    assert (
        f"at the current {before_threshold:.2f}: "
        f"auto-rate {before_rate:.0%} -> {after_rate:.0%}" in block
    )
    assert block.splitlines()[-1] == "exit 1"


def test_the_printed_line_names_the_compared_unit() -> None:
    """The threshold line is keyed by the compared unit's label, which is the question key.

    So the README's ``recommended threshold (auto_refund)`` is what a project whose question is
    named ``auto_refund`` prints; the same cost matrix on a question named ``intent`` prints
    ``(intent)``, because the threshold belongs to the question, not to the action that prices it.
    """
    records = [
        *_side(
            300,
            mode="calibrated",
            model=OLDER,
            start=OLD_START,
            seed=71,
            question_key="auto_refund",
        ),
        *_side(
            300, mode="inflated", model=NEWER, start=NEW_START, seed=72, question_key="auto_refund"
        ),
    ]
    action = CostAction(
        name="auto_refund",
        question="auto_refund",
        when="refund_request",
        cost_false_accept=50_000.0,
        cost_escalate=2_000.0,
        cost_false_reject=0.0,
    )
    block = format_ci_block(attach_thresholds(compare(records), records, [action]), ())

    assert re.search(r"recommended threshold \(auto_refund\): \d\.\d{2} -> \d\.\d{2}", block)
    assert re.search(r"at the current \d\.\d{2}: auto-rate \d+% -> \d+%", block)


def test_the_auto_rate_line_is_the_number_the_check_fails_on() -> None:
    records = _underconfident_log()
    view = attach_thresholds(compare(records), records, [AUTO_REFUND])
    before = _slice(view, "intent", OLDER)
    after = _slice(view, "intent", NEWER)
    before_threshold, after_threshold = before.threshold, after.threshold
    before_rate, after_rate = before.auto_rate, after.auto_rate

    assert before_threshold is not None and after_threshold is not None
    assert before_rate is not None and after_rate is not None
    drop = before_rate - after_rate

    # A model shrunk toward 0.5 escalates more: the recommended line moves down and so does the
    # share of cases the machine handles alone. Both numbers are finite, so the check can fire.
    assert after.ece > before.ece
    assert after_threshold < before_threshold
    assert drop > 0.10

    failures = run_checks(view, parse_fail_on(["auto-rate-drop=0.10"]))
    assert [f.check for f in failures] == ["auto-rate-drop"]
    assert failures[0].value == pytest.approx(drop)

    block = format_ci_block(view, failures)
    assert (
        f"recommended threshold (intent): {before_threshold:.2f} -> {after_threshold:.2f}" in block
    )
    assert (
        f"at the current {before_threshold:.2f}: "
        f"auto-rate {before_rate:.0%} -> {after_rate:.0%}" in block
    )
    assert block.splitlines()[-1] == "exit 1"


def test_the_cheapest_action_wins_when_two_price_the_same_question() -> None:
    records = _inflated_log()
    cheap_escalation = CostAction(
        name="auto_refund_lite",
        question="intent",
        when="refund_request",
        cost_false_accept=90_000.0,
        cost_escalate=100.0,
        cost_false_reject=0.0,
    )
    view = attach_thresholds(compare(records), records, [AUTO_REFUND, cheap_escalation])
    before = _slice(view, "intent", OLDER)
    older = _records_of(records, "intent", OLDER)
    cheaper = min(
        (sweep(action, older) for action in (AUTO_REFUND, cheap_escalation)),
        key=lambda result: result.expected_cost_per_case,
    )

    assert before.threshold == pytest.approx(cheaper.threshold)
    assert f"the cheapest recommendation ({cheaper.action}) is reported" in view.note
    assert "matched 2 cost actions" in view.note


def test_a_slice_too_small_to_sweep_keeps_none_and_the_block_says_so() -> None:
    records = [*_inflated_log(), *_rare_question()]
    # min_slice=5 lets the 12-record question into the comparison so the cost floor is what stops
    # the sweep, which is the case a caller has to be told about.
    view = attach_thresholds(compare(records, min_slice=5), records, [AUTO_REFUND, AUTO_RARE])
    rare_before = _slice(view, "rare_question", OLDER)
    rare_after = _slice(view, "rare_question", NEWER)

    assert rare_before.threshold is None and rare_after.threshold is None
    assert rare_before.auto_rate is None and rare_after.auto_rate is None
    assert _slice(view, "intent", NEWER).threshold is not None
    assert (
        f"rare_question at {OLDER} (12 gold-labeled record(s), under the 30 a sweep needs)"
        in view.note
    )

    block = format_ci_block(view, ())
    assert "recommended threshold (intent): " in block
    assert "recommended threshold (rare_question)" not in block
    assert "not available" not in block
    assert "under the 30 a sweep needs" in block


def test_a_cost_matrix_that_can_price_nothing_does_not_blame_a_missing_matrix() -> None:
    records = _rare_question()
    view = attach_thresholds(compare(records), records, [AUTO_RARE])
    block = format_ci_block(view, ())

    # The only question is below min_slice, so the comparison has no slice at all. The block must
    # not send the reader looking for a cost matrix that was supplied.
    assert view.slices == ()
    assert "not available" in block
    assert "no cost matrix was applied" not in block
    assert "the comparison has no slice to sweep" in block


def test_every_slice_too_small_says_why_instead_of_printing_a_number() -> None:
    records = _rare_question()
    view = attach_thresholds(compare(records, min_slice=5), records, [AUTO_RARE])
    block = format_ci_block(view, ())

    assert len(view.slices) == 2
    assert all(s.threshold is None and s.auto_rate is None for s in view.slices)
    assert "recommended threshold: not available" in block
    assert "no cost matrix was applied" not in block
    assert "under the 30 a sweep needs" in block
    # The count is the sweep's own, not the row count of the log.
    assert "12 gold-labeled record(s)" in block


def test_a_period_comparison_says_which_label_it_cannot_price() -> None:
    records = _inflated_log()
    view = attach_thresholds(compare(records, by_period="W"), records, [AUTO_REFUND])
    block = format_ci_block(view, ())

    # A period slice is labelled by period, not by question, so no action matches it and the
    # threshold stays unset rather than being attributed to a population it was not measured on.
    assert view.slices
    assert all(s.threshold is None and s.auto_rate is None for s in view.slices)
    assert "no cost action on this label" in view.note
    assert "recommended threshold: not available" in block
    assert "no cost matrix was applied" not in block


def test_without_a_cost_matrix_the_block_still_says_not_available() -> None:
    view = compare(_inflated_log())
    block = format_ci_block(view, run_checks(view, parse_fail_on(["auto-rate-drop=0.10"])))

    assert all(s.threshold is None and s.auto_rate is None for s in view.slices)
    assert "recommended threshold: not available (no cost matrix was applied" in block
    # Nothing to compare means the check is skipped, never treated as a pass.
    assert run_checks(view, parse_fail_on(["auto-rate-drop=0.10"])) == ()
    assert block.splitlines()[-1] == "exit 0"


def test_no_actions_at_all_means_no_cost_claim_is_made() -> None:
    records = _inflated_log()
    view = compare(records)

    # An empty cost matrix is not a cost matrix: claiming "cost matrix applied" here would send the
    # reader looking for a sweep that never ran.
    assert attach_thresholds(view, records, []) is view
    block = format_ci_block(view, ())
    assert "no cost matrix was applied" in block
    assert "cost matrix applied:" not in block


def test_attaching_twice_does_not_stack_two_cost_claims() -> None:
    records = _inflated_log()
    once = attach_thresholds(compare(records), records, [AUTO_REFUND])
    twice = attach_thresholds(once, records, [AUTO_REFUND])

    assert twice.note == once.note
    assert twice.note.count("cost matrix applied") == 1
    assert [s.threshold for s in twice.slices] == [s.threshold for s in once.slices]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"steps": 1}, "steps must be >= 2"), ({"alpha": 0.0}, "alpha must be in")],
)
def test_attach_thresholds_rejects_a_grid_or_level_it_cannot_sweep(
    kwargs: dict[str, Any], match: str
) -> None:
    records = _inflated_log(n=60)

    with pytest.raises(ValueError, match=re.escape(match)):
        attach_thresholds(compare(records), records, [AUTO_REFUND], **kwargs)
