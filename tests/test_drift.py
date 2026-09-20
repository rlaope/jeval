"""Drift engine tests.

The fixtures are two-model logs built with ``jeval.synth``, and the newer model is deliberately
worse: its confidence is inflated by a known factor. A correct engine has to report a positive ECE
increase on that question and fail ``ece-increase``. A question that was re-run unchanged has to
come out clean, so the check cannot be passing by firing on everything.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from jeval.drift import (
    DriftCheck,
    compare,
    detect_model_changes,
    format_ci_block,
    parse_fail_on,
    run_checks,
    split_by_period,
    split_models,
)
from jeval.report.model import DriftFailure, DriftSlice, DriftView
from jeval.schema import DecisionRecord
from jeval.synth import SynthSpec, generate

OLDER = "jev-1.13.0"
NEWER = "jev-1.14.0"
INTENT_CLASSES = ("refund_request", "check_balance", "other")
# Noon so a 600-record batch (10 hours of one-minute decisions) cannot spill into the day or the
# ISO week before it.
OLD_START = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
NEW_START = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)


def _two_model_log(*, n: int = 600) -> list[DecisionRecord]:
    """Two model versions over two questions, the newer one inflated on ``department``.

    ``department`` is the drift a correct engine must catch. ``intent`` is the same calibrated
    population on both sides, so it must report a movement inside the noise floor and no failure.
    """
    return [
        *generate(SynthSpec(n=n, mode="calibrated", model=OLDER, start=OLD_START, seed=21)),
        *generate(
            SynthSpec(
                n=n,
                mode="calibrated",
                model=OLDER,
                start=OLD_START,
                seed=23,
                question_key="intent",
                classes=INTENT_CLASSES,
            )
        ),
        *generate(
            SynthSpec(
                n=n,
                mode="inflated",
                inflation=1.35,
                model=NEWER,
                start=NEW_START,
                seed=22,
            )
        ),
        *generate(
            SynthSpec(
                n=n,
                mode="calibrated",
                model=NEWER,
                start=NEW_START,
                seed=24,
                question_key="intent",
                classes=INTENT_CLASSES,
            )
        ),
    ]


def _record(
    ts: datetime,
    *,
    model: str = OLDER,
    confidence: float = 0.9,
    correct: bool = True,
    question_key: str = "department",
) -> DecisionRecord:
    """A gold-labeled choice record, for fixtures that need exact timestamps."""
    return DecisionRecord(
        ts=ts,
        model=model,
        question_key=question_key,
        question_type="choice",
        prediction="billing",
        confidence=confidence,
        label="billing" if correct else "technical",
        label_source="human_override",
    )


def _slice(
    label: str,
    model: str,
    ece: float,
    *,
    threshold: float | None = None,
    auto_rate: float | None = None,
) -> DriftSlice:
    return DriftSlice(
        label=label,
        model=model,
        start="2026-09-01T00:00:00+00:00",
        end="2026-09-05T00:00:00+00:00",
        n=200,
        ece=ece,
        threshold=threshold,
        auto_rate=auto_rate,
    )


def test_split_models_orders_by_first_seen() -> None:
    slices = split_models(_two_model_log(n=2400))

    assert [s.label for s in slices] == [OLDER, NEWER]
    assert [s.model for s in slices] == [OLDER, NEWER]
    older, newer = slices
    assert older.n == 4800 and newer.n == 4800
    # ISO timestamps with the same UTC offset sort chronologically as strings.
    assert older.start < newer.start
    assert older.ece < 0.02, older.ece
    assert newer.ece > older.ece, (older.ece, newer.ece)


def test_split_models_counts_only_measurable_gold_records() -> None:
    records = generate(
        SynthSpec(
            n=200,
            mode="calibrated",
            model=OLDER,
            start=OLD_START,
            seed=31,
            label_fraction=0.4,
            silver_fraction=0.5,
        )
    )
    measured = split_models(records)

    assert len(measured) == 1
    assert measured[0].n == len([r for r in records if r.is_gold])
    assert measured[0].n < len([r for r in records if r.is_labeled])
    assert measured[0].n > 0


def test_split_models_keeps_a_model_with_nothing_measurable() -> None:
    records = generate(
        SynthSpec(
            n=120,
            mode="constant_high",
            question_type="score",
            model=OLDER,
            start=OLD_START,
            seed=32,
        )
    )
    measured = split_models(records)

    assert len(measured) == 1
    assert measured[0].n == 0
    assert measured[0].ece != measured[0].ece  # NaN: no fake number for an unmeasurable model


def test_split_by_period_uses_iso_weeks_ordered_by_start() -> None:
    records = [
        _record(datetime(2026, 9, 14, 9, 0, tzinfo=timezone.utc)),
        _record(datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc)),
        _record(datetime(2026, 9, 7, 9, 0, tzinfo=timezone.utc)),
    ]
    slices = split_by_period(records, period="W")

    assert [s.label for s in slices] == ["2026-W36", "2026-W37", "2026-W38"]
    assert [s.n for s in slices] == [1, 1, 1]


def test_split_by_period_uses_calendar_months() -> None:
    records = [
        _record(datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)),
        _record(datetime(2026, 8, 31, 9, 0, tzinfo=timezone.utc)),
        _record(datetime(2026, 9, 20, 9, 0, tzinfo=timezone.utc)),
    ]
    slices = split_by_period(records, period="M")

    assert [s.label for s in slices] == ["2026-08", "2026-09", "2026-10"]


def test_split_by_period_rejects_an_unknown_period() -> None:
    with pytest.raises(ValueError, match="unknown period"):
        split_by_period([_record(datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc))], period="Y")  # type: ignore[arg-type]


def test_period_slices_carry_the_model_serving_at_the_end() -> None:
    records = [
        _record(datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc), model=OLDER),
        _record(datetime(2026, 9, 8, 9, 0, tzinfo=timezone.utc), model=OLDER),
        _record(datetime(2026, 9, 8, 14, 0, tzinfo=timezone.utc), model=NEWER),
        _record(datetime(2026, 9, 15, 9, 0, tzinfo=timezone.utc), model=NEWER),
    ]
    slices = split_by_period(records, period="W")

    assert [s.label for s in slices] == ["2026-W36", "2026-W37", "2026-W38"]
    assert [s.model for s in slices] == [OLDER, NEWER, NEWER]


def test_detect_model_changes_records_every_transition_in_order() -> None:
    log = _two_model_log(n=40)
    back = generate(
        SynthSpec(
            n=40,
            mode="calibrated",
            model=OLDER,
            start=datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc),
            seed=33,
        )
    )
    changes = detect_model_changes([*back, *reversed(log)])

    assert [c.model for c in changes] == [NEWER, OLDER]
    assert [c.label for c in changes] == [f"{OLDER} -> {NEWER}", f"{NEWER} -> {OLDER}"]
    # The change is dated at the first record of the new model, not at the last one of the old.
    first_newer = min(
        (r.ts for r in log if r.model == NEWER),
        key=lambda ts: ts.astimezone(timezone.utc),
    )
    assert changes[0].at == first_newer.astimezone(timezone.utc).isoformat()
    assert changes[0].at < changes[1].at
    assert detect_model_changes(log[:10]) == ()  # one model so far is not a change


def test_compare_reports_a_positive_ece_increase_and_fails_ece_increase() -> None:
    view = compare(_two_model_log())

    assert view.baseline_label == OLDER
    assert view.current_label == NEWER
    assert [(s.label, s.model) for s in view.slices] == [
        ("department", OLDER),
        ("intent", OLDER),
        ("department", NEWER),
        ("intent", NEWER),
    ]
    department_before, department_after = view.slices[0], view.slices[2]
    intent_before, intent_after = view.slices[1], view.slices[3]
    assert department_before.n == 600 and department_after.n == 600
    assert department_after.ece > department_before.ece
    rise = department_after.ece - department_before.ece
    assert rise > 0.05, rise
    assert intent_after.ece - intent_before.ece < 0.05
    assert [c.label for c in view.changes] == [f"{OLDER} -> {NEWER}"]

    failures = run_checks(view, parse_fail_on(["ece-increase=0.05"]))

    assert len(failures) == 1
    assert failures[0].check == "ece-increase"
    assert failures[0].value == pytest.approx(rise)
    assert failures[0].limit == 0.05
    assert failures[0].detail.startswith("department: ")


def test_compare_omits_small_slices_and_says_so() -> None:
    rare = [
        *generate(
            SynthSpec(
                n=6,
                mode="calibrated",
                model=OLDER,
                start=OLD_START,
                seed=34,
                question_key="rare_question",
            )
        ),
        *generate(
            SynthSpec(
                n=6,
                mode="inflated",
                inflation=1.5,
                model=NEWER,
                start=NEW_START,
                seed=35,
                question_key="rare_question",
            )
        ),
    ]
    view = compare([*_two_model_log(), *rare])
    failures = run_checks(view, parse_fail_on(["ece-increase=0.05"]))

    assert {s.label for s in view.slices} == {"department", "intent"}
    assert "min_slice=30" in view.note
    assert "rare_question (6 vs 6)" in view.note
    assert {f.detail.split(":")[0] for f in failures} == {"department"}


def test_compare_with_one_model_explains_what_drift_needs() -> None:
    view = compare(
        generate(SynthSpec(n=120, mode="calibrated", model=OLDER, start=OLD_START, seed=26))
    )

    assert view.slices == ()
    assert view.changes == ()
    assert "two model versions" in view.note
    assert "period split" in view.note
    assert run_checks(view, parse_fail_on(["ece-increase=0.05"])) == ()

    block = format_ci_block(view, ())
    assert "period split" in block
    assert block.splitlines()[-1] == "exit 0"


def test_compare_defaults_to_the_two_most_recent_models() -> None:
    oldest = generate(
        SynthSpec(
            n=60,
            mode="calibrated",
            model="jev-1.12.0",
            start=datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc),
            seed=61,
        )
    )
    records = [*oldest, *_two_model_log(n=200)]

    assert compare(records).baseline_label == OLDER
    assert compare(records).current_label == NEWER
    explicit = compare(records, baseline="jev-1.12.0", current=NEWER)
    assert explicit.baseline_label == "jev-1.12.0"
    assert {s.model for s in explicit.slices} == {"jev-1.12.0", NEWER}
    with pytest.raises(ValueError, match=re.escape("jev-9.9.9")):
        compare(records, baseline="jev-9.9.9")


def test_compare_measures_the_compared_population_only() -> None:
    """A difference in the table has to be a difference in the data, not in the binning."""
    log = _two_model_log()
    plain = compare(log, baseline=OLDER, current=NEWER)
    unrelated = generate(
        SynthSpec(
            n=800,
            mode="constant_high",
            model="jev-1.15.0-rc",
            start=datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc),
            seed=51,
            question_key="unrelated_question",
            constant_confidence=0.99,
            accuracy_target=0.6,
        )
    )
    extended = compare([*log, *unrelated], baseline=OLDER, current=NEWER)

    assert [(s.label, s.model, s.n, s.ece) for s in extended.slices] == [
        (s.label, s.model, s.n, s.ece) for s in plain.slices
    ]


def test_compare_by_period_reports_a_worsening_trend() -> None:
    weeks = [
        generate(
            SynthSpec(
                n=300,
                mode=mode,
                inflation=inflation,
                model=OLDER,
                start=datetime(2026, 9, day, 12, 0, tzinfo=timezone.utc),
                seed=seed,
            )
        )
        for day, mode, inflation, seed in (
            (8, "calibrated", 1.0, 41),
            (15, "inflated", 1.35, 42),
            (22, "inflated", 1.5, 43),
        )
    ]
    view = compare([record for batch in weeks for record in batch], by_period="W")

    assert [s.label for s in view.slices] == ["2026-W37", "2026-W38", "2026-W39"]
    assert view.baseline_label == "2026-W37"
    assert view.current_label == "2026-W39"
    eces = [s.ece for s in view.slices]
    assert eces[0] < eces[1] < eces[2], eces

    failures = run_checks(view, parse_fail_on(["ece-increase=0.05", "ece-above=0.10"]))

    # The 1.35 inflation is a rise over the limit; 1.35 -> 1.5 is not, and only the last period is
    # judged by the absolute limit.
    assert eces[2] - eces[1] < 0.05, eces
    assert [f.check for f in failures] == ["ece-increase", "ece-above"]
    assert [f.detail.split(":")[0] for f in failures] == ["2026-W38", "2026-W39"]
    assert failures[0].value == pytest.approx(eces[1] - eces[0])
    assert failures[1].value == pytest.approx(eces[2])


def test_ece_above_is_checked_against_the_current_slice_only() -> None:
    view = DriftView(
        baseline_label=OLDER,
        current_label=NEWER,
        slices=(_slice("q1", OLDER, 0.30), _slice("q1", NEWER, 0.02)),
    )

    assert run_checks(view, parse_fail_on(["ece-above=0.10"])) == ()


def test_nan_slices_cannot_fail_a_check() -> None:
    view = DriftView(
        baseline_label=OLDER,
        current_label=NEWER,
        slices=(
            _slice("q1", OLDER, float("nan")),
            _slice("q1", NEWER, float("nan")),
        ),
    )

    assert run_checks(view, parse_fail_on(["ece-increase=0.05", "ece-above=0.05"])) == ()


def test_parse_fail_on_accepts_the_three_checks() -> None:
    checks = parse_fail_on(["ece-increase=0.05", "auto-rate-drop=0.15", "ece-above=0.10"])

    assert [(c.key, c.limit) for c in checks] == [
        ("ece-increase", 0.05),
        ("auto-rate-drop", 0.15),
        ("ece-above", 0.10),
    ]
    assert parse_fail_on([]) == ()


@pytest.mark.parametrize(
    "spec",
    ["nonsense=0.1", "ece-increase=big", "ece-increase", "auto-rate-drop=-1", "ece-above=nan"],
)
def test_parse_fail_on_rejects_an_unknown_key_or_a_bad_limit(spec: str) -> None:
    with pytest.raises(ValueError, match=re.escape(spec)):
        parse_fail_on([spec])


def test_run_checks_compares_each_unit_and_skips_a_missing_auto_rate() -> None:
    view = DriftView(
        baseline_label=OLDER,
        current_label=NEWER,
        slices=(
            _slice("q1", OLDER, 0.10, threshold=0.9, auto_rate=0.6),
            _slice("q2", OLDER, 0.09),
            _slice("q1", NEWER, 0.20, threshold=0.8, auto_rate=0.4),
            _slice("q2", NEWER, 0.095),
        ),
    )
    failures = run_checks(
        view, parse_fail_on(["ece-increase=0.05", "auto-rate-drop=0.15", "ece-above=0.15"])
    )

    assert [f.check for f in failures] == ["ece-increase", "auto-rate-drop", "ece-above"]
    assert [f.detail.split(":")[0] for f in failures] == ["q1", "q1", "q1"]
    assert failures[0].value == pytest.approx(0.10)
    assert failures[1].value == pytest.approx(0.20)
    assert failures[2].value == pytest.approx(0.20)


def test_run_checks_rejects_a_check_it_cannot_evaluate() -> None:
    with pytest.raises(ValueError, match="nonsense"):
        run_checks(
            DriftView(baseline_label=OLDER, current_label=NEWER), [DriftCheck("nonsense", 1.0)]
        )


def test_format_ci_block_prints_the_change_the_rows_and_the_exit_code() -> None:
    view = compare(_two_model_log())
    failures = run_checks(view, parse_fail_on(["ece-increase=0.05"]))
    lines = format_ci_block(view, failures).splitlines()

    assert lines[0] == f"model changed: {OLDER} -> {NEWER} (Sep 15)"
    assert lines[1].split() == ["question", "ECE", "before", "ECE", "after", "delta"]
    assert lines[2].startswith("  department") and lines[2].endswith("FAIL")
    assert lines[3].startswith("  intent") and lines[3].endswith("ok")
    assert sum(1 for line in lines if line.endswith("FAIL")) == 1
    assert lines[-1] == "exit 1"
    assert "\x1b[" not in "\n".join(lines)

    department_before, department_after = view.slices[0], view.slices[2]
    assert re.findall(r"[-+]?\d+\.\d{3}", lines[2]) == [
        f"{department_before.ece:.3f}",
        f"{department_after.ece:.3f}",
        f"{department_after.ece - department_before.ece:+.3f}",
    ]
    intent_before, intent_after = view.slices[1], view.slices[3]
    assert re.findall(r"[-+]?\d+\.\d{3}", lines[3]) == [
        f"{intent_before.ece:.3f}",
        f"{intent_after.ece:.3f}",
        f"{intent_after.ece - intent_before.ece:+.3f}",
    ]


def test_format_ci_block_prints_threshold_and_auto_rate_movement() -> None:
    view = DriftView(
        baseline_label=OLDER,
        current_label=NEWER,
        slices=(
            _slice("department", OLDER, 0.061, threshold=0.89, auto_rate=0.58),
            _slice("department", NEWER, 0.142, threshold=0.82, auto_rate=0.41),
        ),
    )
    block = format_ci_block(view, run_checks(view, parse_fail_on(["ece-increase=0.05"])))

    assert "recommended threshold (department): 0.89 -> 0.82" in block
    assert "at the current 0.89: auto-rate 58% -> 41%" in block
    assert block.splitlines()[-1] == "exit 1"


def test_format_ci_block_says_when_no_threshold_was_computed() -> None:
    view = compare(_two_model_log())
    block = format_ci_block(view, ())

    assert all(s.threshold is None and s.auto_rate is None for s in view.slices)
    assert "recommended threshold: not available" in block
    assert block.splitlines()[-1] == "exit 0"


def test_format_ci_block_without_a_comparison_prints_the_reason() -> None:
    view = DriftView(baseline_label="", current_label="", note="nothing to compare yet")

    assert format_ci_block(view, ()) == "nothing to compare yet\nexit 0"
    assert format_ci_block(
        view, [DriftFailure("ece-above", "q1: current ECE 0.200", 0.2, 0.1)]
    ) == ("nothing to compare yet\nexit 1")
