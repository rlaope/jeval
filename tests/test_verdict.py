"""The verdict rules, one test per branch, plus the markdown the copy button hands over.

Every branch is driven by metrics built for that branch -- no tolerance is loosened to make a
status fire -- and the threshold inputs are the real report dataclasses as well as a duck-typed
stand-in, because the builder is documented to read either.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from jeval.calibration import CalibrationBin, CalibrationMetrics
from jeval.report.model import (
    CLEAR,
    INSUFFICIENT_DATA,
    TOO_HIGH,
    TOO_LOW,
    CostPoint,
    ImpactRow,
    ImpactTable,
    ReportModel,
    ThresholdResult,
    Verdict,
    VerdictStat,
)
from jeval.report.verdict import (
    HEADLINE_CLEAR,
    HEADLINE_INSUFFICIENT_DATA,
    HEADLINE_NEAR_MINIMUM,
    HEADLINE_NO_THRESHOLD,
    HEADLINE_TOO_HIGH,
    HEADLINE_TOO_LOW,
    build_verdict,
    markdown_summary,
)

WORST_LABEL = "0.50-0.75"
WORST_ACCURACY = "68.0%"
LABELS = "1,000"


# --------------------------------------------------------------------------------------
# Fixtures: metrics for each branch, thresholds of every supported shape
# --------------------------------------------------------------------------------------


def make_bin(
    lo: float, hi: float, n: int, confidence: float, accuracy: float, index: int
) -> CalibrationBin:
    return CalibrationBin(
        index=index,
        lo=lo,
        hi=hi,
        n=n,
        mean_confidence=confidence,
        accuracy=accuracy,
        ci_low=max(0.0, accuracy - 0.05),
        ci_high=min(1.0, accuracy + 0.05),
    )


def make_metrics(
    n: int = 1000,
    ece: float = 0.008,
    mce: float = 0.015,
    bins: tuple[CalibrationBin, ...] | None = None,
) -> CalibrationMetrics:
    """Metrics whose worst band is 0.50-0.75 at 68.0% on a fifth of the labels."""
    if bins is None:
        if n <= 0:
            bins = ()
        else:
            small = max(1, n // 5)
            bins = (
                make_bin(0.5, 0.75, small, 0.62, 0.68, index=0),
                make_bin(0.75, 1.0, n - small, 0.93, 0.90, index=1),
            )
    return CalibrationMetrics(
        n=n,
        ece=ece,
        mce=mce,
        brier=0.11,
        ece_ci_low=ece / 2.0,
        ece_ci_high=ece * 1.5,
        bins=bins,
        bins_requested=10,
        binning="quantile",
    )


def make_curve(step: float = 0.05) -> tuple[CostPoint, ...]:
    thresholds = [round(0.10 + step * index, 6) for index in range(17)]
    return tuple(
        CostPoint(
            threshold=threshold,
            expected_cost=0.42,
            auto_rate=0.5,
            accuracy_auto=0.9,
            accuracy_escalated=0.95,
            accept_cost=0.0,
            escalate_cost=1.0,
            n_auto=500,
            n_escalate=500,
            n_wrong_auto=50,
            n_wrong_escalate=25,
        )
        for threshold in thresholds
    )


def make_threshold_result(
    threshold: float = 0.70,
    *,
    curve: tuple[CostPoint, ...] | None = None,
    accuracy_auto: float | None = 0.9,
) -> ThresholdResult:
    return ThresholdResult(
        action="route",
        question="answerable",
        when="always",
        threshold=threshold,
        expected_cost_per_case=0.42,
        auto_rate=0.5,
        accuracy_auto=accuracy_auto if accuracy_auto is not None else 0.9,
        ci_low=0.87,
        ci_high=0.93,
        curve=make_curve() if curve is None else curve,
    )


def duck(thresholds: list[float]) -> SimpleNamespace:
    """A ThresholdResult-shaped object from a module this test does not import."""
    return SimpleNamespace(curve=tuple(SimpleNamespace(threshold=value) for value in thresholds))


# --------------------------------------------------------------------------------------
# Branch 1: not enough labels
# --------------------------------------------------------------------------------------


def test_insufficient_data_when_no_labels_exist() -> None:
    verdict = build_verdict(make_metrics(n=0, ece=float("nan"), mce=float("nan")), threshold=None)

    assert verdict.status == INSUFFICIENT_DATA
    assert verdict.headline == "Not enough labeled data to decide."
    # The detail names the count and says labels, not analysis, are the blocker.
    assert "0 labeled decisions exist" in verdict.detail
    assert "100 are needed" in verdict.detail
    assert "more labels, not more analysis, are the blocker" in verdict.detail


def test_insufficient_data_below_min_labels() -> None:
    verdict = build_verdict(
        make_metrics(n=99), threshold=make_threshold_result(), current_threshold=0.7
    )

    assert verdict.status == INSUFFICIENT_DATA
    assert verdict.headline == HEADLINE_INSUFFICIENT_DATA
    assert "99 labeled decisions exist" in verdict.detail
    assert "100 are needed" in verdict.detail
    # The label-count stat is what the reader needs here.
    assert [stat.label for stat in verdict.stats] == [
        "Current threshold",
        "Measured accuracy",
        "Recommended threshold",
    ]


def test_min_labels_is_configurable_and_labels_at_the_floor_are_enough() -> None:
    assert build_verdict(make_metrics(n=40), threshold=None, min_labels=40).status == CLEAR
    assert (
        build_verdict(make_metrics(n=39), threshold=None, min_labels=40).status == INSUFFICIENT_DATA
    )


def test_label_count_stat_appears_when_there_is_no_threshold() -> None:
    verdict = build_verdict(make_metrics(n=42, ece=0.05, mce=0.09), threshold=None, min_labels=100)

    assert verdict.status == INSUFFICIENT_DATA
    # No threshold to quote, so the label count is the figure the reader needs.
    assert verdict.stats == (
        VerdictStat(label="Measured accuracy", value="85.8%", sub="of 42 labels"),
        VerdictStat(label="Labeled decisions", value="42", sub="need 100"),
    )


# --------------------------------------------------------------------------------------
# Branch 2: calibrated inside tolerance
# --------------------------------------------------------------------------------------


def test_clear_when_ece_and_mce_are_within_tolerance() -> None:
    verdict = build_verdict(make_metrics(), threshold=None)

    assert verdict.status == CLEAR
    assert verdict.headline == "Confidence is trustworthy at the volumes you have."
    assert f"Band {WORST_LABEL} measures {WORST_ACCURACY} accuracy" in verdict.detail
    assert "ECE 0.008" in verdict.detail
    assert "MCE 0.015" in verdict.detail
    assert "0.02 tolerance" in verdict.detail


def test_tolerance_boundary_is_inside_and_just_past_it_is_not() -> None:
    # Within tolerance of zero, inclusive: exactly the tolerance still counts as calibrated.
    assert build_verdict(make_metrics(ece=0.02, mce=0.02)).headline == HEADLINE_CLEAR
    # Past it, with no threshold supplied: the honest fallback, not the trustworthy headline.
    assert build_verdict(make_metrics(ece=0.0205, mce=0.02)).headline == HEADLINE_NO_THRESHOLD


def test_clear_needs_both_ece_and_mce_inside_tolerance() -> None:
    verdict = build_verdict(
        make_metrics(n=1000, ece=0.001, mce=0.03),
        threshold=make_threshold_result(),
        current_threshold=0.7,
    )

    assert verdict.headline != HEADLINE_CLEAR
    assert verdict.status == CLEAR  # the thresholds agree, so the verdict is still clear


# --------------------------------------------------------------------------------------
# Branch 3 and 4: the threshold wants to move
# --------------------------------------------------------------------------------------


def test_too_low_when_the_recommendation_is_more_than_one_sweep_step_up() -> None:
    verdict = build_verdict(
        make_metrics(ece=0.05, mce=0.09),
        threshold=make_threshold_result(threshold=0.70),
        current_threshold=0.50,
    )

    assert verdict.status == TOO_LOW
    assert verdict.headline == HEADLINE_TOO_LOW
    assert verdict.headline == "Your threshold is too low."
    assert "the threshold belongs at 0.70" in verdict.detail
    assert "above the 0.50 in use" in verdict.detail


def test_too_high_when_the_recommendation_is_more_than_one_sweep_step_down() -> None:
    verdict = build_verdict(
        make_metrics(ece=0.05, mce=0.09),
        threshold=make_threshold_result(threshold=0.40),
        current_threshold=0.60,
    )

    assert verdict.status == TOO_HIGH
    assert verdict.headline == HEADLINE_TOO_HIGH
    assert verdict.headline == "Your threshold is too high."
    assert "the threshold belongs at 0.40" in verdict.detail
    assert "below the 0.60 in use" in verdict.detail


def test_exactly_one_sweep_step_is_not_a_move() -> None:
    # The curve steps by 0.05; 0.05 is not "more than one sweep step".
    verdict = build_verdict(
        make_metrics(ece=0.05, mce=0.09),
        threshold=make_threshold_result(threshold=0.55),
        current_threshold=0.50,
    )

    assert verdict.status == CLEAR
    assert verdict.headline == HEADLINE_NEAR_MINIMUM


def test_past_one_sweep_step_by_a_hair_does_move() -> None:
    verdict = build_verdict(
        make_metrics(ece=0.05, mce=0.09),
        threshold=make_threshold_result(threshold=0.5501),
        current_threshold=0.50,
    )

    assert verdict.status == TOO_LOW


def test_explicit_recommendation_without_a_curve_still_moves() -> None:
    # No curve means no known sweep step, so any difference at all is worth acting on.
    verdict = build_verdict(
        make_metrics(ece=0.05, mce=0.09),
        threshold=make_threshold_result(threshold=0.50, curve=()),
        current_threshold=0.50,
        recommended_threshold=0.62,
    )

    assert verdict.status == TOO_LOW
    assert "0.62" in verdict.detail


# --------------------------------------------------------------------------------------
# Branch 5: the thresholds agree
# --------------------------------------------------------------------------------------


def test_agreeing_thresholds_reuse_clear_with_the_cost_minimum_headline() -> None:
    verdict = build_verdict(
        make_metrics(ece=0.05, mce=0.09),
        threshold=make_threshold_result(threshold=0.70),
        current_threshold=0.70,
    )

    assert verdict.status == CLEAR
    assert verdict.headline == "Your current threshold is already near the cost minimum."
    assert "within one sweep step of the 0.70 cost minimum" in verdict.detail


def test_a_lone_recommendation_is_not_reported_as_the_deployed_threshold() -> None:
    # Adversarial QA: with nothing deployed, the recommendation was substituted for the current
    # threshold, so the report claimed "the 0.70 threshold in use" and compared it with itself.
    verdict = build_verdict(
        make_metrics(ece=0.05, mce=0.09),
        threshold=make_threshold_result(threshold=0.70, curve=()),
    )

    assert verdict.status == CLEAR
    assert verdict.headline == HEADLINE_NO_THRESHOLD
    assert "in use" not in verdict.detail


def test_miscalibrated_with_no_threshold_says_so_instead_of_claiming_a_minimum() -> None:
    verdict = build_verdict(make_metrics(ece=0.05, mce=0.09), threshold=None)

    assert verdict.status == CLEAR
    assert verdict.headline == HEADLINE_NO_THRESHOLD
    assert "no threshold was supplied to move" in verdict.detail


# --------------------------------------------------------------------------------------
# The duck-typed threshold input
# --------------------------------------------------------------------------------------


def test_impact_table_shape_is_read_defensively() -> None:
    impact = ImpactTable(
        rows=(
            ImpactRow(label="Auto rate", current="50.0%", recommended="72.0%", change="+22.0pp"),
        ),
        monthly_volume=20_000.0,
        current_threshold=0.60,
        recommended_threshold=0.40,
    )
    verdict = build_verdict(make_metrics(ece=0.05, mce=0.09), threshold=impact)

    assert verdict.status == TOO_HIGH
    assert verdict.detail == (
        f"Band {WORST_LABEL} measures {WORST_ACCURACY} accuracy on 200 decisions "
        f"(of {LABELS} labels); the threshold belongs at 0.40, below the 0.60 in use."
    )


def test_arbitrary_threshold_shaped_object_is_read_defensively() -> None:
    verdict = build_verdict(
        make_metrics(ece=0.05, mce=0.09),
        threshold=SimpleNamespace(current_threshold=0.5, recommended_threshold=0.7),
        current_threshold=None,
        recommended_threshold=None,
    )

    assert verdict.status == TOO_LOW

    # A curve of plain objects works for the sweep step too.
    with_curve = build_verdict(
        make_metrics(ece=0.05, mce=0.09),
        threshold=SimpleNamespace(
            current_threshold=0.5,
            recommended_threshold=0.55,
            curve=duck([0.5, 0.55, 0.6]).curve,
        ),
    )
    assert with_curve.status == CLEAR


def test_a_threshold_object_without_numbers_degrades_to_no_threshold_evidence() -> None:
    verdict = build_verdict(make_metrics(ece=0.05, mce=0.09), threshold=SimpleNamespace(rogue=1.0))

    assert verdict.status == CLEAR
    assert verdict.headline == HEADLINE_NO_THRESHOLD


def test_nan_and_boolean_thresholds_are_ignored() -> None:
    verdict = build_verdict(
        make_metrics(ece=0.05, mce=0.09),
        threshold=None,
        current_threshold=float("nan"),
        recommended_threshold=float("inf"),
    )

    assert verdict.status == CLEAR
    assert verdict.headline == HEADLINE_NO_THRESHOLD


# --------------------------------------------------------------------------------------
# Detail and stats discipline
# --------------------------------------------------------------------------------------


def branch_verdicts() -> list[Verdict]:
    return [
        build_verdict(make_metrics(n=99), threshold=make_threshold_result()),
        build_verdict(make_metrics(), threshold=None),
        build_verdict(
            make_metrics(ece=0.05, mce=0.09),
            threshold=make_threshold_result(threshold=0.70),
            current_threshold=0.50,
        ),
        build_verdict(
            make_metrics(ece=0.05, mce=0.09),
            threshold=make_threshold_result(threshold=0.40),
            current_threshold=0.60,
        ),
        build_verdict(
            make_metrics(ece=0.05, mce=0.09),
            threshold=make_threshold_result(threshold=0.70),
            current_threshold=0.70,
        ),
        build_verdict(make_metrics(ece=0.05, mce=0.09), threshold=None),
    ]


def test_every_detail_is_one_sentence_carrying_the_numbers() -> None:
    for verdict in branch_verdicts():
        assert verdict.detail.endswith("."), verdict.detail
        assert ". " not in verdict.detail, verdict.detail
        assert verdict.detail.count(".") >= 1
        if verdict.status == INSUFFICIENT_DATA:
            continue
        assert WORST_LABEL in verdict.detail
        assert WORST_ACCURACY in verdict.detail
        assert LABELS in verdict.detail


def test_stats_are_three_at_most_with_short_labels() -> None:
    verdict = build_verdict(
        make_metrics(ece=0.05, mce=0.09),
        threshold=make_threshold_result(accuracy_auto=0.912),
        current_threshold=0.50,
        recommended_threshold=0.70,
    )

    assert len(verdict.stats) == 3
    assert [stat.label for stat in verdict.stats] == [
        "Current threshold",
        "Measured accuracy",
        "Recommended threshold",
    ]
    assert [stat.value for stat in verdict.stats] == ["0.50", "91.2%", "0.70"]
    assert all(stat.sub and len(stat.sub) <= 24 for stat in verdict.stats)


def test_measured_accuracy_falls_back_to_the_observed_bins() -> None:
    verdict = build_verdict(make_metrics(), threshold=None)

    measured = verdict.stats[0]
    assert measured.label == "Measured accuracy"
    assert measured.value == "85.6%"  # (0.68 * 200 + 0.90 * 800) / 1000
    assert measured.sub == f"of {LABELS} labels"


def test_stats_never_exceed_three_entries() -> None:
    for verdict in branch_verdicts():
        assert len(verdict.stats) <= 3


# --------------------------------------------------------------------------------------
# markdown_summary
# --------------------------------------------------------------------------------------


def impact_table() -> ImpactTable:
    return ImpactTable(
        rows=(
            ImpactRow(
                label="Auto rate", current="50.0%", recommended="72.0%", change="+22.0 points"
            ),
            ImpactRow(label="Cost per case", current="0.52", recommended="0.42", change="-0.10"),
        ),
        monthly_volume=20_000.0,
        current_threshold=0.50,
        recommended_threshold=0.70,
    )


def model_with_impact() -> ReportModel:
    verdict = build_verdict(
        make_metrics(ece=0.05, mce=0.09),
        threshold=make_threshold_result(threshold=0.70),
        current_threshold=0.50,
    )
    return ReportModel(verdict=verdict, impact=impact_table())


def test_markdown_summary_leads_with_the_verdict_as_one_quoted_paragraph() -> None:
    model = model_with_impact()
    text = markdown_summary(model)

    first_line, rest = text.split("\n", 1)
    assert first_line == f"> **{model.verdict.headline}** {model.verdict.detail}"
    assert rest.startswith("\n")
    assert text.endswith("\n")
    assert not text.endswith("\n\n")
    assert "\x1b" not in text  # no ANSI
    assert "<" not in text  # no HTML


def test_markdown_summary_includes_the_stat_line_and_the_impact_table() -> None:
    text = markdown_summary(model_with_impact())

    assert "Current threshold 0.50 | Measured accuracy 90.0% | Recommended threshold 0.70" in text
    assert "| Change | Now | Recommended | Delta |" in text
    assert "| --- | --- | --- | --- |" in text
    assert "| Auto rate | 50.0% | 72.0% | +22.0 points |" in text
    assert "| Cost per case | 0.52 | 0.42 | -0.10 |" in text


def test_markdown_summary_works_from_a_bare_verdict_without_an_impact_table() -> None:
    verdict = build_verdict(make_metrics(), threshold=None)
    text = markdown_summary(verdict)

    assert text.startswith(f"> **{HEADLINE_CLEAR}**")
    assert "| Change |" not in text
    assert text.endswith("\n")


def test_markdown_summary_escapes_pipes_and_newlines_in_cells() -> None:
    verdict = build_verdict(make_metrics(), threshold=None)
    model = ReportModel(
        verdict=verdict,
        impact=ImpactTable(
            rows=(
                ImpactRow(
                    label="Auto | manual",
                    current="50.0%\nof cases",
                    recommended="72.0%",
                    change="+22.0",
                ),
            )
        ),
    )
    text = markdown_summary(model)

    assert "| Auto \\| manual | 50.0% of cases | 72.0% | +22.0 |" in text
    # Exactly one cell needed escaping, and no cell leaked a raw delimiter.
    assert text.count("\\|") == 1


def test_markdown_summary_is_stable_across_calls() -> None:
    model = model_with_impact()
    assert markdown_summary(model) == markdown_summary(model)


def test_markdown_summary_accepts_any_object_carrying_a_verdict() -> None:
    verdict = build_verdict(make_metrics(), threshold=None)
    text = markdown_summary(SimpleNamespace(verdict=verdict, impact=None))

    assert text.startswith("> **")
    assert HEADLINE_CLEAR in text


def test_markdown_summary_rejects_something_that_is_not_a_report() -> None:
    with pytest.raises(TypeError):
        markdown_summary(SimpleNamespace(status="clear"))
    with pytest.raises(TypeError):
        markdown_summary("Your threshold is too low.")
