"""Paired head-to-head comparison of two model versions on the same requests.

The synthetic log is built with ``synth.generate_paired``: both versions answer the same requests,
share a ``source_key`` per request and share the probability each request is answered correctly,
so every difference the tests expect is one the generator put there on purpose.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from jeval.calibration import mcnemar_exact, paired_bootstrap
from jeval.cli import app
from jeval.drift import MIN_PAIRS, compare_paired, format_paired_lines
from jeval.schema import DecisionRecord
from jeval.store import records_path, write_records
from jeval.synth import SynthSpec, accuracy_of, generate, generate_paired

BASE = SynthSpec(n=1500, mode="calibrated", seed=21, model="jev-1.13.0")


def _paired(current: SynthSpec, *, current_skill: float = 1.0) -> list[DecisionRecord]:
    return generate_paired(BASE, current, current_skill=current_skill)


def _only(records: list[DecisionRecord], **kwargs: object):  # type: ignore[no-untyped-def]
    view = compare_paired(records, n_boot=400, **kwargs)  # type: ignore[arg-type]
    assert len(view.questions) == 1, view
    return view, view.questions[0]


# --- McNemar -------------------------------------------------------------------------------------


def test_mcnemar_exact_known_values() -> None:
    assert mcnemar_exact(0, 10) == pytest.approx(2 / 1024)
    assert mcnemar_exact(0, 10) == pytest.approx(0.001953125)
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 0) == 1.0
    # Two-sided: 2 * P(X <= 2 | n=12, p=0.5) = 2 * 79 / 4096.
    assert mcnemar_exact(2, 10) == pytest.approx(2 * 79 / 4096)


def test_mcnemar_exact_is_symmetric_and_rejects_negative_counts() -> None:
    assert mcnemar_exact(3, 17) == mcnemar_exact(17, 3)
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 4)


def test_paired_bootstrap_of_a_log_against_itself_is_exactly_zero() -> None:
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, size=200)
    hit = rng.random(200) < conf
    edges = [0.5, 0.75, 1.0]
    result = paired_bootstrap(conf, hit, conf, hit, edges, n_boot=100, seed=1)
    assert result.n == 200
    assert result.accuracy.difference == 0.0
    assert result.ece.difference == 0.0
    assert result.brier.difference == 0.0
    assert (result.discordant_baseline_only, result.discordant_current_only) == (0, 0)
    assert result.mcnemar_p == 1.0


# --- synthetic restoration -----------------------------------------------------------------------


def test_generate_paired_shares_requests_and_leaves_generate_untouched() -> None:
    records = _paired(SynthSpec(n=1500, mode="calibrated", seed=22, model="jev-1.14.0"))
    baseline = [r for r in records if r.model == "jev-1.13.0"]
    current = [r for r in records if r.model == "jev-1.14.0"]
    assert len(baseline) == len(current) == 1500
    assert [r.source_key for r in baseline] == [r.source_key for r in current]
    assert len({r.source_key for r in baseline}) == 1500
    # The plain generator is not affected by the paired one existing.
    plain = generate(BASE)
    assert all(record.source_key is None for record in plain)


def test_identical_versions_show_no_difference() -> None:
    records = _paired(SynthSpec(n=1500, mode="calibrated", seed=22, model="jev-1.14.0"))
    _, question = _only(records)
    result = question.comparison
    assert result is not None
    assert question.n_pairs == 1500
    assert result.accuracy.ci_low <= 0.0 <= result.accuracy.ci_high, result.accuracy
    assert result.ece.ci_low <= 0.0 <= result.ece.ci_high, result.ece
    assert result.mcnemar_p > 0.05, result.mcnemar_p
    assert "no accuracy difference the sample can resolve" in question.verdict
    assert "no calibration difference the sample can resolve" in question.verdict


def test_inflated_current_version_is_worse_calibrated_beyond_noise() -> None:
    records = _paired(
        SynthSpec(n=1500, mode="inflated", inflation=1.15, seed=22, model="jev-1.14.0")
    )
    _, question = _only(records)
    result = question.comparison
    assert result is not None
    # Current minus baseline: an inflated current version has the larger ECE.
    assert result.ece.difference > 0.0
    assert result.ece.ci_low > 0.0, result.ece
    assert "worse calibrated beyond noise" in question.verdict


def test_less_accurate_current_version_is_caught_by_mcnemar() -> None:
    current = SynthSpec(n=1500, mode="calibrated", seed=22, model="jev-1.14.0")
    records = _paired(current, current_skill=0.6)
    before = accuracy_of([r for r in records if r.model == "jev-1.13.0"])
    after = accuracy_of([r for r in records if r.model == "jev-1.14.0"])
    assert after < before  # the generator really made it worse
    _, question = _only(records)
    result = question.comparison
    assert result is not None
    assert result.accuracy.difference < 0.0
    assert result.accuracy.ci_high < 0.0, result.accuracy
    assert result.mcnemar_p < 0.05, result.mcnemar_p
    assert result.discordant_baseline_only > result.discordant_current_only
    assert "current is less accurate beyond noise" in question.verdict


# --- pairing rules -------------------------------------------------------------------------------


def test_pairing_refuses_below_the_minimum_and_reports_no_numbers() -> None:
    small = SynthSpec(n=MIN_PAIRS - 1, mode="calibrated", seed=21, model="jev-1.13.0")
    records = generate_paired(
        small, SynthSpec(n=MIN_PAIRS - 1, mode="inflated", seed=22, model="jev-1.14.0")
    )
    _, question = _only(records)
    assert question.n_pairs == MIN_PAIRS - 1
    assert question.comparison is None
    assert f"fewer than the {MIN_PAIRS}" in question.refusal
    text = "\n".join(format_paired_lines(compare_paired(records)))
    assert "accuracy" not in text
    assert "refused" in text


def test_pairing_counts_every_record_it_cannot_use() -> None:
    records = _paired(SynthSpec(n=1500, mode="calibrated", seed=22, model="jev-1.14.0"))
    baseline = [r for r in records if r.model == "jev-1.13.0"]
    current = [r for r in records if r.model == "jev-1.14.0"]
    # 10 baseline records lose their key: unpairable, and their partners are left one-sided.
    for record in baseline[:10]:
        record.source_key = None
    # Request 10 is logged twice by the current model: that key is refused on that side, and its
    # baseline partner is left one-sided.
    duplicate = current[10].model_copy()
    # 5 pairs lose the gold label on the current side only.
    for record in current[20:25]:
        record.label = None
        record.label_source = None
    # 3 pairs where the two sides were labeled with different answers.
    for record in current[30:33]:
        record.label = "__other_answer__"
    # One score record, which is never folded into binary accuracy.
    score = generate(
        SynthSpec(n=1, question_type="score", question_key="csat", mode="calibrated", seed=3)
    )[0]
    score = score.model_copy(update={"model": "jev-1.13.0", "source_key": "req-00000"})
    view = compare_paired([*baseline, *current, duplicate, score], n_boot=50)
    assert view.n_unkeyed == 10
    assert view.n_duplicate == 2  # both current records at the duplicated key
    assert view.n_one_sided == 10 + 1  # partners of the unkeyed records, and of the duplicate
    assert view.n_not_gold_on_both == 5
    assert view.n_label_conflict == 3
    assert view.n_score == 1
    (question,) = view.questions
    assert question.n_pairs == 1500 - 10 - 1 - 5 - 3
    text = "\n".join(format_paired_lines(view))
    assert "10 without source_key" in text
    assert "3 with conflicting labels" in text


def test_pairing_needs_two_models() -> None:
    view = compare_paired(generate(BASE))
    assert view.questions == ()
    assert "two model versions" in view.note


# --- CLI -----------------------------------------------------------------------------------------


def _cli_root(tmp_path: Path) -> Path:
    records = generate_paired(
        SynthSpec(n=400, mode="calibrated", seed=21, model="jev-1.13.0"),
        SynthSpec(n=400, mode="calibrated", seed=22, model="jev-1.14.0"),
    )
    write_records(records, records_path(tmp_path))
    return tmp_path


def test_drift_paired_prints_the_head_to_head_block(tmp_path: Path) -> None:
    root = _cli_root(tmp_path)
    result = CliRunner().invoke(app, ["drift", "--root", str(root), "--paired"])
    assert result.exit_code == 0, result.output
    assert "paired: jev-1.13.0 -> jev-1.14.0" in result.output
    assert "department: 400 pairs" in result.output
    assert "McNemar p=" in result.output
    assert "verdict:" in result.output
    assert result.output.rstrip().endswith("exit 0")


def test_drift_without_paired_is_unchanged(tmp_path: Path) -> None:
    root = _cli_root(tmp_path)
    result = CliRunner().invoke(app, ["drift", "--root", str(root)])
    assert result.exit_code == 0, result.output
    assert "paired" not in result.output
    assert "McNemar" not in result.output


def test_drift_paired_refuses_a_period_split(tmp_path: Path) -> None:
    root = _cli_root(tmp_path)
    result = CliRunner().invoke(app, ["drift", "--root", str(root), "--paired", "--by-period", "W"])
    assert result.exit_code != 0
