"""Regressions for the loop that closed the remaining QA findings.

One test per finding, named after the behaviour it protects. These are the lower-severity half of
the QA report: observability, degenerate inputs, and claims the report made without evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from jeval import collect
from jeval.cli import app
from jeval.collect import response_payloads, track
from jeval.costs import CostAction, sweep, sweep_by_segment
from jeval.planning import ASSUMPTION, plan_labels
from jeval.presets import JEV_NATIVE, skip_reason
from jeval.report.svg import escape
from jeval.store import read_records
from jeval.synth import SynthSpec, generate

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(collect.ENV_FLAG, raising=False)
    monkeypatch.delenv(collect.ENV_ROOT, raising=False)
    collect.reset_stats()


def _demo(tmp_path: Path) -> None:
    runner.invoke(app, ["demo", "--out-dir", str(tmp_path)])


# --- rendering: a value cannot lie about what it says -------------------------------------------


def test_invisible_direction_controls_are_made_visible() -> None:
    """QA: U+202E survived raw in the report, so a label could reverse the text a reader sees."""
    rendered = escape("billing\u202etxen")

    assert "\u202e" not in rendered  # the control itself is gone
    assert "&lt;U+202E&gt;" in rendered  # and the reader can see that one was there
    assert escape("<script>") == "&lt;script&gt;"  # escaping still holds


def test_the_report_says_when_it_was_generated(tmp_path: Path) -> None:
    """QA: 'Generated · source:' printed with no timestamp at all."""
    _demo(tmp_path)
    runner.invoke(
        app,
        [
            "report",
            "--root",
            str(tmp_path),
            "--costs",
            str(tmp_path / "costs.yaml"),
            "--current",
            "0.6",
            "-o",
            str(tmp_path / "r.html"),
        ],
    )
    html = (tmp_path / "r.html").read_text(encoding="utf-8")

    assert "Generated 20" in html  # a date, not an empty field


def test_impossible_report_inputs_are_refused(tmp_path: Path) -> None:
    """QA: `--monthly -5` rendered negative costs and `--current 1.5` was quoted as truth."""
    _demo(tmp_path)

    monthly = runner.invoke(app, ["report", "--root", str(tmp_path), "--monthly", "-5"])
    current = runner.invoke(app, ["report", "--root", str(tmp_path), "--current", "1.5"])

    assert monthly.exit_code == 1
    assert "cannot be negative" in monthly.stdout
    assert current.exit_code == 1
    assert "between 0 and 1" in current.stdout


# --- numbers and messages the tool owes the reader ----------------------------------------------


def test_a_flat_cost_curve_reports_no_interval_and_says_so() -> None:
    """QA: an all-zero cost matrix reported threshold 1.00 with 'ci [1.0, 1.0]' and
    insufficient_data false — a recommendation the data cannot separate."""
    records = [
        r
        for r in generate(SynthSpec(n=300, mode="calibrated", question_key="intent", seed=5))
        if r.is_labeled
    ]
    action = CostAction("free", "intent", records[0].prediction, 0.0, 0.0, 0.0)

    result = sweep(action, records, steps=11, n_boot=20)

    assert result.flat_region is not None
    assert result.flat_region[0] <= 0.0 and result.flat_region[1] >= 1.0
    assert result.ci_low != result.ci_low  # no interval, rather than [1.0, 1.0]


def test_the_plan_reports_the_scaling_exponent_it_fitted() -> None:
    """QA: the k/sqrt(n) assumption drifted ~17% over the range it projected across."""
    records = generate(SynthSpec(n=1200, mode="inflated", question_key="department", seed=4))

    plan = plan_labels(records, target_ci=0.05)[0]

    assert plan.exponent is not None and 0.2 <= plan.exponent <= 0.9
    assert plan.residual is not None and plan.residual < 0.25  # the law held where it was fitted
    from jeval.planning import _assumption_note

    assert "n**a" in ASSUMPTION
    assert "slower than" in _assumption_note(plan)


def test_a_sub_floor_segment_request_is_clamped_not_refused() -> None:
    """QA: min_records=0 raised, contradicting the docstring's own promise."""
    records = [
        r
        for r in generate(SynthSpec(n=400, mode="calibrated", question_key="intent", seed=5))
        if r.is_labeled
    ]
    action = CostAction("a", "intent", records[0].prediction, 10.0, 5.0, 0.0)

    assert sweep_by_segment(action, records, segment_key="lang", min_records=0) == (
        sweep_by_segment(action, records, segment_key="lang", min_records=1)
    )


# --- collector: what the counters say happened --------------------------------------------------


def test_a_token_count_that_arrives_as_text_is_kept() -> None:
    """QA: usage.input_tokens '312' made state_tokens vanish with no counter."""
    payloads = response_payloads(
        {
            "model": "jev-1.13.0",
            "answers": {"q": {"type": "choice", "choice": "a", "probabilities": {"a": 1.0}}},
            "usage": {"input_tokens": "312"},
        }
    )

    assert payloads[0]["state_tokens"] == 312


def test_a_negative_or_boolean_number_is_refused_by_record(tmp_path: Path) -> None:
    """QA: negative latency and `confidence=True` were accepted and corrupted aggregates."""
    target = tmp_path / "records.jsonl"

    assert (
        collect.record(question_key="q", prediction="a", confidence=0.9, latency_ms=-5, path=target)
        is False
    )
    assert collect.record(question_key="q", prediction="a", confidence=True, path=target) is False
    assert collect.stats()["invalid_value"] == 2
    assert not target.exists()


def test_a_sink_that_is_not_a_file_is_not_counted_as_written() -> None:
    """QA: /dev/null reported written=1 for bytes that were discarded."""
    written = collect.record(question_key="q", prediction="a", confidence=0.9, path="/dev/null")

    assert written is False
    assert collect.stats()["sink_not_a_file"] == 1
    assert collect.stats()["written"] == 0


def test_a_raising_call_is_still_counted_as_an_attempt(tmp_path: Path) -> None:
    """QA: a call that raised incremented no counter, so stats() could not tell it from 'never
    called'."""

    class Boom:
        def system_one(self, **kwargs: object) -> dict[str, object]:
            raise ValueError("upstream 500")

    client = track(Boom(), path=tmp_path / "records.jsonl")
    with pytest.raises(ValueError):
        client.system_one(state="s")

    assert collect.stats()["calls"] == 1


# --- messages that name the real cause ----------------------------------------------------------


def test_each_preset_skip_cause_gets_its_own_message() -> None:
    """QA: six distinct causes shared one message that was false for five of them."""
    reasons = {
        skip_reason(JEV_NATIVE, {}),
        skip_reason(JEV_NATIVE, {"response": {"model": "m"}}),
        skip_reason(JEV_NATIVE, {"response": {"answers": []}}),
        skip_reason(JEV_NATIVE, {"response": {"answers": {}}}),
        skip_reason(JEV_NATIVE, {"response": {"answers": {"q": {"type": "choice"}}}}),
    }

    assert len(reasons) == 5
    assert any("is a list" in reason for reason in reasons)


def test_an_axis_the_log_does_not_carry_is_named_not_invented(tmp_path: Path) -> None:
    """QA: `--by nope` rendered a segment called 'nope = unknown', inventing a breakdown."""
    _demo(tmp_path)

    result = runner.invoke(
        app, ["report", "--root", str(tmp_path), "--by", "nope", "-o", str(tmp_path / "r.html")]
    )

    assert result.exit_code == 0
    assert "no record carries segment 'nope'" in result.stdout
    assert "nope = unknown" not in (tmp_path / "r.html").read_text(encoding="utf-8")


def test_init_over_a_file_named_jeval_is_a_message(tmp_path: Path) -> None:
    """QA: a plain file named .jeval produced a FileExistsError traceback."""
    (tmp_path / ".jeval").write_text("not a directory", encoding="utf-8")

    result = runner.invoke(app, ["init", "--root", str(tmp_path)])

    assert result.exit_code == 1
    assert "is not a directory" in result.stdout
    assert "Traceback" not in result.stdout


def test_label_on_a_project_with_no_records_says_so(tmp_path: Path) -> None:
    """QA: it claimed 'every record already has a label' with no records at all."""
    runner.invoke(app, ["init", "--root", str(tmp_path)])

    result = runner.invoke(app, ["label", "--root", str(tmp_path)])
    message = result.stdout + getattr(result, "stderr", "")

    assert "ingest" in message.lower()
    assert "already has a label" not in message


def test_a_reused_client_reports_the_configuration_it_dropped(tmp_path: Path) -> None:
    """QA: a second track() with a path silently ignored it, so the log looked missing."""

    class Client:
        def system_one(self, **kwargs: object) -> dict[str, object]:
            return {"model": "jev-1.13.0", "answers": {}}

    client = Client()
    track(client, path=tmp_path / "a.jsonl")
    track(client, path=tmp_path / "b.jsonl")

    assert collect.stats()["retrack_ignored"] == 1


def test_unparseable_log_lines_still_name_their_file_and_line(tmp_path: Path) -> None:
    """The granular skip work must not have cost the file:line in the ingest errors."""
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"request_id": "r1"}) + "\n", encoding="utf-8")

    result = runner.invoke(
        app, ["ingest", "--preset", "jev-native", str(bad), "--root", str(tmp_path)]
    )

    assert "bad.jsonl row 1" in result.stdout
    assert read_records(tmp_path / ".jeval" / "records.jsonl") == []
