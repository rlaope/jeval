"""Regressions for the defects adversarial QA found.

Each test names one finding and fails on the behaviour that produced it. One defect per test, so a
future change that reintroduces one of them points at exactly what broke.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from jeval import collect, presets
from jeval.calibration import compute_calibration, diagnose
from jeval.cli import app
from jeval.collect import track
from jeval.config import IngestMap
from jeval.costs import CostAction
from jeval.ingest import harvest_file, ingest_files
from jeval.planning import plan_labels
from jeval.schema import DecisionRecord, normalize_record
from jeval.score import measure_score
from jeval.store import read_records

runner = CliRunner()

JEV_MODEL = "jev-1.13.0"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(collect.ENV_FLAG, raising=False)
    monkeypatch.delenv(collect.ENV_ROOT, raising=False)
    monkeypatch.delenv(collect.ENV_MODEL, raising=False)
    collect.reset_stats()


def _record(**overrides: object) -> DecisionRecord:
    payload: dict[str, object] = {
        "model": JEV_MODEL,
        "question_key": "department",
        "question_type": "choice",
        "prediction": "billing",
        "confidence": 0.9,
        "probabilities": {"billing": 0.9, "technical": 0.1},
    }
    payload.update(overrides)
    return DecisionRecord.model_validate(payload)


# --- collector: the switches and the failure modes -------------------------------------------


def test_the_off_switch_beats_an_explicit_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """QA: JEVAL_COLLECT=0 still wrote records when a call site passed path=."""
    monkeypatch.setenv(collect.ENV_FLAG, "0")
    target = tmp_path / "records.jsonl"

    assert collect.record(question_key="q", prediction="yes", confidence=0.9, path=target) is False
    assert not target.exists()


def test_a_flag_value_that_is_not_a_path_does_not_become_a_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """QA: JEVAL_COLLECT=enabled created ./enabled and sent the user's records there."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(collect.ENV_FLAG, "enabled")
    monkeypatch.setenv(collect.ENV_ROOT, str(tmp_path / "project"))

    assert collect.target_path() == tmp_path / "project" / ".jeval" / "records.jsonl"
    collect.record(question_key="q", prediction="yes", confidence=0.9)
    assert not (tmp_path / "enabled").exists()
    assert collect.stats()["unknown_flag_value"] >= 1


def test_a_response_without_a_model_still_records(tmp_path: Path) -> None:
    """QA: a model-less response lost every decision silently (ValidationError, then dropped)."""

    class Client:
        def system_one(self, **kwargs: object) -> dict[str, object]:
            return {
                "answers": {
                    "department": {
                        "type": "choice",
                        "choice": "billing",
                        "probabilities": {"billing": 0.9, "x": 0.1},
                    }
                }
            }

    target = tmp_path / "records.jsonl"
    track(Client(), path=target).system_one(state="s")

    stored = read_records(target)
    assert len(stored) == 1
    assert stored[0].model  # never empty: a record without a model cannot be validated


def test_tracking_a_client_that_refuses_attributes_does_not_raise(tmp_path: Path) -> None:
    """QA: slots / frozen dataclass / pydantic clients made track() raise out of the module."""

    class Slotted:
        __slots__ = ()

        def system_one(self, **kwargs: object) -> dict[str, object]:
            return {"model": JEV_MODEL, "answers": {}}

    client = Slotted()

    assert track(client, path=tmp_path / "records.jsonl") is client
    assert collect.stats()["install_failed"] == 1


# --- collector: the numbers a record is allowed to claim --------------------------------------


def test_confidence_is_the_probability_of_the_stored_prediction() -> None:
    """QA: a record could claim 0.85 confidence for a class its own map gave 0.08."""
    built = normalize_record(
        {
            "model": JEV_MODEL,
            "question_key": "department",
            "prediction": "billing",
            "probabilities": {"technical": 0.85, "billing": 0.08, "sales": 0.07},
        }
    )

    assert built.confidence == pytest.approx(0.08)


def test_a_prediction_its_own_map_cannot_produce_is_refused() -> None:
    """The same contradiction where the winner is absent from the map entirely."""
    with pytest.raises(ValueError, match="not in its own probabilities map"):
        normalize_record(
            {
                "model": JEV_MODEL,
                "question_key": "department",
                "prediction": "legal",
                "probabilities": {"technical": 0.85, "billing": 0.15},
            }
        )


def test_a_noul_probability_is_stored_clamped_and_refused_when_not_finite() -> None:
    """QA: 1.5 was stored verbatim, and NaN became a maximally confident answer."""
    built = normalize_record(
        {"model": JEV_MODEL, "question_key": "is_urgent", "question_type": "noul", "noul": 1.5}
        | {"probability_positive": 1.5}
    )
    assert built.probabilities == {"yes": 1.0, "no": 0.0}

    with pytest.raises(ValueError, match="must be finite"):
        normalize_record(
            {
                "model": JEV_MODEL,
                "question_key": "is_urgent",
                "question_type": "noul",
                "probability_positive": float("nan"),
            }
        )


# --- ingest: identity, bytes and label population ---------------------------------------------


def test_two_questions_of_one_request_get_distinct_ids(tmp_path: Path) -> None:
    """QA: shared ids let one harvested answer land on every question of a request."""
    raw = tmp_path / "log.jsonl"
    raw.write_text(
        json.dumps(
            {
                "id": "req_1",
                "model": JEV_MODEL,
                "questions": [
                    {
                        "question_key": "department",
                        "question_type": "choice",
                        "prediction": "billing",
                        "confidence": 0.9,
                        "probabilities": {"billing": 0.9, "x": 0.1},
                    },
                    {
                        "question_key": "intent",
                        "question_type": "choice",
                        "prediction": "refund",
                        "confidence": 0.8,
                        "probabilities": {"refund": 0.8, "other": 0.2},
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "records.jsonl"
    ingest_files([raw], out, IngestMap())

    stored = read_records(out)
    assert len({record.id for record in stored}) == 2, "a request's questions shared one id"


def test_the_label_rewrite_preserves_crlf_line_endings(tmp_path: Path) -> None:
    """QA: harvesting one label silently rewrote every CRLF in the user's file."""
    target = tmp_path / "records.jsonl"
    rows = [
        {
            "id": "a",
            "model": JEV_MODEL,
            "question_key": "q",
            "question_type": "choice",
            "prediction": "yes",
            "confidence": 0.9,
            "probabilities": {"yes": 0.9, "no": 0.1},
            "label": None,
            "source_key": "t-1",
        },
        {
            "id": "b",
            "model": JEV_MODEL,
            "question_key": "q",
            "question_type": "choice",
            "prediction": "yes",
            "confidence": 0.9,
            "probabilities": {"yes": 0.9, "no": 0.1},
            "label": None,
            "source_key": "t-2",
        },
    ]
    target.write_bytes(("\r\n".join(json.dumps(row) for row in rows) + "\r\n").encode("utf-8"))
    before = target.read_bytes()

    report = harvest_file(
        target,
        [{"ticket_id": "t-1", "answer": "yes"}],
        field="answer",
        source="human_review",
        join_on="ticket_id",
    )

    assert report.n_applied == 1
    after = target.read_bytes()
    assert after.count(b"\r\n") == 2, "line endings were rewritten for the whole file"
    # the untouched line keeps every byte it had
    assert after.split(b"\r\n")[1] == before.split(b"\r\n")[1]


def test_a_label_with_no_source_is_not_gold(tmp_path: Path) -> None:
    """QA: evaluate() counted source-less labels as gold while costs and drift excluded them."""
    record = _record(label="billing")

    assert record.label_source is None
    assert record.is_gold is False  # one definition of gold, shared by every consumer


# --- measurement: the numbers that must not lie ------------------------------------------------


def test_the_ece_interval_contains_its_own_estimate() -> None:
    """QA: the reported CI excluded the ECE printed beside it, and excluded zero when calibrated."""
    confidences = np.array([0.56 + 0.02 * index for index in range(20)])
    correct = np.zeros(20, dtype=bool)
    correct[:15] = True

    metrics = compute_calibration(confidences, correct, alpha=0.05, n_boot=500)

    assert metrics.ece == pytest.approx(0.0, abs=1e-9)
    assert metrics.ece_ci_low <= metrics.ece <= metrics.ece_ci_high
    assert metrics.ece_ci_low == pytest.approx(0.0, abs=1e-6)


def test_a_direction_is_only_claimed_when_the_bin_interval_excludes_it() -> None:
    """QA: a calibrated log was called over/under-confident from a bin whose own CI contained the
    claimed confidence."""
    from jeval.synth import SynthSpec, generate

    for seed in (3, 7, 11):
        records = generate(
            SynthSpec(n=400, mode="calibrated", question_key="department", seed=seed)
        )
        points = [point for point in (record.calibration_point() for record in records) if point]
        metrics = compute_calibration([p[0] for p in points], [p[1] for p in points], n_bins=10)
        reading = diagnose(metrics)
        assert "overconfidence" not in reading and "underconfidence" not in reading, reading


def test_an_extreme_projection_target_does_not_crash() -> None:
    """QA: `jeval plan --target-ci 1e-160` raised OverflowError out of the command."""
    plans = plan_labels([_record(label="billing") for _ in range(300)], target_ci=1e-160)

    assert plans
    assert plans[0].labels_for_target is None  # unprojectable, and it says so
    assert plans[0].reason


def test_score_metrics_never_report_a_non_finite_result() -> None:
    """QA: two finite predictions overflowed the reduction and the report printed 'MAE inf'."""
    records = [
        _record(
            question_type="score",
            prediction="1e308",
            confidence=0.5,
            label="-1e308",
            probabilities=None,
        ),
        _record(
            question_type="score",
            prediction="1e308",
            confidence=0.5,
            label="1e308",
            probabilities=None,
        ),
    ]
    metrics = measure_score(records)

    assert metrics.mae != metrics.mae or np.isfinite(metrics.mae)
    assert metrics.rmse != metrics.rmse or np.isfinite(metrics.rmse)


def test_a_cost_the_arithmetic_cannot_carry_is_refused() -> None:
    """QA: a NaN cost won every comparison and was reported as a measured threshold."""
    with pytest.raises(ValueError, match="finite"):
        CostAction("a", "intent", "yes", float("nan"), 1.0, 0.0)
    with pytest.raises(ValueError, match="negative"):
        CostAction("a", "intent", "yes", -1.0, 1.0, 0.0)


# --- report and CLI: no invented facts ---------------------------------------------------------


def _demo_with_two_actions(tmp_path: Path) -> None:
    """A project carrying a cost matrix, so the report has more than one action to render."""
    runner.invoke(app, ["demo", "--out-dir", str(tmp_path)])


def test_no_deployed_threshold_is_never_invented(tmp_path: Path) -> None:
    """QA: with no thresholds.yaml the report claimed "the 0.99 threshold in use"."""
    _demo_with_two_actions(tmp_path)
    without = tmp_path / "no-current.html"
    runner.invoke(
        app,
        [
            "report",
            "--root",
            str(tmp_path),
            "--costs",
            str(tmp_path / "costs.yaml"),
            "-o",
            str(without),
        ],
    )
    html = without.read_text(encoding="utf-8")

    assert "threshold in use" not in html
    assert "threshold-slider" not in html  # no baseline, so no impact table to drive

    with_current = tmp_path / "with-current.html"
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
            str(with_current),
        ],
    )
    assert "threshold-slider" in with_current.read_text(encoding="utf-8")


def test_each_action_gets_its_own_impact_table_and_control_ids(tmp_path: Path) -> None:
    """QA: every action's cost section rendered the first action's numbers, and three sections
    shared one DOM id."""
    import re

    _demo_with_two_actions(tmp_path)
    runner.invoke(
        app,
        [
            "report",
            "--root",
            str(tmp_path),
            "--costs",
            str(tmp_path / "costs.yaml"),
            "--current",
            "0.6",  # a deployed threshold is required: nothing is fabricated without one
            "--monthly",
            "5000",  # also renders each action's volume control, so ids are checked in full
            "-o",
            str(tmp_path / "r.html"),
        ],
    )
    html = (tmp_path / "r.html").read_text(encoding="utf-8")
    identifiers = re.findall(r'id="(threshold-slider[^"]*)"', html)

    assert len(identifiers) == len(set(identifiers)) >= 6
    assert (
        len(
            {
                identifier.split("threshold-slider-")[-1].removesuffix("-volume")
                for identifier in identifiers
            }
        )
        == 3
    )


def test_a_hostile_drift_note_cannot_inject_script(tmp_path: Path) -> None:
    """QA: the drift note and failure detail were the only unescaped sinks in the report."""
    from jeval.report.charts.drift import render_drift_section
    from jeval.report.model import DriftSlice, DriftView

    view = DriftView(
        baseline_label="<script>window.__pwned=1</script>",
        current_label="new",
        slices=(
            DriftSlice(label="q", model="m", start="", end="", n=40, ece=0.1),
            DriftSlice(label="q", model="n", start="", end="", n=40, ece=0.2),
        ),
        note="<script>window.__pwned=2</script>",
    )
    html = render_drift_section(view)

    assert "<script>window.__pwned" not in html
    assert "&lt;script&gt;" in html


def test_the_embedded_payload_cannot_swallow_the_report_script() -> None:
    """QA: a name containing `<!--<script` left an open comment that killed every control."""
    from jeval.report import svg

    embedded = svg.embed_json('{"name": "<!--<script>alert(1)</script>"}', element_id="d")

    assert "<script" not in embedded.split(">", 1)[1] or embedded.count("<script") == 1
    assert "\\u003c" in embedded
    assert json.loads(embedded.split(">", 1)[1].rsplit("</script>", 1)[0])["name"].startswith(
        "<!--"
    )


def test_one_bin_is_refused_because_it_cannot_support_the_claim(tmp_path: Path) -> None:
    """QA: --bins 1 reported 'confidence is trustworthy' where ten bins reported the opposite."""
    _demo_with_two_actions(tmp_path)

    result = runner.invoke(app, ["report", "--root", str(tmp_path), "--bins", "1"])

    assert result.exit_code == 1
    assert "cannot support a calibration claim" in result.stdout


def test_a_malformed_log_is_a_message_not_a_traceback(tmp_path: Path) -> None:
    """QA: one bad line surfaced as a rich traceback panel."""
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"unterminated\n', encoding="utf-8")

    result = runner.invoke(app, ["ingest", str(bad), "--root", str(tmp_path)])

    assert result.exit_code == 1
    assert "invalid JSON" in result.stdout
    assert "Traceback" not in result.stdout
    assert "bad.jsonl:1" in result.stdout


def test_ingesting_files_and_harvesting_labels_does_both(tmp_path: Path) -> None:
    """QA: `jeval ingest log.jsonl --labels res.jsonl` (the README's own example) ingested
    nothing and said nothing."""
    log = tmp_path / "log.jsonl"
    log.write_text(
        json.dumps(
            {
                "id": "req_1",
                "model": JEV_MODEL,
                "question_key": "department",
                "question_type": "choice",
                "prediction": "billing",
                "confidence": 0.9,
                "probabilities": {"billing": 0.9, "x": 0.1},
                "source_key": "t-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    labels = tmp_path / "res.jsonl"
    labels.write_text(
        json.dumps({"ticket_id": "t-1", "answer": "billing"}) + "\n", encoding="utf-8"
    )

    result = runner.invoke(
        app,
        [
            "ingest",
            str(log),
            "--root",
            str(tmp_path),
            "--labels",
            str(labels),
            "--label-field",
            "answer",
            "--label-source",
            "human_review",
            "--join-on",
            "ticket_id",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "wrote 1 records" in result.stdout
    assert "applied 1 labels" in result.stdout
    assert read_records(tmp_path / ".jeval" / "records.jsonl")[0].label == "billing"


def test_preset_and_labels_together_are_refused_rather_than_half_run(tmp_path: Path) -> None:
    """QA: the harvest was silently skipped (exit 0) when --preset came first."""
    log = tmp_path / "native.jsonl"
    log.write_text(
        json.dumps(
            {
                "request_id": "r1",
                "response": {
                    "model": JEV_MODEL,
                    "answers": {
                        "department": {
                            "type": "choice",
                            "choice": "billing",
                            "probabilities": {"billing": 0.9, "x": 0.1},
                        }
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    labels = tmp_path / "res.jsonl"
    labels.write_text(json.dumps({"ticket_id": "r1", "answer": "billing"}) + "\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "ingest",
            str(log),
            "--preset",
            "jev-native",
            "--labels",
            str(labels),
            "--root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "two commands" in result.stdout


def test_a_preset_answer_carrying_a_label_is_kept(tmp_path: Path) -> None:
    """QA: an inline label inside a native answer object was dropped with no counter."""
    row = {
        "request_id": "r1",
        "response": {
            "model": JEV_MODEL,
            "answers": {
                "department": {
                    "type": "choice",
                    "choice": "billing",
                    "probabilities": {"billing": 0.9, "x": 0.1},
                    "label": "billing",
                    "label_source": "human_review",
                }
            },
        },
    }
    payloads = presets.rows_to_payloads(presets.JEV_NATIVE, row)

    assert payloads[0]["label"] == "billing"
    assert payloads[0]["label_source"] == "human_review"


def test_a_drift_block_does_not_claim_a_model_change_between_periods() -> None:
    """QA: a period comparison printed 'model changed: 2026-W36 -> 2026-W37'."""
    from jeval.drift import format_ci_block
    from jeval.report.model import DriftSlice, DriftView

    view = DriftView(
        baseline_label="2026-W36",
        current_label="2026-W37",
        slices=(
            DriftSlice(label="2026-W36", model="m", start="", end="", n=50, ece=0.1),
            DriftSlice(label="2026-W37", model="m", start="", end="", n=50, ece=0.2),
        ),
        note="",
    )
    block = format_ci_block(view, ())

    assert "period changed: 2026-W36 -> 2026-W37" in block
    assert "model changed" not in block


def test_comparing_a_slice_with_itself_is_not_a_comparison() -> None:
    """QA: `--compare` pointing at the newest model printed 'model changed: new -> new'."""
    from jeval.drift import format_ci_block
    from jeval.report.model import DriftSlice, DriftView

    view = DriftView(
        baseline_label="new",
        current_label="new",
        slices=(
            DriftSlice(label="q", model="new", start="", end="", n=50, ece=0.1),
            DriftSlice(label="q", model="new", start="", end="", n=50, ece=0.1),
        ),
        note="",
    )
    block = format_ci_block(view, ())

    assert "nothing to compare" in block
    assert "model changed" not in block
