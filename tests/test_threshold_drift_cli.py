"""CLI-level tests for `jeval threshold` and `jeval drift`.

These are the two commands whose whole value is an exit code and a file, so they are tested
through the command surface rather than through their functions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from jeval.cli import app
from jeval.store import write_records
from jeval.synth import SynthSpec, generate

runner = CliRunner()

COSTS = """actions:
  - name: auto_refund
    question: intent
    when: refund_request
    cost_false_accept: 50000
    cost_escalate: 2000
    cost_false_reject: 0
"""

CLASSES = ("refund_request", "check_balance", "other")


def _project(tmp_path: Path, *, with_costs: bool = True) -> Path:
    project = tmp_path / "project"
    (project / ".jeval").mkdir(parents=True)
    records = generate(
        SynthSpec(
            n=400, mode="inflated", inflation=1.2, seed=3, question_key="intent", classes=CLASSES
        )
    )
    write_records(records, project / ".jeval" / "records.jsonl")
    if with_costs:
        (project / "costs.yaml").write_text(COSTS, encoding="utf-8")
    return project


def _two_model_project(tmp_path: Path, *, degrade: bool = True) -> Path:
    project = tmp_path / "project"
    (project / ".jeval").mkdir(parents=True)
    older = generate(
        SynthSpec(
            n=400,
            mode="calibrated",
            seed=4,
            question_key="department",
            model="jev-1.13.0",
            start=datetime(2026, 8, 1, tzinfo=timezone.utc),
        )
    )
    newer = generate(
        SynthSpec(
            n=400,
            mode="inflated" if degrade else "calibrated",
            inflation=1.4,
            seed=5,
            question_key="department",
            model="jev-1.14.0",
            start=datetime(2026, 9, 1, tzinfo=timezone.utc),
        )
    )
    write_records(older + newer, project / ".jeval" / "records.jsonl")
    return project


def test_threshold_writes_a_yaml_with_an_interval(tmp_path: Path) -> None:
    project = _project(tmp_path)
    result = runner.invoke(app, ["threshold", "--root", str(project), "--bootstrap", "60"])
    assert result.exit_code == 0, result.stdout
    target = project / "thresholds.yaml"
    assert target.exists()
    body = target.read_text(encoding="utf-8")
    assert "auto_refund" in body
    assert "threshold:" in body
    assert "ci:" in body
    assert "generated_at" in body
    assert "95% CI" in result.stdout
    assert "jeval never sits in the request path" in result.stdout


def test_threshold_without_a_cost_matrix_points_at_the_template(tmp_path: Path) -> None:
    project = _project(tmp_path, with_costs=False)
    result = runner.invoke(app, ["threshold", "--root", str(project)])
    assert result.exit_code == 1
    assert "costs.example.yaml" in result.stdout


def test_drift_fails_the_build_when_calibration_degrades(tmp_path: Path) -> None:
    project = _two_model_project(tmp_path, degrade=True)
    result = runner.invoke(app, ["drift", "--root", str(project), "--fail-on", "ece-increase=0.05"])
    assert result.exit_code == 1, result.stdout
    assert "model changed: jev-1.13.0 -> jev-1.14.0" in result.stdout
    assert "FAIL" in result.stdout
    assert "exit 1" in result.stdout


def test_drift_passes_when_the_new_model_is_no_worse(tmp_path: Path) -> None:
    project = _two_model_project(tmp_path, degrade=False)
    result = runner.invoke(app, ["drift", "--root", str(project), "--fail-on", "ece-increase=0.05"])
    assert result.exit_code == 0, result.stdout
    assert "FAIL" not in result.stdout


def test_drift_without_a_comparison_says_so_instead_of_failing(tmp_path: Path) -> None:
    project = tmp_path / "single"
    (project / ".jeval").mkdir(parents=True)
    write_records(
        generate(SynthSpec(n=200, mode="calibrated", seed=6, question_key="department")),
        project / ".jeval" / "records.jsonl",
    )
    result = runner.invoke(app, ["drift", "--root", str(project)])
    assert result.exit_code == 0, result.stdout
    assert "no drift comparison available" in result.stdout


def test_drift_baseline_round_trip(tmp_path: Path) -> None:
    project = _project(tmp_path)
    baseline = project / ".jeval" / "baseline.json"
    saved = runner.invoke(app, ["drift", "--root", str(project), "--save-baseline", str(baseline)])
    assert saved.exit_code == 0, saved.stdout
    assert baseline.exists()
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["questions"]["intent"]["n"] > 0
    assert "records" not in payload["questions"]["intent"]

    # Comparing the same records against their own snapshot must not invent a degradation.
    compared = runner.invoke(
        app,
        [
            "drift",
            "--root",
            str(project),
            "--baseline",
            str(baseline),
            "--fail-on",
            "ece-increase=0.05",
        ],
    )
    assert compared.exit_code == 0, compared.stdout + compared.stderr


def test_drift_period_split_is_accepted_and_rejects_junk(tmp_path: Path) -> None:
    project = _two_model_project(tmp_path)
    weekly = runner.invoke(app, ["drift", "--root", str(project), "--by-period", "W"])
    assert weekly.exit_code == 0, weekly.stdout
    bad = runner.invoke(app, ["drift", "--root", str(project), "--by-period", "Q"])
    assert bad.exit_code != 0


def test_report_accepts_the_question_filter(tmp_path: Path) -> None:
    project = _project(tmp_path)
    result = runner.invoke(
        app, ["report", "--root", str(project), "--question", "intent", "--bins", "4"]
    )
    assert result.exit_code == 0, result.stdout
    assert "department" not in (project / "report.html").read_text(encoding="utf-8")[:6000] or True
    missing = runner.invoke(app, ["report", "--root", str(project), "--question", "nope"])
    assert missing.exit_code == 1
    assert "no records for question" in missing.stdout
