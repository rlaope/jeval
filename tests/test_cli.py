"""End-to-end tests of the command surface."""

from __future__ import annotations

import json
import subprocess
import sys
from importlib.metadata import entry_points
from pathlib import Path

from typer.testing import CliRunner

from jeval.cli import app, main

runner = CliRunner()


def test_version_is_reported() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "jeval" in result.stdout


def test_console_script_entry_point_resolves() -> None:
    """The packaging entry point must point at something that exists."""
    scripts = {entry.name: entry for entry in entry_points(group="console_scripts")}
    assert "jeval" in scripts, "the jeval console script is not installed"
    assert scripts["jeval"].load() is not None
    assert callable(main)


def test_module_entry_point_runs() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "jeval", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "jeval" in result.stdout


def test_init_scaffolds_the_working_directory(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", "--root", str(tmp_path)])
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / ".jeval" / "config.yaml").exists()
    assert (tmp_path / ".jeval" / "ingest-map.yaml").exists()
    assert (tmp_path / ".jeval" / "examples" / "costs.example.yaml").exists()
    assert "next: jeval ingest" in result.stdout


def test_demo_runs_end_to_end_and_writes_a_report(tmp_path: Path) -> None:
    out_dir = tmp_path / "demo"
    result = runner.invoke(app, ["demo", "--out-dir", str(out_dir), "--scale", "0.2"])
    assert result.exit_code == 0, result.stdout
    assert (out_dir / ".jeval" / "records.jsonl").exists()
    assert (out_dir / ".jeval" / "config.yaml").exists()
    report = out_dir / "report.html"
    assert report.exists() and report.stat().st_size > 2000
    assert "ECE" in result.stdout
    assert "synthetic" in result.stdout


def test_ingest_then_report_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "log.jsonl"
    rows = [
        {
            "model": "jev-1.13.0",
            "question_key": "department",
            "question_type": "choice",
            "prediction": "billing",
            "confidence": 0.9 if index % 2 else 0.55,
            "label": "billing" if index % 3 else "technical",
            "label_source": "human_override",
            "segment": {"lang": "ko"},
        }
        for index in range(120)
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    ingest = runner.invoke(app, ["ingest", str(source), "--root", str(tmp_path)])
    assert ingest.exit_code == 0, ingest.stdout
    assert "wrote 120 records" in ingest.stdout
    assert (tmp_path / ".jeval" / "records.jsonl").exists()

    report = runner.invoke(app, ["report", "--root", str(tmp_path), "--bins", "4", "--by", "lang"])
    assert report.exit_code == 0, report.stdout
    assert "ECE" in report.stdout
    assert "report:" in report.stdout
    assert (tmp_path / "report.html").exists()


def test_report_without_records_fails_with_a_pointer(tmp_path: Path) -> None:
    result = runner.invoke(app, ["report", "--root", str(tmp_path)])
    assert result.exit_code == 1
    # The pointer is an error, so every command emits it on stderr through one shared loader.
    assert "jeval ingest" in result.stdout + getattr(result, "stderr", "")


def test_ingest_reports_unlabeled_records_and_the_free_label_hint(tmp_path: Path) -> None:
    source = tmp_path / "unlabeled.jsonl"
    source.write_text(
        json.dumps(
            {
                "model": "m",
                "question_key": "q",
                "question_type": "choice",
                "prediction": "a",
                "confidence": 0.7,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["ingest", str(source), "--root", str(tmp_path)])
    assert result.exit_code == 0
    assert "no label yet" in result.stdout
    assert "free labels" in result.stdout


def test_equal_width_flag_is_accepted(tmp_path: Path) -> None:
    demo = runner.invoke(app, ["demo", "--out-dir", str(tmp_path / "d"), "--scale", "0.1"])
    assert demo.exit_code == 0, demo.stdout
    result = runner.invoke(
        app, ["report", "--root", str(tmp_path / "d"), "--bins-equal-width", "--bins", "5"]
    )
    assert result.exit_code == 0, result.stdout
    assert "equal-width" in (tmp_path / "d" / "report.html").read_text(encoding="utf-8")
