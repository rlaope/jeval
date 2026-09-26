"""Human answers recorded at run time, and joined by every command without an ingest step.

The loop a service owner wants is two lines of code and no pipeline: ``collect.track`` records what
the model said, ``collect.resolve`` records what a human decided, and ``jeval report`` reads both.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from jeval import collect
from jeval.cli import app
from jeval.ingest import apply_label_events
from jeval.schema import normalize_record
from jeval.store import labels_path, read_label_events

runner = CliRunner()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JEVAL_COLLECT", raising=False)
    monkeypatch.delenv("JEVAL_ROOT", raising=False)
    collect.reset_stats()


def _decision(key: str, question: str = "department", prediction: str = "billing") -> object:
    return normalize_record(
        {
            "model": "m",
            "question_key": question,
            "question_type": "choice",
            "prediction": prediction,
            "probabilities": {"billing": 0.8, "technical": 0.2},
            "source_key": key,
        }
    )


# --- collect.resolve -------------------------------------------------------------------------------


def test_resolve_appends_one_answer_next_to_the_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JEVAL_ROOT", str(tmp_path))
    assert collect.resolve(source_key="T-1", question="department", answer="billing") is True
    lines = labels_path(tmp_path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["source_key"] == "T-1"
    assert event["question_key"] == "department"
    assert event["label"] == "billing"
    assert event["label_source"] == "human_review"
    assert collect.stats()["labels_written"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_key": "", "question": "department", "answer": "billing"},
        {"source_key": "T-1", "question": " ", "answer": "billing"},
        {"source_key": "T-1", "question": "department", "answer": ""},
        {"source_key": "T-1", "question": "department", "answer": "b", "source": "a-guess"},
    ],
)
def test_resolve_never_raises_on_a_bad_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, str]
) -> None:
    monkeypatch.setenv("JEVAL_ROOT", str(tmp_path))
    assert collect.resolve(**kwargs) is False  # type: ignore[arg-type]
    assert not labels_path(tmp_path).exists()
    assert collect.stats()["invalid_value"] == 1


def test_resolve_honours_the_off_switch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEVAL_ROOT", str(tmp_path))
    monkeypatch.setenv("JEVAL_COLLECT", "0")
    assert collect.resolve(source_key="T-1", question="department", answer="billing") is False
    assert not labels_path(tmp_path).exists()


def test_resolve_follows_a_records_path_set_in_the_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "elsewhere" / "records.jsonl"
    monkeypatch.setenv("JEVAL_COLLECT", str(target))
    assert collect.resolve(source_key="T-1", question="department", answer="billing") is True
    assert (target.parent / "labels.jsonl").exists()


# --- applying the events ---------------------------------------------------------------------------


def test_events_label_matching_records_and_count_what_they_could_not() -> None:
    records = [_decision("T-1"), _decision("T-2"), _decision("T-3", question="intent")]
    events = [
        {"source_key": "T-1", "question_key": "department", "label": "technical"},
        {"source_key": "T-1", "question_key": "department", "label": "billing"},  # later wins
        {"source_key": "T-2", "question_key": "department", "label": "nonsense"},  # impossible
        {"source_key": "T-9", "question_key": "department", "label": "billing"},  # no decision
        {"source_key": "T-3", "question_key": "department", "label": "billing"},  # other question
    ]
    for event in events:
        event.setdefault("label_source", "human_review")
    report = apply_label_events(records, events)  # type: ignore[arg-type]
    assert records[0].label == "billing" and records[0].is_gold  # type: ignore[attr-defined]
    assert records[1].label is None  # type: ignore[attr-defined]
    assert report.n_applied == 1
    assert report.n_impossible == 1
    assert report.n_unmatched == 2


def test_events_never_overwrite_a_label_that_is_already_there() -> None:
    existing = normalize_record(
        {
            "model": "m",
            "question_key": "department",
            "question_type": "choice",
            "prediction": "billing",
            "probabilities": {"billing": 0.8, "technical": 0.2},
            "source_key": "T-1",
            "label": "billing",
            "label_source": "human_override",
        }
    )
    records = [existing]
    report = apply_label_events(
        records,  # type: ignore[arg-type]
        [
            {
                "source_key": "T-1",
                "question_key": "department",
                "label": "technical",
                "label_source": "silver",
            }
        ],
    )
    assert records[0].label == "billing"
    assert report.n_kept_existing == 1


# --- the whole loop, through the CLI ---------------------------------------------------------------


def _collected_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n: int = 150) -> Path:
    monkeypatch.setenv("JEVAL_ROOT", str(tmp_path))
    for index in range(n):
        confident = index % 3 != 0
        collect.record(
            question_key="department",
            prediction="billing" if index % 2 else "technical",
            probabilities={"billing": 0.9, "technical": 0.1}
            if index % 2
            else {"billing": 0.35, "technical": 0.65},
            model="jev-1.14.0",
            source_key=f"T-{index}",
        )
        answer = ("billing" if index % 2 else "technical") if confident else "billing"
        collect.resolve(source_key=f"T-{index}", question="department", answer=answer)
    return tmp_path


def test_report_reads_answers_recorded_by_resolve_without_an_ingest_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _collected_project(tmp_path, monkeypatch)
    assert read_label_events(labels_path(project))
    result = runner.invoke(app, ["report", "--root", str(project), "-o", str(project / "r.html")])
    assert result.exit_code == 0, result.output
    assert "labeled decisions (gold): 150" in result.output
    assert "150 applied" in result.output


def test_status_says_what_was_collected_and_what_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _collected_project(tmp_path, monkeypatch, n=40)
    result = runner.invoke(app, ["status", "--root", str(project)])
    assert result.exit_code == 0, result.output
    assert "40 decisions" in result.output
    assert "40 answers" in result.output
    assert "jev-1.14.0" in result.output and "department" in result.output
    assert "60 more gold labels" in result.output  # the verdict needs 100
    assert "no cost matrix" in result.output


def test_status_explains_an_empty_project(tmp_path: Path) -> None:
    result = runner.invoke(app, ["status", "--root", str(tmp_path)])
    assert result.exit_code == 1
    assert "collect.track" in result.output


# --- review findings -------------------------------------------------------------------------------


class _Unprintable:
    def __str__(self) -> str:
        raise RuntimeError("no text for you")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_key": _Unprintable(), "question": "department", "answer": "billing"},
        {"source_key": "T-1", "question": "department", "answer": _Unprintable()},
        {"source_key": "T-1", "question": "department", "answer": "billing", "ts": "2020-01-01"},
        {"source_key": "T-1", "question": "department", "answer": "billing", "path": 123},
    ],
)
def test_resolve_never_raises_whatever_it_is_handed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, object]
) -> None:
    monkeypatch.setenv("JEVAL_ROOT", str(tmp_path))
    assert collect.resolve(**kwargs) is False  # type: ignore[arg-type]


def test_resolve_refuses_a_directory_as_its_records_path(tmp_path: Path) -> None:
    target = tmp_path / "somewhere"
    target.mkdir()
    assert collect.resolve(source_key="T", question="q", answer="a", path=target) is False
    assert not (tmp_path / "labels.jsonl").exists()


def test_a_later_silver_answer_never_replaces_an_earlier_human_one() -> None:
    records = [_decision("T-1")]
    report = apply_label_events(
        records,  # type: ignore[arg-type]
        [
            {
                "source_key": "T-1",
                "question_key": "department",
                "label": "billing",
                "label_source": "human_review",
            },
            {
                "source_key": "T-1",
                "question_key": "department",
                "label": "technical",
                "label_source": "silver",
            },
        ],
    )
    assert records[0].label == "billing" and records[0].is_gold  # type: ignore[attr-defined]
    assert report.n_superseded == 1


def test_yes_no_answers_are_normalised_like_the_schema_does() -> None:
    def noul(key: str) -> object:
        return normalize_record(
            {
                "model": "m",
                "question_key": "urgent",
                "question_type": "noul",
                "probability_positive": 0.8,
                "source_key": key,
            }
        )

    records = [noul("A"), noul("B"), noul("C")]
    apply_label_events(
        records,  # type: ignore[arg-type]
        [
            {"source_key": "A", "question_key": "urgent", "label": "True"},
            {"source_key": "B", "question_key": "urgent", "label": "false"},
            {"source_key": "C", "question_key": "urgent", "label": "1"},
        ],
    )
    assert [r.label for r in records] == ["yes", "no", "yes"]  # type: ignore[attr-defined]


def test_label_apply_never_persists_answers_joined_on_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _collected_project(tmp_path, monkeypatch, n=4)
    sheet = project / "sheet.csv"
    sheet.write_text("record_id,question_key,confidence,prediction,priority,reason,label\n")
    runner.invoke(app, ["label", "--root", str(project), "--apply", str(sheet)])
    stored = [
        json.loads(line) for line in (project / ".jeval/records.jsonl").read_text().splitlines()
    ]
    assert all(row["label"] is None for row in stored), "joined answers were written to disk"


def test_status_handles_mixed_timezones_and_reports_utc(tmp_path: Path) -> None:
    from datetime import datetime, timedelta, timezone

    from jeval.store import records_path, write_records

    aware = normalize_record(
        {
            "model": "m",
            "question_key": "q",
            "question_type": "choice",
            "prediction": "a",
            "probabilities": {"a": 0.9, "b": 0.1},
            "ts": datetime(2026, 1, 1, 9, 0, tzinfo=timezone(timedelta(hours=9))),
        }
    )
    naive = normalize_record(
        {
            "model": "m",
            "question_key": "q",
            "question_type": "choice",
            "prediction": "a",
            "probabilities": {"a": 0.9, "b": 0.1},
            "ts": datetime(2025, 12, 31, 12, 0),
        }
    )
    write_records([aware, naive], records_path(tmp_path))  # type: ignore[list-item]
    result = runner.invoke(app, ["status", "--root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "last 2026-01-01 00:00Z" in result.output


def test_status_does_not_ask_for_labels_a_score_question_cannot_use(tmp_path: Path) -> None:
    from jeval.store import records_path, write_records
    from jeval.synth import SynthSpec, generate

    write_records(
        generate(SynthSpec(n=150, mode="calibrated", question_type="score", seed=1)),
        records_path(tmp_path),
    )
    result = runner.invoke(app, ["status", "--root", str(tmp_path)])
    assert "more gold labels" not in result.output
    assert "score" in result.output
