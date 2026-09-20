"""Tests for the single-file HTML report."""

from __future__ import annotations

from pathlib import Path

from jeval.evaluate import evaluate
from jeval.report.html import render_report, write_report
from jeval.schema import DecisionRecord
from jeval.synth import SynthSpec, demo_dataset, generate


def _record(
    *,
    question_key: str = "department",
    confidence: float,
    correct: bool,
    label_source: str = "human_override",
    question_type: str = "choice",
    segment: dict[str, str] | None = None,
) -> DecisionRecord:
    return DecisionRecord(
        model="jev-1.13.0",
        question_key=question_key,
        question_type=question_type,  # type: ignore[arg-type]
        prediction="a",
        confidence=confidence,
        label="a" if correct else "b",
        label_source=label_source,  # type: ignore[arg-type]
        segment=segment or {},
    )


def test_report_is_one_file_with_no_external_assets() -> None:
    dataset = evaluate(generate(SynthSpec(n=400, mode="inflated", seed=2)), n_boot=100)
    html = render_report(dataset)
    assert html.startswith("<!doctype html>")
    assert "<svg" in html
    assert "<script" not in html.lower()
    assert 'src="http' not in html
    assert 'href="http' not in html
    assert "@import" not in html
    assert "url(http" not in html
    # The only permitted absolute URI is the SVG namespace declaration.
    remaining = html.replace("http://www.w3.org/2000/svg", "")
    assert "http://" not in remaining and "https://" not in remaining


def test_report_states_how_many_score_records_were_excluded() -> None:
    records = generate(
        SynthSpec(
            n=200, mode="calibrated", question_type="score", question_key="satisfaction", seed=3
        )
    )
    dataset = evaluate(records, n_boot=50)
    html = render_report(dataset)
    assert "Score excluded" in html
    assert "not scored as right or wrong" in html
    assert dataset.n_score_excluded == 200
    # Score records can be labeled, but none of them enters binary accuracy.
    assert dataset.n_labeled_gold == 200
    assert dataset.overall.n == 0
    assert "In binary metrics" in html


def test_silver_labels_never_reach_the_gold_headline() -> None:
    records = [
        _record(confidence=0.9, correct=True, label_source="silver"),
        _record(confidence=0.6, correct=False, label_source="human_override"),
    ]
    dataset = evaluate(records, n_boot=20)
    assert dataset.n_labeled_gold == 1
    assert dataset.n_labeled_silver == 1
    assert dataset.overall.n == 1
    html = render_report(dataset)
    assert "Silver labels (separate)" in html
    assert "agreement" in html.lower()


def test_a_silver_only_dataset_raises_a_banner() -> None:
    records = [_record(confidence=0.9, correct=True, label_source="silver") for _ in range(50)]
    dataset = evaluate(records, n_boot=20)
    assert dataset.silver_only is True
    html = render_report(dataset)
    assert "Every label here is silver" in html


def test_an_unlabeled_dataset_says_so_instead_of_guessing() -> None:
    records = [
        DecisionRecord(
            model="m", question_key="q", question_type="choice", prediction="a", confidence=0.8
        )
        for _ in range(30)
    ]
    dataset = evaluate(records, n_boot=20)
    html = render_report(dataset)
    assert "No gold labels" in html
    assert "No labeled decisions" in dataset.overall_diagnosis


def test_segment_breakdown_appears_when_requested() -> None:
    records = []
    for index in range(300):
        records.append(
            _record(
                confidence=0.8 if index % 2 else 0.6,
                correct=index % 3 == 0,
                segment={"lang": "ko" if index % 2 else "en"},
            )
        )
    dataset = evaluate(records, n_boot=50, by=("lang",))
    assert {segment.value for segment in dataset.segments} == {"ko", "en"}
    html = render_report(dataset)
    assert "By segment" in html
    assert "CI width" in html


def test_write_report_creates_the_file(tmp_path: Path) -> None:
    dataset = evaluate(demo_dataset(seed=1, scale=0.15).records, n_boot=50, by=("lang",))
    target = write_report(dataset, tmp_path / "report.html")
    assert target.exists()
    assert target.read_text(encoding="utf-8").count("<html") == 1
    assert target.stat().st_size > 2000
