"""Baseline snapshot tests.

The property that matters: when a snapshot is compared against current records, both sides are
measured over one partition. Per-side binning would re-partition the confidence range between
"before" and "after", so part of the delta would be bin movement dressed up as model drift — the
exact mistake this tool exists to catch in other people's dashboards.
"""

from __future__ import annotations

import numpy as np
import pytest

from jeval.baseline import SNAPSHOT_VERSION, snapshot, view_from_snapshot, write_snapshot
from jeval.calibration import bins_from_edges, expected_calibration_error
from jeval.synth import SynthSpec, generate


def _records(
    *,
    n: int,
    mode: str,
    seed: int,
    question_key: str = "department",
    model: str = "jev-1.13.0",
    **spec_kwargs: object,
):
    return generate(
        SynthSpec(n=n, mode=mode, seed=seed, question_key=question_key, model=model, **spec_kwargs)  # type: ignore[arg-type]
    )


def test_snapshot_holds_measurements_and_edges_but_never_records() -> None:
    payload = snapshot(_records(n=400, mode="calibrated", seed=1))
    assert payload["schema_version"] == SNAPSHOT_VERSION
    assert payload["records"] == 400
    question = payload["questions"]["department"]
    assert question["n"] > 0
    assert len(question["edges"]) >= 3
    assert list(question["edges"]) == sorted(question["edges"])
    assert "records" not in question
    assert isinstance(question["ece"], float)


def test_current_side_is_measured_over_the_snapshot_edges() -> None:
    """The compared ECE must equal the ECE computed over the recorded edges, not fresh ones."""
    older = _records(n=500, mode="calibrated", seed=2)
    # A narrower, differently shaped current distribution: this is what would move fresh edges.
    newer = _records(n=500, mode="overconfident", exponent=0.7, seed=3)

    payload = snapshot(older)
    edges = payload["questions"]["department"]["edges"]
    view = view_from_snapshot(payload, newer)
    current_slice = view.slices[-1]

    pairs = [
        point
        for point in (record.calibration_point() for record in newer if record.is_gold)
        if point is not None
    ]
    confidences = np.asarray([point[0] for point in pairs], dtype=float)
    correct = np.asarray([point[1] for point in pairs], dtype=bool)
    expected = expected_calibration_error(
        bins_from_edges(confidences, correct, edges), int(confidences.size)
    )

    assert current_slice.ece == pytest.approx(expected, abs=1e-12)
    assert "Warning" not in view.note


def test_a_snapshot_without_edges_falls_back_and_says_so() -> None:
    older = _records(n=400, mode="calibrated", seed=4)
    newer = _records(n=400, mode="inflated", inflation=1.2, seed=5)
    payload = snapshot(older)
    payload["questions"]["department"].pop("edges")

    view = view_from_snapshot(payload, newer)
    assert "binned independently" in view.note
    assert any(slice_.ece == slice_.ece for slice_ in view.slices)


def test_same_records_against_their_own_snapshot_show_no_movement() -> None:
    records = _records(n=600, mode="inflated", inflation=1.25, seed=6)
    view = view_from_snapshot(snapshot(records), records)
    per_question = [slice_.ece for slice_ in view.slices]
    assert len(per_question) == 2
    # The snapshot stores ECE at 6 dp, so the comparison is exact to that precision.
    assert per_question[0] == pytest.approx(per_question[1], abs=1e-6)


def test_a_degraded_model_still_shows_a_larger_ece_than_the_baseline() -> None:
    baseline_records = _records(n=600, mode="calibrated", seed=7)
    degraded = _records(n=600, mode="inflated", inflation=1.4, seed=8)
    view = view_from_snapshot(snapshot(baseline_records), degraded)
    before, after = view.slices[0], view.slices[1]
    assert after.ece > before.ece


def test_a_side_below_the_sample_floor_is_omitted_not_reported() -> None:
    """The large saved side still compares; only the too-small current side is dropped."""
    tiny = _records(n=12, mode="calibrated", seed=9)
    view = view_from_snapshot(snapshot(_records(n=300, mode="calibrated", seed=10)), tiny)
    assert [slice_.n for slice_ in view.slices] == [300]
    assert all(slice_.n >= 30 for slice_ in view.slices)


def test_unsupported_snapshot_version_is_rejected() -> None:
    payload = snapshot(_records(n=200, mode="calibrated", seed=11))
    payload["schema_version"] = 99
    with pytest.raises(ValueError, match="unsupported baseline schema_version"):
        view_from_snapshot(payload, _records(n=200, mode="calibrated", seed=12))


def test_write_snapshot_round_trips(tmp_path) -> None:
    import json

    payload = snapshot(_records(n=200, mode="calibrated", seed=13))
    target = write_snapshot(payload, tmp_path / "baseline.json")
    assert json.loads(target.read_text(encoding="utf-8")) == payload


def test_snapshot_excludes_silver_and_score_records() -> None:
    silver = generate(
        SynthSpec(n=200, mode="calibrated", seed=14, label_fraction=1.0, silver_fraction=1.0)
    )
    score = generate(SynthSpec(n=100, mode="calibrated", seed=15, question_type="score"))
    payload = snapshot(list(silver) + list(score))
    assert payload["labeled_gold"] == 0
    assert payload["questions"] == {}
