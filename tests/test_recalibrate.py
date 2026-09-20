"""Tests for recalibration fitting, applying, and export.

The honesty rules are the point of this file. A fit must correct a miscalibrated log, must refuse
to claim anything on a calibrated one, and must round-trip exactly through its export file.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml

from jeval.calibration import compute_calibration
from jeval.recalibrate import (
    MIN_IMPROVEMENT,
    RecalibrationFit,
    apply,
    export_yaml,
    fit_isotonic,
    fit_temperature,
    load_yaml,
)
from jeval.schema import DecisionRecord
from jeval.synth import SynthSpec, generate


def _records(
    mode: str, seed: int, *, n: int = 1200, question_type: str = "choice"
) -> list[DecisionRecord]:
    return generate(SynthSpec(n=n, mode=mode, seed=seed, question_type=question_type))


def _pairs(records: Sequence[DecisionRecord]) -> tuple[list[float], list[bool]]:
    usable = [point for record in records if (point := record.calibration_point()) is not None]
    return ([point[0] for point in usable], [point[1] for point in usable])


@pytest.fixture(scope="module")
def inflated_records() -> list[DecisionRecord]:
    return _records("inflated", 7)


@pytest.fixture(scope="module")
def calibrated_records() -> list[DecisionRecord]:
    return _records("calibrated", 0)


@pytest.fixture(scope="module")
def inflated_temperature_fit(inflated_records: list[DecisionRecord]) -> RecalibrationFit:
    return fit_temperature(inflated_records)


@pytest.fixture(scope="module")
def inflated_isotonic_fit(inflated_records: list[DecisionRecord]) -> RecalibrationFit:
    return fit_isotonic(inflated_records)


@pytest.fixture(scope="module")
def calibrated_temperature_fit(calibrated_records: list[DecisionRecord]) -> RecalibrationFit:
    return fit_temperature(calibrated_records)


@pytest.fixture(scope="module")
def calibrated_isotonic_fit(calibrated_records: list[DecisionRecord]) -> RecalibrationFit:
    return fit_isotonic(calibrated_records)


def test_before_ece_is_the_number_the_rest_of_the_tool_reports(
    inflated_records: list[DecisionRecord], inflated_temperature_fit: RecalibrationFit
) -> None:
    confidences, correct = _pairs(inflated_records)
    reported = compute_calibration(confidences, correct)
    assert inflated_temperature_fit.before_ece == reported.ece
    assert inflated_temperature_fit.n == reported.n


def test_temperature_fit_undoes_inflation(
    inflated_records: list[DecisionRecord], inflated_temperature_fit: RecalibrationFit
) -> None:
    fit = inflated_temperature_fit
    assert fit.method == "temperature"
    assert fit.n == 1200
    assert fit.helps is True
    assert float(fit.params["T"]) > 1.0, "an inflated model needs softening, so T must exceed 1"
    assert fit.after_ece < fit.before_ece
    assert fit.improvement > MIN_IMPROVEMENT
    assert fit.gain_floor is not None and fit.improvement > fit.gain_floor
    assert "Correction retained" in fit.note

    confidences, correct = _pairs(inflated_records)
    corrected = [apply(fit, confidence) for confidence in confidences]
    assert compute_calibration(corrected, correct, n_boot=0).ece < fit.before_ece


def test_isotonic_fit_undoes_inflation(
    inflated_records: list[DecisionRecord], inflated_isotonic_fit: RecalibrationFit
) -> None:
    fit = inflated_isotonic_fit
    assert fit.method == "isotonic"
    assert fit.n == 1200
    assert fit.helps is True
    assert fit.after_ece < fit.before_ece
    assert fit.improvement > MIN_IMPROVEMENT
    assert "Correction retained" in fit.note

    confidences, correct = _pairs(inflated_records)
    corrected = [apply(fit, confidence) for confidence in confidences]
    assert compute_calibration(corrected, correct, n_boot=0).ece < fit.before_ece


def test_reported_after_ece_is_not_the_flattering_in_sample_number(
    inflated_records: list[DecisionRecord], inflated_isotonic_fit: RecalibrationFit
) -> None:
    """An isotonic map scored on its own fitting records looks better than it is.

    jeval reports the cross-validated number, which is worse -- that is the whole point of the
    cross-validation. If this ever flips, the fit has started grading its own homework.
    """
    confidences, correct = _pairs(inflated_records)
    applied = [apply(inflated_isotonic_fit, confidence) for confidence in confidences]
    in_sample = compute_calibration(applied, correct, n_boot=0).ece
    assert in_sample < inflated_isotonic_fit.after_ece


def test_fits_are_deterministic() -> None:
    """Same log, same fold seed, same map: a CI diff of the export must be about the data."""
    records = _records("inflated", 2, n=400)
    first = fit_temperature(records, folds=3, repeats=2)
    second = fit_temperature(records, folds=3, repeats=2)
    assert first == second
    first_isotonic = fit_isotonic(records, folds=3, repeats=2)
    assert first_isotonic == fit_isotonic(records, folds=3, repeats=2)


def test_the_seed_moves_the_measurement_not_the_map(
    inflated_records: list[DecisionRecord],
) -> None:
    """The fold seed perturbs the cross-validated estimate; the exported map stays data-only."""
    base = fit_isotonic(inflated_records, seed=0)
    other = fit_isotonic(inflated_records, seed=99)
    assert base.helps is True and other.helps is True
    assert base.params == other.params
    assert base.before_ece == other.before_ece


def test_isotonic_knots_are_a_monotone_map(inflated_isotonic_fit: RecalibrationFit) -> None:
    knots = inflated_isotonic_fit.params["knots"]
    assert len(knots) >= 2
    confidences = [knot[0] for knot in knots]
    accuracies = [knot[1] for knot in knots]
    assert confidences == sorted(confidences)
    assert len(set(confidences)) == len(confidences), "knot confidences must be strictly increasing"
    assert accuracies == sorted(accuracies), "pooled accuracies must be non-decreasing"


def test_calibrated_log_gets_no_temperature_correction(
    calibrated_temperature_fit: RecalibrationFit,
) -> None:
    fit = calibrated_temperature_fit
    assert fit.helps is False
    assert fit.params == {"T": 1.0}
    assert fit.after_ece == fit.before_ece
    assert fit.improvement == 0.0
    assert fit.after_ece >= fit.before_ece - MIN_IMPROVEMENT
    assert "No correction retained" in fit.note
    assert "identity" in fit.note


def test_calibrated_log_gets_no_isotonic_correction(
    calibrated_isotonic_fit: RecalibrationFit,
) -> None:
    fit = calibrated_isotonic_fit
    assert fit.helps is False
    assert fit.params == {"knots": [[0.0, 0.0], [1.0, 1.0]]}
    assert fit.after_ece == fit.before_ece
    assert fit.improvement == 0.0
    assert "No correction retained" in fit.note


def test_a_rejected_fit_is_a_real_no_op(
    calibrated_temperature_fit: RecalibrationFit, calibrated_isotonic_fit: RecalibrationFit
) -> None:
    """``helps`` false is not a label on a correction that still moves numbers: it is no map."""
    for fit in (calibrated_temperature_fit, calibrated_isotonic_fit):
        for confidence in (0.0, 0.05, 0.25, 0.5, 0.9, 0.99, 1.0):
            assert apply(fit, confidence) == pytest.approx(confidence, abs=1e-12)


@pytest.mark.parametrize("seed", [0, 3, 9, 17])
def test_calibrated_logs_get_no_correction_across_seeds(seed: int) -> None:
    """Calibrated at these seeds means calibrated; the gate may not find a gain to sell."""
    records = _records("calibrated", seed)
    for fit in (fit_temperature(records), fit_isotonic(records)):
        assert fit.helps is False
        assert fit.after_ece == fit.before_ece
        assert fit.improvement == 0.0
        assert fit.n == 1200
        assert fit.gain_floor is not None
        assert fit.gain_floor >= MIN_IMPROVEMENT
        assert fit.improvement <= fit.gain_floor
        assert "No correction retained" in fit.note


@pytest.mark.parametrize("seed", [0, 1])
def test_the_gate_still_offers_a_correction_when_there_is_one(seed: int) -> None:
    """The gate refuses because of the evidence, not because it is built to always refuse."""
    records = _records("inflated", seed, n=800)
    for fit in (fit_temperature(records), fit_isotonic(records)):
        assert fit.helps is True
        assert fit.after_ece < fit.before_ece
        assert fit.gain_floor is not None and fit.improvement > fit.gain_floor


def test_apply_temperature_is_monotone(inflated_temperature_fit: RecalibrationFit) -> None:
    confidences = [0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.999, 1.0]
    corrected = [apply(inflated_temperature_fit, confidence) for confidence in confidences]
    assert corrected == sorted(corrected)
    assert all(0.0 <= value <= 1.0 for value in corrected)
    assert len(set(corrected)) == len(confidences), "softening must stay strictly monotone"


def test_apply_isotonic_is_monotone_and_clamped(inflated_isotonic_fit: RecalibrationFit) -> None:
    confidences = [0.0, 0.01, 0.3, 0.5, 0.75, 0.9, 0.99, 1.0]
    corrected = [apply(inflated_isotonic_fit, confidence) for confidence in confidences]
    assert corrected == sorted(corrected)
    assert all(0.0 <= value <= 1.0 for value in corrected)
    assert apply(inflated_isotonic_fit, -1.0) == corrected[0]
    assert apply(inflated_isotonic_fit, 2.0) == corrected[-1]


@pytest.mark.parametrize(
    "fixture_name",
    [
        "inflated_temperature_fit",
        "inflated_isotonic_fit",
        "calibrated_temperature_fit",
        "calibrated_isotonic_fit",
    ],
)
def test_yaml_round_trip_is_exact(
    fixture_name: str, request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    fit: RecalibrationFit = request.getfixturevalue(fixture_name)
    path = export_yaml(fit, tmp_path / "recalibration.yaml")
    loaded = load_yaml(path)
    assert loaded == fit
    for confidence in (0.0, 0.05, 0.25, 0.5, 0.9, 0.99, 1.0):
        assert apply(loaded, confidence) == apply(fit, confidence)


def test_exported_file_carries_the_evidence(
    inflated_temperature_fit: RecalibrationFit, tmp_path: Path
) -> None:
    path = export_yaml(inflated_temperature_fit, tmp_path / "recalibration.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["method"] == "temperature"
    assert payload["helps"] is True
    assert payload["n"] == 1200
    assert payload["before_ece"] == inflated_temperature_fit.before_ece
    assert payload["after_ece"] == inflated_temperature_fit.after_ece
    assert payload["gain_floor"] == inflated_temperature_fit.gain_floor
    assert isinstance(payload["note"], str) and payload["note"]


def _write(payload: object, path: Path) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_load_yaml_refuses_maps_it_cannot_trust(tmp_path: Path) -> None:
    good: dict[str, object] = {
        "method": "temperature",
        "n": 1200,
        "before_ece": 0.1,
        "after_ece": 0.04,
        "helps": True,
        "gain_floor": 0.02,
        "params": {"T": 1.5},
        "note": "measured",
    }
    assert load_yaml(_write(good, tmp_path / "good.yaml")).params == {"T": 1.5}

    missing_helps = {key: value for key, value in good.items() if key != "helps"}
    with pytest.raises(ValueError, match="helps"):
        load_yaml(_write(missing_helps, tmp_path / "no-helps.yaml"))

    unknown_method = dict(good, method="platt")
    with pytest.raises(ValueError, match="unknown method"):
        load_yaml(_write(unknown_method, tmp_path / "platt.yaml"))

    zero_temperature = dict(good, params={"T": 0.0})
    with pytest.raises(ValueError, match="temperature must be finite and > 0"):
        load_yaml(_write(zero_temperature, tmp_path / "zero.yaml"))

    broken_knots = dict(
        good,
        method="isotonic",
        params={"knots": [[0.9, 0.5], [0.3, 0.6]]},
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        load_yaml(_write(broken_knots, tmp_path / "knots.yaml"))

    falling_accuracy = dict(
        good,
        method="isotonic",
        params={"knots": [[0.1, 0.9], [0.2, 0.5]]},
    )
    with pytest.raises(ValueError, match="non-decreasing"):
        load_yaml(_write(falling_accuracy, tmp_path / "falling.yaml"))

    out_of_range = dict(good, method="isotonic", params={"knots": [[1.5, 0.5]]})
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        load_yaml(_write(out_of_range, tmp_path / "range.yaml"))

    non_numeric = dict(good, n="many")
    with pytest.raises(ValueError, match="finite numeric"):
        load_yaml(_write(non_numeric, tmp_path / "n.yaml"))

    missing_count = {key: value for key, value in good.items() if key != "n"}
    with pytest.raises(ValueError, match="finite numeric 'n'"):
        load_yaml(_write(missing_count, tmp_path / "no-n.yaml"))

    with pytest.raises(ValueError, match="recalibration map"):
        load_yaml(_write(["not", "a", "map"], tmp_path / "list.yaml"))


def test_apply_refuses_an_unknown_method() -> None:
    fit = RecalibrationFit(method="platt", params={"a": 1.0}, before_ece=0.1, after_ece=0.05, n=10)
    with pytest.raises(ValueError, match="unknown method"):
        apply(fit, 0.9)


def test_fit_refuses_logs_it_cannot_use() -> None:
    with pytest.raises(ValueError, match="at least 2 labeled binary decisions"):
        fit_temperature([])
    unlabeled = _records("inflated", 4, n=40)
    for record in unlabeled:
        record.label = None
        record.label_source = None
    with pytest.raises(ValueError, match="at least 2 labeled binary decisions"):
        fit_isotonic(unlabeled)
    single = _records("inflated", 4, n=1)
    assert len(single) == 1
    with pytest.raises(ValueError, match="at least 2 labeled binary decisions"):
        fit_temperature(single)


def test_score_records_are_left_out_of_the_fit() -> None:
    choice = _records("inflated", 5, n=240)
    scored = _records("constant_high", 6, n=120, question_type="score")
    assert all(record.question_type == "score" for record in scored)
    assert all(record.calibration_point() is None for record in scored)
    fit = fit_temperature(choice + scored, folds=3, repeats=1)
    assert fit.n == len(choice)
    assert fit.n == 240


def test_temperature_grid_is_validated(inflated_records: list[DecisionRecord]) -> None:
    small = inflated_records[:60]
    with pytest.raises(ValueError, match="grid must not be empty"):
        fit_temperature(small, grid=())
    with pytest.raises(ValueError, match="finite and > 0"):
        fit_temperature(small, grid=(0.0, 1.5))
    with pytest.raises(ValueError, match="finite and > 0"):
        fit_temperature(small, grid=(float("nan"), 1.5))
    with pytest.raises(ValueError, match="folds must be >= 2"):
        fit_temperature(small, grid=(1.5,), folds=1)
    with pytest.raises(ValueError, match="repeats must be >= 1"):
        fit_temperature(small, grid=(1.5,), repeats=0)


def test_the_identity_is_always_a_candidate(inflated_records: list[DecisionRecord]) -> None:
    """A grid that excludes ``T = 1`` still cannot beat doing nothing on a tie."""
    fit = fit_temperature(inflated_records, grid=(3.0, 4.0))
    assert float(fit.params["T"]) in (1.0, 3.0, 4.0)
    assert fit.after_ece <= fit.before_ece
