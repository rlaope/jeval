"""Recalibration: fit a confidence correction map and hand it to the caller.

Scope
-----
jeval *measures* calibration and *exports a correction map*. It does not host, serve, or
fine-tune a model, and it never silently rewrites a user's data: nothing in this module mutates a
``DecisionRecord``, a stored log, or a prediction in flight. The deliverable is a file
(:func:`export_yaml`) plus :func:`apply`, and the user's application decides where to call it::

    fit = fit_temperature(records)
    if fit.helps:
        export_yaml(fit, "recalibration.yaml")

    # ... later, in the application that owns the data
    fit = load_yaml("recalibration.yaml")
    corrected = apply(fit, model_confidence)

``model_confidence`` is the probability that the answer given is right: the top-1 probability for a
``choice`` question, and ``max(p, 1 - p)`` for a yes/no ``noul`` question whose model returned
``p`` for "yes" (``DecisionRecord.stated_probability``). The map is fitted on that scale.

Two maps are offered. Temperature scaling divides the top-1 logit by a single number ``T`` and
pushes it back through a sigmoid, so the ranking of answers never changes. Isotonic regression
fits a monotone confidence -> accuracy map by pool-adjacent-violators (implemented here in a few
lines rather than pulled in as a scikit-learn dependency) and stores it as knots.

What the numbers mean
---------------------
``before_ece`` is the ECE of the log as it stands, measured with
:func:`jeval.calibration.compute_calibration`, so it is the same number the rest of the tool
reports. ``after_ece`` is *cross-validated*: the map is fitted on all but one fold of the log and
scored on the fold it never saw, repeated over several fold assignments, and what is reported is
the mean of those out-of-fold ECEs. An in-sample ``after_ece`` would flatter any flexible map --
isotonic regression in particular can fit the noise of the very records it is scored on and claim
a two-point gain on data that is already calibrated.

A correction is shipped only when the cross-validated improvement beats ``gain_floor``, the
larger of :data:`MIN_IMPROVEMENT` and :data:`NOISE_MULTIPLIER` standard errors of the log's own
ECE. When it does not, the fit keeps the identity map, reports ``after_ece == before_ece``, sets
``helps`` false, and states the numbers in ``note``: jeval does not invent an improvement, and it
does not claim one it cannot show. A log that is already calibrated therefore comes back saying
no correction is needed -- which is a result, not a failure.

Grid search for ``T``
---------------------
``T`` is chosen by minimising ECE over ``grid`` (Euclidean spacing is wrong here: the useful
range is multiplicative, so the default grid is geometric). ``T = 1`` is always a candidate and
is evaluated first, and ties are broken toward it, so an arbitrary grid point can never beat
doing nothing on a tie.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import yaml
from numpy.typing import NDArray

from jeval.calibration import compute_calibration
from jeval.schema import DecisionRecord

DEFAULT_TEMPERATURE_GRID: tuple[float, ...] = tuple(
    sorted({float(value) for value in np.geomspace(0.5, 5.0, 91)} | {1.0})
)
"""Geometric grid from 0.5 to 5.0 with ``T = 1`` (no correction) included."""

IDENTITY_TEMPERATURE = 1.0
IDENTITY_KNOTS: tuple[tuple[float, float], ...] = ((0.0, 0.0), (1.0, 1.0))
"""The line ``accuracy = confidence``: the map that means "no correction"."""

MIN_IMPROVEMENT = 0.005
"""Smallest ECE movement jeval is willing to ship a map for. Below this the move is noise."""

NOISE_MULTIPLIER = 2.0
"""Standard errors of the log's own ECE that a correction has to clear (see ``_gain_floor``)."""

MIN_FIT_RECORDS = 2
"""Fewer labeled decisions than this cannot support a fit at all."""

DEFAULT_FOLDS = 5
DEFAULT_REPEATS = 4
DEFAULT_SEED = 0
CONFIDENCE_EPSILON = 1e-6
"""Confidences are clipped to ``[epsilon, 1 - epsilon]`` before the logit; ``logit(1)`` is inf."""

_METHODS = ("temperature", "isotonic")

_SubsetFit: TypeAlias = Callable[[NDArray[np.float64], NDArray[np.bool_]], dict[str, Any]]
_MapTransform: TypeAlias = Callable[[NDArray[np.float64], Mapping[str, Any]], NDArray[np.float64]]


@dataclass(frozen=True)
class RecalibrationFit:
    """A fitted correction map plus the evidence for shipping it.

    ``params`` is the map itself: ``{"T": float}`` for temperature scaling or
    ``{"knots": [[confidence, accuracy], ...]}`` for isotonic. ``before_ece`` and ``after_ece``
    are the ECE this map is measured to change (see the module docstring for how ``after_ece``
    is estimated), and ``gain_floor`` is the improvement it had to beat to be shipped at all.
    ``helps`` is false when jeval could not show the map improves anything; in that case
    ``params`` is the identity and ``after_ece`` equals ``before_ece``, so calling :func:`apply`
    is a genuine no-op rather than a silently harmful edit.
    """

    method: str
    params: dict[str, Any]
    before_ece: float
    after_ece: float
    n: int
    helps: bool = False
    gain_floor: float | None = None
    note: str = ""

    @property
    def improvement(self) -> float:
        """ECE the map is measured to remove; ``0.0`` when no correction was retained."""
        return self.before_ece - self.after_ece


def fit_temperature(
    records: Sequence[DecisionRecord],
    *,
    grid: Sequence[float] = DEFAULT_TEMPERATURE_GRID,
    folds: int = DEFAULT_FOLDS,
    repeats: int = DEFAULT_REPEATS,
    seed: int = DEFAULT_SEED,
) -> RecalibrationFit:
    """Fit one temperature ``T`` on the top-1 confidence by minimising ECE over ``grid``.

    The corrected confidence is ``sigmoid(logit(p) / T)``, so ``T > 1`` softens an overconfident
    model and ``T < 1`` sharpens an underconfident one. Records that cannot participate
    (unlabeled, or ``score``-type, which are measured on another scale) are excluded and counted
    out of ``n``. A log with fewer than :data:`MIN_FIT_RECORDS` usable decisions is an error, not
    a fit.

    The retained ``T`` minimises ECE in-sample; whether the map actually helps is decided
    out-of-sample, which is why a log that is already calibrated comes back with ``T = 1`` and
    ``helps`` false.
    """
    candidates = _temperature_candidates(grid)

    def fit_subset(confidences: NDArray[np.float64], correct: NDArray[np.bool_]) -> dict[str, Any]:
        return {"T": _best_temperature(confidences, correct, candidates)}

    return _fit_recalibration(
        records,
        method="temperature",
        fit_subset=fit_subset,
        transform=_temperature_confidence,
        folds=folds,
        repeats=repeats,
        seed=seed,
    )


def fit_isotonic(
    records: Sequence[DecisionRecord],
    *,
    folds: int = DEFAULT_FOLDS,
    repeats: int = DEFAULT_REPEATS,
    seed: int = DEFAULT_SEED,
) -> RecalibrationFit:
    """Fit a monotone confidence -> accuracy map by pool-adjacent-violators, stored as knots.

    Records are sorted by confidence and neighbouring blocks are merged while their accuracies
    decrease. Each surviving block contributes one knot at its highest confidence and its pooled
    accuracy, so the map is non-decreasing in confidence by construction. Between knots the map
    interpolates linearly; outside them it holds the end values, so it never extrapolates.

    Isotonic regression has enough freedom to fit noise, so the honest question -- does the map
    help on records it was not fitted on? -- decides whether it is retained. On a calibrated log
    the answer is no, and the fit returns the identity line.
    """
    return _fit_recalibration(
        records,
        method="isotonic",
        fit_subset=_pav_knots,
        transform=_knot_confidence,
        folds=folds,
        repeats=repeats,
        seed=seed,
    )


def apply(fit: RecalibrationFit, confidence: float) -> float:
    """Apply a fitted map to one confidence and return the corrected confidence in ``[0, 1]``.

    This is the whole runtime interface: the caller's application decides when to call it, and
    jeval never applies a map to anything by itself. The parameters are re-validated on every
    call, so a hand-edited or truncated map is refused with ``ValueError`` instead of being
    applied silently. An identity map (``helps`` false) returns the input unchanged.
    """
    value = np.asarray([float(confidence)], dtype=float)
    if fit.method == "temperature":
        corrected = _temperature_confidence(value, fit.params)
    elif fit.method == "isotonic":
        corrected = _knot_confidence(value, fit.params)
    else:
        raise ValueError(f"unknown method {fit.method!r}; expected one of {_METHODS}")
    return float(corrected[0])


def export_yaml(fit: RecalibrationFit, path: Path | str) -> Path:
    """Write a fitted map to ``path`` as YAML and return the path.

    The file carries the map (``params``) together with the evidence behind it (``before_ece``,
    ``after_ece``, ``helps``, ``gain_floor``, ``n``), so a reader can tell a measured correction
    from one that was shipped anyway. Floats are written at full precision: rounding a correction
    map changes the correction it applies, which is exactly the kind of silent edit jeval refuses
    to make.
    """
    target = Path(path)
    payload: dict[str, Any] = {
        "method": fit.method,
        "n": fit.n,
        "before_ece": float(fit.before_ece),
        "after_ece": float(fit.after_ece),
        "helps": bool(fit.helps),
        "gain_floor": None if fit.gain_floor is None else float(fit.gain_floor),
        "params": _params_for_yaml(fit),
        "note": fit.note,
    }
    target.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return target


def load_yaml(path: Path | str) -> RecalibrationFit:
    """Read a map written by :func:`export_yaml` back into a :class:`RecalibrationFit`.

    Round-trip is exact: the loaded fit equals the exported one and applies identically, because
    nothing is rounded on the way out or in. The map and the evidence for it are both required:
    ``method``, ``params``, ``before_ece``, ``after_ece``, ``n`` and ``helps`` must be present,
    and anything malformed (an unknown method, a non-positive temperature, knots that are out of
    order, values outside ``[0, 1]``) raises ``ValueError``. A correction map that cannot be
    read back is worse than no map, so this loader refuses rather than guesses.
    """
    source = Path(path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:  # pragma: no cover - message text is pyyaml's
        raise ValueError(f"{source} is not valid YAML: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{source} does not contain a recalibration map")
    method = payload.get("method")
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {_METHODS}")
    params = _validated_params(str(method), payload.get("params"))
    return RecalibrationFit(
        method=str(method),
        params=params,
        before_ece=_required_float(payload, "before_ece", source),
        after_ece=_required_float(payload, "after_ece", source),
        n=int(_required_float(payload, "n", source)),
        helps=_required_bool(payload, "helps", source),
        gain_floor=_optional_float(payload.get("gain_floor")),
        note=str(payload.get("note") or ""),
    )


def _usable_points(
    records: Sequence[DecisionRecord],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """``(confidence, correct)`` pairs that can participate in a fit.

    ``DecisionRecord.calibration_point`` returns ``None`` for unlabeled decisions and for
    ``score``-type records, which are measured on a different scale; both are excluded here
    rather than coerced into a binary accuracy they do not have.
    """
    pairs = [point for record in records if (point := record.calibration_point()) is not None]
    if len(pairs) < MIN_FIT_RECORDS:
        raise ValueError(
            f"need at least {MIN_FIT_RECORDS} labeled binary decisions to fit a correction; "
            f"got {len(pairs)}"
        )
    confidences = np.asarray([point[0] for point in pairs], dtype=float)
    correct = np.asarray([point[1] for point in pairs], dtype=bool)
    return (confidences, correct)


def _measured_ece(confidences: NDArray[np.float64], correct: NDArray[np.bool_]) -> float:
    """ECE of a population, measured with the project's own binning.

    ``n_boot=0`` skips the bootstrap interval: fitting needs the point estimate only, and the
    interval is the report's job. The ECE itself is the same number
    :func:`jeval.calibration.compute_calibration` gives at its defaults, so it is comparable with
    every other number jeval prints.
    """
    return float(compute_calibration(confidences, correct, n_boot=0).ece)


def _fit_recalibration(
    records: Sequence[DecisionRecord],
    *,
    method: str,
    fit_subset: _SubsetFit,
    transform: _MapTransform,
    folds: int,
    repeats: int,
    seed: int,
) -> RecalibrationFit:
    """Shared fit path: cross-validated measurement, then the honesty gate."""
    if folds < 2:
        raise ValueError(f"folds must be >= 2; got {folds}")
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1; got {repeats}")

    confidences, correct = _usable_points(records)
    n = int(confidences.size)
    before_ece = _measured_ece(confidences, correct)
    after_ece = _cross_validated_ece(
        confidences,
        correct,
        folds=folds,
        repeats=repeats,
        seed=seed,
        fit_subset=fit_subset,
        transform=transform,
    )
    gain = before_ece - after_ece
    floor = _gain_floor(confidences, correct)
    if gain > floor:
        return RecalibrationFit(
            method=method,
            params=fit_subset(confidences, correct),
            before_ece=before_ece,
            after_ece=after_ece,
            n=n,
            helps=True,
            gain_floor=floor,
            note=_retained_note(before_ece, after_ece, floor, n, folds, repeats),
        )
    return RecalibrationFit(
        method=method,
        params=_identity_params(method),
        before_ece=before_ece,
        after_ece=before_ece,
        n=n,
        helps=False,
        gain_floor=floor,
        note=_rejected_note(before_ece, after_ece, floor, n, folds, repeats),
    )


def _cross_validated_ece(
    confidences: NDArray[np.float64],
    correct: NDArray[np.bool_],
    *,
    folds: int,
    repeats: int,
    seed: int,
    fit_subset: _SubsetFit,
    transform: _MapTransform,
) -> float:
    """Mean ECE of a map fitted without each record, over ``repeats`` fold assignments.

    This is what makes ``after_ece`` an honest estimate: a map that only encodes the noise of the
    records it was fitted on gains nothing here. Each repeat scores every record exactly once, so
    one repeat is one ECE for one fitted map -- the repeats are averaged as ECEs, not as corrected
    confidences, so the reported number stays a measurement of a single map rather than of a
    bagged ensemble of them. The fold assignment is seeded, so two runs over the same log agree.
    """
    n = int(confidences.size)
    effective_folds = max(2, min(folds, n))
    estimates = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        order = np.random.default_rng(seed + repeat).permutation(n)
        corrected = np.empty(n, dtype=float)
        for index in range(effective_folds):
            held = order[index::effective_folds]
            if held.size == 0:
                continue
            train = np.setdiff1d(order, held)
            corrected[held] = transform(
                confidences[held], fit_subset(confidences[train], correct[train])
            )
        estimates[repeat] = _measured_ece(corrected, correct)
    return float(np.mean(estimates))


def _sampling_noise(confidences: NDArray[np.float64], correct: NDArray[np.bool_]) -> float:
    """One standard-error scale for the ECE of this log, from its own bins.

    ECE is a weighted mean of ``|accuracy - confidence|`` over bins. A bin holding ``n_b``
    records measures its accuracy to roughly ``sqrt(acc(1 - acc) / n_b)``, and the bin weight
    ``n_b / n`` turns that into a standard error on the mean. The confidence spread inside a bin
    contributes to the gap too, but this term dominates at the sample sizes jeval is used on, and
    it is the term that shrinks with more labels -- which is the honest thing for the floor to
    track.
    """
    metrics = compute_calibration(confidences, correct, n_boot=0)
    total = 0.0
    for calibration_bin in metrics.bins:
        weight = calibration_bin.n / metrics.n
        accuracy = calibration_bin.accuracy
        total += weight * weight * (accuracy * (1.0 - accuracy) / calibration_bin.n)
    return math.sqrt(total)


def _gain_floor(confidences: NDArray[np.float64], correct: NDArray[np.bool_]) -> float:
    """The ECE improvement a correction has to beat before jeval will ship it.

    The larger of :data:`MIN_IMPROVEMENT` and :data:`NOISE_MULTIPLIER` standard errors of the
    log's own ECE. Two standard errors is deliberately conservative: at the sample sizes jeval is
    used on, a single standard error admits spurious "corrections" on logs that are already
    calibrated, and the cost of the stricter bar is only that a barely-detectable miscalibration
    gets a note instead of a map.
    """
    return max(MIN_IMPROVEMENT, NOISE_MULTIPLIER * _sampling_noise(confidences, correct))


def _retained_note(
    before_ece: float,
    after_ece: float,
    floor: float,
    n: int,
    folds: int,
    repeats: int,
) -> str:
    """One sentence a report can print: what the map is measured to do."""
    return (
        f"Correction retained: cross-validated ECE {after_ece:.4f} against {before_ece:.4f} "
        f"uncorrected (improvement {before_ece - after_ece:+.4f} clears the {floor:.4f} floor; "
        f"n={n}, {folds}-fold x {repeats})."
    )


def _rejected_note(
    before_ece: float,
    after_ece: float,
    floor: float,
    n: int,
    folds: int,
    repeats: int,
) -> str:
    """One sentence a report can print: why no correction is being offered."""
    return (
        f"No correction retained: the best map moves cross-validated ECE by "
        f"{before_ece - after_ece:+.4f} ({after_ece:.4f} against {before_ece:.4f} uncorrected), "
        f"which does not clear the {floor:.4f} floor ({NOISE_MULTIPLIER:.1f} x the ECE sampling "
        f"noise of this log; n={n}, {folds}-fold x {repeats}). Apply() is the identity here: "
        "this log does not need this correction."
    )


def _identity_params(method: str) -> dict[str, Any]:
    """A fresh identity map for ``method``: applying it returns the confidence unchanged."""
    if method == "temperature":
        return {"T": IDENTITY_TEMPERATURE}
    if method == "isotonic":
        return {"knots": [[x, y] for x, y in IDENTITY_KNOTS]}
    raise ValueError(f"unknown method {method!r}; expected one of {_METHODS}")


def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Logistic function, written in two branches so a small ``T`` cannot overflow ``exp``."""
    positive = values >= 0.0
    result = np.empty(values.shape, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative = ~positive
    result[negative] = np.exp(values[negative]) / (1.0 + np.exp(values[negative]))
    return result


def _temperature_confidence(
    values: NDArray[np.float64], params: Mapping[str, Any]
) -> NDArray[np.float64]:
    """``sigmoid(logit(clip(p)) / T)``, kept inside ``(0, 1)``.

    ``T = 1`` returns the input clipped to ``[0, 1]`` and nothing else. That shortcut matters:
    it makes a rejected fit a literal no-op, so a map jeval could not show to help cannot move a
    single prediction, not even the ``logit(1)`` endpoint by an epsilon.
    """
    temperature = _temperature_of(params)
    if temperature == IDENTITY_TEMPERATURE:
        return np.asarray(np.clip(values, 0.0, 1.0), dtype=float)
    clipped = np.clip(values, CONFIDENCE_EPSILON, 1.0 - CONFIDENCE_EPSILON)
    logits = np.log(clipped / (1.0 - clipped))
    return np.asarray(
        np.clip(
            _sigmoid(np.asarray(logits / temperature, dtype=float)),
            CONFIDENCE_EPSILON,
            1.0 - CONFIDENCE_EPSILON,
        ),
        dtype=float,
    )


def _temperature_candidates(grid: Sequence[float]) -> tuple[float, ...]:
    """Validated grid plus ``T = 1``, ordered outwards from the identity.

    Ordering by ``|T - 1|`` is what breaks ties toward doing nothing: the grid search keeps the
    first candidate that is strictly better, so a grid point only wins if it actually wins.
    """
    if len(grid) == 0:
        raise ValueError("grid must not be empty")
    candidates = {float(value) for value in grid} | {IDENTITY_TEMPERATURE}
    for temperature in candidates:
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(f"temperatures must be finite and > 0; got {temperature!r}")
    return tuple(sorted(candidates, key=lambda value: (abs(value - IDENTITY_TEMPERATURE), value)))


def _best_temperature(
    confidences: NDArray[np.float64],
    correct: NDArray[np.bool_],
    candidates: Sequence[float],
) -> float:
    """Grid search for the ``T`` with the lowest ECE, starting from no correction at all."""
    best_temperature = IDENTITY_TEMPERATURE
    best_ece = _measured_ece(confidences, correct)
    for temperature in candidates:
        candidate_ece = _measured_ece(
            _temperature_confidence(confidences, {"T": temperature}), correct
        )
        if candidate_ece < best_ece:
            best_temperature = temperature
            best_ece = candidate_ece
    return best_temperature


def _pav_knots(confidences: NDArray[np.float64], correct: NDArray[np.bool_]) -> dict[str, Any]:
    """Monotone confidence -> accuracy map by pool-adjacent-violators, as isotonic knots.

    One knot per surviving block at ``(highest confidence in the block, pooled accuracy)``. Block
    uppers are strictly increasing and pooled accuracies are non-decreasing, which is exactly the
    invariant :func:`apply` relies on.

    Blocks that end on the same confidence are merged afterwards. Real logs repeat confidences
    (a clipped ``1.0`` shows up hundreds of times), and a map that takes two different values at
    one confidence is not a function; merging pools them, and because two adjacent blocks are
    already ordered, the pooled value stays ordered against both neighbours.
    """
    order = np.argsort(confidences, kind="stable")
    sums: list[float] = []
    weights: list[float] = []
    uppers: list[float] = []
    for index in order:
        sums.append(float(correct[index]))
        weights.append(1.0)
        uppers.append(float(confidences[index]))
        while len(sums) > 1 and sums[-2] / weights[-2] > sums[-1] / weights[-1]:
            merged_sum = sums.pop() + sums.pop()
            merged_weight = weights.pop() + weights.pop()
            merged_upper = uppers.pop()
            uppers.pop()
            sums.append(merged_sum)
            weights.append(merged_weight)
            uppers.append(merged_upper)

    pooled_sum: list[float] = []
    pooled_weight: list[float] = []
    pooled_upper: list[float] = []
    for index in range(len(sums)):
        if pooled_upper and pooled_upper[-1] == uppers[index]:
            pooled_sum[-1] += sums[index]
            pooled_weight[-1] += weights[index]
            continue
        pooled_sum.append(sums[index])
        pooled_weight.append(weights[index])
        pooled_upper.append(uppers[index])
    return {
        "knots": [
            [pooled_upper[i], pooled_sum[i] / pooled_weight[i]] for i in range(len(pooled_upper))
        ]
    }


def _knot_confidence(values: NDArray[np.float64], params: Mapping[str, Any]) -> NDArray[np.float64]:
    """Piecewise-linear monotone interpolation between knots, clamped at both ends."""
    knots = _knots_of(params)
    xs = np.asarray([knot[0] for knot in knots], dtype=float)
    ys = np.asarray([knot[1] for knot in knots], dtype=float)
    return np.asarray(np.interp(values, xs, ys), dtype=float)


def _temperature_of(params: Mapping[str, Any]) -> float:
    """The validated ``T`` from a parameter mapping."""
    raw = params.get("T") if isinstance(params, Mapping) else None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"temperature params need a numeric 'T'; got {raw!r}")
    temperature = float(raw)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be finite and > 0; got {temperature!r}")
    return temperature


def _knots_of(params: Mapping[str, Any]) -> list[list[float]]:
    """The validated isotonic knots: strictly increasing confidence, non-decreasing accuracy."""
    raw = params.get("knots") if isinstance(params, Mapping) else None
    if not isinstance(raw, (list, tuple)) or len(raw) == 0:
        raise ValueError("isotonic params need a non-empty 'knots' list")
    knots: list[list[float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"each knot must be a [confidence, accuracy] pair; got {item!r}")
        confidence, accuracy = item
        if isinstance(confidence, bool) or isinstance(accuracy, bool):
            raise ValueError(f"knot values must be numbers; got {item!r}")
        if not isinstance(confidence, (int, float)) or not isinstance(accuracy, (int, float)):
            raise ValueError(f"knot values must be numbers; got {item!r}")
        confidence_value = float(confidence)
        accuracy_value = float(accuracy)
        if not 0.0 <= confidence_value <= 1.0 or not 0.0 <= accuracy_value <= 1.0:
            raise ValueError(f"knot values must lie in [0, 1]; got {item!r}")
        knots.append([confidence_value, accuracy_value])
    for index in range(1, len(knots)):
        if knots[index][0] <= knots[index - 1][0]:
            raise ValueError("knot confidences must be strictly increasing")
        if knots[index][1] < knots[index - 1][1]:
            raise ValueError("knot accuracies must be non-decreasing")
    return knots


def _params_for_yaml(fit: RecalibrationFit) -> dict[str, Any]:
    """Canonical, YAML-safe form of the map: plain floats in plain lists."""
    if fit.method == "temperature":
        return {"T": _temperature_of(fit.params)}
    if fit.method == "isotonic":
        return {"knots": _knots_of(fit.params)}
    raise ValueError(f"unknown method {fit.method!r}; expected one of {_METHODS}")


def _validated_params(method: str, raw: object) -> dict[str, Any]:
    """Validate loaded parameters and return them in canonical form."""
    if not isinstance(raw, dict):
        raise ValueError(f"params must be a mapping; got {raw!r}")
    if method == "temperature":
        return {"T": _temperature_of(raw)}
    return {"knots": _knots_of(raw)}


def _required_float(payload: dict[str, Any], key: str, source: Path) -> float:
    """A required numeric field, or a ``ValueError`` naming the file and the field."""
    raw = payload.get(key)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
        raise ValueError(f"{source} needs a finite numeric {key!r}; got {raw!r}")
    return float(raw)


def _required_bool(payload: dict[str, Any], key: str, source: Path) -> bool:
    """A required boolean field.

    ``helps`` is deliberately required rather than defaulted: a map whose file does not say
    whether it was measured to help would be applied on the strength of silence.
    """
    raw = payload.get(key)
    if not isinstance(raw, bool):
        raise ValueError(f"{source} needs a boolean {key!r}; got {raw!r}")
    return raw


def _optional_float(raw: object) -> float | None:
    """A number that may be absent, as YAML writes ``null`` for a measurement that was not made."""
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
        raise ValueError(f"expected a finite number or null; got {raw!r}")
    return float(raw)
