"""Score-type measurement: how far the numbers are off, and whether the ordering survives.

``score`` records carry a numeric prediction and a numeric ground truth (a satisfaction
rating, a risk level, a 0-100 judgement), and they cannot be measured with the binary
machinery used for ``choice`` and ``noul`` records:

- **"Correct" is undefined for a number.** A prediction three points away from the truth on a
  1-10 scale is not wrong, and forcing it into right/wrong throws away every difference
  smaller than the whole scale. What can be measured is the size of the miss: MAE (the typical
  error, in the units of the question) and RMSE (the same scale, punishing large misses
  harder).
- **The ordering is often the decision-relevant part.** If the action is to send the highest
  scoring cases to a human reviewer, the model has to agree with the ground truth about which
  cases are worse; the exact values matter less. Spearman's rho measures that agreement.
- **The two are not substitutes**, which is why this module reports both. A model shifted by a
  constant has a large MAE and a perfect rho: the ranking is usable and the numbers need an
  offset correction. A model with small errors but a weak rho ranks cases in an order the
  decision cannot rely on. Either number alone hides the other case.

So ``score`` records are never folded into accuracy, and they are never measured *as* accuracy
either: the numbers below are an error scale plus a rank correlation, and every one of them
would be lost by a right/wrong reduction.

What is measured, and what is not:

- A record counts only when ``question_type == "score"`` and both ``prediction`` and ``label``
  parse as finite numbers. The schema stores both as strings, so values are parsed defensively:
  free text, empty strings, ``nan`` and ``inf`` are not measurements.
- Nothing is dropped silently. Every record inspected lands in exactly one bucket — measured
  (``n``), another question type (``n_other_type``), unlabeled (``n_unlabeled``), or
  unparseable (``n_unparseable``) — and the four buckets sum to ``n_records``.
- With fewer than two usable pairs the summary metrics are ``nan`` and the level view is empty:
  a rank correlation over one point is not a measurement, and neither is an "average" error of
  one sample. ``n`` still records what was there, so the caller can say "not enough data yet".
- The level view splits the *predicted* range into equal-width bands. Binning on the prediction
  is deliberate: the question it answers is "the model says 8 — what is the truth actually like
  up there", which is a coverage-and-bias question, not a calibration curve.

Spearman is implemented directly — rank both sides with average ranks for ties, then Pearson on
the ranks — so the project needs no scipy. This module is pure: no formatting and no I/O (the
numbers are formatted where they are printed, in the CLI/report layer, which this engine module
must not import).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from jeval.schema import DecisionRecord

DEFAULT_LEVEL_BINS = 5
MIN_USABLE_PAIRS = 3
"""Two pairs always produce a perfect rank correlation of +/-1, which is arithmetic, not
agreement: three is the least a rank statistic can say anything from."""
_DEGENERATE_RANGE = 1e-12


@dataclass(frozen=True)
class ScoreLevel:
    """One equal-width band of the predicted range, with what actually happened in it."""

    index: int
    lo: float
    hi: float
    n: int
    mean_predicted: float
    mean_actual: float

    @property
    def mean_error(self) -> float:
        """Mean signed error in the band; positive means the model runs high there."""
        return self.mean_predicted - self.mean_actual


@dataclass(frozen=True)
class ScoreMetrics:
    """Error scale, rank agreement, and the level view for a log's score records."""

    n: int
    mae: float
    rmse: float
    bias: float
    spearman_rho: float
    levels: tuple[ScoreLevel, ...]
    levels_requested: int
    n_records: int
    n_other_type: int
    n_unlabeled: int
    n_unparseable: int

    @property
    def is_measurable(self) -> bool:
        """True when enough usable pairs made the summary numbers meaningful."""
        return self.n >= MIN_USABLE_PAIRS


def parse_score_value(value: object) -> float | None:
    """Return ``value`` as a finite float, or ``None`` when it is not a score.

    ``prediction`` and ``label`` are strings in the record schema, so real logs contain ``"7"``,
    ``" 7.5 "``, ``"7 of 10"``, ``""`` and ``"nan"``. Only the numeric ones are measurements;
    everything else has to be counted rather than coerced into a number, because a coerced
    value would invent an error that was never observed.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(str(value).strip())
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _paired(
    predicted: Sequence[float], actual: Sequence[float]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    pred = np.asarray(list(predicted), dtype=float)
    real = np.asarray(list(actual), dtype=float)
    if pred.size != real.size:
        raise ValueError("predicted and actual must have the same length")
    return pred, real


def _average_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Ranks 1..n in which tied values share the mean of the ranks they span."""
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    ranks = np.empty(int(values.size), dtype=float)
    start = 0
    for stop in range(1, int(values.size) + 1):
        if stop == values.size or ordered[stop] != ordered[start]:
            ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
            start = stop
    return ranks


def _pearson(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Pearson correlation; ``nan`` when either side has no spread to correlate with."""
    x_centered = x - float(x.mean())
    y_centered = y - float(y.mean())
    x_spread = float(np.dot(x_centered, x_centered))
    y_spread = float(np.dot(y_centered, y_centered))
    spread = math.sqrt(x_spread * y_spread)
    if spread == 0.0:
        return float("nan")
    return float(np.dot(x_centered, y_centered) / spread)


def spearman_rho(predicted: Sequence[float], actual: Sequence[float]) -> float:
    """Rank agreement between predictions and ground truth, ties averaged.

    Implemented directly — rank both sides, then Pearson on the ranks — because the whole
    computation is a dozen lines and a scipy dependency is not worth it.

    Returns ``nan``, never raises, when there are fewer than two pairs, when either side is
    constant, or when a value is not finite: a constant prediction has no ordering to agree or
    disagree with, and "undefined" is the honest answer.
    """
    pred, real = _paired(predicted, actual)
    if pred.size < MIN_USABLE_PAIRS:
        return float("nan")
    if not (bool(np.isfinite(pred).all()) and bool(np.isfinite(real).all())):
        return float("nan")
    return _pearson(_average_ranks(pred), _average_ranks(real))


def mean_absolute_error(predicted: Sequence[float], actual: Sequence[float]) -> float:
    """Mean of ``|prediction - label|``, in the units of the question; ``nan`` when empty."""
    pred, real = _paired(predicted, actual)
    if pred.size == 0:
        return float("nan")
    return float(np.mean(np.abs(pred - real)))


def root_mean_squared_error(predicted: Sequence[float], actual: Sequence[float]) -> float:
    """Square root of the mean squared error; ``nan`` when empty."""
    pred, real = _paired(predicted, actual)
    if pred.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred - real) ** 2)))


def mean_signed_error(predicted: Sequence[float], actual: Sequence[float]) -> float:
    """Mean of ``prediction - label``: the size and direction of a systematic offset."""
    pred, real = _paired(predicted, actual)
    if pred.size == 0:
        return float("nan")
    return float(np.mean(pred - real))


def _finite_or_nan(value: float) -> float:
    """Two finite inputs can overflow a reduction (1e308 against -1e308), and `inf` rendered in a
    report reads as a measurement rather than as the absence of one."""
    return value if math.isfinite(value) else float("nan")


def score_levels(
    predicted: Sequence[float], actual: Sequence[float], *, n_bins: int = DEFAULT_LEVEL_BINS
) -> tuple[ScoreLevel, ...]:
    """Equal-width bands over the predicted range, each with its mean prediction and truth.

    Every requested band is returned, including bands no record falls into (``n == 0`` with
    ``nan`` means): a level the model never predicts into is itself a finding, and omitting it
    would make the view look better covered than it is. Band counts sum to the number of pairs.

    A single distinct prediction has no range to divide, so it collapses to one band holding
    all of it (``lo == hi``) rather than producing empty bands with a fabricated width.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    pred, real = _paired(predicted, actual)
    n = int(pred.size)
    if n == 0:
        return ()
    lo = float(pred.min())
    hi = float(pred.max())
    if n_bins == 1 or hi - lo < _DEGENERATE_RANGE:
        return (ScoreLevel(0, lo, hi, n, float(pred.mean()), float(real.mean())),)
    edges = [float(edge) for edge in np.linspace(lo, hi, n_bins + 1)]
    assigned = np.clip(np.digitize(pred, edges[1:-1], right=False), 0, n_bins - 1)
    levels: list[ScoreLevel] = []
    for index in range(n_bins):
        mask = assigned == index
        count = int(mask.sum())
        if count == 0:
            levels.append(
                ScoreLevel(index, edges[index], edges[index + 1], 0, float("nan"), float("nan"))
            )
            continue
        levels.append(
            ScoreLevel(
                index,
                edges[index],
                edges[index + 1],
                count,
                float(pred[mask].mean()),
                float(real[mask].mean()),
            )
        )
    return tuple(levels)


def measure_score(
    records: Iterable[DecisionRecord], *, n_bins: int = DEFAULT_LEVEL_BINS
) -> ScoreMetrics:
    """Measure the ``score`` records in ``records``; nothing else is touched.

    ``choice`` and ``noul`` records are counted (``n_other_type``) and left alone: their
    correctness belongs to the accuracy and calibration path, and folding them in here is the
    metric leak the two paths exist to prevent.

    Records whose prediction or label does not parse as a finite number, and score records with
    no label, are counted in their own buckets and excluded from the metrics — never dropped and
    never invented. With fewer than :data:`MIN_USABLE_PAIRS` usable pairs the metrics are
    ``nan``, the level view is empty, and ``n`` still says how much data was there.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    predicted: list[float] = []
    actual: list[float] = []
    n_records = 0
    n_other_type = 0
    n_unlabeled = 0
    n_unparseable = 0
    for record in records:
        n_records += 1
        if record.question_type != "score":
            n_other_type += 1
            continue
        label = record.label
        if label is None or not label.strip():
            n_unlabeled += 1
            continue
        prediction_value = parse_score_value(record.prediction)
        label_value = parse_score_value(label)
        if prediction_value is None or label_value is None:
            n_unparseable += 1
            continue
        predicted.append(prediction_value)
        actual.append(label_value)

    n = len(predicted)
    if n < MIN_USABLE_PAIRS:
        return ScoreMetrics(
            n=n,
            mae=float("nan"),
            rmse=float("nan"),
            bias=float("nan"),
            spearman_rho=float("nan"),
            levels=(),
            levels_requested=n_bins,
            n_records=n_records,
            n_other_type=n_other_type,
            n_unlabeled=n_unlabeled,
            n_unparseable=n_unparseable,
        )
    return ScoreMetrics(
        n=n,
        mae=_finite_or_nan(mean_absolute_error(predicted, actual)),
        rmse=_finite_or_nan(root_mean_squared_error(predicted, actual)),
        bias=_finite_or_nan(mean_signed_error(predicted, actual)),
        spearman_rho=spearman_rho(predicted, actual),
        levels=score_levels(predicted, actual, n_bins=n_bins),
        levels_requested=n_bins,
        n_records=n_records,
        n_other_type=n_other_type,
        n_unlabeled=n_unlabeled,
        n_unparseable=n_unparseable,
    )
