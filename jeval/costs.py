"""Cost-matrix thresholds: turn what a mistake costs into where a human should take over.

An action is one decision jeval can set a confidence threshold for. It names a question, the
class value of that question it fires on, and three costs:

- ``cost_false_accept``: the action ran automatically and its answer was wrong
- ``cost_escalate``: the decision went to a human instead of running automatically
- ``cost_false_reject``: the case was a ``when`` case and still went to a human

For one action at one threshold, every *gold*-labeled record of the action's question is scored:

- it ran automatically when ``prediction == when and confidence >= threshold``
- a record that ran costs ``0.0`` when its label equals ``when``, else ``cost_false_accept``
- a record that did not run costs ``cost_escalate``, plus ``cost_false_reject`` when its label
  equals ``when``

``expected_cost`` is the mean of those costs over the question's gold-labeled records, so it
reads as a cost per case. ``auto_rate`` is the share of cases the machine handled alone, and
``accuracy_auto`` is the share of those cases it got right.

Two rules keep the output honest rather than merely plausible:

- **Silver labels are never used.** Agreement with another model is not accuracy, so a
  threshold derived from silver labels would be a threshold derived from something else.
- **Thirty gold records or no number.** Below :data:`MIN_GOLD_RECORDS` the sweep returns ``nan``
  and sets ``n_records``, so the caller reports insufficient data instead of inventing a
  threshold from a handful of cases.

Every function here is pure and deterministic given its inputs: the sweep is exhaustive over a
fixed grid, the bootstrap draws from a seeded generator, and nothing is read from the network.

Segments
--------

:func:`sweep_by_segment` answers the follow-up question a report raises as soon as its segments
disagree -- one threshold per segment -- and it is allowed to answer *no*. Each segment is swept
on its own gold records and compared with the global optimum, and adopting the segment's own
threshold only counts as worth it when the two optima are more than one sweep step apart *and*
the change in cost per case exceeds :data:`SPLIT_COST_TOLERANCE` of the global cost per case. A
segment that cannot be swept comes back with ``nan`` figures and a ``reason``, never as a missing
row: a segment silently dropped from the answer is indistinguishable from one that was measured
and found uninteresting.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from jeval.calibration import DEFAULT_ALPHA
from jeval.currency import format_amount
from jeval.report.model import CostPoint, ImpactRow, ImpactTable, ThresholdResult
from jeval.schema import DecisionRecord

__all__ = [
    "DEFAULT_FLAT_TOLERANCE",
    "DEFAULT_MIN_SEGMENT_RECORDS",
    "DEFAULT_N_BOOT",
    "DEFAULT_STEPS",
    "MIN_GOLD_RECORDS",
    "NOT_WORTH_SPLITTING",
    "SEGMENT_UNKNOWN",
    "SPLIT_COST_TOLERANCE",
    "CostAction",
    "SegmentThreshold",
    "bootstrap_threshold_ci",
    "build_impact",
    "evaluate_point",
    "flat_region",
    "load_cost_actions",
    "sweep",
    "sweep_by_segment",
    "write_thresholds_yaml",
]

DEFAULT_STEPS = 101
"""Sweep points from 0.0 to 1.0 inclusive, so one step is 0.01 of confidence."""

DEFAULT_MIN_SEGMENT_RECORDS = 100
"""Gold records a segment needs before it is swept on its own instead of being reported as thin."""

SPLIT_COST_TOLERANCE = 0.02
"""A segment pays off only if its own threshold moves cost per case by more than this share."""

NOT_WORTH_SPLITTING = "splitting does not pay"
"""Every reason of a segment that was swept and should keep the global threshold starts here."""

SEGMENT_UNKNOWN = "unknown"
"""Segment value of a record that does not carry the segment key at all."""

DEFAULT_N_BOOT = 200
"""Bootstrap resamples for the threshold interval."""

DEFAULT_FLAT_TOLERANCE = 0.05
"""A threshold counts as equally good within five percent of the best cost."""

MIN_GOLD_RECORDS = 30
"""Below this many gold-labeled records an action has no threshold, only a note saying so."""

_ACTION_KEYS = frozenset(
    {
        "name",
        "question",
        "when",
        "cost_false_accept",
        "cost_escalate",
        "cost_false_reject",
    }
)

_NAN = float("nan")
_STEP_EPSILON = 1e-12


@dataclass(frozen=True)
class CostAction:
    """One automated decision with the costs of getting it wrong, or of not running it."""

    name: str
    question: str
    when: str
    cost_false_accept: float
    cost_escalate: float
    cost_false_reject: float

    def __post_init__(self) -> None:
        """Refuse a cost the arithmetic cannot carry: a NaN won every comparison (all of them are
        False), so the first grid point was reported as a measured threshold."""
        for name in ("cost_false_accept", "cost_escalate", "cost_false_reject"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{self.name}: {name} must be a finite number, got {value!r}")
            if value < 0:
                raise ValueError(f"{self.name}: {name} must not be negative, got {value!r}")

    @property
    def label(self) -> str:
        """``name`` with its trigger, for messages that must say what is being measured."""
        return f"{self.name} ({self.question} == {self.when})"


def load_cost_actions(path: Path | str) -> list[CostAction]:
    """Load a cost matrix from a ``costs.yaml``-shaped file.

    The file is a YAML mapping with an ``actions`` list, each entry carrying exactly the six
    fields of :class:`CostAction`::

        actions:
          - name: auto_refund
            question: intent
            when: refund_request
            cost_false_accept: 50000   # it ran automatically and was wrong
            cost_escalate: 2000        # a human handled it
            cost_false_reject: 0       # it was right and a human handled it anyway

    Malformed input is an exception, never a default, because a silently dropped or defaulted
    cost figure produces a confidently wrong threshold. A missing file raises
    :class:`FileNotFoundError`; everything else raises :class:`ValueError` naming the file and
    the offending entry: not a mapping, no ``actions`` list, an entry that is not a mapping,
    a missing or unknown field, a non-numeric, non-finite or negative cost, or a duplicate
    action name (``thresholds.yaml`` is keyed by action name, so names must be unique).
    Unknown top-level keys are ignored so the file can carry comments and metadata.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"{source}: cost matrix not found")
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:  # pragma: no cover - message text is pyyaml's
        raise ValueError(f"{source}: not valid YAML: {error}") from error
    if loaded is None:
        raise ValueError(f"{source}: empty cost matrix: expected a top-level 'actions' list")
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{source}: cost matrix must be a YAML mapping with an 'actions' key")
    raw_actions = loaded.get("actions")
    if not isinstance(raw_actions, list):
        raise ValueError(f"{source}: 'actions' must be a list of cost entries")

    actions: list[CostAction] = []
    seen: dict[str, int] = {}
    for index, entry in enumerate(raw_actions):
        where = f"{source}: actions[{index}]"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{where}: each action must be a mapping of cost fields")
        keys = {str(key) for key in entry}
        missing = sorted(_ACTION_KEYS - keys)
        if missing:
            raise ValueError(f"{where}: missing cost field(s): {', '.join(missing)}")
        unknown = sorted(keys - _ACTION_KEYS)
        if unknown:
            expected = ", ".join(sorted(_ACTION_KEYS))
            raise ValueError(
                f"{where}: unknown cost field(s): {', '.join(unknown)}; expected {expected}"
            )
        name = _text(entry, "name", where)
        if name in seen:
            raise ValueError(
                f"{where}: duplicate action name {name!r}, already defined at actions[{seen[name]}]"
            )
        seen[name] = index
        actions.append(
            CostAction(
                name=name,
                question=_text(entry, "question", where),
                when=_text(entry, "when", where),
                cost_false_accept=_cost(entry, "cost_false_accept", where),
                cost_escalate=_cost(entry, "cost_escalate", where),
                cost_false_reject=_cost(entry, "cost_false_reject", where),
            )
        )
    return actions


def _text(entry: Mapping[Any, Any], key: str, where: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: {key} must be a non-empty string")
    return value.strip()


def _cost(entry: Mapping[Any, Any], key: str, where: str) -> float:
    value = entry.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: {key} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{where}: {key} must be a finite number, got {value!r}")
    if number < 0.0:
        raise ValueError(f"{where}: {key} must not be negative, got {value!r}")
    return number


@dataclass(frozen=True)
class _Sample:
    """One action's gold records as parallel arrays, ready to score at any threshold."""

    confidence: NDArray[np.float64]
    correct: NDArray[np.bool_]
    label_is_when: NDArray[np.bool_]
    predicts_when: NDArray[np.bool_]

    @property
    def n(self) -> int:
        return int(self.confidence.size)

    def take(self, index: NDArray[np.int64]) -> _Sample:
        """A resample of these records, used by the bootstrap."""
        return _Sample(
            confidence=self.confidence[index],
            correct=self.correct[index],
            label_is_when=self.label_is_when[index],
            predicts_when=self.predicts_when[index],
        )


@dataclass(frozen=True)
class _CostSeries:
    """The cost figures of one action over every threshold of a sweep, as arrays."""

    cost: NDArray[np.float64]
    n_auto: NDArray[np.int64]
    n_escalate: NDArray[np.int64]
    wrong_auto: NDArray[np.int64]
    wrong_escalate: NDArray[np.int64]
    accept_cost: NDArray[np.float64]
    escalate_cost: NDArray[np.float64]


def _gold_records(action: CostAction, records: Sequence[DecisionRecord]) -> list[DecisionRecord]:
    """The action's question's gold-labeled records, in input order.

    Silver and unlabeled records are dropped. ``n_records`` counts exactly what is returned
    here, so a caller can compare it against the log's own row count and see the difference.
    """
    return [
        record
        for record in records
        if record.question_key == action.question and record.is_gold and record.label is not None
    ]


def _sample(action: CostAction, gold: Sequence[DecisionRecord]) -> _Sample:
    return _Sample(
        confidence=np.asarray([record.confidence for record in gold], dtype=np.float64),
        correct=np.asarray([bool(record.is_correct) for record in gold], dtype=np.bool_),
        label_is_when=np.asarray([record.label == action.when for record in gold], dtype=np.bool_),
        predicts_when=np.asarray(
            [record.prediction == action.when for record in gold], dtype=np.bool_
        ),
    )


def _series(action: CostAction, sample: _Sample, thresholds: NDArray[np.float64]) -> _CostSeries:
    """Score one action at every threshold in one vectorized pass.

    ``wrong_auto`` and ``wrong_escalate`` both count decisions whose prediction differed from the
    gold label, so ``accuracy_auto`` is the machine's accuracy on what it ran alone and
    ``accuracy_escalated`` its accuracy on what a human took. They are not the same as the cost
    terms: ``cost_false_reject`` is charged whenever a ``when`` case was escalated, whether or
    not the model had predicted ``when``.
    """
    n = sample.n
    if n == 0:
        raise ValueError("cannot score an action over zero records")
    auto = sample.predicts_when[:, None] & (sample.confidence[:, None] >= thresholds[None, :])
    wrong = ~sample.correct[:, None]
    when_label = sample.label_is_when[:, None]
    escalated = ~auto

    n_auto = np.asarray(auto.sum(axis=0), dtype=np.int64)
    wrong_auto = np.asarray((auto & wrong).sum(axis=0), dtype=np.int64)
    wrong_escalate = np.asarray((escalated & wrong).sum(axis=0), dtype=np.int64)
    rejected = np.asarray((escalated & when_label).sum(axis=0), dtype=np.int64)
    n_escalate = n - n_auto

    accept_cost = np.asarray(wrong_auto, dtype=np.float64) * action.cost_false_accept
    escalate_cost = (
        np.asarray(n_escalate, dtype=np.float64) * action.cost_escalate
        + np.asarray(rejected, dtype=np.float64) * action.cost_false_reject
    )
    return _CostSeries(
        cost=(accept_cost + escalate_cost) / float(n),
        n_auto=n_auto,
        n_escalate=n_escalate,
        wrong_auto=wrong_auto,
        wrong_escalate=wrong_escalate,
        accept_cost=accept_cost,
        escalate_cost=escalate_cost,
    )


def _share(hits: int, total: int) -> float:
    """A proportion, or ``nan`` when there is no denominator to divide by."""
    return hits / total if total > 0 else _NAN


def _points_from(series: _CostSeries, thresholds: NDArray[np.float64]) -> tuple[CostPoint, ...]:
    points: list[CostPoint] = []
    for index, threshold in enumerate(thresholds):
        n_auto = int(series.n_auto[index])
        n_escalate = int(series.n_escalate[index])
        total = n_auto + n_escalate
        points.append(
            CostPoint(
                threshold=float(threshold),
                expected_cost=float(series.cost[index]),
                auto_rate=n_auto / total if total > 0 else _NAN,
                accuracy_auto=_share(n_auto - int(series.wrong_auto[index]), n_auto),
                accuracy_escalated=_share(
                    n_escalate - int(series.wrong_escalate[index]), n_escalate
                ),
                accept_cost=float(series.accept_cost[index]),
                escalate_cost=float(series.escalate_cost[index]),
                n_auto=n_auto,
                n_escalate=n_escalate,
                n_wrong_auto=int(series.wrong_auto[index]),
                n_wrong_escalate=int(series.wrong_escalate[index]),
            )
        )
    return tuple(points)


def _thresholds(steps: int) -> NDArray[np.float64]:
    """The sweep grid: ``steps`` points from 0.0 to 1.0 inclusive, rounded to stay exact."""
    return np.round(np.linspace(0.0, 1.0, steps), 12)


def _argmin_high(costs: NDArray[np.float64]) -> int:
    """Index of the lowest cost, preferring the highest threshold among exact ties.

    Iterating upward and accepting ``<=`` keeps the last of a run of equal costs, which is the
    conservative choice: when two thresholds cost the same, the higher one escalates more.
    """
    best = 0
    for index in range(1, int(costs.size)):
        if bool(costs[index] <= costs[best]):
            best = index
    return best


def _empty_point(threshold: float) -> CostPoint:
    """A point for a question with no gold-labeled records: counts zero, figures ``nan``."""
    return CostPoint(
        threshold=threshold,
        expected_cost=_NAN,
        auto_rate=_NAN,
        accuracy_auto=_NAN,
        accuracy_escalated=_NAN,
        accept_cost=0.0,
        escalate_cost=0.0,
        n_auto=0,
        n_escalate=0,
        n_wrong_auto=0,
        n_wrong_escalate=0,
    )


def evaluate_point(
    action: CostAction, records: Sequence[DecisionRecord], threshold: float
) -> CostPoint:
    """Score one action at one threshold on the observed gold-labeled records.

    The rule is the one documented at module level: a record runs automatically when the model
    predicted the action's ``when`` class *and* its confidence reached ``threshold``; running a
    case whose label is not ``when`` costs ``cost_false_accept``; everything else costs
    ``cost_escalate``, plus ``cost_false_reject`` when the case's label was ``when`` anyway.
    ``expected_cost`` is the mean over every gold-labeled record of the action's question.

    Silver-labeled and unlabeled records are ignored. A question with no gold-labeled records
    yields a point whose figures are ``nan`` and whose counts are zero, so nothing downstream
    can mistake absence of data for a measured zero.

    Unlike :func:`sweep` this does not apply the :data:`MIN_GOLD_RECORDS` floor: it reports what
    the records say at a threshold the caller chose, which is exactly what is wanted to show
    where a *current* threshold sits while the recommendation is still withheld.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    gold = _gold_records(action, records)
    if not gold:
        return _empty_point(threshold)
    sample = _sample(action, gold)
    grid = np.asarray([threshold], dtype=np.float64)
    return _points_from(_series(action, sample, grid), grid)[0]


def _bootstrap(
    action: CostAction,
    sample: _Sample,
    *,
    steps: int,
    n_boot: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    """Percentile bootstrap over records, re-choosing the threshold on every resample."""
    thresholds = _thresholds(steps)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=np.float64)
    for draw in range(n_boot):
        index = rng.integers(0, sample.n, size=sample.n)
        series = _series(action, sample.take(index), thresholds)
        estimates[draw] = thresholds[_argmin_high(series.cost)]
    return (
        float(np.quantile(estimates, alpha / 2.0)),
        float(np.quantile(estimates, 1.0 - alpha / 2.0)),
    )


def bootstrap_threshold_ci(
    action: CostAction,
    records: Sequence[DecisionRecord],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the recommended threshold.

    Each resample draws ``n`` records with replacement from the action's gold-labeled records
    and re-runs the whole sweep on that resample, so the interval reflects how much the
    *recommendation* moves under sampling noise rather than the cost at a fixed threshold. The
    interval is the 2.5th and 97.5th percentile of the resampled thresholds, on the default
    sweep grid (see :data:`DEFAULT_STEPS`).

    Returns ``(nan, nan)`` when there are fewer than :data:`MIN_GOLD_RECORDS` gold records --
    the same floor that makes :func:`sweep` refuse to recommend a number -- and when
    ``n_boot <= 0``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    gold = _gold_records(action, records)
    if len(gold) < MIN_GOLD_RECORDS or n_boot <= 0:
        return (_NAN, _NAN)
    return _bootstrap(
        action,
        _sample(action, gold),
        steps=DEFAULT_STEPS,
        n_boot=n_boot,
        alpha=alpha,
        seed=seed,
    )


def flat_region(
    curve: Sequence[CostPoint], *, tolerance: float = DEFAULT_FLAT_TOLERANCE
) -> tuple[float, float] | None:
    """The contiguous threshold interval that is as good as the best one.

    A curve point belongs to the region when its expected cost is at most
    ``min_cost * (1 + tolerance)``, and the region is the run of points around the chosen
    minimum -- the highest threshold among tied minima, matching :func:`sweep`. A second, cheaper
    run elsewhere on the curve therefore cannot widen the recommendation; the point is to say
    how far the threshold can drift without changing what it costs.

    Returns ``None`` when the region is narrower than one sweep step, because one point is not a
    region, and when the curve is empty or any cost is not finite. Costs are assumed to be
    non-negative, which :func:`load_cost_actions` enforces.
    """
    if tolerance < 0.0:
        raise ValueError(f"tolerance must be >= 0, got {tolerance}")
    points = sorted(curve, key=lambda point: point.threshold)
    if not points:
        return None
    costs = np.asarray([point.expected_cost for point in points], dtype=np.float64)
    if not bool(np.all(np.isfinite(costs))):
        return None
    best = _argmin_high(costs)
    limit = float(costs[best]) * (1.0 + tolerance)
    low = best
    high = best
    while low > 0 and float(costs[low - 1]) <= limit:
        low -= 1
    while high + 1 < len(points) and float(costs[high + 1]) <= limit:
        high += 1
    step = _sweep_step([point.threshold for point in points])
    if step is None:
        return None
    if points[high].threshold - points[low].threshold < step - _STEP_EPSILON:
        return None
    return (points[low].threshold, points[high].threshold)


def _sweep_step(thresholds: Sequence[float]) -> float | None:
    """Smallest positive gap between consecutive thresholds, or ``None`` when there is none."""
    gaps = [right - left for left, right in pairwise(thresholds) if right > left]
    return min(gaps) if gaps else None


def sweep(
    action: CostAction,
    records: Sequence[DecisionRecord],
    *,
    steps: int = DEFAULT_STEPS,
    alpha: float = DEFAULT_ALPHA,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> ThresholdResult:
    """Find the cost-minimizing threshold for one action, with its interval and flat region.

    The sweep evaluates ``steps`` thresholds from 0.0 to 1.0 inclusive and takes the argmin of
    expected cost, breaking exact ties toward the **higher** threshold: where two thresholds cost
    the same, the one that escalates more is the conservative choice. The interval is the
    percentile bootstrap of :func:`bootstrap_threshold_ci` on the same grid, and the flat region
    is :func:`flat_region` at :data:`DEFAULT_FLAT_TOLERANCE`.

    Below :data:`MIN_GOLD_RECORDS` gold-labeled records the recommendation is withheld: the
    result carries ``nan`` for threshold, cost, rates and interval, with ``n_records``,
    ``models`` and ``silver_only`` filled in so the caller can say what is missing.
    """
    if steps < 2:
        raise ValueError(f"steps must be >= 2 to sweep a range, got {steps}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    gold = _gold_records(action, records)
    n_records = len(gold)
    models = tuple(sorted({record.model for record in gold}))
    silver_only = n_records == 0 and _has_silver(action, records)
    if n_records < MIN_GOLD_RECORDS:
        return ThresholdResult(
            action=action.name,
            question=action.question,
            when=action.when,
            threshold=_NAN,
            expected_cost_per_case=_NAN,
            auto_rate=_NAN,
            accuracy_auto=_NAN,
            ci_low=_NAN,
            ci_high=_NAN,
            curve=(),
            flat_region=None,
            n_records=n_records,
            models=models,
            cost_false_accept=action.cost_false_accept,
            cost_escalate=action.cost_escalate,
            cost_false_reject=action.cost_false_reject,
            silver_only=silver_only,
        )
    sample = _sample(action, gold)
    grid = _thresholds(steps)
    series = _series(action, sample, grid)
    curve = _points_from(series, grid)
    best = curve[_argmin_high(series.cost)]
    ci_low, ci_high = _bootstrap(action, sample, steps=steps, n_boot=n_boot, alpha=alpha, seed=seed)
    region = flat_region(curve)
    if region is not None and region[0] <= 0.0 and region[1] >= 1.0:
        # Every threshold costs the same, so the sweep cannot separate them and a "recommendation"
        # is the documented tie-break rather than a finding. The interval goes with it, because a
        # zero-width interval printed beside a flat region contradicts the region it sits next to.
        ci_low = ci_high = _NAN
    # A narrow-but-real interval is left alone: every resample agreeing on one threshold is
    # information about stability, not a claim of exactness, and the flat region is printed with
    # it.
    return ThresholdResult(
        action=action.name,
        question=action.question,
        when=action.when,
        threshold=best.threshold,
        expected_cost_per_case=best.expected_cost,
        auto_rate=best.auto_rate,
        accuracy_auto=best.accuracy_auto,
        ci_low=ci_low,
        ci_high=ci_high,
        curve=curve,
        flat_region=region,
        n_records=n_records,
        models=models,
        cost_false_accept=action.cost_false_accept,
        cost_escalate=action.cost_escalate,
        cost_false_reject=action.cost_false_reject,
        silver_only=silver_only,
    )


def _has_silver(action: CostAction, records: Sequence[DecisionRecord]) -> bool:
    """Whether the question has silver labels but no gold ones: agreement, not measurement."""
    return any(record.question_key == action.question and record.is_silver for record in records)


@dataclass(frozen=True)
class SegmentThreshold:
    """One segment's own cost-optimal threshold, and whether adopting it pays.

    ``threshold``, ``expected_cost_per_case`` and ``auto_rate`` are measured on this segment's
    gold records alone and are ``nan`` when the segment could not be swept, in which case
    ``reason`` says why. ``cost_delta_vs_global`` is the signed change in cost per case *inside
    the segment* from replacing the global threshold with the segment's own: zero or negative
    whenever the segment was swept, since the segment's optimum is the cheapest point of its own
    curve. ``reason`` is empty exactly when the segment was swept and splitting pays, and starts
    with :data:`NOT_WORTH_SPLITTING` whenever the segment was swept and does not.
    """

    segment_key: str
    segment_value: str
    threshold: float
    expected_cost_per_case: float
    auto_rate: float
    n_records: int
    cost_delta_vs_global: float
    worth_splitting: bool
    reason: str = ""

    @property
    def label(self) -> str:
        """``segment_key = segment_value``, for messages that must say what was measured."""
        return f"{self.segment_key} = {self.segment_value}"


def sweep_by_segment(
    action: CostAction,
    records: Sequence[DecisionRecord],
    *,
    segment_key: str,
    min_records: int = DEFAULT_MIN_SEGMENT_RECORDS,
    steps: int = DEFAULT_STEPS,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[SegmentThreshold, ...]:
    """Sweep one action per segment value and say, per segment, whether splitting pays.

    The global line comes from :func:`sweep` on the action's gold records. Each segment is then
    swept on its own records with exactly the same rule and compared with that line twice:

    - **movement**: the segment's optimum differs from the global one by more than one sweep step
      (``1 / (steps - 1)``), so the two genuinely disagree about where the line goes; a single
      step apart is grid noise.
    - **materiality**: adopting the segment's own threshold instead of the global one changes
      cost per case by more than :data:`SPLIT_COST_TOLERANCE` of the global cost per case.

    Both are required before ``worth_splitting`` becomes true. A segment can move the threshold
    without moving money, and it can move money without moving the threshold; either way the
    report should keep one line. ``cost_delta_vs_global`` is that change, signed, and is zero or
    negative for every swept segment.

    Segments are the distinct ``segment[segment_key]`` values of the action's gold records, in
    sorted order. A record that does not carry the key at all is counted under
    :data:`SEGMENT_UNKNOWN` rather than dropped, so the segments keep adding up to the global
    figure. Records of other questions and silver-labeled records belong to no segment, exactly
    as in :func:`sweep`.

    A segment that cannot be swept is returned with ``nan`` figures and a ``reason``: fewer than
    ``min_records`` gold records (and never fewer than :data:`MIN_GOLD_RECORDS`, whatever the
    caller asks for), or no case on either side of ``when`` -- if nothing is predicted as the
    action's class no threshold routes anything, and if nothing is labeled as it there is no case
    the action was meant to decide. Segments are never omitted, and a segment below the floor
    keeps its ``n_records`` so the caller can see how thin it is.
    """
    if not isinstance(segment_key, str) or not segment_key.strip():
        raise ValueError("segment_key must be a non-empty string")
    # Clamped rather than refused, which is what the docstring promises: a sub-floor request is
    # raised to the module's own floor.
    min_records = max(1, int(min_records))
    if steps < 2:
        raise ValueError(f"steps must be >= 2 to sweep a range, got {steps}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    grouped: dict[str, list[DecisionRecord]] = {}
    for record in _gold_records(action, records):
        grouped.setdefault(record.segment.get(segment_key, SEGMENT_UNKNOWN), []).append(record)

    floor = max(min_records, MIN_GOLD_RECORDS)
    step = 1.0 / float(steps - 1)
    global_result = sweep(action, records, steps=steps, alpha=alpha)

    results: list[SegmentThreshold] = []
    for value in sorted(grouped):
        segment = grouped[value]
        reason = _unsweepable_reason(action, segment, floor=floor)
        if reason is None and not math.isfinite(global_result.threshold):
            # Unreachable while the floor is at least MIN_GOLD_RECORDS, since the global sweep
            # sees at least as many gold records as any one of its segments. Kept so a segment
            # can never be compared against a global threshold that does not exist.
            reason = (
                f"the action's {global_result.n_records} gold records are below "
                f"{MIN_GOLD_RECORDS}, so there is no global threshold to split from"
            )
        if reason is not None:
            results.append(_unswept_segment(segment_key, value, len(segment), reason))
            continue
        results.append(
            _swept_segment(
                action,
                segment,
                segment_key=segment_key,
                segment_value=value,
                steps=steps,
                alpha=alpha,
                global_result=global_result,
                step=step,
            )
        )
    return tuple(results)


def _unsweepable_reason(
    action: CostAction, segment: Sequence[DecisionRecord], *, floor: int
) -> str | None:
    """Why this segment cannot carry a threshold of its own, or ``None`` when it can."""
    if len(segment) < floor:
        return (
            f"only {len(segment)} gold records, below the {floor} a segment needs before a "
            "threshold is recommended for it"
        )
    if not any(record.prediction == action.when for record in segment):
        return (
            f"no record in this segment is predicted as the action's class {action.when!r}, so "
            "no threshold routes anything and none can be chosen"
        )
    if not any(record.label == action.when for record in segment):
        return (
            f"no record in this segment is labeled as the action's class {action.when!r}, so "
            "the segment holds no case the action was meant to decide"
        )
    return None


def _unswept_segment(segment_key: str, value: str, n_records: int, reason: str) -> SegmentThreshold:
    """The row a segment gets when it cannot be swept: counts kept, figures ``nan``, reason set."""
    return SegmentThreshold(
        segment_key=segment_key,
        segment_value=value,
        threshold=_NAN,
        expected_cost_per_case=_NAN,
        auto_rate=_NAN,
        n_records=n_records,
        cost_delta_vs_global=_NAN,
        worth_splitting=False,
        reason=reason,
    )


def _swept_segment(
    action: CostAction,
    segment: Sequence[DecisionRecord],
    *,
    segment_key: str,
    segment_value: str,
    steps: int,
    alpha: float,
    global_result: ThresholdResult,
    step: float,
) -> SegmentThreshold:
    """Sweep one segment and compare its optimum with the global line, in money and in steps."""
    result = sweep(action, segment, steps=steps, alpha=alpha)
    at_global = evaluate_point(action, segment, global_result.threshold)
    delta = result.expected_cost_per_case - at_global.expected_cost
    moved = abs(result.threshold - global_result.threshold) > step + _STEP_EPSILON
    material = abs(delta) > SPLIT_COST_TOLERANCE * global_result.expected_cost_per_case
    worth_splitting = moved and material
    reason = ""
    if not worth_splitting:
        reason = _not_worth_reason(
            moved=moved,
            material=material,
            segment_threshold=result.threshold,
            global_threshold=global_result.threshold,
            cost_delta=delta,
            global_cost=global_result.expected_cost_per_case,
            step=step,
        )
    return SegmentThreshold(
        segment_key=segment_key,
        segment_value=segment_value,
        threshold=result.threshold,
        expected_cost_per_case=result.expected_cost_per_case,
        auto_rate=result.auto_rate,
        n_records=result.n_records,
        cost_delta_vs_global=delta,
        worth_splitting=worth_splitting,
        reason=reason,
    )


def _not_worth_reason(
    *,
    moved: bool,
    material: bool,
    segment_threshold: float,
    global_threshold: float,
    cost_delta: float,
    global_cost: float,
    step: float,
) -> str:
    """Why a swept segment keeps the global line, naming whichever of the two tests it failed."""
    if moved:
        threshold_clause = (
            f"its optimum {segment_threshold:.2f} differs from the global {global_threshold:.2f}"
        )
    else:
        threshold_clause = (
            f"its optimum {segment_threshold:.2f} is within one sweep step ({step:.2f}) of the "
            f"global {global_threshold:.2f}"
        )
    if material:
        share = abs(cost_delta) / global_cost if global_cost > 0.0 else _NAN
        cost_clause = (
            f"the cost change of {abs(cost_delta):.4g} per case is {share:.1%} of the global "
            f"figure, more than the {SPLIT_COST_TOLERANCE:.0%} bar on its own"
        )
    else:
        cost_clause = (
            f"the cost change of {abs(cost_delta):.4g} per case is under the "
            f"{SPLIT_COST_TOLERANCE:.0%} bar"
        )
    return f"{NOT_WORTH_SPLITTING}: {threshold_clause} and {cost_clause}."


def _fmt_threshold(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.2f}"


def _fmt_share(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.0%}"


def _fmt_points(delta: float) -> str:
    return "n/a" if not math.isfinite(delta) else f"{delta * 100.0:+.1f} pt"


def _fmt_money(value: float, currency: str) -> str:
    return format_amount(value, currency)


def _fmt_threshold_delta(recommended: float, current: float) -> str:
    if not (math.isfinite(recommended) and math.isfinite(current)):
        return "n/a"
    return f"{recommended - current:+.2f}"


def _fmt_percent_change(current: float, recommended: float) -> str:
    """Signed relative change, or ``n/a`` when it is undefined (no finite or non-zero base)."""
    if not (math.isfinite(current) and math.isfinite(recommended)):
        return "n/a"
    if abs(current) < _STEP_EPSILON:
        return "n/a"
    return f"{(recommended - current) / current * 100.0:+.1f}%"


def build_impact(
    result: ThresholdResult,
    current_threshold: float,
    monthly_volume: float | None = None,
    currency: str = "USD",
) -> ImpactTable:
    """What changes if the recommended threshold is adopted.

    Every cell is pre-formatted for the report: a threshold in confidence units, a rate or
    accuracy as a percentage, a cost with its currency, and the change column as a signed delta
    (rates and accuracy in points, cost in percent). Cells that cannot be computed -- because the
    recommendation was withheld for insufficient data, or because a branch has no records to
    divide by -- read ``n/a`` instead of a number the sample cannot support. The monthly row
    appears only when ``monthly_volume`` is given, and is cost per case times that volume.

    ``paradox_note`` explains, in one sentence, the case that otherwise looks like a mistake: the
    recommended threshold escalates more cases (auto rate falls) and still costs less per case.
    It is empty whenever that is not what the numbers show.
    """
    current = result.point_at(current_threshold)
    chosen = result.point_at(result.threshold) if math.isfinite(result.threshold) else None
    current_cost = current.expected_cost if current is not None else _NAN
    chosen_cost = chosen.expected_cost if chosen is not None else _NAN
    current_auto = current.auto_rate if current is not None else _NAN
    chosen_auto = chosen.auto_rate if chosen is not None else _NAN
    current_accuracy = current.accuracy_auto if current is not None else _NAN
    chosen_accuracy = chosen.accuracy_auto if chosen is not None else _NAN

    rows = [
        ImpactRow(
            label="confidence threshold",
            current=_fmt_threshold(current_threshold),
            recommended=_fmt_threshold(result.threshold),
            change=_fmt_threshold_delta(result.threshold, current_threshold),
        ),
        ImpactRow(
            label="auto rate",
            current=_fmt_share(current_auto),
            recommended=_fmt_share(chosen_auto),
            change=_fmt_points(chosen_auto - current_auto),
        ),
        ImpactRow(
            label="accuracy (auto)",
            current=_fmt_share(current_accuracy),
            recommended=_fmt_share(chosen_accuracy),
            change=_fmt_points(chosen_accuracy - current_accuracy),
        ),
        ImpactRow(
            label="cost per case",
            current=_fmt_money(current_cost, currency),
            recommended=_fmt_money(chosen_cost, currency),
            change=_fmt_percent_change(current_cost, chosen_cost),
        ),
    ]
    if monthly_volume is not None:
        rows.append(
            ImpactRow(
                label="monthly cost",
                current=_fmt_money(current_cost * monthly_volume, currency),
                recommended=_fmt_money(chosen_cost * monthly_volume, currency),
                change=_fmt_percent_change(current_cost, chosen_cost),
            )
        )

    note = ""
    if current is not None and chosen is not None:
        escalates_more = chosen.auto_rate < current.auto_rate
        cheaper = chosen.expected_cost < current.expected_cost
        if escalates_more and cheaper:
            note = (
                f"Auto rate falls from {current.auto_rate:.0%} to {chosen.auto_rate:.0%} while "
                f"cost per case falls from {_fmt_money(current.expected_cost, currency)} to "
                f"{_fmt_money(chosen.expected_cost, currency)}: the cases the higher threshold "
                "escalates are the ones expensive enough to be worth a human."
            )
    return ImpactTable(
        rows=tuple(rows),
        monthly_volume=monthly_volume,
        currency=currency,
        paradox_note=note,
        current_threshold=current_threshold,
        recommended_threshold=result.threshold,
    )


def _number_or_none(value: float) -> float | None:
    """``value`` rounded for a diffable file, or ``None`` when it was not measured."""
    return round(value, 6) if math.isfinite(value) else None


def write_thresholds_yaml(
    results: Sequence[ThresholdResult], path: Path | str, *, generated_at: str
) -> Path:
    """Write the recommendation file that CI and ``jeval drift`` read back.

    The shape is the header the file needs to be interpretable followed by one entry per action::

        generated_at: 2026-09-20T12:00:00Z
        model: jev-1.13.0
        n_records: 1480
        actions:
          auto_refund:
            question: intent
            when: refund_request
            threshold: 0.89
            expected_cost_per_case: 12.4
            auto_rate: 0.58
            accuracy_auto: 0.94
            ci:
              low: 0.83
              high: 0.93
            flat_region:
              low: 0.86
              high: 0.92
            models: [jev-1.13.0]
            n_records: 402
            insufficient_data: false

    ``model`` joins the distinct models seen across the results; top-level ``n_records`` is the
    sum over actions, so it double-counts records shared by several actions on purpose: it is the
    volume of evidence behind the file, not a distinct-record census.

    Numbers that were never measured are written as ``null``, not ``NaN``, so the file
    round-trips, diffs cleanly, and cannot be read as a measurement. Action names must be unique
    across ``results``: the file is keyed by action name, so a repeat would silently drop one.
    """
    actions: dict[str, dict[str, Any]] = {}
    models: set[str] = set()
    total = 0
    for result in results:
        if result.action in actions:
            raise ValueError(f"duplicate action {result.action!r} in results")
        models.update(result.models)
        total += result.n_records
        ci: dict[str, float] | None = None
        if math.isfinite(result.ci_low) and math.isfinite(result.ci_high):
            ci = {"low": round(result.ci_low, 6), "high": round(result.ci_high, 6)}
        region: dict[str, float] | None = None
        if result.flat_region is not None:
            region = {"low": result.flat_region[0], "high": result.flat_region[1]}
        actions[result.action] = {
            "question": result.question,
            "when": result.when,
            "threshold": _number_or_none(result.threshold),
            "expected_cost_per_case": _number_or_none(result.expected_cost_per_case),
            "auto_rate": _number_or_none(result.auto_rate),
            "accuracy_auto": _number_or_none(result.accuracy_auto),
            "ci": ci,
            "flat_region": region,
            "models": list(result.models),
            "n_records": result.n_records,
            "insufficient_data": not math.isfinite(result.threshold),
        }
    payload: dict[str, Any] = {
        "generated_at": str(generated_at),
        "model": ", ".join(sorted(models)),
        "n_records": total,
        "actions": actions,
    }
    target = Path(path)
    if target.parent != Path(""):
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return target
