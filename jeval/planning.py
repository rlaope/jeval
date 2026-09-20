"""Label-needed planning: project how many more human labels a scope needs.

The projection rests on exactly one assumption, carried on every plan this module returns:

    the width of the bootstrap ECE interval shrinks like ``k / sqrt(n)``

``k`` is not known, so it is fitted from two measurements of the same scope: the observed
interval width at the full sample of ``n`` gold-labeled records, and the observed width at the
first ``n // 2`` of them. Because the relation is multiplicative, ``k`` is fitted in log space
-- the geometric mean of the two per-point constants ``width_i * sqrt(n_i)`` -- and the
projection inverts the same relation to get ``n_needed = (k / target) ** 2``.

Every other number on a plan is a measurement of the records as they are (``n_now``, ``ece``,
``ci_width``). The projected counts are extrapolation from two points under an assumed scaling
law, so they are reported with the assumption attached and are never presented as measurements.
A scope with fewer than :data:`MIN_LABELS_FOR_PROJECTION` gold-labeled records returns
``labels_for_target=None`` and a ``reason``: fitting a scaling law to a handful of records would
produce a number, not a projection.

What counts as a label: only gold labels (``human_review`` / ``human_override``) feed the
interval, because agreement with a model is not accuracy and a silver label does not shrink the
interval. ``score`` records carry no binary calibration point and are excluded from the fit, as
they are everywhere else in jeval.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from jeval.calibration import (
    DEFAULT_ALPHA,
    DEFAULT_BOOTSTRAP_SAMPLES,
    CalibrationMetrics,
    compute_calibration,
)
from jeval.schema import DecisionRecord

Scope = Literal["question", "segment"]

DEFAULT_TARGET_CI = 0.05

# A scaling law fitted from two points needs enough records under each point for the bootstrap
# interval to mean anything. Below this, the honest answer is "not enough labels to say".
ABSURD_LABEL_COUNT = 1e9
"""A projection past this is arithmetic, not advice: nobody labels a billion decisions."""

MIN_LABELS_FOR_PROJECTION = 200

# Widths reported per scope, as fractions of the measured width, besides the caller's target_ci.
TARGET_WIDTH_FRACTIONS: tuple[float, ...] = (0.5, 0.25)

ASSUMPTION = (
    "Estimate, not a measurement: assumes the bootstrap ECE interval width scales as k/sqrt(n), "
    "with k fitted from the observed width at n and at the first n/2 records of this scope."
)


@dataclass(frozen=True)
class LabelPlan:
    """What one scope (a question, or one value of one segment field) needs to be measured.

    ``labels_for_target`` maps a target interval width to the *additional* gold labels the
    scope needs, rounded up and never negative. It is ``None`` -- never an empty tuple, never a
    guess -- when the scope cannot support the fit, and ``reason`` then says why.

    ``key`` / ``value`` identify the scope: for ``scope="question"`` the key is the question key
    and the value is its question type; for ``scope="segment"`` the key is the segment field and
    the value is the segment value. ``fit_k`` and ``n_half`` record the fit the projection came
    from, so a consumer can see how far the two sample points were apart.
    """

    scope: Scope
    key: str
    value: str
    n_now: int
    ece: float
    ci_width: float
    labels_for_target: tuple[tuple[float, int], ...] | None
    assumption: str
    reason: str = ""
    fit_k: float | None = None
    n_half: int = 0


def _calibration_points(records: Sequence[DecisionRecord]) -> tuple[list[float], list[bool]]:
    """``(confidence, correct)`` pairs of the labels that can be measured for this scope.

    Order is preserved, so "the first half of the sample" is the first half of the records as
    they were passed in. Silver labels are dropped on purpose: a silver label is not ground
    truth, so it does not narrow the interval this projection is fitted from.
    """
    confidences: list[float] = []
    correct: list[bool] = []
    for record in records:
        if record.is_silver:
            continue
        point = record.calibration_point()
        if point is None:
            continue
        confidences.append(point[0])
        correct.append(point[1])
    return confidences, correct


def _measure(
    confidences: Sequence[float], correct: Sequence[bool], alpha: float, seed: int
) -> CalibrationMetrics:
    return compute_calibration(
        confidences,
        correct,
        alpha=alpha,
        seed=seed,
        n_boot=DEFAULT_BOOTSTRAP_SAMPLES,
    )


def _fit_width_constant(n: int, width: float, n_half: int, width_half: float) -> float | None:
    """Fit ``k`` in ``width = k / sqrt(n)`` from the full-sample and half-sample widths.

    The two points are combined in log space, which is the natural space for a multiplicative
    model and keeps a single noisy half-sample from dominating the fit. ``None`` means the two
    measurements cannot support a fit at all (a missing, zero, or non-finite width).
    """
    if n <= 0 or n_half <= 0:
        return None
    if not (math.isfinite(width) and math.isfinite(width_half)):
        return None
    if width <= 0.0 or width_half <= 0.0:
        return None
    fitted = math.sqrt((width * math.sqrt(n)) * (width_half * math.sqrt(n_half)))
    if not math.isfinite(fitted) or fitted <= 0.0:
        return None
    return fitted


def _target_widths(ci_width: float, target_ci: float) -> tuple[float, ...]:
    """The widths one plan reports: the caller's target, plus fractions of the measured width."""
    widths = [target_ci]
    if math.isfinite(ci_width) and ci_width > 0.0:
        widths.extend(ci_width * fraction for fraction in TARGET_WIDTH_FRACTIONS)
    return tuple(sorted(set(widths)))


def _labels_needed(k: float, target: float, n_now: int) -> int | None:
    """Additional labels to reach ``target``, or ``None`` when the fit gives an absurd number."""
    if target <= 0.0 or k <= 0.0:
        return None
    # Decided in log space: `(k / target) ** 2` raises OverflowError long before it returns
    # inf, so a tiny target crashed the command instead of being reported as unprojectable.
    log_needed = 2.0 * (math.log(k) - math.log(target))
    if log_needed > math.log(ABSURD_LABEL_COUNT):
        return None
    n_needed = math.exp(log_needed)
    if not math.isfinite(n_needed):
        return None
    return max(0, math.ceil(n_needed) - n_now)


def _unprojectable(
    scope: Scope, key: str, value: str, metrics: CalibrationMetrics, n_now: int, reason: str
) -> LabelPlan:
    """A plan that reports the measurements but refuses to invent a label count."""
    return LabelPlan(
        scope=scope,
        key=key,
        value=value,
        n_now=n_now,
        ece=metrics.ece,
        ci_width=metrics.ece_ci_span,
        labels_for_target=None,
        assumption=ASSUMPTION,
        reason=reason,
    )


def _plan_scope(
    scope: Scope,
    key: str,
    value: str,
    records: Sequence[DecisionRecord],
    *,
    target_ci: float,
    alpha: float,
    seed: int,
) -> LabelPlan:
    """Measure one scope, then project its label needs under the stated assumption."""
    confidences, correct = _calibration_points(records)
    n_now = len(confidences)
    metrics = _measure(confidences, correct, alpha, seed)
    if n_now < MIN_LABELS_FOR_PROJECTION:
        return _unprojectable(
            scope,
            key,
            value,
            metrics,
            n_now,
            reason=(
                f"only {n_now} gold-labeled records; at least {MIN_LABELS_FOR_PROJECTION} are "
                "needed before the k/sqrt(n) scaling can be fitted"
            ),
        )

    n_half = n_now // 2
    half = _measure(confidences[:n_half], correct[:n_half], alpha, seed)
    fit_k = _fit_width_constant(n_now, metrics.ece_ci_span, n_half, half.ece_ci_span)
    if fit_k is None:
        return _unprojectable(
            scope,
            key,
            value,
            metrics,
            n_now,
            reason=(
                "the bootstrap ECE interval was not usable at n or at n/2 "
                "(width <= 0 or not finite), so no scaling constant could be fitted"
            ),
        )

    projected: list[tuple[float, int]] = []
    for target in _target_widths(metrics.ece_ci_span, target_ci):
        count = _labels_needed(fit_k, target, n_now)
        if count is None:
            return _unprojectable(
                scope,
                key,
                value,
                metrics,
                n_now,
                reason="the fitted scaling produced a non-finite label count for a target width",
            )
        projected.append((target, count))

    return LabelPlan(
        scope=scope,
        key=key,
        value=value,
        n_now=n_now,
        ece=metrics.ece,
        ci_width=metrics.ece_ci_span,
        labels_for_target=tuple(projected),
        assumption=ASSUMPTION,
        fit_k=fit_k,
        n_half=n_half,
    )


def plan_labels(
    records: Sequence[DecisionRecord],
    *,
    target_ci: float = DEFAULT_TARGET_CI,
    by: Sequence[str] = (),
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> tuple[LabelPlan, ...]:
    """Plan the labels each scope needs for a target interval width.

    One plan is returned per question key, plus one per value of each segment field named in
    ``by``. Scopes are never dropped for being small: a scope below
    :data:`MIN_LABELS_FOR_PROJECTION` is still returned, with its measurements and a ``reason``
    in place of a label count, because "this segment has no labels at all" is the most useful
    thing a labeling plan can say about it.

    The method and its one assumption are documented in the module docstring; the assumption
    travels on every returned plan in its ``assumption`` field.
    """
    if target_ci <= 0.0:
        raise ValueError("target_ci must be > 0")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    segment_keys = tuple(dict.fromkeys(by))
    by_question: dict[str, list[DecisionRecord]] = {}
    by_segment: dict[str, dict[str, list[DecisionRecord]]] = {key: {} for key in segment_keys}
    for record in records:
        by_question.setdefault(record.question_key, []).append(record)
        for segment_key in segment_keys:
            value = record.segment.get(segment_key, "unknown")
            by_segment[segment_key].setdefault(value, []).append(record)

    plans: list[LabelPlan] = []
    for question_key in sorted(by_question):
        rows = by_question[question_key]
        plans.append(
            _plan_scope(
                "question",
                question_key,
                rows[0].question_type,
                rows,
                target_ci=target_ci,
                alpha=alpha,
                seed=seed,
            )
        )
    for segment_key in segment_keys:
        for value in sorted(by_segment[segment_key]):
            plans.append(
                _plan_scope(
                    "segment",
                    segment_key,
                    value,
                    by_segment[segment_key][value],
                    target_ci=target_ci,
                    alpha=alpha,
                    seed=seed,
                )
            )
    return tuple(plans)
