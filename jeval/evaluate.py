"""Turn a set of decision records into a report-ready evaluation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from jeval.calibration import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_N_BINS,
    CalibrationMetrics,
    compute_calibration,
    diagnose,
)
from jeval.schema import DecisionRecord


@dataclass(frozen=True)
class QuestionReport:
    """Calibration for a single question key."""

    question_key: str
    question_type: str
    metrics: CalibrationMetrics
    n_records: int
    n_unlabeled: int
    n_score_excluded: int
    diagnosis: str
    silver_metrics: CalibrationMetrics | None = None
    n_silver: int = 0


@dataclass(frozen=True)
class SegmentReport:
    """Calibration for one segment value of one segment key."""

    key: str
    value: str
    metrics: CalibrationMetrics
    diagnosis: str


@dataclass
class DatasetReport:
    """Everything the HTML report renders."""

    overall: CalibrationMetrics
    questions: list[QuestionReport] = field(default_factory=list)
    segments: list[SegmentReport] = field(default_factory=list)
    n_records: int = 0
    n_labeled_gold: int = 0
    n_labeled_silver: int = 0
    n_unlabeled: int = 0
    n_score_excluded: int = 0
    n_questions: int = 0
    models: list[str] = field(default_factory=list)
    segment_keys: list[str] = field(default_factory=list)
    silver_metrics: CalibrationMetrics | None = None
    overall_diagnosis: str = ""

    @property
    def silver_only(self) -> bool:
        """True when every label in the dataset is silver, which weakens every conclusion."""
        return self.n_labeled_gold == 0 and self.n_labeled_silver > 0


def _split_labels(
    records: Sequence[DecisionRecord],
) -> tuple[list[DecisionRecord], list[DecisionRecord], list[DecisionRecord]]:
    gold: list[DecisionRecord] = []
    silver: list[DecisionRecord] = []
    unlabeled: list[DecisionRecord] = []
    for record in records:
        if not record.is_labeled:
            unlabeled.append(record)
        elif record.is_silver:
            silver.append(record)
        else:
            gold.append(record)
    return gold, silver, unlabeled


def _points(records: Iterable[DecisionRecord]) -> tuple[list[float], list[bool]]:
    confidences: list[float] = []
    correct: list[bool] = []
    for record in records:
        point = record.calibration_point()
        if point is None:
            continue
        confidences.append(point[0])
        correct.append(point[1])
    return confidences, correct


@dataclass(frozen=True)
class MeasureOptions:
    """Measurement settings shared by every population in one evaluation."""

    n_bins: int = DEFAULT_N_BINS
    equal_width: bool = False
    alpha: float = 0.05
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES
    seed: int = 0


def _measure(records: Sequence[DecisionRecord], options: MeasureOptions) -> CalibrationMetrics:
    confidences, correct = _points(records)
    return compute_calibration(
        confidences,
        correct,
        n_bins=options.n_bins,
        equal_width=options.equal_width,
        alpha=options.alpha,
        n_boot=options.n_boot,
        seed=options.seed,
    )


def evaluate(
    records: Sequence[DecisionRecord],
    *,
    n_bins: int = DEFAULT_N_BINS,
    equal_width: bool = False,
    alpha: float = 0.05,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 0,
    by: Sequence[str] = (),
    min_segment_size: int = 10,
) -> DatasetReport:
    """Evaluate a record set.

    Gold-labeled records drive every headline number. Silver-labeled records are measured
    separately and never mixed into the gold aggregate, because agreement with a model is not
    accuracy.
    """
    gold, silver, unlabeled = _split_labels(records)
    options = MeasureOptions(
        n_bins=n_bins, equal_width=equal_width, alpha=alpha, n_boot=n_boot, seed=seed
    )
    overall = _measure(gold, options)

    by_question: dict[str, list[DecisionRecord]] = defaultdict(list)
    for record in records:
        by_question[record.question_key].append(record)

    questions: list[QuestionReport] = []
    for key in sorted(by_question):
        group = by_question[key]
        group_gold, group_silver, group_unlabeled = _split_labels(group)
        metrics = _measure(group_gold, options)
        silver_metrics = _measure(group_silver, options) if group_silver else None
        questions.append(
            QuestionReport(
                question_key=key,
                question_type=group[0].question_type,
                metrics=metrics,
                n_records=len(group),
                n_unlabeled=len(group_unlabeled),
                n_score_excluded=sum(1 for r in group if r.question_type == "score"),
                diagnosis=diagnose(metrics),
                silver_metrics=silver_metrics,
                n_silver=len(group_silver),
            )
        )

    segments: list[SegmentReport] = []
    for segment_key in by:
        grouped: dict[str, list[DecisionRecord]] = defaultdict(list)
        for record in gold:
            grouped[record.segment.get(segment_key, "unknown")].append(record)
        for value in sorted(grouped):
            group = grouped[value]
            if len(group) < min_segment_size:
                continue
            metrics = _measure(group, options)
            segments.append(
                SegmentReport(
                    key=segment_key,
                    value=value,
                    metrics=metrics,
                    diagnosis=diagnose(metrics),
                )
            )

    return DatasetReport(
        overall=overall,
        questions=questions,
        segments=segments,
        n_records=len(records),
        n_labeled_gold=len(gold),
        n_labeled_silver=len(silver),
        n_unlabeled=len(unlabeled),
        n_score_excluded=sum(1 for r in records if r.question_type == "score"),
        n_questions=len(by_question),
        models=sorted({record.model for record in records}),
        segment_keys=list(by),
        silver_metrics=_measure(silver, options) if silver else None,
        overall_diagnosis=diagnose(overall),
    )
