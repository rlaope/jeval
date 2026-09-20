"""Report data model.

Pure data. This module owns the shape of everything the report renders, so chart modules, the
template, and the CLI can be built against one frozen contract without importing each other.

Nothing here computes statistics: the builders live in ``jeval.costs``, ``jeval.drift``,
``jeval.evaluate``, and ``jeval.report.verdict``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

# Fixed argument order. The report is an argument document: it is read top to bottom, so the
# order is part of the specification, not a layout preference.
SECTION_ORDER: tuple[str, ...] = (
    "verdict",
    "reliability",
    "cost",
    "impact",
    "segments",
    "drift",
    "data_quality",
)

CLEAR = "clear"
TOO_LOW = "too_low"
TOO_HIGH = "too_high"
WELL_CALIBRATED = "well_calibrated"
INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class VerdictStat:
    """One headline figure in the verdict block."""

    label: str
    value: str
    sub: str = ""


@dataclass(frozen=True)
class Verdict:
    """The conclusion, readable without scrolling."""

    status: str
    headline: str
    detail: str
    stats: tuple[VerdictStat, ...] = ()


@dataclass(frozen=True)
class CostPoint:
    """Expected cost of one threshold decision, evaluated on the observed records."""

    threshold: float
    expected_cost: float
    auto_rate: float
    accuracy_auto: float
    accuracy_escalated: float
    accept_cost: float
    escalate_cost: float
    n_auto: int
    n_escalate: int
    n_wrong_auto: int
    n_wrong_escalate: int


@dataclass(frozen=True)
class ThresholdResult:
    """The chosen threshold with the evidence that supports it."""

    action: str
    question: str
    when: str
    threshold: float
    expected_cost_per_case: float
    auto_rate: float
    accuracy_auto: float
    ci_low: float
    ci_high: float
    curve: tuple[CostPoint, ...] = ()
    flat_region: tuple[float, float] | None = None
    n_records: int = 0
    models: tuple[str, ...] = ()
    cost_false_accept: float = 0.0
    cost_escalate: float = 0.0
    cost_false_reject: float = 0.0
    silver_only: bool = False

    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low

    def point_at(self, threshold: float) -> CostPoint | None:
        """Curve point nearest a threshold, used by the report's slider."""
        if not self.curve:
            return None
        return min(self.curve, key=lambda point: abs(point.threshold - threshold))


@dataclass(frozen=True)
class ImpactRow:
    """One row of the persuasion table."""

    label: str
    current: str
    recommended: str
    change: str = ""


@dataclass(frozen=True)
class ImpactTable:
    """What changes if the recommended threshold is adopted."""

    rows: tuple[ImpactRow, ...] = ()
    monthly_volume: float | None = None
    currency: str = "USD"
    paradox_note: str = ""
    current_threshold: float = 0.0
    recommended_threshold: float = 0.0


@dataclass(frozen=True)
class SegmentBar:
    """One segment's ECE, drawn as a horizontal bar."""

    key: str
    value: str
    ece: float
    n: int
    too_few_samples: bool = False

    @property
    def label(self) -> str:
        return f"{self.key} = {self.value}"


@dataclass(frozen=True)
class HeatmapCell:
    x: str
    y: str
    ece: float
    n: int


@dataclass(frozen=True)
class SegmentView:
    """Segment breakdown; a heatmap only when the grid stays readable."""

    bars: tuple[SegmentBar, ...] = ()
    heatmap: tuple[HeatmapCell, ...] = ()
    x_axis: str = ""
    y_axis: str = ""
    min_samples: int = 30


@dataclass(frozen=True)
class DriftSlice:
    """One period or model version, measured on its own."""

    label: str
    model: str
    start: str
    end: str
    n: int
    ece: float
    threshold: float | None = None
    auto_rate: float | None = None


@dataclass(frozen=True)
class ModelChange:
    """A point where the serving model changed."""

    model: str
    at: str
    label: str


@dataclass(frozen=True)
class DriftFailure:
    """One failed drift check, with the numbers that failed it."""

    check: str
    detail: str
    value: float
    limit: float


@dataclass(frozen=True)
class DriftView:
    """Model-version and period comparison; absent unless there is something to compare."""

    baseline_label: str
    current_label: str
    slices: tuple[DriftSlice, ...] = ()
    changes: tuple[ModelChange, ...] = ()
    failures: tuple[DriftFailure, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class DataQuality:
    """Sample and label state, plus the limitations this report does not hide."""

    rows: tuple[tuple[str, str], ...] = ()
    limitations: tuple[str, ...] = ()
    sparse_bins: int = 0
    total_bins: int = 0
    silver_only: bool = False
    no_gold_labels: bool = False


@dataclass(frozen=True)
class ScoreLevelRow:
    """One predicted-value band and what the actual values were inside it."""

    lo: float
    hi: float
    n: int
    mean_predicted: float
    mean_actual: float


@dataclass(frozen=True)
class ScoreView:
    """Score-type questions: error and rank agreement, never right/wrong accuracy."""

    n: int
    mae: float
    rmse: float
    spearman_rho: float
    levels: tuple[ScoreLevelRow, ...] = ()
    n_other_type: int = 0
    n_unlabeled: int = 0
    n_unparseable: int = 0


@dataclass(frozen=True)
class RecalibrationView:
    """A fitted correction, only ever shown with the cross-validated gain and the floor."""

    method: str
    before_ece: float
    after_ece: float
    n: int
    helps: bool
    note: str = ""


@dataclass(frozen=True)
class LabelPlanRow:
    """One scope's projection of how many more labels a tighter interval needs."""

    scope: str
    key: str
    n_now: int
    ece: float
    ci_width: float
    needed: str = ""
    reason: str = ""


@dataclass(frozen=True)
class SegmentThresholdRow:
    """A segment's own optimum, and whether splitting pays for itself."""

    label: str
    threshold: float
    cost_per_case: float
    delta: float
    n: int
    worth_splitting: bool
    reason: str = ""


@dataclass
class ReportModel:
    """Everything one report renders, in reading order."""

    verdict: Verdict
    reliability_sections: list[Any] = field(default_factory=list)
    thresholds: tuple[ThresholdResult, ...] = ()
    impact: ImpactTable | None = None
    impacts: Mapping[str, ImpactTable] = field(default_factory=dict)
    segments: SegmentView | None = None
    drift: DriftView | None = None
    score: ScoreView | None = None
    recalibration: RecalibrationView | None = None
    label_plan: tuple[LabelPlanRow, ...] = ()
    segment_thresholds: tuple[SegmentThresholdRow, ...] = ()
    data_quality: DataQuality = field(default_factory=DataQuality)
    generated_at: str = ""
    source_note: str = ""
    demo_note: str = ""
    active_question: str = ""
    section_flags: Mapping[str, bool] = field(default_factory=dict)

    def has_section(self, name: str) -> bool:
        if self.section_flags:
            return bool(self.section_flags.get(name, False))
        return True


DEFAULT_LIMITATIONS: tuple[str, ...] = (
    "Thresholds depend entirely on the cost figures you provided.",
    "This measures calibration on the records you supplied. It is not a benchmark of any model.",
    "Labels from human overrides may be biased toward cases a human noticed; silent errors are "
    "underrepresented.",
)


def to_jsonable(value: Any) -> Any:
    """Convert a dataclass tree into JSON-ready primitives."""
    if isinstance(
        value, (Verdict, ThresholdResult, ImpactTable, SegmentView, DriftView, DataQuality)
    ):
        return json.loads(json.dumps(asdict(value), default=_fallback))
    return json.loads(json.dumps(value, default=_fallback))


def _fallback(value: Any) -> Any:
    if isinstance(value, float):
        return None if value != value else value
    if isinstance(value, (tuple, list)):
        return list(value)
    if isinstance(value, Mapping):
        return dict(value)
    return str(value)


def sanitize(value: Any) -> Any:
    """Replace non-finite floats with ``null``.

    A threshold that could not be computed is a NaN in the data model and a JSON error in the
    document, so the conversion happens once, here, instead of at every call site.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    return value


def dumps(payload: Mapping[str, Any]) -> str:
    """Compact JSON for embedding; separators trimmed because file size is a requirement."""
    return json.dumps(sanitize(payload), separators=(",", ":"), allow_nan=False, default=_fallback)


def stat_rows(rows: Sequence[Sequence[str]]) -> tuple[tuple[str, str], ...]:
    return tuple((str(left), str(right)) for left, right in rows)
