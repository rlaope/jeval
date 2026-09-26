"""Active-learning labeling queue: spend the next label where it buys the most.

Labels are the scarce resource in jeval. ``jeval report`` can already say that an interval is
too wide, but only more labels close it -- so the operational question is "which records should
a human label next?", and this module answers it with a *ranking*, never a sample. The priority
order is the one in the specification:

1. **Records inside the threshold candidate band.** A label just above or below the candidate
   threshold is the one that can move the recommended threshold, because the recommendation is
   decided by the cases sitting on the line. A label far away changes nothing.
2. **Records whose probability distribution is flat.** A near-uniform answer over the record's own
   classes is evidence that the *criteria* for the question are unclear, not that the model is
   uncertain about a question it understands. Labeling one settles what the question means.
3. **Records in sparsely labeled confidence bins.** Calibration is measured per confidence bin;
   a bin with nothing in it is a region where nothing is measured, and no amount of analysis
   fills it.

Everything else still gets queued, after those three groups, because a label in an already
covered region refines a measurement instead of filling a gap.

Four rules keep the queue honest:

- **Nothing is sampled.** Every eligible record is ranked; the seed only rotates exact ties, so
  the same input and seed always produce the same queue and a different seed never drops or
  duplicates a record.
- **A labeled record is never queued.** Gold, silver or anything else: the queue is work that is
  still outstanding. An all-labeled log produces an empty queue, not a re-labeling suggestion.
- **No threshold is invented.** A band exists only where the caller supplied a threshold
  candidate (``jeval.costs.sweep``'s recommendation, or the deployed threshold). Without one, the
  first group is empty and the reasons say so; a threshold the tool made up would rank records
  against a number nobody has agreed to.
- **Every reason says what the label buys.** ``QueueItem.reason`` is a plain sentence about the
  measurement, so a reviewer reading the sheet knows why the record is there.

Records whose ``probabilities`` map is missing cannot be flat -- absence of a distribution is not
a flat one -- and they keep their place in the other groups. Nothing here raises on thin or
unlabeled input: an empty queue is a legitimate answer.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import log
from pathlib import Path
from typing import Any, get_args

from jeval.calibration import DEFAULT_N_BINS, MIN_BIN_SIZE
from jeval.report.model import ThresholdResult
from jeval.schema import DecisionRecord, LabelSource

__all__ = [
    "DEFAULT_BAND_HALF_WIDTH",
    "DEFAULT_LIMIT",
    "FLAT_ENTROPY_MIN",
    "KEY_COLUMN",
    "LABEL_COLUMN",
    "MIN_LABELED_PER_BIN",
    "PRIORITY_BAND",
    "PRIORITY_FLAT",
    "PRIORITY_NAMES",
    "PRIORITY_REMAINDER",
    "PRIORITY_SPARSE",
    "SHEET_COLUMNS",
    "QueueItem",
    "apply_labels",
    "build_queue",
    "export_session",
    "format_queue",
    "priority_breakdown",
]

DEFAULT_LIMIT = 50
"""How many records one labeling session asks for when the caller does not say."""

DEFAULT_BAND_HALF_WIDTH = 0.05
"""Half-width of the threshold candidate band, in confidence units.

Five sweep steps of ``jeval.costs.DEFAULT_STEPS``: a label inside this window is close enough to
the line to change which side of it a threshold recommendation lands on.
"""

MIN_LABELED_PER_BIN = MIN_BIN_SIZE
"""Labeled records a confidence bin needs before it counts as covered.

The same floor the calibration bins use for a well-populated bin (``jeval.calibration``), so the
queue and the reliability curve agree about which regions are measured.
"""

FLAT_ENTROPY_MIN = 0.9
"""Normalized entropy at which an answer counts as flat.

Entropy divided by the maximum the record's own class count allows, so 1.0 is a uniform answer.
0.9 asks for a distribution within a tenth of uniform: ``{0.36, 0.32, 0.32}`` qualifies (0.999),
``{0.60, 0.25, 0.15}`` does not (0.853).
"""

PRIORITY_BAND = 1
"""Priority of a record inside the threshold candidate band."""

PRIORITY_FLAT = 2
"""Priority of a record whose probability distribution is flat."""

PRIORITY_SPARSE = 3
"""Priority of a record in a sparsely labeled confidence bin."""

PRIORITY_REMAINDER = 4
"""Priority of everything else: still worth a label, but only as a top-up."""

PRIORITY_NAMES: dict[int, str] = {
    PRIORITY_BAND: "threshold band",
    PRIORITY_FLAT: "flat distribution",
    PRIORITY_SPARSE: "sparse bin",
    PRIORITY_REMAINDER: "remainder",
}
"""Class name per priority, in queue order."""

SHEET_COLUMNS: tuple[str, ...] = (
    "record_id",
    "question_key",
    "confidence",
    "prediction",
    "priority",
    "reason",
    "label",
)
"""Columns of the labeling sheet ``export_session`` writes, in order.

The first six are jeval's own report of why the record is queued; ``label`` is left empty for the
reviewer to fill in, and is the only column ``apply_labels`` reads back.
"""

KEY_COLUMN = "record_id"
"""Sheet column that joins a row back to a record: ``DecisionRecord.id``."""

LABEL_COLUMN = "label"
"""Sheet column holding the human answer."""

_LABEL_SOURCES: frozenset[str] = frozenset(str(option) for option in get_args(LabelSource))
"""Every ``label_source`` the schema accepts, read off the schema's own Literal."""


@dataclass(frozen=True)
class QueueItem:
    """One record worth labeling, with the reason it is in the queue."""

    record_id: str
    question_key: str
    confidence: float
    prediction: str
    priority: int
    reason: str

    @property
    def priority_name(self) -> str:
        """The class name of :attr:`priority`, e.g. ``"threshold band"``."""
        return PRIORITY_NAMES[self.priority]


@dataclass(frozen=True)
class _Ranked:
    """A queue item with its sort keys, kept out of the public type."""

    item: QueueItem
    order: float
    digest: str


def _threshold_map(thresholds: Mapping[str, float | ThresholdResult] | None) -> dict[str, float]:
    """Normalize the caller's threshold candidates to ``question -> confidence``.

    A value may be a plain confidence or a :class:`~jeval.report.model.ThresholdResult` from
    ``jeval.costs.sweep``, whose ``threshold`` is used. A candidate that is not finite -- the
    ``nan`` a sweep returns when it withheld a recommendation -- contributes no band, because
    there is no number to rank against. A finite candidate outside ``[0, 1]``, or a value of any
    other type, is a caller bug and raises instead of being clamped into a band nobody defined.
    """
    resolved: dict[str, float] = {}
    if thresholds is None:
        return resolved
    for question, value in thresholds.items():
        if isinstance(value, ThresholdResult):
            number = float(value.threshold)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
        else:
            raise ValueError(
                f"thresholds[{question!r}] must be a confidence or a ThresholdResult, "
                f"got {type(value).__name__}"
            )
        if not math.isfinite(number):
            continue
        if not 0.0 <= number <= 1.0:
            raise ValueError(
                f"thresholds[{question!r}] must be a confidence in [0, 1], got {number}"
            )
        resolved[str(question)] = number
    return resolved


def _bin_index(confidence: float, n_bins: int) -> int:
    """Index of the equal-width confidence bin a confidence falls in.

    Bins are ``[0, 1]`` cut into ``n_bins`` equal widths with the top edge included in the last
    bin, and the same helper counts observations and looks them up, so a bin's occupancy can
    never disagree with the bin a record is assigned to.
    """
    return min(int(confidence * n_bins), n_bins - 1)


def _bin_bounds(index: int, n_bins: int) -> tuple[float, float]:
    """``(low, high)`` confidence bounds of one bin, for a reason line."""
    return (index / n_bins, (index + 1) / n_bins)


def _bin_counts(labeled: Sequence[DecisionRecord], n_bins: int) -> list[int]:
    """How many labeled records sit in each confidence bin."""
    counts = [0] * n_bins
    for record in labeled:
        counts[_bin_index(record.stated_probability, n_bins)] += 1
    return counts


def _normalized_entropy(probabilities: Mapping[str, float] | None) -> float | None:
    """Entropy of a probability map, as a share of the maximum its class count allows.

    ``1.0`` is a uniform answer and ``0.0`` a single-class one, so the figure is comparable
    across questions with different class counts. Returns ``None`` when there is no usable
    distribution: a missing or single-class map, a negative value, or a map that sums to zero.
    """
    if not probabilities or len(probabilities) < 2:
        return None
    values = [float(value) for value in probabilities.values()]
    if any(value < 0.0 for value in values):
        return None
    total = sum(values)
    if total <= 0.0:
        return None
    shares = [value / total for value in values if value > 0.0]
    if len(shares) < 2:
        return None
    entropy = -sum(share * log(share) for share in shares)
    return entropy / log(len(values))


def _plural(count: int, noun: str, plural: str | None = None) -> str:
    """``1 record`` / ``3 records`` -- reason lines are read by people."""
    return f"{count} {noun}" if count == 1 else f"{count} {plural or noun + 's'}"


def _tiebreak(record_id: str, seed: int) -> str:
    """A stable digest that rotates exact ties without depending on input order."""
    return sha256(f"{seed}:{record_id}".encode()).hexdigest()


def _band_reason(record: DecisionRecord, threshold: float) -> str:
    return (
        f"inside the threshold candidate band for {record.question_key} "
        f"({threshold:.2f} +/- {DEFAULT_BAND_HALF_WIDTH:.2f}): this label lands on the decision "
        "line, so it moves the recommended threshold more than any other label in the log"
    )


def _flat_reason(record: DecisionRecord, entropy: float, classes: int) -> str:
    # A flat answer always spans at least two classes, so the plural is safe here.
    return (
        f"flat distribution over {classes} classes (normalized entropy {entropy:.2f}): "
        f"a near-uniform answer is a sign the criteria for {record.question_key} are unclear, so "
        "this label settles what the question means rather than how sure the model was"
    )


def _sparse_reason(lo: float, hi: float, count: int) -> str:
    held = "no labeled records yet" if count == 0 else f"only {_plural(count, 'labeled record')}"
    return (
        f"confidence {lo:.2f}-{hi:.2f} holds {held}: this label fills a gap in the reliability "
        "curve, where nothing is measured today"
    )


def _remainder_reason(
    record: DecisionRecord, threshold: float | None, lo: float, hi: float, count: int
) -> str:
    if threshold is None:
        support = f"no threshold candidate is known for {record.question_key}"
    else:
        distance = abs(record.stated_probability - threshold)
        support = (
            f"confidence {record.stated_probability:.2f} sits {distance:.2f} outside the "
            f"{record.question_key} candidate band ({threshold:.2f})"
        )
    return (
        f"confidence {lo:.2f}-{hi:.2f} already holds {_plural(count, 'labeled record')} and "
        f"{support}: this label refines an existing measurement rather than filling a gap"
    )


def _rank(
    record: DecisionRecord,
    thresholds: Mapping[str, float],
    counts: Sequence[int],
    seed: int,
) -> _Ranked:
    """Place one unlabeled record in a priority class and build its reason."""
    # The scale thresholds and calibration bins are drawn on, so a yes/no record lands in the
    # band its threshold was computed in.
    confidence = record.stated_probability
    index = _bin_index(confidence, DEFAULT_N_BINS)
    lo, hi = _bin_bounds(index, DEFAULT_N_BINS)
    bin_count = counts[index]
    threshold = thresholds.get(record.question_key)

    if threshold is not None and abs(confidence - threshold) <= DEFAULT_BAND_HALF_WIDTH:
        item = QueueItem(
            record_id=record.id,
            question_key=record.question_key,
            confidence=confidence,
            prediction=record.prediction,
            priority=PRIORITY_BAND,
            reason=_band_reason(record, threshold),
        )
        # Closest to the line first: the nearer record can flip the recommendation.
        return _Ranked(
            item=item, order=abs(confidence - threshold), digest=_tiebreak(record.id, seed)
        )

    entropy = _normalized_entropy(record.probabilities)
    if entropy is not None and entropy >= FLAT_ENTROPY_MIN:
        item = QueueItem(
            record_id=record.id,
            question_key=record.question_key,
            confidence=confidence,
            prediction=record.prediction,
            priority=PRIORITY_FLAT,
            reason=_flat_reason(record, entropy, len(record.probabilities or {})),
        )
        # Flattest first: the most uniform answer is the strongest signal about the criteria.
        return _Ranked(item=item, order=-entropy, digest=_tiebreak(record.id, seed))

    if bin_count < MIN_LABELED_PER_BIN:
        item = QueueItem(
            record_id=record.id,
            question_key=record.question_key,
            confidence=confidence,
            prediction=record.prediction,
            priority=PRIORITY_SPARSE,
            reason=_sparse_reason(lo, hi, bin_count),
        )
        # Emptiest bin first: the widest gap in the reliability curve.
        return _Ranked(item=item, order=float(bin_count), digest=_tiebreak(record.id, seed))

    item = QueueItem(
        record_id=record.id,
        question_key=record.question_key,
        confidence=confidence,
        prediction=record.prediction,
        priority=PRIORITY_REMAINDER,
        reason=_remainder_reason(record, threshold, lo, hi, bin_count),
    )
    return _Ranked(item=item, order=float(bin_count), digest=_tiebreak(record.id, seed))


def build_queue(
    records: Sequence[DecisionRecord],
    *,
    thresholds: Mapping[str, float | ThresholdResult] | None = None,
    limit: int = DEFAULT_LIMIT,
    seed: int = 0,
) -> tuple[QueueItem, ...]:
    """Rank the unlabeled records of ``records`` and return at most ``limit`` of them.

    The order is the specification's: threshold candidate band, then flat probability
    distributions, then sparsely labeled confidence bins, then everything else. Within a class the
    most informative record comes first -- nearest the candidate threshold, flattest distribution,
    emptiest bin -- and exact ties are rotated by a digest of ``(seed, record id)``, so the queue
    is stable for a given seed and independent of the input order.

    ``thresholds`` maps a question key to its threshold candidate, either a confidence or a
    ``jeval.costs.sweep`` result. With no candidate for a question the band class is empty for
    that question and its records are ranked on the other two signals: jeval does not invent a
    threshold to rank against.

    Records that already carry a label are work that is done, so they are never queued; an empty
    log, an all-labeled log, ``limit=0`` and records without a probability map all return a
    queue instead of raising. A negative ``limit``, a threshold value of the wrong type, and a
    finite threshold outside ``[0, 1]`` raise, because those are caller bugs and a silently
    ignored one would rank labels against a number nobody agreed to.
    """
    if limit < 0:
        raise ValueError(f"limit must not be negative, got {limit}")
    resolved = _threshold_map(thresholds)
    candidates = [record for record in records if not record.is_labeled]
    if limit == 0 or not candidates:
        return ()
    counts = _bin_counts([record for record in records if record.is_labeled], DEFAULT_N_BINS)
    ranked = [_rank(record, resolved, counts, seed) for record in candidates]
    ranked.sort(key=lambda entry: (entry.item.priority, entry.order, entry.digest))
    return tuple(entry.item for entry in ranked[:limit])


def priority_breakdown(queue: Sequence[QueueItem]) -> dict[str, int]:
    """How many queued records fall in each priority class.

    Every class is present, including the empty ones, in queue order, so a caller can print a
    fixed set of lines and the report of a partial queue cannot be mistaken for a full one.
    """
    counts = {name: 0 for name in PRIORITY_NAMES.values()}
    for item in queue:
        counts[item.priority_name] += 1
    return counts


def format_queue(queue: Sequence[QueueItem]) -> str:
    """Render a queue as plain text, one line per record, with no ANSI escapes.

    Each line carries the position, the record id, the question, the confidence, the prediction
    and the reason, in that order, so the sheet and the terminal show the same facts.
    """
    if not queue:
        return "labeling queue is empty: every record already has a label."
    width = max(len(item.record_id) for item in queue)
    lines = [
        f"{position:>3}  {item.record_id:<{width}}  {item.question_key}  "
        f"confidence {item.confidence:.2f}  prediction {item.prediction}  "
        f"[{item.priority_name}] {item.reason}"
        for position, item in enumerate(queue, start=1)
    ]
    return "\n".join(lines)


def export_session(queue: Sequence[QueueItem], path: Path | str) -> Path:
    """Write the labeling sheet for one session and return the path it wrote.

    The sheet is a CSV with the columns of :data:`SHEET_COLUMNS`: jeval's own description of each
    queued record, and one empty ``label`` cell per row for the reviewer to fill in. Confidence is
    written to four decimal places as a display value; ``record_id`` is the join key and the
    other columns are never read back, so :func:`apply_labels` can be pointed at the same file
    (or at anything with the same two column names).
    """
    target = Path(path)
    if target.parent != Path(""):
        target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SHEET_COLUMNS))
        writer.writeheader()
        for item in queue:
            writer.writerow(
                {
                    "record_id": item.record_id,
                    "question_key": item.question_key,
                    "confidence": f"{item.confidence:.4f}",
                    "prediction": item.prediction,
                    "priority": item.priority,
                    "reason": item.reason,
                    "label": "",
                }
            )
    return target


def _cell(row: Mapping[str, Any], column: str) -> str | None:
    """A trimmed cell value, or ``None`` when the cell is missing or blank."""
    value = row.get(column)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _apply_one(record: DecisionRecord, answer: str, source: str) -> None:
    """Set one record's label and source through the schema's own normalization.

    Re-validating a copy means the label is normalized exactly as a freshly ingested one would
    be -- a ``noul`` answer of ``"Yes"`` becomes ``"yes"`` -- so a sheet entry cannot introduce a
    label the rest of jeval would read differently. Only the two label fields are written back.
    """
    normalized = DecisionRecord.model_validate(
        {**record.model_dump(), "label": answer, "label_source": source}
    )
    record.label = normalized.label
    record.label_source = normalized.label_source


def apply_labels(
    records: Sequence[DecisionRecord],
    labeled_rows: Iterable[Mapping[str, Any]],
    *,
    key: str,
    label: str,
    source: LabelSource,
) -> int:
    """Write a reviewer's answers back onto ``records`` and return how many records were labeled.

    ``labeled_rows`` is the filled-in sheet as mappings -- ``csv.DictReader`` over the exported
    file is the intended use -- with ``key`` holding the record id (``DecisionRecord.id``) and
    ``label`` holding the answer. Records are updated in place, because the caller is expected to
    rewrite its log afterwards; the returned count is over *records*, so a record repeated in the
    sheet counts once and the sheet's last answer for it wins.

    Rows that are not counted are the ones with nothing to apply: a row whose ``label`` cell is
    still empty (the common case -- the reviewer answered only part of the sheet) and a row whose
    key names no record in ``records``. ``source`` is stamped on every applied label and must be
    one of the schema's ``label_source`` values.

    A sheet that carries neither column at all raises, because "0 applied" would otherwise hide a
    column-name mistake; an empty sheet returns 0, because there is nothing to apply.
    """
    rows = list(labeled_rows)
    if not rows:
        return 0
    first = rows[0]
    missing = [column for column in (key, label) if column not in first]
    if missing:
        present = ", ".join(sorted(str(column) for column in first)) or "none"
        raise ValueError(
            f"labeling rows have no {', '.join(missing)} column; columns present: {present}"
        )
    if source not in _LABEL_SOURCES:
        raise ValueError(
            f"unknown label source {source!r}; expected one of {', '.join(sorted(_LABEL_SOURCES))}"
        )

    by_id = {record.id: record for record in records}
    applied: set[str] = set()
    for row in rows:
        record_id = _cell(row, key)
        answer = _cell(row, label)
        if record_id is None or answer is None:
            continue
        record = by_id.get(record_id)
        if record is None:
            continue
        _apply_one(record, answer, source)
        applied.add(record_id)
    return len(applied)
