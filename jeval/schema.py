"""Decision record schema.

A decision record is the single unit of data in jeval: one question, one model answer,
one confidence, and (optionally) the ground truth for that answer.
"""

from __future__ import annotations

import math
import secrets
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

QuestionType = Literal["choice", "score", "noul"]
LabelSource = Literal["human_review", "human_override", "silver"]

ALL_LABEL_SOURCES: tuple[str, ...] = tuple(get_args(LabelSource))
GOLD_LABEL_SOURCES: frozenset[str] = frozenset({"human_review", "human_override"})
SILVER_LABEL_SOURCE = "silver"

_YES_TOKENS = {"yes", "y", "true", "t", "1", "positive", "pos"}
_NO_TOKENS = {"no", "n", "false", "f", "0", "negative", "neg"}


def new_record_id() -> str:
    """Return a fresh record id, e.g. ``rec_01H...``."""
    return f"rec_{secrets.token_hex(8)}"


def _coerce_binary_label(value: Any) -> str | None:
    """Normalize a yes/no-ish label to ``"yes"`` / ``"no"``; return ``None`` when unknown."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if float(value) == 1.0:
            return "yes"
        if float(value) == 0.0:
            return "no"
        return None
    token = str(value).strip().lower()
    if token in _YES_TOKENS:
        return "yes"
    if token in _NO_TOKENS:
        return "no"
    return None


class DecisionRecord(BaseModel):
    """One model decision on one question, with optional ground truth."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=new_record_id)
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model: str
    question_key: str
    question_type: QuestionType
    prediction: str
    confidence: float = Field(ge=0.0, le=1.0, description="Top-1 probability, normalized.")
    probabilities: dict[str, float] | None = None
    label: str | None = None
    label_source: LabelSource | None = None
    source_key: str | None = Field(
        default=None,
        description="Key of the source row this record came from; the join key for free labels.",
    )
    segment: dict[str, str] = Field(default_factory=dict)
    state_tokens: int | None = None
    latency_ms: float | None = None
    cost_usd: float | None = None

    @field_validator("segment", mode="before")
    @classmethod
    def _stringify_segment(cls, value: Any) -> Any:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return {str(k): str(v) for k, v in value.items()}
        return value

    @model_validator(mode="after")
    def _normalize(self) -> DecisionRecord:
        record = self
        if record.question_type == "noul":
            record.prediction = _coerce_binary_label(record.prediction) or record.prediction
            if record.label is not None:
                record.label = _coerce_binary_label(record.label) or record.label
        if record.label is None:
            record.label_source = None
        return record

    @property
    def is_labeled(self) -> bool:
        return self.label is not None

    @property
    def is_gold(self) -> bool:
        return self.label_source in GOLD_LABEL_SOURCES

    @property
    def is_silver(self) -> bool:
        return self.label_source == SILVER_LABEL_SOURCE

    @property
    def is_correct(self) -> bool | None:
        """Whether the prediction matched the label, or ``None`` when unlabeled."""
        if self.label is None:
            return None
        return self.prediction == self.label

    @property
    def positive_probability(self) -> float | None:
        """Raw ``P(positive)`` for ``noul`` records when it can be recovered."""
        if self.question_type != "noul":
            return None
        if self.probabilities:
            for key, value in self.probabilities.items():
                if _coerce_binary_label(key) == "yes":
                    return float(value)
        if self.prediction == "yes":
            return 0.5 + self.confidence / 2
        return 0.5 - self.confidence / 2

    def calibration_point(self) -> tuple[float, bool] | None:
        """Return ``(confidence, correct)`` for binary-accuracy metrics.

        Returns ``None`` when the record cannot participate: ``score`` records are evaluated
        on a separate scale, and unlabeled records have nothing to compare against.
        """
        if self.question_type == "score":
            return None
        correct = self.is_correct
        if correct is None:
            return None
        return (self.confidence, correct)


def confidence_from_probabilities(probabilities: Mapping[str, float]) -> tuple[str, float]:
    """Return the top-1 ``(label, probability)`` pair of a probability map."""
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    top_label = max(probabilities, key=lambda key: probabilities[key])
    return top_label, float(probabilities[top_label])


def normalize_record(payload: Mapping[str, Any]) -> DecisionRecord:
    """Build a :class:`DecisionRecord` from a loosely typed mapping.

    Confidence handling follows the type rules in the specification:

    - ``choice``: confidence is the top-1 probability (taken from ``probabilities`` when given).
    - ``noul``: confidence is ``|p - 0.5| * 2`` so a coin-flip answer lands at 0.
    - ``score``: whatever confidence the producer reported is kept as-is.
    """
    question_type = str(payload.get("question_type", "choice")).strip().lower()
    if question_type not in ("choice", "score", "noul"):
        raise ValueError(f"unsupported question_type: {question_type!r}")

    raw_probabilities = payload.get("probabilities")
    probabilities: dict[str, float] | None = None
    if raw_probabilities:
        probabilities = {str(k): float(v) for k, v in dict(raw_probabilities).items()}

    prediction = payload.get("prediction")
    confidence = payload.get("confidence")

    if question_type == "noul":
        p_yes = payload.get("probability_positive")
        if p_yes is None and probabilities:
            for key, value in probabilities.items():
                if _coerce_binary_label(key) == "yes":
                    p_yes = value
                    break
        if p_yes is None and confidence is not None:
            normalized = _coerce_binary_label(prediction)
            half = float(confidence) / 2
            p_yes = 0.5 + half if normalized == "yes" else 0.5 - half
        if p_yes is None:
            raise ValueError("noul records need probability_positive, probabilities, or confidence")
        raw_yes = float(p_yes)
        if not math.isfinite(raw_yes):
            # `max(0.0, nan)` is 0.0, which would become a maximally confident "no".
            raise ValueError(f"noul probability must be finite, got {raw_yes!r}")
        p_yes = min(1.0, max(0.0, raw_yes))
        # Rebuilt from the clamped value: storing the caller's map unchanged let a probability of
        # 1.5 reach the record and the threshold sweep.
        probabilities = {"yes": p_yes, "no": 1.0 - p_yes}
        # The probability is authoritative for a yes/no answer: deriving the prediction from it
        # keeps `confidence` the distance from the fence for the answer that was stored. A caller
        # passing "yes" alongside p_yes = 0.0 used to store a maximally confident "yes".
        prediction = "yes" if p_yes > 0.5 else "no"
        confidence = abs(p_yes - 0.5) * 2
    elif question_type == "choice":
        if probabilities:
            top_label, top_prob = confidence_from_probabilities(probabilities)
            if prediction is None:
                prediction = top_label
                confidence = top_prob
            else:
                matching = next((key for key in probabilities if str(key) == str(prediction)), None)
                if matching is None:
                    # A winner its own distribution gives no mass to is not a decision: keep
                    # confidence = P(prediction) would be 0, and top-1 would describe another class.
                    raise ValueError(
                        f"choice prediction {prediction!r} is not in its own probabilities map "
                        f"({sorted(probabilities)})"
                    )
                # The confidence must be the probability of the answer that was stored, never
                # the largest number in the map when those differ.
                confidence = probabilities[matching]
        if prediction is None:
            raise ValueError("choice records need a prediction or a probabilities map")
        if confidence is None:
            raise ValueError("choice records need a confidence or a probabilities map")

    if confidence is None:
        raise ValueError("score records need a confidence")

    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence out of range: {confidence}")

    known = {
        "id",
        "ts",
        "model",
        "question_key",
        "question_type",
        "prediction",
        "confidence",
        "probabilities",
        "label",
        "label_source",
        "source_key",
        "segment",
        "state_tokens",
        "latency_ms",
        "cost_usd",
    }
    extra = {key: value for key, value in payload.items() if key in known and value is not None}
    extra["question_type"] = question_type
    extra["prediction"] = str(prediction)
    extra["confidence"] = confidence
    extra["probabilities"] = probabilities

    if extra.get("source_key") is not None:
        extra["source_key"] = str(extra["source_key"])

    label = payload.get("label")
    if question_type == "noul" and label is not None:
        extra["label"] = _coerce_binary_label(label) or str(label)

    return DecisionRecord.model_validate(extra)
