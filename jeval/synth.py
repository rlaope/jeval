"""Synthetic decision logs with known miscalibration.

This module is the credibility of the whole project. A calibration tool that cannot restore a
known miscalibration is worthless, so the generator is written first and the measurement code
has to pass against it before anything else is trusted.

Modes:

- ``calibrated``: reported confidence equals the probability the outcome was drawn with, so
  measured ECE must land inside its bootstrap interval of zero.
- ``inflated``: reported confidence is the true probability scaled up by a known factor.
- ``overconfident``: reported confidence is pushed up by a fixed exponent.
- ``underconfident``: reported confidence is shrunk toward 0.5.
- ``constant_high``: one fixed high confidence with a known lower accuracy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
from numpy.typing import NDArray

from jeval.schema import DecisionRecord

SYNTH_MODES = ("calibrated", "inflated", "overconfident", "underconfident", "constant_high")

# The highest confidence a generated record may claim: certainty is not a probability.
CERTAINTY_CAP = 0.999

DEFAULT_CLASSES: tuple[str, ...] = ("billing", "technical", "other")


@dataclass(frozen=True)
class SynthSpec:
    """Parameters of one synthetic question."""

    n: int = 1200
    mode: str = "calibrated"
    seed: int = 7
    inflation: float = 1.15
    exponent: float = 0.6
    question_key: str = "department"
    question_type: str = "choice"
    classes: tuple[str, ...] = DEFAULT_CLASSES
    accuracy_target: float = 0.95
    constant_confidence: float = 0.99
    score_bias: float = 0.0  # systematic optimism for score questions, in answer units
    label_fraction: float = 1.0
    silver_fraction: float = 0.0
    model: str = "jev-1.13.0"
    languages: tuple[str, ...] = ()
    tiers: tuple[str, ...] = ()
    state_tokens_range: tuple[int, int] = (200, 4000)
    start: datetime = field(default_factory=lambda: datetime(2026, 9, 1, tzinfo=timezone.utc))


def _transform(probabilities: NDArray[np.float64], spec: SynthSpec) -> NDArray[np.float64]:
    """Map true probabilities onto reported confidence for the requested mode."""
    if spec.mode == "calibrated":
        reported = probabilities
    elif spec.mode == "inflated":
        reported = probabilities * spec.inflation
    elif spec.mode == "overconfident":
        reported = probabilities**spec.exponent
    elif spec.mode == "underconfident":
        reported = 0.5 + (probabilities - 0.5) * spec.exponent
    elif spec.mode == "constant_high":
        reported = np.full_like(probabilities, spec.constant_confidence)
    else:
        raise ValueError(f"unknown mode {spec.mode!r}; expected one of {SYNTH_MODES}")
    # Clipping a continuous confidence at 1.0 stacks a large share of records on exactly-certain —
    # 45% of them at inflation 1.22 — and a threshold sweep then finds a free bucket at 1.00 and
    # recommends automating nothing. A producer that reports a probability does not claim certainty
    # on half its traffic, so the cap sits just below it.
    return np.clip(reported, 0.5000001, CERTAINTY_CAP)


def _true_probabilities(rng: np.random.Generator, n: int) -> NDArray[np.float64]:
    """Draw top-1 probabilities skewed toward the high end, as real classifiers produce."""
    return np.clip(0.5 + 0.5 * rng.beta(1.6, 1.2, size=n), 0.5000001, 1.0)


def _segment_of(index: int, spec: SynthSpec, rng: np.random.Generator) -> dict[str, str]:
    segment: dict[str, str] = {}
    if spec.languages:
        segment["lang"] = spec.languages[index % len(spec.languages)]
    if spec.tiers:
        segment["tier"] = spec.tiers[int(rng.integers(0, len(spec.tiers)))]
    return segment


def generate(spec: SynthSpec) -> list[DecisionRecord]:
    """Generate a synthetic labeled decision log for one question."""
    return _generate(spec)


def generate_paired(
    baseline: SynthSpec,
    current: SynthSpec,
    *,
    current_skill: float = 1.0,
    requests_seed: int = 0,
    key_prefix: str = "req",
) -> list[DecisionRecord]:
    """Two model versions answering the same requests, as shadow traffic logs them.

    Request ``i`` gets ``source_key`` ``f"{key_prefix}-{i:05d}"`` on both sides, one true answer,
    and one true probability of being answered correctly, all drawn once with ``requests_seed``:
    the shared difficulty is what a paired comparison removes, and the shared answer means both
    versions are judged against the same truth. Each version then reports confidence through its
    own ``mode`` and draws its own outcome, label availability and segment with its own ``seed``,
    so two specs that differ only in seed are two equally good versions whose answers still differ.

    ``current_skill`` in ``(0, 1]`` makes the current version genuinely less accurate: its true
    probability is the shared one shrunk toward a coin flip, ``0.5 + (p - 0.5) * current_skill``,
    and its ``mode`` applies on top of that — a ``calibrated`` current version with skill 0.6 is
    worse and knows it. Both specs must describe the same question and the same number of requests;
    ``score`` and ``constant_high`` have no per-request probability to share and are refused.
    """
    if baseline.n != current.n:
        raise ValueError("paired versions must answer the same number of requests")
    if (baseline.question_key, baseline.question_type, baseline.classes) != (
        current.question_key,
        current.question_type,
        current.classes,
    ):
        raise ValueError("paired versions must answer the same question")
    if baseline.model == current.model:
        raise ValueError("paired versions need two different model values")
    for spec in (baseline, current):
        if spec.question_type == "score" or spec.mode == "constant_high":
            raise ValueError("paired generation needs a per-request probability to share")
    if not 0.0 < current_skill <= 1.0:
        raise ValueError("current_skill must be in (0, 1]")
    requests = np.random.default_rng(requests_seed)
    shared = _true_probabilities(requests, baseline.n)
    answers = ("yes", "no") if baseline.question_type == "noul" else baseline.classes
    answers = answers or DEFAULT_CLASSES
    truths = [answers[int(i)] for i in requests.integers(0, len(answers), size=baseline.n)]
    shrunk = np.clip(0.5 + (shared - 0.5) * current_skill, 0.5000001, 1.0)
    records: list[DecisionRecord] = []
    for spec, true_p in ((baseline, shared), (current, shrunk)):
        for index, record in enumerate(_generate(spec, true_p, truths)):
            record.source_key = f"{key_prefix}-{index:05d}"
            records.append(record)
    return records


def _generate(
    spec: SynthSpec,
    true_p: NDArray[np.float64] | None = None,
    truths: Sequence[str] | None = None,
) -> list[DecisionRecord]:
    """The generator; ``true_p`` and ``truths`` are supplied only by :func:`generate_paired`.

    With both absent the random stream is exactly the one ``generate`` has always drawn, so every
    existing spec keeps producing the same records.
    """
    if spec.mode not in SYNTH_MODES:
        raise ValueError(f"unknown mode {spec.mode!r}; expected one of {SYNTH_MODES}")
    rng = np.random.default_rng(spec.seed)

    if true_p is None:
        if spec.mode == "constant_high":
            true_p = np.full(spec.n, spec.accuracy_target, dtype=float)
        else:
            true_p = _true_probabilities(rng, spec.n)
    reported = _transform(true_p, spec)

    correct = rng.random(spec.n) < true_p
    if spec.n == 0:
        return []

    records: list[DecisionRecord] = []
    for index in range(spec.n):
        label_known = bool(rng.random() < spec.label_fraction)
        label_source = None
        if label_known:
            label_source = "silver" if rng.random() < spec.silver_fraction else "human_override"
        segment = _segment_of(index, spec, rng)
        state_tokens = int(rng.integers(spec.state_tokens_range[0], spec.state_tokens_range[1]))
        ts = spec.start - timedelta(seconds=60 * index)

        if spec.question_type == "noul":
            if truths is None:
                positive = bool(rng.random() < 0.5)
            else:
                # The answer given is the shared truth when right and the other one when wrong.
                positive = (truths[index] == "yes") == bool(correct[index])
            decision = "yes" if positive else "no"
            opposite = "no" if positive else "yes"
            prediction = decision
            p_yes = float(reported[index]) if positive else 1.0 - float(reported[index])
            label = (decision if bool(correct[index]) else opposite) if label_known else None
            records.append(
                DecisionRecord(
                    ts=ts,
                    model=spec.model,
                    question_key=spec.question_key,
                    question_type="noul",
                    prediction=prediction,
                    confidence=abs(p_yes - 0.5) * 2,
                    probabilities={"yes": p_yes, "no": 1.0 - p_yes},
                    label=label,
                    label_source=label_source,  # type: ignore[arg-type]
                    segment=segment,
                    state_tokens=state_tokens,
                )
            )
            continue

        if spec.question_type == "score":
            # A numeric answer fails by being *off*, not by naming the wrong class, so the
            # injected miscalibration here is a systematic offset plus small noise. Rank
            # correlation survives an offset; MAE and the gap column do not.
            actual = float(rng.uniform(0.05, 0.95))
            offset = spec.score_bias if spec.mode != "calibrated" else 0.0
            noise = float(rng.normal(0.0, 0.04))
            predicted = min(1.0, max(0.0, actual + offset + noise))
            records.append(
                DecisionRecord(
                    ts=ts,
                    model=spec.model,
                    question_key=spec.question_key,
                    question_type="score",
                    prediction=f"{predicted:.3f}",
                    confidence=predicted,
                    label=f"{actual:.3f}" if label_known else None,
                    label_source=label_source,  # type: ignore[arg-type]
                    segment=segment,
                    state_tokens=state_tokens,
                )
            )
            continue

        classes = spec.classes or DEFAULT_CLASSES
        # The prediction follows the sampled outcome instead of always being `classes[0]`. A
        # generator whose classifier only ever answers the first class cannot exercise a per-class
        # threshold, a per-class cost action, or any slice by predicted class — every artifact built
        # from it looks like a tool that routes exactly one class.
        truth = classes[int(rng.integers(0, len(classes)))] if truths is None else truths[index]
        if bool(correct[index]):
            prediction = truth
        else:
            alternatives = [name for name in classes if name != truth]
            prediction = alternatives[int(rng.integers(0, len(alternatives)))]
        label = truth if label_known else None
        remaining = [name for name in classes if name != prediction] or ["__other__"]
        top = float(reported[index])
        leftover = max(0.0, 1.0 - top)
        weights = rng.random(len(remaining))
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(remaining))
        probabilities = {prediction: top}
        for name, weight in zip(remaining, weights, strict=True):
            probabilities[name] = float(leftover * weight)
        records.append(
            DecisionRecord(
                ts=ts,
                model=spec.model,
                question_key=spec.question_key,
                question_type="choice" if spec.question_type not in ("score",) else "score",
                prediction=prediction,
                confidence=top,
                probabilities=probabilities,
                label=label,
                label_source=label_source,  # type: ignore[arg-type]
                segment=segment,
                state_tokens=state_tokens,
            )
        )
    return records


def accuracy_of(records: Sequence[DecisionRecord]) -> float:
    """Observed accuracy over labeled records; ``nan`` when nothing is labeled."""
    labeled = [record for record in records if record.is_labeled]
    if not labeled:
        return float("nan")
    hits = sum(1 for record in labeled if record.is_correct)
    return hits / len(labeled)


@dataclass(frozen=True)
class DemoDataset:
    """A mixed multi-question synthetic dataset used by ``jeval demo``."""

    records: list[DecisionRecord]
    description: str


def demo_dataset(seed: int = 11, scale: float = 1.0) -> DemoDataset:
    """Build the demo log: three scored questions with different failure modes.

    The mix is deliberate. One question is nearly honest, one is inflated, one is a
    binary (``noul``) question, and one is a ``score`` question that jeval refuses to fold
    into binary accuracy.
    """

    def size(base: int) -> int:
        return max(40, int(base * scale))

    specs = [
        SynthSpec(
            n=size(620),
            mode="overconfident",
            exponent=0.75,
            question_key="department",
            seed=seed,
            languages=("ko", "en"),
            tiers=("free", "pro"),
            label_fraction=0.86,
            silver_fraction=0.05,
            model="jev-1.13.0",
        ),
        SynthSpec(
            n=size(540),
            mode="inflated",
            inflation=1.22,
            question_key="intent",
            classes=("refund_request", "check_balance", "other"),
            seed=seed + 1,
            languages=("ko", "en"),
            tiers=("free", "pro"),
            label_fraction=0.9,
            model="jev-1.13.0",
        ),
        SynthSpec(
            n=size(430),
            mode="calibrated",
            question_key="is_urgent",
            question_type="noul",
            seed=seed + 2,
            languages=("ko", "en"),
            label_fraction=0.88,
            model="jev-1.13.0",
        ),
        SynthSpec(
            n=size(220),
            mode="inflated",
            question_key="satisfaction",
            question_type="score",
            score_bias=0.18,  # answers run 0.18 optimistic, on purpose
            seed=seed + 3,
            label_fraction=0.8,
            model="jev-1.13.0",
        ),
    ]
    records: list[DecisionRecord] = []
    for spec in specs:
        records.extend(generate(spec))
    description = (
        f"{len(records)} synthetic decisions across 4 questions "
        "(2 choice, 1 noul, 1 score), generated with seed "
        f"{seed}. Two of the answered questions are miscalibrated on purpose, so the report "
        "has something real to find, and one numeric score question whose answers run 0.18 "
        "optimistic on purpose."
    )
    return DemoDataset(records=records, description=description)
