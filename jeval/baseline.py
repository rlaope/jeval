"""Baseline snapshots: prove that calibration changed since a known-good run.

A model-version comparison answers "is the new model worse than the old one". A baseline snapshot
answers the question CI actually asks: "is the model worse than it was when we last agreed the
numbers were acceptable" — which is the only comparison that matters when the model string never
changes but the behaviour does.

The snapshot holds measurements, never records: it is safe to commit, small, and diffable.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jeval.calibration import compute_calibration
from jeval.report.model import DriftSlice, DriftView, ModelChange
from jeval.schema import DecisionRecord

SNAPSHOT_VERSION = 1


def _gold(records: Sequence[DecisionRecord]) -> list[DecisionRecord]:
    return [
        record for record in records if record.is_gold and record.calibration_point() is not None
    ]


def _ece(group: Sequence[DecisionRecord], *, alpha: float, n_boot: int) -> tuple[float, int]:
    pairs = [point for point in (record.calibration_point() for record in group) if point]
    metrics = compute_calibration(
        [point[0] for point in pairs],
        [point[1] for point in pairs],
        alpha=alpha,
        n_boot=n_boot,
    )
    return metrics.ece, metrics.n


def snapshot(
    records: Sequence[DecisionRecord], *, generated_at: str | None = None
) -> dict[str, Any]:
    """Freeze the current per-question measurements, plus the conditions they were taken under."""
    gold = _gold(records)
    questions: dict[str, list[DecisionRecord]] = {}
    for record in gold:
        questions.setdefault(record.question_key, []).append(record)
    overall_ece, overall_n = _ece(gold, alpha=0.05, n_boot=200)
    stamps = sorted(record.ts for record in records)
    return {
        "schema_version": SNAPSHOT_VERSION,
        "generated_at": generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models": sorted({record.model for record in records}),
        "records": len(records),
        "labeled_gold": len(gold),
        "overall": {
            "ece": None if math.isnan(overall_ece) else round(overall_ece, 6),
            "n": overall_n,
        },
        "date_range": {
            "start": stamps[0].isoformat() if stamps else "",
            "end": stamps[-1].isoformat() if stamps else "",
        },
        "questions": {
            key: {"ece": round(_ece(group, alpha=0.05, n_boot=200)[0], 6), "n": len(group)}
            for key, group in sorted(questions.items())
            if len(group) >= 2
        },
    }


def view_from_snapshot(
    snapshot_payload: Mapping[str, Any],
    records: Sequence[DecisionRecord],
    *,
    min_slice: int = 30,
    alpha: float = 0.05,
    n_boot: int = 200,
) -> DriftView:
    """Compare a saved snapshot against the current records.

    Questions present on only one side are reported as such instead of being silently dropped:
    a new question appearing is itself worth knowing.
    """
    version = snapshot_payload.get("schema_version")
    if version != SNAPSHOT_VERSION:
        raise ValueError(
            f"unsupported baseline schema_version {version!r}: expected {SNAPSHOT_VERSION}"
        )
    saved_questions = snapshot_payload.get("questions") or {}
    saved_models = ", ".join(snapshot_payload.get("models") or []) or "baseline"
    saved_at = str(snapshot_payload.get("generated_at") or "baseline")

    gold = _gold(records)
    current_by_question: dict[str, list[DecisionRecord]] = {}
    for record in gold:
        current_by_question.setdefault(record.question_key, []).append(record)

    slices: list[DriftSlice] = []
    labels = sorted(set(saved_questions) | set(current_by_question))
    for key in labels:
        saved = saved_questions.get(key)
        current_group = current_by_question.get(key, [])
        if saved and int(saved.get("n", 0)) >= min_slice:
            slices.append(
                DriftSlice(
                    label=key,
                    model=saved_models,
                    start=saved_at,
                    end=saved_at,
                    n=int(saved["n"]),
                    ece=float(saved["ece"]),
                )
            )
        if len(current_group) >= min_slice:
            current_ece, current_n = _ece(current_group, alpha=alpha, n_boot=n_boot)
            slices.append(
                DriftSlice(
                    label=key,
                    model=", ".join(sorted({record.model for record in current_group})),
                    start=saved_at,
                    end=str(snapshot_payload.get("date_range", {}).get("end") or "now"),
                    n=current_n,
                    ece=current_ece,
                )
            )

    changes = tuple(
        ModelChange(
            model=model,
            at=saved_at,
            label=f"{saved_models} -> {model}",
        )
        for model in sorted({record.model for record in records})
        if model not in (snapshot_payload.get("models") or [])
    )
    note = (
        f"compared against the baseline saved {saved_at} ({saved_models}); "
        f"{len(slices) // 2 or 0} question(s) measurable on both sides"
    )
    return DriftView(
        baseline_label=saved_models,
        current_label=", ".join(sorted({record.model for record in records})) or "current",
        slices=tuple(slices),
        changes=changes,
        failures=(),
        note=note,
    )


def write_snapshot(payload: Mapping[str, Any], path: Path | str) -> Path:
    import json

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
