"""Ingest raw logs (JSONL or CSV) into decision records."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jeval.config import IngestMap
from jeval.schema import DecisionRecord, normalize_record


@dataclass
class IngestReport:
    """What an ingest run did, in numbers a user can act on."""

    out_path: Path
    n_rows: int = 0
    n_records: int = 0
    n_skipped: int = 0
    n_unlabeled: int = 0
    n_duplicate_questions: int = 0
    per_question: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.n_records > 0


def _coerce_timestamp(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    return value


def iter_rows(path: Path | str) -> Iterator[dict[str, Any]]:
    """Yield raw rows from a JSONL or CSV file, chosen by suffix."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"no input file at {source}")
    if source.suffix.lower() in (".csv", ".tsv"):
        delimiter = "\t" if source.suffix.lower() == ".tsv" else ","
        with source.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter=delimiter):
                yield dict(row)
        return
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON ({exc.msg})") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{source}:{line_number}: expected a JSON object per line")
            yield payload


def _shared_fields(row: Mapping[str, Any], mapping: IngestMap) -> dict[str, Any]:
    """Row-level fields shared by every question the row carries."""
    shared = mapping.apply(row)
    shared.pop("question_key", None)
    shared.pop("question_type", None)
    shared.pop("prediction", None)
    shared.pop("confidence", None)
    shared.pop("probabilities", None)
    shared.pop("label", None)
    shared.pop("label_source", None)
    if "ts" in shared:
        shared["ts"] = _coerce_timestamp(shared["ts"])
    return shared


def records_from_row(row: Mapping[str, Any], mapping: IngestMap) -> list[DecisionRecord]:
    """Build one record per question carried by a raw row.

    A row may carry several questions; per-question fields live under the configured
    ``questions_field`` list and override the row defaults. Aggregating several questions
    into one record would hide the per-question failure modes this tool exists to expose.
    """
    shared = _shared_fields(row, mapping)
    questions = row.get(mapping.questions_field)
    if isinstance(questions, list) and questions:
        payloads: list[Mapping[str, Any]] = questions
        records: list[DecisionRecord] = []
        for payload in payloads:
            merged = {**shared, **{k: v for k, v in payload.items() if v not in (None, "")}}
            records.append(normalize_record(merged))
        return records
    return [normalize_record(mapping.apply(row))]


def ingest_files(
    inputs: Sequence[Path | str],
    out_path: Path | str,
    mapping: IngestMap,
    *,
    append: bool = False,
) -> IngestReport:
    """Read every input file and append the resulting records to ``out_path``."""
    report = IngestReport(out_path=Path(out_path))
    records: list[DecisionRecord] = []
    for source in inputs:
        for row in iter_rows(source):
            report.n_rows += 1
            try:
                built = records_from_row(row, mapping)
            except Exception as exc:
                report.n_skipped += 1
                if len(report.errors) < 10:
                    report.errors.append(f"{Path(source).name} row {report.n_rows}: {exc}")
                continue
            seen: set[str] = set()
            for record in built:
                if record.question_key in seen:
                    report.n_duplicate_questions += 1
                    continue
                seen.add(record.question_key)
                if not record.is_labeled:
                    report.n_unlabeled += 1
                report.per_question[record.question_key] = (
                    report.per_question.get(record.question_key, 0) + 1
                )
                records.append(record)

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with target.open(mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
    report.n_records = len(records)
    return report
