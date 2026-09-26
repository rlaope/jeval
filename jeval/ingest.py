"""Ingest raw logs (JSONL or CSV) into decision records."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Collection, Iterable, Iterator, Mapping, MutableSequence, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jeval.config import IngestMap
from jeval.schema import ALL_LABEL_SOURCES, DecisionRecord, LabelSource, normalize_record
from jeval.store import read_records


@dataclass
class IngestReport:
    """What an ingest run did, in numbers a user can act on."""

    out_path: Path
    n_rows: int = 0
    n_records: int = 0
    n_skipped: int = 0
    n_unlabeled: int = 0
    n_duplicate_questions: int = 0
    n_impossible_labels: int = 0
    per_question: dict[str, int] = field(default_factory=dict)
    impossible_labels: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.n_records > 0


def _short(value: Any, limit: int = 40) -> str:
    """A value as it will appear inside a one-line error."""
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _row_error(exc: Exception) -> str:
    """One line naming what to change, instead of a validation dump.

    A mapping mistake is the likeliest reason a row is skipped, and the person reading it is holding
    a log whose field names they control. The pydantic dump names neither the field to fix nor the
    file to fix it in, so the same mistake gets read as "jeval is broken".
    """
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return str(exc)
    parts: list[str] = []
    for item in errors():
        if not isinstance(item, Mapping):
            continue
        located = item.get("loc") or ()
        field = ".".join(str(piece) for piece in located) or "record"
        kind = str(item.get("type", ""))
        if kind == "missing":
            parts.append(f"nothing to use for {field!r}: map it in field_map or set it in defaults")
        elif kind == "dict_type":
            parts.append(
                f"{field!r} must be an object of names, got {_short(item.get('input'))}: map an "
                f"object column, or map a flat column and jeval names the segment after it"
            )
        else:
            parts.append(f"{field!r} {item.get('msg', kind)}: got {_short(item.get('input'))}")
    return "; ".join(parts) or str(exc)


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
    ``questions_field`` and override the row defaults. Aggregating several questions into one
    record would hide the per-question failure modes this tool exists to expose.

    The container may be a list of question objects, or an object keyed by question name — the
    shape a decision API returns (``{"answers": {"department": {...}, "is_urgent": {...}}}``).
    The key becomes the question key when the payload does not name one. ``questions_field``
    accepts a dotted path, so a container nested inside a response object is reachable.
    """
    shared = _shared_fields(row, mapping)
    questions = _read_path(row, mapping.questions_field)
    if questions is _MISSING:
        questions = row.get(mapping.questions_field)
    if isinstance(questions, Mapping) and questions:
        # A container keyed by question name, as a decision API returns it: the key is the
        # question key unless the payload names one itself.
        questions = [
            {**dict(payload), "question_key": dict(payload).get("question_key", name)}
            if isinstance(payload, Mapping)
            else payload
            for name, payload in questions.items()
        ]
    if isinstance(questions, list) and questions:
        payloads: list[Mapping[str, Any]] = questions
        records: list[DecisionRecord] = []
        base_id = shared.get("id")
        for payload in payloads:
            merged = {**shared, **{k: v for k, v in payload.items() if v not in (None, "")}}
            if base_id is not None and len(payloads) > 1:
                # A request's questions are separate decisions, so they cannot share an id: the
                # harvest and the labeling sheet both address records BY id, and a duplicated id
                # made one answer land on every question of the request.
                merged["id"] = f"{base_id}:{merged.get('question_key', 'question')}"
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
                    report.errors.append(
                        f"{Path(source).name} row {report.n_rows}: {_row_error(exc)}"
                    )
                continue
            seen: set[str] = set()
            for record in built:
                if record.question_key in seen:
                    report.n_duplicate_questions += 1
                    continue
                seen.add(record.question_key)
                record = _drop_impossible_label(record, report)
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


def ingest_preset_files(
    inputs: Sequence[Path | str],
    out_path: Path | str,
    preset: Any,
    *,
    append: bool = False,
    keys: Mapping[str, str] | None = None,
    source_key_field: str | None = None,
) -> IngestReport:
    """Ingest a log a product already writes, using a preset to name its shape.

    Rows that carry no recognizable response are skipped and counted with the location the preset
    looked in, rather than being forced into a record that would measure the wrong thing.
    """
    from jeval import presets as preset_module
    from jeval.presets import rows_to_payloads

    report = IngestReport(out_path=Path(out_path))
    records: list[DecisionRecord] = []
    for source in inputs:
        for row in iter_rows(source):
            report.n_rows += 1
            try:
                payloads = rows_to_payloads(
                    preset, row, keys=keys, source_key_field=source_key_field
                )
            except Exception as exc:
                report.n_skipped += 1
                if len(report.errors) < 10:
                    report.errors.append(
                        f"{Path(source).name} row {report.n_rows}: {_row_error(exc)}"
                    )
                continue
            if not payloads:
                report.n_skipped += 1
                if len(report.errors) < 10:
                    report.errors.append(
                        f"{Path(source).name} row {report.n_rows}: "
                        f"{preset_module.skip_reason(preset, row)}"
                    )
                continue
            for payload in payloads:
                try:
                    built = normalize_record(payload)
                except Exception as exc:
                    report.n_skipped += 1
                    if len(report.errors) < 10:
                        report.errors.append(
                            f"{Path(source).name} row {report.n_rows}: {_row_error(exc)}"
                        )
                    continue
                built = _drop_impossible_label(built, report)
                if not built.is_labeled:
                    report.n_unlabeled += 1
                report.per_question[built.question_key] = (
                    report.per_question.get(built.question_key, 0) + 1
                )
                records.append(built)

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with target.open(mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
    report.n_records = len(records)
    return report


_MISSING = object()
_JSON = json.JSONDecoder()


def _read_path(row: Mapping[str, Any], path: str) -> Any:
    """Read a dotted path (``resolution.final_department``) out of a raw row.

    Returns ``_MISSING`` when any segment is absent, so "no such column" and "column present
    but empty" stay distinguishable: the first is a mapping mistake worth an error, the second
    is just a row without an answer yet.
    """
    current: Any = row
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _label_text(value: Any) -> str:
    """Render an external label as the string the schema stores."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value).strip()


def _describe_columns(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = sorted({str(key) for row in rows for key in row})
    return ", ".join(columns) if columns else "none"


@dataclass(frozen=True)
class LabelApplication:
    """One record that received an external label."""

    record_id: str
    question_key: str
    label: str
    label_source: LabelSource


@dataclass
class HarvestReport:
    """What a label harvest did: applied, unmatched, and skipped, never a silent drop."""

    field: str
    source: LabelSource
    join_on: str
    question: str | None = None
    n_records: int = 0
    n_label_rows: int = 0
    n_applied: int = 0
    n_unmatched: int = 0
    n_skipped_without_key: int = 0
    n_kept_existing: int = 0
    n_rows_without_key: int = 0
    n_rows_without_label: int = 0
    n_other_question: int = 0
    n_unlisted_label: int = 0
    applications: tuple[LabelApplication, ...] = ()
    unmatched_keys: tuple[str, ...] = ()
    kept_existing_question_keys: tuple[str, ...] = ()
    unlisted_labels: tuple[str, ...] = ()

    @property
    def applied_question_keys(self) -> tuple[str, ...]:
        """The questions that gained at least one label, sorted."""
        return tuple(sorted({application.question_key for application in self.applications}))

    @property
    def ok(self) -> bool:
        return self.n_applied > 0


def harvest_labels(
    records: MutableSequence[DecisionRecord],
    label_rows: Iterable[Mapping[str, Any]],
    *,
    field: str,
    source: LabelSource,
    join_on: str,
    overwrite: bool = False,
    question: str | None = None,
    allow_unlisted: bool = False,
) -> HarvestReport:
    """Apply external human answers to records that already exist.

    ``field`` is the column (or dotted path) in ``label_rows`` that holds the answer,
    ``join_on`` is the column that holds the key tying a label row back to a record's
    ``source_key``, and ``source`` is the provenance recorded next to the answer. The labels
    are the ones the user is already producing: the human answer on an escalated case, the
    reversal of an auto-processed one.

    Records are updated in place, so the caller can hand the same list to the report or to
    :func:`rewrite_labels_atomic`. That means labels appear on the sequence that was passed:
    a reference to a record taken *before* the harvest is not updated. Every applied value
    goes through schema validation, so a ``noul`` answer of ``"FALSE"`` is stored as ``"no"``.

    Existing labels are never replaced unless ``overwrite=True``: a silver label silently
    overwriting a human review would be the most expensive bug this tool could have, so those
    records are counted in ``n_kept_existing`` and named in ``kept_existing_question_keys``
    instead. Label rows that match no record are counted in ``n_unmatched`` and named in
    ``unmatched_keys``; they are a symptom (a stale key, a wrong column) and are never dropped
    quietly.

    Two guards stop this function from writing a label that is merely well-formed:

    ``question`` scopes the harvest to one question. A join key is usually shared by every
    question of a request, so ``resolution.final_department`` joined on ``ticket_id`` matches the
    request's *intent* record too, and without scoping the department answer would be written as
    the intent answer. Records of another question are counted in ``n_other_question``.

    A label is also refused when the record itself proves it cannot be one: a ``choice`` label
    outside the record's own ``probabilities`` keys is not something that question can answer, and
    a ``noul`` label outside yes/no is not a yes/no answer. Those are counted in
    ``n_unlisted_label`` and named in ``unlisted_labels``. ``allow_unlisted=True`` lifts the second
    guard for classifiers whose ``probabilities`` map lists only the top candidates.
    """
    if not field.strip() or not join_on.strip():
        raise ValueError("harvest_labels needs both a label field and a join key")
    if source not in ALL_LABEL_SOURCES:
        allowed = ", ".join(ALL_LABEL_SOURCES)
        raise ValueError(f"label source {source!r} is not one of {allowed}")

    rows = [dict(row) for row in label_rows]
    if rows and not any(_read_path(row, join_on) is not _MISSING for row in rows):
        raise ValueError(
            f"label rows have no {join_on!r} column to join on; columns present: "
            f"{_describe_columns(rows)}"
        )
    if rows and not any(_read_path(row, field) is not _MISSING for row in rows):
        raise ValueError(
            f"label rows have no {field!r} field; columns present: {_describe_columns(rows)}"
        )

    report = HarvestReport(
        field=field,
        source=source,
        join_on=join_on,
        question=question,
        n_records=len(records),
        n_label_rows=len(rows),
    )

    labels_by_key: dict[str, str] = {}
    for row in rows:
        raw_key = _read_path(row, join_on)
        key = "" if raw_key is _MISSING or raw_key is None else str(raw_key).strip()
        if not key:
            report.n_rows_without_key += 1
            continue
        raw_label = _read_path(row, field)
        if raw_label is _MISSING or raw_label is None:
            report.n_rows_without_label += 1
            continue
        text = _label_text(raw_label)
        if not text:
            report.n_rows_without_label += 1
            continue
        # A resolution log that answers the same case twice is applied in file order.
        labels_by_key[key] = text

    applications: list[LabelApplication] = []
    matched: set[str] = set()
    kept: list[str] = []
    unlisted: set[str] = set()
    for index, record in enumerate(records):
        key = record.source_key.strip() if record.source_key else ""
        if not key:
            report.n_skipped_without_key += 1
            continue
        if key not in labels_by_key:
            continue
        if question is not None and record.question_key != question:
            report.n_other_question += 1
            continue
        matched.add(key)
        if record.label is not None and not overwrite:
            report.n_kept_existing += 1
            kept.append(record.question_key)
            continue
        updated = DecisionRecord.model_validate(
            {**record.model_dump(), "label": labels_by_key[key], "label_source": source}
        )
        if updated.label is None:  # pragma: no cover - the schema keeps any non-empty answer
            report.n_rows_without_label += 1
            continue
        if not allow_unlisted and not _label_is_possible(record, updated.label):
            report.n_unlisted_label += 1
            unlisted.add(f"{key}:{record.question_key}={updated.label}")
            continue
        records[index] = updated
        report.n_applied += 1
        applications.append(
            LabelApplication(
                record_id=updated.id,
                question_key=updated.question_key,
                label=updated.label,
                label_source=source,
            )
        )

    report.applications = tuple(applications)
    report.unmatched_keys = tuple(sorted(set(labels_by_key) - matched))
    report.n_unmatched = len(report.unmatched_keys)
    report.kept_existing_question_keys = tuple(sorted(set(kept)))
    report.unlisted_labels = tuple(sorted(unlisted))
    return report


def _drop_impossible_label(record: DecisionRecord, report: IngestReport) -> DecisionRecord:
    """Refuse an inline label the record's own question could not have produced.

    The harvest path already refuses these, but a label that arrives inside the log skipped the
    check entirely — and an impossible label is worse than a missing one, because it is counted as
    a wrong answer forever and drags the measured accuracy down while looking like real ground
    truth. The prediction is kept: it is legitimate evidence either way.
    """
    if record.label is None or _label_is_possible(record, record.label):
        return record
    report.n_impossible_labels += 1
    if len(report.impossible_labels) < 10:
        report.impossible_labels.append(f"{record.question_key}={record.label!r}")
    return DecisionRecord.model_validate(
        {**record.model_dump(), "label": None, "label_source": None}
    )


@dataclass
class LabelEventReport:
    """What happened to the answers ``collect.resolve`` recorded, when they were joined on read."""

    n_events: int = 0
    n_applied: int = 0
    n_kept_existing: int = 0
    n_impossible: int = 0
    n_unmatched: int = 0
    n_invalid: int = 0

    def summary(self) -> str:
        """One line for the terminal, naming every answer that did not become a label."""
        parts = [f"{self.n_applied} applied"]
        for count, words in (
            (self.n_kept_existing, "kept an existing label"),
            (self.n_impossible, "not an answer the question can give"),
            (self.n_unmatched, "matched no decision"),
            (self.n_invalid, "unreadable"),
        ):
            if count:
                parts.append(f"{count} {words}")
        return f"labels: {self.n_events} answers from labels.jsonl, " + ", ".join(parts)


def apply_label_events(
    records: MutableSequence[DecisionRecord],
    events: Iterable[Mapping[str, Any]],
) -> LabelEventReport:
    """Join run-time answers onto records in memory, by ``(source_key, question_key)``.

    The same guards as :func:`harvest_labels`: a record that already carries a label keeps it (a
    later silver answer must never replace a human review), and an answer the record's own question
    could not have produced is refused. When one case is answered twice, the later line wins —
    an agent correcting their own resolution is the common reason. Nothing is written back: the
    records file stays what the collector wrote, and the join is repeated on every read.
    """
    report = LabelEventReport()
    latest: dict[tuple[str, str], tuple[str, str]] = {}
    for event in events:
        report.n_events += 1
        key = str(event.get("source_key") or "").strip()
        question = str(event.get("question_key") or "").strip()
        label = str(event.get("label") or "").strip()
        source = str(event.get("label_source") or "human_review")
        if not key or not question or not label or source not in ALL_LABEL_SOURCES:
            report.n_invalid += 1
            continue
        latest[(key, question)] = (label, source)

    matched: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        pair = ((record.source_key or "").strip(), record.question_key)
        if pair not in latest:
            continue
        matched.add(pair)
        label, source = latest[pair]
        if record.label is not None:
            report.n_kept_existing += 1
            continue
        if not _label_is_possible(record, label):
            report.n_impossible += 1
            continue
        records[index] = DecisionRecord.model_validate(
            {**record.model_dump(), "label": label, "label_source": source}
        )
        report.n_applied += 1
    report.n_unmatched = len(latest) - len(matched)
    return report


def _label_is_possible(record: DecisionRecord, label: str) -> bool:
    """Whether ``label`` is an answer this record's own question could have produced."""
    if record.question_type == "choice":
        candidates = record.probabilities
        if not candidates:
            return True  # nothing to check against; the caller's mapping is all we have
        return label in candidates
    if record.question_type == "noul":
        return label.strip().lower() in {"yes", "no"}
    if record.question_type == "score":
        try:
            float(label)
        except (TypeError, ValueError):
            return False
        return True


def harvest_file(
    records_path: Path | str,
    label_rows: Iterable[Mapping[str, Any]],
    *,
    field: str,
    source: LabelSource,
    join_on: str,
    overwrite: bool = False,
    question: str | None = None,
    allow_unlisted: bool = False,
) -> HarvestReport:
    """Harvest labels into a records file and rewrite that file atomically."""
    path = Path(records_path)
    records = read_records(path)
    report = harvest_labels(
        records,
        label_rows,
        field=field,
        source=source,
        join_on=join_on,
        overwrite=overwrite,
        question=question,
        allow_unlisted=allow_unlisted,
    )
    if report.applications:
        rewrite_labels_atomic(path, report.applications)
    return report


def _skip_whitespace(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    return index


def _top_level_spans(line: str, keys: Collection[str]) -> dict[str, tuple[int, int]]:
    """Spans of the raw JSON values for ``keys`` in one flat, single-line JSON object."""
    wanted = set(keys)
    spans: dict[str, tuple[int, int]] = {}
    index = _skip_whitespace(line, 0)
    if index >= len(line) or line[index] != "{":
        return spans
    index += 1
    while True:
        index = _skip_whitespace(line, index)
        if index >= len(line) or line[index] == "}":
            return spans
        if line[index] == ",":
            index += 1
            continue
        try:
            key, index = _JSON.raw_decode(line, index)
        except json.JSONDecodeError:
            return spans
        if not isinstance(key, str):
            return spans
        index = _skip_whitespace(line, index)
        if index >= len(line) or line[index] != ":":
            return spans
        index = _skip_whitespace(line, index + 1)
        try:
            _, end = _JSON.raw_decode(line, index)
        except json.JSONDecodeError:
            return spans
        if key in wanted:
            spans[key] = (index, end)
        index = end


def _set_json_field(line: str, key: str, value: Any) -> str:
    """Replace one top-level field of a JSON line, leaving every other byte alone."""
    encoded = json.dumps(value, ensure_ascii=False)
    span = _top_level_spans(line, (key,)).get(key)
    if span is not None:
        return line[: span[0]] + encoded + line[span[1] :]
    body = line.rstrip("\r\n")
    tail = line[len(body) :]
    close = body.rfind("}")
    if close == -1:
        return line
    separator = "" if body[:close].rstrip().endswith("{") else ","
    addition = f"{separator}{json.dumps(key)}:{encoded}"
    return body[:close] + addition + body[close:] + tail


def _write_atomically(target: Path, text: str) -> None:
    """Replace ``target`` with ``text`` through a temp file in the same directory."""
    handle, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temp = Path(temp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def rewrite_labels_atomic(
    path: Path | str,
    applications: Iterable[LabelApplication],
) -> int:
    """Write harvested labels into a records file, atomically. Returns the lines changed.

    Only the ``label`` and ``label_source`` values of the named records are touched: every
    other byte of the file — every other field, the key order, the number formatting, the
    lines of records that were not labeled — is copied through untouched. ``records.jsonl`` is
    the user's data file, so a harvest must not be an excuse to reformat it.

    The new content is written to a temp file next to the original and moved into place with
    ``os.replace``, so an interrupted run leaves either the complete old file or the complete
    new one. A caller that expects every application to land in the file should compare the
    returned count with ``len(applications)``.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"no records file at {target}")
    updates = {application.record_id: application for application in applications}
    if not updates:
        return 0

    # Read the bytes: `read_text` applies universal-newline translation, which silently rewrote
    # every CRLF line ending in the user's file while the docstring promised untouched bytes.
    lines = target.read_bytes().decode("utf-8").splitlines(keepends=True)
    rebuilt: list[str] = []
    changed = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            rebuilt.append(line)
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{target}: invalid JSON ({exc.msg})") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{target}: expected a JSON object per line")
        update = updates.get(str(payload.get("id")))
        if update is None:
            rebuilt.append(line)
            continue
        patched = _set_json_field(line, "label", update.label)
        patched = _set_json_field(patched, "label_source", update.label_source)
        changed += 1
        rebuilt.append(patched)
    _write_atomically(target, "".join(rebuilt))
    return changed
