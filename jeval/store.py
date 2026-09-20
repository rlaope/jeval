"""JSONL persistence. Files on disk, no database."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from jeval.schema import DecisionRecord

DATA_DIR_NAME = ".jeval"
RECORDS_FILE_NAME = "records.jsonl"
CONFIG_FILE_NAME = "config.yaml"


def data_dir(root: Path | str = ".") -> Path:
    """Path to the ``.jeval`` working directory for a project root."""
    return Path(root) / DATA_DIR_NAME


def records_path(root: Path | str = ".") -> Path:
    return data_dir(root) / RECORDS_FILE_NAME


def config_path(root: Path | str = ".") -> Path:
    return data_dir(root) / CONFIG_FILE_NAME


def write_records(
    records: Iterable[DecisionRecord],
    path: Path | str,
    *,
    append: bool = False,
) -> int:
    """Write records as JSONL. Returns the number of records written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    mode = "a" if append else "w"
    with target.open(mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json(exclude_none=False) + "\n")
            count += 1
    return count


def iter_records(path: Path | str) -> Iterator[DecisionRecord]:
    """Yield records from a JSONL file, skipping blank lines."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"no records file at {source}")
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON ({exc.msg})") from exc
            yield DecisionRecord.model_validate(payload)


def read_records(path: Path | str) -> list[DecisionRecord]:
    return list(iter_records(path))


def load_records(root: Path | str = ".") -> list[DecisionRecord]:
    """Read the records file of a project, with a clear error when it is missing."""
    path = records_path(root)
    if not path.exists():
        raise FileNotFoundError(
            f"no decision records found at {path}. Run `jeval ingest <file>` first."
        )
    return read_records(path)
