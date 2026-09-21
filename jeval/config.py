"""Project configuration and ingest mapping files."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from jeval.calibration import DEFAULT_ALPHA, DEFAULT_N_BINS
from jeval.schema import ALL_LABEL_SOURCES
from jeval.store import DATA_DIR_NAME

DEFAULT_CONFIG: dict[str, Any] = {
    "version": 1,
    "bins": DEFAULT_N_BINS,
    "bins_equal_width": False,
    "bootstrap_samples": 1000,
    "alpha": DEFAULT_ALPHA,
    "by": [],
    "min_segment_size": 10,
    "ingest_map": "ingest-map.yaml",
}


@dataclass(frozen=True)
class Config:
    """Runtime settings, loaded from ``.jeval/config.yaml`` with defaults filled in."""

    root: Path = Path(".")
    bins: int = DEFAULT_N_BINS
    bins_equal_width: bool = False
    bootstrap_samples: int = 1000
    alpha: float = DEFAULT_ALPHA
    by: tuple[str, ...] = ()
    min_segment_size: int = 10
    ingest_map: str = "ingest-map.yaml"
    raw: Mapping[str, Any] = field(default_factory=dict)


def write_default_config(root: Path | str, *, force: bool = False) -> Path:
    """Scaffold ``.jeval/config.yaml``. Existing files are kept unless ``force``."""
    directory = Path(root) / DATA_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "config.yaml"
    if target.exists() and not force:
        return target
    target.write_text(yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False), encoding="utf-8")
    return target


def load_config(root: Path | str = ".") -> Config:
    """Load configuration for a project root, falling back to defaults."""
    root_path = Path(root)
    path = root_path / DATA_DIR_NAME / "config.yaml"
    raw: dict[str, Any] = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            raw = loaded
    merged = {**DEFAULT_CONFIG, **raw}
    return Config(
        root=root_path,
        bins=int(merged["bins"]),
        bins_equal_width=bool(merged["bins_equal_width"]),
        bootstrap_samples=int(merged["bootstrap_samples"]),
        alpha=float(merged["alpha"]),
        by=tuple(str(item) for item in merged.get("by") or ()),
        min_segment_size=int(merged["min_segment_size"]),
        ingest_map=str(merged.get("ingest_map") or "ingest-map.yaml"),
        raw=merged,
    )


@dataclass(frozen=True)
class LabelFromSpec:
    """The ``label_from`` block of an ingest map, parsed and checked."""

    field: str
    source: str
    join_on: str
    question: str | None = None

    @property
    def known_source(self) -> bool:
        return self.source in ALL_LABEL_SOURCES


@dataclass(frozen=True)
class IngestMap:
    """How raw log fields map onto the decision-record schema."""

    field_map: Mapping[str, str] = field(default_factory=dict)
    defaults: Mapping[str, Any] = field(default_factory=dict)
    questions_field: str = "questions"
    label_from: Mapping[str, Any] = field(default_factory=dict)

    def resolve(self, row: Mapping[str, Any], field: str) -> Any:
        """Read ``field`` from a raw row, honouring the configured source name."""
        source = self.field_map.get(field, field)
        # A column the map already assigns to another field must not double as an implicit source
        # here. An application that logs its prediction in a column called ``label`` would otherwise
        # feed that same column to the schema's ``label``, producing a ground truth equal to the
        # prediction: a perfect score out of nothing, in the field this tool exists to measure.
        if source in self._claimed_elsewhere(field):
            return None
        return row.get(source)

    def _claimed_elsewhere(self, field: str) -> frozenset[str]:
        """Raw columns this map assigns to a schema field other than ``field``."""
        return frozenset(
            column for other, column in self.field_map.items() if other != field and column
        )

    def source_key_source(self) -> str:
        """Raw field name that carries the join key for free labels.

        ``field_map`` wins when it names a source for ``source_key``; otherwise
        ``label_from.join_on`` is used, because the key that ties a label row back to a request
        is normally called the same thing in the log as in the resolution file. With neither
        configured, a log that already uses the schema name is read as-is.
        """
        mapped = self.field_map.get("source_key")
        if mapped:
            return str(mapped)
        join_on = str(self.label_from.get("join_on") or "").strip()
        return join_on or "source_key"

    def label_spec(self) -> LabelFromSpec | None:
        """Parse ``label_from``; ``None`` when the block is absent.

        A half-filled block raises instead of being ignored: the README promises this mapping
        applies labels, so a silently dropped key would leave a user believing they had ground
        truth when they had none.
        """
        if not self.label_from:
            return None
        missing = [
            key
            for key in ("field", "source", "join_on")
            if not str(self.label_from.get(key) or "").strip()
        ]
        if missing:
            raise ValueError(
                f"ingest map: label_from is missing {', '.join(missing)} "
                "(it needs field, source and join_on)"
            )
        raw_question = str(self.label_from.get("question") or "").strip()
        spec = LabelFromSpec(
            field=str(self.label_from["field"]).strip(),
            source=str(self.label_from["source"]).strip(),
            join_on=str(self.label_from["join_on"]).strip(),
            question=raw_question or None,
        )
        if not spec.known_source:
            allowed = ", ".join(ALL_LABEL_SOURCES)
            raise ValueError(
                f"ingest map: label_from.source is {spec.source!r}, expected one of {allowed}"
            )
        return spec

    def apply(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Project a raw row onto schema field names, then add defaults."""
        payload: dict[str, Any] = {}
        for name in (
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
            "segment",
            "state_tokens",
            "latency_ms",
            "cost_usd",
        ):
            value = self.resolve(row, name)
            if value not in (None, ""):
                if name == "segment" and not isinstance(value, Mapping):
                    # A log that carries `lang: ko` as a column has no object to map, and rejecting
                    # the row made the documented `--by lang` unreachable for anyone whose segment
                    # is a flat field. The column name becomes the segment key.
                    value = {str(self.field_map.get("segment", "segment")): str(value)}
                payload[name] = value
        if "source_key" not in payload:
            key_field = self.source_key_source()
            value = row.get(key_field)
            if value not in (None, ""):
                payload["source_key"] = value
        for key, value in self.defaults.items():
            payload.setdefault(key, value)
        return payload


def load_ingest_map(path: Path | str) -> IngestMap:
    """Load an ingest mapping file. A missing file yields an identity mapping."""
    source = Path(path)
    if not source.exists():
        return IngestMap()
    loaded = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{source}: ingest map must be a YAML mapping")
    return IngestMap(
        field_map={str(k): str(v) for k, v in (loaded.get("field_map") or {}).items()},
        defaults=dict(loaded.get("defaults") or {}),
        questions_field=str(loaded.get("questions_field") or "questions"),
        label_from=dict(loaded.get("label_from") or {}),
    )
