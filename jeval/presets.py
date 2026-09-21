"""Ingest presets: where a product's spelling is written down.

The collector and the ingest core understand a *shape* — typed answers keyed by question name,
each carrying a distribution and a confidence. A product spells that shape its own way, and a
preset is the small data record that says how. Keeping presets here, and keeping the core free of
product conditionals, is what lets a new product be supported by adding a record instead of a
branch.

``jev-native`` describes a log line that carries a request and the response object beside it::

    {"request_id": "req_00042",
     "response": {"model": "jev-1.13.0",
                  "answers": {"department": {"type": "choice", "choice": "technical",
                                             "probabilities": {...}, "confidence": 0.82}},
                  "usage": {"input_tokens": 312, "output_tokens": 48}}}

The response's own ``model`` is what gets recorded, never the requested alias: a log that says
``jev-latest`` while the service answered ``jev-1.13.0`` is exactly the change jeval exists to
detect.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from jeval.collect import response_payloads


@dataclass(frozen=True)
class IngestPreset:
    """How one product spells the answer shape, and where its response sits in a log line."""

    name: str
    description: str
    container: str = "answers"
    response_field: str | None = "response"
    source_key_field: str | None = None
    keys: Mapping[str, str] = field(default_factory=dict)
    nesting: tuple[str, ...] = ()

    def response_of(self, row: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """The response object inside one log line, or ``None`` when it is not there."""
        candidates: list[str] = []
        if self.response_field:
            candidates.append(self.response_field)
        candidates.extend(self.nesting)
        fallback: Mapping[str, Any] | None = None
        for key in candidates:
            candidate: Any = row
            for part in key.split("."):
                if not isinstance(candidate, Mapping) or part not in candidate:
                    candidate = None
                    break
                candidate = candidate[part]
            if not isinstance(candidate, Mapping):
                continue
            if self.container in candidate:
                return candidate
            if fallback is None and key:
                # Kept so a response that exists but lacks the answers container can be reported
                # as exactly that, rather than as a missing response.
                fallback = candidate
        return fallback


JEV_NATIVE = IngestPreset(
    name="jev-native",
    description=(
        "A decision API's own response, recorded beside the request: typed answers keyed by "
        "question name, each with probabilities and a confidence."
    ),
    container="answers",
    response_field="response",
    source_key_field="request_id",
    nesting=("response", ""),
    keys={
        "question_type": "type",
        "choice": "choice",
        "noul": "noul",
        "score": "score",
        "probabilities": "probabilities",
        "confidence": "confidence",
    },
)

PRESETS: Mapping[str, IngestPreset] = {JEV_NATIVE.name: JEV_NATIVE}


def available() -> tuple[str, ...]:
    """Preset names, sorted, for error messages and `--help`."""
    return tuple(sorted(PRESETS))


def get(name: str) -> IngestPreset:
    """Look a preset up by name, naming the alternatives when it is unknown."""
    preset = PRESETS.get(name)
    if preset is None:
        raise ValueError(f"unknown preset {name!r}; available: {', '.join(available())}")
    return preset


def rows_to_payloads(
    preset: IngestPreset,
    row: Mapping[str, Any],
    *,
    keys: Mapping[str, str] | None = None,
    source_key_field: str | None = None,
) -> list[dict[str, Any]]:
    """Canonical payloads for one log line, or an empty list when it carries no response."""
    response = preset.response_of(row)
    if response is None:
        return []
    key_field = source_key_field if source_key_field is not None else preset.source_key_field
    source_key: str | None = None
    if key_field:
        value = _read(row, key_field)
        if value is not None and str(value).strip():
            source_key = str(value).strip()
    return response_payloads(
        response,
        source_key=source_key,
        keys=dict(keys) if keys is not None else dict(preset.keys),
        container=preset.container,
    )


def skip_reason(preset: IngestPreset, row: Mapping[str, Any]) -> str:
    """Why a log line produced nothing, in the terms this preset looks in.

    One message for six causes told the user the response was missing even when it was present and
    only its container had the wrong shape.
    """
    response = preset.response_of(row)
    if response is None:
        return (
            f"no response object at {preset.response_field!r} with answers under "
            f"{preset.container!r}"
        )
    container = response.get(preset.container)
    if container is None:
        return f"response has no {preset.container!r}"
    if not isinstance(container, Mapping):
        return f"{preset.container!r} is a {type(container).__name__}, not an object of answers"
    if not container:
        return f"{preset.container!r} is empty"
    return (
        f"every answer under {preset.container!r} was unusable: no recognizable type, answer or "
        "probability"
    )


def payloads_from_rows(
    preset: IngestPreset,
    rows: Sequence[Mapping[str, Any]],
    *,
    keys: Mapping[str, str] | None = None,
    source_key_field: str | None = None,
) -> list[dict[str, Any]]:
    """Concatenate one log line's payloads across many log lines."""
    payloads: list[dict[str, Any]] = []
    for row in rows:
        payloads.extend(rows_to_payloads(preset, row, keys=keys, source_key_field=source_key_field))
    return payloads


# Where a user's own spelling is expected upstream: jeval never guesses silently at a renamed key,
# it reports the keys it looked for.
def describe(preset: IngestPreset) -> str:
    """One-line summary of what a preset expects, for user-facing messages."""
    location = preset.response_field or "the row itself"
    # No name prefix: the CLI prints `preset: <name> (<this>)`, and repeating it read as
    # "preset: jev-native (jev-native: response at ...)".
    return (
        f"response {location!r}, answers {preset.container!r}, join key {preset.source_key_field!r}"
    )


def _read(row: Mapping[str, Any], path: str) -> Any:
    candidate: Any = row
    for part in path.split("."):
        if not isinstance(candidate, Mapping) or part not in candidate:
            return None
        candidate = candidate[part]
    return candidate
