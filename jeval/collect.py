"""Collect decisions where they are made, so nobody has to write a logging line.

The problem this solves: a decision API returns typed answers with probabilities, but nothing
persists them. The call site is the only place that knows the trace id, the ticket id and how long
the call took, and it is also the only place that sees the response. So collection belongs in a
wrapper, and the wrapper belongs to the caller's process.

Provider neutrality is the point: :func:`record` takes fields, not products, and
:func:`answer_payloads` recognizes a shape (typed answers keyed by question name) rather than a
vendor. A preset names the keys when a product spells them differently; see :mod:`jeval.presets`.

Everything here is best effort. A decision tool that breaks the application it measures is worse
than no tool, so nothing raises out of this module: failures are counted and readable from
:func:`stats`. Set ``JEVAL_COLLECT=0`` to switch collection off, or set it to a path to choose
where records land.
"""

from __future__ import annotations

import functools
import inspect
import json
import os
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jeval.schema import DecisionRecord, normalize_record
from jeval.store import records_path

ENV_FLAG = "JEVAL_COLLECT"
ENV_ROOT = "JEVAL_ROOT"
ENV_MODEL = "JEVAL_MODEL"

_OFF = frozenset({"0", "off", "false", "no"})
_ON = frozenset({"1", "on", "true", "yes"})

_TRACKED_ATTR = "_jeval_tracked"

_STATS: dict[str, int] = {
    "written": 0,
    "dropped": 0,
    "unsupported": 0,
    "calls": 0,
    "already_tracked": 0,
    "install_failed": 0,
    "unknown_flag_value": 0,
}


def stats() -> dict[str, int]:
    """Counter snapshot: what was written, what was dropped, what shape was not understood."""
    return dict(_STATS)


def reset_stats() -> None:
    """Zero the counters. Intended for tests and for long-running processes that rotate logs."""
    for key in _STATS:
        _STATS[key] = 0


def target_path(explicit: str | Path | None = None) -> Path | None:
    """Where records should go, or ``None`` when collection is switched off.

    ``JEVAL_COLLECT=0`` disables it. ``JEVAL_COLLECT=/some/records.jsonl`` sends records there.
    With the flag unset, records go to ``$JEVAL_ROOT/.jeval/records.jsonl`` (the current directory
    by default), which is the same file ``jeval report`` already reads.
    """
    # The off switch wins over everything, including an explicit path: a user killing collection
    # during an incident must not have records land because one call site passed a path.
    flag = os.environ.get(ENV_FLAG)
    if flag is not None and flag.strip().lower() in _OFF:
        return None
    if explicit is not None:
        return Path(explicit)
    if flag is not None and flag.strip():
        candidate = flag.strip()
        if candidate.lower() not in _ON:
            if _looks_like_a_path(candidate):
                return Path(candidate)
            # `JEVAL_COLLECT=enabled` is not a filename. Counted, then treated as "on".
            _STATS["unknown_flag_value"] += 1
    return records_path(os.environ.get(ENV_ROOT, "."))


def _looks_like_a_path(value: str) -> bool:
    """Whether a flag value names a file, rather than being a truthy word typed by mistake."""
    return (
        os.path.isabs(value)
        or "/" in value
        or "\\" in value
        or value.lower().endswith((".jsonl", ".json", ".ndjson", ".log"))
    )


def enabled() -> bool:
    """Whether collection is currently switched on."""
    return os.environ.get(ENV_FLAG, "").strip().lower() not in _OFF


def _now() -> datetime:
    return datetime.now(timezone.utc)


def record(
    *,
    question_key: str,
    prediction: str,
    question_type: str | None = None,
    confidence: float | None = None,
    probabilities: Mapping[str, Any] | None = None,
    model: str | None = None,
    source_key: str | None = None,
    segment: Mapping[str, str] | None = None,
    state_tokens: int | None = None,
    latency_ms: float | None = None,
    cost_usd: float | None = None,
    label: str | None = None,
    label_source: str | None = None,
    ts: datetime | None = None,
    path: str | Path | None = None,
) -> bool:
    """Append one decision record. Returns whether it was written; never raises."""
    destination = target_path(path)
    if destination is None:
        return False
    payload: dict[str, Any] = {
        "ts": ts or _now(),
        "model": model or os.environ.get(ENV_MODEL) or "unknown",
        "question_key": question_key,
        "prediction": prediction,
    }
    if question_type:
        payload["question_type"] = question_type
    for name, value in (
        ("confidence", confidence),
        ("probabilities", dict(probabilities) if probabilities else None),
        ("source_key", source_key),
        ("segment", dict(segment) if segment else None),
        ("state_tokens", state_tokens),
        ("latency_ms", latency_ms),
        ("cost_usd", cost_usd),
        ("label", label),
        ("label_source", label_source),
    ):
        if value is not None:
            payload[name] = value
    try:
        built = normalize_record(payload)
    except Exception:
        _STATS["dropped"] += 1
        return False
    return _append(built, destination)


def record_many(payloads: Sequence[Mapping[str, Any]], *, path: str | Path | None = None) -> int:
    """Append several records from canonical payloads. Returns how many were written."""
    written = 0
    for payload in payloads:
        destination = target_path(path)
        if destination is None:
            return written
        try:
            built = normalize_record(payload)
        except Exception:
            _STATS["dropped"] += 1
            continue
        if _append(built, destination):
            written += 1
    return written


def _append(record: DecisionRecord, destination: Path) -> bool:
    """One JSONL line, best effort: a failed write must never reach the caller."""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")
    except Exception:
        _STATS["dropped"] += 1
        return False
    _STATS["written"] += 1
    return True


def answer_payloads(
    answers: Mapping[str, Any],
    *,
    model: str | None = None,
    usage: Mapping[str, Any] | None = None,
    source_key: str | None = None,
    segment: Mapping[str, str] | None = None,
    latency_ms: float | None = None,
    keys: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Canonical payloads for a container of typed answers keyed by question name.

    ``keys`` renames the fields when a product spells them differently (``type`` for
    ``question_type``, ``choice`` for ``prediction``). An unrecognizable answer is counted in
    ``unsupported`` and skipped rather than guessed at.

    Every answer type maps onto the same three fields jeval measures with, so nothing downstream
    needs to know which product produced the log:

    - ``choice``: the winning key becomes the prediction, with the full distribution kept.
    - ``noul``: the probability is kept as ``{yes, no}`` and the confidence is derived from it.
    - ``score``: the reported score becomes the prediction, which is measured as error and rank
      agreement rather than as right or wrong.
    """
    names = dict(keys or {})
    type_key = names.get("question_type", "type")
    prediction_keys = {
        "choice": names.get("choice", "choice"),
        "score": names.get("score", "score"),
    }
    noul_key = names.get("noul", "noul")
    probability_key = names.get("probabilities", "probabilities")
    confidence_key = names.get("confidence", "confidence")

    payloads: list[dict[str, Any]] = []
    for question_key, raw in answers.items():
        answer = dict(raw) if isinstance(raw, Mapping) else {}
        question_type = str(answer.get(type_key, "choice")).strip().lower()
        probabilities = answer.get(probability_key)
        common: dict[str, Any] = {
            "question_key": str(question_key),
            "question_type": question_type,
        }
        # Never omit the model: a record without one cannot be validated, so the whole answer
        # would be dropped silently — total data loss from a response that merely omitted a field.
        common["model"] = model or os.environ.get(ENV_MODEL) or "unknown"
        if source_key is not None:
            common["source_key"] = str(source_key)
        if segment:
            common["segment"] = dict(segment)
        if latency_ms is not None:
            common["latency_ms"] = latency_ms
        if isinstance(usage, Mapping):
            tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
            if isinstance(tokens, int):
                common["state_tokens"] = tokens
        if isinstance(probabilities, Mapping) and probabilities:
            common["probabilities"] = dict(probabilities)
        for schema_name, default_key in (("label", "label"), ("label_source", "label_source")):
            answer_key = names.get(schema_name, default_key)
            value = answer.get(answer_key)
            if value not in (None, ""):
                common[schema_name] = str(value)

        confidence = answer.get(confidence_key)
        if question_type == "noul":
            probability = answer.get(noul_key)
            if probability is None and isinstance(probabilities, Mapping):
                probability = probabilities.get("yes")
            if probability is None:
                _STATS["unsupported"] += 1
                continue
            positive = float(probability)
            common["probabilities"] = {"yes": positive, "no": 1.0 - positive}
            common["probability_positive"] = positive
            common["prediction"] = "yes" if positive > 0.5 else "no"
        elif question_type == "score":
            score = answer.get(prediction_keys["score"])
            if score is None:
                _STATS["unsupported"] += 1
                continue
            common["prediction"] = str(score)
            if isinstance(confidence, (int, float)):
                common["confidence"] = float(confidence)
        else:
            winner = answer.get(prediction_keys["choice"])
            if winner is None:
                _STATS["unsupported"] += 1
                continue
            common["prediction"] = str(winner)
            if isinstance(confidence, (int, float)):
                common["confidence"] = float(confidence)
        payloads.append(common)
    return payloads


def response_payloads(
    response: Mapping[str, Any],
    *,
    source_key: str | None = None,
    keys: Mapping[str, str] | None = None,
    container: str = "answers",
) -> list[dict[str, Any]]:
    """Canonical payloads for a whole response object, model and usage included."""
    answers = response.get(container)
    if not isinstance(answers, Mapping) or not answers:
        return []
    model = response.get("model")
    usage = response.get("usage")
    return answer_payloads(
        answers,
        model=str(model) if model else None,
        usage=usage if isinstance(usage, Mapping) else None,
        source_key=source_key,
        keys=keys,
    )


def track(
    client: Any,
    *,
    source_key: str | Callable[..., Any] | None = None,
    keys: Mapping[str, str] | None = None,
    container: str = "answers",
    method_names: Sequence[str] = ("system_one", "systemOne", "systemone"),
    path: str | Path | None = None,
    clock: Callable[[], datetime] = _now,
) -> Any:
    """Patch a client so every answered call is recorded, then return it.

    ``source_key`` may be a literal, or a callable receiving the call's keyword arguments — the
    trace id or ticket id usually arrives as an argument, and it is what later lets a human answer
    be joined back onto the decision.

    The client is patched in place and returned, so ``client = track(Client())`` and
    ``track(client)`` both work. Tracking is **idempotent**: a client that is already tracked is
    left alone, because wrapping twice would record every call twice and silently double-weight
    the whole log. The second call is counted in ``stats()["already_tracked"]``. Nothing here
    imports a vendor SDK: the wrapper looks for a method that takes questions and returns typed
    answers.
    """
    if not enabled():
        return client
    for method_name in method_names:
        try:
            original = getattr(client, method_name, None)
            if not callable(original):
                continue
            if getattr(original, _TRACKED_ATTR, False) is True:
                _STATS["already_tracked"] += 1
                return client
            wrapper = _wrap(
                original,
                source_key=source_key,
                keys=keys,
                container=container,
                path=path,
                clock=clock,
            )
            setattr(wrapper, _TRACKED_ATTR, True)
            setattr(client, method_name, wrapper)
            # Verify the install: a read-only facade whose __setattr__ drops writes raises
            # nothing, so without this readback tracking would silently collect nothing.
            if getattr(getattr(client, method_name, None), _TRACKED_ATTR, False) is not True:
                _STATS["install_failed"] += 1
                return client
        except Exception:
            # A property that raises, __slots__, a frozen dataclass, a pydantic model whose
            # __setattr__ refuses: all of them must leave the caller's client working, unwrapped.
            _STATS["install_failed"] += 1
            return client
        break
    return client


def _wrap(
    original: Callable[..., Any],
    *,
    source_key: str | Callable[..., Any] | None,
    keys: Mapping[str, str] | None,
    container: str,
    path: str | Path | None,
    clock: Callable[[], datetime],
) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            started = clock()
            response = await original(*args, **kwargs)
            _capture(response, args, kwargs, source_key, keys, container, path, started, clock)
            return response

        return async_wrapper

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        started = clock()
        response = original(*args, **kwargs)
        _capture(response, args, kwargs, source_key, keys, container, path, started, clock)
        return response

    return wrapper


def _capture(
    response: Any,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    source_key: str | Callable[..., Any] | None,
    keys: Mapping[str, str] | None,
    container: str,
    path: str | Path | None,
    started: datetime,
    clock: Callable[[], datetime],
) -> None:
    """Record one call's answers. Every failure mode is counted, none is raised."""
    _STATS["calls"] += 1
    try:
        if hasattr(response, "model_dump"):
            response = response.model_dump()
        if not isinstance(response, Mapping):
            _STATS["unsupported"] += 1
            return
        key: str | None = None
        if callable(source_key):
            resolved = source_key(*args, **kwargs)
            key = None if resolved is None else str(resolved)
        elif source_key is not None:
            key = str(source_key)
        latency = (clock() - started).total_seconds() * 1000.0
        payloads = response_payloads(response, source_key=key, keys=keys, container=container)
        if not payloads:
            _STATS["unsupported"] += 1
            return
        for payload in payloads:
            payload.setdefault("latency_ms", round(latency, 1))
            payload["ts"] = started
        record_many(payloads, path=path)
    except Exception:
        _STATS["dropped"] += 1


def dump(payloads: Sequence[Mapping[str, Any]]) -> str:
    """Render payloads as JSONL, the format the collector writes and ``jeval ingest`` reads."""
    return "".join(json.dumps(dict(payload), default=str) + "\n" for payload in payloads)
