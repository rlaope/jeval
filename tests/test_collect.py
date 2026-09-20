"""Tests for the collector: decisions recorded where they are made, best effort.

The contract under test is not just "it writes records" but "it can never be the reason a service
falls over": collection off is respected, a failed write is counted instead of raised, and a
response shape nobody understands is skipped rather than guessed at.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from jeval import collect
from jeval.store import read_records

JEV_RESPONSE: dict[str, Any] = {
    "model": "jev-1.13.0",
    "answers": {
        "department": {
            "type": "choice",
            "choice": "technical",
            "probabilities": {"billing": 0.08, "technical": 0.85, "sales": 0.07},
            "confidence": 0.82,
        },
        "is_urgent": {"type": "noul", "noul": 0.999},
        "satisfaction": {
            "type": "score",
            "score": 1.035,
            "probabilities": {"0": 0.0, "1": 0.96, "2": 0.04},
            "confidence": 0.61,
        },
    },
    "usage": {"input_tokens": 312, "output_tokens": 48},
}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No test may inherit another's env or counters."""
    monkeypatch.delenv(collect.ENV_FLAG, raising=False)
    monkeypatch.delenv(collect.ENV_ROOT, raising=False)
    monkeypatch.delenv(collect.ENV_MODEL, raising=False)
    collect.reset_stats()


# --- switching -------------------------------------------------------------------------------


def test_collection_writes_jsonl_that_the_rest_of_the_tool_reads(tmp_path: Path) -> None:
    target = tmp_path / "records.jsonl"
    written = collect.record(
        question_key="department",
        prediction="technical",
        probabilities={"technical": 0.85, "billing": 0.15},
        model="jev-1.13.0",
        source_key="req_1",
        path=target,
    )

    assert written is True
    stored = read_records(target)
    assert len(stored) == 1
    assert stored[0].question_key == "department"
    assert stored[0].model == "jev-1.13.0"
    assert stored[0].source_key == "req_1"
    assert stored[0].confidence == pytest.approx(0.85)  # top-1, from the distribution


def test_the_env_flag_switches_collection_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(collect.ENV_FLAG, "0")
    monkeypatch.setenv(collect.ENV_ROOT, str(tmp_path))

    assert collect.enabled() is False
    assert collect.target_path() is None
    assert collect.record(question_key="department", prediction="billing") is False
    assert not (tmp_path / ".jeval" / "records.jsonl").exists()


def test_the_env_flag_can_name_the_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    destination = tmp_path / "somewhere" / "records.jsonl"
    monkeypatch.setenv(collect.ENV_FLAG, str(destination))

    assert collect.record(
        question_key="department", prediction="billing", confidence=0.91, model="jev-1.13.0"
    )
    assert read_records(destination)[0].prediction == "billing"


def test_the_default_target_sits_under_the_project_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(collect.ENV_ROOT, str(tmp_path))

    assert collect.target_path() == tmp_path / ".jeval" / "records.jsonl"


def test_a_failed_write_is_counted_and_never_raised(tmp_path: Path) -> None:
    blocked = tmp_path / "a-file"
    blocked.write_text("not a directory", encoding="utf-8")
    result = collect.record(
        question_key="department", prediction="billing", path=blocked / "nested" / "records.jsonl"
    )

    assert result is False
    assert collect.stats()["dropped"] == 1


def test_a_decision_with_no_confidence_is_refused_rather_than_invented(tmp_path: Path) -> None:
    # A choice record needs a confidence or a distribution. Inventing either would fabricate the
    # one number the report is built on, so the record is dropped and counted instead.
    written = collect.record(
        question_key="department", prediction="billing", path=tmp_path / "r.jsonl"
    )

    assert written is False
    assert collect.stats()["dropped"] == 1
    assert not (tmp_path / "r.jsonl").exists()


def test_a_model_can_come_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(collect.ENV_MODEL, "jev-1.13.0")
    collect.record(
        question_key="department", prediction="billing", confidence=0.91, path=tmp_path / "r.jsonl"
    )

    assert read_records(tmp_path / "r.jsonl")[0].model == "jev-1.13.0"


# --- reading a response ----------------------------------------------------------------------


def test_every_answer_type_becomes_a_measurable_record() -> None:
    payloads = collect.response_payloads(JEV_RESPONSE, source_key="req_1")
    by_key = {payload["question_key"]: payload for payload in payloads}

    assert set(by_key) == {"department", "is_urgent", "satisfaction"}
    assert by_key["department"]["prediction"] == "technical"
    assert by_key["department"]["confidence"] == pytest.approx(0.82)
    assert by_key["is_urgent"]["prediction"] == "yes"
    assert by_key["is_urgent"]["probabilities"] == {"yes": 0.999, "no": pytest.approx(0.001)}
    assert by_key["satisfaction"]["prediction"] == "1.035"
    for payload in payloads:
        assert payload["model"] == "jev-1.13.0"  # the answered model, never the requested alias
        assert payload["state_tokens"] == 312
        assert payload["source_key"] == "req_1"


def test_an_answer_nobody_can_interpret_is_counted_not_guessed() -> None:
    response = {"model": "jev-1.13.0", "answers": {"mystery": {"type": "choice"}}}

    assert collect.response_payloads(response) == []
    assert collect.stats()["unsupported"] == 1


def test_a_response_without_answers_yields_nothing() -> None:
    assert collect.response_payloads({"model": "jev-1.13.0"}) == []
    assert collect.answer_payloads({}) == []


# --- wrapping a client -----------------------------------------------------------------------


class FakeClient:
    """Stands in for a decision-API SDK client."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def system_one(
        self, *, state: str, questions: Any, trace_id: str | None = None
    ) -> dict[str, Any]:
        self.calls.append({"state": state, "trace_id": trace_id})
        return json.loads(json.dumps(JEV_RESPONSE))


class FakeAsyncClient:
    async def system_one(self, *, state: str, trace_id: str | None = None) -> dict[str, Any]:
        return json.loads(json.dumps(JEV_RESPONSE))


def test_wrapping_a_client_records_every_answer_and_leaves_the_call_alone(tmp_path: Path) -> None:
    client = FakeClient()
    ticks = iter([0.0, 0.21])
    wrapped = collect.track(
        client,
        source_key=lambda **kwargs: kwargs.get("trace_id"),
        path=tmp_path / "records.jsonl",
        clock=lambda: __import__("datetime").datetime.fromtimestamp(
            next(ticks), tz=__import__("datetime").timezone.utc
        ),
    )

    assert wrapped is client  # patched in place and returned
    response = wrapped.system_one(state="payouts failing", questions={}, trace_id="t-9")

    assert response["model"] == "jev-1.13.0"  # the caller sees exactly what the client returned
    stored = read_records(tmp_path / "records.jsonl")
    assert {record.question_key for record in stored} == {"department", "is_urgent", "satisfaction"}
    assert all(record.source_key == "t-9" for record in stored)
    assert all(record.latency_ms == pytest.approx(210.0, abs=0.1) for record in stored)


def test_an_async_client_is_recorded_too(tmp_path: Path) -> None:
    client = collect.track(FakeAsyncClient(), path=tmp_path / "records.jsonl")

    response = asyncio.run(client.system_one(state="payouts failing"))

    assert response["model"] == "jev-1.13.0"
    assert len(read_records(tmp_path / "records.jsonl")) == 3


def test_a_wrapped_client_still_returns_what_it_returned_before(tmp_path: Path) -> None:
    client = FakeClient()
    original = client.system_one
    collect.track(client, path=tmp_path / "records.jsonl")

    # Bound methods are built fresh on each access, so this compares by function and instance.
    assert client.system_one.__wrapped__ == original  # type: ignore[attr-defined]


def test_wrapping_a_disabled_collector_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(collect.ENV_FLAG, "off")
    client = FakeClient()

    assert collect.track(client) is client
    assert "system_one" not in vars(client)  # nothing was installed on the instance


def test_an_unwrappable_response_is_counted_and_the_call_still_succeeds(tmp_path: Path) -> None:
    class Odd:
        def system_one(self, **kwargs: Any) -> str:
            return "not a response object"

    client = collect.track(Odd(), path=tmp_path / "records.jsonl")

    assert client.system_one() == "not a response object"
    assert collect.stats()["unsupported"] == 1
    assert collect.stats()["calls"] == 1


def test_a_client_without_a_known_method_is_returned_untouched(tmp_path: Path) -> None:
    class Nothing:
        pass

    client = Nothing()

    assert collect.track(client, path=tmp_path / "records.jsonl") is client
    assert not (tmp_path / "records.jsonl").exists()


def test_the_stats_expose_what_was_dropped(tmp_path: Path) -> None:
    collect.record(
        question_key="department", prediction="billing", confidence=0.9, path=tmp_path / "r.jsonl"
    )
    collect.record(
        question_key="department", prediction="billing", confidence=0.9, path=tmp_path / "r.jsonl"
    )

    assert collect.stats()["written"] == 2
    assert collect.stats()["dropped"] == 0


def test_tracking_twice_records_one_record_per_call(tmp_path: Path) -> None:
    """Wrapping twice would double every record and silently double-weight the whole log."""
    target = tmp_path / "records.jsonl"
    once = collect.track(FakeClient(), path=target)
    twice = collect.track(once, path=target)

    assert twice is once
    twice.system_one(state="payouts failing", questions={})

    stored = read_records(target)
    assert len(stored) == 3  # one per answer, not six
    assert collect.stats()["already_tracked"] == 1


def test_tracking_twice_under_another_method_name_is_counted(tmp_path: Path) -> None:
    target = tmp_path / "records.jsonl"
    client = collect.track(FakeClient(), path=target)

    collect.track(client, path=target, method_names=("system_one", "systemOne", "systemone"))

    assert collect.stats()["already_tracked"] == 1
    assert not target.exists()  # and nothing was recorded by a second wrapper
