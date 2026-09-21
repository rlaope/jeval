"""Write the two-version log the README's drift block is captured from.

The README quotes a `jeval drift` run, so the log behind it has to be reproducible by a reader
rather than living in the author's shell history:

    uv run python examples/make-drift-log.py /tmp/jeval-drift
    uv run jeval drift --root /tmp/jeval-drift --fail-on ece-increase=0.05

1,800 synthetic decisions for the same question, split across two model versions. The older one is
calibrated; the newer one is overconfident, which is the change a drift gate exists to catch. The
same generator the test suite restores its miscalibration with, so a reader can check both the
failure and the tool's ability to detect it.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from jeval.schema import DecisionRecord
from jeval.store import write_records
from jeval.synth import SynthSpec, generate

OLDER_START = datetime(2026, 8, 20, tzinfo=timezone.utc)
NEWER_START = datetime(2026, 9, 17, tzinfo=timezone.utc)


def build() -> list[DecisionRecord]:
    older = generate(
        SynthSpec(
            n=1200,
            mode="calibrated",
            question_key="department",
            classes=("billing", "technical", "other"),
            model="jev-1.13.0",
            start=OLDER_START,
            seed=41,
        )
    )
    newer = generate(
        SynthSpec(
            n=600,
            mode="overconfident",
            # Strong enough that the gate fires with room to spare: ECE 0.028 -> 0.141, +0.113.
            exponent=0.3,
            question_key="department",
            classes=("billing", "technical", "other"),
            model="jev-1.14.0",
            start=NEWER_START,
            seed=42,
        )
    )
    for index, record in enumerate(older):
        # One row per minute going back from the start, so the two versions do not interleave.
        record.ts = OLDER_START - timedelta(minutes=index)
    return [*older, *newer]


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path("/tmp/jeval-drift")
    records_dir = root / ".jeval"
    records_dir.mkdir(parents=True, exist_ok=True)
    records = build()
    write_records(records, records_dir / "records.jsonl")
    (root / "costs.yaml").write_text(
        "# costs for the drift example: a wrong auto-route is worth 25 hand-offs\n"
        "actions:\n"
        "  - name: auto_route\n"
        "    question: department\n"
        "    when: billing\n"
        "    cost_false_accept: 50000\n"
        "    cost_escalate: 2000\n"
        "    cost_false_reject: 0\n",
        encoding="utf-8",
    )
    print(f"wrote {len(records)} decision records to {records_dir / 'records.jsonl'}")
    print(f"wrote {root / 'costs.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
