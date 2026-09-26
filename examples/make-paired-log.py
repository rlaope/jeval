"""Write the shadow-traffic log the README's `jeval drift --paired` block is captured from.

    uv run python examples/make-paired-log.py /tmp/jeval-paired
    uv run jeval drift --root /tmp/jeval-paired --paired

SYNTHETIC data. 900 requests, each answered by two model versions that share the request's
`source_key`, its true answer and how hard it is. The older version is calibrated; the newer one is
calibrated too but genuinely less accurate (`current_skill=0.8`), which is the change an ECE gate
cannot see: a worse model that knows it is worse. Each side has its own 90% of gold labels, so some
pairs lack a label on one side and are counted rather than used.
"""

from __future__ import annotations

import sys
from pathlib import Path

from jeval.schema import DecisionRecord
from jeval.store import write_records
from jeval.synth import SynthSpec, generate_paired


def build() -> list[DecisionRecord]:
    return generate_paired(
        SynthSpec(
            n=900,
            mode="calibrated",
            question_key="department",
            label_fraction=0.9,
            model="jev-1.13.0",
            seed=51,
        ),
        SynthSpec(
            n=900,
            mode="calibrated",
            question_key="department",
            label_fraction=0.9,
            model="jev-1.14.0",
            seed=52,
        ),
        current_skill=0.8,
        requests_seed=50,
    )


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path("/tmp/jeval-paired")
    records_dir = root / ".jeval"
    records_dir.mkdir(parents=True, exist_ok=True)
    records = build()
    write_records(records, records_dir / "records.jsonl")
    print(f"wrote {len(records)} synthetic decision records to {records_dir / 'records.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
