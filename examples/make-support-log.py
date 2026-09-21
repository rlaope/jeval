"""Write the support-routing log the walkthrough on a real system runs on.

A two-version log for one question, in the shape an application actually writes: the model's own
field names, no labels, and the human's answer arriving separately from the ticketing tool. The
miscalibration is injected and known, so what the tool measures can be checked against what was put
in.

    uv run python examples/make-support-log.py /tmp/support
    cd /tmp/support && jeval init && jeval ingest logs/aug-app-log.jsonl

The newer version is degraded (an overconfident exponent of 0.35), which is the change
`jeval drift --fail-on ece-increase=0.05` catches.
"""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from jeval.synth import SynthSpec, generate

CLASSES = ("billing", "technical", "other")


def write_pair(records, logs: Path, prefix: str, *, coverage: float, seed: int) -> tuple[int, int]:
    """Write the app log and the resolution log, with the human's answer only in the second."""
    rng = random.Random(seed)
    log_path = logs / f"{prefix}-app-log.jsonl"
    res_path = logs / f"{prefix}-resolutions.jsonl"
    rows = resolved = 0
    with log_path.open("w", encoding="utf-8") as log, res_path.open("w", encoding="utf-8") as res:
        for index, record in enumerate(records):
            request_id = f"tr_{prefix}{index:05d}"
            log.write(
                json.dumps(
                    {
                        "request_id": request_id,
                        "created_at": record.ts.isoformat().replace("+00:00", "Z"),
                        "model_name": record.model,
                        "task": record.question_key,
                        "label": record.prediction,  # the model's own column name for its answer
                        "score_0_to_1": round(record.confidence, 4),
                        "lang": (record.segment or {}).get("lang", "ko"),
                        "context_tokens": record.state_tokens,
                    }
                )
                + "\n"
            )
            rows += 1
            # Not every ticket is reviewed by a person: the escalated ones are, and some auto-routed
            # ones come back corrected by the customer.
            if record.label and rng.random() < coverage:
                res.write(
                    json.dumps(
                        {
                            "request_id": request_id,
                            "final_department": record.label,
                            "handled_by": rng.choice(["human_review", "auto_corrected"]),
                            "closed_at": record.ts.isoformat().replace("+00:00", "Z"),
                        }
                    )
                    + "\n"
                )
                resolved += 1
    print(f"{log_path.name}: {rows} responses, {res_path.name}: {resolved} human answers")
    return rows, resolved


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path("/tmp/support")
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    # The model in production today: inflated, the failure this tool exists to measure.
    write_pair(
        generate(
            SynthSpec(
                n=900,
                mode="inflated",
                inflation=1.12,
                question_key="department",
                classes=CLASSES,
                languages=("ko", "en"),
                model="jev-1.13.0",
                start=datetime(2026, 8, 1, tzinfo=timezone.utc),
                seed=41,
            )
        ),
        logs,
        "aug",
        coverage=0.62,
        seed=11,
    )
    # The model after a quiet upgrade: same interface, worse confidence.
    write_pair(
        generate(
            SynthSpec(
                n=500,
                mode="overconfident",
                exponent=0.35,
                question_key="department",
                classes=CLASSES,
                languages=("ko", "en"),
                model="jev-1.14.0",
                start=datetime(2026, 9, 12, tzinfo=timezone.utc),
                seed=42,
            )
        ),
        logs,
        "sep",
        coverage=0.62,
        seed=12,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
