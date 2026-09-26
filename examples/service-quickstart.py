"""SYNTHETIC. A stand-in service for docs/library.md and docs/instrumenting-a-service.md.

A fake ticket classifier, the two jeval lines, and a fake human who closes most tickets later. Run it
with the directory the records should land in:

    JEVAL_ROOT=/tmp/jeval-service uv run python examples/service-quickstart.py

Nothing here calls a real model. The classifier returns made-up answers with made-up probabilities;
it claims a little more than it earns, and a little more again for Korean tickets, so the report has
something real to find. The same seed gives the same files every time.
"""

import random

from jeval import collect

DEPARTMENTS = ("billing", "technical", "other")


class TicketClassifier:
    """Stands in for a Jev client: returns typed answers with probabilities."""

    def classify(self, *, ticket_id: str, text: str, lang: str) -> dict:
        rng = random.Random(ticket_id)
        top = rng.choice(DEPARTMENTS)
        p = rng.uniform(0.45, 0.99)
        rest = [d for d in DEPARTMENTS if d != top]
        split = rng.uniform(0.3, 0.7)
        return {
            "model": "jev-1.14.0",
            "answers": {
                "department": {
                    "type": "choice",
                    "choice": top,
                    "probabilities": {
                        top: p,
                        rest[0]: (1 - p) * split,
                        rest[1]: (1 - p) * (1 - split),
                    },
                },
            },
        }


client = collect.track(
    TicketClassifier(),
    method_names=("classify",),
    source_key=lambda **kw: kw["ticket_id"],
    segment=lambda **kw: {"lang": kw["lang"]},
)

rng = random.Random(7)
for n in range(600):
    ticket_id = f"T-{n}"
    lang = rng.choice(["ko", "en"])
    answer = client.classify(ticket_id=ticket_id, text="...", lang=lang)
    department = answer["answers"]["department"]
    predicted = department["choice"]
    claimed = department["probabilities"][predicted]
    # How often it is really right: a little below what it claims, more so in Korean.
    right = rng.random() < claimed - (0.12 if lang == "ko" else 0.04)
    final = predicted if right else rng.choice([d for d in DEPARTMENTS if d != predicted])
    # Later, a human closes the ticket. Four in five have closed so far.
    if n % 5:
        collect.resolve(source_key=ticket_id, question="department", answer=final)

print(collect.stats())
