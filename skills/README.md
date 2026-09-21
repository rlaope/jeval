# The jeval skills

Six skills that teach a coding agent to measure a classifier's confidence and set the human hand-off
line. They are written for the agent, not for a person: each carries the commands, the check that
proves it worked, the failure modes that really happen, and what it must not claim.

| Skill | The question it answers | The artifact it produces |
| --- | --- | --- |
| [`jeval-handoff`](jeval-handoff/SKILL.md) | Nothing yet — hand the whole job to an agent | the report, and whatever the job needed first |
| [`jeval-instrument-service`](jeval-instrument-service/SKILL.md) | Where do decisions come from? | `.jeval/records.jsonl` filling as the service runs |
| [`jeval-labels-harvest`](jeval-labels-harvest/SKILL.md) | Where do the answers come from? | labeled records, silver kept apart from gold |
| [`jeval-calibration-audit`](jeval-calibration-audit/SKILL.md) | Is this confidence worth anything? | `report.html` |
| [`jeval-threshold-from-costs`](jeval-threshold-from-costs/SKILL.md) | Where should the line be? | `thresholds.yaml` |
| [`jeval-drift-gate`](jeval-drift-gate/SKILL.md) | Did the model change when I was not looking? | a build that fails, with the movement named |

## Install them

```sh
sh install-skills.sh --list                       # hosts, destinations, and the skills in the pack
sh install-skills.sh --host all                    # .agents/skills + .claude/skills
sh install-skills.sh --host cursor --project .
```

`--host all` writes the two roots that between them cover every host jeval targets; naming a single
host installs only that host's own root. `--uninstall` removes what the script installed and nothing
else.

## One source, generated copies

**This directory is the source of truth.** `skills/<name>/SKILL.md` is the only hand-edited copy of
each skill; the per-host layouts under `skills/hosts/` are generated from it by
`tools/export_skills.py` and are checked byte-for-byte by `tests/test_skills_pack.py`.

Do not hand-edit anything under `skills/hosts/`: the change will be lost on the next regeneration,
and the guard will fail the suite before it can reach a release. Edit the canonical file, then run
the generator:

```sh
uv run python tools/export_skills.py
uv run python tools/export_skills.py --check
```

[`docs/skills.md`](../docs/skills.md) records where each host reads a skill, with the source for each
destination, and why some conventions were deliberately not shipped.
