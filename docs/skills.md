# Agent skills

Six skills that teach a coding agent to measure a classifier's confidence and set the human hand-off
line with jeval. They are written for the agent, not for a person: each one carries the commands, the
check that proves it worked, the failure modes that really happen, and the claims it must not make.

| Skill | The question it answers | The artifact it produces |
| --- | --- | --- |
| `jeval-handoff` | Nothing yet — hand the whole job to an agent | the report, and whatever else the job needed |
| `jeval-instrument-service` | Where do decisions come from? | `.jeval/records.jsonl` filling as the service runs |
| `jeval-labels-harvest` | Where do the answers come from? | labeled records (gold), kept apart from silver |
| `jeval-calibration-audit` | Is this confidence worth anything? | `report.html` |
| `jeval-threshold-from-costs` | Where should the line be? | `thresholds.yaml` |
| `jeval-drift-gate` | Did the model change when I was not looking? | a build that fails, with the movement named |

The canonical copies live in this repository under `skills/<name>/SKILL.md`. Everything else on this
page is generated from them. What the pack does when an agent runs it, in order:

```sh
jeval report --root /var/lib/jeval --current 0.88 --costs costs.yaml     # measure, and compare with what is deployed
jeval threshold --root /var/lib/jeval --costs costs.yaml                 # write thresholds.yaml for the application
jeval drift --root /var/lib/jeval --fail-on ece-increase=0.05            # fail a build when calibration moves
```

## Install

```sh
# what would be installed, and where each host reads it
curl -fsSL https://raw.githubusercontent.com/rlaope/jeval/main/install-skills.sh | sh -s -- --list

# the two roots that between them cover every host below
curl -fsSL https://raw.githubusercontent.com/rlaope/jeval/main/install-skills.sh | sh -s -- --host all

# one host, into one project
sh install-skills.sh --host cursor --project .
```

From a checkout the script finds the pack itself; piped from the network it unpacks the latest
release archive first. It uses no package manager, needs no root, has `--dry-run`, and
`--uninstall` removes what it installed.

Parallel installations that overlap are the only subtlety: `--host all` writes `.agents/skills` and
`.claude/skills`, and a host that reads both roots lists a skill from the first one it finds. Naming
a single host avoids that.

Every installed file carries the marker `jeval-agent-skills` — a YAML comment inside the frontmatter
of a skill, and an HTML comment around the block appended to an `AGENTS.md`. An uninstall removes
exactly the files carrying it, needs no pack and no network, leaves any file that is not ours alone,
and refuses to report success when it installed nothing.

## Where each host reads a skill

Verified against each host's own documentation on 2026-09-21. Paths are shown as project-relative;
the global form is the same path under `$HOME`. The generated tree in `skills/hosts/` mirrors these
paths exactly, so installing is a copy of a subtree.

| Host | Reads | Source |
| --- | --- | --- |
| Codex CLI | `.agents/skills/`, scanned from the working directory up to the repository root; global `~/.agents/skills`; required keys `name`, `description` | [developers.openai.com/codex/skills](https://developers.openai.com/codex/skills) |
| Hermes Agent | `.agents/skills/` and `.hermes/skills/` (project tier, trust-gated by `hermes skills trust`); global `~/.hermes/skills`; optional `metadata.hermes` keys | [hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills) |
| OpenClaw | `<workspace>/skills/` (highest precedence), `<workspace>/.agents/skills/`, personal `~/.agents/skills`; global `~/.openclaw/skills`; **description one line under 160 characters**, frontmatter keys must be single lines | [docs.openclaw.ai/tools/creating-skills](https://docs.openclaw.ai/tools/creating-skills) |
| Pi | `.pi/skills/`, `.agents/skills/`; global `~/.pi/agent/skills`; `name` up to 64 characters, `description` up to 1024 | [pi.dev/docs/latest/skills](https://pi.dev/docs/latest/skills) |
| Cursor | `.cursor/skills/`, `.agents/skills/` (plus compatibility reads of `.claude/skills/` and `.codex/skills/`); global `~/.cursor/skills`; `name` must match the parent directory | [cursor.com/docs/skills](https://cursor.com/docs/skills) |
| OpenCode | `.opencode/skills/`; global `~/.config/opencode/skills`; also reads `.claude/skills/` and `.agents/skills/`; keys `name`, `description` | [opencode.ai/docs/skills](https://opencode.ai/docs/skills) |
| Claude Code | `.claude/skills/`; global `~/.claude/skills`; **does not read `AGENTS.md`** — it reads `CLAUDE.md` | [code.claude.com/docs/en/skills](https://code.claude.com/docs/en/skills) |
| Anything that reads one instruction file | `AGENTS.md` — Codex CLI, OpenCode, Cursor, OpenClaw, Pi and Hermes all read it | per host above |

The shared layout is the [Agent Skills specification](https://agentskills.io/specification):
`name` (≤64 characters, lowercase letters, digits and hyphens, matching the parent directory) and
`description` are the required keys, plus optional `license`, `compatibility`, `metadata` and
`allowed-tools`. `.agents/skills` is the root the most hosts read; Claude Code is the exception, and
gets `.claude/skills`.

### What we deliberately did not ship

* **Cursor `.mdc` rules.** Project rules are `.cursor/rules/*.mdc`, but a rule is not a skill: since
  Cursor 2.4 its own `/migrate-to-skills` converts dynamic rules to skills. Shipping rules would
  give the same instructions through a second, weaker mechanism.
* **`~/.codex/skills`.** The 2025-12-15 vendor document located experimental skills there; the
  current documentation says `.agents/skills`. We follow the current one.
* **A per-host description rewrite.** OpenClaw gets the first sentence of the canonical description
  rather than a shortened variant written by hand, so there is still exactly one description per
  skill. A first sentence over 160 characters fails the generator instead of being trimmed.
* **OpenClaw and Opencode as separate skill roots in `--host all`.** Both read `.agents/skills`, so a
  second copy under their own root would list the same skill twice.

### What is not certain

The table above records what each host's own documentation states, retrieved 2026-09-21. Where that
documentation is silent, contradictory, or undated, the question stays open here so a future change
can start with it rather than with a guess:

* **Codex CLI** — whether it still scans the legacy `$CODEX_HOME/skills` path is not stated in the
  current documentation; that path appears only in a vendor document dated 2025-12-15. The pack ships
  `.agents/skills` only.
* **Cursor** — the discovery path for `AGENTS.md` is documented as a rule *type*, not as a path, so the
  digest is offered for projects that already read `AGENTS.md`.
* **OpenClaw** — the current name and state directory are confirmed by the vendor documentation, but
  the rename dates come from third-party sources. Its description budget (one line, under 160
  characters) is written as guidance rather than a hard limit, and the pack keeps to it anyway.
* **Hermes Agent** — per-profile skill paths, and the date the project tier was added, are not
  published.
* **Pi** — only two of its documentation pages are directly fetchable; the instruction-file facts come
  from the vendor repository's own docs file.
* **OpenCode** — the V2 skill-source layout carries no published release date; the pack targets the
  fixed roots documented for the current version.
* **Claude Code** — no maximum length is documented for `name`, and the only size guidance for
  `SKILL.md` is to keep it under 500 lines (the pack caps a skill at 120 lines).

## Keeping the pack true

One source, one generator, one guard:

```sh
uv run python tools/export_skills.py           # regenerate skills/hosts/
uv run python tools/export_skills.py --check    # fail when the committed tree is stale
```

* `tests/test_skills_pack.py` fails when the committed tree stops matching its sources, when a skill
  breaks a documented frontmatter budget, when the aggregate `AGENTS.md` drops a skill or its
  honest-limits section, or when an install stops being reversible.
* `tests/test_documented_features.py` parses every `jeval ...` line inside a skill and checks it
  against the real command surface, exactly as it does for the README. A skill that tells an agent to
  run a flag that does not exist fails the suite.
* The release workflow runs `--check` at tag time and attaches `jeval-skills.tar.gz`, whose stable
  asset name is what lets `releases/latest/download/jeval-skills.tar.gz` resolve.

The pack is versioned with the project: the tarball records the version in `manifest.json`, and the
digest for `AGENTS.md` names it in its header.
