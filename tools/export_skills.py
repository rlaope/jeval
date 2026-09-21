#!/usr/bin/env python
"""Export the canonical jeval skills into the layout each agent host expects.

The canonical sources live in `skills/<name>/SKILL.md` and are the only hand-edited copies. This
script renders them into `skills/hosts/<host>/...`, one directory per host, and the result is
committed so that a reader on GitHub, a release archive, and a raw URL all get the same bytes.

Two rules make the output trustworthy:

* Deterministic. No timestamps, no version drift beyond the project version, sorted iteration:
  running it twice produces identical bytes, and `--check` fails when the committed tree differs
  from what the sources say.
* Only verified layouts. Every host below is listed with the path and frontmatter rules that host's
  own documentation states; the sources are recorded in docs/skills.md beside each host's row. A
  layout that could not be confirmed from primary documentation is not here.

Usage:
    uv run python tools/export_skills.py            # write skills/hosts/
    uv run python tools/export_skills.py --check     # fail if the committed tree is stale
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CANONICAL = REPO / "skills"
OUTPUT = CANONICAL / "hosts"

# The destination each host reads, relative to the project directory (or to $HOME for a global
# install). The generated tree carries this path verbatim, so installing is a copy of a subtree and
# the file in this repository sits exactly where the host will look for it.
TARGETS: dict[str, str] = {
    # Read by six hosts: Codex CLI, Hermes Agent, OpenClaw, Pi, Cursor and OpenCode.
    "agents-skills": ".agents/skills",
    "claude-code": ".claude/skills",
    "cursor": ".cursor/skills",
    "hermes": ".hermes/skills",
    "opencode": ".opencode/skills",
    "pi": ".pi/skills",
    "openclaw": ".openclaw/skills",
    # Not a skill root: the single instruction document other hosts read, one digest for all skills.
    "agents-md": "",
}

# Which renderer each host uses. `portable` is the shared Agent Skills layout (name + description,
# body preserved), which is what most hosts read verbatim.
RENDERER_FOR: dict[str, str] = {
    "agents-skills": "portable",
    "claude-code": "portable",
    "cursor": "portable",
    "opencode": "portable",
    "pi": "portable",
    "hermes": "hermes",
    "openclaw": "openclaw",
    "agents-md": "digest",
}

# Documented limits. Enforced here rather than worked around: a description that OpenClaw would
# truncate is a description the project should have written differently.
DESCRIPTION_MAX = 1024  # Agent Skills spec
FIRST_SENTENCE_MAX = 160  # OpenClaw: "keep it one line and under 160 characters"
NAME_MAX = 64  # Agent Skills spec and OpenCode
BODY_LINE_MAX = 500  # Agent Skills spec: keep SKILL.md under 500 lines


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    source: Path

    @property
    def first_sentence(self) -> str:
        """The description up to its first sentence end, which is all some hosts will read."""
        match = re.match(r"(.+?[.!?])(\s|$)", self.description)
        return match.group(1) if match else self.description


def project_version() -> str:
    """The version this pack belongs to, read from pyproject.toml so both agree."""
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if match is None:
        raise SystemExit('pyproject.toml has no version = "..." line')
    return match.group(1)


def parse_skill(path: Path) -> Skill:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise SystemExit(f"{path}: missing YAML frontmatter")
    _, front, body = text.split("---\n", 2)
    name = re.search(r'^name:\s*"?(?P<v>[^"\n]+)"?\s*$', front, re.MULTILINE)
    description = re.search(r'^description:\s*"?(?P<v>[^\n]+?)"?\s*$', front, re.MULTILINE)
    if name is None or description is None:
        raise SystemExit(f"{path}: frontmatter needs a name and a description")
    return Skill(
        name=name.group("v").strip(),
        description=description.group("v").strip(),
        body=body.lstrip("\n"),
        source=path,
    )


def validate(skill: Skill) -> None:
    """Refuse to generate a pack a host would silently distort or drop."""
    where = skill.source
    if skill.source.parent.name != skill.name:
        raise SystemExit(f"{where}: name '{skill.name}' must match its directory")
    if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", skill.name):
        raise SystemExit(f"{where}: name must be lowercase letters, digits and hyphens")
    if len(skill.name) > NAME_MAX:
        raise SystemExit(f"{where}: name is {len(skill.name)} characters, the limit is {NAME_MAX}")
    if len(skill.description) > DESCRIPTION_MAX:
        raise SystemExit(f"{where}: description is longer than {DESCRIPTION_MAX} characters")
    if len(skill.first_sentence) > FIRST_SENTENCE_MAX:
        raise SystemExit(
            f"{where}: the first sentence of the description is {len(skill.first_sentence)} "
            f"characters. OpenClaw reads one line under {FIRST_SENTENCE_MAX}, so the first "
            "sentence has to stand on its own inside that budget."
        )
    if "\n" in skill.description.strip():
        raise SystemExit(f"{where}: the description must be a single line")
    if len(skill.body.splitlines()) > BODY_LINE_MAX:
        raise SystemExit(f"{where}: the body is longer than {BODY_LINE_MAX} lines")


def load_skills() -> list[Skill]:
    if not CANONICAL.is_dir():
        raise SystemExit(f"{CANONICAL} does not exist")
    paths = sorted(p for p in CANONICAL.glob("*/SKILL.md"))
    if not paths:
        raise SystemExit(f"no skills found under {CANONICAL}/*/SKILL.md")
    skills = [parse_skill(p) for p in paths]
    for skill in skills:
        validate(skill)
    names = [s.name for s in skills]
    if len(set(names)) != len(names):
        raise SystemExit(f"two skills share a name: {sorted(names)}")
    return skills


def strip_heading(body: str, title: str) -> str:
    """The text under one `## title` section, without the heading itself."""
    pattern = re.compile(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", re.MULTILINE | re.DOTALL)
    match = pattern.search(body)
    return match.group(1).strip("\n") if match else ""


# --- renderers -----------------------------------------------------------------------------------


def render_portable(skills: list[Skill]) -> dict[str, str]:
    """The shared Agent Skills layout: frontmatter name + description, body unchanged."""
    files: dict[str, str] = {}
    for skill in skills:
        head = f"---\nname: {skill.name}\ndescription: {skill.description}\n---\n\n"
        files[f"{skill.name}/SKILL.md"] = head + skill.body
    return files


def render_hermes(skills: list[Skill]) -> dict[str, str]:
    """Hermes reads the same file and accepts an optional metadata.hermes block."""
    files: dict[str, str] = {}
    for skill in skills:
        head = (
            f"---\nname: {skill.name}\ndescription: {skill.description}\n"
            "metadata:\n  hermes:\n    tags: [jeval, calibration, thresholds]\n"
            "    category: evaluation\n---\n\n"
        )
        files[f"{skill.name}/SKILL.md"] = head + skill.body
    return files


def render_openclaw(skills: list[Skill]) -> dict[str, str]:
    """OpenClaw shows one line and asks for under 160 characters, so it gets the first sentence."""
    files: dict[str, str] = {}
    for skill in skills:
        head = f"---\nname: {skill.name}\ndescription: {skill.first_sentence}\n---\n\n"
        files[f"{skill.name}/SKILL.md"] = head + skill.body
    return files


def render_agents_digest(skills: list[Skill], version: str) -> dict[str, str]:
    """One file for hosts that read a single instruction document.

    The digest is generated from the same sections the skills carry, so it cannot drift into a
    second, softer version of the truth: the steps and their verification lines are copied, only the
    prose around them is trimmed. Claude Code does not read AGENTS.md, which is why it is given the
    skill tree instead — see docs/skills.md.
    """
    lines = [
        "# jeval",
        "",
        "How to measure a classifier's confidence and set the human/AI hand-off line with jeval",
        f"({version}). Generated from the canonical skills in the jeval repository; each section",
        "names the artifact it produces. Run `jeval <command> --help` for the flags of any step,",
        "and `jeval report` once a step has produced records.",
        "",
        "Read every number as an estimate with an interval. jeval measures what records support",
        "and refuses to answer when they do not: no labels means no measurement, and it says so.",
        "",
    ]
    for skill in skills:
        lines.append(f"## {skill.name}")
        lines.append("")
        lines.append(skill.description)
        lines.append("")
        steps = strip_heading(skill.body, "What to do")
        if steps:
            lines.append(steps)
            lines.append("")
        verify = strip_heading(skill.body, "Verify")
        if verify:
            lines.append("**Verify**")
            lines.append("")
            lines.append(verify)
            lines.append("")
        limits = strip_heading(skill.body, "Do not claim")
        if limits:
            lines.append("**Do not claim**")
            lines.append("")
            lines.append(limits)
            lines.append("")
    return {"AGENTS.md": "\n".join(lines).rstrip("\n") + "\n"}


def build(version: str | None = None) -> dict[str, str]:
    """Every generated file, keyed by its path relative to skills/hosts/."""
    version = version or project_version()
    skills = load_skills()
    tree: dict[str, str] = {}
    written: dict[str, list[str]] = {}
    for host, target in TARGETS.items():
        renderer = RENDERER_FOR[host]
        if renderer == "portable":
            files = render_portable(skills)
        elif renderer == "hermes":
            files = render_hermes(skills)
        elif renderer == "openclaw":
            files = render_openclaw(skills)
        else:
            files = render_agents_digest(skills, version)
        prefix = f"{target}/" if target else ""
        for relative, content in sorted(files.items()):
            tree[f"{host}/{prefix}{relative}"] = content
            written.setdefault(host, []).append(f"{prefix}{relative}")
        # The destination this host reads, as one line a shell installer can read: the pack tree
        # mirrors the install location, and this is the mapping stated once.
        tree[f"{host}/target.txt"] = f"{target}\n"

    # Two indexes on purpose. `manifest.json` is for tools; `skills.txt` is one skill name per line
    # so a shell installer can read it without parsing JSON. Parsing the JSON array from a shell is
    # how an installer silently installs nothing: with `indent=2` the names sit on their own lines,
    # so a line-oriented match yields an empty list and an exit code of 0.
    #
    # The file list is taken from what was actually written, never recomputed: a manifest built from
    # a second call into the renderers could describe a tree that differs from the one on disk.
    tree["manifest.json"] = (
        json.dumps(
            {
                "version": version,
                "canonical_dir": "skills",
                "output_dir": "skills/hosts",
                "targets": TARGETS,
                "hosts": {host: sorted(files) for host, files in written.items()},
                "skills": [s.name for s in skills],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    tree["skills.txt"] = (
        "# One skill name per line, from tools/export_skills.py. Installers read this file\n"
        "# instead of parsing manifest.json, where the same names span several lines.\n"
        + "".join(f"{s.name}\n" for s in skills)
    )
    return tree


def display(path: Path) -> str:
    """A path for a human, relative to the repository when it is inside it."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def write(tree: dict[str, str]) -> list[str]:
    written: list[str] = []
    for relative, content in tree.items():
        target = OUTPUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_text(encoding="utf-8") != content:
            target.write_text(content, encoding="utf-8")
            written.append(display(target))
    return written


def check(tree: dict[str, str]) -> list[str]:
    """Differences between what the sources say and what is committed."""
    drift: list[str] = []
    for relative, content in tree.items():
        target = OUTPUT / relative
        if not target.exists():
            drift.append(f"missing: skills/hosts/{relative}")
        elif target.read_text(encoding="utf-8") != content:
            drift.append(f"stale: skills/hosts/{relative}")
    if OUTPUT.is_dir():
        expected = {str(OUTPUT / r) for r in tree}
        for path in sorted(OUTPUT.rglob("*")):
            if path.is_file() and str(path) not in expected:
                drift.append(f"unexpected: {path.relative_to(REPO)}")
    return drift


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export the canonical jeval skills into per-host layouts."
    )
    parser.add_argument("--check", action="store_true", help="fail if skills/hosts/ is out of date")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    tree = build()
    if args.check:
        drift = check(tree)
        if drift:
            print("skills/hosts/ is out of date with skills/:", file=sys.stderr)
            for line in drift:
                print(f"  {line}", file=sys.stderr)
            print("run: uv run python tools/export_skills.py", file=sys.stderr)
            return 1
        if not args.quiet:
            print(f"skills/hosts/ matches {len(tree)} generated files")
        return 0

    written = write(tree)
    if not args.quiet:
        print(f"{len(tree)} files across {len(TARGETS)} hosts ({len(written)} changed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
