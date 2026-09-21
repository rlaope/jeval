"""Guard the agent skill pack: sources, generated layouts, and the installer.

Why this file exists: a skill pack is a second copy of the instructions an agent follows, and a
second copy is where the truth forks. The pack therefore has one canonical source
(`skills/<name>/SKILL.md`), a generator (`tools/export_skills.py`) that renders it into the layout
each host expects, and this guard, which checks the laws that keep them from drifting:

1. **One source.** Every canonical skill has well-formed frontmatter and the fixed section order,
   and the committed per-host tree is byte-identical to what the sources render — hand-editing a
   generated file fails here rather than silently diverging.
2. **Only layouts the hosts document.** Each host's destination and frontmatter budget comes from
   that host's own documentation (recorded in `docs/skills.md`): a name that matches its directory,
   a description inside the tightest budget any host enforces, and single-line keys, because
   OpenClaw's parser reads one line per key.
3. **Commands are real.** Every `jeval ...` line in a skill is parsed by
   `tests/test_documented_features.py` against the real CLI, so a skill cannot instruct an agent to
   run something that does not exist.
4. **Installing is reversible and non-destructive.** The installer is valid POSIX sh, records its
   own marker the way each file format allows (a YAML comment inside frontmatter — a marker before
   the opening `---` would hide the frontmatter from the host), removes only its own files, and
   refuses to report success when it has installed nothing.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SKILLS = REPO / "skills"
HOSTS = SKILLS / "hosts"
EXPORTER = REPO / "tools" / "export_skills.py"
INSTALLER = REPO / "install-skills.sh"
MARKER = "jeval-agent-skills"

REQUIRED_SECTIONS = ("## Use when", "## What to do", "## Verify", "## Pitfalls", "## Do not claim")

# The tightest budget any host applies, from docs/skills.md: OpenClaw shows one line under 160
# characters, so a skill's first sentence has to stand alone inside it.
FIRST_SENTENCE_MAX = 160

sys.path.insert(0, str(REPO / "tools"))
import export_skills  # noqa: E402  (the generated tree is produced by this module)


def _skill_paths() -> list[Path]:
    return sorted(SKILLS.glob("*/SKILL.md"))


def _frontmatter(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{path.name}: frontmatter must be the first thing in the file"
    _, front, body = text.split("---\n", 2)
    values: dict[str, str] = {}
    for line in front.splitlines():
        match = re.match(r"^([a-z_]+):\s*(.*)$", line)
        if match:
            values[match.group(1)] = match.group(2).strip().strip('"')
    return values, body


def _first_sentence(description: str) -> str:
    match = re.match(r"(.+?[.!?])(\s|$)", description)
    return match.group(1) if match else description


def test_the_pack_has_the_skills_it_claims() -> None:
    names = sorted(path.parent.name for path in _skill_paths())
    assert len(names) >= 5, f"the canonical pack looks empty: {names}"
    for required in ("jeval-handoff", "jeval-calibration-audit"):
        assert required in names, f"{required} is the pack's entry point and is missing"


def test_every_skill_is_well_formed() -> None:
    for path in _skill_paths():
        front, body = _frontmatter(path)
        directory = path.parent.name

        assert set(front) == {"name", "description"}, (
            f"{directory}: frontmatter should carry exactly name and description, found {sorted(front)}"
        )
        assert front["name"] == directory, (
            f"{directory}: frontmatter name is {front['name']!r}; every host resolves the skill by "
            "its directory, so the two must agree"
        )
        assert len(front["name"]) <= 64, f"{directory}: names are capped at 64 characters"
        description = front["description"]
        assert description.startswith("Use when "), (
            f"{directory}: the description is the trigger the host matches on; it must start with "
            "'Use when '"
        )
        assert len(_first_sentence(description)) <= FIRST_SENTENCE_MAX, (
            f"{directory}: the first sentence of the description is "
            f"{len(_first_sentence(description))} characters; OpenClaw shows one line under "
            f"{FIRST_SENTENCE_MAX}, so it has to make sense on its own"
        )
        assert len(body.splitlines()) <= 120, f"{directory}: body is longer than 120 lines"


def test_every_skill_keeps_the_five_sections_in_order() -> None:
    for path in _skill_paths():
        _, body = _frontmatter(path)
        positions = [body.find(title) for title in REQUIRED_SECTIONS]
        assert all(position != -1 for position in positions), (
            f"{path.parent.name}: missing one of {REQUIRED_SECTIONS}"
        )
        assert positions == sorted(positions), (
            f"{path.parent.name}: the sections are out of order; a reader follows them in sequence"
        )


def test_every_skill_is_english_only() -> None:
    """The repository is English-only; a skill an agent reads is no exception."""
    for path in _skill_paths():
        text = path.read_text(encoding="utf-8")
        foreign = re.findall(r"[\u3000-\u9fff\uac00-\ud7af\u0400-\u04ff]", text)
        assert foreign == [], f"{path.parent.name}: non-English characters {foreign[:5]}"


def test_the_committed_host_tree_matches_the_sources() -> None:
    """The sync law. Regenerating must be a no-op, or the pack has two versions of the truth."""
    drift = export_skills.check(export_skills.build())
    assert drift == [], (
        "skills/hosts/ is not what the canonical skills render. Run "
        "`uv run python tools/export_skills.py` and commit the result. Drift:\n  "
        + "\n  ".join(drift)
    )


def test_generation_is_deterministic() -> None:
    """Two runs must produce identical bytes, so a diff always means a real change."""
    first = export_skills.build()
    second = export_skills.build()
    assert first == second
    for relative, content in first.items():
        assert not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:", content), (
            f"{relative}: a timestamp would make every regeneration a diff"
        )


def test_the_target_map_is_stated_once() -> None:
    """The installer reads target.txt, and the manifest carries the same map."""
    manifest = json.loads((HOSTS / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["targets"]) == set(manifest["hosts"]), "targets and hosts disagree"
    for host, target in manifest["targets"].items():
        stated = (HOSTS / host / "target.txt").read_text(encoding="utf-8").strip()
        assert stated == target, f"{host}/target.txt says {stated!r}, manifest says {target!r}"
        if target:
            for relative in manifest["hosts"][host]:
                assert relative.startswith(f"{target}/"), (
                    f"{host}: {relative} is not under the destination {target} this host reads"
                )


def test_every_host_gets_every_skill() -> None:
    manifest = json.loads((HOSTS / "manifest.json").read_text(encoding="utf-8"))
    canonical = {path.parent.name for path in _skill_paths()}

    assert set(manifest["skills"]) == canonical, "the manifest lists a different set of skills"
    assert manifest["hosts"], "no host targets were generated"

    for host, files in manifest["hosts"].items():
        assert files, f"{host}: generated no files"
        for relative in files:
            assert (HOSTS / host / relative).exists(), f"{host}: {relative} is listed but missing"

    for skill in canonical:
        assert (HOSTS / "agents-skills" / ".agents" / "skills" / skill / "SKILL.md").exists(), (
            f"the portable root is missing {skill}, which six hosts read"
        )
        assert (HOSTS / "claude-code" / ".claude" / "skills" / skill / "SKILL.md").exists(), (
            f"Claude Code does not read .agents/skills, so it needs its own copy of {skill}"
        )


def test_the_generated_copy_carries_the_canonical_body() -> None:
    """A generated file must be a rendering of the source, not a second, softer version of it."""
    for path in _skill_paths():
        _, body = _frontmatter(path)
        generated = (
            HOSTS / "agents-skills" / ".agents" / "skills" / path.parent.name / "SKILL.md"
        ).read_text(encoding="utf-8")
        assert body.strip() in generated, (
            f"{path.parent.name}: the generated copy has a different body from the source"
        )


def test_openclaw_gets_one_line_per_key() -> None:
    """OpenClaw's frontmatter parser reads single-line keys, so nothing may be nested or wrapped."""
    for path in _skill_paths():
        generated = (
            HOSTS / "openclaw" / ".openclaw" / "skills" / path.parent.name / "SKILL.md"
        ).read_text(encoding="utf-8")
        front = generated.split("---\n")[1]
        for line in front.splitlines():
            if not line.strip():
                continue
            assert re.match(r"^[a-z-]+: \S", line), (
                f"{path.parent.name}: OpenClaw cannot read this frontmatter line: {line!r}"
            )
        description = re.search(r"^description: (.+)$", front, re.MULTILINE)
        assert description is not None
        assert len(description.group(1)) <= FIRST_SENTENCE_MAX


def test_the_aggregate_digest_carries_every_skill() -> None:
    digest = (HOSTS / "agents-md" / "AGENTS.md").read_text(encoding="utf-8")
    for path in _skill_paths():
        assert f"## {path.parent.name}" in digest, f"AGENTS.md is missing {path.parent.name}"
    assert "**Do not claim**" in digest, "the digest dropped the honest-limits section"


def test_the_shell_index_agrees_with_the_manifest() -> None:
    """The installer reads skills.txt; a drift between it and the manifest would install a subset."""
    manifest = json.loads((HOSTS / "manifest.json").read_text(encoding="utf-8"))
    listed = [
        line
        for line in (HOSTS / "skills.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert listed == sorted(listed), "skills.txt must be sorted, so a diff stays readable"
    assert listed == sorted(manifest["skills"]), (
        "skills.txt and manifest.json disagree about which skills exist"
    )


def test_the_installer_is_valid_shell() -> None:
    result = subprocess.run(["sh", "-n", str(INSTALLER)], capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0, f"install-skills.sh does not parse: {result.stderr}"


def test_the_installer_marks_its_files_without_breaking_frontmatter() -> None:
    """A skill file is discovered by its frontmatter, which must stay the first thing in the file."""
    script = INSTALLER.read_text(encoding="utf-8")
    assert MARKER in script, "the installer does not record what it installed"
    assert 'echo "# $MARKER"' in script, (
        "the marker belongs in the frontmatter as a YAML comment; anything before the opening --- "
        "stops the host from reading the frontmatter at all"
    )
    assert not re.search(r"echo \"<!-- \$MARKER -->\"\s*\n\s*cat", script), (
        "the installer must not prepend an HTML comment to a skill file"
    )


def test_the_installer_lists_its_destinations() -> None:
    result = subprocess.run(
        ["sh", str(INSTALLER), "--list", "--source", str(HOSTS)],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert result.returncode == 0, result.stderr
    for host, destination in (
        ("agents-skills", ".agents/skills"),
        ("claude-code", ".claude/skills"),
        ("cursor", ".cursor/skills"),
        ("hermes", ".hermes/skills"),
        ("opencode", ".config/opencode/skills"),
        ("pi", ".pi/agent/skills"),
        ("openclaw", ".openclaw/skills"),
        ("agents-md", ".codex"),
    ):
        assert host in result.stdout, f"--list does not mention {host}"
        assert destination in result.stdout, f"--list does not say where {host} installs"


def test_an_empty_skill_index_is_refused(tmp_path: Path) -> None:
    """Installing nothing and exiting 0 is how a broken install looks exactly like a working one.

    This is not hypothetical: the first version of the installer parsed the skill names out of
    `manifest.json` with `sed`, and because `json.dumps(indent=2)` puts each name on its own line,
    the match produced an empty list — so it installed nothing and reported success.
    """
    empty = tmp_path / "pack" / "skills" / "hosts"
    empty.mkdir(parents=True)
    (empty / "manifest.json").write_text('{"skills": []}\n', encoding="utf-8")
    (empty / "skills.txt").write_text("# only comments\n", encoding="utf-8")

    result = subprocess.run(
        [
            "sh",
            str(INSTALLER),
            "--host",
            "claude-code",
            "--project",
            str(tmp_path),
            "--source",
            str(empty),
        ],
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
        cwd=str(tmp_path),
    )
    assert result.returncode != 0, "an empty pack must not exit 0"
    assert "lists no skills" in result.stderr


@pytest.mark.parametrize("host", ["claude-code", "cursor", "agents-skills", "agents-md"])
def test_install_then_uninstall_leaves_nothing_behind(host: str, tmp_path: Path) -> None:
    """The property a user cares about: installing is reversible, and a foreign file is untouched."""
    project = tmp_path / "project"
    project.mkdir()
    env = {"HOME": str(tmp_path / "home"), "PATH": "/usr/bin:/bin"}

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["sh", str(INSTALLER), *args, "--source", str(HOSTS)],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(project),
        )
        assert result.returncode == 0, f"{args}: {result.stderr}{result.stdout}"
        return result

    # A file that is not ours must survive an uninstall untouched.
    foreign = project / "AGENTS.md" if host == "agents-md" else None
    if foreign is not None:
        foreign.write_text("# our own agent notes\n", encoding="utf-8")

    run("--host", host, "--project", ".")
    installed = [p for p in project.rglob("*") if p.is_file()]
    assert installed, f"{host}: nothing was installed into the project"
    for path in installed:
        assert MARKER in path.read_text(encoding="utf-8"), f"{path} carries no marker"

    run("--host", host, "--uninstall", "--project", ".")
    if foreign is not None:
        assert foreign.read_text(encoding="utf-8") == "# our own agent notes\n"

    leftover = [
        p
        for p in project.rglob("*")
        if p.is_file() and p.name != "AGENTS.md" and MARKER in p.read_text(encoding="utf-8")
    ]
    assert leftover == [], f"{host}: uninstall left marked files behind: {leftover}"


def test_an_uninstall_needs_no_pack(tmp_path: Path) -> None:
    """The marker is what makes a file ours, so removal must not depend on the pack still existing."""
    project = tmp_path / "project"
    project.mkdir()
    env = {"HOME": str(tmp_path / "home"), "PATH": "/usr/bin:/bin"}

    install = subprocess.run(
        ["sh", str(INSTALLER), "--host", "claude-code", "--project", ".", "--source", str(HOSTS)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(project),
    )
    assert install.returncode == 0, install.stderr
    assert list((project / ".claude" / "skills").rglob("SKILL.md"))

    # No --source, and no pack anywhere near the project.
    remove = subprocess.run(
        ["sh", str(INSTALLER), "--host", "claude-code", "--project", ".", "--uninstall"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(project),
    )
    assert remove.returncode == 0, remove.stderr
    assert not list((project / ".claude" / "skills").rglob("SKILL.md")), (
        "an uninstall without the pack left files behind"
    )


def test_all_covers_every_host_it_claims() -> None:
    """--host all must land somewhere each host above actually reads, or it is a false promise."""
    script = INSTALLER.read_text(encoding="utf-8")
    default_all = re.search(r"^DEFAULT_ALL=\"([^\"]+)\"", script, re.MULTILINE)
    assert default_all is not None, "the default host set is not declared"
    hosts = default_all.group(1).split()
    assert "agents-skills" in hosts, ".agents/skills is the root six hosts read"
    assert "claude-code" in hosts, "Claude Code does not read .agents/skills"


def test_the_installer_refuses_to_clobber_a_foreign_agents_md(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    agents = project / "AGENTS.md"
    agents.write_text("# house rules\n", encoding="utf-8")

    result = subprocess.run(
        ["sh", str(INSTALLER), "--host", "agents-md", "--project", ".", "--source", str(HOSTS)],
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"},
        cwd=str(project),
    )
    assert result.returncode == 0, result.stderr
    text = agents.read_text(encoding="utf-8")
    assert text.startswith("# house rules\n"), "an existing AGENTS.md was overwritten"
    assert MARKER in text, "the jeval block was not appended"
