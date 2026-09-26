"""Guard against documented-but-unimplemented features.

The defect this file exists to catch: the README and the CLI both told users to map `label_from`
in their ingest map, and nothing ever read it. No test failed, because every test asked whether
the code did what it said, and nobody asked whether the documentation matched the code.

The documents an agent reads are now a first-class surface: `README.md`, `llms.txt` (the
machine-readable entry point) and `docs/agent-setup.md` (the setup playbook). A wrong flag in any of
them is a wrong instruction to an agent, so all three are checked, not just the README.

Two directions, both mechanical:

1. Every `jeval <command>` and every flag the README *shows as a command* must exist on the real
   command surface, and every command the README calls a milestone must still be missing — so
   shipping it forces the status table to be updated.
2. Every key the ingest-map scaffold tells the user to fill in must be consumed by code that acts
   on it, not merely declared on a dataclass and parsed out of the file.
"""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest
from typer import main as typer_main
from typer.testing import CliRunner

from jeval.cli import app
from jeval.config import IngestMap

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"
PACKAGE = REPO / "jeval"

# Every document that tells a reader (or an agent) to run something.
DOC_PATHS = (
    README,
    REPO / "llms.txt",
    REPO / "docs" / "agent-setup.md",
    REPO / "docs" / "instrumenting-a-service.md",
    REPO / "docs" / "library.md",
    REPO / "docs" / "skills.md",
)

# `jeval ...` may be shown wrapped in an installer, or bare. The whole prefix is optional: without
# the trailing `?` this matched only launcher-prefixed lines and silently ignored every plain
# `jeval ...` in the documents, so the guard read a fraction of what it claimed to check.
LAUNCHERS = r"(?:uv run |uvx(?: --from \S+)? )?"

runner = CliRunner()

# Scaffolded keys the user is told to fill in, so something must act on them.
INGEST_MAP_KEYS = ("field_map", "defaults", "questions_field", "label_from")

# Keys the mapping type itself applies while transforming a row: consuming them inside config.py
# is real consumption.
APPLIED_BY_THE_MAPPING = frozenset({"field_map", "defaults", "questions_field"})

# `label_from` is different in kind: config.py parsing and validating the spec is not the feature.
# Something outside config.py has to act on label rows, or the key is a documented no-op again.
REQUIRES_EXTERNAL_CONSUMER = frozenset({"label_from"})

# Commands the README documents as milestones rather than as available. If one of these starts
# existing, the README status table is stale and this test says so.
DOCUMENTED_AS_MILESTONE: tuple[str, ...] = ()

# `version` is a document marker in the ingest map: the loader accepts and ignores it. It is the
# one tolerated non-consumed key, named here rather than hidden.
TOLERATED_KEYS = ("version",)


def _readme() -> str:
    return README.read_text(encoding="utf-8")


# The canonical agent skills are instructions an agent follows, so they are held to the same
# standard as the docs: a flag that does not exist is a wrong instruction, not a typo. They are
# generated into per-host layouts by tools/export_skills.py, and only the sources are parsed here —
# the generated copies are checked for byte equality elsewhere.
SKILLS_DIR = REPO / "skills"


def _skill_texts() -> list[tuple[str, str]]:
    return [
        (f"skills/{path.parent.name}/SKILL.md", path.read_text(encoding="utf-8"))
        for path in sorted(SKILLS_DIR.glob("*/SKILL.md"))
    ]


def _doc_texts() -> list[tuple[str, str]]:
    """(name, text) for every document that shows commands."""
    pairs = [(path.name, path.read_text(encoding="utf-8")) for path in DOC_PATHS if path.exists()]
    return pairs + _skill_texts()


def _registered_commands() -> set[str]:
    names = set()
    for info in app.registered_commands:
        if info.name:
            names.add(info.name)
        elif info.callback is not None:
            names.add(info.callback.__name__.replace("_", "-"))
    return names


def _ansi_free(text: str) -> str:
    """Drop ANSI styling.

    Under CI (``GITHUB_ACTIONS=true``) rich forces colour on, and its highlighter splits an option
    name into separately styled spans, so the literal text ``--out-dir`` is not contiguous in the
    byte stream. Anything that searches rendered help must strip styling first.
    """
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _declared_flags(command: str | None = None) -> set[str]:
    """The options a command really declares.

    Read from Click's command objects rather than from rendered help: help text is a rendering
    (colour, wrapping, width) and the rendering is not the contract.
    """
    group = typer_main.get_command(app)
    target: Any = group if command is None else group.commands[command]
    flags: set[str] = set()
    for param in target.params:
        for opt in (*getattr(param, "opts", ()), *getattr(param, "secondary_opts", ())):
            if opt.startswith("--"):
                flags.add(opt)
    flags.add("--help")  # Click always provides it, and never lists it as a parameter
    return flags


def _help_flags(command: str | None = None) -> set[str]:
    """Declared flags, with `--help` checked as a live smoke test of the help path."""
    args = [command, "--help"] if command else ["--help"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, f"`jeval {' '.join(args)}` failed: {result.stdout}"
    rendered = _ansi_free(result.stdout)
    assert re.search(r"(Usage|Options)", rendered), (
        f"`jeval {' '.join(args)}` rendered no usage text; stripped output was {rendered[:200]!r}"
    )
    return _declared_flags(command)


def _invocations() -> list[tuple[str, str, set[str]]]:
    """(document, command, flags) for every `jeval <command>` the documents show as a command.

    Only code fences and inline code spans count: prose like "jeval measures ..." is a sentence,
    not a command. A command shown inside `uvx --from ...` or `uv run` is still a command.
    """
    found: list[tuple[str, str, set[str]]] = []
    for name, text in _doc_texts():
        snippets: list[str] = re.findall(r"```[a-z]*\n(.*?)```", text, re.DOTALL)
        snippets += re.findall(r"`([^`\n]+)`", text)
        for snippet in snippets:
            for line in snippet.splitlines():
                match = re.match(
                    rf"\s*\$?\s*{LAUNCHERS}jeval\s+(?P<cmd>[a-z][a-z-]*)(?P<rest>.*)$", line
                )
                if match:
                    found.append(
                        (
                            name,
                            match.group("cmd"),
                            set(re.findall(r"--[a-z][a-z-]+", match.group("rest"))),
                        )
                    )
    return found


def test_every_documented_command_exists() -> None:
    available = _registered_commands()
    missing = sorted(
        {
            command
            for _, command, _ in _invocations()
            if command not in available and command not in DOCUMENTED_AS_MILESTONE
        }
    )
    assert missing == [], f"the docs show commands that do not exist: {missing}"


# A wrong install line does not fail loudly: it runs someone else's code. `pip install jeval` on
# PyPI is an unrelated project, so a line naming it must say so.
INSTALL_HAZARD_MARKERS = ("unrelated", "belongs to", "not ours", "someone else")


def test_no_document_sends_a_reader_to_the_wrong_distribution() -> None:
    """The distribution is `jeval-cli`; `jeval` on PyPI belongs to another account."""
    offenders: list[str] = []
    for name, text in _doc_texts():
        for number, line in enumerate(text.splitlines(), 1):
            if re.search(r"pip install jeval(?![-_]?cli)", line) and not any(
                marker in line for marker in INSTALL_HAZARD_MARKERS
            ):
                offenders.append(f"{name}:{number}")
    assert offenders == [], (
        f"these lines install a different project from PyPI: {offenders}. "
        "Name the distribution `jeval-cli`, or say in the line that `jeval` is not ours."
    )


def test_every_agent_facing_document_is_actually_parsed() -> None:
    """A guard that quietly stopped reading a document would pass while the docs lied.

    Every parse target — the README, `llms.txt`, the docs, and each canonical skill — has to
    contribute at least one parsed command, or this file is checking less than it claims. Skills
    are counted explicitly: a glob that matched nothing would otherwise pass as coverage.
    """
    expected = {name for name, _ in _doc_texts()}
    covered = {name for name, _, _ in _invocations()}

    missing = sorted(expected - covered)
    assert missing == [], f"no command was parsed out of {missing}; the guard does not read them"

    skills = _skill_texts()
    assert len(skills) >= 5, f"expected the canonical skill pack, found {len(skills)} skills"


def test_a_milestone_command_is_not_secretly_shipped() -> None:
    """The status table is a claim; shipping a milestone without updating it is a stale claim."""
    available = _registered_commands()
    shipped = sorted(command for command in DOCUMENTED_AS_MILESTONE if command in available)
    assert shipped == [], (
        f"{shipped} now exist, so the README status table still marks them as milestones. "
        "Move them to implemented (and out of DOCUMENTED_AS_MILESTONE) in the same change."
    )


def test_every_documented_flag_exists_on_its_command() -> None:
    """Flags are checked per command, so a flag shown against the wrong command is caught."""
    available = _registered_commands()
    offenders: list[str] = []
    for _, command, flags in _invocations():
        if command not in available or not flags:
            continue
        unknown = flags - _help_flags(command)
        if unknown:
            offenders.append(f"jeval {command}: {sorted(unknown)}")
    assert offenders == [], f"the docs show flags their command does not accept: {offenders}"


def test_the_paired_drift_flag_is_documented_and_declared() -> None:
    """`--paired` is shown in the docs as a command and exists on `jeval drift`.

    The generic flag check only catches a documented flag that does not exist. This pins the other
    direction for the head-to-head comparison: dropping the flag, or the docs that show it, fails
    here in the same change.
    """
    documented = {
        name
        for name, command, flags in _invocations()
        if command == "drift" and "--paired" in flags
    }
    assert {"README.md", "llms.txt"} <= documented, documented
    assert "--paired" in _help_flags("drift")


def test_the_documented_demo_command_is_real() -> None:
    """The example the README tells readers to run is reproduced literally."""
    line = next(
        (
            candidate
            for candidate in _readme().splitlines()
            if "jeval demo" in candidate and "--out-dir" in candidate
        ),
        None,
    )
    assert line is not None, "the README no longer shows how to rebuild the example report"
    flags = set(re.findall(r"--[a-z][a-z-]+", line))
    assert flags <= _help_flags("demo"), f"demo does not accept: {sorted(flags)}"


@pytest.mark.parametrize("key", INGEST_MAP_KEYS)
def test_every_ingest_map_key_is_consumed(key: str) -> None:
    """A scaffolded key nobody reads is a documented no-op.

    Consumption means code acts on the value: either the mapping type itself (`self.<key>`) or
    another module through the attribute. Constructing the dataclass with a keyword is not
    consumption, which is exactly how `label_from` shipped as a no-op.
    """
    config_source = (PACKAGE / "config.py").read_text(encoding="utf-8")
    consumed_inside = re.search(rf"self\.{key}\b", config_source) is not None
    readers = [
        path.relative_to(REPO).as_posix()
        for path in sorted(PACKAGE.rglob("*.py"))
        if path.name not in {"config.py", "__init__.py"}
        and re.search(rf"\.{key}\b", path.read_text(encoding="utf-8"))
    ]
    if key in REQUIRES_EXTERNAL_CONSUMER:
        # The sanctioned way to read this block is the parsed accessor, so a module that calls
        # `label_spec()` counts as a consumer.
        accessor = key.replace("_from", "_spec")  # label_from -> label_spec
        readers += [
            path.relative_to(REPO).as_posix()
            for path in sorted(PACKAGE.rglob("*.py"))
            if path.name not in {"config.py", "__init__.py"}
            and re.search(rf"\.{accessor}\(\)", path.read_text(encoding="utf-8"))
        ]
        assert readers, (
            f"{key!r} is offered to users in the scaffold and the README, and config.py only "
            f"parses and validates it. Nothing outside config.py acts on it, so the feature is a "
            f"documented no-op — which is the defect this test exists for."
        )
        return
    assert consumed_inside or readers, (
        f"ingest-map key {key!r} is offered to users in the scaffold and the README, but nothing "
        f"reads it: not the mapping type, and no module outside config.py"
    )


def test_scaffolded_ingest_keys_match_the_mapping_type() -> None:
    declared = {field.name for field in fields(IngestMap)} | set(TOLERATED_KEYS)
    scaffold = re.search(
        r'INGEST_SCAFFOLD = """(.*?)"""',
        (PACKAGE / "cli.py").read_text(encoding="utf-8"),
        re.DOTALL,
    )
    assert scaffold is not None
    documented = {
        line.split(":")[0].strip()
        for line in scaffold.group(1).splitlines()
        if line and not line.startswith((" ", "#", '"'))
    }
    assert documented <= declared, (
        f"the scaffold offers keys the mapping cannot hold: {sorted(documented - declared)}"
    )


def test_the_reported_version_is_the_installed_distribution() -> None:
    """`jeval --version` reported 0.1.0 from a 0.1.2 wheel.

    The package asked for the distribution `jeval`, which is not its name on PyPI: the lookup raised
    and fell back to a hardcoded string, and with the unrelated `jeval` project installed it would
    have reported *that* version instead. The fallback is now a value that cannot be mistaken for a
    release.
    """
    from importlib.metadata import version

    import jeval

    assert jeval.__version__ == version(jeval.DISTRIBUTION)
    assert not jeval.__version__.startswith("0.0.0"), (
        "the installed distribution was not found, so the version shown is a placeholder"
    )
