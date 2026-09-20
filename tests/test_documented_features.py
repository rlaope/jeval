"""Guard against documented-but-unimplemented features.

The defect this file exists to catch: the README and the CLI both told users to map `label_from`
in their ingest map, and nothing ever read it. No test failed, because every test asked whether
the code did what it said, and nobody asked whether the documentation matched the code.

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


def _invocations() -> list[tuple[str, set[str]]]:
    """(command, flags) for every `jeval <command>` the README shows as a command.

    Only code fences and inline code spans count: prose like "jeval measures ..." is a sentence,
    not a command.
    """
    readme = _readme()
    snippets: list[str] = re.findall(r"```[a-z]*\n(.*?)```", readme, re.DOTALL)
    snippets += re.findall(r"`([^`\n]+)`", readme)
    found: list[tuple[str, set[str]]] = []
    for snippet in snippets:
        for line in snippet.splitlines():
            match = re.match(r"\s*\$?\s*jeval\s+(?P<cmd>[a-z][a-z-]*)(?P<rest>.*)$", line)
            if match:
                found.append(
                    (match.group("cmd"), set(re.findall(r"--[a-z][a-z-]+", match.group("rest"))))
                )
    return found


def test_every_documented_command_exists() -> None:
    available = _registered_commands()
    missing = sorted(
        {
            command
            for command, _ in _invocations()
            if command not in available and command not in DOCUMENTED_AS_MILESTONE
        }
    )
    assert missing == [], f"README shows commands that do not exist: {missing}"


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
    for command, flags in _invocations():
        if command not in available or not flags:
            continue
        unknown = flags - _help_flags(command)
        if unknown:
            offenders.append(f"jeval {command}: {sorted(unknown)}")
    assert offenders == [], f"README shows flags their command does not accept: {offenders}"


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
