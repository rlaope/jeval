"""The installer is the primary install path, so it is a tested artifact.

The README tells a reader to pipe a file from `main` into `sh`. Nothing else checks that the file is
valid shell, that the URL in the document is the file that exists, or that the command it leaves
behind is the command the rest of the docs use. A syntax error there is a broken install for every
user, and it is invisible to every other test in this suite.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INSTALLER = REPO / "install.sh"
DOCS = (REPO / "README.md", REPO / "llms.txt", REPO / "docs" / "agent-setup.md")


def test_the_installer_is_valid_posix_shell() -> None:
    result = subprocess.run(
        ["sh", "-n", str(INSTALLER)], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, f"sh -n rejected the installer: {result.stderr}"


def test_the_documented_install_url_is_the_file_that_exists() -> None:
    """A raw URL is a claim about a path on `main`; renaming the file breaks every install."""
    urls = {
        url
        for doc in DOCS
        for url in re.findall(
            r"https://raw\.githubusercontent\.com/\S+/install\.sh", doc.read_text()
        )
    }

    assert urls, "no document shows how to install the tool"
    for url in urls:
        assert url == "https://raw.githubusercontent.com/rlaope/jeval/main/install.sh", url
    assert INSTALLER.exists(), "a document points at install.sh, which is not in the repository"


def test_the_installer_puts_a_jeval_command_on_path() -> None:
    """The point of the installer: `jeval`, not `python -m jeval` or a path under a venv."""
    script = INSTALLER.read_text(encoding="utf-8")

    assert re.search(r'ln -sfn "\$VENV/bin/jeval" "\$LINK"', script), (
        "the installer no longer links the jeval executable"
    )
    assert 'BIN_DIR="${JEVAL_BIN_DIR:-$HOME/.local/bin}"' in script
    assert 'if on_path "$BIN_DIR"; then' in script, (
        "the installer must say when the bin directory is not on PATH, or the user gets a command "
        "that does not resolve"
    )


def _installer_code() -> str:
    """What the installer actually runs: no usage text, no comments.

    The script names uv and pipx in the sentence that says it needs neither, so a search over the
    whole file reports the documentation as a violation.
    """
    script = INSTALLER.read_text(encoding="utf-8")
    script = re.sub(r"<<'USAGE'\n.*?\nUSAGE\n", "", script, flags=re.DOTALL)
    return "\n".join(line for line in script.splitlines() if not line.lstrip().startswith("#"))


def test_the_installer_needs_no_uv_no_pipx_and_no_root() -> None:
    """It exists because the user asked for an install that is not a wrapper they must remember."""
    code = _installer_code()

    for foreign in ("uvx", "pipx", "sudo", "uv tool"):
        assert not re.search(rf"(?:^|[;&|]\s*){re.escape(foreign)}(?:\s|$)", code, re.MULTILINE), (
            f"the installer runs {foreign}"
        )
    assert "--force-reinstall" in code, "a re-run must upgrade rather than do nothing"
    assert "uninstall" in code, "an installer that cannot be undone is a trap"
