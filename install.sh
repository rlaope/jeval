#!/usr/bin/env sh
# Install jeval as a `jeval` command on PATH.
#
# No uv, no pipx, no root, no PyPI: it downloads the wheel a release tag produced, installs it into
# a private virtual environment, and links the `jeval` executable into a directory that is on PATH.
#
#   curl -fsSL https://raw.githubusercontent.com/rlaope/jeval/main/install.sh | sh
#
# Re-run the same command to upgrade. `install.sh uninstall` removes what it created.
set -eu

REPO="rlaope/jeval"
VERSION="${JEVAL_VERSION:-}"                    # empty = the newest release
PREFIX="${JEVAL_HOME:-$HOME/.local/share/jeval}"
BIN_DIR="${JEVAL_BIN_DIR:-$HOME/.local/bin}"
MODE="install"
MODIFY_PATH="0"

usage() {
    cat <<'USAGE'
Install jeval as a `jeval` command on PATH, without uv, pipx or root.

Usage:
  install.sh [options]
  install.sh uninstall [options]

Options:
  --version TAG     install a specific release tag (default: the newest release)
  --home DIR        where the private virtual environment lives
                    (default: ~/.local/share/jeval)
  --bin-dir DIR     where the `jeval` link is created (default: ~/.local/bin)
  --modify-path     append the bin directory to the shell's rc file if it is missing from PATH
  -h, --help        this message

Environment: JEVAL_VERSION, JEVAL_HOME, JEVAL_BIN_DIR

What it does: downloads the wheel built for the release, creates a virtual environment under
--home, installs the wheel into it, links `jeval` into --bin-dir, and runs `jeval --version`.
Nothing outside those two directories is touched, and nothing is written into the current directory.
USAGE
}

say() { printf '%s\n' "$*"; }
die() { printf '%s\n' "install.sh: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        uninstall|--uninstall) MODE="uninstall"; shift ;;
        --version) [ $# -ge 2 ] || die "--version needs a tag"; VERSION="$2"; shift 2 ;;
        --version=*) VERSION="${1#--version=}"; shift ;;
        --home) [ $# -ge 2 ] || die "--home needs a directory"; PREFIX="$2"; shift 2 ;;
        --home=*) PREFIX="${1#--home=}"; shift ;;
        --bin-dir) [ $# -ge 2 ] || die "--bin-dir needs a directory"; BIN_DIR="$2"; shift 2 ;;
        --bin-dir=*) BIN_DIR="${1#--bin-dir=}"; shift ;;
        --modify-path) MODIFY_PATH="1"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (try --help)" ;;
    esac
done

VENV="$PREFIX/venv"
LINK="$BIN_DIR/jeval"

on_path() {
    case ":${PATH:-}:" in
        *":$1:"*) return 0 ;;
        *) return 1 ;;
    esac
}

rc_file() {
    # The file to edit when asked to fix PATH. zsh is the macOS default, bash elsewhere.
    if [ -n "${SHELL:-}" ] && [ "$(basename "${SHELL}")" = "zsh" ]; then
        printf '%s\n' "${ZDOTDIR:-$HOME}/.zshrc"
    elif [ -n "${SHELL:-}" ] && [ "$(basename "${SHELL}")" = "bash" ]; then
        printf '%s\n' "$HOME/.bashrc"
    else
        printf '%s\n' "$HOME/.profile"
    fi
}

if [ "$MODE" = "uninstall" ]; then
    removed=0
    if [ -L "$LINK" ]; then rm -f "$LINK"; say "removed $LINK"; removed=1; fi
    if [ -d "$PREFIX" ]; then rm -rf "$PREFIX"; say "removed $PREFIX"; removed=1; fi
    [ "$removed" = "1" ] || say "nothing to remove at $LINK or $PREFIX"
    exit 0
fi

# --- a python new enough to run jeval, and able to make a virtual environment --------------------
PY=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' \
        >/dev/null 2>&1; then
        PY="$candidate"
        break
    fi
done
[ -n "$PY" ] || die "no python3.10 or newer on PATH. Install Python 3.10+ and run this again."
say "python:  $PY ($("$PY" -c 'import platform; print(platform.python_version())'))"

# --- which release to install --------------------------------------------------------------------
if [ -z "$VERSION" ]; then
    command -v curl >/dev/null 2>&1 || die "curl is required to find the newest release"
    VERSION="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
        | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1)"
    [ -n "$VERSION" ] || die "could not read the newest release tag. Pass --version vX.Y.Z."
fi
NUMBER="${VERSION#v}"
URL=""
# Ask the release what it carries instead of guessing the file name: the distribution was renamed
# once already, so older tags hold a differently named wheel for the same tool.
if command -v curl >/dev/null 2>&1; then
    URL="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/tags/$VERSION" \
        | sed -n 's/.*"browser_download_url"[[:space:]]*:[[:space:]]*"\([^"]*\.whl\)".*/\1/p' \
        | head -n 1)"
fi
if [ -z "$URL" ]; then
    # The API can be rate limited, so fall back to the conventional name — but only after checking
    # the asset is really there, so a wrong tag is one clear line instead of pip's error output.
    candidate="https://github.com/$REPO/releases/download/$VERSION/jeval_cli-${NUMBER}-py3-none-any.whl"
    if command -v curl >/dev/null 2>&1 && curl -fsIL -o /dev/null "$candidate" 2>/dev/null >/dev/null; then
        URL="$candidate"
    fi
fi
[ -n "$URL" ] || die "no release wheel for $VERSION. Check the tag, or see https://github.com/$REPO/releases"
say "release: $VERSION"
say "wheel:   $URL"

# --- install into a private environment ----------------------------------------------------------
mkdir -p "$PREFIX" "$BIN_DIR"
if [ ! -x "$VENV/bin/python" ]; then
    say "creating a virtual environment in $VENV"
    "$PY" -m venv "$VENV" \
        || die "could not create a virtual environment. On Debian/Ubuntu install python3-venv."
fi

# --force-reinstall makes a re-run an upgrade rather than a no-op.
"$VENV/bin/python" -m pip install --disable-pip-version-check --no-input --quiet \
    --upgrade --force-reinstall "$URL" || die "installing $URL failed"

ln -sfn "$VENV/bin/jeval" "$LINK"
[ -x "$LINK" ] || die "the link at $LINK is not executable; the wheel may not ship a jeval script"

# --- say what happened, and what the shell still needs -------------------------------------------
REPORTED="$("$LINK" --version 2>/dev/null || true)"
[ -n "$REPORTED" ] || die "installed, but '$LINK --version' printed nothing"
say "installed: $LINK -> $REPORTED"

if on_path "$BIN_DIR"; then
    say "done. Try: jeval demo"
else
    line="export PATH=\"$BIN_DIR:\$PATH\""
    if [ "$MODIFY_PATH" = "1" ]; then
        rc="$(rc_file)"
        mkdir -p "$(dirname "$rc")"
        if [ -f "$rc" ] && grep -qF "$BIN_DIR" "$rc"; then
            say "$rc already mentions $BIN_DIR"
        else
            printf '\n# added by the jeval installer\n%s\n' "$line" >>"$rc"
            say "appended to $rc: $line"
        fi
        say "open a new shell, or run: . \"$rc\""
    else
        say ""
        say "$BIN_DIR is not on your PATH yet. Add it with:"
        say "    $line"
        say "or re-run this installer with --modify-path to append that line for you."
    fi
    say ""
    say "Either way, until the PATH is updated: $LINK demo"
fi
