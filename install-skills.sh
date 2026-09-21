#!/bin/sh
# Install the jeval agent skills into the layout your agent host expects.
#
#   sh install-skills.sh --list
#   sh install-skills.sh --host claude-code
#   sh install-skills.sh --host cursor --project .
#   sh install-skills.sh --host all --global
#   sh install-skills.sh --host agents-md --project . --dry-run
#   sh install-skills.sh --host claude-code --uninstall
#
# The pack is a set of markdown files; nothing here runs jeval, installs a package, or needs root.
# Run it from a checkout (the pack is found under skills/hosts) or pipe it from the network, in
# which case the latest release archive is unpacked into a temporary directory first.
set -eu

JEVAL_SKILLS_URL="${JEVAL_SKILLS_URL:-https://github.com/rlaope/jeval/releases/latest/download/jeval-skills.tar.gz}"
MARKER="jeval-agent-skills"

usage() {
    cat <<'USAGE'
Install the jeval agent skills for the coding agent you use.

Usage: sh install-skills.sh [options]

  --host NAME      agents-skills | claude-code | cursor | hermes | opencode | pi | openclaw |
                   agents-md | all                                    (default: all)
  --global         install for your user account (default)
  --project DIR    install into a project directory instead
  --list           show the hosts, their destinations and the skills in the pack
  --dry-run        print what would be written, write nothing
  --force          replace an AGENTS.md that exists and is not ours
  --uninstall      remove what this script installed for that host
  --source DIR     use a pack directory or archive instead of finding one
  -h, --help       this text

--host all installs the two roots that between them cover every host above: .agents/skills, which
Codex CLI, Hermes, OpenClaw, Pi, Cursor and OpenCode all read, and .claude/skills, which is what
Claude Code reads. A host that reads both roots will list a skill from the first one it finds.

Every installed file carries the marker "jeval-agent-skills": a YAML comment inside the frontmatter
of a skill, and an HTML comment around the block appended to an AGENTS.md. An uninstall removes
exactly the files and blocks carrying that marker and nothing else, and it needs no pack and no
network access to do it.
USAGE
}

host="all"
scope="global"
project=""
dry_run=0
force=0
uninstall=0
source_dir=""
mode_list=0
tmp_root=""
pack_dir=""

ALL_HOSTS="agents-skills claude-code cursor hermes opencode pi openclaw agents-md"
DEFAULT_ALL="agents-skills claude-code"

while [ $# -gt 0 ]; do
    case "$1" in
        --host) host="${2:?--host needs a value}"; shift 2 ;;
        --global) scope="global"; shift ;;
        --project) scope="project"; project="${2:?--project needs a directory}"; shift 2 ;;
        --list) mode_list=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        --force) force=1; shift ;;
        --uninstall) uninstall=1; shift ;;
        --source) source_dir="${2:?--source needs a directory}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# Status-neutral on purpose: this runs as an EXIT trap, and a failing last command in a trap
# becomes the script's exit status — so an uninstall that removed everything would still report
# failure because the temporary directory name was empty.
cleanup() {
    if [ -n "$tmp_root" ] && [ -d "$tmp_root" ]; then
        rm -rf "$tmp_root"
    fi
}
trap cleanup EXIT INT TERM

# --- find the pack -------------------------------------------------------------------------------

use_pack_dir() {
    for candidate in "$1" "$1/skills/hosts" "$1/hosts"; do
        if [ -f "$candidate/manifest.json" ]; then
            pack_dir=$(CDPATH= cd -- "$candidate" && pwd)
            return 0
        fi
    done
    return 1
}

find_pack() {
    if [ -n "$source_dir" ]; then
        if use_pack_dir "$source_dir"; then
            return
        fi
        if [ -f "$source_dir" ] && tar -tzf "$source_dir" >/dev/null 2>&1; then
            tmp_root=$(mktemp -d)
            tar -xzf "$source_dir" -C "$tmp_root"
            use_pack_dir "$tmp_root" || { echo "$source_dir: holds no manifest.json" >&2; exit 1; }
            return
        fi
        echo "$source_dir: not a pack directory and not a tar archive" >&2
        exit 1
    fi

    script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" 2>/dev/null && pwd || echo .)
    if use_pack_dir "$script_dir" || use_pack_dir "."; then
        return
    fi

    # An uninstall does not need the pack: it removes files by their marker, so it keeps working
    # after the pack is gone.
    if [ "$uninstall" = 1 ]; then
        return
    fi

    if ! command -v curl >/dev/null 2>&1; then
        echo "No local pack found and curl is unavailable." >&2
        echo "Clone the repository or download $JEVAL_SKILLS_URL by hand." >&2
        exit 1
    fi
    tmp_root=$(mktemp -d)
    echo "fetching $JEVAL_SKILLS_URL"
    if ! curl -fsSL "$JEVAL_SKILLS_URL" | tar -xz -C "$tmp_root" 2>/dev/null; then
        echo "could not download the skills archive." >&2
        echo "Clone https://github.com/rlaope/jeval and run this script from the checkout instead." >&2
        exit 1
    fi
    use_pack_dir "$tmp_root" || { echo "the archive holds no manifest.json" >&2; exit 1; }
}

skills_in_pack() {
    if [ -f "$pack_dir/skills.txt" ]; then
        grep -v '^#' "$pack_dir/skills.txt" | grep -v '^$' | sort
        return 0
    fi
    # Fallback for a pack built before skills.txt existed: this JSON parse only works while the
    # array sits on one line, which is why the generated index exists.
    sed -n 's/.*"skills": \[\(.*\)\].*/\1/p' "$pack_dir/manifest.json" |
        tr -d ' "' | tr ',' '\n' | grep -v '^$' | sort
    return 0
}

# Where this host's files sit inside the pack, which mirrors where the host reads them.
host_target_in_pack() {
    if [ -f "$pack_dir/$1/target.txt" ]; then
        cat "$pack_dir/$1/target.txt"
    fi
}

# An empty skill list must fail the run: installing nothing and reporting success is how a broken
# install looks identical to a working one.
require_skills() {
    if [ -z "$(skills_in_pack)" ]; then
        echo "the pack at $pack_dir lists no skills; nothing was installed." >&2
        echo "Re-download the pack, or pass --source with a directory that has skills.txt." >&2
        exit 1
    fi
}

# $1 = host, $2 = global|project, $3 = project dir
dest_root_for() {
    case "$1" in
        agents-skills) [ "$2" = project ] && echo "$3/.agents/skills" || echo "$HOME/.agents/skills" ;;
        claude-code)   [ "$2" = project ] && echo "$3/.claude/skills" || echo "$HOME/.claude/skills" ;;
        cursor)        [ "$2" = project ] && echo "$3/.cursor/skills" || echo "$HOME/.cursor/skills" ;;
        hermes)        [ "$2" = project ] && echo "$3/.hermes/skills" || echo "$HOME/.hermes/skills" ;;
        opencode)      [ "$2" = project ] && echo "$3/.opencode/skills" || echo "$HOME/.config/opencode/skills" ;;
        pi)            [ "$2" = project ] && echo "$3/.pi/skills" || echo "$HOME/.pi/agent/skills" ;;
        openclaw)      [ "$2" = project ] && echo "$3/skills" || echo "$HOME/.openclaw/skills" ;;
        agents-md)     [ "$2" = project ] && echo "$3" || echo "$HOME/.codex" ;;
        *) return 1 ;;
    esac
}

# --- writing -------------------------------------------------------------------------------------
#
# POSIX sh has no local variables, so every helper below prefixes its own variables with a tag.
# Without that, a helper called inside install_host's loop overwrote the loop's `target` and `src`
# on the second iteration: the uninstall removed one skill and then reported a missing pack file.

# Copy a skill file, recording the marker as a YAML comment so the frontmatter stays valid: a
# comment before the opening --- would stop the host from seeing the frontmatter at all.
write_skill() {
    ws_src="$1" ws_dst="$2"
    if [ "$dry_run" = 1 ]; then
        echo "would write $ws_dst"
        return
    fi
    mkdir -p "$(dirname -- "$ws_dst")"
    if head -n 1 "$ws_src" | grep -q '^---$'; then
        { head -n 1 "$ws_src"; echo "# $MARKER"; tail -n +2 "$ws_src"; } > "$ws_dst.tmp"
    else
        { echo "# $MARKER"; cat "$ws_src"; } > "$ws_dst.tmp"
    fi
    mv "$ws_dst.tmp" "$ws_dst"
    echo "wrote $ws_dst"
}

remove_if_ours() {
    ours_file="$1"
    [ -f "$ours_file" ] || return 0
    if grep -q "$MARKER" "$ours_file" 2>/dev/null; then
        if [ "$dry_run" = 1 ]; then echo "would remove $ours_file"
        else rm -f "$ours_file"; echo "removed $ours_file"; fi
    else
        echo "not ours, leaving alone: $ours_file"
    fi
}

write_agents_md() {
    am_src="$1" am_dst="$2"
    if [ "$dry_run" = 1 ]; then echo "would write the jeval block in $am_dst"; return; fi
    mkdir -p "$(dirname -- "$am_dst")"
    if [ -f "$am_dst" ] && ! grep -q "$MARKER" "$am_dst" && [ "$force" != 1 ]; then
        # The leading newline is the separator, and remove_agents_block takes it back out again, so
        # a project's own AGENTS.md is byte-identical after an install and an uninstall.
        {
            printf '\n<!-- %s:begin -->\n' "$MARKER"
            cat "$am_src"
            printf '<!-- %s:end -->\n' "$MARKER"
        } >> "$am_dst"
        echo "appended the jeval block to $am_dst (your existing file was kept)"
        return
    fi
    {
        printf '<!-- %s:begin -->\n' "$MARKER"
        cat "$am_src"
        printf '<!-- %s:end -->\n' "$MARKER"
    } > "$am_dst.tmp" && mv "$am_dst.tmp" "$am_dst"
    echo "wrote $am_dst"
}

remove_agents_block() {
    rb_dst="$1"
    [ -f "$rb_dst" ] || { echo "nothing to remove: $rb_dst"; return; }
    if ! grep -q "$MARKER" "$rb_dst"; then
        echo "not ours, leaving alone: $rb_dst"
        return
    fi
    if [ "$dry_run" = 1 ]; then echo "would strip the jeval block from $rb_dst"; return; fi
    # Buffer the file, drop the block, and drop one trailing empty line — the separator the append
    # added. Anything else in the file is left exactly as it was.
    awk -v m="$MARKER" '
        index($0, m ":begin") { skip = 1; next }
        index($0, m ":end")   { skip = 0; next }
        !skip                 { lines[++n] = $0 }
        END {
            if (n > 0 && lines[n] == "") n--
            for (i = 1; i <= n; i++) print lines[i]
        }
    ' "$rb_dst" > "$rb_dst.tmp" && mv "$rb_dst.tmp" "$rb_dst"
    if ! grep -q '[^[:space:]]' "$rb_dst"; then
        # The file only ever held our block, which this script created: leaving an empty file
        # behind is litter, not a courtesy.
        rm -f "$rb_dst"
        echo "removed $rb_dst (it held nothing but the jeval block)"
        return
    fi
    echo "stripped the jeval block from $rb_dst"
}

# Removes by marker, with no pack and no skill list: this is the path an uninstall takes when the
# pack is no longer on the machine.
uninstall_by_marker() {
    ubm_host="$1" ubm_root="$2"
    case "$ubm_host" in
        agents-md)
            remove_agents_block "$ubm_root/AGENTS.md"
            ;;
        *)
            if [ ! -d "$ubm_root" ]; then
                echo "nothing to remove: $ubm_root"
                return 0
            fi
            find "$ubm_root" -maxdepth 3 -type f | while read -r ubm_found; do
                remove_if_ours "$ubm_found"
            done
            find "$ubm_root" -mindepth 1 -maxdepth 2 -type d -empty -delete 2>/dev/null || true
            ;;
    esac
}

install_host() {
    host_name="$1"
    root=$(dest_root_for "$host_name" "$scope" "$project") || {
        echo "unknown host: $host_name ($ALL_HOSTS | all)" >&2
        return 1
    }

    if [ "$uninstall" = 1 ] && [ -z "$pack_dir" ]; then
        uninstall_by_marker "$host_name" "$root"
        return 0
    fi

    if [ "$host_name" = agents-md ]; then
        src="$pack_dir/agents-md/AGENTS.md"
        [ -f "$src" ] || { echo "missing in pack: $src" >&2; return 1; }
        if [ "$uninstall" = 1 ]; then remove_agents_block "$root/AGENTS.md"
        else write_agents_md "$src" "$root/AGENTS.md"; fi
        return 0
    fi

    target=$(host_target_in_pack "$host_name")
    for skill in $(skills_in_pack); do
        src="$pack_dir/$host_name/$target/$skill/SKILL.md"
        [ -f "$src" ] || { echo "missing in pack: $src" >&2; return 1; }
        if [ "$uninstall" = 1 ]; then remove_if_ours "$root/$skill/SKILL.md"
        else write_skill "$src" "$root/$skill/SKILL.md"; fi
    done
    return 0
}

# --- main ----------------------------------------------------------------------------------------

find_pack

if [ "$mode_list" = 1 ]; then
    if [ -z "$pack_dir" ]; then
        echo "no pack found; pass --source with a pack directory or archive" >&2
        exit 1
    fi
    echo "pack: $pack_dir"
    echo "skills:"
    for skill in $(skills_in_pack); do echo "  $skill"; done
    echo "hosts (global -> project):"
    for h in $ALL_HOSTS; do
        printf '  %-14s %s\n' "$h" "$(dest_root_for "$h" global .) -> $(dest_root_for "$h" project './your-project')"
    done
    exit 0
fi

if [ "$uninstall" != 1 ]; then
    require_skills
fi

if [ "$host" = all ]; then
    for h in $DEFAULT_ALL; do install_host "$h"; done
else
    install_host "$host"
fi

if [ "$dry_run" = 1 ]; then
    echo "(dry run: nothing was written)"
fi
