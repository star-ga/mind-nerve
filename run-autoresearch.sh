#!/usr/bin/env bash
# autoresearch launcher with CLI rotation, auth load, and Telegram start ping.
#
# Usage:
#   ./run-autoresearch.sh            # default: claude (uses ANTHROPIC_API_KEY)
#   ./run-autoresearch.sh codex      # use codex CLI
#   ./run-autoresearch.sh gemini     # use gemini CLI
#
# Each CLI authenticates differently:
#   - claude needs ANTHROPIC_API_KEY (loaded from ~/.claude-ultimate/.env)
#   - codex uses local CLI auth (no env var needed) — spawned with
#     -s workspace-write so the agent can apply edits to target_files
#   - gemini uses local CLI auth (no env var needed)
#
# When one CLI hits its rate / quota limit, restart with the next:
#   ./run-autoresearch.sh codex
# The harness picks up where it stopped (results_file accumulates).

set -e

CLI="${1:-claude}"
TAG="$(date +%b%d | tr A-Z a-z)$(printf %s "$CLI" | head -c1)"

# Load .env keys without echoing them.
if [ -f "$HOME/.claude-ultimate/.env" ]; then
    set -a
    . "$HOME/.claude-ultimate/.env"
    set +a
fi

# Sanity: required key for chosen CLI.
case "$CLI" in
    claude)
        if [ -z "$ANTHROPIC_API_KEY" ]; then
            echo "error: ANTHROPIC_API_KEY not set after sourcing ~/.claude-ultimate/.env" >&2
            exit 2
        fi
        ;;
    codex|gemini)
        # local CLI auth; no env var probe.
        ;;
    *)
        echo "error: unknown CLI '$CLI' (expected: claude | codex | gemini)" >&2
        exit 2
        ;;
esac

echo "[autoresearch] launching with CLI=$CLI tag=$TAG cwd=$PWD"

# Telegram ping on session start. One message per launch (not per
# iteration). Failure is non-fatal; the loop runs either way.
TG_HELPER="$HOME/.openclaw/workspace/outbound/telegram_send.py"
if [ -f "$TG_HELPER" ]; then
    python3 "$TG_HELPER" msg \
        "🔬 mind-nerve autoresearch started · CLI=$CLI · tag=$TAG · branch=autoresearch/$TAG" \
        > /dev/null 2>&1 || true
fi

# autorun.py accepts --cli on the command line as an override of the
# YAML cli field. Use it so the same config supports any backend.
exec python3 /home/n/autoresearch/autorun.py \
    --config autoresearch.yaml \
    --cli "$CLI" \
    --tag "$TAG"
