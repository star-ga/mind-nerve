#!/usr/bin/env bash
# autoresearch launcher with CLI rotation, auth load, and Telegram start ping.
#
# Usage:
#   ./run-autoresearch.sh            # default: claude (uses ANTHROPIC_API_KEY)
#   ./run-autoresearch.sh codex      # use codex CLI
#   ./run-autoresearch.sh gemini     # use gemini CLI
#
# Each CLI authenticates via its local CLI auth (subscription / OAuth):
#   - claude uses ~/.claude credentials (Claude Code subscription) — we
#     EXPLICITLY UNSET ANTHROPIC_API_KEY so the CLI doesn't accidentally
#     fall back to a pay-per-token API key with a separate (smaller) budget
#   - codex uses local CLI auth — spawned with
#     --dangerously-bypass-approvals-and-sandbox to skip bwrap (broken on host)
#   - gemini uses local CLI auth (--yolo)
#   - vibe uses local CLI auth
#
# When one CLI hits its rate / quota limit, restart with the next:
#   ./run-autoresearch.sh codex
# The harness picks up where it stopped (results_file accumulates).

set -e

CLI="${1:-claude}"
TAG="$(date +%b%d | tr A-Z a-z)$(printf %s "$CLI" | head -c1)"

# Sanity: chosen CLI is in the supported set.
case "$CLI" in
    claude)
        # Use the user's Claude Code subscription auth, NOT a pay-per-token
        # API key. Unset ANTHROPIC_API_KEY explicitly so the CLI falls
        # back to its stored OAuth credentials (~/.claude/.credentials.json).
        unset ANTHROPIC_API_KEY
        ;;
    codex|gemini|vibe)
        # local CLI auth; no env var probe.
        ;;
    *)
        echo "error: unknown CLI '$CLI' (expected: claude | codex | gemini | vibe)" >&2
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
