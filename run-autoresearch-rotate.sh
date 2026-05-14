#!/usr/bin/env bash
# autoresearch CLI-rotation launcher.
#
# Wraps run-autoresearch.sh and rotates through CLIs when one's quota
# runs out. The inner autorun.py exits with code 2 when its
# max_consecutive_failures budget is exhausted (after this commit:
# 5055b8b... in mind-nerve / engine patch in /home/n/autoresearch/);
# we treat that exit code as "this CLI is dead for now, try the next."
#
# Order: claude → gemini → codex → vibe, then back to the top after a
# long sleep so transient quotas (gemini's hourly 429, claude's daily
# extra-usage cap) get a chance to recover.
#
# Usage:
#     ./run-autoresearch-rotate.sh                 # default rotation
#     CLIS="gemini,codex" ./run-autoresearch-rotate.sh   # custom order
#
# Telegram notifications fire from autorun.py itself; this wrapper
# only adds a rotation-event ping when it cycles CLI.

set -u

# vibe is excluded by default: its programmatic mode (-p) takes the
# prompt as argv, and our prompts (target_files + INDEX.md verbatim)
# blow past Linux ARG_MAX (~128 KB). Override with CLIS env if you
# want to test it with a smaller config.
CLIS_DEFAULT="claude,gemini,codex"
CLIS="${CLIS:-$CLIS_DEFAULT}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-1800}"   # 30 min nap between full rotations
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TG_HELPER="$HOME/.openclaw/workspace/outbound/telegram_send.py"

tg() {
    if [ -f "$TG_HELPER" ]; then
        python3 "$TG_HELPER" msg "$1" > /dev/null 2>&1 || true
    fi
}

tg "🔄 autoresearch rotation launcher armed — order: $CLIS"

IFS=',' read -ra CLI_ARR <<< "$CLIS"
ROUND=0

while :; do
    ROUND=$((ROUND + 1))
    any_kept_this_round=0

    for CLI in "${CLI_ARR[@]}"; do
        CLI="$(echo "$CLI" | xargs)"   # strip whitespace
        [ -z "$CLI" ] && continue

        echo "[rotate] round=$ROUND  cli=$CLI  $(date -Iseconds)"
        tg "▶️ autoresearch round=$ROUND · trying cli=$CLI"

        # Snapshot TSV size so we can detect whether any iterations
        # actually landed on this CLI's run.
        TSV="$HERE/autoresearch_results.tsv"
        before=$([ -f "$TSV" ] && wc -l < "$TSV" || echo 0)

        "$HERE/run-autoresearch.sh" "$CLI"
        rc=$?

        after=$([ -f "$TSV" ] && wc -l < "$TSV" || echo 0)
        delta=$((after - before))

        echo "[rotate] cli=$CLI exited rc=$rc  (TSV grew by $delta rows)"

        # Any keep row in the new rows?
        if [ "$delta" -gt 0 ] && tail -n "$delta" "$TSV" | awk -F'\t' '$4=="keep"{found=1} END{exit !found}'; then
            any_kept_this_round=1
        fi

        # rc=0 = clean shutdown (Ctrl-C). rc=2 = crash budget hit.
        # Anything else (1, sigkill, etc.) is unexpected — log and rotate.
        if [ "$rc" -eq 0 ]; then
            echo "[rotate] cli=$CLI exited cleanly. Stopping."
            tg "⏹️ autoresearch stopped cleanly (cli=$CLI, round=$ROUND)"
            exit 0
        fi

        tg "⚠️ cli=$CLI exhausted (rc=$rc, +$delta rows). Rotating."
    done

    if [ "$any_kept_this_round" -eq 0 ]; then
        echo "[rotate] full pass produced 0 kept iterations. Cooling down ${COOLDOWN_SECONDS}s."
        tg "😴 all CLIs blocked. Sleeping ${COOLDOWN_SECONDS}s before retrying."
        sleep "$COOLDOWN_SECONDS"
    else
        echo "[rotate] round $ROUND produced kept iterations. Looping immediately."
    fi
done
