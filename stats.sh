#!/usr/bin/env bash
# stats.sh — one-shot snapshot of the three improvement loops.
#
# Composes:
#   - autoresearch iteration history (composite score over time)
#   - arch-mind structural metric snapshot (Q16 scores out of 10000)
#   - skill-improver agent ranking (mean/40 across the mind-* set)
#
# Usage:
#   ./stats.sh            # print snapshot
#   ./stats.sh --watch    # refresh every 30s
#   ./stats.sh --csv      # csv-friendly output for piping

set -e

CSV=0
WATCH=0
for arg in "$@"; do
    case "$arg" in
        --csv)    CSV=1 ;;
        --watch)  WATCH=1 ;;
    esac
done

render_once() {
    local now
    now=$(date '+%Y-%m-%d %H:%M:%S')

    if [ "$CSV" = "0" ]; then
        printf '\n========================================================================\n'
        printf '  mind-nerve improvement loops — snapshot @ %s\n' "$now"
        printf '========================================================================\n'
    fi

    # --- autoresearch ----------------------------------------------------
    if [ "$CSV" = "0" ]; then
        printf '\n[ autoresearch ] composite score per iteration\n'
        printf '  (composite = arch_purity + skill_mean*100 + rfcs*1000)\n\n'
    fi
    if [ -f autoresearch_results.tsv ]; then
        # Print without the description column; convert keep/discard into ✓/×
        if [ "$CSV" = "1" ]; then
            cat autoresearch_results.tsv
        else
            awk -F'\t' '
                NR == 1 { printf "  %-9s  %-12s  %-7s\n", "commit", "composite", "status"; next }
                {
                    mark = ($4 == "keep") ? "✓" : (($4 == "crash") ? "!" : "×")
                    printf "  %-9s  %-12.2f  %s %s\n", $1, $2, mark, $4
                }
            ' autoresearch_results.tsv | tail -25
            iters=$(($(wc -l < autoresearch_results.tsv) - 1))
            kept=$(awk -F'\t' '$4=="keep"{n++} END{print n+0}' autoresearch_results.tsv)
            printf '\n  %d iterations · %d kept · %d discarded/crash\n' "$iters" "$kept" "$((iters - kept))"
        fi
    else
        [ "$CSV" = "0" ] && printf '  (no autoresearch_results.tsv yet — loop not started)\n'
    fi

    # --- arch-mind --------------------------------------------------------
    if [ "$CSV" = "0" ]; then
        printf '\n[ arch-mind ] structural invariants (raw / 10000)\n\n'
    fi
    if [ -x /home/n/arch-mind/bin/arch-mind ]; then
        local tmp=/tmp/_stats_scan.json
        /home/n/arch-mind/bin/arch-mind sidecar-scan \
            --repo . --out "$tmp" >/dev/null 2>&1 || true
        /home/n/arch-mind/bin/arch-mind scan \
            --fixture "$tmp" --out "${tmp%.json}_scored.json" >/dev/null 2>&1 || true
        if [ "$CSV" = "1" ]; then
            /home/n/arch-mind/bin/arch-mind explain --scan "${tmp%.json}_scored.json" 2>/dev/null \
                | awk '/^  [a-z]/ {gsub(/≈/, "~"); printf "%s,%s\n", $1, $NF}'
        else
            /home/n/arch-mind/bin/arch-mind explain --scan "${tmp%.json}_scored.json" 2>/dev/null \
                | awk '/^  [a-z]/ {
                    name = $1
                    # $NF is "raw≈10000.00"; strip everything up to and
                    # including the ≈ (multibyte) by matching digits-dot-digits.
                    if (match($0, /[0-9]+\.[0-9]+$/)) {
                        val = substr($0, RSTART, RLENGTH) + 0
                    } else {
                        val = 0
                    }
                    pct = (val / 10000.0) * 100.0
                    bars = int(pct / 5.0)
                    if (bars > 20) bars = 20
                    bar = ""
                    for (i = 0; i < bars; i++) bar = bar "█"
                    for (i = bars; i < 20; i++) bar = bar "·"
                    printf "  %-30s %7.2f  %s\n", name, val, bar
                  }'
        fi
    else
        [ "$CSV" = "0" ] && printf '  (arch-mind binary not found)\n'
    fi

    # --- skill-improver ---------------------------------------------------
    if [ "$CSV" = "0" ]; then
        printf '\n[ skill-improver ] agent ranking (mean/40, fails out of 8 tests)\n\n'
    fi
    if [ -x /home/n/.local/bin/skill-improve ]; then
        if [ "$CSV" = "1" ]; then
            /home/n/.local/bin/skill-improve report 2>/dev/null | tail -n +2
        else
            /home/n/.local/bin/skill-improve report 2>/dev/null \
                | awk 'NR == 1 {
                    printf "  %-30s %7s  %6s\n", "skill", "mean/40", "fails"
                    next
                  }
                  {
                    name = $1
                    mean = $4
                    fails = $5
                    pct = (mean / 40.0) * 100.0
                    bars = int(pct / 5.0)
                    bar = ""
                    for (i = 0; i < bars; i++) bar = bar "█"
                    for (i = bars; i < 20; i++) bar = bar "·"
                    flag = (fails + 0 > 2) ? " ⚠" : ""
                    printf "  %-30s %7s  %6s  %s%s\n", name, mean, fails, bar, flag
                  }'
        fi
    else
        [ "$CSV" = "0" ] && printf '  (skill-improve binary not found)\n'
    fi

    # --- git activity -----------------------------------------------------
    if [ "$CSV" = "0" ]; then
        printf '\n[ git ] recent commits (main branch)\n\n'
        git log main --oneline -10 | awk '{printf "  %s\n", $0}'
        printf '\n[ git ] autoresearch branches active\n\n'
        git branch | grep autoresearch || printf '  (none)\n'
    fi

    [ "$CSV" = "0" ] && printf '\n========================================================================\n\n'
}

if [ "$WATCH" = "1" ]; then
    while true; do
        clear
        render_once
        sleep 30
    done
else
    render_once
fi
