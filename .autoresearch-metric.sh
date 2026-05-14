#!/usr/bin/env bash
# autoresearch composite-metric extractor for mind-nerve.
# Chains arch-mind + skill-improver and emits a single composite_score
# line that autorun.py greps. Writes detailed gate output to run.log.
set -e

# --- Gate 1: arch-mind 5-language scan ---
/home/n/arch-mind/bin/arch-mind sidecar-scan \
  --repo . \
  --lang python,typescript,go,rust,mind \
  --out /tmp/_ar_scan.json > /tmp/_ar_run.log 2>&1

/home/n/arch-mind/bin/arch-mind scan \
  --fixture /tmp/_ar_scan.json \
  --out /tmp/_ar_scored.json >> /tmp/_ar_run.log 2>&1

ARCH=$(/home/n/arch-mind/bin/arch-mind explain --scan /tmp/_ar_scored.json \
  | grep q16_determinism_purity \
  | awk -F'≈' '{print $2}' | tr -d ' ' || echo 0)
[ -z "$ARCH" ] && ARCH=0

# --- Gate 2: skill-improver mean across mind-* agents ---
SKILL=$(/home/n/.local/bin/skill-improve report 2>/dev/null \
  | awk '/^mind-/ {sum += $4; n++} END {if (n>0) printf "%.2f", sum/n; else print "0"}')
[ -z "$SKILL" ] && SKILL=0

# --- Gate 3a: drafted RFC count (informational, no weight) ---
# Each RFC is a level-1 header (`# RFC-NNN — …`) inside RFCs/INDEX.md.
# Use `grep | wc -l` (not `grep -c`) so empty matches yield 0 cleanly
# without the `|| echo 0` fallback emitting a second line.
RFCS_DRAFTED=$(grep -E '^# RFC-' RFCs/INDEX.md 2>/dev/null | wc -l)

# --- Gate 3b: IMPLEMENTED RFC count (composite weight 5000 each) ---
# This is the IMPLEMENT-phase metric. An RFC is implemented when its
# section in RFCs/INDEX.md carries a `**Status:** IMPLEMENTED` line
# (added by the agent after landing code in src/). SKIPPED RFCs do not
# count — skipping is progress but not worth the same as shipping.
RFCS_IMPLEMENTED=$(grep -E '^\*\*Status:\*\*\s+IMPLEMENTED' RFCs/INDEX.md 2>/dev/null | wc -l)
RFCS_SKIPPED=$(grep -E '^\*\*Status:\*\*\s+SKIPPED' RFCs/INDEX.md 2>/dev/null | wc -l)
RFCS_QUEUED=$((RFCS_DRAFTED - RFCS_IMPLEMENTED - RFCS_SKIPPED))

# --- Gate 4: multi-LLM consensus on the latest diff ---
# Runs after every commit. If the iteration introduced a new
# `**Status:** IMPLEMENTED` marker, 3 fleet providers (grok/deepseek/
# mistral) review the diff against mind-nerve's six non-negotiables.
# Pass = ≥2 approves AND 0 rejects. Failure subtracts CONSENSUS_PENALTY
# from the composite — big enough to push the iteration below baseline
# so autorun.py discards it (git_reset_hard the commit).
#
# Output values mirror into run.log. If no IMPLEMENTED marker was
# added this iteration, consensus_skipped=1 fires and no penalty.
CONSENSUS_OUT=$(python3 .autoresearch-consensus.py 2>&1 || true)
CONSENSUS_SKIPPED=$(printf "%s\n" "$CONSENSUS_OUT" | awk '/^consensus_skipped:/ {print $2; exit}')
CONSENSUS_PASSED=$(printf "%s\n" "$CONSENSUS_OUT" | awk '/^consensus_passed:/ {print $2; exit}')
CONSENSUS_SCORE=$(printf "%s\n" "$CONSENSUS_OUT" | awk '/^consensus_score:/ {print $2; exit}')
[ -z "$CONSENSUS_SKIPPED" ] && CONSENSUS_SKIPPED=0
[ -z "$CONSENSUS_PASSED" ] && CONSENSUS_PASSED=0
[ -z "$CONSENSUS_SCORE" ] && CONSENSUS_SCORE=0

CONSENSUS_PENALTY=0
if [ "$CONSENSUS_SKIPPED" = "1" ]; then
    # No vote was taken — no penalty. (Baseline iter, SKIPPED-only
    # iter, or process change.)
    :
elif [ "$CONSENSUS_PASSED" = "1" ]; then
    # Voted approve. No penalty.
    :
else
    # Voted reject or unparseable. Push composite below baseline so
    # autorun discards the iteration.
    CONSENSUS_PENALTY=6000
fi

# --- Composite ---
# composite = arch_mind_purity
#           + skill_improver_mean*100
#           + rfcs_implemented*5000
#           + (rfcs_implemented + rfcs_skipped)*100   # rfcs_processed nudge
#           - consensus_penalty
#
# The +100/processed nudge ensures SKIPPED iterations beat the prior
# baseline by enough to be kept (otherwise the SKIPPED marker would
# get reverted on discard and the agent would loop on the same RFC).
# Big rfcs_implemented weight (5000) keeps shipping dominant.
COMPOSITE=$(python3 -c "
arch = float('${ARCH}')
skill = float('${SKILL}')
impl = float('${RFCS_IMPLEMENTED}')
skip = float('${RFCS_SKIPPED}')
pen = float('${CONSENSUS_PENALTY}')
val = arch + skill * 100 + impl * 5000 + (impl + skip) * 100 - pen
print(f'{val:.2f}')")

{
  echo "arch_mind_purity: ${ARCH}"
  echo "skill_improver_mean: ${SKILL}"
  echo "rfcs_drafted: ${RFCS_DRAFTED}"
  echo "rfcs_implemented: ${RFCS_IMPLEMENTED}"
  echo "rfcs_skipped: ${RFCS_SKIPPED}"
  echo "rfcs_queued: ${RFCS_QUEUED}"
  echo "consensus_skipped: ${CONSENSUS_SKIPPED}"
  echo "consensus_passed: ${CONSENSUS_PASSED}"
  echo "consensus_score: ${CONSENSUS_SCORE}"
  echo "consensus_penalty: ${CONSENSUS_PENALTY}"
  echo "autoresearch_composite: ${COMPOSITE}"
  echo "---"
  printf "%s\n" "$CONSENSUS_OUT" | sed -n 's/^#  //p'
} > run.log
