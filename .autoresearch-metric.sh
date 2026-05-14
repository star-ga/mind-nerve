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

# --- Composite ---
# composite = arch_mind_purity + skill_improver_mean*100 + rfcs_implemented*5000
COMPOSITE=$(python3 -c "print(f'{float('${ARCH}') + float('${SKILL}') * 100 + float('${RFCS_IMPLEMENTED}') * 5000:.2f}')")

{
  echo "arch_mind_purity: ${ARCH}"
  echo "skill_improver_mean: ${SKILL}"
  echo "rfcs_drafted: ${RFCS_DRAFTED}"
  echo "rfcs_implemented: ${RFCS_IMPLEMENTED}"
  echo "rfcs_skipped: ${RFCS_SKIPPED}"
  echo "rfcs_queued: ${RFCS_QUEUED}"
  echo "autoresearch_composite: ${COMPOSITE}"
} > run.log
