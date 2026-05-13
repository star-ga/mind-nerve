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

# --- Gate 3: accepted RFC count ---
RFCS=$(ls research/RFC-*.md 2>/dev/null | wc -l)

# --- Composite ---
COMPOSITE=$(python3 -c "print(f'{float('${ARCH}') + float('${SKILL}') * 100 + float('${RFCS}') * 1000:.2f}')")

{
  echo "arch_mind_purity: ${ARCH}"
  echo "skill_improver_mean: ${SKILL}"
  echo "rfcs_accepted: ${RFCS}"
  echo "autoresearch_composite: ${COMPOSITE}"
} > run.log
