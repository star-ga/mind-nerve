#!/usr/bin/env bash
# Mass-clone agent-skill / claude-skill / mcp-server collection repos.
# Shallow clones (--depth 1) into catalog-data/sources/.

set -uo pipefail

ROOT=catalog-data
SOURCES=$ROOT/sources
MANIFEST=$ROOT/index/clone_manifest.tsv
LOG=$ROOT/index/clone_log.tsv

mkdir -p "$SOURCES"
echo -e "ts\tfull_name\toutcome\tlocal_dir\tsize_kb\tnotes" > "$LOG"

total=0
ok=0
fail=0
skipped=0

tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r stars size full desc; do
  total=$((total+1))
  ts=$(date -Iseconds)
  safe_name=$(echo "$full" | tr '/' '__')
  dest="$SOURCES/$safe_name"

  if [[ -d "$dest/.git" ]]; then
    echo -e "$ts\t$full\tSKIPPED\t$safe_name\t0\talready cloned" >> "$LOG"
    skipped=$((skipped+1))
    continue
  fi

  if git clone --depth 1 --quiet "https://github.com/${full}.git" "$dest" 2>>"$ROOT/index/clone_errors.log"; then
    actual_kb=$(du -sk "$dest" | awk '{print $1}')
    echo -e "$ts\t$full\tOK\t$safe_name\t$actual_kb\t" >> "$LOG"
    ok=$((ok+1))
  else
    echo -e "$ts\t$full\tFAIL\t$safe_name\t0\tsee clone_errors.log" >> "$LOG"
    fail=$((fail+1))
    rm -rf "$dest" 2>/dev/null
  fi
done

cloned_total_kb=$(awk -F'\t' '$3=="OK" {sum+=$5} END {print sum+0}' "$LOG")
echo ""
echo "=== clone summary ==="
echo "  attempted: $total"
echo "  ok       : $ok"
echo "  skipped  : $skipped"
echo "  failed   : $fail"
echo "  total kb : $cloned_total_kb"
