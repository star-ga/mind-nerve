#!/usr/bin/env bash
# fake-mind-nerve.sh — Local test stub for the mind-nerve binary.
#
# Reads a JSON payload from stdin and echoes a canned top_k response.
# Use this to smoke-test the hook before the real binary is built.
#
# Usage:
#   MIND_NERVE_BIN=/path/to/fake-mind-nerve.sh node dist/preselect.js

set -euo pipefail

subcommand="${1:-}"
if [[ "$subcommand" != "preselect" ]]; then
  echo '{"outcome":"error","message":"unknown subcommand"}' >&2
  exit 1
fi

# Read and discard stdin (the hook always sends a full payload).
input="$(cat)"

# Extract k from the JSON payload using a simple regex (no jq dependency).
k="$(echo "$input" | grep -oP '"k"\s*:\s*\K[0-9]+' | head -1 || echo "5")"

# Build a fake top_k reply using the first k skill IDs from the registry.
ids="$(echo "$input" | grep -oP '"id"\s*:\s*"\K[^"]+' | head -"$k")"

selected_json="["
first=1
while IFS= read -r id; do
  [[ -z "$id" ]] && continue
  if [[ "$first" -eq 1 ]]; then
    first=0
  else
    selected_json+=","
  fi
  selected_json+="\"$id\""
done <<< "$ids"
selected_json+="]"

cat <<EOF
{"outcome":"top_k","selected":${selected_json},"scores":[0.95,0.90,0.85,0.80,0.75],"version":1}
EOF
