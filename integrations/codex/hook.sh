#!/usr/bin/env bash
# codex preselection hook.
#
# Reads the user prompt and catalog hash from environment variables set by
# codex's hook protocol, calls mind-nerve, writes the top-K skill IDs back as
# newline-separated stdout so codex can filter its skill listing.
#
# Fails open: any error causes the hook to print nothing and exit 0, which
# codex treats as "no preselection, load default skills".

set -u

MIND_NERVE_BIN="${MIND_NERVE_BIN:-mind-nerve}"
TOP_K="${MIND_NERVE_TOP_K:-5}"
TIMEOUT_SECS="${MIND_NERVE_TIMEOUT:-1}"

# Required hook variables from codex.
REQUEST="${CODEX_USER_PROMPT:-}"
CATALOG_HASH="${CODEX_SKILL_CATALOG_HASH:-}"

# Skip silently if either is unset.
if [ -z "$REQUEST" ] || [ -z "$CATALOG_HASH" ]; then
    exit 0
fi

# Skip silently if mind-nerve isn't installed.
if ! command -v "$MIND_NERVE_BIN" > /dev/null 2>&1; then
    exit 0
fi

# Build JSON payload without depending on jq.
payload=$(printf '{"request":%s,"catalog_hash":"%s","k":%s}' \
    "$(printf '%s' "$REQUEST" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')" \
    "$CATALOG_HASH" \
    "$TOP_K")

# Call mind-nerve with timeout, capture stdout.
response=$(printf '%s' "$payload" | timeout "$TIMEOUT_SECS" "$MIND_NERVE_BIN" preselect 2>/dev/null || echo "")

# Empty response means failure; fail open.
[ -z "$response" ] && exit 0

# Extract route IDs, one per line.
printf '%s' "$response" | python3 -c '
import sys, json
try:
    resp = json.loads(sys.stdin.read())
    for r in resp.get("routes", []):
        print(r["id"])
except Exception:
    pass
'
