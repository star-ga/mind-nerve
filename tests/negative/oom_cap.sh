#!/bin/sh
# tests/negative/oom_cap.sh
#
# Allocation-bomb regression gate for the loader DoS caps
# (src/loader.mind MAX_CATALOG_BYTES / MAX_WEIGHTS_BYTES).
#
# A catalog file larger than MAX_CATALOG_BYTES (64 MiB) MUST be rejected as an
# IO error (CLI exit 7 = MN_EXIT_IO_ERROR) BEFORE the loader allocates a
# read buffer for it — i.e. with a bounded resident set, never an OOM. The
# file is a sparse 100 MiB truncate (no disk cost); without the size cap the
# loader would __mind_alloc(100 MiB) and grow RSS to ~100 MiB (or be
# OOM-killed under a tight limit).
#
# Usage:  sh tests/negative/oom_cap.sh [path-to-mind-nerve-cpu]
#
# Exit codes:
#   0  cap fired: binary exited 7 with bounded RSS
#   1  cap did NOT fire (wrong exit code or excessive RSS) — REGRESSION
#   2  environment/setup error

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BIN="${1:-${REPO_DIR}/mind-nerve-cpu}"
MODEL="${REPO_DIR}/tests/bit_identity/fixtures/model.weights"

if [ ! -x "$BIN" ]; then
    echo "oom_cap: binary not built/executable: $BIN" >&2
    exit 2
fi
if [ ! -f "$MODEL" ]; then
    echo "oom_cap: model fixture missing: $MODEL" >&2
    exit 2
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT INT TERM

HUGE_CAT="${WORK}/huge.cat"
# 100 MiB > MAX_CATALOG_BYTES (64 MiB); sparse, so it costs no real disk.
truncate -s 104857600 "$HUGE_CAT"

REQ="${WORK}/request.mic2"
{
    printf 'mic@2/mind-nerve/preselect\n'
    printf 'model: %s\n' "$MODEL"
    printf 'catalog: %s\n' "$HUGE_CAT"
    printf 'k: 5\n'
    printf 'tokens: 1,2,3\n'
    printf '.\n'
} > "$REQ"

# RSS ceiling: 60 MiB. Legit operation (before the oversize catalog is even
# read) touches only a 64 KiB stdin buffer + small tables; the cap must keep
# us far under the 100 MiB the un-capped path would allocate.
RSS_CEIL_KB=61440

TIME_OUT="${WORK}/time.txt"
set +e
/usr/bin/env timeout 30 /usr/bin/time -v "$BIN" < "$REQ" > /dev/null 2>"$TIME_OUT"
CODE=$?
set -e

MAXRSS_KB="$(awk -F': ' '/Maximum resident set size/ {print $2}' "$TIME_OUT")"

echo "oom_cap: exit=${CODE} max_rss_kb=${MAXRSS_KB:-unknown}"

if [ "$CODE" -ne 7 ]; then
    echo "oom_cap: FAIL — expected exit 7 (MN_EXIT_IO_ERROR), got ${CODE}" >&2
    cat "$TIME_OUT" >&2
    exit 1
fi

if [ -z "${MAXRSS_KB:-}" ]; then
    echo "oom_cap: FAIL — could not read Maximum resident set size" >&2
    exit 1
fi

if [ "$MAXRSS_KB" -gt "$RSS_CEIL_KB" ]; then
    echo "oom_cap: FAIL — RSS ${MAXRSS_KB} KiB exceeds ${RSS_CEIL_KB} KiB ceiling (cap did not fire before alloc)" >&2
    exit 1
fi

echo "oom_cap: PASS — oversize catalog rejected as IO error with bounded RSS (${MAXRSS_KB} KiB <= ${RSS_CEIL_KB} KiB)"
exit 0
