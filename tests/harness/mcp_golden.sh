#!/bin/sh
# tests/harness/mcp_golden.sh — golden-transcript gate for the native `mcp`
# subcommand (src/mcp.mind), fail-closed. POSIX sh (bash/dash).
#
# WHAT THIS GATES
#   1. REGRESSION (frozen native golden). Replay tests/harness/mcp_session.jsonl
#      through the native binary and assert the response stream is byte-for-byte
#      equal to tests/harness/mcp_golden.native.txt. Any drift in the JSON-RPC
#      framing, tool schema, error codes, id-echo, or route payload trips here.
#      SECURITY REGRESSION GUARD: the LAST session line carries a ~4000-digit
#      numeric `id` (> MCP_ID_CAP, src/mcp.mind). Uncapped, mcp_put_id copied it
#      verbatim into the 16 KiB MCP_RESP_CAP scratch — a heap overflow reachable
#      on the first message. The frozen golden's last line MUST echo `"id":null`
#      (over-long id rejected at capture); a giant-id echo or a crash here means
#      the id cap regressed.
#   2. JSON WELL-FORMEDNESS. Every emitted line must parse as JSON.
#   3. PROTOCOL CONFORMANCE vs the Python server (python/mind_nerve/mcp_server.py).
#      The `initialize` frame MUST be byte-identical to the Python server's
#      compact output; the `notifications/initialized` notification MUST be
#      suppressed on both sides; error frames MUST carry the same {id, code}.
#      (Deliberate divergence, NOT gated for byte-identity: the native tool
#      ranks IN-PROCESS, so its tools/list description carries no daemon
#      language, tools/call returns route ids + Q16.16 scores rather than daemon
#      route metadata, and error `message` text is fixed — native has no dynamic
#      string interpolation of a method/tool name. Only the framing subset above
#      is byte-compared.)
#
# CROSS-SUBSTRATE (the wedge) — NOT run here; flagged for CI: the frozen native
# golden must be byte-identical when produced by the x86 (avx2) and ARM (neon)
# builds of the binary. Wire it into tests/bit_identity/ alongside the preselect
# fixtures once the ARM build of the single binary is produced in CI.
#
# USAGE:  BIN=./target/debug/mind-nerve sh tests/harness/mcp_golden.sh
set -eu

BIN="${BIN:-./target/debug/mind-nerve}"
HERE=$(dirname "$0")
SESSION="$HERE/mcp_session.jsonl"
GOLDEN="$HERE/mcp_golden.native.txt"

[ -x "$BIN" ] || { echo "FAIL: binary not found/executable: $BIN"; exit 1; }
[ -f "$SESSION" ] || { echo "FAIL: session fixture missing: $SESSION"; exit 1; }
[ -f "$GOLDEN" ] || { echo "FAIL: golden missing: $GOLDEN"; exit 1; }

OUT=$(mktemp)
trap 'rm -f "$OUT"' EXIT

# (1) regression vs frozen native golden — byte-for-byte.
"$BIN" < "$SESSION" > "$OUT" 2>/dev/null || { echo "FAIL: binary exited non-zero"; exit 1; }
if ! cmp -s "$OUT" "$GOLDEN"; then
    echo "FAIL: native output drifted from frozen golden"
    diff -u "$GOLDEN" "$OUT" || true
    exit 1
fi
echo "PASS leg 1: native output byte-identical to frozen golden"

# (2) JSON well-formedness + (3) protocol conformance vs Python.
python3 - "$OUT" <<'PY'
import json, sys
sys.path.insert(0, "python")
from mind_nerve import mcp_server as M

native = open(sys.argv[1]).read().splitlines()
for ln in native:
    json.loads(ln)                       # raises on malformed
print(f"PASS leg 2: all {len(native)} native lines are valid JSON")

def pyline(msg):
    r = M.handle(msg)
    return None if r is None else json.dumps(r, separators=(",", ":"))

# initialize: byte-identical framing.
init_msg = {"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
assert pyline(init_msg) == native[0], "initialize frame diverged from Python"
# notifications/initialized: suppressed on both sides.
assert pyline({"jsonrpc":"2.0","method":"notifications/initialized"}) is None
# error frames: same {id, code} (message text deliberately fixed on native side).
perr = json.loads(native[5]); assert perr["id"] == 6 and perr["error"]["code"] == -32601
pparse = json.loads(native[6]); assert pparse["id"] is None and pparse["error"]["code"] == -32700
print("PASS leg 3: initialize byte-identical; notification suppressed; error {id,code} match")
PY

echo "mcp_golden: ALL LEGS PASS"
