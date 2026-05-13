#!/usr/bin/env bash
# fake-mind-nerve.sh — Local test stub for the mind-nerve binary.
#
# Reads a mic@2 text frame from stdin, emits a mic-b binary response
# on stdout matching the wire spec in cli/main.mind §WIRE PROTOCOL.
#
# Fake behaviour:
#   - Parses `k:` from the mic@2 stdin frame.
#   - Parses the `catalog:` path and derives skill IDs from the catalog
#     hash (not available here), so uses fixed fake IDs: skill-a, skill-b, ...
#   - Emits a valid mic-b frame: MNB1 magic, k, k×RouteId, k×score, 212-byte
#     zero envelope.
#   - RouteId = SHA-256("skill-<letter>") for slots 0..k-1.
#   - Scores are fixed Q16.16 values: 0.95, 0.90, 0.85, 0.80, 0.75 (repeating).
#
# Usage:
#   MIND_NERVE_BIN=/path/to/fake-mind-nerve.sh node dist/preselect.js

set -euo pipefail

subcommand="${1:-}"
if [[ "$subcommand" != "preselect" ]]; then
  printf 'mic@2/mind-nerve/error\ncode: ParseError\ndetail: unknown subcommand\n.\n' >&2
  exit 6
fi

# Read stdin until terminator line "."
input=""
while IFS= read -r line; do
  input+="$line"$'\n'
  if [[ "$line" == "." ]]; then
    break
  fi
done

# Extract k from the mic@2 frame.
k="$(printf '%s\n' "$input" | grep -oP '^k:\s*\K[0-9]+' | head -1 || echo "5")"
if [[ -z "$k" ]]; then
  k=5
fi

# Fixed fake skill IDs for the test stub (letters a-z, wrapping).
letters=(a b c d e f g h i j k l m n o p q r s t u v w x y z)

# Fixed Q16.16 scores * 65536: 0.95→62259, 0.90→58982, 0.85→55705, 0.80→52428, 0.75→49152
q1616=(62259 58982 55705 52428 49152)

# Build the mic-b frame using Python (portable, handles binary output correctly).
python3 - <<PYEOF
import sys
import struct
import hashlib

k = $k
letters = list('abcdefghijklmnopqrstuvwxyz')
q1616 = [62259, 58982, 55705, 52428, 49152]

# magic
out = b'MNB1'

# k as u16 LE
out += struct.pack('<H', k)

# k * RouteId: SHA-256 of "skill-<letter>" (32 bytes each)
for i in range(k):
    name = 'skill-' + letters[i % len(letters)]
    out += hashlib.sha256(name.encode('utf-8')).digest()

# k * score: i32 LE Q16.16
for i in range(k):
    out += struct.pack('<i', q1616[i % len(q1616)])

# 212-byte attestation envelope (zeroes — tests don't verify chain integrity)
out += b'\x00' * 212

sys.stdout.buffer.write(out)
sys.stdout.buffer.flush()
PYEOF
