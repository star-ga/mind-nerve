#!/bin/sh
# tests/bit_identity/run.sh
#
# Cross-architecture bit-identity gate for mind-nerve.
#
# POSIX sh. Runs correctly under bash and dash.
#
# Purpose:
#   For every (request_fixture, catalog_fixture) pair listed in
#   fixtures/MANIFEST, run the mind-nerve binary under every detected backend,
#   capture stdout, strip the timestamp_ms slot (envelope bytes 8-15), compute
#   SHA-256 of the masked frame, and compare every backend's hash against:
#     (a) the committed golden hash in fixtures/expected/
#     (b) every other backend's hash for the same fixture pair
#
#   One byte of divergence exits non-zero.
#
# Usage:
#   bash tests/bit_identity/run.sh              -- normal gate mode
#   bash tests/bit_identity/run.sh --generate-golden  -- populate goldens
#   bash tests/bit_identity/run.sh --backend cpu      -- cpu only (skips pairwise)
#   bash tests/bit_identity/run.sh --backend cuda     -- cuda only
#
# Environment:
#   MIND_NERVE_CPU    path to x86/ARM cpu binary (default: ./mind-nerve-cpu)
#   MIND_NERVE_CUDA   path to cuda binary         (default: ./mind-nerve-cuda)
#   MIND_NERVE_ARM    path to ARM cpu binary       (default: ./mind-nerve-arm)
#   MIND_NERVE_MODEL  path to weights file         (default: fixtures/model.weights)
#   MIND_NERVE_TEST_INJECT_MS  fixed timestamp to pin clock (default: 1000000)
#
# Exit codes:
#   0   all backends agree with golden and pairwise
#   1   divergence detected
#   2   binary not built — CI fast-fail (expected at Phase 1.2)
#   3   usage error
#   4   fixture files missing

set -eu

# ---------------------------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FIXTURE_DIR="${SCRIPT_DIR}/fixtures"
EXPECTED_DIR="${FIXTURE_DIR}/expected"
MANIFEST="${FIXTURE_DIR}/MANIFEST"
WORK_DIR="/tmp/mind-nerve-bit-identity-$$"

MIND_NERVE_CPU="${MIND_NERVE_CPU:-${SCRIPT_DIR}/../../mind-nerve-cpu}"
MIND_NERVE_CUDA="${MIND_NERVE_CUDA:-${SCRIPT_DIR}/../../mind-nerve-cuda}"
MIND_NERVE_ARM="${MIND_NERVE_ARM:-${SCRIPT_DIR}/../../mind-nerve-arm}"
MIND_NERVE_MODEL="${MIND_NERVE_MODEL:-${FIXTURE_DIR}/model.weights}"
MIND_NERVE_TEST_INJECT_MS="${MIND_NERVE_TEST_INJECT_MS:-1000000}"

# Byte offsets in the mic-b output frame for the timestamp_ms field.
# mic-b layout:
#   [0:4]    magic "MNB1"
#   [4:6]    k (u16 LE)
#   [6:6+32*k]  k RouteIds
#   [6+32*k:6+36*k]  k scores (i32 LE)
#   [6+36*k:]  212-byte attestation envelope
#
# Envelope layout (little-endian, packed):
#   offset 0  version (1 byte)
#   offset 1  entry_kind (1 byte)
#   offset 2  wire_version (u16)
#   offset 4  k (u32)
#   offset 8  timestamp_ms (i64)  <-- 8 bytes to zero out
#   offset 16 architecture (u8)   <-- 1 byte to zero out (backend-specific)
#   offset 17 reserved (u8)
#   ...
#
# The harness zeros two envelope fields before hashing:
#   - timestamp_ms (8 bytes): intentionally non-identical across runs
#   - architecture (1 byte):  intentionally different per backend
#
# Zeroing these fields produces a deterministic preimage that MUST be
# identical across every backend if the bit-identity contract holds.

TIMESTAMP_OFFSET_IN_ENVELOPE=8   # bytes into envelope
TIMESTAMP_LEN=8                   # i64 = 8 bytes
ARCH_OFFSET_IN_ENVELOPE=16        # bytes into envelope
ARCH_LEN=1                        # u8 = 1 byte

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

GENERATE_GOLDEN=0
FORCE_BACKEND=""
SINGLE_BACKEND_MODE=0

for arg in "$@"; do
    case "$arg" in
        --generate-golden)
            GENERATE_GOLDEN=1
            ;;
        --backend)
            # next arg is backend name, handled below
            ;;
        cpu|cuda|arm)
            FORCE_BACKEND="$arg"
            SINGLE_BACKEND_MODE=1
            ;;
        --backend=*)
            FORCE_BACKEND="${arg#--backend=}"
            SINGLE_BACKEND_MODE=1
            ;;
        *)
            # Allow --backend <value> as two args
            ;;
    esac
done

# Re-parse --backend <value> as two-token form.
prev=""
for arg in "$@"; do
    if [ "$prev" = "--backend" ]; then
        FORCE_BACKEND="$arg"
        SINGLE_BACKEND_MODE=1
    fi
    prev="$arg"
done

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

log()  { printf '[bit-id] %s\n' "$*" >&2; }
pass() { printf '[bit-id] PASS  %s\n' "$*" >&2; }
fail() { printf '[bit-id] FAIL  %s\n' "$*" >&2; }
info() { printf '[bit-id] INFO  %s\n' "$*" >&2; }

FAILURES=0
PASSES=0

record_fail() {
    fail "$*"
    FAILURES=$(( FAILURES + 1 ))
}

record_pass() {
    pass "$*"
    PASSES=$(( PASSES + 1 ))
}

# ---------------------------------------------------------------------------
# Prerequisite: sha256sum or shasum
# ---------------------------------------------------------------------------

if command -v sha256sum >/dev/null 2>&1; then
    SHA256_CMD="sha256sum"
elif command -v shasum >/dev/null 2>&1; then
    SHA256_CMD="shasum -a 256"
else
    log "ERROR: neither sha256sum nor shasum found in PATH"
    exit 3
fi

sha256_of_file() {
    # Returns just the hex digest, no filename.
    $SHA256_CMD "$1" | awk '{print $1}'
}

sha256_of_stdin() {
    # Returns just the hex digest.
    $SHA256_CMD | awk '{print $1}'
}

# ---------------------------------------------------------------------------
# Python availability check (needed for timestamp masking helper)
# ---------------------------------------------------------------------------

if ! command -v python3 >/dev/null 2>&1; then
    log "ERROR: python3 not found; required for timestamp masking"
    exit 3
fi

# ---------------------------------------------------------------------------
# Fixture sanity check
# ---------------------------------------------------------------------------

if [ ! -f "$MANIFEST" ]; then
    log "ERROR: fixture manifest not found at ${MANIFEST}"
    log "       Run: python3 tests/bit_identity/gen_fixtures.py"
    exit 4
fi

for fixture_check in \
    "${FIXTURE_DIR}/request_001.mic2" \
    "${FIXTURE_DIR}/request_002.mic2" \
    "${FIXTURE_DIR}/request_003.mic2" \
    "${FIXTURE_DIR}/catalog_44.bin" \
    "${FIXTURE_DIR}/catalog_440.bin" \
    "${FIXTURE_DIR}/catalog_4400.bin"
do
    if [ ! -f "$fixture_check" ]; then
        log "ERROR: missing fixture: ${fixture_check}"
        log "       Run: python3 tests/bit_identity/gen_fixtures.py"
        exit 4
    fi
done

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

detect_backends() {
    detected=""

    if [ -n "$FORCE_BACKEND" ]; then
        # Honour explicit override even if binary doesn't exist (the binary
        # check later will produce the right fast-fail message).
        detected="$FORCE_BACKEND"
        echo "$detected"
        return
    fi

    # CPU: always in scope for Phase 1.
    detected="cpu"

    # CUDA: present iff nvidia-smi exits 0.
    if nvidia-smi >/dev/null 2>&1; then
        detected="${detected} cuda"
    else
        info "nvidia-smi not found or failed — CUDA backend skipped"
    fi

    # ARM: present iff running on aarch64.
    arch_name="$(uname -m)"
    if [ "$arch_name" = "aarch64" ]; then
        detected="${detected} arm"
    else
        info "uname -m = ${arch_name} — ARM backend skipped"
    fi

    echo "$detected"
}

DETECTED_BACKENDS="$(detect_backends)"
log "Detected backends: ${DETECTED_BACKENDS}"

# ---------------------------------------------------------------------------
# Binary presence check — fail fast with clear message if not built.
# ---------------------------------------------------------------------------

backend_binary() {
    case "$1" in
        cpu)  echo "$MIND_NERVE_CPU"  ;;
        cuda) echo "$MIND_NERVE_CUDA" ;;
        arm)  echo "$MIND_NERVE_ARM"  ;;
        *)    echo "" ;;
    esac
}

BINARY_MISSING=0
for backend in $DETECTED_BACKENDS; do
    bin="$(backend_binary "$backend")"
    if [ -z "$bin" ] || [ ! -x "$bin" ]; then
        log "BINARY NOT BUILT: ${backend} binary not found at: ${bin:-<unset>}"
        log "  Build the project first: mindc build --target ${backend}"
        BINARY_MISSING=1
    fi
done

if [ "$BINARY_MISSING" -ne 0 ]; then
    log "Bit-identity check SKIPPED — binary not built."
    log "This is expected at Phase 1.2 (toolchain bringup)."
    log "CI FAILS FAST here by design — fix: build the binary."
    exit 2
fi

# ---------------------------------------------------------------------------
# Timestamp masking helper — zero out timestamp_ms and architecture byte
# in the envelope so the SHA-256 preimage is backend-agnostic.
#
# The envelope starts at byte offset: 6 + 36*k
#   (4 magic + 2 k_u16 + 32*k route_ids + 4*k scores)
# Within the envelope:
#   timestamp_ms at offset 8, length 8
#   architecture at offset 16, length 1
#
# We parse k from bytes [4:6] of the frame (u16 LE).
# ---------------------------------------------------------------------------

mask_frame() {
    # mask_frame <input_file> <output_file>
    # Writes the masked frame to output_file.
    python3 - "$1" "$2" <<'PYEOF'
import sys, struct

inp = sys.argv[1]
out = sys.argv[2]

with open(inp, 'rb') as f:
    data = bytearray(f.read())

if len(data) < 6:
    print(f"mask_frame: frame too short ({len(data)} bytes)", file=sys.stderr)
    sys.exit(1)

# Verify magic.
if data[0:4] != b'MNB1':
    print(f"mask_frame: bad magic {data[0:4].hex()}", file=sys.stderr)
    sys.exit(1)

k = struct.unpack_from('<H', data, 4)[0]
envelope_start = 6 + 36 * k  # = 4_magic + 2_k + 32*k_routes + 4*k_scores

expected_total = 218 + 36 * k  # 6 + 36*k + 212
if len(data) != expected_total:
    print(
        f"mask_frame: wrong size: got {len(data)}, expected {expected_total} "
        f"(k={k})",
        file=sys.stderr,
    )
    sys.exit(1)

# Zero timestamp_ms (envelope offset 8, 8 bytes).
ts_start = envelope_start + 8
for i in range(8):
    data[ts_start + i] = 0

# Zero architecture byte (envelope offset 16, 1 byte).
arch_pos = envelope_start + 16
data[arch_pos] = 0

with open(out, 'wb') as f:
    f.write(data)
PYEOF
}

# ---------------------------------------------------------------------------
# Single inference: invoke binary, capture output, mask, hash.
# Returns the hex SHA-256 of the masked frame on stdout.
# Sets LAST_RUN_EXIT to the binary's exit code.
# ---------------------------------------------------------------------------

LAST_RUN_EXIT=0

run_inference() {
    # run_inference <backend> <request_mic2> <catalog_bin>
    backend="$1"
    request_file="$2"
    catalog_bin="$3"

    bin="$(backend_binary "$backend")"
    raw_out="${WORK_DIR}/raw_${backend}.bin"
    masked_out="${WORK_DIR}/masked_${backend}.bin"

    # Substitute placeholder paths in the mic@2 frame.
    # We produce a modified frame in WORK_DIR with real paths.
    request_patched="${WORK_DIR}/request_${backend}.mic2"
    sed \
        -e "s|__MIND_NERVE_MODEL__|${MIND_NERVE_MODEL}|g" \
        -e "s|__MIND_NERVE_CATALOG__|${catalog_bin}|g" \
        "$request_file" > "$request_patched"

    # Run the binary.
    MIND_NERVE_TEST_INJECT_MS="${MIND_NERVE_TEST_INJECT_MS}" \
        "$bin" < "$request_patched" > "$raw_out" 2>/dev/null
    LAST_RUN_EXIT=$?

    if [ "$LAST_RUN_EXIT" -ne 0 ]; then
        log "  binary exited ${LAST_RUN_EXIT} for backend=${backend}"
        echo "ERROR_EXIT_${LAST_RUN_EXIT}"
        return
    fi

    # Mask timestamp_ms and architecture byte.
    if ! mask_frame "$raw_out" "$masked_out"; then
        log "  mask_frame failed for backend=${backend}"
        echo "ERROR_MASK"
        return
    fi

    sha256_of_file "$masked_out"
}

# ---------------------------------------------------------------------------
# Main gate loop
# ---------------------------------------------------------------------------

mkdir -p "$WORK_DIR"
trap 'rm -rf "$WORK_DIR"' EXIT INT TERM

log "=== Bit-identity gate starting ==="
log "  Timestamp pin: MIND_NERVE_TEST_INJECT_MS=${MIND_NERVE_TEST_INJECT_MS}"
log "  Generate golden: ${GENERATE_GOLDEN}"
log "  Single backend mode: ${SINGLE_BACKEND_MODE}"

BACKENDS_LIST="$DETECTED_BACKENDS"

while IFS='|' read -r req_file cat_file golden_file || [ -n "$req_file" ]; do
    # Skip comments and blank lines from MANIFEST.
    case "$req_file" in
        '#'*|'') continue ;;
    esac

    req_path="${FIXTURE_DIR}/${req_file}"
    cat_path="${FIXTURE_DIR}/${cat_file}"
    golden_path="${EXPECTED_DIR}/${golden_file}"
    pair_label="${req_file}+${cat_file}"

    # Verify fixture files exist.
    if [ ! -f "$req_path" ]; then
        record_fail "${pair_label}: request fixture not found: ${req_path}"
        continue
    fi
    if [ ! -f "$cat_path" ]; then
        record_fail "${pair_label}: catalog fixture not found: ${cat_path}"
        continue
    fi

    log "--- Pair: ${pair_label} ---"

    # Run all backends, collect hashes.
    hashes=""
    backend_names=""
    all_ok=1

    for backend in $BACKENDS_LIST; do
        hash="$(run_inference "$backend" "$req_path" "$cat_path")"
        case "$hash" in
            ERROR_*)
                record_fail "${pair_label} backend=${backend}: inference error (${hash})"
                all_ok=0
                ;;
            *)
                log "  ${backend}: ${hash}"
                hashes="${hashes}${hash}:${backend} "
                backend_names="${backend_names}${backend} "
                ;;
        esac
    done

    # Skip comparison if any backend errored.
    if [ "$all_ok" -eq 0 ]; then
        continue
    fi

    # --generate-golden: write the cpu hash as the golden value.
    if [ "$GENERATE_GOLDEN" -eq 1 ]; then
        cpu_hash=""
        for item in $hashes; do
            bhash="${item%%:*}"
            bname="${item##*:}"
            if [ "$bname" = "cpu" ]; then
                cpu_hash="$bhash"
            fi
        done
        if [ -z "$cpu_hash" ]; then
            record_fail "${pair_label}: --generate-golden but no cpu hash available"
        else
            printf '%s\n' "$cpu_hash" > "$golden_path"
            info "${pair_label}: wrote golden ${cpu_hash}"
        fi
        continue
    fi

    # Load golden hash.
    if [ ! -f "$golden_path" ]; then
        record_fail "${pair_label}: golden file not found: ${golden_path}"
        continue
    fi
    golden_hash="$(head -1 "$golden_path" | tr -d '[:space:]')"
    case "$golden_hash" in
        PENDING*)
            record_fail "${pair_label}: golden hash is PENDING — run with --generate-golden first"
            continue
            ;;
        [0-9a-f][0-9a-f][0-9a-f][0-9a-f]*)
            : # looks like a hex hash, proceed
            ;;
        *)
            record_fail "${pair_label}: golden file malformed: ${golden_path}"
            continue
            ;;
    esac

    # Compare each backend against golden.
    for item in $hashes; do
        bhash="${item%%:*}"
        bname="${item##*:}"
        if [ "$bhash" != "$golden_hash" ]; then
            record_fail "${pair_label} backend=${bname}: DIVERGES FROM GOLDEN"
            printf '    expected: %s\n' "$golden_hash" >&2
            printf '    actual:   %s\n' "$bhash" >&2
        else
            record_pass "${pair_label} backend=${bname} vs golden: OK"
        fi
    done

    # Pairwise comparison — every backend against every other backend.
    # Single-backend mode skips this (explicit --backend flag).
    if [ "$SINGLE_BACKEND_MODE" -eq 0 ]; then
        hash_count=0
        for item_a in $hashes; do
            hash_count=$(( hash_count + 1 ))
        done
        if [ "$hash_count" -ge 2 ]; then
            n=0
            for item_a in $hashes; do
                n=$(( n + 1 ))
                m=0
                for item_b in $hashes; do
                    m=$(( m + 1 ))
                    if [ "$m" -le "$n" ]; then continue; fi
                    ha="${item_a%%:*}"
                    ba="${item_a##*:}"
                    hb="${item_b%%:*}"
                    bb="${item_b##*:}"
                    if [ "$ha" != "$hb" ]; then
                        record_fail "${pair_label} PAIRWISE: ${ba} vs ${bb}"
                        printf '    %s: %s\n' "$ba" "$ha" >&2
                        printf '    %s: %s\n' "$bb" "$hb" >&2
                    else
                        record_pass "${pair_label} pairwise ${ba} vs ${bb}: identical"
                    fi
                done
            done
        fi
    fi

done < "$MANIFEST"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

log "=== Summary ==="
log "  PASS: ${PASSES}"
log "  FAIL: ${FAILURES}"

if [ "$GENERATE_GOLDEN" -eq 1 ]; then
    log "Golden generation complete. Commit fixtures/expected/ to lock the hashes."
    exit 0
fi

if [ "$FAILURES" -ne 0 ]; then
    log "BIT-IDENTITY GATE: FAILED (${FAILURES} divergence(s))"
    log "Run: python3 tests/bit_identity/verify.py <frame.bin> for diagnostics."
    exit 1
fi

log "BIT-IDENTITY GATE: PASSED"
exit 0
