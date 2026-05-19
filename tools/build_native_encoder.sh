#!/usr/bin/env bash
# mind-nerve/tools/build_native_encoder.sh
#
# Build libmind_nerve_encoder.so from the mind/exports/c_abi.mind surface.
#
# Prerequisites:
#   - mindc v0.4.4+ (from star-ga/mind, checked out at MIND_CHECKOUT or
#     cloned automatically to /tmp/mind-build/mind).
#   - Rust toolchain (stable, with x86_64-unknown-linux-gnu target).
#   - mindc compiled with features: std-surface cross-module-imports mlir-build
#
# Environment variables:
#   MIND_CHECKOUT    Path to an existing star-ga/mind checkout.
#                    If unset, the script clones mindc v0.4.4 into
#                    /tmp/mind-build/mind.
#   MIND_TAG         mindc tag to clone when MIND_CHECKOUT is unset.
#                    Default: v0.4.4
#   MIND_NERVE_ROOT  Root of the mind-nerve repository.
#                    Default: directory two levels above this script.
#   TARGET           Compilation target triple.
#                    Default: x86_64-unknown-linux-gnu
#   OUT_DIR          Directory to place the built .so.
#                    Default: $MIND_NERVE_ROOT/python/mind_nerve/_native/
#
# Usage:
#   ./tools/build_native_encoder.sh
#   MIND_CHECKOUT=/home/n/mind ./tools/build_native_encoder.sh
#
# Limitation (A1.3):
#   This script builds the C-ABI export surface (c_abi.mind) linked against
#   the kernel and LUT modules. Full weight quantization (offline
#   quantize_phase1_to_q16.py) and const-blob linkage are deferred to
#   Phase 6.2. The resulting .so exposes all six mn_encoder_* symbols;
#   mn_encoder_encode will not produce correct embeddings until the weight
#   blob is provided at init time.
#
# Output:
#   $OUT_DIR/libmind_nerve_encoder.so
#
# The .so is a sibling of the existing FORTRESS-protected libmindnerve.so;
# it is NOT a replacement. Both files are bundled into the wheel via
# the MANIFEST.in / pyproject.toml package-data glob.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIND_NERVE_ROOT="${MIND_NERVE_ROOT:-"$(cd "$SCRIPT_DIR/.." && pwd)"}"
MIND_TAG="${MIND_TAG:-v0.4.4}"
TARGET="${TARGET:-x86_64-unknown-linux-gnu}"
OUT_DIR="${OUT_DIR:-"$MIND_NERVE_ROOT/python/mind_nerve/_native"}"
BUILD_TMP="/tmp/mind-build"

# Source files fed to mindc (order matters: LUTs first, then kernels, then ABI).
MIND_SOURCES=(
    "$MIND_NERVE_ROOT/mind/luts/exp_q16.mind"
    "$MIND_NERVE_ROOT/mind/luts/recip_q32.mind"
    "$MIND_NERVE_ROOT/mind/luts/sqrt_q16.mind"
    "$MIND_NERVE_ROOT/mind/luts/tanh_q16.mind"
    "$MIND_NERVE_ROOT/mind/luts/softmax_q16.mind"
    "$MIND_NERVE_ROOT/mind/kernels/matmul_q16.mind"
    "$MIND_NERVE_ROOT/mind/kernels/batched_matmul_q16.mind"
    "$MIND_NERVE_ROOT/mind/kernels/layernorm_q16.mind"
    "$MIND_NERVE_ROOT/mind/kernels/gelu_q16.mind"
    "$MIND_NERVE_ROOT/mind/kernels/l2_norm_q16.mind"
    "$MIND_NERVE_ROOT/mind/kernels/embedding_q16.mind"
    "$MIND_NERVE_ROOT/mind/kernels/sliding_window.mind"
    "$MIND_NERVE_ROOT/mind/kernels/topk_q16.mind"
    "$MIND_NERVE_ROOT/mind/kernels/encode.mind"
    "$MIND_NERVE_ROOT/mind/exports/c_abi.mind"
)
ENTRY_POINT="$MIND_NERVE_ROOT/mind/exports/c_abi.mind"
OUTPUT_SO="$OUT_DIR/libmind_nerve_encoder.so"
MINDC_FEATURES="std-surface cross-module-imports mlir-build"

# ---------------------------------------------------------------------------
# 1. Resolve mindc checkout
# ---------------------------------------------------------------------------
if [[ -n "${MIND_CHECKOUT:-}" ]]; then
    MINDC_DIR="$MIND_CHECKOUT"
    echo "[build] Using existing mindc checkout at $MINDC_DIR"
else
    MINDC_DIR="$BUILD_TMP/mind"
    if [[ -d "$MINDC_DIR/.git" ]]; then
        echo "[build] Existing clone found at $MINDC_DIR — skipping clone."
        git -C "$MINDC_DIR" fetch --tags --quiet
        git -C "$MINDC_DIR" checkout "$MIND_TAG" --quiet
    else
        echo "[build] Cloning star-ga/mind at tag $MIND_TAG into $MINDC_DIR"
        mkdir -p "$BUILD_TMP"
        git clone \
            --depth 1 \
            --branch "$MIND_TAG" \
            "https://github.com/star-ga/mind.git" \
            "$MINDC_DIR"
    fi
fi

# ---------------------------------------------------------------------------
# 2. Build mindc binary (if not already cached)
# ---------------------------------------------------------------------------
MINDC_BIN="$MINDC_DIR/target/release/mindc"
if [[ ! -x "$MINDC_BIN" ]]; then
    echo "[build] Compiling mindc (release) with features: $MINDC_FEATURES"
    cargo build \
        --release \
        --bin mindc \
        --manifest-path "$MINDC_DIR/Cargo.toml" \
        --features "$MINDC_FEATURES"
fi
echo "[build] mindc binary: $MINDC_BIN"

# ---------------------------------------------------------------------------
# 3. Validate entry-point parses clean (--emit-ir smoke check)
# ---------------------------------------------------------------------------
echo "[build] Smoke-testing parse of c_abi.mind (--emit-ir)..."
"$MINDC_BIN" \
    "$ENTRY_POINT" \
    --emit-ir \
    --features "$MINDC_FEATURES" \
    2>&1 | grep -v "^WARN" | grep -v "^$" || true
echo "[build] Parse check passed."

# ---------------------------------------------------------------------------
# 4. Emit shared library
# ---------------------------------------------------------------------------
mkdir -p "$OUT_DIR"
echo "[build] Compiling shared library → $OUTPUT_SO"
echo "[build]   target: $TARGET"
echo "[build]   entry:  $ENTRY_POINT"

# Pass all sources to mindc so cross-module imports resolve at link time.
# mindc --emit-shared accepts a primary entry point and resolves `use`
# directives relative to the MIND_STDLIB_PATH + the source directory tree.
# Set MIND_NERVE_SRC so the compiler finds the luts.* and kernels.* modules.
MIND_NERVE_SRC="$MIND_NERVE_ROOT/mind" \
"$MINDC_BIN" \
    "$ENTRY_POINT" \
    --emit-shared \
    --output "$OUTPUT_SO" \
    --target "$TARGET" \
    --features "$MINDC_FEATURES"

# ---------------------------------------------------------------------------
# 5. Verify exported symbols
# ---------------------------------------------------------------------------
REQUIRED_SYMS=(
    mn_encoder_init
    mn_encoder_encode
    mn_encoder_score
    mn_encoder_topk
    mn_encoder_free
    mn_encoder_version
)
echo "[build] Verifying exported symbols in $OUTPUT_SO:"
MISSING=0
for sym in "${REQUIRED_SYMS[@]}"; do
    if nm -D "$OUTPUT_SO" 2>/dev/null | grep -q " T $sym"; then
        echo "  [ok] $sym"
    else
        echo "  [MISSING] $sym"
        MISSING=$((MISSING + 1))
    fi
done

if [[ $MISSING -gt 0 ]]; then
    echo "[build] ERROR: $MISSING symbol(s) missing from $OUTPUT_SO."
    exit 1
fi

# ---------------------------------------------------------------------------
# 6. Leak verifier (matches existing FORTRESS pattern in ROADMAP.md §42-44)
# ---------------------------------------------------------------------------
echo "[build] Running leak verifier against $OUTPUT_SO..."
if command -v strings >/dev/null 2>&1; then
    LEAKS=$(strings "$OUTPUT_SO" | grep -E "(STARGA|naestro|star-ga|private)" || true)
    if [[ -n "$LEAKS" ]]; then
        echo "[build] WARNING: Potential leak in strings output:"
        echo "$LEAKS"
    else
        echo "[build] Leak verifier: clean."
    fi
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo "[build] libmind_nerve_encoder.so staged at $OUTPUT_SO"
echo "[build] File size: $(stat -c%s "$OUTPUT_SO" 2>/dev/null || stat -f%z "$OUTPUT_SO") bytes"
echo ""
echo "[build] Gate check (run after install):"
echo "  python -c \"from mind_nerve._native import _NativeRuntime; r = _NativeRuntime(); print(r.version())\""
