#!/bin/sh
# tests/mindc_gate.sh — fail-closed mindc 0.10.2 gate for the MIND tree.
#
# POSIX sh. Runs correctly under bash and dash.
#
# Replaces the v0.4.4-era CI loop (`mindc $f --emit-ir` per file), which
# passed vacuously: v0.4.4 single-file mode never name-checked fn bodies
# (mind#23), so unresolved cross-module calls were invisible. mindc 0.10.2
# checks fn bodies and resolves user-module imports only in PROJECT mode
# (whole-project module table seeded from mind/Mind.toml's explicit
# [targets.cpu].sources list).
#
# ANTI-SILENT-GREEN DISCIPLINE (audit 2026-08-10, findings D1/D2):
# `mindc build` and `mindc test` both exit 0 in degraded states:
#   - a module that fails to compile is embedded as a runtime-JIT fallback
#     with only a [WARN] line, and the build still exits 0;
#   - a failed native LINK emits a launcher-script fallback, exit 0;
#   - `mindc test` prints "running 0 tests / test result: ok" and exits 0
#     when discovery finds nothing (e.g. every file failed to parse).
# This script therefore NEVER trusts an exit code alone: it pattern-checks
# the output and asserts exact test counts. Any unrecognized line shape
# fails the gate.
#
# Legs:
#   1. Project-mode NATIVE build of all 19 CI-gated files in mind/
#      (`mindc build`). Requires ZERO error-severity diagnostics, ZERO
#      JIT/embedded fallback, ZERO launcher-link fallback, real native
#      objects for ALL 19 modules, and a REAL ELF linked artifact. The four
#      C-shim modules (kernels/matmul_blas.mind + luts/{rsqrt,softmax,tanh}
#      _q16.mind) now compile natively: their five __mind_nerve_* C-ABI
#      callees are registered in mindc's STD_SURFACE_INTRINSICS and DEFINED
#      at link time by the manifest [targets.cpu].native_sources C runtime
#      (runtime/blas_shims_i64.c + runtime/lut_cache.c). (Prior gate rigged:
#      it whitelisted exactly 4 E2024 fallbacks + a launcher and passed on a
#      degraded non-native artifact — the exact silent-green trap.)
#   2. LUT bit-identity smoke: merge mind/luts/*.mind + the generated
#      tests/luts_smoke.mind fixture + the #[test] wrappers in
#      tests/luts_smoke_test.mind into one self-contained module (mindc
#      test resolves only bundled std imports, never user modules) and
#      run it under `mindc test`. HARD FAIL unless exactly 5 tests run
#      and pass — "running 0 tests" is a failure, never a pass.
#   3. src/sha256.mind (wave-1 pure-MIND port): `mindc check` clean, then
#      merged src+test image under `mindc test` with exactly 16 tests
#      required to run and pass.
#   4. src/q16_16.mind (wave-1 pure-MIND port): merged image with the two
#      pinned LUT tables it delegates to (mind/luts/exp_q16.mind,
#      mind/luts/sqrt_q16.mind); `mindc check` clean on the src-merged
#      image, then `mindc test` on the full merged image with exactly 86
#      tests required to run and pass.
#   5. src/lib.mind (wave-1 pure-MIND port): `mindc check` clean, then
#      merged src+test image under `mindc test` with exactly 12 tests
#      required to run and pass.
#   6. src/top_k.mind (wave-2 pure-MIND port): `mindc check` clean on the
#      merged src image (src/sha256.mind + src/top_k.mind — same merged-image
#      discipline as leg 4, since top_k imports src.sha256), then the full
#      merged image (+ tests/unit/test_top_k.mind) under `mindc test` with
#      exactly 30 tests required to run and pass.
#   7. src/tokenizer.mind (wave-2 pure-MIND port): merged-image check
#      (src/lib.mind + src/sha256.mind + src/tokenizer.mind), then the full
#      merged image (+ tests/unit/test_tokenizer.mind) under `mindc test`
#      with exactly 18 tests required to run and pass.
#   8. src/chain_log.mind (wave-2 pure-MIND port): merged-image check
#      (src/lib.mind + src/sha256.mind + src/chain_log.mind), then the full
#      merged image (+ tests/unit/test_chain_log.mind) under `mindc test`
#      with exactly 5 tests required to run and pass.
#   9. src/runtime_ffi.mind (wave-3 pure-MIND port): merged-image check
#      (src/runtime_ffi.mind alone — the extern "C" block checks clean;
#      E2024 intrinsic-table WARNs on the __mind_nerve_rt_* symbols are the
#      documented link-time shim mechanism, same as leg 1), then the merged
#      image (+ tests/unit/test_runtime_ffi.mind) under `mindc test` with
#      exactly 3 tests required to run and pass. The extern bodies are NOT
#      executed by the unit gate (no host shim in the test interpreter) —
#      the legs gate the status-logic surface only.
#  10. src/clock.mind (wave-3 pure-MIND port): merged-image check
#      (src/runtime_ffi.mind + src/clock.mind), then the full merged image
#      (+ tests/unit/test_clock.mind) under `mindc test` with exactly 5
#      tests required to run and pass. The host-clock path is NOT executed
#      by the unit gate; the legs gate the pure ns→ms conversion, the
#      injection fixture, and the counter fallback's monotonicity.
#  11. src/evidence.mind (wave-4 port): merged-image check (src/lib.mind +
#      src/sha256.mind + src/evidence.mind), then + tests/unit/
#      test_evidence.mind under `mindc test`, exactly 36 tests.
#  12. src/encoder_kernels.mind (wave-4 port): merged-image check (pinned
#      luts + src/lib.mind + src/q16_16.mind + src/encoder_kernels.mind),
#      then + tests/unit/test_encoder_kernels.mind, exactly 17 tests.
#  13. src/model.mind (wave-4 port): same image + src/model.mind, then
#      + tests/unit/test_model.mind, exactly 8 tests.
#  14. src/loader.mind (wave-4 port): complete-stack merged image (all of
#      the above + src/runtime_ffi.mind + src/sha256.mind + src/loader.mind
#      — completeness is load-bearing: an unresolved reference inside a
#      CALLED function makes the 0.10.2 test interpreter report a
#      misleading "unknown variable" in the caller; see leg-14 comment),
#      then + tests/unit/test_loader.mind, exactly 12 tests.
#  15. src/inference.mind (wave-4 port): three parts. (a) `mindc check` on
#      the complete merged src image. (b) `mindc test` on
#      tests/unit/test_inference.mind, exactly 9 tests (gate surfaces plus
#      the three end-to-end oracle scenarios). (c) NATIVE run-harness:
#      tests/harness/inference_main.mind builds into a REAL ELF (with
#      tests/harness/rt_stub.mind standing in for the extern FFI layer —
#      the only harness substitution; see its header) and the gate executes
#      it: `file` must report an ELF and the run must exit 0 (every
#      non-zero exit code identifies a specific failed oracle check). The
#      native execution is the stronger evidence for the composed
#      pipeline; it also guards the class of fixture OOB that the
#      interpreter silently tolerates (repro:
#      mind-flow/target/repro-0.10.2/r11_oob_interpreter_corruption/).
#
# Usage:
#   bash tests/mindc_gate.sh
#
# Environment:
#   MINDC   path to the mindc 0.10.2 binary (default: `mindc` from PATH,
#           then ~/.local/bin/mindc)
#
# Exit codes:
#   0  all legs passed
#   1  gate failure (see output)
#   2  mindc 0.10.2 not found / wrong version

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_DIR="$(mktemp -d /tmp/mind-nerve-mindc-gate.XXXXXX)"
trap 'rm -rf "${WORK_DIR}"' EXIT

fail() {
    echo "GATE FAIL: $*" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Locate + version-pin mindc
# ---------------------------------------------------------------------------
if [ -n "${MINDC:-}" ]; then
    :
elif command -v mindc >/dev/null 2>&1; then
    MINDC="$(command -v mindc)"
elif [ -x "${HOME}/.local/bin/mindc" ]; then
    MINDC="${HOME}/.local/bin/mindc"
else
    echo "error: mindc not found. Install mindc 0.10.2:" >&2
    echo "  cargo install --git https://github.com/star-ga/mind --tag v0.10.2 \\" >&2
    echo "      --bin mindc --features std-surface,cross-module-imports,mlir-build" >&2
    exit 2
fi

VER="$("${MINDC}" --version 2>/dev/null | head -1)"
case "${VER}" in
    "mind 0.10.2"*) : ;;
    *)
        echo "error: gate is pinned to mindc 0.10.2; got: ${VER:-<unknown>}" >&2
        echo "  binary: ${MINDC}" >&2
        echo "  Re-pin deliberately: validate the tree against the new version," >&2
        echo "  then update this script, mind/Mind.toml, and ci.yml together." >&2
        exit 2
        ;;
esac
echo "mindc: ${MINDC} (${VER})"

# ---------------------------------------------------------------------------
# Leg 0 — manifest completeness: every .mind under mind/{luts,kernels,exports}
# must be listed in mind/Mind.toml's sources, and vice versa. A kernel added
# to the tree but not to the manifest would otherwise compile nothing and gate
# nothing — silently ungated (2026-08-10 codex audit).
# ---------------------------------------------------------------------------
echo "== Leg 0: manifest completeness (filesystem vs mind/Mind.toml)"
ON_DISK=$(find mind/luts mind/kernels mind/exports -name '*.mind' -type f | sort)
IN_MANIFEST=$(grep -oE '"[A-Za-z0-9_/]+\.mind"' mind/Mind.toml | tr -d '"' | sort -u)
# The manifest lists paths relative to mind/ (e.g. "luts/exp_q16.mind").
IN_MANIFEST_FULL=$(echo "$IN_MANIFEST" | sed 's|^|mind/|')
if [ "$ON_DISK" != "$IN_MANIFEST_FULL" ]; then
    echo "FAIL: mind/ tree and mind/Mind.toml sources disagree:" >&2
    diff <(echo "$ON_DISK") <(echo "$IN_MANIFEST_FULL") >&2
    echo "  Add the file to BOTH the tree and mind/Mind.toml [targets.cpu].sources." >&2
    exit 2
fi
COUNT=$(echo "$ON_DISK" | wc -l)
echo "   OK: ${COUNT} .mind files on disk == ${COUNT} manifest entries"

# ---------------------------------------------------------------------------
# Leg 1 — project-mode NATIVE build of the 19-file kernel tree, asserting a
# REAL ELF with ZERO JIT/launcher fallback.
#
# HARDENED (2026-08-16): the four C-shim modules (kernels/matmul_blas.mind +
# luts/{rsqrt,softmax,tanh}_q16.mind) now compile NATIVELY — their five
# __mind_nerve_* C-ABI callees are registered in mindc's STD_SURFACE_INTRINSICS
# (so the MLIR backend emits a plain `func.call @__mind_nerve_*` instead of the
# E2024 self-host-only advisory + runtime-JIT fallback) and are DEFINED at link
# time by the manifest `[targets.cpu].native_sources` C runtime
# (runtime/blas_shims_i64.c + runtime/lut_cache.c). There is therefore NO
# reason for any module to embed or any link to fall back.
#
# The PRIOR gate whitelisted exactly 4 E2024 fallbacks + a launcher fallback
# and reported GREEN on that degraded, non-native artifact — the exact
# silent-green trap this script exists to prevent (a public "compiled MIND end
# to end, no runtime, bit-identical bytes" claim must be LITERALLY true of the
# built artifact). Leg 1 now:
#   * FAILS on ANY "not natively compiled" / "runtime-JIT fallback" line;
#   * FAILS on ANY "native link failed" / launcher-script fallback;
#   * asserts the LINKED artifact is a real ELF (magic \x7fELF, `file` says ELF,
#     NOT a #! shell script, NO "mind-runtime" exec shim);
#   * asserts all 19 modules produced real native objects (pub fn symbols
#     present, ELF relocatable), keyed to the SAME module set registered above.
# mindc build exits 0 in degraded states, so NONE of these trusts the exit
# code — every check pattern-matches the log and inspects the artifact bytes.
# ---------------------------------------------------------------------------
echo "== Leg 1: project-mode NATIVE build (mind/ tree, 19 modules, real ELF)"

cd "${REPO_ROOT}/mind"
# Fresh object dir + artifact every run + bypass the incremental object cache:
# a stale target/obj (or a leftover launcher artifact from a degraded prior
# run) could otherwise mask a source/link regression.
rm -rf target/obj target/debug/mind-nerve-kernels
BUILD_LOG="${WORK_DIR}/build.log"
# Default (binary) emit → linked native executable at target/debug/<output>.
# The generous tool timeout covers the -O2 C-runtime compile of the 57 KiB
# blas_shims_i64.c on a cold machine.
MINDC_BUILD_TOOL_TIMEOUT_SECS="${MINDC_BUILD_TOOL_TIMEOUT_SECS:-900}" \
    "${MINDC}" build --no-cache >"${BUILD_LOG}" 2>&1 || \
    fail "mindc build exited non-zero (see below)
$(cat "${BUILD_LOG}")"

# Any error-severity mindc diagnostic fails the gate. (E2003 cross-module
# failures surface inside the [WARN] fallback parenthetical, so scanning
# the whole log — not just exit codes — is the load-bearing check.)
if grep -q 'error\[' "${BUILD_LOG}"; then
    fail "error-severity diagnostic in project build:
$(grep 'error\[' "${BUILD_LOG}" | head -20)"
fi

# ZERO-fallback policy. Any embedded/JIT fallback or launcher-link fallback is
# a HARD failure — the artifact would not be a real native binary. These are
# the exact silent-green shapes mindc emits while still exiting 0.
if grep -qE 'not natively compiled|runtime-JIT fallback' "${BUILD_LOG}"; then
    fail "JIT/embedded fallback in native build (expected ZERO):
$(grep -nE 'not natively compiled|runtime-JIT fallback' "${BUILD_LOG}")"
fi
if grep -q 'native link failed' "${BUILD_LOG}"; then
    fail "native link fell back to a launcher (expected a real ELF):
$(grep -n 'native link failed' "${BUILD_LOG}")"
fi
if grep -q '\[WARN\]' "${BUILD_LOG}"; then
    fail "unexpected [WARN] line in native build (expected none):
$(grep -n '\[WARN\]' "${BUILD_LOG}")"
fi

# Assert the LINKED artifact is a real native ELF executable, not a launcher.
ARTIFACT="target/debug/mind-nerve-kernels"
[ -f "${ARTIFACT}" ] || fail "no build artifact at ${ARTIFACT}"
# (1) `file` must report ELF.
if ! file "${ARTIFACT}" | grep -q 'ELF'; then
    fail "artifact is not an ELF: $(file "${ARTIFACT}")"
fi
# (2) First 4 bytes must be the ELF magic \x7fELF (0x7f 45 4c 46).
MAGIC="$(od -An -tx1 -N4 "${ARTIFACT}" | tr -d ' \n')"
[ "${MAGIC}" = "7f454c46" ] || \
    fail "artifact magic is ${MAGIC:-<empty>}, expected 7f454c46 (\\x7fELF) — not a native ELF"
# (3) It must NOT be a shell script and must carry no mind-runtime exec shim
#     (the launcher fallback is a #!/bin/bash script that execs mind-runtime).
if head -c 2 "${ARTIFACT}" | grep -q '#!'; then
    fail "artifact begins with a #! shebang — it is a launcher script, not an ELF"
fi
if grep -aq 'mind-runtime' "${ARTIFACT}"; then
    fail "artifact contains a 'mind-runtime' exec shim — degraded launcher, not a native binary"
fi
echo "   OK: linked artifact is a real native ELF (magic 7f454c46, no launcher/shim)"

# All 19 project modules must have produced REAL native objects (their pub fn
# symbols present), proving parse + type-check + IR lowering + native codegen.
# An embedded-fallback object would contain only mind_module_*_get_ir/get_source.
NATIVE_MODULES="exports_c_abi kernels_batched_matmul_q16 kernels_embedding_q16 kernels_encode kernels_gelu_q16 kernels_l2_norm_q16 kernels_layernorm_q16 kernels_matmul_blas kernels_matmul_q16 kernels_sliding_window kernels_topk_q16 kernels_wordpiece luts_exp_q16 luts_recip_q32 luts_rsqrt_q16 luts_softmax_q16 luts_sqrt_q16 luts_tanh_q16 luts_vocab_wp"
mod_count=0
for m in ${NATIVE_MODULES}; do
    obj="target/obj/${m}.o"
    [ -f "${obj}" ] || fail "missing object for module ${m}"
    file "${obj}" | grep -q 'ELF' || fail "object ${m}.o is not an ELF: $(file "${obj}")"
    if command -v nm >/dev/null 2>&1; then
        # Native-vs-embedded is decided by symbol IDENTITY, not count: an
        # embedded runtime-JIT fallback object carries ONLY the
        # `mind_module_<name>_get_ir` / `_get_source` marker pair and none of
        # the module's own code, whereas a natively-compiled object defines its
        # `pub fn` text symbols and NO get_ir/get_source marker. (A count
        # threshold is wrong: a real native module with a single `pub fn` — e.g.
        # luts/rsqrt_q16.mind — has just one `T` symbol.)
        if nm "${obj}" 2>/dev/null | grep -qE 'mind_module_.*_get_ir|mind_module_.*_get_source|get_ir$|get_source$'; then
            fail "module ${m} is an embedded runtime-JIT fallback (get_ir/get_source marker present), not native"
        fi
        tsyms="$(nm "${obj}" 2>/dev/null | grep -c ' [Tt] ' || true)"
        [ "${tsyms}" -ge 1 ] || \
            fail "module ${m} has no native text symbols (${tsyms}) — not natively compiled?"
    fi
    mod_count=$((mod_count + 1))
done
[ "${mod_count}" -eq 19 ] || fail "expected 19 native module objects, verified ${mod_count}"
echo "   OK: 19/19 modules compiled to real native objects; zero JIT/launcher fallback"

# ---------------------------------------------------------------------------
# Leg 2 — LUT bit-identity smoke via mindc test (merged self-contained image)
# ---------------------------------------------------------------------------
echo "== Leg 2: LUT smoke (mindc test, merged image)"

MERGED="${WORK_DIR}/luts_smoke_merged.mind"
# shellcheck disable=SC2035
cat "${REPO_ROOT}/mind/luts/exp_q16.mind" \
    "${REPO_ROOT}/mind/luts/recip_q32.mind" \
    "${REPO_ROOT}/mind/luts/sqrt_q16.mind" \
    "${REPO_ROOT}/mind/luts/tanh_q16.mind" \
    "${REPO_ROOT}/mind/luts/softmax_q16.mind" \
    "${REPO_ROOT}/mind/luts/rsqrt_q16.mind" \
    "${REPO_ROOT}/tests/luts_smoke.mind" \
    "${REPO_ROOT}/tests/luts_smoke_test.mind" > "${MERGED}"

TEST_LOG="${WORK_DIR}/test.log"
rc=0
"${MINDC}" test "${MERGED}" >"${TEST_LOG}" 2>&1 || rc=$?

# ANTI-SILENT-GREEN: `mindc test` exits 0 with "running 0 tests" when
# discovery finds nothing (e.g. a parse failure). That is a HARD failure
# here: the gate requires exactly 5 executed tests.
if grep -q "running 0 tests" "${TEST_LOG}"; then
    fail "mindc test discovered 0 tests (silent-green trap). Output:
$(cat "${TEST_LOG}")"
fi
grep -q "running 5 tests" "${TEST_LOG}" || \
    fail "expected 'running 5 tests'. Output:
$(cat "${TEST_LOG}")"
grep -q "5 passed; 0 failed" "${TEST_LOG}" || \
    fail "expected '5 passed; 0 failed'. Output:
$(cat "${TEST_LOG}")"
[ "${rc}" -eq 0 ] || fail "mindc test exited ${rc}. Output:
$(cat "${TEST_LOG}")"
echo "   OK: 5/5 LUT smoke tests executed and passed (tables + lookup arithmetic verified)"

# ---------------------------------------------------------------------------
# Leg 3 — src/sha256.mind (wave-1 port): compile gate + executing tests
#
# (a) `mindc check --no-fmt --no-lint` must be clean (0 error-severity
#     diagnostics). sha256.mind imports only bundled std (std.sha256), so
#     single-file check is the correct per-file gate; project-mode wiring of
#     the full src/ tree arrives when the dormant backlog is fully ported.
# (b/c) merged self-contained image (src/sha256.mind + its test file) run
#     under `mindc test`. HARD FAIL unless exactly 16 tests run and pass —
#     "running 0 tests" is a failure, never a pass.
# ---------------------------------------------------------------------------
echo "== Leg 3: src/sha256.mind (mindc check + mindc test, 16 tests)"

CHECK_LOG="${WORK_DIR}/check_sha256.log"
"${MINDC}" check --no-fmt --no-lint "${REPO_ROOT}/src/sha256.mind" >"${CHECK_LOG}" 2>&1 || \
    fail "mindc check failed for src/sha256.mind:
$(cat "${CHECK_LOG}")"
if grep -q 'error\[' "${CHECK_LOG}"; then
    fail "error-severity diagnostic for src/sha256.mind:
$(grep 'error\[' "${CHECK_LOG}" | head -20)"
fi

SHA256_MERGED="${WORK_DIR}/sha256_merged.mind"
cat "${REPO_ROOT}/src/sha256.mind" \
    "${REPO_ROOT}/tests/unit/test_sha256.mind" > "${SHA256_MERGED}"

SHA256_TEST_LOG="${WORK_DIR}/test_sha256.log"
rc=0
"${MINDC}" test "${SHA256_MERGED}" >"${SHA256_TEST_LOG}" 2>&1 || rc=$?

if grep -q "running 0 tests" "${SHA256_TEST_LOG}"; then
    fail "mindc test discovered 0 tests for sha256 (silent-green trap). Output:
$(cat "${SHA256_TEST_LOG}")"
fi
grep -q "running 16 tests" "${SHA256_TEST_LOG}" || \
    fail "expected 'running 16 tests' for sha256. Output:
$(cat "${SHA256_TEST_LOG}")"
grep -q "16 passed; 0 failed" "${SHA256_TEST_LOG}" || \
    fail "expected '16 passed; 0 failed' for sha256. Output:
$(cat "${SHA256_TEST_LOG}")"
[ "${rc}" -eq 0 ] || fail "mindc test (sha256) exited ${rc}. Output:
$(cat "${SHA256_TEST_LOG}")"
echo "   OK: src/sha256.mind checks clean; 16/16 FIPS KAT + fixture tests executed and passed"

# ---------------------------------------------------------------------------
# Leg 4 — src/q16_16.mind (wave-1 port): merged-image check + executing tests
#
# q16_16 delegates all table lookups to the pinned LUTs (operator decision:
# delegate, never duplicate), so the merged image prepends the two pinned
# table modules. Single-file check of src/q16_16.mind alone cannot resolve
# user-module imports (mindc resolves those only in project mode; the full
# src/ project build arrives when the dormant backlog is fully ported), so
# the compile gate runs on the merged src image — same discipline as Leg 2.
# HARD FAIL unless exactly 86 tests run and pass.
# ---------------------------------------------------------------------------
echo "== Leg 4: src/q16_16.mind (merged check + mindc test, 86 tests)"

Q16_SRC_MERGED="${WORK_DIR}/q16_src_merged.mind"
cat "${REPO_ROOT}/mind/luts/exp_q16.mind" \
    "${REPO_ROOT}/mind/luts/sqrt_q16.mind" \
    "${REPO_ROOT}/src/q16_16.mind" > "${Q16_SRC_MERGED}"

Q16_CHECK_LOG="${WORK_DIR}/check_q16.log"
"${MINDC}" check --no-fmt --no-lint "${Q16_SRC_MERGED}" >"${Q16_CHECK_LOG}" 2>&1 || \
    fail "mindc check failed for src/q16_16.mind merged image:
$(cat "${Q16_CHECK_LOG}")"
if grep -q 'error\[' "${Q16_CHECK_LOG}"; then
    fail "error-severity diagnostic for src/q16_16.mind merged image:
$(grep 'error\[' "${Q16_CHECK_LOG}" | head -20)"
fi

Q16_MERGED="${WORK_DIR}/q16_merged.mind"
cat "${Q16_SRC_MERGED}" \
    "${REPO_ROOT}/tests/unit/test_q16_16.mind" > "${Q16_MERGED}"

Q16_TEST_LOG="${WORK_DIR}/test_q16.log"
rc=0
"${MINDC}" test "${Q16_MERGED}" >"${Q16_TEST_LOG}" 2>&1 || rc=$?

if grep -q "running 0 tests" "${Q16_TEST_LOG}"; then
    fail "mindc test discovered 0 tests for q16_16 (silent-green trap). Output:
$(cat "${Q16_TEST_LOG}")"
fi
grep -q "running 86 tests" "${Q16_TEST_LOG}" || \
    fail "expected 'running 86 tests' for q16_16. Output:
$(cat "${Q16_TEST_LOG}")"
grep -q "86 passed; 0 failed" "${Q16_TEST_LOG}" || \
    fail "expected '86 passed; 0 failed' for q16_16. Output:
$(cat "${Q16_TEST_LOG}")"
[ "${rc}" -eq 0 ] || fail "mindc test (q16_16) exited ${rc}. Output:
$(cat "${Q16_TEST_LOG}")"
echo "   OK: src/q16_16.mind merged image checks clean; 86/86 pinned-LUT + arithmetic tests executed and passed"

# ---------------------------------------------------------------------------
# Leg 5 — src/lib.mind (wave-1 port): compile gate + executing tests
#
# lib.mind imports nothing (pure const hub), so single-file
# `mindc check --no-fmt --no-lint` is the complete per-file compile gate.
# HARD FAIL unless exactly 12 tests run and pass.
# ---------------------------------------------------------------------------
echo "== Leg 5: src/lib.mind (mindc check + mindc test, 12 tests)"

LIB_CHECK_LOG="${WORK_DIR}/check_lib.log"
"${MINDC}" check --no-fmt --no-lint "${REPO_ROOT}/src/lib.mind" >"${LIB_CHECK_LOG}" 2>&1 || \
    fail "mindc check failed for src/lib.mind:
$(cat "${LIB_CHECK_LOG}")"
if grep -q 'error\[' "${LIB_CHECK_LOG}"; then
    fail "error-severity diagnostic for src/lib.mind:
$(grep 'error\[' "${LIB_CHECK_LOG}" | head -20)"
fi

LIB_MERGED="${WORK_DIR}/lib_merged.mind"
cat "${REPO_ROOT}/src/lib.mind" \
    "${REPO_ROOT}/tests/unit/test_lib.mind" > "${LIB_MERGED}"

LIB_TEST_LOG="${WORK_DIR}/test_lib.log"
rc=0
"${MINDC}" test "${LIB_MERGED}" >"${LIB_TEST_LOG}" 2>&1 || rc=$?

if grep -q "running 0 tests" "${LIB_TEST_LOG}"; then
    fail "mindc test discovered 0 tests for lib (silent-green trap). Output:
$(cat "${LIB_TEST_LOG}")"
fi
grep -q "running 12 tests" "${LIB_TEST_LOG}" || \
    fail "expected 'running 12 tests' for lib. Output:
$(cat "${LIB_TEST_LOG}")"
grep -q "12 passed; 0 failed" "${LIB_TEST_LOG}" || \
    fail "expected '12 passed; 0 failed' for lib. Output:
$(cat "${LIB_TEST_LOG}")"
[ "${rc}" -eq 0 ] || fail "mindc test (lib) exited ${rc}. Output:
$(cat "${LIB_TEST_LOG}")"
echo "   OK: src/lib.mind checks clean; 12/12 model_hash-binding const pins executed and passed"

# ---------------------------------------------------------------------------
# Leg 6 — src/top_k.mind (wave-2 port): merged-image check + executing tests
#
# top_k imports src.sha256 (mn_sha256), which single-file check cannot
# resolve, so the compile gate runs on the merged src image — same
# discipline as leg 4. HARD FAIL unless exactly 30 tests run and pass.
# ---------------------------------------------------------------------------
echo "== Leg 6: src/top_k.mind (merged check + mindc test, 30 tests)"

TOPK_SRC_MERGED="${WORK_DIR}/topk_src_merged.mind"
cat "${REPO_ROOT}/src/sha256.mind" \
    "${REPO_ROOT}/src/top_k.mind" > "${TOPK_SRC_MERGED}"

TOPK_CHECK_LOG="${WORK_DIR}/check_top_k.log"
"${MINDC}" check --no-fmt --no-lint "${TOPK_SRC_MERGED}" >"${TOPK_CHECK_LOG}" 2>&1 || \
    fail "mindc check failed for src/top_k.mind merged image:
$(cat "${TOPK_CHECK_LOG}")"
if grep -q 'error\[' "${TOPK_CHECK_LOG}"; then
    fail "error-severity diagnostic for src/top_k.mind merged image:
$(grep 'error\[' "${TOPK_CHECK_LOG}" | head -20)"
fi

TOPK_MERGED="${WORK_DIR}/topk_merged.mind"
cat "${TOPK_SRC_MERGED}" \
    "${REPO_ROOT}/tests/unit/test_top_k.mind" > "${TOPK_MERGED}"

TOPK_TEST_LOG="${WORK_DIR}/test_top_k.log"
rc=0
"${MINDC}" test "${TOPK_MERGED}" >"${TOPK_TEST_LOG}" 2>&1 || rc=$?

if grep -q "running 0 tests" "${TOPK_TEST_LOG}"; then
    fail "mindc test discovered 0 tests for top_k (silent-green trap). Output:
$(cat "${TOPK_TEST_LOG}")"
fi
grep -q "running 30 tests" "${TOPK_TEST_LOG}" || \
    fail "expected 'running 30 tests' for top_k. Output:
$(cat "${TOPK_TEST_LOG}")"
grep -q "30 passed; 0 failed" "${TOPK_TEST_LOG}" || \
    fail "expected '30 passed; 0 failed' for top_k. Output:
$(cat "${TOPK_TEST_LOG}")"
[ "${rc}" -eq 0 ] || fail "mindc test (top_k) exited ${rc}. Output:
$(cat "${TOPK_TEST_LOG}")"
echo "   OK: src/top_k.mind merged image checks clean; 30/30 determinism + tie-break + mask + catalog tests executed and passed"

# ---------------------------------------------------------------------------
# Leg 7 — src/tokenizer.mind (wave-2 port): merged-image check + executing
# tests. tokenizer imports src.lib (MAX_REQUEST_TOKENS) and src.sha256
# (mn_sha256), so the compile gate runs on the merged src image — same
# discipline as legs 4/6. HARD FAIL unless exactly 18 tests run and pass.
# ---------------------------------------------------------------------------
echo "== Leg 7: src/tokenizer.mind (merged check + mindc test, 18 tests)"

TOK_SRC_MERGED="${WORK_DIR}/tok_src_merged.mind"
cat "${REPO_ROOT}/src/lib.mind" \
    "${REPO_ROOT}/src/sha256.mind" \
    "${REPO_ROOT}/src/tokenizer.mind" > "${TOK_SRC_MERGED}"

TOK_CHECK_LOG="${WORK_DIR}/check_tokenizer.log"
"${MINDC}" check --no-fmt --no-lint "${TOK_SRC_MERGED}" >"${TOK_CHECK_LOG}" 2>&1 || \
    fail "mindc check failed for src/tokenizer.mind merged image:
$(cat "${TOK_CHECK_LOG}")"
if grep -q 'error\[' "${TOK_CHECK_LOG}"; then
    fail "error-severity diagnostic for src/tokenizer.mind merged image:
$(grep 'error\[' "${TOK_CHECK_LOG}" | head -20)"
fi

TOK_MERGED="${WORK_DIR}/tok_merged.mind"
cat "${TOK_SRC_MERGED}" \
    "${REPO_ROOT}/tests/unit/test_tokenizer.mind" > "${TOK_MERGED}"

TOK_TEST_LOG="${WORK_DIR}/test_tokenizer.log"
rc=0
"${MINDC}" test "${TOK_MERGED}" >"${TOK_TEST_LOG}" 2>&1 || rc=$?

if grep -q "running 0 tests" "${TOK_TEST_LOG}"; then
    fail "mindc test discovered 0 tests for tokenizer (silent-green trap). Output:
$(cat "${TOK_TEST_LOG}")"
fi
grep -q "running 18 tests" "${TOK_TEST_LOG}" || \
    fail "expected 'running 18 tests' for tokenizer. Output:
$(cat "${TOK_TEST_LOG}")"
grep -q "18 passed; 0 failed" "${TOK_TEST_LOG}" || \
    fail "expected '18 passed; 0 failed' for tokenizer. Output:
$(cat "${TOK_TEST_LOG}")"
[ "${rc}" -eq 0 ] || fail "mindc test (tokenizer) exited ${rc}. Output:
$(cat "${TOK_TEST_LOG}")"
echo "   OK: src/tokenizer.mind merged image checks clean; 18/18 encode + error-surface + hash-oracle tests executed and passed"

# ---------------------------------------------------------------------------
# Leg 8 — src/chain_log.mind (wave-2 port): merged-image check + executing
# tests. chain_log imports src.lib and src.sha256, so the compile gate runs
# on the merged src image — same discipline as legs 4/6/7. HARD FAIL unless
# exactly 5 tests run and pass.
# ---------------------------------------------------------------------------
echo "== Leg 8: src/chain_log.mind (merged check + mindc test, 5 tests)"

CL_SRC_MERGED="${WORK_DIR}/cl_src_merged.mind"
cat "${REPO_ROOT}/src/lib.mind" \
    "${REPO_ROOT}/src/sha256.mind" \
    "${REPO_ROOT}/src/chain_log.mind" > "${CL_SRC_MERGED}"

CL_CHECK_LOG="${WORK_DIR}/check_chain_log.log"
"${MINDC}" check --no-fmt --no-lint "${CL_SRC_MERGED}" >"${CL_CHECK_LOG}" 2>&1 || \
    fail "mindc check failed for src/chain_log.mind merged image:
$(cat "${CL_CHECK_LOG}")"
if grep -q 'error\[' "${CL_CHECK_LOG}"; then
    fail "error-severity diagnostic for src/chain_log.mind merged image:
$(grep 'error\[' "${CL_CHECK_LOG}" | head -20)"
fi

CL_MERGED="${WORK_DIR}/cl_merged.mind"
cat "${CL_SRC_MERGED}" \
    "${REPO_ROOT}/tests/unit/test_chain_log.mind" > "${CL_MERGED}"

CL_TEST_LOG="${WORK_DIR}/test_chain_log.log"
rc=0
"${MINDC}" test "${CL_MERGED}" >"${CL_TEST_LOG}" 2>&1 || rc=$?

if grep -q "running 0 tests" "${CL_TEST_LOG}"; then
    fail "mindc test discovered 0 tests for chain_log (silent-green trap). Output:
$(cat "${CL_TEST_LOG}")"
fi
grep -q "running 5 tests" "${CL_TEST_LOG}" || \
    fail "expected 'running 5 tests' for chain_log. Output:
$(cat "${CL_TEST_LOG}")"
grep -q "5 passed; 0 failed" "${CL_TEST_LOG}" || \
    fail "expected '5 passed; 0 failed' for chain_log. Output:
$(cat "${CL_TEST_LOG}")"
[ "${rc}" -eq 0 ] || fail "mindc test (chain_log) exited ${rc}. Output:
$(cat "${CL_TEST_LOG}")"
echo "   OK: src/chain_log.mind merged image checks clean; 5/5 chain advancement + session-id tests executed and passed"

# ---------------------------------------------------------------------------
# Leg 9 — src/runtime_ffi.mind (wave-3 port): merged-image check + executing
# tests. The extern "C" declarations check clean (E2024 warns on the
# __mind_nerve_rt_* symbols are the documented link-time shim mechanism —
# same as the four shim modules in leg 1). HARD FAIL unless exactly 3 tests
# run and pass.
# ---------------------------------------------------------------------------
echo "== Leg 9: src/runtime_ffi.mind (merged check + mindc test, 3 tests)"

RFFI_SRC_MERGED="${WORK_DIR}/rffi_src_merged.mind"
cat "${REPO_ROOT}/src/runtime_ffi.mind" > "${RFFI_SRC_MERGED}"

RFFI_CHECK_LOG="${WORK_DIR}/check_runtime_ffi.log"
"${MINDC}" check --no-fmt --no-lint "${RFFI_SRC_MERGED}" >"${RFFI_CHECK_LOG}" 2>&1 || \
    fail "mindc check failed for src/runtime_ffi.mind merged image:
$(cat "${RFFI_CHECK_LOG}")"
if grep -q 'error\[' "${RFFI_CHECK_LOG}"; then
    fail "error-severity diagnostic for src/runtime_ffi.mind merged image:
$(grep 'error\[' "${RFFI_CHECK_LOG}" | head -20)"
fi

RFFI_MERGED="${WORK_DIR}/rffi_merged.mind"
cat "${RFFI_SRC_MERGED}" \
    "${REPO_ROOT}/tests/unit/test_runtime_ffi.mind" > "${RFFI_MERGED}"

RFFI_TEST_LOG="${WORK_DIR}/test_runtime_ffi.log"
rc=0
"${MINDC}" test "${RFFI_MERGED}" >"${RFFI_TEST_LOG}" 2>&1 || rc=$?

if grep -q "running 0 tests" "${RFFI_TEST_LOG}"; then
    fail "mindc test discovered 0 tests for runtime_ffi (silent-green trap). Output:
$(cat "${RFFI_TEST_LOG}")"
fi
grep -q "running 3 tests" "${RFFI_TEST_LOG}" || \
    fail "expected 'running 3 tests' for runtime_ffi. Output:
$(cat "${RFFI_TEST_LOG}")"
grep -q "3 passed; 0 failed" "${RFFI_TEST_LOG}" || \
    fail "expected '3 passed; 0 failed' for runtime_ffi. Output:
$(cat "${RFFI_TEST_LOG}")"
[ "${rc}" -eq 0 ] || fail "mindc test (runtime_ffi) exited ${rc}. Output:
$(cat "${RFFI_TEST_LOG}")"
echo "   OK: src/runtime_ffi.mind merged image checks clean; 3/3 status-code + classification tests executed and passed"

# ---------------------------------------------------------------------------
# Leg 10 — src/clock.mind (wave-3 port): merged-image check + executing
# tests. clock imports src.runtime_ffi, so the compile gate runs on the
# merged src image — same discipline as legs 4/6/7/8. HARD FAIL unless
# exactly 5 tests run and pass.
# ---------------------------------------------------------------------------
echo "== Leg 10: src/clock.mind (merged check + mindc test, 5 tests)"

CLK_SRC_MERGED="${WORK_DIR}/clk_src_merged.mind"
cat "${REPO_ROOT}/src/runtime_ffi.mind" \
    "${REPO_ROOT}/src/clock.mind" > "${CLK_SRC_MERGED}"

CLK_CHECK_LOG="${WORK_DIR}/check_clock.log"
"${MINDC}" check --no-fmt --no-lint "${CLK_SRC_MERGED}" >"${CLK_CHECK_LOG}" 2>&1 || \
    fail "mindc check failed for src/clock.mind merged image:
$(cat "${CLK_CHECK_LOG}")"
if grep -q 'error\[' "${CLK_CHECK_LOG}"; then
    fail "error-severity diagnostic for src/clock.mind merged image:
$(grep 'error\[' "${CLK_CHECK_LOG}" | head -20)"
fi

CLK_MERGED="${WORK_DIR}/clk_merged.mind"
cat "${CLK_SRC_MERGED}" \
    "${REPO_ROOT}/tests/unit/test_clock.mind" > "${CLK_MERGED}"

CLK_TEST_LOG="${WORK_DIR}/test_clock.log"
rc=0
"${MINDC}" test "${CLK_MERGED}" >"${CLK_TEST_LOG}" 2>&1 || rc=$?

if grep -q "running 0 tests" "${CLK_TEST_LOG}"; then
    fail "mindc test discovered 0 tests for clock (silent-green trap). Output:
$(cat "${CLK_TEST_LOG}")"
fi
grep -q "running 5 tests" "${CLK_TEST_LOG}" || \
    fail "expected 'running 5 tests' for clock. Output:
$(cat "${CLK_TEST_LOG}")"
grep -q "5 passed; 0 failed" "${CLK_TEST_LOG}" || \
    fail "expected '5 passed; 0 failed' for clock. Output:
$(cat "${CLK_TEST_LOG}")"
[ "${rc}" -eq 0 ] || fail "mindc test (clock) exited ${rc}. Output:
$(cat "${CLK_TEST_LOG}")"
echo "   OK: src/clock.mind merged image checks clean; 5/5 ns→ms + injection + counter monotonicity tests executed and passed"

# ---------------------------------------------------------------------------
# Leg 11 — src/evidence.mind (wave-4 port): merged-image check + executing
# tests. evidence imports src.lib and src.sha256, so the compile gate runs
# on the merged src image — same discipline as legs 4/6/7/8/10. HARD FAIL
# unless exactly 36 tests run and pass.
# ---------------------------------------------------------------------------
echo "== Leg 11: src/evidence.mind (merged check + mindc test, 36 tests)"

EV_SRC_MERGED="${WORK_DIR}/ev_src_merged.mind"
cat "${REPO_ROOT}/src/lib.mind" \
    "${REPO_ROOT}/src/sha256.mind" \
    "${REPO_ROOT}/src/evidence.mind" > "${EV_SRC_MERGED}"

EV_CHECK_LOG="${WORK_DIR}/check_evidence.log"
"${MINDC}" check --no-fmt --no-lint "${EV_SRC_MERGED}" >"${EV_CHECK_LOG}" 2>&1 || \
    fail "mindc check failed for src/evidence.mind merged image:
$(cat "${EV_CHECK_LOG}")"
if grep -q 'error\[' "${EV_CHECK_LOG}"; then
    fail "error-severity diagnostic for src/evidence.mind merged image:
$(grep 'error\[' "${EV_CHECK_LOG}" | head -20)"
fi

EV_MERGED="${WORK_DIR}/ev_merged.mind"
cat "${EV_SRC_MERGED}" \
    "${REPO_ROOT}/tests/unit/test_evidence.mind" > "${EV_MERGED}"

EV_TEST_LOG="${WORK_DIR}/test_evidence.log"
rc=0
"${MINDC}" test "${EV_MERGED}" >"${EV_TEST_LOG}" 2>&1 || rc=$?

if grep -q "running 0 tests" "${EV_TEST_LOG}"; then
    fail "mindc test discovered 0 tests for evidence (silent-green trap). Output:
$(cat "${EV_TEST_LOG}")"
fi
grep -q "running 36 tests" "${EV_TEST_LOG}" || \
    fail "expected 'running 36 tests' for evidence. Output:
$(cat "${EV_TEST_LOG}")"
grep -q "36 passed; 0 failed" "${EV_TEST_LOG}" || \
    fail "expected '36 passed; 0 failed' for evidence. Output:
$(cat "${EV_TEST_LOG}")"
[ "${rc}" -eq 0 ] || fail "mindc test (evidence) exited ${rc}. Output:
$(cat "${EV_TEST_LOG}")"
echo "   OK: src/evidence.mind merged image checks clean; 36/36 envelope layout + chain integrity + tamper tests executed and passed"

# ---------------------------------------------------------------------------
# Leg 12 — src/encoder_kernels.mind (wave-4 port): merged-image check +
# executing tests. encoder_kernels imports src.lib and src.q16_16, which in
# turn delegates to the pinned LUTs, so the merged src image prepends
# mind/luts/exp_q16.mind + mind/luts/sqrt_q16.mind — same discipline as
# leg 4. HARD FAIL unless exactly 17 tests run and pass.
# ---------------------------------------------------------------------------
echo "== Leg 12: src/encoder_kernels.mind (merged check + mindc test, 17 tests)"

EK_SRC_MERGED="${WORK_DIR}/ek_src_merged.mind"
cat "${REPO_ROOT}/mind/luts/exp_q16.mind" \
    "${REPO_ROOT}/mind/luts/sqrt_q16.mind" \
    "${REPO_ROOT}/src/lib.mind" \
    "${REPO_ROOT}/src/q16_16.mind" \
    "${REPO_ROOT}/src/encoder_kernels.mind" > "${EK_SRC_MERGED}"

EK_CHECK_LOG="${WORK_DIR}/check_encoder_kernels.log"
"${MINDC}" check --no-fmt --no-lint "${EK_SRC_MERGED}" >"${EK_CHECK_LOG}" 2>&1 || \
    fail "mindc check failed for src/encoder_kernels.mind merged image:
$(cat "${EK_CHECK_LOG}")"
if grep -q 'error\[' "${EK_CHECK_LOG}"; then
    fail "error-severity diagnostic for src/encoder_kernels.mind merged image:
$(grep 'error\[' "${EK_CHECK_LOG}" | head -20)"
fi

EK_MERGED="${WORK_DIR}/ek_merged.mind"
cat "${EK_SRC_MERGED}" \
    "${REPO_ROOT}/tests/unit/test_encoder_kernels.mind" > "${EK_MERGED}"

EK_TEST_LOG="${WORK_DIR}/test_encoder_kernels.log"
rc=0
"${MINDC}" test "${EK_MERGED}" >"${EK_TEST_LOG}" 2>&1 || rc=$?

if grep -q "running 0 tests" "${EK_TEST_LOG}"; then
    fail "mindc test discovered 0 tests for encoder_kernels (silent-green trap). Output:
$(cat "${EK_TEST_LOG}")"
fi
grep -q "running 17 tests" "${EK_TEST_LOG}" || \
    fail "expected 'running 17 tests' for encoder_kernels. Output:
$(cat "${EK_TEST_LOG}")"
grep -q "17 passed; 0 failed" "${EK_TEST_LOG}" || \
    fail "expected '17 passed; 0 failed' for encoder_kernels. Output:
$(cat "${EK_TEST_LOG}")"
[ "${rc}" -eq 0 ] || fail "mindc test (encoder_kernels) exited ${rc}. Output:
$(cat "${EK_TEST_LOG}")"
echo "   OK: src/encoder_kernels.mind merged image checks clean; 17/17 window math + mask + projection + attention + pooling tests executed and passed"

# ---------------------------------------------------------------------------
# Leg 13 — src/model.mind (wave-4 port): merged-image check + executing
# tests. model imports src.lib + src.q16_16 + src.encoder_kernels (LUT
# delegation chain), so the merged src image prepends the pinned LUTs —
# same discipline as legs 4/12. HARD FAIL unless exactly 8 tests run and
# pass.
# ---------------------------------------------------------------------------
echo "== Leg 13: src/model.mind (merged check + mindc test, 8 tests)"

MD_SRC_MERGED="${WORK_DIR}/md_src_merged.mind"
cat "${REPO_ROOT}/mind/luts/exp_q16.mind" \
    "${REPO_ROOT}/mind/luts/sqrt_q16.mind" \
    "${REPO_ROOT}/src/lib.mind" \
    "${REPO_ROOT}/src/q16_16.mind" \
    "${REPO_ROOT}/src/encoder_kernels.mind" \
    "${REPO_ROOT}/src/model.mind" > "${MD_SRC_MERGED}"

MD_CHECK_LOG="${WORK_DIR}/check_model.log"
"${MINDC}" check --no-fmt --no-lint "${MD_SRC_MERGED}" >"${MD_CHECK_LOG}" 2>&1 || \
    fail "mindc check failed for src/model.mind merged image:
$(cat "${MD_CHECK_LOG}")"
if grep -q 'error\[' "${MD_CHECK_LOG}"; then
    fail "error-severity diagnostic for src/model.mind merged image:
$(grep 'error\[' "${MD_CHECK_LOG}" | head -20)"
fi

MD_MERGED="${WORK_DIR}/md_merged.mind"
cat "${MD_SRC_MERGED}" \
    "${REPO_ROOT}/tests/unit/test_model.mind" > "${MD_MERGED}"

MD_TEST_LOG="${WORK_DIR}/test_model.log"
rc=0
"${MINDC}" test "${MD_MERGED}" >"${MD_TEST_LOG}" 2>&1 || rc=$?

if grep -q "running 0 tests" "${MD_TEST_LOG}"; then
    fail "mindc test discovered 0 tests for model (silent-green trap). Output:
$(cat "${MD_TEST_LOG}")"
fi
grep -q "running 8 tests" "${MD_TEST_LOG}" || \
    fail "expected 'running 8 tests' for model. Output:
$(cat "${MD_TEST_LOG}")"
grep -q "8 passed; 0 failed" "${MD_TEST_LOG}" || \
    fail "expected '8 passed; 0 failed' for model. Output:
$(cat "${MD_TEST_LOG}")"
[ "${rc}" -eq 0 ] || fail "mindc test (model) exited ${rc}. Output:
$(cat "${MD_TEST_LOG}")"
echo "   OK: src/model.mind merged image checks clean; 8/8 encoder composition + scoring-head tests executed and passed"

# ---------------------------------------------------------------------------
# Leg 14 — src/loader.mind (wave-4 port): merged-image check + executing
# tests. The merged image must be COMPLETE — every transitively referenced
# symbol defined (luts + lib + q16_16 + encoder_kernels + model +
# runtime_ffi + sha256 + loader): the 0.10.2 test interpreter poisons the
# caller's environment with a misleading "unknown variable" error when a
# CALLED function references an undefined symbol (repro:
# mind-flow/target/repro-0.10.2/r10_test_unresolved_callee_env.mind), so an
# incomplete image fails unrelated tests. HARD FAIL unless exactly 12 tests
# run and pass.
# ---------------------------------------------------------------------------
echo "== Leg 14: src/loader.mind (merged check + mindc test, 12 tests)"

LD_SRC_MERGED="${WORK_DIR}/ld_src_merged.mind"
cat "${REPO_ROOT}/mind/luts/exp_q16.mind" \
    "${REPO_ROOT}/mind/luts/sqrt_q16.mind" \
    "${REPO_ROOT}/src/lib.mind" \
    "${REPO_ROOT}/src/q16_16.mind" \
    "${REPO_ROOT}/src/encoder_kernels.mind" \
    "${REPO_ROOT}/src/model.mind" \
    "${REPO_ROOT}/src/runtime_ffi.mind" \
    "${REPO_ROOT}/src/sha256.mind" \
    "${REPO_ROOT}/src/loader.mind" > "${LD_SRC_MERGED}"

LD_CHECK_LOG="${WORK_DIR}/check_loader.log"
"${MINDC}" check --no-fmt --no-lint "${LD_SRC_MERGED}" >"${LD_CHECK_LOG}" 2>&1 || \
    fail "mindc check failed for src/loader.mind merged image:
$(cat "${LD_CHECK_LOG}")"
if grep -q 'error\[' "${LD_CHECK_LOG}"; then
    fail "error-severity diagnostic for src/loader.mind merged image:
$(grep 'error\[' "${LD_CHECK_LOG}" | head -20)"
fi

LD_MERGED="${WORK_DIR}/ld_merged.mind"
cat "${LD_SRC_MERGED}" \
    "${REPO_ROOT}/tests/unit/test_loader.mind" > "${LD_MERGED}"

LD_TEST_LOG="${WORK_DIR}/test_loader.log"
rc=0
"${MINDC}" test "${LD_MERGED}" >"${LD_TEST_LOG}" 2>&1 || rc=$?

if grep -q "running 0 tests" "${LD_TEST_LOG}"; then
    fail "mindc test discovered 0 tests for loader (silent-green trap). Output:
$(cat "${LD_TEST_LOG}")"
fi
grep -q "running 12 tests" "${LD_TEST_LOG}" || \
    fail "expected 'running 12 tests' for loader. Output:
$(cat "${LD_TEST_LOG}")"
grep -q "12 passed; 0 failed" "${LD_TEST_LOG}" || \
    fail "expected '12 passed; 0 failed' for loader. Output:
$(cat "${LD_TEST_LOG}")"
[ "${rc}" -eq 0 ] || fail "mindc test (loader) exited ${rc}. Output:
$(cat "${LD_TEST_LOG}")"
echo "   OK: src/loader.mind merged image checks clean; 12/12 parser gates + UTF-8 + dequantize + hashlib-oracle tests executed and passed"

# ---------------------------------------------------------------------------
# Leg 15 — src/inference.mind (wave-4 port): check + gate suite + NATIVE
# end-to-end run-harness (see the header leg list for why the composed
# pipeline is gated natively, not via the test interpreter).
# ---------------------------------------------------------------------------
echo "== Leg 15: src/inference.mind (check + 9 tests + native e2e harness)"

INF_SRC_MERGED="${WORK_DIR}/inf_src_merged.mind"
cat "${REPO_ROOT}/mind/luts/exp_q16.mind" \
    "${REPO_ROOT}/mind/luts/sqrt_q16.mind" \
    "${REPO_ROOT}/src/lib.mind" \
    "${REPO_ROOT}/src/q16_16.mind" \
    "${REPO_ROOT}/src/encoder_kernels.mind" \
    "${REPO_ROOT}/src/model.mind" \
    "${REPO_ROOT}/src/runtime_ffi.mind" \
    "${REPO_ROOT}/src/sha256.mind" \
    "${REPO_ROOT}/src/top_k.mind" \
    "${REPO_ROOT}/src/tokenizer.mind" \
    "${REPO_ROOT}/src/evidence.mind" \
    "${REPO_ROOT}/src/chain_log.mind" \
    "${REPO_ROOT}/src/loader.mind" \
    "${REPO_ROOT}/src/inference.mind" > "${INF_SRC_MERGED}"

INF_CHECK_LOG="${WORK_DIR}/check_inference.log"
"${MINDC}" check --no-fmt --no-lint "${INF_SRC_MERGED}" >"${INF_CHECK_LOG}" 2>&1 || \
    fail "mindc check failed for src/inference.mind merged image:
$(cat "${INF_CHECK_LOG}")"
if grep -q 'error\[' "${INF_CHECK_LOG}"; then
    fail "error-severity diagnostic for src/inference.mind merged image:
$(grep 'error\[' "${INF_CHECK_LOG}" | head -20)"
fi

INF_MERGED="${WORK_DIR}/inf_merged.mind"
cat "${INF_SRC_MERGED}" \
    "${REPO_ROOT}/tests/unit/test_inference.mind" > "${INF_MERGED}"

INF_TEST_LOG="${WORK_DIR}/test_inference.log"
rc=0
"${MINDC}" test "${INF_MERGED}" >"${INF_TEST_LOG}" 2>&1 || rc=$?

if grep -q "running 0 tests" "${INF_TEST_LOG}"; then
    fail "mindc test discovered 0 tests for inference (silent-green trap). Output:
$(cat "${INF_TEST_LOG}")"
fi
grep -q "running 9 tests" "${INF_TEST_LOG}" || \
    fail "expected 'running 9 tests' for inference. Output:
$(cat "${INF_TEST_LOG}")"
grep -q "9 passed; 0 failed" "${INF_TEST_LOG}" || \
    fail "expected '9 passed; 0 failed' for inference. Output:
$(cat "${INF_TEST_LOG}")"
[ "${rc}" -eq 0 ] || fail "mindc test (inference) exited ${rc}. Output:
$(cat "${INF_TEST_LOG}")"
echo "   OK: src/inference.mind merged image checks clean; 9/9 gate + error-mapping + e2e oracle tests executed and passed"

# Leg 15c — native end-to-end run-harness. Build the harness image in a
# scratch single-file project (the pipeline references no cross-module
# symbols once merged), verify the artifact is a REAL ELF, and require
# exit 0. Every non-zero exit code names a specific failed check (see
# tests/harness/inference_main.mind).
HARNESS_DIR="${WORK_DIR}/inf_harness"
mkdir -p "${HARNESS_DIR}"
cat "${REPO_ROOT}/mind/luts/exp_q16.mind" \
    "${REPO_ROOT}/mind/luts/sqrt_q16.mind" \
    "${REPO_ROOT}/src/lib.mind" \
    "${REPO_ROOT}/src/q16_16.mind" \
    "${REPO_ROOT}/src/encoder_kernels.mind" \
    "${REPO_ROOT}/src/model.mind" \
    "${REPO_ROOT}/tests/harness/rt_stub.mind" \
    "${REPO_ROOT}/src/sha256.mind" \
    "${REPO_ROOT}/src/top_k.mind" \
    "${REPO_ROOT}/src/tokenizer.mind" \
    "${REPO_ROOT}/src/evidence.mind" \
    "${REPO_ROOT}/src/chain_log.mind" \
    "${REPO_ROOT}/src/loader.mind" \
    "${REPO_ROOT}/src/inference.mind" \
    "${REPO_ROOT}/tests/harness/inference_main.mind" > "${HARNESS_DIR}/main.mind"
cat > "${HARNESS_DIR}/Mind.toml" <<'EOF'
[package]
name = "inf-harness"
version = "0.1.0"

[build]
entry = "main.mind"
output = "app"

[targets.cpu]
backend = "cpu"
sources = ["main.mind"]
EOF

HARNESS_BUILD_LOG="${WORK_DIR}/harness_build.log"
(cd "${HARNESS_DIR}" && "${MINDC}" build --release --no-cache) >"${HARNESS_BUILD_LOG}" 2>&1 || \
    fail "harness build exited non-zero:
$(cat "${HARNESS_BUILD_LOG}")"
if grep -q 'error\[\|not natively compiled' "${HARNESS_BUILD_LOG}"; then
    fail "harness build degraded (error or JIT fallback):
$(cat "${HARNESS_BUILD_LOG}")"
fi
HARNESS_BIN="${HARNESS_DIR}/target/release/inf-harness"
[ -f "${HARNESS_BIN}" ] || fail "harness binary missing (build produced no artifact)"
if ! file "${HARNESS_BIN}" | grep -q 'ELF 64-bit'; then
    fail "harness artifact is not a real ELF: $(file "${HARNESS_BIN}")"
fi
HARNESS_RUN_LOG="${WORK_DIR}/harness_run.log"
rc=0
"${HARNESS_BIN}" >"${HARNESS_RUN_LOG}" 2>&1 || rc=$?
[ "${rc}" -eq 0 ] || \
    fail "inference e2e harness exited ${rc} (code names the failed check; see tests/harness/inference_main.mind). Output:
$(cat "${HARNESS_RUN_LOG}")"
echo "   OK: native e2e harness — real ELF, exit 0 (stateful oracle + chain link, full-path bytes oracle, stateless chain-of-one)"

echo "GATE PASS: mindc 0.10.2 — 19/19 kernel-tree modules natively compile + link into a real ELF (zero JIT/launcher fallback); 5/5 LUT smoke tests green; src/sha256.mind 16/16, src/q16_16.mind 86/86, src/lib.mind 12/12, src/top_k.mind 30/30, src/tokenizer.mind 18/18, src/chain_log.mind 5/5, src/runtime_ffi.mind 3/3, src/clock.mind 5/5, src/evidence.mind 36/36, src/encoder_kernels.mind 17/17, src/model.mind 8/8, src/loader.mind 12/12, src/inference.mind 9/9 + native e2e harness green"
