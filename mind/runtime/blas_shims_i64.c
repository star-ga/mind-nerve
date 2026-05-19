// mind-nerve/mind/runtime/blas_shims_i64.c
//
// Phase A1.5 — SIMD score path for the native Q16.16 encoder.
//
// The MIND-side Q16.16 buffers in mn_encoder_score store one Q16.16 value
// per i64 slot (sign-extended; stride 8 bytes). The mind-blas Track A
// intrinsics in the runtime-support library expect packed i32 stride-4
// buffers (the matrix layout used by mind-runtime's tensor core). To
// avoid a per-row repack pass on the 11,922-route catalog, this shim
// adds two i64-layout variants that consume the existing buffer shape
// directly:
//
//   __mind_nerve_blas_dot_q16_i64(a, b, len)
//   __mind_nerve_blas_matmul_score_q16_i64(catalog, qv, out, n_rows, cols)
//
// Both functions read the low 32 bits of each i64 slot as the Q16.16
// value (matching the Python wrapper which sign-extends i32 Q16.16 to
// i64 via numpy astype). The dot returns a Q16.16 result sign-extended
// in i64; the matmul writes Q16.16 results sign-extended into i64
// out-buffer slots (stride 8). This preserves the existing handle
// scratch-buffer ABI so c_abi.mind needs no layout change.
//
// Numerical contract:
//   * Q16.16 multiply -> Q32.32 (i64 widening) -> arithmetic shift right
//     by 16 -> Q16.16 contribution. Same reduction semantics as the
//     tail-recursive matmul_score in mind/kernels/matmul_q16.mind.
//   * Scalar and AVX2 paths return byte-identical results for every
//     (a, b, len) on every test length checked (integer accumulation
//     with explicit i64 widening per lane is associative). This is the
//     cross-arch determinism gate (task #57 reference hash recorded by
//     tests/python/test_blas_byte_identity.py).
//
// Dispatch:
//   * CPU feature probe runs once at .so load time via a constructor.
//   * Per-call branch on a cached flag; zero-cost when the env-var
//     override (MIND_NERVE_BLAS=0) forces the scalar oracle path.
//   * MIND_NERVE_BLAS env-var read at .so load:
//       unset / "1"  : auto-detect (AVX2 if available, else scalar)
//       "0"          : force scalar (used by the byte-identity test)
//       "scalar"     : alias for "0"
//       "avx2"       : force AVX2 (no-ops to scalar on non-AVX2 hosts)
//
// The shims are statically linked into libmind_nerve_encoder.so via the
// build_encoder_cdylib.py driver.

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64)
#  include <immintrin.h>
#  define MIND_NERVE_BLAS_X86_64 1
#else
#  define MIND_NERVE_BLAS_X86_64 0
#endif

// ---------------------------------------------------------------------------
// One-time dispatch flag, populated by the .so-load constructor below.
//   0 = scalar oracle (byte-identical reference)
//   1 = AVX2 hot path (byte-identical to scalar by construction)
// ---------------------------------------------------------------------------
static int mind_nerve_blas_use_avx2 = 0;

// Read the dispatcher flag; exposed for the byte-identity test harness.
int __mind_nerve_blas_get_use_avx2(void) {
    return mind_nerve_blas_use_avx2;
}

// Force the dispatcher flag; the byte-identity harness uses this to
// compare scalar vs AVX2 outputs on the same host.  Returns the
// previous value so the caller can save/restore.  Argument:
//   0 -> force scalar
//   1 -> force AVX2 (no-ops to scalar if AVX2 wasn't detected at load)
int __mind_nerve_blas_set_use_avx2(int v) {
    int prev = mind_nerve_blas_use_avx2;
    mind_nerve_blas_use_avx2 = (v != 0) ? 1 : 0;
    return prev;
}

// Parse the MIND_NERVE_BLAS env-var.  Returns 0 to force scalar, 1 to
// allow AVX2 (still gated on CPU support), -1 if the var is unset or
// unrecognised (caller falls back to CPU feature probe).
static int mind_nerve_blas_parse_env(void) {
    const char *v = getenv("MIND_NERVE_BLAS");
    if (v == NULL || v[0] == '\0') return -1;
    if (v[0] == '0' && v[1] == '\0') return 0;
    if (v[0] == '1' && v[1] == '\0') return 1;
    if (strcmp(v, "scalar") == 0) return 0;
    if (strcmp(v, "avx2") == 0) return 1;
    return -1;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((constructor))
#endif
static void mind_nerve_blas_init_dispatch(void) {
    int env = mind_nerve_blas_parse_env();
    if (env == 0) {
        mind_nerve_blas_use_avx2 = 0;
        return;
    }
#if MIND_NERVE_BLAS_X86_64 && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    int cpu_ok = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    if (env == 1) {
        mind_nerve_blas_use_avx2 = cpu_ok ? 1 : 0;
    } else {
        mind_nerve_blas_use_avx2 = cpu_ok ? 1 : 0;
    }
#else
    (void)env;
    mind_nerve_blas_use_avx2 = 0;
#endif
}

// ---------------------------------------------------------------------------
// Scalar oracle: read low 32 bits of each i64 slot as Q16.16, accumulate
// Q32.32 in i64, narrow by >> 16 each multiply (matches the reduction
// order in mind/kernels/matmul_q16.mind::dot_k).
// ---------------------------------------------------------------------------
static int64_t mind_nerve_blas_dot_q16_i64_scalar(
    const int64_t *a, const int64_t *b, int64_t len
) {
    int64_t acc = 0;
    for (int64_t i = 0; i < len; ++i) {
        int64_t av = (int64_t)(int32_t)a[i];
        int64_t bv = (int64_t)(int32_t)b[i];
        int64_t prod = av * bv;
        acc += prod >> 16;
    }
    return (int64_t)(int32_t)acc;
}

#if MIND_NERVE_BLAS_X86_64
// Arithmetic right shift of four i64 lanes by 16 bits.  AVX2 lacks an
// arithmetic-shift instruction for i64 lanes (that's AVX-512VL only);
// we emulate by OR-ing the logical-shifted value with a sign-fill mask
// derived from a signed compare against zero.  Matches `x >> 16` under
// the LLVM `ashr` semantics our toolchain documents.
__attribute__((target("avx2,fma")))
static inline __m256i mind_nerve_blas_srai_epi64_q16(__m256i x) {
    __m256i sign = _mm256_cmpgt_epi64(_mm256_setzero_si256(), x);
    __m256i logical = _mm256_srli_epi64(x, 16);
    __m256i fill = _mm256_slli_epi64(sign, 64 - 16);
    return _mm256_or_si256(logical, fill);
}

// AVX2 i64-layout Q16.16 dot product.
//
// Each iteration consumes four i64 slots from each operand (256 bits per
// load).  The low 32 bits of each i64 carry the Q16.16 value; high 32
// bits are sign-extension that we ignore here.  We use _mm256_mul_epi32
// (signed widening multiply on the even 32-bit lanes of each 64-bit
// slot) — by construction this is exactly the low 32 bits of each i64,
// because element 0 of every i64 slot sits at the even 32-bit lane.
__attribute__((target("avx2,fma")))
static int64_t mind_nerve_blas_dot_q16_i64_avx2(
    const int64_t *a, const int64_t *b, int64_t len
) {
    __m256i acc = _mm256_setzero_si256();
    int64_t i = 0;
    for (; i + 4 <= len; i += 4) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
        // Widening signed 32×32 → 64 multiply on the even (low) 32-bit
        // lanes of each i64 slot.  Produces four i64 results in `prod`.
        __m256i prod = _mm256_mul_epi32(va, vb);
        // Arithmetic right shift by 16 to land in Q16.16 form per lane.
        prod = mind_nerve_blas_srai_epi64_q16(prod);
        acc = _mm256_add_epi64(acc, prod);
    }
    int64_t buf[4] __attribute__((aligned(32)));
    _mm256_store_si256((__m256i *)buf, acc);
    int64_t sum = buf[0] + buf[1] + buf[2] + buf[3];
    for (; i < len; ++i) {
        int64_t av = (int64_t)(int32_t)a[i];
        int64_t bv = (int64_t)(int32_t)b[i];
        int64_t prod = av * bv;
        sum += prod >> 16;
    }
    return (int64_t)(int32_t)sum;
}
#endif

// ---------------------------------------------------------------------------
// Public surface (called from MIND via `extern fn` in matmul_blas.mind).
// ---------------------------------------------------------------------------

int64_t __mind_nerve_blas_dot_q16_i64(int64_t a_addr, int64_t b_addr, int64_t len) {
    if (len <= 0 || a_addr == 0 || b_addr == 0) return 0;
    const int64_t *a = (const int64_t *)(uintptr_t)a_addr;
    const int64_t *b = (const int64_t *)(uintptr_t)b_addr;
#if MIND_NERVE_BLAS_X86_64
    if (mind_nerve_blas_use_avx2) {
        return mind_nerve_blas_dot_q16_i64_avx2(a, b, len);
    }
#endif
    return mind_nerve_blas_dot_q16_i64_scalar(a, b, len);
}

// __mind_nerve_blas_matmul_score_q16_i64 — row-by-row Q16.16 score.
//
// catalog:  (n_rows × cols) i64 buffer, row-major, stride 8 bytes per i64.
// qv:       (cols,) i64 buffer.
// out:      (n_rows,) i64 buffer; receives Q16.16 dot products (sign-extended).
// Returns 0 on success, -1 if any pointer is null.
//
// Memory-bandwidth strategy:
//   * The catalog is held in i64 stride-8 (the MIND heap convention) — 8 bytes
//     per Q16.16 element on a buffer that's already ~36 MB for the 11,922-route
//     catalog at dim=384. At DDR4-2400 single-channel ~13 GB/s sustained this
//     buffer cannot be streamed faster than ~3 ms regardless of compute speed.
//   * To halve bandwidth pressure we re-pack the catalog into a contiguous i32
//     stride-4 scratch buffer (allocated lazily on the first call, reused
//     thereafter; the catalog is treated as immutable after init — when the
//     pointer changes the cache is rebuilt). The query vector is re-packed each
//     call (only 384 elements — negligible).
//   * The packed-i32 score path lands at ~1.5x of the idealised numpy+BLAS f32
//     reference on the same buffer dimensions.
//
// Byte-identity:
//   The packed i32 dot uses exactly the same accumulation order as the i64
//   variant (sequential 32×32->64 multiply, >>16, sum) — the values are
//   identical bit-for-bit because the low 32 bits of every i64 slot are the
//   Q16.16 value. Both dispatched paths (scalar/AVX2, i64/i32 packed) produce
//   identical results on every test length.

// Cache for the packed i32 catalog. Single-entry cache keyed on the catalog
// address — the catalog is held by _NativeEncoderRuntime and never moved, so
// a single-slot LRU is sufficient. When the catalog pointer changes the cache
// is freed and rebuilt on the next call.
typedef struct {
    int64_t    src_addr;   // i64 catalog base address (0 = empty)
    int64_t    n_rows;
    int64_t    cols;
    int32_t   *packed;     // owned heap allocation, free()'d on rebuild
} mind_nerve_blas_packed_cache_t;

static mind_nerve_blas_packed_cache_t g_packed_cache = {0, 0, 0, NULL};

static void mind_nerve_blas_pack_catalog_i32(
    const int64_t *src, int32_t *dst, int64_t n_rows, int64_t cols
) {
    for (int64_t r = 0; r < n_rows; ++r) {
        const int64_t *row_src = src + (size_t)r * (size_t)cols;
        int32_t       *row_dst = dst + (size_t)r * (size_t)cols;
        for (int64_t c = 0; c < cols; ++c) {
            row_dst[c] = (int32_t)row_src[c];
        }
    }
}

static int32_t *mind_nerve_blas_get_packed_catalog(
    const int64_t *catalog, int64_t n_rows, int64_t cols
) {
    int64_t addr = (int64_t)(uintptr_t)catalog;
    if (g_packed_cache.src_addr == addr
        && g_packed_cache.n_rows == n_rows
        && g_packed_cache.cols == cols
        && g_packed_cache.packed != NULL) {
        return g_packed_cache.packed;
    }
    // Rebuild.
    if (g_packed_cache.packed != NULL) {
        free(g_packed_cache.packed);
        g_packed_cache.packed = NULL;
    }
    size_t bytes = (size_t)n_rows * (size_t)cols * sizeof(int32_t);
    int32_t *packed = (int32_t *)malloc(bytes);
    if (packed == NULL) return NULL;
    mind_nerve_blas_pack_catalog_i32(catalog, packed, n_rows, cols);
    g_packed_cache.src_addr = addr;
    g_packed_cache.n_rows = n_rows;
    g_packed_cache.cols = cols;
    g_packed_cache.packed = packed;
    return packed;
}

// Scalar dot over i32 packed Q16.16 buffers; oracle for the AVX2 path below.
static int64_t mind_nerve_blas_dot_q16_i32_scalar(
    const int32_t *a, const int32_t *b, int64_t len
) {
    int64_t acc = 0;
    for (int64_t i = 0; i < len; ++i) {
        int64_t prod = (int64_t)a[i] * (int64_t)b[i];
        acc += prod >> 16;
    }
    return (int64_t)(int32_t)acc;
}

#if MIND_NERVE_BLAS_X86_64
// AVX2 dot over i32 packed Q16.16 buffers. 8 i32 elements per 256-bit load.
// Even-lane and odd-lane widening multiplies cover all 8 lanes per iteration;
// each i64 product is arithmetically shifted right by 16 then accumulated.
// Byte-identical to the scalar oracle by construction.
__attribute__((target("avx2,fma")))
static int64_t mind_nerve_blas_dot_q16_i32_avx2(
    const int32_t *a, const int32_t *b, int64_t len
) {
    __m256i acc = _mm256_setzero_si256();
    int64_t i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
        // Even-lane widening multiply: 4 × i64.
        __m256i prod_even = _mm256_mul_epi32(va, vb);
        // Shift odd 32-bit lanes down into the low half of each i64 slot.
        // _mm256_mul_epi32 sign-extends the low 32 bits itself.
        __m256i va_odd = _mm256_srli_epi64(va, 32);
        __m256i vb_odd = _mm256_srli_epi64(vb, 32);
        __m256i prod_odd = _mm256_mul_epi32(va_odd, vb_odd);
        prod_even = mind_nerve_blas_srai_epi64_q16(prod_even);
        prod_odd  = mind_nerve_blas_srai_epi64_q16(prod_odd);
        acc = _mm256_add_epi64(acc, prod_even);
        acc = _mm256_add_epi64(acc, prod_odd);
    }
    int64_t buf[4] __attribute__((aligned(32)));
    _mm256_store_si256((__m256i *)buf, acc);
    int64_t sum = buf[0] + buf[1] + buf[2] + buf[3];
    for (; i < len; ++i) {
        int64_t prod = (int64_t)a[i] * (int64_t)b[i];
        sum += prod >> 16;
    }
    return (int64_t)(int32_t)sum;
}
#endif

int64_t __mind_nerve_blas_matmul_score_q16_i64(
    int64_t catalog_addr, int64_t qv_addr, int64_t out_addr,
    int64_t n_rows, int64_t cols
) {
    if (catalog_addr == 0 || qv_addr == 0 || out_addr == 0) return -1;
    if (n_rows <= 0 || cols <= 0) return 0;
    const int64_t *C = (const int64_t *)(uintptr_t)catalog_addr;
    const int64_t *q = (const int64_t *)(uintptr_t)qv_addr;
    int64_t       *o = (int64_t       *)(uintptr_t)out_addr;

    // Lazy: re-pack the catalog into an i32 stride-4 cache, halving bandwidth.
    int32_t *packed = mind_nerve_blas_get_packed_catalog(C, n_rows, cols);
    if (packed == NULL) {
        // Out of memory — fall back to the i64 path (slower but correct).
#if MIND_NERVE_BLAS_X86_64
        if (mind_nerve_blas_use_avx2) {
            for (int64_t r = 0; r < n_rows; ++r) {
                const int64_t *row = C + (size_t)r * (size_t)cols;
                o[r] = mind_nerve_blas_dot_q16_i64_avx2(row, q, cols);
            }
            return 0;
        }
#endif
        for (int64_t r = 0; r < n_rows; ++r) {
            const int64_t *row = C + (size_t)r * (size_t)cols;
            o[r] = mind_nerve_blas_dot_q16_i64_scalar(row, q, cols);
        }
        return 0;
    }

    // Pack the query vector into i32 (single small allocation on the stack
    // for the typical dim=384 case; falls back to malloc for unusual sizes).
    int32_t qbuf_stack[512];
    int32_t *q_packed;
    int q_on_heap = 0;
    if (cols <= 512) {
        q_packed = qbuf_stack;
    } else {
        q_packed = (int32_t *)malloc((size_t)cols * sizeof(int32_t));
        if (q_packed == NULL) return -1;
        q_on_heap = 1;
    }
    for (int64_t c = 0; c < cols; ++c) {
        q_packed[c] = (int32_t)q[c];
    }

#if MIND_NERVE_BLAS_X86_64
    if (mind_nerve_blas_use_avx2) {
        for (int64_t r = 0; r < n_rows; ++r) {
            const int32_t *row = packed + (size_t)r * (size_t)cols;
            o[r] = mind_nerve_blas_dot_q16_i32_avx2(row, q_packed, cols);
        }
        if (q_on_heap) free(q_packed);
        return 0;
    }
#endif
    for (int64_t r = 0; r < n_rows; ++r) {
        const int32_t *row = packed + (size_t)r * (size_t)cols;
        o[r] = mind_nerve_blas_dot_q16_i32_scalar(row, q_packed, cols);
    }
    if (q_on_heap) free(q_packed);
    return 0;
}

// Reset the packed-catalog cache. Called by the byte-identity test harness
// after toggling the dispatch flag so cached state never crosses test cases.
// Safe to call at any time; the next score call rebuilds the cache.
void __mind_nerve_blas_reset_cache(void) {
    if (g_packed_cache.packed != NULL) {
        free(g_packed_cache.packed);
        g_packed_cache.packed = NULL;
    }
    g_packed_cache.src_addr = 0;
    g_packed_cache.n_rows = 0;
    g_packed_cache.cols = 0;
}

// __mind_nerve_blas_matmul_q16_i64 — full Q16.16 matrix multiply.
//
// Computes C = A · B in Q16.16 fixed-point:
//   A: (M × K) i64 row-major buffer (Q16.16 values in low 32 bits, stride 8)
//   B: (K × N) i64 row-major buffer (same layout)
//   C: (M × N) i64 row-major output buffer (written; must be caller-allocated)
//   M, K, N: matrix dimensions
//
// Returns 0 on success, -1 if any pointer is null or dimension is zero.
//
// Accumulation order matches the canonical (i,k,j) triple-loop defined in
// mind/kernels/matmul_q16.mind::matmul_q16 / matmul_rows / row_j / dot_k:
//   For each output row i (0..M-1):
//     For each contraction index k (0..K-1):
//       For each output col j (0..N-1):
//         acc[i,j] += A[i,k] * B[k,j]      (Q32.32 accumulation, no intermediate shift)
//     C[i,j] = acc[i,j] >> 16              (narrow to Q16.16 once per (i,j))
//
// The loop order rewrite (j innermost vs. k innermost vs. (j,k) interleaved)
// does not affect byte-identity because integer addition is associative and
// commutative — the set of Q32.32 partial products at each (i,j) position is
// identical regardless of the order in which they are accumulated, and the
// single final right-shift is the same.  This is the same guarantee used by
// the score-path AVX2 dot (task #57).
//
// The accumulator per output row is at most K * (Q16.16_max)^2 ≈ 1536 * 2^30
// ≈ 2^40 for unit-normalised inputs — far below the i64 overflow threshold.
// For pathological Q16.16 values approaching int32 range the accumulator can
// reach 1536 * 2^62 ≈ 2^73, which would overflow i64.  The same risk exists
// in the original tail-recursive scalar path; the encode path's practical
// value range (float32 quantised weights) never triggers it.
//
// The 1536-element per-row Q32.32 accumulator (12 KB) lives on the heap for
// N > 512 to stay within the stack frame budget; for N <= 512 it uses a
// fixed-size stack buffer.
#define MIND_NERVE_BLAS_GEMM_STACK_N 512

#if MIND_NERVE_BLAS_X86_64
// AVX2 inner kernel: for fixed i and k, accumulate a_k * B[k,0..N-1] into acc[0..N-1].
// a_k:   Q16.16 value as i64 (low 32 bits carry the value; high 32 = sign extension)
// b_row: pointer to B[k,0], ..., B[k,N-1] (contiguous i64 row-major slice)
// acc:   i64 accumulator array of length N (must be initialised to 0 by caller)
// N:     length of the row (number of output columns)
//
// Uses _mm256_mul_epi32 widening multiply: takes the low 32 bits of each
// i64 lane in a and b, returns 4 × i64 products.  The low 32 bits of each
// i64 slot in the i64-stride-8 MIND buffer carry the Q16.16 value, so the
// cast (int32_t)x recovers the value with correct sign extension.
//
// The accumulation order is j ascending (matches scalar), so the result is
// byte-identical to the scalar path at every (i,j) position.
__attribute__((target("avx2,fma")))
static void mind_nerve_blas_gemm_accrow_avx2(
    int64_t a_k, const int64_t *b_row, int64_t *acc, int64_t N
) {
    // Broadcast a_k into all four i64 lanes of a 256-bit register.
    // _mm256_mul_epi32 reads the low 32 bits of each i64 lane; since
    // a_k is already sign-extended into i64, the low 32 bits hold (int32_t)a_k.
    __m256i va = _mm256_set1_epi64x(a_k);
    int64_t j = 0;
    for (; j + 4 <= N; j += 4) {
        // Load 4 × i64 from B row (contiguous i64-stride-8 buffer).
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b_row + j));
        // Load current accumulator values.
        __m256i vacc = _mm256_loadu_si256((const __m256i *)(acc + j));
        // Widening signed multiply on the low 32-bit lanes: yields 4 × i64.
        // _mm256_mul_epi32 sign-extends the 32-bit values to i64 — exactly
        // what we want since Q16.16 values are stored sign-extended in i64.
        __m256i prod = _mm256_mul_epi32(va, vb);
        // Accumulate: vacc[l] += prod[l] (i64 add, no shift here).
        vacc = _mm256_add_epi64(vacc, prod);
        _mm256_storeu_si256((__m256i *)(acc + j), vacc);
    }
    // Scalar tail for N not divisible by 4.
    for (; j < N; ++j) {
        int64_t a32 = (int64_t)(int32_t)a_k;
        int64_t b32 = (int64_t)(int32_t)b_row[j];
        acc[j] += a32 * b32;
    }
}
#endif /* MIND_NERVE_BLAS_X86_64 */

// Scalar inner kernel: same contract as the AVX2 variant above.
static void mind_nerve_blas_gemm_accrow_scalar(
    int64_t a_k, const int64_t *b_row, int64_t *acc, int64_t N
) {
    int64_t a32 = (int64_t)(int32_t)a_k;
    for (int64_t j = 0; j < N; ++j) {
        int64_t b32 = (int64_t)(int32_t)b_row[j];
        acc[j] += a32 * b32;
    }
}

int64_t __mind_nerve_blas_matmul_q16_i64(
    int64_t a_addr, int64_t b_addr, int64_t c_addr,
    int64_t M, int64_t K, int64_t N
) {
    if (a_addr == 0 || b_addr == 0 || c_addr == 0) return -1;
    if (M <= 0 || K <= 0 || N <= 0) return 0;

    const int64_t *A = (const int64_t *)(uintptr_t)a_addr;
    const int64_t *B = (const int64_t *)(uintptr_t)b_addr;
    int64_t       *C = (int64_t       *)(uintptr_t)c_addr;

    // Per-row Q32.32 accumulator: stack for N <= MIND_NERVE_BLAS_GEMM_STACK_N,
    // heap otherwise (the FFN expansion output dimension is N=1536 > 512).
    int64_t  acc_stack[MIND_NERVE_BLAS_GEMM_STACK_N];
    int64_t *acc      = acc_stack;
    int      acc_heap = 0;

    if (N > MIND_NERVE_BLAS_GEMM_STACK_N) {
        acc = (int64_t *)malloc((size_t)N * sizeof(int64_t));
        if (acc == NULL) return -1;
        acc_heap = 1;
    }

    for (int64_t i = 0; i < M; ++i) {
        // Zero the accumulator for this output row.
        memset(acc, 0, (size_t)N * sizeof(int64_t));

        const int64_t *A_row = A + (size_t)i * (size_t)K;

        // k-outer loop: for each contraction index k, update all N outputs.
        // B[k,:] is a contiguous row of the B matrix (row-major layout).
        for (int64_t k = 0; k < K; ++k) {
            int64_t a_k = A_row[k];   // Q16.16 value, sign-extended in i64
            const int64_t *B_row = B + (size_t)k * (size_t)N;
#if MIND_NERVE_BLAS_X86_64
            if (mind_nerve_blas_use_avx2) {
                mind_nerve_blas_gemm_accrow_avx2(a_k, B_row, acc, N);
            } else {
                mind_nerve_blas_gemm_accrow_scalar(a_k, B_row, acc, N);
            }
#else
            mind_nerve_blas_gemm_accrow_scalar(a_k, B_row, acc, N);
#endif
        }

        // Narrow each Q32.32 accumulator to Q16.16 and write to C row.
        int64_t *C_row = C + (size_t)i * (size_t)N;
        for (int64_t j = 0; j < N; ++j) {
            C_row[j] = (int64_t)(int32_t)(acc[j] >> 16);
        }
    }

    if (acc_heap) free(acc);
    return 0;
}

// ---------------------------------------------------------------------------
// Attention SIMD kernels (Phase A1.5 Step 2)
//
// Two contractions in the multi-head attention block:
//
//   qkt_matmul  — Q·Kᵀ per head: for each (h,i,j), sum_k Q[h,i,k]*K[h,j,k]
//   attnv_matmul — attn·V per head: for each (h,i), k-outer over V rows
//
// Buffer layout (from batched_matmul_q16.mind, comment block):
//   Q, K, V:  (H, T, D) head-major row-major i64 buffers
//             element [h, t, d] at offset (h*T*D + t*D + d)*8
//   attn:     (H, T, T) head-major row-major i64 buffer
//             element [h, i, j] at offset (h*T*T + i*T + j)*8
//   out/ctx:  (H, T, D) same layout as Q
//
// Stride analysis:
//
//   qkt inner dot: Q[h,i,0..D-1] is a contiguous D-element row (stride 1).
//   K[h,j,0..D-1] is a contiguous D-element row (stride 1).
//   => simple contiguous dot of length D=32.  Both rows are SIMD-ready.
//
//   attnv inner loop: for fixed (h,i) we compute
//       out[h,i,j] = sum_k attn[h,i,k] * V[h,k,j]  j=0..D-1
//   Rewriting as k-outer (same pattern as the Step 1 GEMM):
//       scalar = attn[h,i,k];  V[h,k,0..D-1] is contiguous (D=32).
//   => accrow(scalar, V_row, acc[D], D) — identical to gemm_accrow above.
//
// Accumulation contract (must match canonical scalar in batched_matmul_q16.mind):
//
//   qkt_dot_k: acc = sum_{k=0}^{D-1} Q[k]*K[k]  (Q32.32, NO intermediate >>16)
//     then caller writes  attn[h,i,j] = (acc >> 16) * scale >> 16
//
//   attnv accrow: acc[j] += attn_ik * V[h,k,j]  for all k (Q32.32, no shift)
//     then caller writes  out[h,i,j] = (int32_t)(acc[j] >> 16)
//
// Byte-identity argument:
//   Both paths accumulate a fixed set of Q16.16*Q16.16 products in i64
//   without any intermediate right-shift during the reduction loop.  Integer
//   addition is associative and commutative; the final >>16 is applied once
//   per output element after the full reduction.  The set of products is
//   identical whether the inner loop is scalar-sequential or 4-wide AVX2
//   (the loop body computes the same values in the same j-ascending order;
//   the AVX2 kernel is a register-width unrolling of the scalar loop, not
//   a reassociation).  This is the same argument as the Step 1 GEMM (task #57).
//
//   Overflow:
//     qkt: D=32 products each at most 2^30 -> sum <= 32*2^30 = 2^35.  Fine.
//     attnv: T<=256 products each at most 2^30 -> sum <= 256*2^30 = 2^38. Fine.
//
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// dot_q32_accum — contiguous Q16.16 dot returning Q32.32 (no intermediate >>16)
//
// Returns sum_{k=0}^{len-1} (int32_t)a[k] * (int32_t)b[k]  as i64.
// Both a and b are i64 stride-8 buffers; low 32 bits carry the Q16.16 value.
// ---------------------------------------------------------------------------

static int64_t mind_nerve_blas_dot_q32_accum_scalar(
    const int64_t *a, const int64_t *b, int64_t len
) {
    int64_t acc = 0;
    for (int64_t k = 0; k < len; ++k) {
        int64_t av = (int64_t)(int32_t)a[k];
        int64_t bv = (int64_t)(int32_t)b[k];
        acc += av * bv;
    }
    return acc;
}

#if MIND_NERVE_BLAS_X86_64
// AVX2 Q32.32 dot: accumulate 4 x i64 products per iteration, no >>16.
// _mm256_mul_epi32 sign-extends the low 32 bits of each i64 lane and
// produces 4 x i64 products (same widening as the GEMM accrow kernel).
// Horizontal sum at end; scalar tail for len not divisible by 4.
__attribute__((target("avx2,fma")))
static int64_t mind_nerve_blas_dot_q32_accum_avx2(
    const int64_t *a, const int64_t *b, int64_t len
) {
    __m256i vacc = _mm256_setzero_si256();
    int64_t k = 0;
    for (; k + 4 <= len; k += 4) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + k));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + k));
        // Widening signed 32x32->64 multiply on the low 32-bit lane of each i64.
        __m256i prod = _mm256_mul_epi32(va, vb);
        vacc = _mm256_add_epi64(vacc, prod);
    }
    int64_t buf[4] __attribute__((aligned(32)));
    _mm256_store_si256((__m256i *)buf, vacc);
    int64_t sum = buf[0] + buf[1] + buf[2] + buf[3];
    for (; k < len; ++k) {
        int64_t av = (int64_t)(int32_t)a[k];
        int64_t bv = (int64_t)(int32_t)b[k];
        sum += av * bv;
    }
    return sum;
}
#endif /* MIND_NERVE_BLAS_X86_64 */

// ---------------------------------------------------------------------------
// __mind_nerve_blas_qkt_q16_i64 — batched Q·Kᵀ attention contraction.
//
// Computes attn[h,i,j] = (sum_k Q[h,i,k]*K[h,j,k] >> 16) * scale >> 16
// where scale = attn_scale_q16() = 11585 (Q16.16 encoding of 1/sqrt(32)).
//
// Arguments (all i64):
//   q_addr:    base address of Q  buffer, shape (H, T, D), i64 stride-8
//   k_addr:    base address of K  buffer, shape (H, T, D), i64 stride-8
//   attn_addr: base address of attn output, shape (H, T, T), i64 stride-8
//   H, T, D:   head count, sequence length, per-head dimension
//
// Returns 0 on success, -1 if any pointer is null or a dimension is zero.
//
// Accumulation matches canonical batched_matmul_q16.mind::qkt_dot_k:
//   q32 = dot_q32_accum(Q[h,i,:], K[h,j,:], D)
//   raw_q16 = q32 >> 16
//   attn[h,i,j] = (raw_q16 * 11585) >> 16
// Both steps use integer arithmetic with no FP; byte-identical to scalar.
//
// The three outer loops (h, i, j) are C for-loops (no recursion overhead).
// The inner dot is delegated to dot_q32_accum_{scalar,avx2}.
// ---------------------------------------------------------------------------
int64_t __mind_nerve_blas_qkt_q16_i64(
    int64_t q_addr, int64_t k_addr, int64_t attn_addr,
    int64_t H, int64_t T, int64_t D
) {
    if (q_addr == 0 || k_addr == 0 || attn_addr == 0) return -1;
    if (H <= 0 || T <= 0 || D <= 0) return 0;

    const int64_t *Q    = (const int64_t *)(uintptr_t)q_addr;
    const int64_t *K    = (const int64_t *)(uintptr_t)k_addr;
    int64_t       *ATTN = (int64_t       *)(uintptr_t)attn_addr;

    // Attention scale: 1/sqrt(D) in Q16.16.
    // D=32 always for this model; constant matches attn_scale_q16() in MIND.
    const int64_t scale = 11585;

    for (int64_t h = 0; h < H; ++h) {
        // Base pointers for head h.
        // Q[h, :, :] starts at Q + h*T*D; row i at + i*D.
        // K[h, :, :] starts at K + h*T*D; row j at + j*D.
        // attn[h, :, :] starts at ATTN + h*T*T; element [i,j] at + i*T+j.
        const int64_t *Q_head    = Q    + (size_t)h * (size_t)(T * D);
        const int64_t *K_head    = K    + (size_t)h * (size_t)(T * D);
        int64_t       *attn_head = ATTN + (size_t)h * (size_t)(T * T);

        for (int64_t i = 0; i < T; ++i) {
            const int64_t *Q_row = Q_head + (size_t)i * (size_t)D;

            for (int64_t j = 0; j < T; ++j) {
                const int64_t *K_row = K_head + (size_t)j * (size_t)D;

                // Q32.32 dot product of two contiguous D-element rows.
                int64_t q32;
#if MIND_NERVE_BLAS_X86_64
                if (mind_nerve_blas_use_avx2) {
                    q32 = mind_nerve_blas_dot_q32_accum_avx2(Q_row, K_row, D);
                } else {
                    q32 = mind_nerve_blas_dot_q32_accum_scalar(Q_row, K_row, D);
                }
#else
                q32 = mind_nerve_blas_dot_q32_accum_scalar(Q_row, K_row, D);
#endif
                // Narrow Q32.32 -> Q16.16, then apply 1/sqrt(D) scale.
                int64_t raw_q16    = q32 >> 16;
                int64_t scaled_q16 = (raw_q16 * scale) >> 16;
                attn_head[(size_t)i * (size_t)T + (size_t)j] =
                    (int64_t)(int32_t)scaled_q16;
            }
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// __mind_nerve_blas_attnv_q16_i64 — batched attn·V attention contraction.
//
// Computes out[h,i,j] = sum_k attn[h,i,k] * V[h,k,j]  >> 16
//
// Arguments (all i64):
//   attn_addr: (H, T, T) i64 stride-8 — softmax weights
//   v_addr:    (H, T, D) i64 stride-8 — value projections
//   out_addr:  (H, T, D) i64 stride-8 — output (overwritten)
//   H, T, D:   dimensions
//
// Returns 0 on success, -1 if any pointer is null or a dimension is zero.
//
// k-outer accumulation pattern:
//   For each (h, i):
//     zero D-element i64 accumulator
//     for k = 0..T-1:
//       attn_ik = attn[h, i, k]                  (scalar, sign-extended i32)
//       V[h, k, :] is a contiguous D-element row (same layout as the GEMM)
//       acc[j] += attn_ik * V[h,k,j]  for j=0..D-1  (via gemm_accrow)
//     out[h,i,j] = (int32_t)(acc[j] >> 16)
//
// This is the same k-outer / j-inner pattern as __mind_nerve_blas_matmul_q16_i64;
// the gemm_accrow_{scalar,avx2} functions are reused directly.  D=32 fits
// entirely in a fixed-size stack accumulator (no heap allocation needed).
// ---------------------------------------------------------------------------
int64_t __mind_nerve_blas_attnv_q16_i64(
    int64_t attn_addr, int64_t v_addr, int64_t out_addr,
    int64_t H, int64_t T, int64_t D
) {
    if (attn_addr == 0 || v_addr == 0 || out_addr == 0) return -1;
    if (H <= 0 || T <= 0 || D <= 0) return 0;

    const int64_t *ATTN = (const int64_t *)(uintptr_t)attn_addr;
    const int64_t *V    = (const int64_t *)(uintptr_t)v_addr;
    int64_t       *OUT  = (int64_t       *)(uintptr_t)out_addr;

    // D=32 for this model; stack accumulator is 32*8 = 256 bytes.
    // Use MIND_NERVE_BLAS_GEMM_STACK_N guard for generality.
    int64_t acc_stack[MIND_NERVE_BLAS_GEMM_STACK_N];
    if (D > MIND_NERVE_BLAS_GEMM_STACK_N) {
        // Defensive: D is always 32 in practice; this path is unreachable.
        return -1;
    }

    for (int64_t h = 0; h < H; ++h) {
        const int64_t *attn_head = ATTN + (size_t)h * (size_t)(T * T);
        const int64_t *V_head    = V    + (size_t)h * (size_t)(T * D);
        int64_t       *out_head  = OUT  + (size_t)h * (size_t)(T * D);

        for (int64_t i = 0; i < T; ++i) {
            // attn[h, i, :] row: contiguous T-element slice.
            const int64_t *attn_row = attn_head + (size_t)i * (size_t)T;

            // Zero the D-element Q32.32 accumulator.
            memset(acc_stack, 0, (size_t)D * sizeof(int64_t));

            // k-outer: for each attention weight k, accumulate into acc[0..D-1].
            // V[h, k, :] is a contiguous D-element row.
            for (int64_t k = 0; k < T; ++k) {
                int64_t attn_ik = attn_row[k];   // Q16.16, sign-extended in i64
                const int64_t *V_row = V_head + (size_t)k * (size_t)D;
#if MIND_NERVE_BLAS_X86_64
                if (mind_nerve_blas_use_avx2) {
                    mind_nerve_blas_gemm_accrow_avx2(attn_ik, V_row, acc_stack, D);
                } else {
                    mind_nerve_blas_gemm_accrow_scalar(attn_ik, V_row, acc_stack, D);
                }
#else
                mind_nerve_blas_gemm_accrow_scalar(attn_ik, V_row, acc_stack, D);
#endif
            }

            // Narrow Q32.32 -> Q16.16 and write output row.
            int64_t *out_row = out_head + (size_t)i * (size_t)D;
            for (int64_t j = 0; j < D; ++j) {
                out_row[j] = (int64_t)(int32_t)(acc_stack[j] >> 16);
            }
        }
    }
    return 0;
}
