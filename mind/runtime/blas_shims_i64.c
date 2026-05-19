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
