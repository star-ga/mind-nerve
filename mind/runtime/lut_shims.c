// mind-nerve/mind/runtime/lut_shims.c
//
// Phase A1.5 — C shims for the three LUT wrapper functions that the kernel
// .mind files reference as external symbols but which are not yet defined
// in the MIND LUT source tree.
//
// Design gap: the kernel files (gelu_q16.mind, layernorm_q16.mind,
// l2_norm_q16.mind, encode.mind) were written expecting single-argument
// convenience wrappers (tanh_q16, rsqrt_q16, softmax_q16) that the LUT
// files don't implement. The LUT files expose a two-phase init+lookup API
// that requires explicit handle management, which was not threaded through
// the kernel call chain.
//
// This shim file closes the link gap for A1.5 by providing C implementations
// using libm math functions. The Q16.16 fixed-point convention is identical
// to the MIND LUT convention:
//   Q16.16 value x_q16 = int64_t(x_real * 65536)
//   x_real = x_q16 / 65536.0
//
// NOTE: These C shims are NOT bit-identical to the MIND LUT implementations
// because libm uses IEEE 754 float arithmetic rather than integer LUT lookup.
// The bit-identity gate (cross-arch SHA-256 determinism) requires the final
// MIND LUT wrappers to be implemented in pure MIND with the existing LUT API.
// This shim is an A1.5 MEASUREMENT ONLY bridge; it MUST NOT ship in production.
//
// The shims are statically linked into the .so via the build script.

#include <stdint.h>
#include <math.h>

#define Q16_ONE 65536LL
#define Q16_MAX INT32_MAX
#define Q16_MIN INT32_MIN

// ---------------------------------------------------------------------------
// tanh_q16(x_q16: i64) -> i64
//
// Called by gelu_q16.mind: `let th = tanh_q16(arg)` where arg is Q16.16.
// Expected signature: one i64 in, one i64 out.
// ---------------------------------------------------------------------------
int64_t tanh_q16(int64_t x_q16) {
    double x_real = (double)x_q16 / (double)Q16_ONE;
    double y_real = tanh(x_real);
    int64_t y_q16 = (int64_t)(y_real * (double)Q16_ONE);
    if (y_q16 > Q16_MAX) y_q16 = Q16_MAX;
    if (y_q16 < Q16_MIN) y_q16 = Q16_MIN;
    return y_q16;
}

// ---------------------------------------------------------------------------
// rsqrt_q16(x_q16: i64) -> i64
//
// Called by layernorm_q16.mind and l2_norm_q16.mind.
// Computes 1/sqrt(x) in Q16.16.
// ---------------------------------------------------------------------------
int64_t rsqrt_q16(int64_t x_q16) {
    if (x_q16 <= 0) return Q16_MAX;  // sentinel for non-positive input
    double x_real = (double)x_q16 / (double)Q16_ONE;
    double y_real = 1.0 / sqrt(x_real);
    int64_t y_q16 = (int64_t)(y_real * (double)Q16_ONE);
    if (y_q16 > Q16_MAX) y_q16 = Q16_MAX;
    if (y_q16 < 0) y_q16 = 0;
    return y_q16;
}

// ---------------------------------------------------------------------------
// softmax_q16(buf: i64, n_rows: i64, row_len: i64) -> i64
//
// Called by encode.mind: `softmax_q16(attn_raw, nh * T, T)`.
// Apply softmax in-place to n_rows rows of row_len Q16.16 elements.
// buf is a flat i64 array of n_rows * row_len Q16.16 elements.
// Returns 0 on success.
//
// Implementation: for each row, compute max, subtract, exp, sum, divide.
// Uses double arithmetic for the exponential (not LUT) — A1.5 shim only.
// ---------------------------------------------------------------------------
int64_t softmax_q16(int64_t buf_addr, int64_t n_rows, int64_t row_len) {
    if (buf_addr == 0 || n_rows <= 0 || row_len <= 0) return 0;

    int64_t *buf = (int64_t *)(uintptr_t)buf_addr;

    for (int64_t r = 0; r < n_rows; r++) {
        int64_t *row = buf + r * row_len;

        // Find max in Q16.16 units
        int64_t max_q16 = row[0];
        for (int64_t c = 1; c < row_len; c++) {
            if (row[c] > max_q16) max_q16 = row[c];
        }

        // Compute exp(x_i - max) in real domain, accumulate sum
        double sum = 0.0;
        double tmp[row_len];  // VLA — acceptable for A1.5 measurement
        for (int64_t c = 0; c < row_len; c++) {
            double x_shifted = ((double)(row[c] - max_q16)) / (double)Q16_ONE;
            tmp[c] = exp(x_shifted);
            sum += tmp[c];
        }

        // Normalize and write back as Q16.16
        double inv_sum = (sum > 0.0) ? 1.0 / sum : 0.0;
        for (int64_t c = 0; c < row_len; c++) {
            double y_real = tmp[c] * inv_sum;
            int64_t y_q16 = (int64_t)(y_real * (double)Q16_ONE);
            if (y_q16 > Q16_MAX) y_q16 = Q16_MAX;
            if (y_q16 < 0) y_q16 = 0;
            row[c] = y_q16;
        }
    }
    return 0;
}
