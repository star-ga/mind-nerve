// mind-nerve/mind/runtime/lut_cache.c
//
// Phase A1.5 — handle cache for the pure-MIND Q16.16 LUT wrappers.
//
// This file SUPERSEDES the former lut_shims.c, which computed tanh /
// rsqrt / softmax with libm float arithmetic (NOT Q16.16 cross-arch
// bit-identical, MEASUREMENT ONLY).
//
// Design contract:
//   * This file performs NO arithmetic. Every numeric value — every LUT
//     table entry and every lookup/interpolation step — is produced by
//     the pure-MIND LUT sources (mind/luts/*.mind). Those functions are
//     deterministic integer Q16.16 routines, so their output is
//     bit-identical across x86 / ARM / CUDA / photonic backends by
//     construction (task #57 cross-arch gate).
//   * The only thing C does here is lazily call each pure-MIND
//     `*_init()` exactly once and cache the returned i64 table handle in
//     a static, so callers do not rebuild the (32 KiB) tables per call.
//     This matches the prior shim's implicit caching contract (the LUT
//     tables were resident for the lifetime of the .so) so kernel call
//     sites need no semantic change.
//
// The pure-MIND symbols referenced here (resolved at .so link time):
//   tanh_q16_init()                    -> i64   (mind/luts/tanh_q16.mind)
//   rsqrt_q16_init()                   -> i64   (mind/luts/sqrt_q16.mind)
//   exp_q16_init()                     -> i64   (mind/luts/exp_q16.mind)
//   recip_q32_init()                   -> i64   (mind/luts/recip_q32.mind)
//
// Exposed to the MIND LUT wrappers (called by bare name from MIND, the
// same MIND<->C link convention used by __mind_alloc and the
// __mind_nerve_blas_* score shims — no `extern fn` declaration needed):
//   __mind_nerve_lut_tanh_h()          -> i64   cached tanh table handle
//   __mind_nerve_lut_rsqrt_h()         -> i64   cached rsqrt table handle
//   __mind_nerve_lut_exp_h()           -> i64   cached exp table handle
//   __mind_nerve_lut_recip_h()         -> i64   cached recip table handle
//
// Statically linked into libmind_nerve_encoder.so via
// tools/build_encoder_cdylib.py.

#include <stdint.h>

// Pure-MIND table builders (defined in the merged MIND module).
extern int64_t tanh_q16_init(void);
extern int64_t rsqrt_q16_init(void);
extern int64_t exp_q16_init(void);
extern int64_t recip_q32_init(void);

// Cached table handles. 0 = not yet built. The first call to each
// accessor builds the table once; every subsequent call returns the
// cached pointer. Single-threaded init is sufficient here: the encoder
// is driven from one thread per handle and the tables are immutable
// once built (idempotent — a benign double-build would only leak one
// table, never corrupt results).
static int64_t lut_tanh_handle  = 0;
static int64_t lut_rsqrt_handle = 0;
static int64_t lut_exp_handle   = 0;
static int64_t lut_recip_handle = 0;

int64_t __mind_nerve_lut_tanh_h(void) {
    if (lut_tanh_handle == 0) {
        lut_tanh_handle = tanh_q16_init();
    }
    return lut_tanh_handle;
}

int64_t __mind_nerve_lut_rsqrt_h(void) {
    if (lut_rsqrt_handle == 0) {
        lut_rsqrt_handle = rsqrt_q16_init();
    }
    return lut_rsqrt_handle;
}

int64_t __mind_nerve_lut_exp_h(void) {
    if (lut_exp_handle == 0) {
        lut_exp_handle = exp_q16_init();
    }
    return lut_exp_handle;
}

int64_t __mind_nerve_lut_recip_h(void) {
    if (lut_recip_handle == 0) {
        lut_recip_handle = recip_q32_init();
    }
    return lut_recip_handle;
}
