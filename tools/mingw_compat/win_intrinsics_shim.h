/* deferred: mind_intrinsics.c's _WIN32 branch assumes a UCRT toolchain
 * (timespec_get gated on _UCRT) and calls POSIX sysconf() unconditionally
 * in the pthread pool sizer (no _WIN32 branch there at all). Ubuntu's
 * mingw-w64 packages target the legacy MSVCRT, not UCRT, so neither symbol
 * exists. Rather than patch the upstream star-ga/mind runtime-support file
 * from this repo, redirect both call sites (via -D on the compile command)
 * to the Win32-native implementations below. Upgrade path: once
 * runtime-support/mind_intrinsics.c gains an explicit non-UCRT / sysconf
 * Windows branch upstream, delete this shim and the -D flags that pull it
 * in (see tools/build_encoder_cdylib_windows.py).
 */
#ifndef MIND_NERVE_WIN_INTRINSICS_SHIM_H
#define MIND_NERVE_WIN_INTRINSICS_SHIM_H

#include <time.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

static __inline int mind_nerve_shim_timespec_get(struct timespec *ts, int base) {
    FILETIME ft;
    GetSystemTimePreciseAsFileTime(&ft);
    ULARGE_INTEGER uli;
    uli.LowPart = ft.dwLowDateTime;
    uli.HighPart = ft.dwHighDateTime;
    /* 100ns intervals since 1601-01-01 -> seconds/nanoseconds since 1970-01-01 */
    unsigned long long t = uli.QuadPart - 116444736000000000ULL;
    ts->tv_sec = (time_t)(t / 10000000ULL);
    ts->tv_nsec = (long)((t % 10000000ULL) * 100);
    return base;
}

static __inline long mind_nerve_shim_sysconf(int name) {
    (void)name;
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (long)si.dwNumberOfProcessors;
}

#endif /* MIND_NERVE_WIN_INTRINSICS_SHIM_H */
