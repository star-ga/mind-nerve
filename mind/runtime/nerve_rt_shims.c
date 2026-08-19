// mind-nerve/mind/runtime/nerve_rt_shims.c
//
// Host-runtime I/O shims for the mind-nerve native ELF (root Mind.toml
// [targets.cpu]). These DEFINE the eight `__mind_nerve_rt_*` C-ABI symbols
// declared in src/runtime_ffi.mind's `extern "C"` block (and registered in
// mindc's STD_SURFACE_INTRINSICS). Without them the native link fails with
// `undefined reference to __mind_nerve_rt_*` and mindc drops to a launcher.
//
// ABI (mirrors src/runtime_ffi.mind, i64 in / i64 out, pointers as i64
// addresses, one byte per offset; the MIND caller owns every buffer):
//   __mind_nerve_rt_monotonic_ns()                       -> ns since boot
//   __mind_nerve_rt_read_stdin(buf, cap)                 -> bytes read (0=EOF)
//   __mind_nerve_rt_write_stdout(buf, len)               -> bytes written
//   __mind_nerve_rt_write_stderr(buf, len)               -> bytes written
//   __mind_nerve_rt_file_size(path, path_len)            -> byte size
//   __mind_nerve_rt_read_file(path, path_len, buf, cap)  -> bytes read
//   __mind_nerve_rt_os_entropy(buf, len)                 -> RT_OK / RT_ERR_*
//   __mind_nerve_rt_exit(code)                           -> does not return
//
// Status codes match src/runtime_ffi.mind: RT_OK=0, RT_ERR_IO=-1,
// RT_ERR_UNSUPPORTED=-2. A negative return is an error VALUE, never a silent
// empty buffer (the audit-chain contract).
//
// DETERMINISM: monotonic_ns + os_entropy are inherently non-deterministic
// (clock / RNG). The attestation envelope strips the timestamp (offset 8)
// and arch byte (offset 16) before hashing, so a clock is tolerable; all
// other shims are plain host I/O with no hidden non-determinism (no
// buffering, no locale, no thread state). ARCH-CLEAN: fixed-width int64_t,
// byte-granular syscalls, no host-width or endian assumptions — cross-
// compiles unchanged to aarch64.

#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <errno.h>
#include <signal.h>
#include <sys/stat.h>

#if defined(__linux__)
#  include <sys/syscall.h>
#endif

// ---------------------------------------------------------------------------
// SIGPIPE discipline. Without this, a consumer that closes the read end early
// (e.g. `mind-nerve ... | head -c1`) delivers SIGPIPE on the write() and the
// process is killed by the signal — exiting on a signal, NOT through the
// documented exit codes, with a truncated/absent MNB1 envelope on stdout.
// Ignoring SIGPIPE turns that into a plain EPIPE returned from write(), which
// mind_nerve_rt_write_fd surfaces as RT_ERR_IO; cli/main.mind then maps the
// short write to MN_EXIT_IO_ERROR (7). The constructor runs before main, needs
// no MIND-side wiring, and does not touch any emitted stdout byte, so the
// bit-identity golden is unaffected.
__attribute__((constructor))
static void mind_nerve_rt_init(void) {
    signal(SIGPIPE, SIG_IGN);
}

// src/runtime_ffi.mind status codes.
#define MIND_NERVE_RT_OK              0
#define MIND_NERVE_RT_ERR_IO         (-1)
#define MIND_NERVE_RT_ERR_UNSUPPORTED (-2)

// Maximum path length accepted across the ABI (the path arrives as raw
// UTF-8 bytes with no NUL terminator; we copy into a bounded stack buffer
// for open()).
#define MIND_NERVE_RT_PATH_MAX 4096

// ---------------------------------------------------------------------------
// Monotonic nanoseconds. CLOCK_MONOTONIC_RAW is immune to NTP/settimeofday
// steps; falls back to CLOCK_MONOTONIC where RAW is unavailable.
// ---------------------------------------------------------------------------
int64_t __mind_nerve_rt_monotonic_ns(void) {
    struct timespec ts;
#if defined(CLOCK_MONOTONIC_RAW)
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
        if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
            return MIND_NERVE_RT_ERR_IO;
        }
    }
#else
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return MIND_NERVE_RT_ERR_IO;
    }
#endif
    return (int64_t)ts.tv_sec * (int64_t)1000000000 + (int64_t)ts.tv_nsec;
}

// ---------------------------------------------------------------------------
// Read up to `cap` bytes from fd 0 into `buf`. Loops across short reads;
// stops at EOF or when `cap` bytes have been read. Returns total bytes read
// (0 = immediate EOF) or RT_ERR_IO.
// ---------------------------------------------------------------------------
int64_t __mind_nerve_rt_read_stdin(int64_t buf, int64_t cap) {
    if (cap < 0) return MIND_NERVE_RT_ERR_IO;
    if (cap == 0 || buf == 0) return 0;
    unsigned char *p = (unsigned char *)(uintptr_t)buf;
    int64_t total = 0;
    while (total < cap) {
        ssize_t n = read(0, p + total, (size_t)(cap - total));
        if (n < 0) {
            if (errno == EINTR) continue;
            return MIND_NERVE_RT_ERR_IO;
        }
        if (n == 0) break; // EOF
        total += (int64_t)n;
    }
    return total;
}

// ---------------------------------------------------------------------------
// Write exactly `len` bytes from `buf` to fd `fd`, looping across short
// writes. Returns bytes written (== len on success) or RT_ERR_IO.
// ---------------------------------------------------------------------------
static int64_t mind_nerve_rt_write_fd(int fd, int64_t buf, int64_t len) {
    if (len < 0) return MIND_NERVE_RT_ERR_IO;
    if (len == 0) return 0;
    if (buf == 0) return MIND_NERVE_RT_ERR_IO;
    const unsigned char *p = (const unsigned char *)(uintptr_t)buf;
    int64_t total = 0;
    while (total < len) {
        ssize_t n = write(fd, p + total, (size_t)(len - total));
        if (n < 0) {
            if (errno == EINTR) continue;
            return MIND_NERVE_RT_ERR_IO;
        }
        total += (int64_t)n;
    }
    return total;
}

int64_t __mind_nerve_rt_write_stdout(int64_t buf, int64_t len) {
    return mind_nerve_rt_write_fd(1, buf, len);
}

int64_t __mind_nerve_rt_write_stderr(int64_t buf, int64_t len) {
    return mind_nerve_rt_write_fd(2, buf, len);
}

// ---------------------------------------------------------------------------
// Copy a non-NUL-terminated path (raw UTF-8 bytes, `path_len` long) into a
// bounded NUL-terminated stack buffer. Returns 0 on success, -1 if the path
// is too long / malformed.
// ---------------------------------------------------------------------------
static int mind_nerve_rt_copy_path(int64_t path, int64_t path_len,
                                   char *dst, size_t dst_cap) {
    if (path == 0 || path_len < 0) return -1;
    if ((size_t)path_len + 1 > dst_cap) return -1;
    const unsigned char *src = (const unsigned char *)(uintptr_t)path;
    for (int64_t i = 0; i < path_len; ++i) {
        char c = (char)src[i];
        if (c == '\0') return -1; // embedded NUL is not a valid path
        dst[i] = c;
    }
    dst[path_len] = '\0';
    return 0;
}

// ---------------------------------------------------------------------------
// Size probe: byte size of the file at `path`. Returns size (>= 0) or
// RT_ERR_IO.
// ---------------------------------------------------------------------------
int64_t __mind_nerve_rt_file_size(int64_t path, int64_t path_len) {
    char cpath[MIND_NERVE_RT_PATH_MAX];
    if (mind_nerve_rt_copy_path(path, path_len, cpath, sizeof(cpath)) != 0) {
        return MIND_NERVE_RT_ERR_IO;
    }
    struct stat st;
    if (stat(cpath, &st) != 0) return MIND_NERVE_RT_ERR_IO;
    if (!S_ISREG(st.st_mode)) return MIND_NERVE_RT_ERR_IO;
    return (int64_t)st.st_size;
}

// ---------------------------------------------------------------------------
// Read up to `cap` bytes of the file at `path` into `buf`. Loops across
// short reads; stops at EOF or `cap`. Returns bytes read (>= 0) or
// RT_ERR_IO.
// ---------------------------------------------------------------------------
int64_t __mind_nerve_rt_read_file(int64_t path, int64_t path_len,
                                  int64_t buf, int64_t cap) {
    if (cap < 0) return MIND_NERVE_RT_ERR_IO;
    char cpath[MIND_NERVE_RT_PATH_MAX];
    if (mind_nerve_rt_copy_path(path, path_len, cpath, sizeof(cpath)) != 0) {
        return MIND_NERVE_RT_ERR_IO;
    }
    int fd = open(cpath, O_RDONLY);
    if (fd < 0) return MIND_NERVE_RT_ERR_IO;
    if (cap == 0 || buf == 0) { close(fd); return 0; }
    unsigned char *p = (unsigned char *)(uintptr_t)buf;
    int64_t total = 0;
    while (total < cap) {
        ssize_t n = read(fd, p + total, (size_t)(cap - total));
        if (n < 0) {
            if (errno == EINTR) continue;
            close(fd);
            return MIND_NERVE_RT_ERR_IO;
        }
        if (n == 0) break; // EOF
        total += (int64_t)n;
    }
    close(fd);
    return total;
}

// ---------------------------------------------------------------------------
// Fill `len` bytes of `buf` with cryptographic-quality host entropy.
// Returns RT_OK on a full fill, RT_ERR_UNSUPPORTED where no host RNG is
// available, RT_ERR_IO on read error.
// ---------------------------------------------------------------------------
int64_t __mind_nerve_rt_os_entropy(int64_t buf, int64_t len) {
    if (len < 0) return MIND_NERVE_RT_ERR_IO;
    if (len == 0) return MIND_NERVE_RT_OK;
    if (buf == 0) return MIND_NERVE_RT_ERR_IO;
    unsigned char *p = (unsigned char *)(uintptr_t)buf;

#if defined(__linux__) && defined(SYS_getrandom)
    int64_t total = 0;
    while (total < len) {
        long n = syscall(SYS_getrandom, p + total, (size_t)(len - total), 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            if (errno == ENOSYS) break; // fall through to /dev/urandom
            return MIND_NERVE_RT_ERR_IO;
        }
        total += (int64_t)n;
    }
    if (total == len) return MIND_NERVE_RT_OK;
#endif

    // Portable fallback: /dev/urandom (also the aarch64/non-getrandom path).
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd < 0) return MIND_NERVE_RT_ERR_UNSUPPORTED;
    int64_t got = 0;
    while (got < len) {
        ssize_t n = read(fd, p + got, (size_t)(len - got));
        if (n < 0) {
            if (errno == EINTR) continue;
            close(fd);
            return MIND_NERVE_RT_ERR_IO;
        }
        if (n == 0) { close(fd); return MIND_NERVE_RT_ERR_IO; }
        got += (int64_t)n;
    }
    close(fd);
    return MIND_NERVE_RT_OK;
}

// ---------------------------------------------------------------------------
// Read environment variable `name` (name_len UTF-8 bytes, no NUL) into `buf`
// (up to `cap` bytes). Returns the value length (>= 0) when set,
// RT_ERR_UNSUPPORTED when unset, RT_ERR_IO on a malformed name or a buffer
// smaller than the value. Read-only host state; the `mcp` subcommand uses it
// to resolve MIND_NERVE_RUNTIME_DIR without baking a path into the binary.
// Never on a hash-preimage path (getenv output is not attested).
// ---------------------------------------------------------------------------
int64_t __mind_nerve_rt_getenv(int64_t name, int64_t name_len,
                               int64_t buf, int64_t cap) {
    if (name == 0 || name_len < 0 || cap < 0) return MIND_NERVE_RT_ERR_IO;
    char nbuf[256];
    if ((size_t)name_len + 1 > sizeof(nbuf)) return MIND_NERVE_RT_ERR_IO;
    const unsigned char *np = (const unsigned char *)(uintptr_t)name;
    for (int64_t i = 0; i < name_len; ++i) {
        char c = (char)np[i];
        if (c == '\0' || c == '=') return MIND_NERVE_RT_ERR_IO; // invalid name byte
        nbuf[i] = c;
    }
    nbuf[name_len] = '\0';
    const char *val = getenv(nbuf);
    if (val == NULL) return MIND_NERVE_RT_ERR_UNSUPPORTED;
    int64_t vlen = 0;
    while (val[vlen] != '\0') ++vlen;
    if (cap < vlen) return MIND_NERVE_RT_ERR_IO; // buffer too small for the value
    if (buf != 0) {
        unsigned char *bp = (unsigned char *)(uintptr_t)buf;
        for (int64_t i = 0; i < vlen; ++i) bp[i] = (unsigned char)val[i];
    }
    return vlen;
}

// ---------------------------------------------------------------------------
// Terminate the process with `code`. exit() flushes any C stdio; the shims
// above use raw fds (unbuffered), so stdout/stderr are already durable.
// Does not return; the i64 return keeps the C-ABI shape uniform.
// ---------------------------------------------------------------------------
int64_t __mind_nerve_rt_exit(int64_t code) {
    exit((int)code);
    return MIND_NERVE_RT_OK; // unreachable
}
