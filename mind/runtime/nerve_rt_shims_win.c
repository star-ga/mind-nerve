// mind-nerve/mind/runtime/nerve_rt_shims_win.c
//
// WINDOWS (PE) host-runtime I/O shims for the mind-nerve native binary
// (Mind.toml [targets.windows]). Windows-native counterpart of
// nerve_rt_shims.c: DEFINES the same nine `__mind_nerve_rt_*` C-ABI symbols
// with the IDENTICAL i64-in / i64-out ABI, status codes, and buffer semantics,
// so the MIND payload code (cli/main.mind + src/*.mind) is byte-for-byte
// unchanged across Linux and Windows. Only the host-I/O boundary differs
// (POSIX syscalls -> Win32 / MSVCRT).
//
// ABI (identical to nerve_rt_shims.c; pointers arrive as i64 addresses, MIND
// owns every buffer; one byte per offset):
//   __mind_nerve_rt_monotonic_ns()                       -> ns (QueryPerformanceCounter)
//   __mind_nerve_rt_read_stdin(buf, cap)                 -> bytes read (0=EOF)
//   __mind_nerve_rt_write_stdout(buf, len)               -> bytes written
//   __mind_nerve_rt_write_stderr(buf, len)               -> bytes written
//   __mind_nerve_rt_file_size(path, path_len)            -> byte size
//   __mind_nerve_rt_read_file(path, path_len, buf, cap)  -> bytes read
//   __mind_nerve_rt_os_entropy(buf, len)                 -> RT_OK / RT_ERR_*
//   __mind_nerve_rt_getenv(name, name_len, buf, cap)     -> value len / RT_ERR_*
//   __mind_nerve_rt_exit(code)                           -> does not return
//
// Status codes match src/runtime_ffi.mind: RT_OK=0, RT_ERR_IO=-1,
// RT_ERR_UNSUPPORTED=-2. A negative return is an error VALUE, never a silent
// empty buffer (the audit-chain contract).
//
// *** BYTE-IDENTITY: the load-bearing Windows difference is BINARY-MODE STDIO. ***
// The Windows CRT defaults stdin/stdout/stderr to TEXT mode, which translates
// '\n' <-> '\r\n' on read/write. That would inject 0x0D bytes into the raw MNB1
// attestation frame on stdout and DIVERGE from the Linux/ARM golden. The
// constructor below forces _O_BINARY on fds 0/1/2 before main, so every byte is
// passed through verbatim -- this is what keeps cross-substrate byte-identity
// holding on Windows. It REPLACES the Linux SIGPIPE-ignore constructor: Windows
// has no SIGPIPE; a closed read end surfaces as a failed _write returning an
// error, which mind_nerve_rt_write_fd maps to RT_ERR_IO -> cli/main.mind's
// MN_EXIT_IO_ERROR (7), exactly as on Linux.
//
// DETERMINISM: monotonic_ns + os_entropy are inherently non-deterministic
// (clock / RNG). The attestation envelope strips the timestamp (offset 8) and
// arch byte (offset 16) before hashing, so a clock is tolerable; all other
// shims are plain host I/O with no hidden non-determinism. Pure-integer i64,
// little-endian x86-64 -- identical arithmetic to the Linux/ARM payload; the
// SysV<->Win64 ABI difference is arg-register-internal and never touches an
// emitted result byte.

#include <stdint.h>
#include <stdlib.h>
#include <io.h>          // _read _write _open _close _setmode
#include <fcntl.h>       // _O_RDONLY _O_BINARY
#include <sys/stat.h>    // _stat64 _S_IFMT _S_IFREG
#include <errno.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>     // QueryPerformanceCounter / QueryPerformanceFrequency
#include <bcrypt.h>      // BCryptGenRandom (link -lbcrypt)

#define MIND_NERVE_RT_OK               0
#define MIND_NERVE_RT_ERR_IO          (-1)
#define MIND_NERVE_RT_ERR_UNSUPPORTED (-2)
#define MIND_NERVE_RT_PATH_MAX        4096
#define MIND_NERVE_RT_IO_CHUNK        0x7fffffff  // _read/_write take unsigned int

// Force binary mode on the standard fds BEFORE main so no CRLF translation ever
// touches the MNB1 wire bytes. Load-bearing for cross-substrate byte-identity;
// runs before main, needs no MIND-side wiring, touches no emitted stdout byte.
__attribute__((constructor))
static void mind_nerve_rt_init(void) {
    _setmode(0, _O_BINARY);
    _setmode(1, _O_BINARY);
    _setmode(2, _O_BINARY);
}

// Monotonic nanoseconds via QPC. Overflow-safe scale (ns = cnt*1e9/freq split
// into whole-seconds + remainder so cnt*1e9 never overflows i64).
int64_t __mind_nerve_rt_monotonic_ns(void) {
    LARGE_INTEGER c, f;
    if (!QueryPerformanceFrequency(&f) || f.QuadPart <= 0) return MIND_NERVE_RT_ERR_IO;
    if (!QueryPerformanceCounter(&c)) return MIND_NERVE_RT_ERR_IO;
    int64_t freq = (int64_t)f.QuadPart;
    int64_t cnt  = (int64_t)c.QuadPart;
    int64_t sec  = cnt / freq;
    int64_t rem  = cnt % freq;
    return sec * (int64_t)1000000000 + (rem * (int64_t)1000000000) / freq;
}

// Read up to `cap` bytes from fd 0 into `buf`. Loops across short reads; stops
// at EOF or `cap`. Returns total bytes read (0 = immediate EOF) or RT_ERR_IO.
int64_t __mind_nerve_rt_read_stdin(int64_t buf, int64_t cap) {
    if (cap < 0) return MIND_NERVE_RT_ERR_IO;
    if (cap == 0 || buf == 0) return 0;
    unsigned char *p = (unsigned char *)(uintptr_t)buf;
    int64_t total = 0;
    while (total < cap) {
        int64_t left = cap - total;
        unsigned int want = left > MIND_NERVE_RT_IO_CHUNK ? MIND_NERVE_RT_IO_CHUNK : (unsigned int)left;
        int n = _read(0, p + total, want);
        if (n < 0) {
            if (errno == EINTR) continue;
            return MIND_NERVE_RT_ERR_IO;
        }
        if (n == 0) break; // EOF
        total += (int64_t)n;
    }
    return total;
}

// Write exactly `len` bytes from `buf` to fd `fd`, looping across short writes.
// A closed read end returns an error here (no SIGPIPE on Windows) -> RT_ERR_IO.
static int64_t mind_nerve_rt_write_fd(int fd, int64_t buf, int64_t len) {
    if (len < 0) return MIND_NERVE_RT_ERR_IO;
    if (len == 0) return 0;
    if (buf == 0) return MIND_NERVE_RT_ERR_IO;
    const unsigned char *p = (const unsigned char *)(uintptr_t)buf;
    int64_t total = 0;
    while (total < len) {
        int64_t left = len - total;
        unsigned int want = left > MIND_NERVE_RT_IO_CHUNK ? MIND_NERVE_RT_IO_CHUNK : (unsigned int)left;
        int n = _write(fd, p + total, want);
        if (n < 0) {
            if (errno == EINTR) continue;
            return MIND_NERVE_RT_ERR_IO;
        }
        total += (int64_t)n;
    }
    return total;
}

int64_t __mind_nerve_rt_write_stdout(int64_t buf, int64_t len) { return mind_nerve_rt_write_fd(1, buf, len); }
int64_t __mind_nerve_rt_write_stderr(int64_t buf, int64_t len) { return mind_nerve_rt_write_fd(2, buf, len); }

// Copy a non-NUL-terminated path (raw UTF-8 bytes, `path_len` long) into a
// bounded NUL-terminated stack buffer. Returns 0 on success, -1 if too long /
// malformed. (Identical to the Linux shim; ANSI path -> _open/_stat64.)
static int mind_nerve_rt_copy_path(int64_t path, int64_t path_len, char *dst, size_t dst_cap) {
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

// Byte size of the regular file at `path`. Returns size (>= 0) or RT_ERR_IO.
int64_t __mind_nerve_rt_file_size(int64_t path, int64_t path_len) {
    char cpath[MIND_NERVE_RT_PATH_MAX];
    if (mind_nerve_rt_copy_path(path, path_len, cpath, sizeof(cpath)) != 0) return MIND_NERVE_RT_ERR_IO;
    struct _stat64 st;
    if (_stat64(cpath, &st) != 0) return MIND_NERVE_RT_ERR_IO;
    if ((st.st_mode & _S_IFMT) != _S_IFREG) return MIND_NERVE_RT_ERR_IO;
    return (int64_t)st.st_size;
}

// Read up to `cap` bytes of the file at `path` into `buf`. _O_BINARY: no CRLF
// translation. Loops across short reads; stops at EOF or `cap`.
int64_t __mind_nerve_rt_read_file(int64_t path, int64_t path_len, int64_t buf, int64_t cap) {
    if (cap < 0) return MIND_NERVE_RT_ERR_IO;
    char cpath[MIND_NERVE_RT_PATH_MAX];
    if (mind_nerve_rt_copy_path(path, path_len, cpath, sizeof(cpath)) != 0) return MIND_NERVE_RT_ERR_IO;
    int fd = _open(cpath, _O_RDONLY | _O_BINARY);
    if (fd < 0) return MIND_NERVE_RT_ERR_IO;
    if (cap == 0 || buf == 0) { _close(fd); return 0; }
    unsigned char *p = (unsigned char *)(uintptr_t)buf;
    int64_t total = 0;
    while (total < cap) {
        int64_t left = cap - total;
        unsigned int want = left > MIND_NERVE_RT_IO_CHUNK ? MIND_NERVE_RT_IO_CHUNK : (unsigned int)left;
        int n = _read(fd, p + total, want);
        if (n < 0) {
            if (errno == EINTR) continue;
            _close(fd);
            return MIND_NERVE_RT_ERR_IO;
        }
        if (n == 0) break; // EOF
        total += (int64_t)n;
    }
    _close(fd);
    return total;
}

// Fill `len` bytes of `buf` with cryptographic-quality host entropy via the
// system-preferred RNG. RT_OK on full fill, RT_ERR_UNSUPPORTED on RNG failure.
int64_t __mind_nerve_rt_os_entropy(int64_t buf, int64_t len) {
    if (len < 0) return MIND_NERVE_RT_ERR_IO;
    if (len == 0) return MIND_NERVE_RT_OK;
    if (buf == 0) return MIND_NERVE_RT_ERR_IO;
    unsigned char *p = (unsigned char *)(uintptr_t)buf;
    int64_t total = 0;
    while (total < len) {
        int64_t left  = len - total;
        ULONG   chunk = left > MIND_NERVE_RT_IO_CHUNK ? (ULONG)MIND_NERVE_RT_IO_CHUNK : (ULONG)left;
        NTSTATUS s = BCryptGenRandom(NULL, (PUCHAR)(p + total), chunk,
                                     BCRYPT_USE_SYSTEM_PREFERRED_RNG);
        if (s < 0) return MIND_NERVE_RT_ERR_UNSUPPORTED; // NTSTATUS < 0 == failure
        total += (int64_t)chunk;
    }
    return MIND_NERVE_RT_OK;
}

// Read environment variable `name` (name_len UTF-8 bytes, no NUL) into `buf`
// (up to `cap` bytes). Returns the value length (>= 0) when set,
// RT_ERR_UNSUPPORTED when unset, RT_ERR_IO on a malformed name or a buffer
// smaller than the value. Read-only host state; the `mcp` subcommand uses it
// to resolve MIND_NERVE_RUNTIME_DIR without baking a path into the binary.
// Never on a hash-preimage path (getenv output is not attested). Byte-for-byte
// identical contract + implementation to nerve_rt_shims.c: the CRT getenv() is
// standard C (MSVCRT/MinGW), and for a normally-launched process its view of the
// environment matches the Win32 environment block. No NUL is written; MIND owns
// the returned length. This keeps the `mcp` path working on the PE target
// exactly as on Linux (the missing definition would otherwise be an undefined
// reference at link, or a call through an unbacked address at runtime).
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

// Terminate the process with `code`. The shims above use raw fds (unbuffered),
// so stdout/stderr are already durable. Does not return; i64 return keeps the
// C-ABI shape uniform.
int64_t __mind_nerve_rt_exit(int64_t code) {
    exit((int)code);
    return MIND_NERVE_RT_OK; // unreachable
}
