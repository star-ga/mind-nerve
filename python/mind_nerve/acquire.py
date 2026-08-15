"""mind-nerve acquire — find, vet, and install external skills/agents/MCP servers.

Pipeline: ``search`` (query curated sources) -> ``fetch`` (download into a
size- and file-count-capped quarantine dir under the runtime dir) ->
``vet`` (security_scan verdict + discovery license classification) ->
``install`` (copy into the hub, write a hash manifest, reindex the route
table, restart the routing daemon so hooked CLIs see the new entries).

Determinism rules honoured throughout: no wall-clock anywhere in manifests
or reports (a fetched git repo is identified by its commit SHA, never by a
timestamp), and every listing is sorted. All network I/O goes through one
injectable helper with a hard timeout; network errors degrade to an empty
result from that source, never a crash.

Trust model: ``install`` always reindexes with ``trusted=False`` — the
discovery license gate applies to acquired content even when the hub itself
is a first-party trust root.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from . import discovery, ensure
from .security_scan import scan_path

NETWORK_TIMEOUT = 30.0
MAX_QUARANTINE_BYTES = 25 * 1024 * 1024
MAX_QUARANTINE_FILES = 2000
_MAX_JSON_BYTES = 16 * 1024 * 1024
MANIFEST_NAME = ".mind-nerve-install.json"

HttpGet = Callable[[str, dict[str, str]], bytes]


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

DEFAULT_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "id": "anthropics-skills",
        "type": "github-repo",
        "repo": "anthropics/skills",
        "description": "Official Anthropic skills",
        "enabled": True,
    },
    {
        "id": "mcp-servers",
        "type": "github-repo",
        "repo": "modelcontextprotocol/servers",
        "description": "Official MCP servers",
        "enabled": True,
    },
    {
        "id": "superpowers",
        "type": "github-repo",
        "repo": "obra/superpowers",
        "description": "Community skills (obra/superpowers)",
        "enabled": True,
    },
    {
        "id": "wshobson-agents",
        "type": "github-repo",
        "repo": "wshobson/agents",
        "description": "Community agents (wshobson/agents)",
        "enabled": True,
    },
    {
        "id": "claude-code-templates",
        "type": "github-repo",
        "repo": "davila7/claude-code-templates",
        "description": "Community templates (davila7/claude-code-templates)",
        "enabled": True,
    },
    {
        "id": "mcp-registry",
        "type": "mcp-registry",
        "url": "https://registry.modelcontextprotocol.io/v0/servers",
        "description": "MCP registry API (JSON)",
        "enabled": True,
    },
    {
        "id": "glama-mcp",
        "type": "glama-mcp",
        "url": "https://glama.ai/api/mcp/v1/servers",
        "description": "Glama MCP directory (no-auth JSON API)",
        "enabled": True,
    },
    {
        "id": "smithery-mcp",
        "type": "smithery-mcp",
        "url": "https://registry.smithery.ai/servers",
        "description": "Smithery MCP registry (no-auth search API)",
        "enabled": True,
    },
    {
        "id": "github-search",
        "type": "github-search",
        "description": "Generic GitHub repository search fallback",
        "enabled": True,
    },
)


def _runtime_root(runtime_dir: str | Path | None = None) -> Path:
    """Resolve the acquire state root WITHOUT triggering the HF auto-seed.

    Acquire only needs a scratch area (quarantine/, acquire_sources.json);
    importing ``inference._resolve_runtime_dir`` would download the model
    runtime on a bare machine just to place a download. Explicit argument,
    then ``$MIND_NERVE_RUNTIME_DIR``, then the default user location —
    preferring the populated STARGA-local dash dir exactly as
    ``inference._default_user_runtime_dir`` does, so quarantine/prune land
    next to the table the daemon actually serves.
    """
    if runtime_dir is not None:
        return Path(str(runtime_dir)).expanduser()
    env_dir = os.environ.get("MIND_NERVE_RUNTIME_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    dash = Path.home() / ".local" / "share" / "mind-nerve-runtime"
    if (dash / "manifest.json").exists():
        return dash
    return Path.home() / ".local" / "share" / "mind-nerve" / "runtime"


def _default_hub_dir() -> Path:
    return Path(
        os.environ.get("MIND_NERVE_SOURCE_DIR", str(Path.home() / ".agents" / "skills-hub"))
    ).expanduser()


def load_sources(runtime_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Default registry plus optional user extension.

    ``<runtime_root>/acquire_sources.json`` holds a JSON list of source
    dicts in the same shape as ``DEFAULT_SOURCES``; entries whose ``id``
    matches a default override it (so a user can disable a default with
    ``{"id": "github-search", "enabled": false}``). Malformed files are
    ignored — the defaults always work.
    """
    sources: dict[str, dict[str, Any]] = {s["id"]: dict(s) for s in DEFAULT_SOURCES}
    cfg = _runtime_root(runtime_dir) / "acquire_sources.json"
    try:
        if cfg.is_file():
            extra = json.loads(cfg.read_text(encoding="utf-8"))
            if isinstance(extra, list):
                for entry in extra:
                    if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                        sources[entry["id"]] = entry
    except (OSError, ValueError):
        pass
    return [sources[k] for k in sorted(sources)]


# ---------------------------------------------------------------------------
# HTTP (single injectable choke point; tests mock this)
# ---------------------------------------------------------------------------


def _http_get(url: str, headers: dict[str, str] | None = None) -> bytes:
    hdrs = {"User-Agent": "mind-nerve-acquire/1"}
    hdrs.update(headers or {})
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=NETWORK_TIMEOUT) as resp:  # noqa: S310
        # Read cap must cover the largest legitimate payload — the 25 MB
        # quarantine byte cap, not the smaller JSON-API cap — or a valid
        # 16-25 MB tarball arrives truncated and fails to extract.
        return resp.read(MAX_QUARANTINE_BYTES + 1)


def _http_json(
    url: str, headers: dict[str, str] | None = None, http_get: HttpGet | None = None
) -> Any:
    data = (http_get or _http_get)(url, headers or {})
    if len(data) > _MAX_JSON_BYTES:
        raise ValueError("response exceeds size cap")
    return json.loads(data.decode("utf-8", "replace"))


# ---------------------------------------------------------------------------
# Candidates + search
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    name: str
    source: str  # source id
    url: str
    description: str = ""
    license: str = ""
    kind: str = "skill"

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def _query_tokens(query: str) -> list[str]:
    return sorted({t for t in re.split(r"[^a-z0-9]+", query.lower()) if t})


def _relevance(candidate: Candidate, tokens: list[str]) -> int:
    haystack = f"{candidate.name}\n{candidate.description}".lower()
    return sum(1 for t in tokens if t in haystack)


# Agent artifacts inside a curated repo: `agents/foo.md` / `subagents/foo.md`.
# These become PER-ARTIFACT candidates (raw file URL, kind="agent") — collapsing
# them to one parent-dir "skill" candidate flattened `agents/foo.md` to a
# root-level `foo.md` that discovery could never classify (audit, 2026-08 r4).
_AGENT_MD = re.compile(r"(^|/)(?:sub)?agents?/[^/]+\.md$", re.IGNORECASE)


def _search_github_repo(
    source: dict[str, Any], tokens: list[str], http_get: HttpGet
) -> list[Candidate]:
    """List a repo's git tree and surface every SKILL.md / agent .md dir."""
    repo = source["repo"]
    url = f"https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1"
    tree = _http_json(url, http_get=http_get)
    if not isinstance(tree, dict) or not isinstance(tree.get("tree"), list):
        return []
    seen: set[str] = set()
    out: list[Candidate] = []
    for entry in tree["tree"]:
        if not isinstance(entry, dict) or entry.get("type") != "blob":
            continue
        path = str(entry.get("path") or "")
        base = path.rsplit("/", 1)[-1]
        if base != "SKILL.md" and not base.endswith(".md"):
            continue
        if base != "SKILL.md" and _AGENT_MD.search(path):
            # One candidate per agent artifact; the raw file URL preserves
            # kind and relative path through fetch and indexing.
            stem = base[:-3]
            if path in seen or not stem:
                continue
            seen.add(path)
            out.append(
                Candidate(
                    name=stem,
                    source=source["id"],
                    url=f"https://raw.githubusercontent.com/{repo}/HEAD/{path}",
                    description=f"{source.get('description', '')}: {path}".strip(": "),
                    kind="agent",
                )
            )
            continue
        parent = path.rsplit("/", 1)[0] if "/" in path else ""
        name = parent.rsplit("/", 1)[-1] if parent else repo.rsplit("/", 1)[-1]
        if parent in seen or not name:
            continue
        seen.add(parent)
        web = (
            f"https://github.com/{repo}/tree/HEAD/{parent}"
            if parent
            else f"https://github.com/{repo}"
        )
        out.append(
            Candidate(
                name=name,
                source=source["id"],
                url=web,
                description=f"{source.get('description', '')}: {path}".strip(": "),
            )
        )
    return out


def _search_mcp_registry(
    source: dict[str, Any], query: str, tokens: list[str], http_get: HttpGet
) -> list[Candidate]:
    url = f"{source['url']}?search={urllib.parse.quote(query)}"
    payload = _http_json(url, http_get=http_get)
    servers = payload.get("servers") if isinstance(payload, dict) else None
    if not isinstance(servers, list):
        return []
    out: list[Candidate] = []
    for entry in servers:
        if not isinstance(entry, dict):
            continue
        body = entry.get("server") if isinstance(entry.get("server"), dict) else entry
        name = str(body.get("name") or "").strip()
        if not name:
            continue
        repo = body.get("repository")
        repo_url = repo.get("url") if isinstance(repo, dict) else None
        out.append(
            Candidate(
                name=name.rsplit("/", 1)[-1],
                source=source["id"],
                url=str(repo_url or body.get("url") or ""),
                description=str(body.get("description") or ""),
                kind="mcp",
            )
        )
    return out


def _search_github_search(source: dict[str, Any], query: str, http_get: HttpGet) -> list[Candidate]:
    headers: dict[str, str] = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"https://api.github.com/search/repositories?q={urllib.parse.quote(query)}&per_page=10"
    try:
        payload = _http_json(url, headers, http_get)
    except urllib.error.HTTPError as e:
        if e.code == 403:
            return []  # unauthenticated rate limit — degrade, don't crash
        raise
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []
    out: list[Candidate] = []
    for item in items:
        if not isinstance(item, dict) or not item.get("full_name"):
            continue
        lic = item.get("license")
        out.append(
            Candidate(
                name=str(item["full_name"]).rsplit("/", 1)[-1],
                source=source["id"],
                url=str(item.get("html_url") or ""),
                description=str(item.get("description") or ""),
                license=str(lic.get("spdx_id") or "") if isinstance(lic, dict) else "",
            )
        )
    return out


def _search_glama(source: dict[str, Any], query: str, http_get: HttpGet) -> list[Candidate]:
    """Glama MCP directory — no-auth JSON API (verified live 2026-08-10)."""
    url = f"{source['url']}?query={urllib.parse.quote(query)}"
    payload = _http_json(url, http_get=http_get)
    servers = payload.get("servers") if isinstance(payload, dict) else None
    if not isinstance(servers, list):
        return []
    out: list[Candidate] = []
    for s in servers:
        if not isinstance(s, dict) or not s.get("name"):
            continue
        repo = s.get("repository")
        # repository.url is installable (git); the glama page URL is not.
        repo_url = repo.get("url") if isinstance(repo, dict) else None
        lic = s.get("spdxLicense")
        lic_name = lic.get("name", "") if isinstance(lic, dict) else ""
        out.append(
            Candidate(
                name=str(s["name"]),
                source=source["id"],
                url=str(repo_url or s.get("url") or ""),
                description=str(s.get("description") or ""),
                license=re.sub(r"\s+License$", "", str(lic_name)),
                kind="mcp",
            )
        )
    return out


def _search_smithery(source: dict[str, Any], query: str, http_get: HttpGet) -> list[Candidate]:
    """Smithery registry — no-auth search API (verified live 2026-08-10).

    Smithery entries link a hosted page, not a repo; these candidates are
    discovery-only (install of a homepage URL fails fetch by design).
    """
    url = f"{source['url']}?q={urllib.parse.quote(query)}&page_size=10"
    payload = _http_json(url, http_get=http_get)
    servers = payload.get("servers") if isinstance(payload, dict) else None
    if not isinstance(servers, list):
        return []
    out: list[Candidate] = []
    for s in servers:
        if not isinstance(s, dict):
            continue
        name = str(s.get("qualifiedName") or s.get("displayName") or "").strip()
        if not name:
            continue
        out.append(
            Candidate(
                name=name,
                source=source["id"],
                url=str(s.get("homepage") or ""),
                description=str(s.get("description") or ""),
                kind="mcp",
            )
        )
    return out


def _search_local_dir(source: dict[str, Any], tokens: list[str]) -> list[Candidate]:
    """Offline source: every immediate subdir of ``path`` is a package."""
    root = Path(str(source.get("path") or "")).expanduser()
    if not root.is_dir():
        return []
    out: list[Candidate] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        desc = ""
        lic = ""
        skill = child / "SKILL.md"
        try:
            if skill.is_file():
                fm = discovery._parse_frontmatter(
                    skill.read_text(encoding="utf-8", errors="replace")[:8192]
                )
                desc = fm.get("description", "")
                lic = fm.get("license", "")
        except OSError:
            continue
        out.append(
            Candidate(
                name=child.name,
                source=source["id"],
                url=str(child),
                description=desc,
                license=lic,
            )
        )
    return out


def search(
    query: str,
    *,
    sources: list[dict[str, Any]] | None = None,
    runtime_dir: str | Path | None = None,
    http_get: HttpGet | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Query every enabled source; per-source failures degrade to []."""
    tokens = _query_tokens(query)
    errors: dict[str, str] = {}
    found: list[Candidate] = []
    for source in sources if sources is not None else load_sources(runtime_dir):
        if not source.get("enabled", True):
            continue
        sid = str(source.get("id") or "?")
        try:
            stype = source.get("type")
            if stype == "github-repo":
                found.extend(_search_github_repo(source, tokens, http_get or _http_get))
            elif stype == "mcp-registry":
                found.extend(_search_mcp_registry(source, query, tokens, http_get or _http_get))
            elif stype == "glama-mcp":
                found.extend(_search_glama(source, query, http_get or _http_get))
            elif stype == "smithery-mcp":
                found.extend(_search_smithery(source, query, http_get or _http_get))
            elif stype == "github-search":
                found.extend(_search_github_search(source, query, http_get or _http_get))
            elif stype == "local-dir":
                found.extend(_search_local_dir(source, tokens))
        except Exception as e:  # noqa: BLE001 — one bad source must not sink search
            errors[sid] = str(e)
    # Rank: relevance desc, then (source, name) for a deterministic order.
    ranked = sorted(found, key=lambda c: (-_relevance(c, tokens), c.source, c.name.lower()))
    return {
        "query": query,
        "candidates": [c.as_dict() for c in ranked[:limit]],
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Fetch into quarantine
# ---------------------------------------------------------------------------


def _redact_url(url: str) -> str:
    """Provenance-safe form of a source URL: userinfo and query/fragment
    stripped. Credential-bearing clone URLs and signed-download query
    strings must never persist in manifests, error messages, or CLI output
    (audit finding, 2026-08 r4). The OPERATIONAL url keeps its secrets;
    only the recorded/printed form is redacted. Local paths (no scheme)
    pass through untouched.
    """
    parsed = urllib.parse.urlsplit(url)
    if not parsed.scheme:
        return url
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urllib.parse.urlunsplit((parsed.scheme, host, parsed.path, "", ""))


class FetchError(RuntimeError):
    """Raised when a candidate cannot be fetched within the caps."""


def _check_caps(total_bytes: int, total_files: int, max_bytes: int, max_files: int) -> None:
    if total_files > max_files:
        raise FetchError(f"file-count cap exceeded ({total_files} > {max_files})")
    if total_bytes > max_bytes:
        raise FetchError(f"size cap exceeded ({total_bytes} > {max_bytes} bytes)")


def _copy_local(src: Path, dest: Path, max_bytes: int, max_files: int) -> None:
    total_bytes = 0
    total_files = 0
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if set(rel.parts) & discovery.SKIP_DIRS:
            continue
        if p.is_symlink():
            # Preserve the link (never follow it) so the security scanner can
            # judge it: escapes and targets in skipped dirs are FAILed at vet
            # time. Dropping it here would silently install an empty package.
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            os.symlink(os.readlink(p), dest / rel)
            continue
        if p.is_dir():
            (dest / rel).mkdir(parents=True, exist_ok=True)
            continue
        total_files += 1
        total_bytes += p.stat().st_size
        _check_caps(total_bytes, total_files, max_bytes, max_files)
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest / rel)


def _check_member_safe(m: tarfile.TarInfo) -> None:
    """Fail-closed checks for one tarball member (any Python version)."""
    name = m.name
    parts = re.split(r"[/\\]", name)
    if (
        name.startswith(("/", "\\"))
        or re.match(r"^[A-Za-z]:[\\/]", name)  # Windows drive-qualified
        or ".." in parts
    ):
        raise FetchError(f"archive member escapes extraction root: {name!r}")
    # Only regular files, dirs, and safe links may be extracted — on any
    # Python version. A FIFO member would block a reader forever, and
    # device nodes have no legitimate place in a skill package; the
    # tarfile data_filter that would catch these only exists on 3.12+
    # (3.10.12/3.11.4 backports), so we cannot rely on it at our floor.
    if not (m.isfile() or m.isdir() or m.issym() or m.islnk()):
        raise FetchError(f"unsupported archive member type: {name!r}")
    if (m.issym() or m.islnk()) and (
        m.linkname.startswith(("/", "\\"))
        or re.match(r"^[A-Za-z]:[\\/]", m.linkname)
        or ".." in re.split(r"[/\\]", m.linkname)
    ):
        raise FetchError(f"archive link escapes extraction root: {name!r}")


def _skipped_member(name: str) -> bool:
    """True when a member lives under a skipped dir (dist/, node_modules/, …).

    The scanner prunes those dirs, so content there is NEVER vetted — it
    must not reach quarantine at all, exactly as ``_copy_local`` filters it
    on the local fetch path (audit finding, 2026-08 r3).
    """
    return bool(set(re.split(r"[/\\]", name)) & discovery.SKIP_DIRS)


# tarfile's extraction ``filter`` (and ``data_filter``) landed in 3.12 and
# was backported to 3.10.12 / 3.11.4 — but pyproject floors at 3.10, where
# early patch releases lack the kwarg entirely. Feature-detect instead of
# version-sniffing; on the unpatched floor our own per-member escape
# checks + byte/file caps still guard the extraction.
_TAR_FILTER: str | None = "data" if hasattr(tarfile, "data_filter") else None


def _fetch_tarball(url: str, dest: Path, max_bytes: int, max_files: int, http_get: HttpGet) -> None:
    data = http_get(url, {})
    if len(data) > max_bytes:
        raise FetchError(f"size cap exceeded ({len(data)} > {max_bytes} bytes)")
    blob = dest / ".payload.tar.gz"
    blob.write_bytes(data)
    try:
        with tarfile.open(blob) as tf:
            members: list[tarfile.TarInfo] = []
            files = 0
            uncompressed = 0
            # Stream members (tarfile iteration) instead of getmembers(): the
            # caps trip DURING the read, so a bomb of millions of tiny headers
            # is refused before its whole member list is materialised.
            for m in tf:
                _check_member_safe(m)
                if _skipped_member(m.name):
                    continue
                members.append(m)
                if m.isfile():
                    files += 1
                    if files > max_files:
                        raise FetchError(f"file-count cap exceeded ({files} > {max_files})")
                    # Cap the UNCOMPRESSED payload: a tar bomb is tiny on the
                    # wire and huge on disk. Member size headers are
                    # authoritative for extraction, so the running sum is an
                    # exact upper bound.
                    uncompressed += m.size
                    if uncompressed > max_bytes:
                        raise FetchError(
                            f"uncompressed size cap exceeded ({uncompressed} > {max_bytes} bytes)"
                        )
            kwargs = {"filter": _TAR_FILTER} if _TAR_FILTER is not None else {}
            tf.extractall(dest, members=members, **kwargs)
    except tarfile.TarError as e:
        raise FetchError(f"not a readable tarball: {e}") from e
    except KeyError as e:
        # A hard link whose target member was filtered out (skipped dir).
        raise FetchError(f"archive link target unavailable: {e}") from e
    finally:
        blob.unlink(missing_ok=True)
    # Strip the single top-level "<repo>-<sha>/" wrapper GitHub adds.
    children = list(dest.iterdir())
    if len(children) == 1 and children[0].is_dir():
        inner = children[0]
        for item in sorted(inner.iterdir()):
            shutil.move(str(item), dest / item.name)
        inner.rmdir()


def _tree_bytes(root: Path) -> int:
    """Total size of regular files under *root* (symlinks not followed)."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                if not p.is_symlink():
                    total += p.stat().st_size
            except OSError:
                continue
    return total


def _clone_monitored(argv: list[str], dest: Path, max_bytes: int) -> str | None:
    """Run git clone, killing it once *dest* grows past *max_bytes*.

    Measuring caps only AFTER the clone lets a huge repo fill the disk
    first (audit finding, 2026-08 r3) — the destination is polled while the
    clone runs. The poll counts ``.git`` too, so the effective cap is
    slightly tighter than the post-clone measure; fail-closed on purpose.
    Returns the trimmed stderr on a non-zero exit, None on success.
    """
    proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
        argv,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 120
    try:
        while proc.poll() is None:
            if _tree_bytes(Path(dest)) > max_bytes:
                proc.kill()
                proc.wait()
                raise FetchError(f"size cap exceeded during git clone (> {max_bytes} bytes)")
            if time.monotonic() > deadline:
                proc.kill()
                proc.wait()
                raise FetchError("git clone timed out after 120s")
            time.sleep(0.1)
        stderr = proc.stderr.read() if proc.stderr is not None else b""
        if proc.returncode:
            return stderr.decode("utf-8", "replace").strip()[:200]
        return None
    finally:
        if proc.poll() is None:
            proc.kill()


def _git_clone(
    url: str, dest: Path, ref: str | None = None, max_bytes: int | None = None
) -> str | None:
    """``git clone --depth 1`` into *dest*; returns the resolved commit SHA.

    ``ref`` (branch or tag) is honoured with ``--branch``; a raw commit SHA
    ref fails loudly (shallow clone cannot address arbitrary SHAs without a
    server round-trip we do not make). ``HEAD``/None clones the default
    branch. The returned SHA is always the ACTUAL checked-out commit, read
    back with rev-parse — never the requested ref. With ``max_bytes`` set
    the clone is aborted mid-flight once the destination exceeds the cap.
    """
    for value, label in ((url, "URL"), (ref, "ref")):
        if value and value.startswith("-"):
            # A leading-dash value is parsed by git as more options
            # (e.g. ``--upload-pack=...``) — refuse before exec.
            raise FetchError(f"refusing git {label} that looks like an option: {value!r}")
    argv = ["git", "clone", "--depth", "1"]
    if ref and ref != "HEAD":
        argv += ["--branch", ref]
    argv += ["--", url, str(dest)]
    if max_bytes is None:
        try:
            subprocess.run(  # noqa: S603 — fixed argv, no shell
                argv,
                check=True,
                capture_output=True,
                timeout=120,
            )
        except subprocess.CalledProcessError as e:
            # git echoes the full URL (userinfo included) in its errors —
            # scrub the operational URL from the detail before propagating.
            detail = (
                e.stderr.decode("utf-8", "replace").strip()[:200].replace(url, _redact_url(url))
            )
            raise FetchError(
                f"git clone failed for {_redact_url(url)!r}"
                + (f" at ref {ref!r}" if ref and ref != "HEAD" else "")
                + f": {detail}"
            ) from e
    else:
        stderr = _clone_monitored(argv, Path(dest), max_bytes)
        if stderr is not None:
            detail = stderr.replace(url, _redact_url(url))
            raise FetchError(
                f"git clone failed for {_redact_url(url)!r}"
                + (f" at ref {ref!r}" if ref and ref != "HEAD" else "")
                + f": {detail}"
            )
    proc = subprocess.run(  # noqa: S603
        ["git", "-C", str(dest), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return proc.stdout.strip() or None


_GITHUB_TREE_URL = re.compile(
    r"^https://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)"
    r"/tree/(?P<ref>[^/]+)(?:/(?P<subdir>.*?))?/?$"
)

# Single-file raw URLs — the shape per-artifact agent candidates carry.
_RAW_GITHUB_URL = re.compile(
    r"^https://raw\.githubusercontent\.com/(?P<owner>[A-Za-z0-9_.-]+)"
    r"/(?P<repo>[A-Za-z0-9_.-]+)/(?P<ref>[^/]+)/(?P<path>.+)$"
)


def _fetch_raw_file(match: re.Match[str], dest: Path, max_bytes: int, http_get: HttpGet) -> None:
    """Fetch ONE raw file (per-artifact agent candidates).

    The parent dir NAME is preserved (``qdir/agents/foo.md``) so discovery's
    kind-by-path classification still sees ``agents/`` — flattening to
    ``qdir/foo.md`` made acquired agents unroutable (audit finding,
    2026-08 r4 / codex#11).
    """
    relpath = match["path"]
    parts = relpath.split("/")
    if relpath.startswith("/") or ".." in parts:
        raise FetchError(f"raw path escapes the repository: {relpath!r}")
    data = http_get(match.group(0), {})
    if len(data) > max_bytes:
        raise FetchError(f"size cap exceeded ({len(data)} > {max_bytes} bytes)")
    target = dest / parts[-2] / parts[-1] if len(parts) > 1 else dest / parts[-1]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def _measure_caps(root: Path, max_bytes: int, max_files: int) -> None:
    """Enforce the quarantine caps on an already-materialised tree (git clone).

    The tarball path caps mid-extraction; a clone has no such hook, so we
    measure after the fact and fail closed (the fetch caller wipes the
    quarantine dir on any raise). Symlinks are not followed.
    """
    total_bytes = 0
    total_files = 0
    for p in sorted(root.rglob("*")):
        if p.is_symlink() or not p.is_file():
            continue
        total_files += 1
        total_bytes += p.stat().st_size
        _check_caps(total_bytes, total_files, max_bytes, max_files)


def _purge_skipped_dirs(root: Path) -> None:
    """Remove skipped dirs (dist/, node_modules/, …) from a fetched tree.

    The scanner never walks those dirs, so anything left in them would be
    installed unvetted — the tarball and local fetch paths already filter
    them; a plain git clone does not, unless we purge here (audit finding,
    2026-08 r3). A symlink NAMED like a skipped dir is left in place
    (rmtree refuses symlinks) — the scanner judges it at vet time.
    """
    for dirpath, dirnames, _filenames in os.walk(root):
        for d in list(dirnames):
            if d in discovery.SKIP_DIRS:
                target = Path(dirpath) / d
                if not target.is_symlink():
                    shutil.rmtree(target, ignore_errors=True)
                dirnames.remove(d)


def fetch(
    candidate: Candidate | dict[str, Any],
    *,
    runtime_dir: str | Path | None = None,
    max_bytes: int = MAX_QUARANTINE_BYTES,
    max_files: int = MAX_QUARANTINE_FILES,
    http_get: HttpGet | None = None,
    allow_local: bool = False,
) -> dict[str, Any]:
    """Fetch *candidate* into ``<runtime_root>/quarantine/<sha>/``.

    Supports local paths / ``file://`` URLs (copy), git URLs (shallow
    clone, records the commit SHA), GitHub ``/tree/<ref>/<subdir>`` URLs
    (clone the parent repo, take *subdir* as the package root), and http(s)
    tarballs (capped stream, escape-checked extraction). The quarantine dir
    name is ``sha256(url)[:16]`` — content of the address, no clock.

    ``allow_local`` marks an OPERATOR-TYPED target (the CLI install path):
    only then are local directories / ``file://`` URLs honoured and plain
    ``http://`` tolerated. Registry-sourced candidates (the default) fetch
    over ``https://`` only — candidate URLs come verbatim from third-party
    registries, and ``"url": "/home/victim/.ssh"`` must never become a
    local copy (audit finding, 2026-08 r3).
    """
    if not isinstance(candidate, Candidate):
        candidate = Candidate(
            **{k: v for k, v in candidate.items() if k in Candidate.__dataclass_fields__}
        )
    url = candidate.url
    local = url.startswith("file://")
    local_path = Path(urllib.parse.unquote(url[7:])) if local else Path(url)
    tree = _GITHUB_TREE_URL.match(url)
    if not allow_local and (local or local_path.is_dir() or not url.startswith("https://")):
        raise FetchError(
            "registry-sourced candidates fetch over https:// only; local dirs "
            f"and file:// require an operator-typed target: {_redact_url(url)!r}"
        )
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    qdir = _runtime_root(runtime_dir) / "quarantine" / key
    if qdir.exists():
        shutil.rmtree(qdir)
    qdir.mkdir(parents=True)

    commit_sha: str | None = None
    repo_url: str | None = None
    subdir: str | None = None
    try:
        if local or local_path.is_dir():
            if not local_path.is_dir():
                raise FetchError(f"local source {local_path} is not a directory")
            _copy_local(local_path, qdir, max_bytes, max_files)
        elif tree:
            # Search candidates point at a subdirectory of a repo; the
            # package root for vet/install is that subdir, not the repo.
            repo_url = f"https://github.com/{tree['owner']}/{tree['repo']}"
            ref = tree.group("ref")
            subdir = tree.group("subdir") or ""
            if subdir:
                parts = subdir.split("/")
                if subdir.startswith("/") or ".." in parts:
                    # A crafted subdir must never reach outside the clone —
                    # it would copy arbitrary local content while the
                    # manifest records GitHub provenance.
                    raise FetchError(f"subdir escapes the repository: {subdir!r}")
            scratch = qdir.parent / f".clone-{key}"
            if scratch.exists():
                shutil.rmtree(scratch)
            try:
                commit_sha = _git_clone(repo_url, scratch, ref=ref, max_bytes=max_bytes)
                pkg_root = scratch / subdir if subdir else scratch
                if subdir and not pkg_root.is_dir():
                    raise FetchError(f"subdir {subdir!r} not found in {repo_url}")
                _copy_local(pkg_root, qdir, max_bytes, max_files)
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
        elif url.endswith(".git") or re.match(r"^https://github\.com/[^/]+/[^/]+/?$", url):
            commit_sha = _git_clone(url, qdir, max_bytes=max_bytes)
            gitdir = qdir / ".git"
            if gitdir.exists():
                shutil.rmtree(gitdir)
            _purge_skipped_dirs(qdir)
            _measure_caps(qdir, max_bytes, max_files)
        elif url.startswith(("http://", "https://")):
            raw = _RAW_GITHUB_URL.match(url)
            if raw is not None:
                _fetch_raw_file(raw, qdir, max_bytes, http_get or _http_get)
            else:
                _fetch_tarball(url, qdir, max_bytes, max_files, http_get or _http_get)
        else:
            raise FetchError(f"unsupported source URL: {_redact_url(url)!r}")
    except Exception:
        shutil.rmtree(qdir, ignore_errors=True)
        raise

    total_bytes = 0
    total_files = 0
    for p in qdir.rglob("*"):
        if p.is_file():
            total_files += 1
            total_bytes += p.stat().st_size
    return {
        "quarantine_dir": str(qdir),
        "commit_sha": commit_sha,
        "repo_url": repo_url,
        "subdir": subdir,
        "files": total_files,
        "bytes": total_bytes,
    }


# ---------------------------------------------------------------------------
# Vet
# ---------------------------------------------------------------------------


def vet(package_dir: str | Path, *, use_clamav: bool = False) -> dict[str, Any]:
    """Security scan + license classification for a fetched package."""
    package_dir = Path(package_dir)
    report = scan_path(package_dir, use_clamav=use_clamav)
    licenses: list[dict[str, Any]] = []
    for item in discovery._walk_dir(package_dir, source_repo="acquire"):
        licenses.append(
            {
                "path": item["source_path"],
                "name": item["name"],
                "kind": item["kind"],
                "bucket": item["_license_bucket"],
                "reasons": item["_license_reasons"],
            }
        )
    licenses.sort(key=lambda lic: lic["path"])
    commercial = [lic for lic in licenses if lic["bucket"] == "commercial_risk"]
    verdict = report.verdict
    if commercial and verdict != "FAIL":
        verdict = "FAIL"
    return {
        "verdict": verdict,
        "security": report.as_dict(),
        "licenses": licenses,
        "commercial_risk": [lic["path"] for lic in commercial],
    }


# ---------------------------------------------------------------------------
# Install / list / remove
# ---------------------------------------------------------------------------


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "acquired-skill"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(
    target: Path,
    candidate: Candidate,
    commit_sha: str | None,
    verdict: str,
    repo_url: str | None = None,
    subdir: str | None = None,
) -> dict[str, Any]:
    files = []
    for p in sorted(target.rglob("*")):
        if p.is_file() and p.name != MANIFEST_NAME:
            files.append(
                {
                    "path": str(p.relative_to(target)),
                    "sha256": _sha256_file(p),
                    "size": p.stat().st_size,
                }
            )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "name": target.name,
        "source": candidate.source,
        # Redacted provenance: the operational URL may carry userinfo or a
        # signed query string; neither persists (audit finding, 2026-08 r4).
        "url": _redact_url(candidate.url),
        "commit_sha": commit_sha,
        "verdict": verdict,
        "files": files,
    }
    if repo_url is not None:
        # Tree-URL install: the repo cloned and the package root within it.
        manifest["repo_url"] = _redact_url(repo_url)
        manifest["subdir"] = subdir or ""
    (target / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _claim_name(hub: Path, candidate: Candidate) -> str:
    """Slugified unique hub name; reinstall of the same URL reuses its dir."""
    base = _slugify(candidate.name)
    for i in range(1, 100):
        name = base if i == 1 else f"{base}-{i}"
        target = hub / name
        if not target.exists():
            return name
        manifest = target / MANIFEST_NAME
        if manifest.is_file():
            try:
                # Manifests store the REDACTED url — compare redacted forms.
                if json.loads(manifest.read_text(encoding="utf-8")).get("url") == _redact_url(
                    candidate.url
                ):
                    return name  # reinstall over our own previous install
            except ValueError:
                pass
    raise FetchError(f"could not claim a hub name for {candidate.name!r}")


def install(
    candidate: Candidate | dict[str, Any],
    *,
    hub_dir: str | Path | None = None,
    runtime_dir: str | Path | None = None,
    accept_warnings: bool = False,
    use_clamav: bool = False,
    http_get: HttpGet | None = None,
    rescan: Callable[..., dict[str, Any]] | None = None,
    restart: Callable[[], dict[str, Any]] | None = None,
    register_mcp: bool = False,
    mcp_targets: dict[str, str | Path] | None = None,
    allow_local: bool = False,
) -> dict[str, Any]:
    """fetch -> vet -> copy into hub -> manifest -> reindex -> restart daemon.

    A FAIL verdict (security or license) leaves the package in quarantine
    and returns ``{"installed": False, ...}``; nothing touches the hub.

    ``register_mcp`` is opt-in: only then, and only for ``kind="mcp"``
    candidates, the installed package's server entry point is written into
    the active CLIs' MCP configs (see :func:`register_mcp_server`). Vetting
    content is not license to execute a server, so the default is OFF.

    ``allow_local`` is forwarded to :func:`fetch` — set it only for
    operator-typed targets (the CLI install path).
    """
    if not isinstance(candidate, Candidate):
        candidate = Candidate(
            **{k: v for k, v in candidate.items() if k in Candidate.__dataclass_fields__}
        )
    fetched = fetch(
        candidate,
        runtime_dir=runtime_dir,
        http_get=http_get,
        allow_local=allow_local,
    )
    qdir = Path(fetched["quarantine_dir"])
    report = vet(qdir, use_clamav=use_clamav)
    verdict = report["verdict"]
    if verdict == "FAIL":
        return {
            "installed": False,
            "reason": "vet-failed",
            "quarantine_dir": str(qdir),
            "vet": report,
        }
    if verdict == "WARN" and not accept_warnings:
        return {
            "installed": False,
            "reason": "warnings-present",
            "quarantine_dir": str(qdir),
            "vet": report,
        }

    hub = Path(hub_dir).expanduser() if hub_dir else _default_hub_dir()
    hub.mkdir(parents=True, exist_ok=True)
    name = _claim_name(hub, candidate)
    target = hub / name
    if target.exists():
        shutil.rmtree(target)  # our own previous install (manifest-checked)
    try:
        # symlinks=True: every link in quarantine already survived the
        # vetter (in-package, non-dangling, never into a skipped dir), so it
        # stays a link — dereferencing here (symlinks=False) is what turned
        # a vetted DIRECTORY symlink into real copies of host secrets in
        # the hub (audit finding, 2026-08 r3). Skipped-dir names are excluded
        # outright: their content is never scanned, so it is never
        # installed, whether real dir or symlink.
        shutil.copytree(
            qdir,
            target,
            symlinks=True,
            ignore=shutil.ignore_patterns(*sorted(discovery.SKIP_DIRS)),
        )
    except Exception as e:
        # Never leave a manifest-less partial dir behind: remove() refuses
        # to delete dirs without a manifest, so a torn copy would be stuck
        # in the hub forever.
        shutil.rmtree(target, ignore_errors=True)
        raise FetchError(f"install copy failed: {e}") from e
    manifest = _write_manifest(
        target,
        candidate,
        fetched["commit_sha"],
        verdict,
        repo_url=fetched.get("repo_url"),
        subdir=fetched.get("subdir"),
    )

    out: dict[str, Any] = {
        "installed": True,
        "name": name,
        "hub_dir": str(hub),
        "verdict": verdict,
        "vet": report,
        "manifest": manifest,
    }

    # Reinstall with changed content: drop the package's existing route rows
    # BEFORE reindexing. discovery.scan dedups by content sha and is
    # append-only, so without this source-path upsert the old sha/embedding
    # rows keep pointing at the replaced bytes (audit finding, 2026-08 r4 /
    # codex#10). Both table rewrites are atomic (jsonl+npy lockstep); a scan
    # failure after the prune is reported loudly via reindex_error and heals
    # on re-run. Uses acquire's own root resolver — never the HF-seeding one.
    try:
        out["pruned_stale"] = _prune_routes_for_path(_runtime_root(runtime_dir), target).get(
            "pruned", 0
        )
    except Exception as e:  # noqa: BLE001 — prune failure must not sink install
        out["prune_error"] = str(e)

    scan_fn = rescan if rescan is not None else discovery.scan
    try:
        out["reindex"] = scan_fn(
            target,
            source_repo=f"acquire:{candidate.source}",
            trusted=False,
            runtime_dir=str(runtime_dir)
            if runtime_dir is not None
            else discovery._DEFAULT_RUNTIME_DIR,
        )
        out["reindexed"] = True
    except Exception as e:  # noqa: BLE001 — install stands, but the CLI must
        # exit non-zero: without the reindex the daemon never routes here.
        out["reindexed"] = False
        out["reindex_error"] = str(e)

    # A "successful" reindex that adds ZERO routes means the package is
    # unusable through the router: the license gate excluded everything, or
    # there was no routable content at all. Surface it instead of silently
    # succeeding — a benign reinstall (all rows already indexed) stays quiet.
    if out.get("reindexed"):
        reindex = out["reindex"]
        skipped = reindex.get("skipped", {})
        excluded = skipped.get("license_excluded", 0) + skipped.get("unknown_excluded", 0)
        if reindex.get("added", 0) == 0 and not (
            skipped.get("already_indexed", 0) > 0 and excluded == 0
        ):
            out["routes_added"] = 0
            out["warning"] = (
                "no-routable-content: reindex added 0 routes "
                f"(license/unknown-excluded: {excluded}); the daemon will "
                "never route to this package"
            )

    restart_fn = restart if restart is not None else ensure.restart_daemon
    try:
        out["daemon_restart"] = restart_fn()
    except Exception as e:  # noqa: BLE001
        out["daemon_restart_error"] = str(e)

    if register_mcp:
        if candidate.kind != "mcp":
            out["mcp_registration"] = {
                "registered": False,
                "error": f"--register-mcp requires kind='mcp', got {candidate.kind!r}",
            }
        else:
            out["mcp_registration"] = register_mcp_server(name, target, targets=mcp_targets)
    return out


# ---------------------------------------------------------------------------
# MCP-pair registration (--register-mcp)
# ---------------------------------------------------------------------------
#
# The skill teaches the model the tools exist; the MCP config makes the tools
# real. Registration writes the canonical ``mcpServers`` JSON entry (the same
# shape the TS installer emits) into each active CLI's config file — atomically
# (tmp + os.replace), only into files that already exist (an absent config
# means the CLI is not installed), and only when the package carries a
# recognizable server entry point. TOML-target CLIs (codex/vibe/grok/kimi)
# are reported as skipped: the stdlib has no TOML *writer*, and a half-edit
# of a TOML config is worse than none.

# tomllib is stdlib from 3.11; the package floors at 3.10. Guarded: on 3.10
# the pyproject entry-point detection is simply unavailable.
try:
    import tomllib as _tomllib
except ImportError:  # pragma: no cover - exercised only on Python 3.10
    _tomllib = None  # type: ignore[assignment]

MCP_JSON_TARGETS: dict[str, str] = {
    "claude-code": "~/.claude/settings.json",
    "cursor": "~/.cursor/mcp.json",
    "gemini": "~/.gemini/settings.json",
    "qwen": "~/.qwen/settings.json",
}


def _atomic_replace_preserving_mode(path: Path, content: str) -> None:
    """Exclusive-temp + fsync + os.replace, preserving the file's mode.

    A fresh umask-derived inode would widen a 0600 secret-bearing CLI config
    to 0644, and a fixed-name temp left behind after a failure would sit
    there with those wide permissions (audit finding, 2026-08 r4 / codex#8).
    """
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp_name, stat.S_IMODE(os.stat(path).st_mode))
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _mcp_entry_point(pkg_dir: Path) -> dict[str, Any] | None:
    """Recognize the server entry point of an installed package.

    Order: pyproject.toml [project.scripts] -> package.json bin. Returns the
    canonical ``{"command": ..., "args": [...]}`` entry, or None (refuse to
    register).

    Only LOCAL entry points are eligible: both shapes resolve to content
    inside the vetted package directory. A ``server.json`` (MCP registry
    manifest) names an npm/PyPI artifact that was never quarantined or
    scanned — writing ``npx -y <ident>`` / ``uvx <ident>`` into a CLI config
    would execute it sight unseen, and an unsanitized identifier (leading
    ``-``, whitespace) is option injection into npx/uvx. Registry references
    are therefore refused outright (audit finding, 2026-08 r3).
    """
    pkg_dir = pkg_dir.resolve()
    # 1. pyproject.toml with a console script -> FLEET-standard uvx argv
    pyproject = pkg_dir / "pyproject.toml"
    if pyproject.is_file() and _tomllib is not None:
        try:
            data = _tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            data = None
        if isinstance(data, dict):
            scripts = data.get("project", {}).get("scripts")
            if isinstance(scripts, dict) and scripts:
                script = sorted(scripts)[0]
                return {"command": "uvx", "args": ["--from", str(pkg_dir), script]}

    # 3. package.json bin -> direct node argv on the installed file
    package_json = pkg_dir / "package.json"
    if package_json.is_file():
        try:
            data = json.loads(package_json.read_text(encoding="utf-8"))
        except ValueError:
            data = None
        if isinstance(data, dict):
            bin_field = data.get("bin")
            bin_path: str | None = None
            if isinstance(bin_field, str):
                bin_path = bin_field
            elif isinstance(bin_field, dict) and bin_field:
                bin_path = str(bin_field[sorted(bin_field)[0]])
            if bin_path:
                target = (pkg_dir / bin_path).resolve()
                # The bin must live INSIDE the installed package.
                if target == pkg_dir or pkg_dir in target.parents:
                    return {"command": "node", "args": [str(target)]}
    return None


def register_mcp_server(
    server_name: str,
    package_dir: str | Path,
    *,
    targets: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Write the server's ``mcpServers`` entry into each active CLI config.

    Fail-closed: no recognizable entry point -> ``registered: False`` with a
    clear message, and no config file is touched. Writes are atomic
    (tmp + os.replace); a non-object or absent config is skipped, never
    clobbered. The entry carries ``x-managed-by: mind-nerve-acquire`` so a
    future unregister can tell our rows from hand-written ones.
    """
    pkg_dir = Path(package_dir)
    entry = _mcp_entry_point(pkg_dir)
    if entry is None:
        return {
            "registered": False,
            "error": "no recognizable server entry point (need a LOCAL one: "
            "pyproject.toml [project.scripts] or package.json bin — server.json "
            "registry identifiers are refused: they name artifacts that were "
            "never vetted)",
        }
    entry["x-managed-by"] = "mind-nerve-acquire"

    table = MCP_JSON_TARGETS if targets is None else targets
    results: dict[str, str] = {}
    for cli in sorted(table):
        cfg_path = Path(table[cli]).expanduser()
        if not cfg_path.is_file():
            results[cli] = "skipped (no config file)"
            continue
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if not isinstance(cfg, dict):
                raise ValueError("config root is not a JSON object")
            servers = cfg.setdefault("mcpServers", {})
            if not isinstance(servers, dict):
                raise ValueError("mcpServers is not a JSON object")
            servers[server_name] = entry
            _atomic_replace_preserving_mode(cfg_path, json.dumps(cfg, indent=2) + "\n")
            results[cli] = "registered"
        except (OSError, ValueError) as e:
            results[cli] = f"error: {e}"
    registered_any = any(v == "registered" for v in results.values())
    out: dict[str, Any] = {
        "registered": registered_any,
        "entry": entry,
        "targets": results,
    }
    if not registered_any:
        out["error"] = "no MCP-capable CLI config accepted the entry"
    return out


def list_installed(hub_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Manifests of every acquire-installed package in the hub."""
    hub = Path(hub_dir).expanduser() if hub_dir else _default_hub_dir()
    out: list[dict[str, Any]] = []
    if not hub.is_dir():
        return out
    for child in sorted(hub.iterdir()):
        manifest_path = child / MANIFEST_NAME
        if not child.is_dir() or not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except ValueError:
            manifest = {"name": child.name, "error": "corrupt manifest"}
        manifest.setdefault("name", child.name)
        manifest["hub_dir"] = str(child)
        out.append(manifest)
    return out


def _prune_routes_for_path(runtime_dir: Path, removed_dir: Path) -> dict[str, Any]:
    """Drop route-table rows whose source_path lives under *removed_dir*.

    The jsonl metadata and the row-aligned npy embeddings are rewritten
    together via discovery's atomic save — pruning one without the other
    silently mis-scores every route after the deletion.
    """
    emb_path = runtime_dir / "route_table.npy"
    meta_path = runtime_dir / "route_table.jsonl"
    if not (emb_path.is_file() and meta_path.is_file()):
        return {"pruned": 0, "reason": "no route table"}
    emb, meta = discovery._load_table(runtime_dir)
    prefix = str(removed_dir)
    # Exact-or-boundary match: "/hub/pdf-tools-2/x" startswith "/hub/pdf-tools"
    # as a string but is NOT under it — a bare startswith prune would delete a
    # sibling package's routes when its name extends the removed one.
    boundary = prefix + os.sep

    def _under_removed(source_path: str) -> bool:
        return source_path == prefix or source_path.startswith(boundary)

    keep = [i for i, m in enumerate(meta) if not _under_removed(str(m.get("source_path", "")))]
    pruned = len(meta) - len(keep)
    if pruned:
        discovery._save_table_atomic(runtime_dir, emb[keep], [meta[i] for i in keep])
        discovery.load_default_runtime.cache_clear()  # type: ignore[attr-defined]
    return {"pruned": pruned}


def remove(
    name: str,
    *,
    hub_dir: str | Path | None = None,
    runtime_dir: str | Path | None = None,
    restart: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Delete an acquire-installed package and reindex.

    SAFETY: only directories carrying our install manifest are touched.
    A hub directory placed there by any other means (hand-installed,
    first-party) is never deleted by this command.
    """
    hub = Path(hub_dir).expanduser() if hub_dir else _default_hub_dir()
    target = hub / _slugify(name)
    if not target.is_dir():
        return {"removed": False, "error": f"{name!r} is not installed in {hub}"}
    if not (target / MANIFEST_NAME).is_file():
        return {
            "removed": False,
            "error": f"refusing to remove {target}: no {MANIFEST_NAME} manifest",
        }
    shutil.rmtree(target)

    out: dict[str, Any] = {"removed": True, "name": target.name}
    try:
        # Resolve through acquire's own root resolver — NOT
        # inference._resolve_runtime_dir, which auto-seeds from Hugging Face
        # (a network call) when the user runtime dir is unpopulated. remove
        # must stay local; with no route table the prune is a no-op anyway.
        out["prune"] = _prune_routes_for_path(_runtime_root(runtime_dir), target)
    except Exception as e:  # noqa: BLE001
        out["prune_error"] = str(e)

    restart_fn = restart if restart is not None else ensure.restart_daemon
    try:
        out["daemon_restart"] = restart_fn()
    except Exception as e:  # noqa: BLE001
        out["daemon_restart_error"] = str(e)
    return out
