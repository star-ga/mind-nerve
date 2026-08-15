# Acquiring skills (`mind-nerve acquire`)

`mind-nerve acquire` is the operator-facing pipeline for pulling **external**
skills, agents, and MCP servers into the local hub (`~/.agents/skills-hub` by
default, overridable via `MIND_NERVE_SOURCE_DIR`) so the routing daemon — and
every hooked CLI — can route to them.

Pipeline: **search → fetch to quarantine → vet → install → reindex**.

## Threat model

Third-party skill content is **untrusted input**. A skill is a prompt fragment
an agent will follow; an MCP server description is text an agent will read; a
package can ship scripts. The known attack classes, and the corresponding
defences:

| Attack class | Defence |
| --- | --- |
| Shell-pipe installers (`curl … \| sh`, `iwr … \| iex`) | FAIL rule in the static vetter |
| Reverse/bind shells (`/dev/tcp`, `nc -e`, `socat exec`, socket+dup2) | FAIL rule |
| Crypto miners (stratum, xmrig, …) | FAIL rule |
| Exfiltration endpoints (webhook.site/requestbin-style collectors, raw-IP URLs) | FAIL rule |
| Prompt injection / MCP tool poisoning ("ignore all previous instructions", "do not tell the user", …) | FAIL rule |
| Archive path escapes (`../../..` members, absolute and drive-qualified `C:\…` members/links, symlink escapes) | FAIL rule, checked in zips, tars, and the fetched tree itself — file AND directory symlinks (`os.walk` reports dir links in `dirnames`, never `filenames`), plus dangling links (they would tear the install copy) |
| Payloads hidden in skipped dirs (`dist/`, `build/`, `node_modules/`, `target/`, `__pycache__`) | the scanner never walks those dirs, so they are filtered at fetch on every path (local copy, tarball, git clone) and excluded again at install — skipped content never reaches the hub |
| Malicious content inside archives | archive member *content* is line-scanned (bounded: 1 MiB/member, 8 MiB/archive, 4096 members/archive; nested archives are marked opaque with a WARN); dispatch is by magic bytes, so a zip renamed `.jar` still gets the escape checks |
| NUL-byte content-scan bypass | only known-binary extensions skip the line rules; text-shaped files AND archive members with NULs are scanned with NULs stripped plus a WARN |
| Opaque binaries (`.so`, `.exe`, `.dll`, `.pyc`, `.o`, unparseable `.7z`/`.rar`/`.jar`) | `unscanned-binary` WARN — executable-shaped content is never skipped silently |
| Tar bombs | uncompressed-size and file-count caps enforced pre-extraction; git clones are size-monitored DURING the clone (killed past the cap) and re-measured after |
| Obfuscation (long base64/hex blobs, high-entropy strings) | WARN rule |
| Dynamic execution (`eval`, `exec`, `subprocess`, `child_process`, `new Function`) | WARN rule |
| Credential-path access (`~/.ssh`, `.aws/credentials`, browser cookie stores, `.env` harvesting) | WARN rule |
| Persistence / config hijack (`LD_PRELOAD`, crontab, shell rc writes, hooks tampering) | WARN rule |
| License laundering (commercial/restricted content in the hub) | discovery's license gate (`_classify`) applied at vet AND at reindex with `trusted=False` |

Design rules:

- **Fail-closed.** Any internal scanner error (unreadable file, corrupt
  archive, clamscan failure) produces a `scanner-error` FAIL finding. A
  scanner that cannot see must never report PASS.
- **Quarantine.** Downloads land in `<runtime_dir>/quarantine/<sha256(url)[:16]>/`
  with a size cap (25 MB) and file-count cap (2000) enforced *during*
  fetch/extraction (git clones are polled and killed mid-flight past the
  byte cap). A FAIL verdict leaves the package in quarantine for
  inspection; nothing reaches the hub.
- **Deterministic.** The vetter is a pure function of file bytes — no
  network, no clock, no execution of scanned content. Manifests carry the
  source commit SHA (git fetches) and per-file SHA-256 hashes; no
  timestamps anywhere, and source URLs are stored REDACTED (userinfo and
  query/fragment stripped — credential-bearing clone URLs and signed
  download tokens never persist). Multi-line evasion is folded before
  matching:
  backslash-newline continuations and a sliding 2-line window are evaluated
  alongside the per-line view, and a whole-file whitespace-normalized view
  catches injection phrases split across 3+ newlines; symlinks into skipped
  dirs
  (`dist/`, `build/`, …) FAIL closed because their targets are never walked.
  The same rules bind DIRECTORY symlinks, and dangling symlinks FAIL at
  vet. Vetted in-package links are copied into the hub AS links
  (`copytree(symlinks=True)`), never dereferenced.
- **Optional ClamAV (opt-in).** `use_clamav=True` / `--clamav` folds
  `clamscan` in when installed. It is OFF by default precisely because
  clamav verdicts depend on the host's signature database — an opt-in
  clamav report is not the deterministic artifact the default report is.
  Findings carry the signature name only, never host paths.

Verdicts: `PASS` installs; `WARN` installs only with `--accept-warnings`;
`FAIL` never installs. Every finding carries `file`, `line`, `rule`, and a
capped `excerpt`, sorted deterministically.

## Source registry

`mind-nerve acquire sources` prints the active registry. Shipped defaults:

- `anthropics-skills` — `github:anthropics/skills` (official Anthropic skills)
- `mcp-servers` — `github:modelcontextprotocol/servers` (official MCP servers)
- `superpowers` — `github:obra/superpowers` (community skills)
- `wshobson-agents` — `github:wshobson/agents` (community agents)
- `claude-code-templates` — `github:davila7/claude-code-templates` (community templates)
- `mcp-registry` — `https://registry.modelcontextprotocol.io/v0/servers` (JSON)
- `glama-mcp` — `https://glama.ai/api/mcp/v1/servers` (no-auth JSON API,
  verified live 2026-08-11: HTTP 200, `{pageInfo, servers: [...]}` with
  `repository.url` + `spdxLicense`)
- `smithery-mcp` — `https://registry.smithery.ai/servers` (no-auth JSON API,
  verified live 2026-08-11: HTTP 200, `{servers: [...]}` with
  `qualifiedName` + `homepage`; discovery-only — entries link a hosted page,
  not an installable repo)
- `github-search` — generic GitHub repo search fallback (uses `GITHUB_TOKEN`
  when set; degrades gracefully on the unauthenticated 403 rate limit)

Directories checked but **not** wired (no public no-auth JSON API at
verification time, 2026-08-11) — use them as manual sources: browse, then
`acquire install <repo-url>`:

- PulseMCP — the v0beta API is being sunset (HTTP 410 `API_SUNSET`) and the
  v0.1 API requires an `X-API-Key` header (HTTP 401 without one).
- mcp.so — HTML SPA only; no JSON endpoint answered (HTTP 404 HTML on the
  probed paths).

Users extend or override the registry with `<runtime_dir>/acquire_sources.json`,
a JSON list of source objects:

```json
[
  {"id": "team-hub", "type": "local-dir", "path": "/srv/skills", "enabled": true},
  {"id": "github-search", "enabled": false}
]
```

An entry whose `id` matches a default replaces it (the example disables the
GitHub fallback). Types: `github-repo`, `mcp-registry`, `glama-mcp`,
`smithery-mcp`, `github-search`, `local-dir` (offline; every immediate
subdirectory is a candidate package).

All network I/O has a 30 s timeout; a failing source contributes an empty
result and an entry in the `errors` map — search never crashes.

## Usage

```bash
mind-nerve acquire sources                      # the active registry
mind-nerve acquire search "pdf"                 # ranked candidates (JSON)
mind-nerve acquire vet ./some-dir               # scan only, exit 0 iff PASS
mind-nerve acquire install <url|file://|dir> [--name N] [--accept-warnings]
                                              [--kind skill|agent|mcp]
                                              [--register-mcp]   # kind=mcp only
mind-nerve acquire list                         # packages installed via acquire
mind-nerve acquire remove <name>                # delete + reindex (manifest-gated)
```

`install` accepts git repo URLs, tarballs, `file://` URLs, local
directories, and GitHub `/tree/<ref>/<subdir>` URLs (the shape `search`
returns — the parent repo is cloned and `<subdir>` becomes the package root;
the manifest records `repo_url` + `subdir` + commit SHA). Local paths and
`file://` URLs are honoured only for the operator-typed CLI target;
candidates coming from third-party registries fetch over `https://` only, so
a registry entry can never turn install into a copy of an arbitrary local
directory.

`install` writes `<hub>/<name>/.mind-nerve-install.json` — schema version,
source URL (redacted), commit SHA (git) or `null`, vetting verdict, and
sorted per-file SHA-256 hashes — then reindexes the route table
(`discovery.scan` with `trusted=False`, so the license gate applies) and
restarts `mind-nerve-routed`. Reinstalling changed content first prunes the
package's existing route rows (the table dedups by content sha and is
append-only — without the source-path prune, stale rows would keep pointing
at replaced bytes), then re-adds fresh ones; both table writes are atomic
and row-aligned. The daemon holds the table in memory, so the restart is
what makes new routes live; the next CLI invocation re-spawns it via the
flock-guarded `mind-nerve-routed-ensure`.

GitHub-repo sources surface AGENT files (`agents/foo.md`,
`subagents/foo.md`) as per-artifact `kind="agent"` candidates whose URL is
the raw file; fetch preserves the parent dir name (`agents/foo.md` lands at
`quarantine/<sha>/agents/foo.md`) so discovery's kind-by-path
classification survives, and the hook routes the daemon's `kind` +
`source_path` directly (root-confined: hub or agent dirs only).

`remove` is manifest-gated: it refuses any hub directory that lacks
`.mind-nerve-install.json`, so hand-installed and first-party hub content can
never be deleted by it. It prunes the package's rows from `route_table.jsonl`
and `route_table.npy` **together** (row-aligned, atomic rewrite) before
restarting the daemon.

## Exit codes and empty packages

`acquire install` exits `0` only when the package installed AND the reindex
added at least one route (or every row was already indexed — a benign
reinstall). Exit `1`: vet FAIL, WARN without `--accept-warnings`, or a
reindex error. Exit `2`: fetch failure (caps, escapes, unsupported URL).
Exit `3`: the install completed but the reindex added **zero** routes —
the license gate excluded everything or there is no routable content —
which would otherwise look like success while the daemon can never route
to the package.

**MCP candidates:** `kind="mcp"` search results are vetted and installed as
*content* (their repo lands in the hub, hash-manifested like any package).
By default that is ALL install does — vetting content is not license to
execute a server.

Opt-in MCP-pair registration: `acquire install --kind mcp --register-mcp`
additionally writes the server's `mcpServers` entry (the pair pattern: the
skill teaches the model the tools exist; the MCP config makes them real)
into each *installed* JSON-config CLI (`~/.claude/settings.json`,
`~/.cursor/mcp.json`, `~/.gemini/settings.json`, `~/.qwen/settings.json`).
Registration is fail-closed and conservative:

- The package must carry a recognizable **local** server entry point —
  a `pyproject.toml` `[project.scripts]` console script (registered as
  `uvx --from <installed dir> <script>`) or a `package.json` `bin`
  (registered as `node <installed path>`) — or registration refuses with a
  clear message and no config file is touched. `server.json` registry
  identifiers are refused outright: `npx -y <ident>` / `uvx <ident>` would
  execute an npm/PyPI artifact that was never quarantined or vetted, and an
  unsanitized identifier (leading `-`, whitespace) is option injection into
  npx/uvx. Only entry points resolving INSIDE the vetted package directory
  are ever registered.
- Writes are atomic (exclusive temp + fsync + `os.replace`), preserve the
  config file's permission mode (a fresh umask inode would widen a 0600
  secret-bearing config), only into config files that
  already exist (an absent config means the CLI is not installed), and a
  non-object config is skipped, never clobbered. Existing `mcpServers`
  entries are preserved; ours carries `x-managed-by: mind-nerve-acquire`
  so a future unregister can tell it apart from hand-written rows.
- TOML-config CLIs are out of scope: the stdlib has no TOML writer.

## How reindex reaches each CLI

The hook (`integrations/hook/mind-nerve-hook`) never reads the hub directly
for routing — it asks the daemon over its UNIX socket, then projects the
winning hub skills as symlinks into the CLI's skills dir. So the acquire
install path is: hub dir → `discovery.scan` (embed + atomic table append) →
daemon restart → next prompt in any hooked CLI routes to the new skill, and
the hook projects it like any first-party skill. When no local skill matches
at all, the hook's no-match context points the operator at
`mind-nerve acquire search "<query>"`.
