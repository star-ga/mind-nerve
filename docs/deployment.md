# Deployment

This document covers the supported deployment shapes for `mind-nerve`:
local CLI use, the long-lived UNIX-socket daemon, the stdio MCP server,
and the OCI container image.

The current public targets are **Claude Code**, **Codex**, **Gemini
CLI**, and **Vibe** (Mistral CLI), plus any other MCP-aware client over
stdio.

## At a glance

| Mode | When to use | Latency profile |
| --- | --- | --- |
| One-shot CLI (`mind-nerve route ...`) | Quick checks, scripts, CI smoke | cold load each call (~5 s warmup, then ~250 ms encode) |
| Long-lived daemon (`mind-nerve-routed`) | Interactive editors, hot prompt path | warm: ~23 ms p95 GPU, ~90 ms p95 4-core CPU (Phase 1 PyTorch) |
| stdio MCP server (`mind-nerve-mcp`) | Any MCP-aware host (Claude Code, Cursor, Codex, ...) | bound by host's MCP transport |
| Container (`mind-nerve:local`) | Headless services, reproducible local dev | matches daemon mode |

## Docker quickstart

The repository ships a multi-stage `Dockerfile` and a `docker-compose.yml`
that brings up the daemon with persistent volumes for runtime artefacts
and the Hugging Face cache.

```bash
# Build the image
docker compose build

# Start the daemon
docker compose up -d

# Tail logs while the model warms up (first run downloads the encoder)
docker compose logs -f mind-nerve-daemon

# Verify the daemon is responsive
docker compose exec mind-nerve-daemon mind-nerve --version
docker compose exec mind-nerve-daemon mind-nerve-routed-ensure
```

To call the daemon from your host, mount the socket volume into another
container or run the CLI inside the same container:

```bash
docker compose exec mind-nerve-daemon \
    sh -c 'echo "git status" | mind-nerve route --top-k 5'
```

### Volumes

| Volume | Mount point | Purpose |
| --- | --- | --- |
| `mind-nerve-runtime` | `/var/lib/mind-nerve/runtime` | Encoder checkpoint + route table |
| `mind-nerve-cache` | `/var/cache/mind-nerve` | XDG cache + Hugging Face download cache |
| `mind-nerve-socket` | `/var/run/mind-nerve` | UNIX socket directory |

Removing the cache volume forces a re-download of the encoder on the
next start; removing the runtime volume forces a re-seed.

### Environment variables (container)

The image sets sensible defaults for a daemon-only deployment:

```text
MIND_NERVE_RUNTIME_DIR=/var/lib/mind-nerve/runtime
MIND_NERVE_SOCKET=/var/run/mind-nerve/mind-nerve.sock
XDG_CACHE_HOME=/var/cache/mind-nerve
HF_HOME=/var/cache/mind-nerve/huggingface
```

To pin the Hugging Face revision (recommended for reproducible
deployments), set `MIND_NERVE_HF_REVISION` to a commit SHA when bringing
the container up:

```yaml
environment:
  MIND_NERVE_HF_REVISION: "<pinned-revision-sha>"
```

## Daemon mode (host)

`mind-nerve-routed` is the production hot path. It loads the encoder
once at startup and answers single-line JSON requests over a UNIX
socket.

```bash
mind-nerve-routed &
mind-nerve-routed-ensure       # idempotent — also used by the SessionStart hook
echo '{"prompt":"deploy staging","top_k":5}' | nc -U /run/user/$(id -u)/mind-nerve.sock
```

The `ensure` helper is safe to invoke from any number of parallel
sessions: a non-blocking `flock` on a sibling lock file serialises the
"should I spawn?" decision so at most one daemon process ever exists
per socket.

### Systemd user unit (Linux)

Installing the daemon as a long-lived systemd user unit gives it its
own cgroup and real `Restart=` semantics:

```bash
mind-nerve-install install --cli claude-code --with-systemd
systemctl --user status mind-nerve-routed.service
```

## MCP mode

`mind-nerve-mcp` is a stdio MCP server exposing a single tool,
`mind_nerve_route`, which proxies to either the running daemon (warm
path) or a fresh encode (cold path). Every public target wires it up
through `mind-nerve-install`:

```bash
mind-nerve-install install --cli claude-code
mind-nerve-install install --cli codex
mind-nerve-install install --cli gemini
mind-nerve-install install --cli vibe
mind-nerve-install install --cli cursor
mind-nerve-install install --cli claude-desktop
```

Add the optional preselector hook for Claude Code so the top-K skills
are projected into `~/.claude/skills` on every prompt:

```bash
mind-nerve-install install --cli claude-code --with-preselect
```

## Healthcheck and readiness contract

The container `HEALTHCHECK` does two things, in order:

1. `mind-nerve-routed-ensure` — exits 0 only after the daemon has bound
   its UNIX socket. This is the same probe the SessionStart hook uses,
   so container readiness matches editor-side readiness.
2. `mind-nerve --version` — sanity check that the Python entry point
   resolves.

Readiness states:

| State | Signal |
| --- | --- |
| Bootstrapping | container is up, socket not yet present (encoder loading, ~5 s typical) |
| Ready | socket file exists, `ensure` returns 0, daemon answers `{"prompt":"ping","top_k":1}` |
| Unhealthy | `ensure` succeeded but the daemon never answered the latest request — restart |

Compose's `start_period: 60s` gives the encoder time to load on first
boot without the orchestrator declaring the container unhealthy.

## Logging defaults

The daemon writes a startup line to stderr containing the resolved
runtime directory, route count, and socket path. When started via the
systemd user unit, it also appends to `~/.mind-nerve/daemon.log`
(append-only, line-delimited).

The container image inherits these defaults; orchestrators should
attach to stdout/stderr (`docker logs`, `journalctl --user-unit ...`,
or a sidecar log collector) rather than reading the file.

`mind-nerve` does not transmit prompts or routing results off the
machine. Local catalog scans (`mind-nerve learn`) honour the discovery
license gate documented in the package metadata.

## Rollback procedure

Every config write performed by `mind-nerve-install` goes through the
atomic `safe_write` helper, which:

1. Creates parent directories on demand.
2. Copies the existing file (if any) to `<path>.bak`.
3. Writes the new content to a temp file in the same directory.
4. Atomically renames the temp file over the target (`os.replace`).

To undo the most recent installer write for a given target:

```bash
mind-nerve rollback --target claude
mind-nerve rollback --target codex
mind-nerve rollback --target gemini
mind-nerve rollback --target vibe
mind-nerve rollback --target cursor
mind-nerve rollback --target claude-desktop
```

(`mind-nerve-install rollback --target <name>` is also accepted.)

The command restores each known config file for the target from its
sibling `.bak`, atomically. Paths that were never installed by
`mind-nerve` simply have no `.bak` and are skipped; the command reports
which paths were restored, which were missing, and any I/O errors.

## Supply-chain provenance

Public release wheels and source distributions published to PyPI are
built and published from `.github/workflows/release.yml` using PyPI's
[trusted-publishing](https://docs.pypi.org/trusted-publishers/) OIDC
flow — no long-lived `PYPI_API_TOKEN` secret is required.

Each release attaches:

- `actions/attest-build-provenance` SLSA-style build provenance for the
  wheel and sdist.
- A `SHA256SUMS` file containing `sha256sum`-format checksums of every
  published artefact.

To verify a downloaded wheel:

```bash
python -m pip download mind-nerve==<version> --no-deps -d ./pkgs
cd pkgs
gh release download v<version> --pattern SHA256SUMS
sha256sum --check SHA256SUMS
```

The repository's CI build job validates the public OSS surface (Python
imports, lint, build, tests). The native runtime component bundled into
production wheels is built and signed in a separate channel; see
`LICENSE.md` for the dual-licensing details.
