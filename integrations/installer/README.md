# @mind-nerve/installer

Install matrix for mind-nerve. Wires 20 AI CLI clients to the mind-nerve MCP
server and skill projection system.

## Quick start

```bash
mind-nerve install --all          # detect + install every present CLI
mind-nerve install claude-code    # install one client
mind-nerve uninstall claude-code  # reverse, restores backup
mind-nerve list-clients           # show all 20 clients + detection status
mind-nerve status                 # show active installs
```

## Supported clients

| Client | Config surface | MCP surface | Instruction block |
|---|---|---|---|
| claude-code | `~/.claude/settings.json` | yes (json-servers) | — |
| codex | `~/.codex/config.toml` | yes (toml-codex) | — |
| vibe | `~/.vibe/config.toml` | yes (toml-vibe) | — |
| gemini | `~/.gemini/settings.json` | yes (json-servers) | — |
| grok | `~/.grok/config.toml` | yes (toml-vibe) | — |
| kimi | `~/.kimi-code/config.toml` | yes (json-servers, `~/.kimi-code/mcp.json`) | — |
| qwen | `~/.qwen/settings.json` | yes (json-servers) | — |
| cursor | `~/.cursor/mcp.json` | yes (json-cursor) | `.cursorrules` |
| windsurf | `~/.codeium/windsurf/mcp_config.json` | yes (json-windsurf) | `.windsurfrules` |
| continue | `~/.continue/config.json` | yes (json-servers) | — |
| cline | VSCode globalStorage | yes (json-servers) | `.clinerules` |
| roo | VSCode globalStorage | yes (json-servers) | `.roo/system-prompt.md` |
| zed | `~/.config/zed/settings.json` | yes (json-zed) | — |
| openclaw | `~/.openclaw/openclaw.json` | yes (json-servers) | — |
| nanoclaw | `~/.nanoclaw/nanoclaw.json` | yes (json-servers) | — |
| nemoclaw | `~/.nemoclaw/nemoclaw.json` | yes (json-servers) | — |
| aider | `.aider.conf.yml` | — | `.aider.conf.yml` |
| copilot | — | — | `.github/copilot-instructions.md` |
| cody | `.cody/config.json` | — | `.cody/config.json` |
| qodo | `.codium/ai-rules.md` | — | `.codium/ai-rules.md` |

## Per-client install steps

Each `install <client>` run does up to four things:

1. **Detection** — probes binary on `$PATH` and config dirs on disk. Skips
   silently if the client is not detected.
2. **Projection dir** — creates `~/.mind-nerve/projections/<client>/` for
   clients with a skill surface (claude-code, codex, gemini, grok, kimi,
   qwen). The runtime hook populates this per-turn.
3. **MCP rewire** — opens the client's MCP config file and injects a
   `mind-nerve` entry pointing at the `mind-nerve-mcp` console script
   (or `uvx --from mind-nerve mind-nerve-mcp` with `--mcp-launcher uvx`).
   When the local runtime dir exists and is populated (manifest.json or
   route_table.jsonl), the entry pins
   `MIND_NERVE_RUNTIME_DIR` so the server loads the local route table.
   Existing entries are preserved. A timestamped backup is created before
   any write.
4. **Instruction block** — for workspace-rules clients (cursor, windsurf,
   aider, copilot, cody, qodo, cline, roo), appends a `# mind-nerve managed`
   block to the rules file. Re-runs are no-ops.

## Flags

```
mind-nerve install --all               Detect + install every CLI present
mind-nerve install --mcp <client>      MCP-only mode (skip skill projection)
mind-nerve install --shared a,b,c      STARGA power-user: one shared projection dir
mind-nerve install --mcp-launcher uvx <client>   MCP entry via uvx, no venv needed
mind-nerve verify [--cli <name|all>] [--json]    Self-test existing installs
```

## MCP launcher: venv (default) vs uvx

The `mind-nerve` MCP entry each client gets can be launched two ways:

- **`venv` (default)** — the entry pins the absolute path of the mind-nerve
  binary inside a pre-built virtualenv. Zero first-launch latency and works
  fully offline, but the entry breaks if that venv is moved or deleted.
- **`uvx`** — the entry is written as `uvx --from mind-nerve mind-nerve-mcp`
  (argv array form, per each client's config schema). uv resolves the
  published PyPI package on first launch, so the host needs no pre-built
  venv. The trade-off: the first launch pays a package download, and hosts
  without network access must stay on `venv`.

Either way, env pins in the entry are preserved.

## Verifying an installation

`mind-nerve verify` self-tests an existing installation per CLI and exits
non-zero if any check FAILs:

- config exists, parses (JSON/TOML per client), and still carries the
  mind-nerve managed block;
- the hook script exists, is executable, and answers `{}` on stdin with valid
  JSON within 5s (the fail-open contract);
- the skills projection dir/symlink is consistent (a symlink target must
  exist and contain `mind-nerve-router`);
- the MCP entry is present and its command binary resolves;
- vibe: the entry carries the `transport = "stdio"` discriminator Vibe
  2.9.6 requires (FAIL when missing);
- the MCP entry's `MIND_NERVE_RUNTIME_DIR` env pin matches a populated local
  runtime dir — WARN (never FAIL) when the pin is missing or stale;
- env pins in the hook wrapper are intact;
- the daemon socket answers — WARN (with an ensure/start hint), never FAIL:
  the daemon is allowed to be on-demand.

With no `--cli`, only clients that have something on disk are checked.
`--json` prints a machine-readable report.

## STARGA power-user: shared projection

If you have the STARGA shared `~/.agents/skills/` setup, pass `--shared` to
use a single projection directory instead of per-CLI projections:

```bash
mind-nerve install --shared claude-code,gemini,codex
```

This creates `~/.mind-nerve/projections/shared/` and points all listed
clients at it. Saves disk and keeps projections in sync.

## Uninstall

Uninstall is always reversible:

```bash
mind-nerve uninstall claude-code      # restore claude-code config from .bak
mind-nerve uninstall --all            # uninstall all clients
```

To wipe all mind-nerve state completely:

```bash
mind-nerve uninstall --all && rm -rf ~/.mind-nerve/
```

Backup files are named `<config>.bak-mind-nerve-<unix-timestamp>`. Only the
most recent backup is restored on `uninstall`. Older backups are left in
place for manual recovery.

## Idempotency

Re-running `install <client>` on an already-installed client is a no-op. No
spurious backup files are created and no config files are modified.

## Backup discipline

- Backup is created before every config write.
- Backup file naming: `<config>.bak-mind-nerve-<unix-timestamp>`.
- `uninstall <client>` restores the most recent backup byte-for-byte.
- Re-running `install` on an already-installed client does not produce new
  backup files.

## Development

```bash
npm install --legacy-peer-deps
npm test          # vitest suite
npx tsc --noEmit  # type-check clean
```

## License

Apache-2.0. Copyright 2026 STARGA Inc.
