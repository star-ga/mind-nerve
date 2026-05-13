# @mind-nerve/installer

Install matrix for mind-nerve. Wires 17 AI CLI clients to the mind-nerve MCP
façade and skill projection system.

## Quick start

```bash
mind-nerve install --all          # detect + install every present CLI
mind-nerve install claude-code    # install one client
mind-nerve uninstall claude-code  # reverse, restores backup
mind-nerve list-clients           # show all 17 clients + detection status
mind-nerve status                 # show active installs
```

## Supported clients

| Client | Config surface | MCP surface | Instruction block |
|---|---|---|---|
| claude-code | `~/.claude/settings.json` | yes (json-servers) | — |
| codex | `~/.codex/config.toml` | yes (toml-codex) | — |
| vibe | `~/.vibe/config.toml` | yes (toml-vibe) | — |
| gemini | `~/.gemini/settings.json` | yes (json-servers) | — |
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
   silently if the client is not detected (use `--force` to override).
2. **Projection dir** — creates `~/.mind-nerve/projections/<client>/` for
   clients with a skill surface (currently claude-code only). The runtime
   hook populates this per-turn.
3. **MCP rewire** — opens the client's MCP config file and injects a
   `mind-nerve` entry pointing at `mind-nerve mcp-facade`. Existing entries
   are preserved. A timestamped backup is created before any write.
4. **Instruction block** — for workspace-rules clients (cursor, windsurf,
   aider, copilot, cody, qodo, cline, roo), appends a `# mind-nerve managed`
   block to the rules file. Re-runs are no-ops.

## Flags

```
mind-nerve install --all               Detect + install every CLI present
mind-nerve install --mcp <client>      MCP-only mode (skip skill projection)
mind-nerve install --shared a,b,c      STARGA power-user: one shared projection dir
```

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
npm test          # all 98 tests pass
npx tsc --noEmit  # type-check clean
```

## License

Apache-2.0. Copyright 2026 STARGA Inc.
