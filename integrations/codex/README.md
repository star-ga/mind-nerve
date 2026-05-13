# codex integration

Installs mind-nerve as a codex skill-preselection hook.

## Install (once mind-nerve binary is available)

1. Make `mind-nerve` available on `$PATH` (or set `MIND_NERVE_BIN`).

2. Drop `hook.sh` somewhere stable and mark it executable:

   ```
   cp hook.sh ~/.codex/hooks/mind-nerve-preselect.sh
   chmod +x ~/.codex/hooks/mind-nerve-preselect.sh
   ```

3. Register it in `~/.codex/config.toml`:

   ```toml
   [hooks]
   skill_preselection = "~/.codex/hooks/mind-nerve-preselect.sh"
   ```

4. Restart codex.

## Behaviour

- codex sets `CODEX_USER_PROMPT` and `CODEX_SKILL_CATALOG_HASH` before each
  turn.
- The hook calls `mind-nerve preselect`, captures the top-K skill IDs.
- Output is newline-separated skill IDs on stdout. codex uses these to filter
  the skill catalog for that turn.

## Safety

Fails open in all error paths: missing binary, timeout, malformed JSON.
Hook never blocks codex execution.

## Status

Phase 1 stretch. The hook code is correct against codex's hook protocol as
documented in 2026-05; revise if the protocol changes.
