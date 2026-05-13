# Claude Code integration

Installs mind-nerve as a UserPromptSubmit hook so Claude Code's skill loader
only surfaces the top-K relevant skills per turn.

## Install (once mind-nerve binary is available)

1. Make sure the `mind-nerve` binary is on your `$PATH`. Alternatively, set
   `MIND_NERVE_BIN` to an absolute path.

2. Drop `preselect.ts` into `~/.claude/hooks/`:

   ```
   cp preselect.ts ~/.claude/hooks/mind-nerve-preselect.ts
   ```

3. Register it in `~/.claude/settings.json` under `hooks.UserPromptSubmit`:

   ```json
   {
     "hooks": {
       "UserPromptSubmit": [
         { "type": "command", "command": "node ~/.claude/hooks/mind-nerve-preselect.ts" }
       ]
     }
   }
   ```

4. Restart Claude Code. The hook is now active.

## Behaviour

- For every user prompt, the hook calls `mind-nerve preselect` with the prompt
  text and the current skill-catalog hash.
- mind-nerve returns the top 5 skill IDs.
- Claude Code's skill loader filters the system-prompt skill listing to those
  5 IDs only.
- If the binary is missing, times out, or errors: the hook fails open and
  Claude Code uses its default skill-loading behaviour.

## Safety

The hook is non-destructive:

- It does not modify the user's prompt.
- It does not block on its own failure.
- If skill preselection returns the wrong top-K for a given prompt, Claude
  Code is no worse off than if the hook were not installed — it just sees a
  smaller skill listing than the full catalog.

## Verification

After installation, every Claude Code turn appends one attestation envelope
to `~/.mind-nerve/evidence.log`. The log is local-only, never sent over the
network, and can be cleared at any time.

To verify a specific envelope manually:

```
mind-nerve verify <base64-envelope>
```
