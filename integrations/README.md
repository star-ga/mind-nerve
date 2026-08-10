# mind-nerve CLI integrations

Wires mind-nerve into every AI coding CLI on the machine, so a large skills hub
stays *reachable* without being *announced*.

## The problem

A hub of ~1,374 skills symlinked into a CLI's skills directory is bulk-announced
into every session. Re-measured 2026-08-10: the `name` + `description`
frontmatter across 1,374 parseable `SKILL.md` files is **378,508 chars ≈ 95k
tokens per announce**, per CLI, per session. (An earlier figure of ~115k was an
over-estimate; a `^description:\s*(.+)$` measurement is also wrong in the other
direction — it truncates multi-line descriptions at the first newline and
under-reports by ~40%. Fold continuation lines before counting.) Loading every
skill *body* would be ~3.3 M tokens, roughly 16x a 200k context window — which
is the sharper reason routing is the only viable design, not merely the cheaper
one.

The fix has three independent parts, and all three are needed — each one alone
leaves a real gap.

| Part | What it does | Without it |
|------|--------------|------------|
| **1. Structural** | The CLI's skills dir becomes a real directory holding only `mind-nerve-router`. Announce drops ~95k → ~2k tokens. | The hub is announced in full. |
| **2. Automatic** | A hook queries the routing daemon per prompt, projects the relevant skills, and injects a ranked route table with absolute `SKILL.md` paths. | The router is announced but nothing routes; the model has to guess. |
| **3. Hygiene** | Dead/renamed routes are pruned from `route_table.jsonl` **and** its row-aligned `route_table.npy`. | Phantom routes out-rank live skills. |

Two refinements come from measurement rather than design:

- **Score floor** (`MIND_NERVE_MIN_SCORE`, default `0.40`, recalibrated
  2026-08-08). For the prompt *"prove an exact minimum move floor for an
  ARC-AGI-3 level"*, the daemon returned the correct skill at **0.437** followed
  by seven unrelated skills at **0.267–0.282** — GraphQL depth-limit attacks, SOC
  tabletop exercises, `yeet`. All eight were injected and projected: **7 of 8
  rows were noise.** Below the floor the hook injects an explicit *"no strong
  skill match — proceed without one"* line instead of misleading rows.

  The floor is a **backstop, not the primary filter** — the intent gate (which
  rejects harness-shaped and contentless turns before embedding) does the real
  work. An earlier n=78 calibration suggested `0.476`, but it built its positives
  from each route's *own indexed text* with the name stripped — a near
  self-match, which upper-bounds the score and manufactured a separation gap real
  queries do not have. Measured against hand-written paraphrases instead, three
  of six rank-1-**correct** hits fall below `0.476`, including `diagnose` at
  `0.456`. Hence `0.40`. The honest cost: two of five observed noise cases
  (`yeet` 0.45, `check-work` 0.44) survive this gate and are caught downstream.
  Both populations are pinned in `test/hook_gates.test.ts` so the trade-off stays
  visible and any recalibration starts from evidence. Full derivation is in the
  dated decision record above `MIN_SCORE` in `hook/mind-nerve-hook`.
- **Dedup across `source_repo`.** The catalog carries the same skill under
  `"starga"` and `"local"`; a raw top-5 was effectively a top-3 distinct. The
  hook over-fetches (default 80) and dedups before applying top-K.

## Layout

```
integrations/
  hook/
    mind-nerve-hook                  the ONE CLI-agnostic router hook (python3, stdlib only)
    assets/mind-nerve-router.SKILL.md the router skill body installed into each CLI
  installer/                          mind-nerve-installer (TypeScript)
    src/registry.ts                   client registry incl. the six skill surfaces
    src/skills_dir.ts                 Part 1 — structural, with the never-delete rules
    src/hook_wiring.ts                Part 2 — JSON/TOML hook registration
    src/wire.ts                       Parts 1+2 applied per client
    src/hygiene.ts                    Part 3 — route table + embeddings in lockstep
    src/npy.ts                        minimal .npy v1.0 codec (numpy-verified)
```

## What `install` does, per CLI

```
mind-nerve-installer install --all        # detect + wire every CLI present
mind-nerve-installer install claude-code  # one client
mind-nerve-installer install --no-wire X  # MCP/instructions only, skip Parts 1+2
```

For each **detected** client with a skill surface it:

1. copies the shared hook to `~/.mind-nerve/bin/mind-nerve-hook`;
2. writes a per-CLI `sh` wrapper that exports this CLI's env and `exec`s it;
3. registers that wrapper on `UserPromptSubmit` + `SessionStart` (timeout 8s);
4. replaces the skills path with a real directory holding only the router.

Undetected clients are skipped. Every mutated config is backed up to
`<config>.bak-mind-nerve-<unix-ms>` first.

### The six skill surfaces (verified on disk)

| CLI | skills dir | hook config | format | verification |
|-----|-----------|-------------|--------|--------------|
| `claude-code` | `~/.claude/skills` | `~/.claude/settings.json` | JSON | config read live; `~/.claude/hooks/` exists; skills dir real (2 entries) |
| `codex` | `~/.codex/skills` | `~/.codex/config.toml` | TOML | config read live (`skills.config` present); native binary: `UserPromptSubmit`×9, `SessionStart`×36, `SKILL.md`×101 |
| `gemini` | `~/.gemini/skills` | `~/.gemini/settings.json` | JSON | config read live; bundle maps `UserPromptSubmit → BeforeAgent`; `SessionStart` in 18 chunks |
| `grok` | `~/.grok/skills` | `~/.grok/config.toml` | TOML | already wired live — the working reference |
| `kimi` | `~/.kimi-code/skills` | `~/.kimi-code/config.toml` | TOML | config read live (`extra_skill_dirs = []`); binary: `UserPromptSubmit`×23, `SessionStart`×69, `SKILL.md`×30 |
| `qwen` | `~/.qwen/skills` | `~/.qwen/settings.json` | JSON | config read live; chunks contain `hasHooksForEvent("UserPromptSubmit")` + `createHookOutput` |

TOML configs are edited as **text inside a marked block**, never round-tripped
through a serialiser — codex's live config is 125 KB of commented
`skills.config` that a re-serialise would silently strip.

## Hook configuration

All optional; the installer bakes per-CLI values into the wrapper.

| Variable | Default | Meaning |
|----------|---------|---------|
| `MIND_NERVE_SOCKET` | `/run/user/<uid>/mind-nerve.sock` | routing daemon socket |
| `MIND_NERVE_SOURCE_DIR` | `~/.agents/skills-hub` | the hub |
| `MIND_NERVE_PROJECTED_DIR` | *(per CLI)* | skills dir to rewrite each turn |
| `MIND_NERVE_AGENT_DIRS` | `~/.claude/agents:~/.agents/agents` | agent `.md` roots |
| `MIND_NERVE_TOP_K` | `8` | routes injected after dedup + floor |
| `MIND_NERVE_OVERFETCH` | `80` | raw routes requested from the daemon |
| `MIND_NERVE_MIN_SCORE` | `0.40` | score floor (backstop; see above) |
| `MIND_NERVE_CORE_SKILLS` | `mind-nerve-router` | always-projected skills |
| `MIND_NERVE_SOCKET_TIMEOUT` | `2.0` | seconds |
| `MIND_NERVE_LOG` | `~/.mind-nerve/logs/<cli>-hook.log` | JSONL log |

The hook queries the **daemon over its UNIX socket**, not the MCP server — the
MCP process loads its own in-process runtime and does not see the pinned local
route table.

**Fail-open is absolute.** Socket missing, socket timeout, malformed daemon
reply, malformed stdin, missing hub, unwritable projection: every path prints
`{}` and exits 0. A broken router must never block a prompt.

## Route-table hygiene

```
mind-nerve-installer hygiene --dry-run
mind-nerve-installer hygiene --repoint /old/hub=/new/hub
mind-nerve-installer hygiene --no-prune --repoint /old/hub=/new/hub
```

Repoints `source_path` prefixes, drops routes whose `SKILL.md` no longer exists,
and drops the matching **row-aligned** `.npy` rows in the same pass. Repointing
happens *before* the liveness check, so a hub rename does not look like a
catalog-wide deletion.

It **refuses to write** when the JSONL row count and the `.npy` row count
disagree — before or after. That mismatch is the silent killer: dropping a JSONL
line without its embedding row shifts every later embedding onto the wrong
route, with no error and no crash, degrading ranking permanently.

## How to undo

```
mind-nerve-installer uninstall claude-code
mind-nerve-installer uninstall --all
```

Restores from the state recorded at install time (`~/.mind-nerve/state/<cli>.json`):

- **hub symlink** → re-created, pointing at the original target;
- **pre-existing real directory** → moved back from `<dir>.bak-mind-nerve-<ts>`;
- **nothing before** → left absent;
- hook registration removed from the config (foreign hooks untouched);
- per-CLI wrapper deleted.

### Safety rules

- A skills path that is a **symlink** is unlinked — the hub it points at is never
  touched.
- A **real directory** is *never* deleted. It is renamed to
  `<dir>.bak-mind-nerve-<unix-ms>`, so nothing is unrecoverable.
- `uninstall` refuses to remove a real directory it does not manage (i.e. one
  without the router skill) and tells you to move it aside yourself.

Manual undo, if you prefer:

```sh
rm -rf ~/.<cli>/skills && ln -s ~/.agents/skills-hub ~/.<cli>/skills
```

## Tests

```sh
cd integrations/installer && npm install && npm test && npm run typecheck
```

The hook suite spawns the **real** `mind-nerve-hook` against a fake daemon on a
real UNIX socket — the fail-open contract can only be proven by the real process.
The `.npy` codec is cross-verified against NumPy: byte-identical round-trip on a
`numpy.save` fixture and on the live 1437×384 embeddings, with NumPy loading the
TypeScript-pruned output bit-for-bit.
