---
name: mind-nerve-router
description: >
  Structural skill router for the STARGA skills hub. Use FIRST on any non-trivial
  task that might need a specialized procedure, domain skill, security workflow,
  MIND guidance, deploy, review, or research. Route with mind-nerve, then read
  the winning hub SKILL.md and follow it. Hub skills are NOT listed in the system
  prompt — this router is the only automatic entry point to the catalog.
when-to-use: >
  Any task where a specialized skill would help; before guessing which of the
  1300+ hub skills applies; when the user asks for a workflow the hub covers;
  MIND/.mind work, security, deploy, review, research.
---

# mind-nerve-router

The skills hub is large (1300+ skills, ~460k chars ≈ 115k tokens if announced in
full). It is therefore **not** injected into this session. mind-nerve knows the
whole catalog and routes on demand; your job is to use the routes, then load only
the winning skill body.

## Automatic routes

On every prompt the `mind-nerve-hook` injects a ranked route table with absolute
`SKILL.md` paths. **Use it.** If a table is present, skip straight to step 3.

If the injected context says *"No strong skill match — proceed without one"*,
that is a deliberate answer, not a failure: nothing cleared the confidence floor.
Do not go hunting through the hub. Use your own judgement.

## Manual flow (when no route table was injected)

1. **Route.** Call the MCP tool `mind_nerve_route` (server `mind-nerve`) with
   `query` = a concise restatement of the user intent, `top_k` = 5.

2. **Pick.** Prefer the top result with `kind` = `skill` and a resolvable path.
   On near-ties prefer: local/STARGA source, the more specific name, MIND-related
   skills for `.mind`/mindc work, and skills over agents when you only need a
   procedure.

3. **Load.** Read the skill body at its absolute path — the `source_path` from
   the route when present, else `<hub>/<name>/SKILL.md`.

4. **Follow.** Treat the loaded SKILL.md as the procedure for this task.

## Rules

- **Never list or dump the hub directory.** That is exactly the cost this router
  exists to avoid.
- **Never invent a skill body from memory.** Read the file or do without.
- Load **one** body at a time. Re-route with a sharper query if it was wrong.
- Low scores mean *no match*, not *the best of a bad set*.
