# mind-nerve Phase 1 Implementation

## Mission
Build mind-nerve — a tiny CPU-first intent-classification preselector that
sits in front of any agent host (Claude Code, codex, gemini, vibe, OpenClaw,
MCP servers) and filters skill/agent/tool registries down to a top-K
candidate set BEFORE the registry hits the LLM's system prompt.

Goal: let hosts ship thousands of skills/agents/MCP tools without burning
context-window tokens. Today Claude Code truncates skill descriptions past
~250 entries. mind-nerve eliminates that ceiling.

Repo: github.com/star-ga/mind-nerve (private, Apache-2.0 architecture +
integrations, STARGA Commercial weights).

## Read first
- README.md, ROADMAP.md
- spec/architecture.md, spec/quality_targets.md, spec/integration_surface.md
- integrations/claude-code/preselect.ts, integrations/codex/hook.sh
- src/lib.mind, src/model.mind, src/inference.mind, src/evidence.mind
- cli/main.mind

## Phase 1 deliverables
1. **Pure-MIND model bodies** — fill `unimplemented!()` stubs in
   src/model.mind, src/inference.mind, src/evidence.mind. Encoder/decoder,
   classifier head, top-K extraction, Q16.16 throughout. Pin reduction
   orders. Compile-time tensor shape checks pass.
2. **CLI binary** — cli/main.mind reads stdin (intent string + registry
   manifest), writes stdout (top-K skill/agent/tool IDs as JSON). Single
   static binary. p95 ≤ 30ms on x86 CPU, ≤ 50ms on ARM.
3. **Cross-arch bit-identity test** — tests/bit_identity/ compiles and
   runs identical inputs on x86_64 + ARM64 + (optional) CUDA, asserts
   SHA-256 of output JSON matches byte-for-byte. CI gate.
4. **Claude Code hook integration** — wire integrations/claude-code/preselect.ts
   into ~/.claude/hooks/UserPromptSubmit, prove it filters skills before
   system-prompt rendering. Validate against /home/n/.claude/skills/*
   (440+ entries).
5. **MCP façade** — integrations/mcp/ implements stdio-transparent proxy
   sitting in front of mind-mem's MCP server. Filters 84+ tools to top-K
   per call. Performance budget: ≤5ms preselect overhead.

## Workflow
- Use **arch-mind** (global binary, ~/.local/bin/arch-mind) for pre-session
  scan and post-session audit. Reject commits that fail invariants.
- Use **mind-auditor** agent on every .mind file before commit.
- Use **mind-dev** agent for .mind implementations. STRICTLY refuse
  Python/TS in src/ — those go in integrations/ only.
- Use **tdd-guide** agent — write tests BEFORE implementation. 80% coverage
  minimum.
- Use **code-reviewer** agent after every meaningful change.
- Use **planner** agent before starting Phase 1.2 (architecture decisions
  in spec/architecture.md must be honoured, not redesigned).

## Constraints
- All commits: STARGA Inc <noreply@star.ga>. No co-authors. No AI
  attribution. Conventional commits.
- mindc compile speed never regresses (1.8–15.5 µs frontend). Module-level
  gating only.
- No upstream attribution. Generic engineering names. No "needle" /
  "function-call model" / specific paper references in code or commits.
- Cross-arch bit-identity is non-negotiable. If Q16.16 ops produce
  different bytes on x86 vs ARM, fix the kernel, do not relax the
  invariant.
- Per-neuron attestation envelope: every weight tensor SHA-256-hashed at
  load time, refuse to run if hash chain breaks.
- Local commits only until I authorize push.

## Training data note
Phase 1 ships with dummy/seed weights for integration testing. Real
weights land in mind-mem v4.1 joint release (router distilled from
mind-mem 4b teacher). Do NOT train weights in Phase 1 — focus on
architecture, inference path, integration surface, and bit-identity
discipline.

## Out of scope for Phase 1
- gemini and vibe CLI integrations (Phase 2 — their hook surfaces
  unstable)
- GPU backend codegen (Phase 2 — CPU bit-identity first)
- Real model weights (gates on mind-mem v4.1 retry2e completion)

## Definition of done
- arch-mind audit passes
- All bit-identity tests green
- Claude Code hook demonstrably filters /home/n/.claude/skills/* in
  ≤30ms
- MCP façade proves ≤5ms overhead against mind-mem MCP server
- code-reviewer and mind-auditor pass on every commit
- README updated with quickstart for each integration

Begin by reading the three spec docs end-to-end, then run planner
agent against Phase 1 deliverables to produce a per-task execution
plan.
