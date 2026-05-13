# Integration Surface

How mind-nerve connects to its three host classes. Same model, same binary,
same evidence chain across all three.

## Host 1: Agent CLI runtimes (Claude Code, codex, gemini, vibe)

**Problem**: every turn loads the full skill library into the system prompt.
At 440 skills, the system prompt dominates the context window.

**Integration shape**: each CLI host exposes a preselection hook. mind-nerve
runs as the host's preselector and returns the top-K skill IDs. The host
loads only those into the system prompt for that turn.

**Per-CLI status**:

| CLI | Hook interface | Phase 1 ships? |
|---|---|---|
| Claude Code | TypeScript hook (`integrations/claude-code/preselect.ts`) | yes |
| codex | Shell hook via `.codex/config.toml` (`integrations/codex/hook.sh`) | yes (Phase 1 stretch) |
| gemini | Hook surface still unstable as of 2026-05 | Phase 2 |
| vibe (Mistral) | Pending vibe CLI hook documentation | Phase 2 |

**Wire protocol** (CLI → mind-nerve):

```json
{
  "request": "user message text",
  "context": "optional prior turn text",
  "catalog_hash": "sha256 hex",
  "k": 5
}
```

**Wire protocol** (mind-nerve → CLI):

```json
{
  "version": 1,
  "routes": [
    {"id": "skill.foo", "score": 0.92},
    {"id": "skill.bar", "score": 0.81}
  ],
  "attestation_envelope": "base64-encoded 132 bytes"
}
```

The CLI host MAY ignore the attestation envelope. mind-nerve always emits it
to the local evidence log regardless of host behaviour.

## Host 2: MCP servers

**Problem**: MCP servers (mind-mem, others) expose dozens of tools. The
calling LLM sees the full registry on every call. mind-mem v4 alone has 84+
MCP tools today.

**Integration shape**: mind-nerve runs as an MCP server in its own right,
positioned in front of the actual tool-providing MCP server. When the
calling LLM dispatches a tool query, mind-nerve intercepts, classifies the
intent, and returns only the top-K tool descriptors as the visible registry
for that call.

**Connection topology**:

```
calling LLM
    |
    v
mind-nerve MCP façade  ----intent classification---->  mind-nerve inference
    |
    | (top-K tool IDs only)
    v
underlying MCP server (mind-mem v4, etc.)
```

**Phase 1 ships**:

- mind-mem v4 façade — `integrations/mcp/mind-mem.facade.mind`
- Generic MCP façade — `integrations/mcp/generic.facade.mind` for arbitrary
  MCP servers, configured by tool catalog file

The façade is the only mind-nerve component that speaks MCP wire protocol.
The inference binary itself is wire-protocol-agnostic.

## Cross-host invariants

These hold regardless of which host integrates mind-nerve:

1. **One mind-nerve binary serves all hosts.** Three host classes do not mean
   three binaries; they mean three façades over one inference engine.
2. **One evidence chain.** Every inference, regardless of host, appends to
   the same attestation chain. The chain orders by monotonic timestamp; a
   replay can verify any historical inference from any host.
3. **Same model hash everywhere.** A given mind-nerve checkpoint has exactly
   one `model_hash`, used in every host's attestation envelope. Upgrading
   the model is an atomic event across all hosts.
4. **Catalog hashes are host-scoped.** Each host class has its own catalog
   (skills for CLI, tools for MCP). Catalog hashes are not portable across
   hosts.

## What integrations are NOT in scope

Explicit non-goals so reviewers don't ask:

- IDE integrations (VS Code, JetBrains). mind-nerve is a runtime
  preselector, not an IDE plugin. Skill libraries inside IDEs route through
  whichever CLI they spawn (Claude Code, etc.).
- Browser integrations. mind-nerve runs in WebGPU for inference but the
  surface for routing in browser-based agent runtimes is a separate
  question.
- Voice agent runtimes. Different latency budget, different input modality,
  different concerns. mind-voice handles voice; mind-nerve handles text.
- Multi-modal routing. Images and audio routing are out of scope for the
  current architecture. A future mind-nerve-multimodal variant would
  require a different encoder.

## Status by host (Phase 1)

| Host | Spec done | Shim drafted | Implementation gated on |
|---|---|---|---|
| Claude Code | yes | yes | Phase 1 model checkpoint |
| codex | yes | partial | Phase 1 model checkpoint |
| gemini | no | no | gemini hook API stabilising |
| vibe | no | no | vibe hook API documentation |
| mind-mem MCP | yes | no | Phase 1 model + façade |
| Generic MCP | yes | no | Phase 1 model + façade |
