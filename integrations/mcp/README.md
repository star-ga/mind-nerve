# MCP integration

mind-nerve runs as an MCP server façade in front of any upstream MCP server
that exposes too many tools for a single calling-LLM prompt budget.

## Topology

```
calling LLM
    |
    | MCP protocol
    v
mind-nerve MCP façade  -->  mind-nerve inference  -->  top-K tool selection
    |
    | (filtered tool list, MCP protocol)
    v
upstream MCP server (mind-mem v4, or any other)
```

The calling LLM thinks it's talking to a normal MCP server. The façade
intercepts `tools/list` calls, runs the user's last request through
mind-nerve preselection, and returns only the top-K tool descriptors
relevant to that intent.

Tool execution (`tools/call`) passes through transparently — the façade does
not interpose on actual tool execution, only on the discovery surface.

## Phase 1 ships

- **mind-mem v4 façade** — `mind-mem.facade.mind` — reads the mind-mem v4 MCP
  schema, registers itself as a façade upstream, routes by intent.
- **Generic façade** — `generic.facade.mind` — configurable façade that
  accepts a tool catalog manifest at startup and faces any MCP server.

## Behaviour notes

- The façade is stateless per call. State lives in the underlying MCP server.
- The catalog hash is computed at façade startup time, pinned for the
  lifetime of the façade process, and emitted in every attestation envelope.
  A new tool registered upstream requires restarting the façade for the new
  tool to be eligible for selection.
- If mind-nerve is unavailable, the façade falls back to passing the full
  tool catalog through unchanged. Fail-open, never blocks the caller.

## Configuration

`mind-nerve.mcp.toml`:

```toml
[upstream]
command = "mind-mem"
args    = ["mcp", "serve"]

[preselect]
top_k = 7

[fallback]
on_error = "passthrough"   # or "deny"
```

## Status

Phase 1 ships against mind-mem v4 specifically; generic façade is Phase 1
stretch. Implementation gated on the Phase 1 model checkpoint.
