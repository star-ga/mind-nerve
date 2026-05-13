# MCP Façade — Design

> Authoritative architecture for the mind-nerve MCP façade, Phase 1 deliverable #5.
> Implementer should be able to build from this document without re-reading the
> integration ADR.
>
> Status: draft, 2026-05-13.

## What this is

A stdio-transparent proxy that sits between an MCP client (Claude Code,
OpenClaw, etc.) and an upstream MCP server (mind-mem, generic). It intercepts
`tools/list` to filter the catalog down to top-K via mind-nerve, byte-forwards
everything else. The client believes it is talking to a normal MCP server; the
upstream believes it is talking to a normal MCP client.

The façade is the only mind-nerve component that speaks MCP wire protocol. The
inference binary itself is wire-protocol-agnostic and only consumes mic@2
frames over a private subprocess channel.

---

## 1. Architecture diagram

```
+----------------------+   stdio   +----------------------+   stdio   +----------------------+
|   MCP client         |<--------->|   mind-nerve façade  |<--------->|   upstream MCP       |
|   (Claude Code)      |  JSON-RPC |   (this component)   |  JSON-RPC |   (mind-mem, etc.)   |
+----------------------+           +----------+-----------+           +----------------------+
                                              |
                                              | mic@2 frames over pipe
                                              v
                                   +----------------------+
                                   |   mind-nerve binary  |
                                   |   (warm subprocess,  |
                                   |    --serve-stdin)    |
                                   +----------------------+
```

### Six message paths

1. **`initialize`**: byte-forward to upstream. Capture upstream's reply,
   inspect `serverInfo.name` + `protocolVersion` to pin the upstream profile
   (mind-mem vs generic), then byte-forward the reply back to client.
2. **`tools/list`**: intercept. The façade emits its own reply containing the
   top-K subset selected by mind-nerve. The upstream's full list is fetched
   eagerly (once per refresh cycle) and cached.
3. **`tools/call`**: byte-forward. The façade never parses tool arguments or
   results. Even tools that were filtered out of the most recent `tools/list`
   are forwarded if the LLM names them (see §6).
4. **`prompts/get`**: byte-forward, but use this method as a per-turn signal
   to re-run mind-nerve preselection against the latest user-visible prompt
   (see §5).
5. **`resources/list` / `resources/read`**: byte-forward. mind-nerve does not
   filter resources in Phase 1.
6. **Anything else** (`prompts/list`, `notifications/*`, `logging/*`,
   `roots/*`, `sampling/*`, unknown methods): byte-forward both directions.
   The façade is a pass-through for every method it does not own.

The façade is stateless between calls except for: (a) the cached upstream
`tools/list` response, (b) the catalog hash derived from it, (c) the last
per-turn top-K decision, (d) the upstream profile pinned at `initialize`. All
four are evicted on `chain_reset` (catalog change, model swap, clock reset).

---

## 2. Per-method behaviour matrix

| Method                | Direction              | Behaviour    | Cache?       | Refresh policy                            |
|-----------------------|------------------------|--------------|--------------|-------------------------------------------|
| `initialize`          | client → upstream      | forward      | profile only | once per façade lifetime                  |
| `initialize` (reply)  | upstream → client      | forward      | n/a          | n/a                                       |
| `tools/list`          | client → upstream      | **intercept**| yes          | TTL 60s, or `catalog_refresh` notification |
| `tools/list` (reply)  | synthesised            | **filter**   | yes          | per-turn re-classify, see §5              |
| `tools/call`          | client → upstream      | forward      | no           | n/a (log `out_of_band_tool_call` if filtered) |
| `tools/call` (reply)  | upstream → client      | forward      | no           | n/a                                       |
| `prompts/get`         | client → upstream      | forward + tap| no           | re-classification trigger                 |
| `prompts/list`        | client → upstream      | forward      | no           | n/a                                       |
| `resources/list`      | client → upstream      | forward      | no           | n/a                                       |
| `resources/read`      | client → upstream      | forward      | no           | n/a                                       |
| `notifications/*`     | bidirectional          | forward      | no           | special-case `catalog_refresh`            |
| `logging/*`           | bidirectional          | forward      | no           | n/a                                       |
| `roots/*`             | bidirectional          | forward      | no           | n/a                                       |
| `sampling/*`          | bidirectional          | forward      | no           | n/a                                       |
| `ping`                | bidirectional          | forward      | no           | n/a                                       |
| unknown               | bidirectional          | forward      | no           | n/a                                       |

"Forward" means byte-level: the façade reads one JSON-RPC frame off the
incoming stream and writes it unchanged to the outgoing stream. No parsing
beyond extracting the JSON-RPC `method` field via a single 2KB peek. The
`method` peek does not allocate the full message into a structured object.

---

## 3. mind-nerve warm-subprocess lifecycle

The façade spawns the mind-nerve binary at startup and holds it open for the
entire façade process lifetime. One subprocess per façade. No subprocess
pooling. mind-nerve internally is single-threaded for inference; concurrency
inside mind-nerve is out of scope.

### Spawn

- Command: `${mind_nerve_bin} --serve-stdin` (argv-list, never shell).
- `mind_nerve_bin` resolved from config (`[mcp.<name>].mind_nerve_bin`), not
  `$PATH` (see §10 security).
- Working directory: `~/.mind-nerve/`.
- Environment: inherits façade env, plus `MIND_NERVE_FACADE=1`.
- stdin/stdout: piped to façade. stderr: inherited (façade's stderr).
- Spawned synchronously at façade startup; if spawn fails, the façade enters
  **passthrough mode** (see §9) and logs `subprocess_spawn_failed`.

### Per-call protocol

mind-nerve subprocess speaks mic@2 frames on stdin, replies in mic@2 on
stdout. Each frame is length-prefixed (4-byte big-endian unsigned, max 4 MB).

Request frame fields (mic@2):

```
op            = "preselect" | "noop" | "version"
request_id    = u64 (façade-assigned, monotonic)
request_text  = utf-8 bytes (user prompt, ≤ 4096 tokens after BPE)
catalog_hash  = 32 bytes (SHA-256 over canonical tool catalog serialization)
catalog_body  = utf-8 bytes (newline-separated tool_id + description SHA-256),
                present only when the catalog_hash is new to mind-nerve
k             = u8 (top-K, default 7, max 64)
```

Reply frame fields:

```
op             = "preselect_reply" | "noop_reply" | "version_reply" | "error"
request_id     = u64 (echo)
route_ids      = list of utf-8 tool_id strings (length K)
scores_q16_16  = list of i32 raw Q16.16 scores (length K)
envelope       = 212 bytes mic-b attestation envelope (v2)
```

### Latency expectations

- Warm pipe (subprocess already serving prior call): 50 ms upper bound,
  typical 0.5–2 ms framing + 5–20 ms mind-nerve forward pass.
- Cold pipe (first call after spawn): 250 ms upper bound, dominated by weight
  load. Cold pipe latency is amortised over façade lifetime, so it is not
  budgeted into the per-call ≤5 ms façade overhead.

### Health check

Every 30 s the façade writes a `noop` frame and expects a `noop_reply` within
500 ms. On timeout or framing error:

1. Log `mind_nerve_health_failed` with reason.
2. SIGTERM the subprocess.
3. If subprocess does not exit within 1 s, SIGKILL.
4. Respawn. If respawn fails twice in 60 s, enter passthrough mode.

### Shutdown

On façade `SIGTERM` / clean exit:

1. Close stdin pipe (causes mind-nerve to exit cleanly).
2. Wait up to 1 s for subprocess to exit.
3. If still alive, SIGKILL.
4. `waitpid` to reap zombie.

---

## 4. Catalog hash + refresh path

The catalog hash is the binding between the upstream's tool set and
mind-nerve's classification. If the upstream registers a new tool and the
façade does not refresh, the new tool is invisible to clients. The architect
audit (PHASE_1_ARCH_AUDIT.md §4) flagged this as the dynamic-catalog problem
and required a refresh path before Phase 1 ships. This is that path.

### Hash computation

The façade builds the catalog hash from the upstream `tools/list` response.
Canonical serialization (matches `spec/architecture.md §Catalog hashing`):

```
for each tool in tools sorted by SHA-256(tool.name):
    emit length-prefixed tool.name
    emit length-prefixed SHA-256(canonical_json(tool.inputSchema || tool.description))
```

The hash is sent to mind-nerve in every `preselect` frame. mind-nerve caches
the catalog hash internally so it does not need to re-encode the catalog
body on every call; only the first call after a new hash carries
`catalog_body`.

### Refresh triggers

1. **TTL expiry (60 s default).** A wall-clock timer; on expiry, the façade
   silently issues `tools/list` to upstream, re-hashes, and chains a
   `chain_reset_reason: ttl_refresh` envelope (only if the hash actually
   changed).
2. **`catalog_refresh` notification.** If the upstream supports
   `notifications/catalog_refresh` (a mind-mem v4 extension), the façade
   immediately re-fetches `tools/list` on receipt. Generic upstreams may not
   send this; TTL is the safety net.
3. **Explicit `tools/list` from client.** If the client itself calls
   `tools/list` (some clients do this on focus change), the façade serves
   from cache if fresh, otherwise refreshes and serves the new top-K.
4. **`InvalidParams` from upstream `tools/call`** referencing a tool the
   façade thinks exists. Indicates upstream catalog has shrunk; force
   refresh.

### Hash mismatch handling

If two `tools/list` responses produce different hashes (catalog changed):

1. Emit a `chain_reset` envelope to `~/.mind-nerve/evidence.log` with
   `chain_reset_reason: catalog_changed`.
2. Invalidate the per-turn top-K cache.
3. Re-run preselection on the next `tools/list` interception.
4. Continue serving; do not interrupt in-flight `tools/call`.

---

## 5. Filtering semantics

### Top-K selection

Default `K = 7`. Configurable per upstream in `[mcp.<name>].top_k`. Hard
cap `K ≤ 64` (matches `spec/architecture.md` functional contract).

When `tools/list` arrives from the client:

1. If cached top-K for the current turn is valid, synthesise reply directly
   from cache (≤0.3 ms).
2. Otherwise, run mind-nerve with the latest user-prompt text (captured via
   `prompts/get`, see below) and the catalog hash. Cache the resulting top-K
   for the duration of the turn.
3. Build the synthesised `tools/list` reply by filtering the cached upstream
   response to only the top-K `route_ids`, preserving the upstream's
   `inputSchema`, `description`, and any annotations byte-for-byte.

### Per-turn re-classification trigger

Most MCP clients invoke `prompts/get` at the start of every turn (Claude
Code does this to render its system prompt). The façade taps this method:

- Forward the request to upstream unchanged.
- Asynchronously, in parallel with waiting for the upstream reply, extract
  the `arguments.user_message` (or equivalent — the field name varies, see
  §12) and stash it as the latest prompt.
- The next `tools/list` interception uses this latest prompt for
  preselection.

If `prompts/get` is not used by the client (architect open question Q2 — see
§12), fall back to **TTL-based re-classification**: re-run mind-nerve every
5 s (configurable as `[mcp.<name>].fallback_turn_ttl_ms = 5000`).

### Empty / unknown prompts

If no prompt text is available (no `prompts/get` seen, TTL expired but no
new prompt captured), mind-nerve receives an empty `request_text`. The model
is trained to return a low-confidence broadly-relevant top-K in this case.
The façade does not error.

---

## 6. Tool-call routing for filtered-out tools

The LLM may call a tool that was not in the most recent top-K. Two cases:

### Case A: tool exists upstream, was filtered out

This happens when the client's LLM remembers a tool name from prior context
or when the user explicitly references a tool. Behaviour:

1. Byte-forward the `tools/call` to upstream unchanged.
2. Log `out_of_band_tool_call` to stderr at INFO level with fields
   `tool_name`, `request_id`, `last_top_k_hash`.
3. Do NOT append an evidence envelope; `tools/call` is not a filtering
   event (§11).

This is non-blocking. The tool executes normally. The log entry exists so
operators can detect cases where mind-nerve consistently filters out tools
the LLM still wants — that's a signal the classifier is mis-routing.

### Case B: tool does not exist upstream

Detected when the upstream returns JSON-RPC `-32602 Invalid params` with a
"tool not found" indicator, or when the façade has a fresh `tools/list`
cache that does not contain the named tool.

The façade returns a JSON-RPC error to the client directly without
forwarding:

```
{ "jsonrpc": "2.0",
  "id": <request id>,
  "error": { "code": -32601, "message": "Method not found",
             "data": { "tool_name": "<name>" } } }
```

`-32601 Method not found` is the correct JSON-RPC code for "the requested
tool does not exist in the upstream's published catalog." The client is
expected to surface this to its LLM and let it self-correct.

---

## 7. Latency budget breakdown

Per `spec/quality_targets.md`, the façade must add ≤5 ms p95 overhead vs raw
upstream MCP. Decomposition per call type:

### `tools/list` (intercept)

| Stage                                 | Cached path | Cold reclassify |
|---------------------------------------|-------------|-----------------|
| Read JSON-RPC frame from client       | 0.1 ms      | 0.1 ms          |
| Peek `method` field (no full parse)   | 0.05 ms     | 0.05 ms         |
| mic@2 frame serialization to subproc  | —           | 0.5 ms          |
| Subprocess write + read (warm pipe)   | —           | 0.5 ms          |
| mind-nerve forward pass               | —           | 18–30 ms (out of overhead budget, see note) |
| Cache lookup for top-K                | 0.05 ms     | —               |
| Filter cached upstream `tools/list`   | 0.3 ms      | 0.3 ms          |
| Synthesise reply, write to client     | 0.5 ms      | 0.5 ms          |
| Evidence log append (mic-b, locked)   | 0.5 ms      | 0.5 ms          |
| **Total**                             | **≤1.5 ms** | **≤30 ms**      |

The cold-reclassify path includes the mind-nerve forward pass, which is
already inside the model's own 30 ms p95 budget (`spec/quality_targets.md`
§Latency). The façade adds only ≤1.5 ms on top of that. Cached path runs on
every subsequent `tools/list` in the same turn — most calls.

### `tools/call` (forward)

| Stage                                 | Cost     |
|---------------------------------------|----------|
| Read frame, peek method, write frame  | 0.2 ms   |
| Match opening tool_name (for §6 case) | 0.05 ms  |
| **Total**                             | **≤0.3 ms** |

No parsing, no allocation of result body. The façade never touches tool
inputs or outputs.

### Everything else

Pure byte-forward: ≤0.2 ms per direction. Effectively zero overhead.

### Budget margin

`tools/list` interception is the only stage where the façade adds
meaningful latency. p95 budget ≤5 ms is met with 70% margin on the cached
path. Cold reclassify is bounded by mind-nerve's own latency, which is
already specified.

---

## 8. Config schema

The façade is configured via `~/.config/mind-nerve/config.toml`. One
`[mcp.<name>]` block per upstream. The façade can host multiple upstreams
simultaneously, one façade process per `[mcp.<name>]` entry.

```toml
# mind-mem example: Python upstream, warm mind-nerve binary
[mcp.mind_mem]
upstream_command = "python"
upstream_args = ["-m", "mind_mem.mcp_server"]
upstream_env = { LOG_LEVEL = "INFO" }
upstream_cwd = "/home/n/mind-mem"

mind_nerve_bin = "/home/n/.local/bin/mind-nerve"
top_k = 7
fallback_turn_ttl_ms = 5000
catalog_refresh_ttl_ms = 60000
health_check_interval_ms = 30000

on_subprocess_error = "passthrough"  # or "deny"
telemetry = "evidence_log"            # or "disabled"

# Generic example: Node upstream, cold mind-nerve binary
[mcp.generic_example]
upstream_command = "node"
upstream_args = ["server.js"]
upstream_env = {}
upstream_cwd = "/srv/example-mcp"

mind_nerve_bin = "/usr/local/bin/mind-nerve"
top_k = 5
fallback_turn_ttl_ms = 3000
catalog_refresh_ttl_ms = 30000
health_check_interval_ms = 30000

on_subprocess_error = "passthrough"
telemetry = "disabled"
```

Schema rules:
- `mind_nerve_bin` MUST be an absolute path (no `$PATH` lookup).
- `upstream_command` MAY be a bare name (resolved via `$PATH` at façade
  startup; this is an upstream trust decision, not a mind-nerve one).
- `top_k ∈ [1, 64]`.
- `telemetry ∈ {"evidence_log", "disabled"}`.
- `on_subprocess_error ∈ {"passthrough", "deny"}`. `deny` means the façade
  returns JSON-RPC `-32603 Internal error` on any mind-nerve failure;
  `passthrough` forwards the full upstream `tools/list`.

---

## 9. Failure modes

Each failure: what the client sees, what is logged, what is recovered.

### F1: mind-nerve binary missing at startup

- **Client sees**: full upstream `tools/list` (passthrough mode).
- **Logged**: `subprocess_spawn_failed binary=<path> error=ENOENT` on stderr.
- **Recovery**: façade continues in passthrough mode. On every health-check
  interval (30 s), retry spawn. On successful retry, exit passthrough mode
  and chain a `chain_reset_reason: model_loaded` envelope.

### F2: mind-nerve subprocess crashes mid-call

- **Client sees**: for the in-flight `tools/list`, the full upstream list
  (single-call passthrough). Subsequent calls retry once with a respawned
  subprocess; if that retry also fails, persistent passthrough.
- **Logged**: `subprocess_crash request_id=<id> exit_code=<n>` plus a
  `chain_reset_reason: subprocess_restart` envelope on the next successful
  call.
- **Recovery**: SIGKILL + respawn within 1 s. Failed-twice-in-60-s policy
  applies (§3 Health check).

### F3: upstream MCP server crashes

- **Client sees**: stdio disconnect (façade closes the client-facing pipe).
  Standard MCP behaviour.
- **Logged**: `upstream_disconnect command=<cmd> exit_code=<n>`.
- **Recovery**: façade exits. Process supervisor (systemd, launchctl, agent
  runtime) is expected to restart the façade, which respawns upstream from
  scratch. mind-nerve subprocess is also respawned by the new façade.

### F4: Catalog hash mismatch on `catalog_refresh`

- **Client sees**: nothing immediately. Next `tools/list` returns a possibly
  different top-K reflecting the new catalog.
- **Logged**: warning + `chain_reset_reason: catalog_changed` evidence
  envelope.
- **Recovery**: cache invalidation. No restart.

### F5: mic@2 parse error from mind-nerve reply

- **Client sees**: for this single call, full upstream `tools/list`
  (single-call passthrough).
- **Logged**: `mic2_parse_error request_id=<id> bytes=<hex_preview>` plus
  the subprocess is killed and respawned (the framing is corrupt; safer to
  reset than to skip-and-pray).
- **Recovery**: SIGKILL + respawn. Same retry policy as F2.

### F6: Upstream `tools/list` exceeds 4 MB cap

- **Client sees**: JSON-RPC `-32603 Internal error` with
  `data: { reason: "upstream_tools_list_too_large" }`.
- **Logged**: error with bytes count.
- **Recovery**: none — this is a real misconfiguration. Operator must
  reduce upstream catalog or raise the cap (config tunable not exposed in
  Phase 1; future work).

### F7: Configuration file missing or invalid

- **Client sees**: façade exits before binding stdio. Client sees connection
  refused.
- **Logged**: stderr, structured error pointing to the bad key.
- **Recovery**: operator fixes config and restarts.

---

## 10. Security

### Subprocess spawn

All `exec` calls use argv-list form. No shell interpolation. Spawning is
done with `subprocess.Popen(args=[bin, *flags], shell=False)` semantics or
equivalent across implementation languages. No `os.system`, no
`shell=True`. Eliminates command-injection vectors via tool names or config
values.

### Path resolution

`mind_nerve_bin` is resolved **only** from config; `$PATH` is never
consulted. This prevents PATH-injection attacks where a malicious actor
plants a binary named `mind-nerve` higher in `$PATH` than the legitimate
install.

`upstream_command` IS resolved via `$PATH` if it is a bare name — this is
the upstream's trust boundary, identical to what the calling MCP client
would have done if it had launched the upstream directly. The façade is
not introducing a new trust boundary here; it is preserving the existing
one.

### Environment inheritance

The upstream MCP server is spawned with the same environment as the façade
itself, including secrets the user has placed in their shell environment.
This is a trust boundary the façade documents explicitly:

> **Trust boundary:** the façade does NOT strip environment variables from
> the upstream subprocess. Any secret in the façade's environment is
> visible to the upstream. Operators concerned about leakage should run
> the façade under a restricted shell or use `env -i` invocation.

mind-nerve subprocess gets the façade environment PLUS
`MIND_NERVE_FACADE=1`. mind-nerve does not need most env vars and will
ignore them.

### Evidence log

Written to `~/.mind-nerve/evidence.log` with file flags
`O_APPEND | O_CLOEXEC` (POSIX) and protected by `flock(LOCK_EX)` during
each append. `flock` prevents two concurrent façades (one per upstream)
from interleaving partial mic-b records. File mode is `0600` —
owner-only. Directory mode `0700`.

### Mic@2 frame size cap

Frames from mind-nerve subprocess are length-prefixed and capped at 4 MB.
Larger frames are treated as corrupt (F5). Prevents memory-exhaustion
attacks via a compromised mind-nerve binary.

### Input validation

The façade does NOT validate JSON-RPC frame contents beyond the `method`
field. The upstream is responsible for validating tool arguments; the
client is responsible for validating tool results. The façade is wire-only.

---

## 11. Telemetry + audit

### Evidence log

Every `tools/list` interception appends one envelope to
`~/.mind-nerve/evidence.log`. Format: mic-b records, one per line
(newline-framed binary; line is base64-encoded mic-b for grep-ability).
Each record is exactly 212 bytes raw / 284 bytes base64-encoded + 1 byte
newline.

Envelope binds (per `spec/architecture.md §Attestation envelope` v2):
`model_hash`, `catalog_hash`, `request_hash`, `result_hash`,
`architecture`, `timestamp_ms`, `chain_prev`, `chain_curr`, plus the
v2-bound fields `k`, `tokenizer_hash`, `wire_version`, `entry_kind`.

### What is logged

- `tools/list` interceptions → one envelope each.
- `chain_reset` events → one envelope each, with `chain_reset_reason ∈
  {catalog_changed, model_swap, clock_reset, subprocess_restart,
  ttl_refresh, model_loaded}`.
- Catalog hash computation (at startup or on refresh) → one envelope with
  `entry_kind = catalog_load`.

### What is NOT logged

- `tools/call` invocations. Filtering is the audit point, not execution.
  Logging every tool execution would be high-volume and out of scope for
  the preselector. The upstream MCP server's own audit chain (if any) is
  the source of truth for tool execution.
- `prompts/get`, `resources/*`, `notifications/*`. The façade does not
  interpose on these.
- Health-check `noop` frames.

### Opt-out

`telemetry = "disabled"` in `[mcp.<name>]` suppresses the evidence log
entirely for that façade instance. mind-nerve still produces envelopes
internally (for verification); they are written to `/dev/null` rather than
the log. The model's own behaviour does not change.

### Verification

The evidence log is replay-verifiable: given the log, the model checkpoint
on disk, and the upstream's catalog at a point in time, an auditor can
reproduce every routing decision byte-for-byte and confirm the chain
linkage. See `spec/architecture.md §Bit-identity contract`.

---

## 12. Open questions

These are explicitly unresolved at the close of design and feed Phase 1
implementation probes (see `research/PHASE_1_PLAN.md` §5 risk-gated tasks).

1. **Does Claude Code invoke `prompts/get` every turn?** Architect open
   question Q2 (P1.4.0 / probe queued). If yes, per-turn re-classification
   triggered by `prompts/get` is the primary mechanism. If no, TTL-based
   fallback is the only path and the `fallback_turn_ttl_ms` default may
   need to drop to 1500 ms to keep top-K freshness aligned with turn pace.
   The argument field name (`arguments.user_message` vs `prompt` vs
   `messages[-1].content`) is also client-specific.

2. **MCP `tools/list` size ceiling.** JSON-RPC over stdio has no formal
   message size limit, but practical clients and frame buffers tend to
   choke past 4 MB. mind-mem v4.0.2 with 84 tools produces ~120 KB
   `tools/list` — comfortably under. A future upstream with 5,000 tools
   and verbose descriptions could approach the cap. Is 4 MB the right
   ceiling, or should it be configurable?

3. **MCP server `roots` and `sampling` features.** Both are bidirectional
   conversation patterns that the façade currently byte-forwards. If
   either feature ever requires interposed state (e.g., the upstream's
   `sampling/createMessage` needs to know which tools are visible to the
   client), pure byte-forward breaks. Phase 1 ships byte-forward; Phase 2
   may need to lift this.

4. **Multi-façade flock contention.** Two façades sharing the same
   `~/.mind-nerve/evidence.log` (e.g., one for mind-mem and one for a
   generic upstream, running concurrently in the same user session) will
   serialize on `flock`. At 1 envelope per `tools/list` and a few-Hz call
   rate, this is benign. At higher call rates the lock becomes the
   bottleneck. Should each upstream get its own evidence log, or is the
   single global log the right design?

5. **Hot-reload of config.** The current design reads
   `~/.config/mind-nerve/config.toml` at startup and never re-reads it.
   Operators changing `top_k` or `mind_nerve_bin` must restart the façade.
   Phase 1 accepts this. Phase 2 could add a `SIGHUP` reload path, but
   the semantics of mid-conversation `top_k` changes need thought
   (in-flight `tools/list` cache invalidation, chain reset envelope,
   etc.).

---

*This document is the contract. Implementation deviating from it is a bug
against this document, not against the code.*
