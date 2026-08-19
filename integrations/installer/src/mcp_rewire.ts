// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.

import path from "node:path";
import type { McpFmt } from "./registry.js";
import { InstallerError } from "./errors.js";
import TOML from "@iarna/toml";

// ---------------------------------------------------------------------------
// MCP server spec — what we inject into each client's MCP config.
// ---------------------------------------------------------------------------

export interface McpServerSpec {
  command: string;
  args: readonly string[];
  env: Readonly<Record<string, string>>;
}

/**
 * How the MCP server entry is launched:
 *  - "venv": absolute path to the mind-nerve-mcp console script, resolved as
 *    the sibling of the mind-nerve binary in the same venv. Zero first-launch
 *    latency, works offline, but breaks on hosts without that venv.
 *  - "uvx":  `uvx --from mind-nerve mind-nerve-mcp` — uv resolves the
 *    published PyPI package on first launch, so the host needs no pre-built
 *    venv. First launch pays a download; offline hosts should stay on "venv".
 */
export type McpLauncher = "venv" | "uvx";

/** PyPI package the uvx launcher resolves. */
export const UVX_PACKAGE = "mind-nerve";
/** Entry point inside the published package. */
export const UVX_ENTRYPOINT = "mind-nerve-mcp";

/**
 * Returns the MCP server spec for the mind-nerve MCP server.
 *
 * The command is ALWAYS the `mind-nerve-mcp` entry point with empty args —
 * the Python package has no `mcp-facade` subcommand; entries written in that
 * shape by older installers are dead (verify flags them).
 *
 * Env pins:
 *  - TRANSFORMERS_NO_TORCHVISION=1 always (the torch-vision import path is
 *    flaky on minimal hosts and unused).
 *  - MIND_NERVE_RUNTIME_DIR when a runtime dir is configured or the default
 *    exists AND is populated (manifest.json/route_table.jsonl) — without it
 *    the MCP silently loads the OSS 11.9k catalog instead of the local route
 *    table (fleet incident 2026-07-10). Null for a fresh PyPI host whose
 *    runtime auto-seeds from Hugging Face.
 */
export function buildMcpSpec(
  mindNerveBin: string,
  launcher: McpLauncher = "venv",
  runtimeDir: string | null = null,
): McpServerSpec {
  const env: Record<string, string> = { TRANSFORMERS_NO_TORCHVISION: "1" };
  if (runtimeDir !== null) {
    env["MIND_NERVE_RUNTIME_DIR"] = runtimeDir;
  }
  if (launcher === "uvx") {
    return {
      command: "uvx",
      args: ["--from", UVX_PACKAGE, UVX_ENTRYPOINT],
      env,
    };
  }
  return {
    // The console script sits next to the mind-nerve binary in the same venv.
    command: path.join(path.dirname(mindNerveBin), UVX_ENTRYPOINT),
    args: [],
    env,
  };
}

// ---------------------------------------------------------------------------
// Marker used in MCP entries to detect idempotency.
// ---------------------------------------------------------------------------
const MANAGED_MARKER = "mind-nerve managed";

/**
 * gemini-cli's settings.json schema is closed (unknown top-level keys on an
 * MCP server entry are rejected/stripped — b10 live-integration), so it
 * cannot carry a `_comment` field the way the other `mcpServers`-shaped
 * clients do. Its managed marker instead lives inside `env`, a field every
 * MCP client already accepts arbitrary keys in.
 */
const MANAGED_ENV_KEY = "MIND_NERVE_MANAGED";

// ---------------------------------------------------------------------------
// JSON-based mergers
// ---------------------------------------------------------------------------

/**
 * Generic { "mcpServers": { "mind-nerve": {...} } } format.
 * Used by: gemini, continue, cursor, cline, roo, windsurf, openclaw, nanoclaw, nemoclaw.
 *
 * `clientName` selects the managed-marker shape: every client gets a
 * `_comment` field EXCEPT gemini, which gets an env-var marker instead (see
 * `MANAGED_ENV_KEY`).
 */
function mergeJsonServers(
  existing: Record<string, unknown>,
  srv: McpServerSpec,
  clientName: string | null = null,
): { updated: Record<string, unknown>; changed: boolean } {
  const out: Record<string, unknown> = JSON.parse(JSON.stringify(existing));
  const servers = (out["mcpServers"] ?? {}) as Record<string, unknown>;

  const target: Record<string, unknown> =
    clientName === "gemini"
      ? {
          command: srv.command,
          args: [...srv.args],
          env: { ...srv.env, [MANAGED_ENV_KEY]: "1" },
        }
      : {
          command: srv.command,
          args: [...srv.args],
          env: { ...srv.env },
          _comment: MANAGED_MARKER,
        };

  const existing_entry = servers["mind-nerve"];
  if (existing_entry !== undefined && JSON.stringify(existing_entry) === JSON.stringify(target)) {
    return { updated: out, changed: false };
  }

  servers["mind-nerve"] = target;
  out["mcpServers"] = servers;
  return { updated: out, changed: true };
}

/**
 * Zed uses `context_servers` instead of `mcpServers`.
 */
function mergeJsonZed(
  existing: Record<string, unknown>,
  srv: McpServerSpec,
): { updated: Record<string, unknown>; changed: boolean } {
  const out: Record<string, unknown> = JSON.parse(JSON.stringify(existing));
  const ctx = (out["context_servers"] ?? {}) as Record<string, unknown>;

  const target = {
    source: "custom",
    command: srv.command,
    args: [...srv.args],
    env: { ...srv.env },
    _comment: MANAGED_MARKER,
  };

  const existing_entry = ctx["mind-nerve"];
  if (existing_entry !== undefined && JSON.stringify(existing_entry) === JSON.stringify(target)) {
    return { updated: out, changed: false };
  }

  ctx["mind-nerve"] = target;
  out["context_servers"] = ctx;
  return { updated: out, changed: true };
}

/**
 * Claude Code hooks format: adds mind-nerve to the mcpServers block,
 * same JSON shape as generic servers.
 */
function mergeJsonClaudeHooks(
  existing: Record<string, unknown>,
  srv: McpServerSpec,
): { updated: Record<string, unknown>; changed: boolean } {
  return mergeJsonServers(existing, srv);
}

/**
 * OpenClaw hooks format: same mcpServers key as generic.
 */
function mergeJsonOpenclawHooks(
  existing: Record<string, unknown>,
  srv: McpServerSpec,
): { updated: Record<string, unknown>; changed: boolean } {
  return mergeJsonServers(existing, srv);
}

// ---------------------------------------------------------------------------
// TOML-based mergers
// ---------------------------------------------------------------------------

/**
 * Codex CLI TOML format:
 *   [mcp_servers.mind-nerve]
 *   command = "..."
 *   args = ["..."]
 *   env = { KEY = "val" }
 */
function mergeTomlCodex(existingText: string, srv: McpServerSpec): { updated: string; changed: boolean } {
  const argsToml = "[" + srv.args.map((a) => JSON.stringify(a)).join(", ") + "]";
  const envPairs = Object.entries(srv.env)
    .map(([k, v]) => `${k} = ${JSON.stringify(v)}`)
    .join(", ");
  const envSection = envPairs ? `{ ${envPairs} }` : "{}";

  const newSection =
    `# ${MANAGED_MARKER}\n` +
    `[mcp_servers.mind-nerve]\n` +
    `command = ${JSON.stringify(srv.command)}\n` +
    `args = ${argsToml}\n` +
    `env = ${envSection}\n`;

  const text = existingText ?? "";

  // Fast idempotency path: if the exact canonical section is already present
  // and the command matches, return unchanged.
  if (text.includes(newSection.trimEnd())) {
    return { updated: text, changed: false };
  }

  // Remove any existing mind-nerve section (including sub-tables and leading marker comment).
  const cleaned = text.replace(
    /(?:# mind-nerve managed\n)?(?:\[mcp_servers\.mind-nerve(?:\.[^\]]+)?\][^\[]*)+/g,
    "",
  ).replace(/\n{3,}/g, "\n\n");

  const candidate = cleaned.trimEnd();
  const separator = candidate.length > 0 ? "\n\n" : "";
  const updated = candidate + separator + newSection;

  return { updated, changed: true };
}

/**
 * Vibe CLI TOML format:
 *   mcp_servers = [
 *     { name = "mind-nerve", transport = "stdio", command = "...", args = [...], env = {...} }
 *   ]
 *
 * `transport = "stdio"` is mandatory: Vibe 2.9.6 rejects entries without the
 * discriminator (codex#16).
 *
 * SCOPING HAZARD (observed live 2026-08-10 on kimi's config.toml): a bare
 * top-level TOML key appended at EOF lands INSIDE the last table or
 * `[[array-element]]` — the write is dead weight in the wrong scope. The
 * block must therefore be inserted ABOVE the first table header, and an
 * existing block found below a header is relocated, not edited in place.
 */
function mergeTomlVibe(existingText: string, srv: McpServerSpec): { updated: string; changed: boolean } {
  const argsToml = "[" + srv.args.map((a) => JSON.stringify(a)).join(", ") + "]";
  const envPairs = Object.entries(srv.env)
    .map(([k, v]) => `${k} = ${JSON.stringify(v)}`)
    .join(", ");
  const envSection = envPairs ? `{ ${envPairs} }` : "{}";

  const mmEntry =
    `  { name = "mind-nerve", transport = "stdio", ` +
    `command = ${JSON.stringify(srv.command)}, ` +
    `args = ${argsToml}, env = ${envSection} } # ${MANAGED_MARKER}`;

  const text = existingText ?? "";
  const newBlock = `mcp_servers = [\n${mmEntry}\n]`;

  const blockMatch = /^mcp_servers\s*=\s*\[([\s\S]*?)\]\s*$/m.exec(text);
  const misScoped =
    blockMatch !== null && !isTopLevelScope(text, blockMatch.index);

  // Idempotency: the COMPLETE managed entry (transport + command + args +
  // env + marker) is present verbatim and not stranded inside another table.
  // Comparing only marker+command let changed args/env pins silently rot
  // (codex#18). The entry line is canonical by construction, so a verbatim
  // substring check is a full equality check.
  if (!misScoped && text.includes(mmEntry)) {
    return { updated: text, changed: false };
  }

  if (blockMatch === null) {
    return { updated: insertAtTopScope(text, newBlock), changed: true };
  }

  // Merge our entry into the matched block's existing body (foreign entries
  // preserved) — the same result whether the block then stays in place or is
  // relocated out of a wrong section.
  const body = blockMatch[1] ?? "";
  const cleanedBody = body
    .split("\n")
    .filter((l) => !/"mind-nerve"/.test(l))
    .join("\n")
    .trim()
    .replace(/,\s*$/, "");

  const newBody = cleanedBody ? cleanedBody + ",\n" + mmEntry : mmEntry;
  const mergedBlock = `mcp_servers = [\n${newBody}\n]`;

  if (misScoped) {
    // Cut the stranded block out of the wrong section and re-insert the
    // merged block where TOML scoping puts it at the top level.
    const stripped =
      text.slice(0, blockMatch.index) +
      text.slice(blockMatch.index + blockMatch[0].length);
    return { updated: insertAtTopScope(stripped, mergedBlock), changed: true };
  }

  const start = blockMatch.index;
  const end = start + blockMatch[0].length;
  return {
    updated: text.slice(0, start) + mergedBlock + text.slice(end),
    changed: true,
  };
}

/**
 * True when `offset` sits above the first `[table]`/`[[array]]` header —
 * the only region where a bare key is top-level. Line-anchored heuristic,
 * same as the rest of this text-based TOML editing: a line inside a
 * multi-line array value that starts with `[` would look like a header.
 */
function isTopLevelScope(text: string, offset: number): boolean {
  return !/^[ \t]*\[\[?/m.test(text.slice(0, offset));
}

/** Inserts a bare-key block into the top-level scope of a TOML document. */
function insertAtTopScope(text: string, block: string): string {
  const header = /^[ \t]*\[\[?/m.exec(text);
  if (header === null) {
    const base = text.trimEnd();
    return (base.length > 0 ? base + "\n\n" : "") + block + "\n";
  }
  const before = text.slice(0, header.index).trimEnd();
  const after = text.slice(header.index);
  return (before.length > 0 ? before + "\n\n" : "") + block + "\n\n" + after;
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

export type MergeJsonResult = { updated: Record<string, unknown>; changed: boolean };
export type MergeTextResult = { updated: string; changed: boolean };

/**
 * Merges a JSON-format MCP config in-place.
 * Returns the updated object and whether it changed.
 */
export function mergeJsonMcp(
  fmt: McpFmt,
  existing: Record<string, unknown>,
  srv: McpServerSpec,
  clientName: string,
): MergeJsonResult {
  switch (fmt) {
    case "mcp-json-servers":
      return mergeJsonServers(existing, srv, clientName);
    case "mcp-json-cursor":
      return mergeJsonServers(existing, srv, clientName);
    case "mcp-json-windsurf":
      return mergeJsonServers(existing, srv, clientName);
    case "mcp-json-zed":
      return mergeJsonZed(existing, srv);
    default:
      // TOML formats and null must not reach this dispatcher.
      throw new InstallerError(
        "INVALID_CONFIG_FORMAT",
        clientName,
        `mergeJsonMcp called with non-JSON format: ${String(fmt)}`,
      );
  }
}

/**
 * Merges a TOML-format MCP config in-place.
 * Returns the updated text and whether it changed.
 */
export function mergeTomlMcp(
  fmt: McpFmt,
  existingText: string,
  srv: McpServerSpec,
  clientName: string,
): MergeTextResult {
  switch (fmt) {
    case "mcp-toml-codex":
      return mergeTomlCodex(existingText, srv);
    case "mcp-toml-vibe":
      return mergeTomlVibe(existingText, srv);
    default:
      throw new InstallerError(
        "INVALID_CONFIG_FORMAT",
        clientName,
        `mergeTomlMcp called with non-TOML format: ${String(fmt)}`,
      );
  }
}

/** Returns true if the McpFmt is TOML-based. */
export function isTomlMcpFmt(fmt: McpFmt): fmt is "mcp-toml-codex" | "mcp-toml-vibe" {
  return fmt === "mcp-toml-codex" || fmt === "mcp-toml-vibe";
}

/** Returns true if the McpFmt is JSON-based. */
export function isJsonMcpFmt(
  fmt: McpFmt,
): fmt is "mcp-json-servers" | "mcp-json-cursor" | "mcp-json-windsurf" | "mcp-json-zed" {
  return (
    fmt === "mcp-json-servers" ||
    fmt === "mcp-json-cursor" ||
    fmt === "mcp-json-windsurf" ||
    fmt === "mcp-json-zed"
  );
}

// Re-export for JSON config mergers used by non-MCP formatters.
export { mergeJsonClaudeHooks, mergeJsonOpenclawHooks };
