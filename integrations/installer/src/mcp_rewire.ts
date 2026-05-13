// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.

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
 * Returns the MCP server spec for mind-nerve facade.
 * The facade binary is referenced by an absolute path resolved at install time.
 */
export function buildMcpSpec(mindNerveBin: string, upstreamConfig: string): McpServerSpec {
  return {
    command: mindNerveBin,
    args: ["mcp-facade", "--config", upstreamConfig],
    env: {},
  };
}

// ---------------------------------------------------------------------------
// Marker used in MCP entries to detect idempotency.
// ---------------------------------------------------------------------------
const MANAGED_MARKER = "mind-nerve managed";

// ---------------------------------------------------------------------------
// JSON-based mergers
// ---------------------------------------------------------------------------

/**
 * Generic { "mcpServers": { "mind-nerve": {...} } } format.
 * Used by: gemini, continue, cursor, cline, roo, windsurf, openclaw, nanoclaw, nemoclaw.
 */
function mergeJsonServers(
  existing: Record<string, unknown>,
  srv: McpServerSpec,
): { updated: Record<string, unknown>; changed: boolean } {
  const out: Record<string, unknown> = JSON.parse(JSON.stringify(existing));
  const servers = (out["mcpServers"] ?? {}) as Record<string, unknown>;

  const target = {
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
 *     { name = "mind-nerve", command = "...", args = [...], env = {...} }
 *   ]
 */
function mergeTomlVibe(existingText: string, srv: McpServerSpec): { updated: string; changed: boolean } {
  const argsToml = "[" + srv.args.map((a) => JSON.stringify(a)).join(", ") + "]";
  const envPairs = Object.entries(srv.env)
    .map(([k, v]) => `${k} = ${JSON.stringify(v)}`)
    .join(", ");
  const envSection = envPairs ? `{ ${envPairs} }` : "{}";

  const mmEntry =
    `  { name = "mind-nerve", command = ${JSON.stringify(srv.command)}, ` +
    `args = ${argsToml}, env = ${envSection} } # ${MANAGED_MARKER}`;

  const text = existingText ?? "";

  // Idempotency: marker present AND command matches.
  if (text.includes(MANAGED_MARKER) && text.includes(JSON.stringify(srv.command))) {
    return { updated: text, changed: false };
  }

  const blockMatch = /^mcp_servers\s*=\s*\[([\s\S]*?)\]\s*$/m.exec(text);
  if (blockMatch === null) {
    const newBlock = `mcp_servers = [\n${mmEntry}\n]\n`;
    const separator = text.trimEnd().length > 0 ? "\n\n" : "";
    return { updated: text.trimEnd() + separator + newBlock, changed: true };
  }

  const body = blockMatch[1] ?? "";
  const cleanedBody = body
    .split("\n")
    .filter((l) => !/"mind-nerve"/.test(l))
    .join("\n")
    .trim()
    .replace(/,\s*$/, "");

  const newBody = cleanedBody ? cleanedBody + ",\n" + mmEntry : mmEntry;
  const newBlock = `mcp_servers = [\n${newBody}\n]`;
  const start = blockMatch.index;
  const end = start + blockMatch[0].length;
  return {
    updated: text.slice(0, start) + newBlock + text.slice(end),
    changed: true,
  };
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
      return mergeJsonServers(existing, srv);
    case "mcp-json-cursor":
      return mergeJsonServers(existing, srv);
    case "mcp-json-windsurf":
      return mergeJsonServers(existing, srv);
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
