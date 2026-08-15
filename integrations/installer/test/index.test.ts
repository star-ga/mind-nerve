// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.
//
// Tests for the `install` argv parser. Audit finding (2026-08, round 3):
// the documented `install --mcp-launcher uvx <client>` order hard-failed —
// cmdInstall took args[0] as the client, so a leading flag hit the
// startsWith("--") guard. Flags may now precede OR follow the client name.

import { describe, it, expect } from "vitest";
import { parseInstallArgs } from "../src/index.js";

describe("parseInstallArgs", () => {
  it("takes the first non-flag token as the client", () => {
    const r = parseInstallArgs(["claude-code"]);
    expect(r).not.toHaveProperty("error");
    if ("error" in r) return;
    expect(r.clientArg).toBe("claude-code");
    expect(r.mcpLauncher).toBe("venv");
  });

  it("accepts the documented flag-first --mcp-launcher order", () => {
    const r = parseInstallArgs(["--mcp-launcher", "uvx", "claude-code"]);
    expect(r).not.toHaveProperty("error");
    if ("error" in r) return;
    expect(r.clientArg).toBe("claude-code");
    expect(r.mcpLauncher).toBe("uvx");
  });

  it("accepts flags after the client name too", () => {
    const r = parseInstallArgs(["claude-code", "--mcp-launcher", "uvx"]);
    expect(r).not.toHaveProperty("error");
    if ("error" in r) return;
    expect(r.clientArg).toBe("claude-code");
    expect(r.mcpLauncher).toBe("uvx");
  });

  it("--no-wire <client> resolves the client", () => {
    const r = parseInstallArgs(["--no-wire", "gemini"]);
    expect(r).not.toHaveProperty("error");
    if ("error" in r) return;
    expect(r.clientArg).toBe("gemini");
    expect(r.noWire).toBe(true);
  });

  it("--mcp <client> resolves the client and MCP-only mode", () => {
    const r = parseInstallArgs(["--mcp", "codex"]);
    expect(r).not.toHaveProperty("error");
    if ("error" in r) return;
    expect(r.clientArg).toBe("codex");
    expect(r.mcpOnly).toBe(true);
  });

  it("--shared consumes its value, the client is still resolved", () => {
    const r = parseInstallArgs(["--shared", "a,b", "claude-code"]);
    expect(r).not.toHaveProperty("error");
    if ("error" in r) return;
    expect(r.clientArg).toBe("claude-code");
    expect(r.sharedClients).toEqual(["a", "b"]);
  });

  it("flags anywhere in argv: --all is order-independent", () => {
    const r = parseInstallArgs(["--mcp-launcher", "uvx", "--all"]);
    expect(r).not.toHaveProperty("error");
    if ("error" in r) return;
    expect(r.all).toBe(true);
    expect(r.mcpLauncher).toBe("uvx");
  });

  it("rejects a bad --mcp-launcher value", () => {
    const r = parseInstallArgs(["--mcp-launcher", "bogus", "claude-code"]);
    expect(r).toHaveProperty("error");
  });

  it("rejects --mcp-launcher with no value", () => {
    const r = parseInstallArgs(["--mcp-launcher"]);
    expect(r).toHaveProperty("error");
  });

  it("leaves clientArg undefined when no positional token exists", () => {
    const r = parseInstallArgs(["--mcp"]);
    expect(r).not.toHaveProperty("error");
    if ("error" in r) return;
    expect(r.clientArg).toBeUndefined();
  });
});
