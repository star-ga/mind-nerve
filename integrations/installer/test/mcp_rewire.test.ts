// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.

import { describe, it, expect } from "vitest";
import {
  buildMcpSpec,
  mergeJsonMcp,
  mergeTomlMcp,
  isJsonMcpFmt,
  isTomlMcpFmt,
} from "../src/mcp_rewire.js";

const FAKE_BIN = "/usr/local/bin/mind-nerve";
const FAKE_UPSTREAM = "/home/user/.config/mind-nerve/mcp/test.toml";

const SRV = buildMcpSpec(FAKE_BIN, FAKE_UPSTREAM);

describe("buildMcpSpec", () => {
  it("uses the provided binary and upstream config", () => {
    expect(SRV.command).toBe(FAKE_BIN);
    expect(SRV.args).toContain(FAKE_UPSTREAM);
    expect(SRV.args[0]).toBe("mcp-facade");
  });
});

describe("mergeJsonMcp — mcp-json-servers (generic)", () => {
  it("adds mind-nerve entry to empty mcpServers", () => {
    const { updated, changed } = mergeJsonMcp("mcp-json-servers", {}, SRV, "test");
    expect(changed).toBe(true);
    const servers = updated["mcpServers"] as Record<string, unknown>;
    expect(servers).toHaveProperty("mind-nerve");
  });

  it("adds mind-nerve to existing mcpServers without touching other entries", () => {
    const existing = {
      mcpServers: { "other-tool": { command: "node", args: ["server.js"] } },
    };
    const { updated, changed } = mergeJsonMcp("mcp-json-servers", existing, SRV, "test");
    expect(changed).toBe(true);
    const servers = updated["mcpServers"] as Record<string, unknown>;
    expect(servers).toHaveProperty("other-tool");
    expect(servers).toHaveProperty("mind-nerve");
  });

  it("is idempotent on second call", () => {
    const { updated: first } = mergeJsonMcp("mcp-json-servers", {}, SRV, "test");
    const { changed: secondChanged } = mergeJsonMcp("mcp-json-servers", first, SRV, "test");
    expect(secondChanged).toBe(false);
  });

  it("contains managed marker", () => {
    const { updated } = mergeJsonMcp("mcp-json-servers", {}, SRV, "test");
    const raw = JSON.stringify(updated);
    expect(raw).toContain("mind-nerve managed");
  });
});

describe("mergeJsonMcp — mcp-json-zed", () => {
  it("adds mind-nerve to context_servers", () => {
    const { updated, changed } = mergeJsonMcp("mcp-json-zed", {}, SRV, "zed");
    expect(changed).toBe(true);
    const ctx = updated["context_servers"] as Record<string, unknown>;
    expect(ctx).toHaveProperty("mind-nerve");
  });

  it("sets source=custom on zed entries", () => {
    const { updated } = mergeJsonMcp("mcp-json-zed", {}, SRV, "zed");
    const ctx = updated["context_servers"] as Record<string, unknown>;
    const entry = ctx["mind-nerve"] as Record<string, unknown>;
    expect(entry["source"]).toBe("custom");
  });

  it("is idempotent", () => {
    const { updated: first } = mergeJsonMcp("mcp-json-zed", {}, SRV, "zed");
    const { changed: second } = mergeJsonMcp("mcp-json-zed", first, SRV, "zed");
    expect(second).toBe(false);
  });

  it("preserves existing assistant settings", () => {
    const existing = { assistant: { model: "claude-opus-4-7" }, context_servers: {} };
    const { updated } = mergeJsonMcp("mcp-json-zed", existing, SRV, "zed");
    expect(updated["assistant"]).toEqual({ model: "claude-opus-4-7" });
  });
});

describe("mergeTomlMcp — mcp-toml-codex", () => {
  it("adds mcp_servers section to empty file", () => {
    const { updated, changed } = mergeTomlMcp("mcp-toml-codex", "", SRV, "codex");
    expect(changed).toBe(true);
    expect(updated).toContain("[mcp_servers.mind-nerve]");
    expect(updated).toContain(FAKE_BIN);
  });

  it("adds mcp_servers section preserving existing content", () => {
    const existing = "[model]\nname = \"codex-latest\"\n";
    const { updated, changed } = mergeTomlMcp("mcp-toml-codex", existing, SRV, "codex");
    expect(changed).toBe(true);
    expect(updated).toContain("[model]");
    expect(updated).toContain("[mcp_servers.mind-nerve]");
  });

  it("is idempotent on second call", () => {
    const { updated: first } = mergeTomlMcp("mcp-toml-codex", "", SRV, "codex");
    const { changed: secondChanged } = mergeTomlMcp("mcp-toml-codex", first, SRV, "codex");
    expect(secondChanged).toBe(false);
  });

  it("replaces an existing mind-nerve section cleanly", () => {
    const existing =
      "# mind-nerve managed\n[mcp_servers.mind-nerve]\ncommand = \"/old/bin\"\nargs = []\nenv = {}\n";
    const { updated, changed } = mergeTomlMcp("mcp-toml-codex", existing, SRV, "codex");
    expect(changed).toBe(true);
    expect(updated).toContain(FAKE_BIN);
    expect(updated).not.toContain("/old/bin");
    // Should not have duplicate sections.
    const count = (updated.match(/\[mcp_servers\.mind-nerve\]/g) ?? []).length;
    expect(count).toBe(1);
  });

  it("contains the managed marker", () => {
    const { updated } = mergeTomlMcp("mcp-toml-codex", "", SRV, "codex");
    expect(updated).toContain("mind-nerve managed");
  });
});

describe("mergeTomlMcp — mcp-toml-vibe", () => {
  it("creates mcp_servers array if missing", () => {
    const { updated, changed } = mergeTomlMcp("mcp-toml-vibe", "[model]\nname = \"x\"\n", SRV, "vibe");
    expect(changed).toBe(true);
    expect(updated).toContain('name = "mind-nerve"');
  });

  it("appends to existing mcp_servers array without removing other entries", () => {
    const existing =
      'mcp_servers = [\n  { name = "other", command = "node", args = [], env = {} }\n]\n';
    const { updated, changed } = mergeTomlMcp("mcp-toml-vibe", existing, SRV, "vibe");
    expect(changed).toBe(true);
    expect(updated).toContain('"other"');
    expect(updated).toContain('"mind-nerve"');
  });

  it("is idempotent", () => {
    const { updated: first } = mergeTomlMcp("mcp-toml-vibe", "", SRV, "vibe");
    const { changed: secondChanged } = mergeTomlMcp("mcp-toml-vibe", first, SRV, "vibe");
    expect(secondChanged).toBe(false);
  });

  it("replaces existing mind-nerve entry without creating duplicates", () => {
    const existing =
      `mcp_servers = [\n  { name = "mind-nerve", command = "/old/bin", args = [], env = {} } # mind-nerve managed\n]\n`;
    const { updated } = mergeTomlMcp("mcp-toml-vibe", existing, SRV, "vibe");
    const count = (updated.match(/"mind-nerve"/g) ?? []).length;
    expect(count).toBe(1);
    expect(updated).toContain(FAKE_BIN);
  });
});

describe("format discriminators", () => {
  it("isJsonMcpFmt returns true for JSON formats", () => {
    expect(isJsonMcpFmt("mcp-json-servers")).toBe(true);
    expect(isJsonMcpFmt("mcp-json-cursor")).toBe(true);
    expect(isJsonMcpFmt("mcp-json-windsurf")).toBe(true);
    expect(isJsonMcpFmt("mcp-json-zed")).toBe(true);
    expect(isJsonMcpFmt("mcp-toml-codex")).toBe(false);
    expect(isJsonMcpFmt(null)).toBe(false);
  });

  it("isTomlMcpFmt returns true for TOML formats", () => {
    expect(isTomlMcpFmt("mcp-toml-codex")).toBe(true);
    expect(isTomlMcpFmt("mcp-toml-vibe")).toBe(true);
    expect(isTomlMcpFmt("mcp-json-servers")).toBe(false);
    expect(isTomlMcpFmt(null)).toBe(false);
  });
});
