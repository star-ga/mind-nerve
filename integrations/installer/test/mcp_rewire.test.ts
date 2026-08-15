// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.

import { describe, it, expect } from "vitest";
import path from "node:path";
import TOML from "@iarna/toml";
import {
  buildMcpSpec,
  mergeJsonMcp,
  mergeTomlMcp,
  isJsonMcpFmt,
  isTomlMcpFmt,
} from "../src/mcp_rewire.js";

const FAKE_BIN = "/usr/local/bin/mind-nerve";

const SRV = buildMcpSpec(FAKE_BIN);

describe("buildMcpSpec", () => {
  it("pins the sibling mind-nerve-mcp console script with empty args", () => {
    expect(SRV.command).toBe(path.join(path.dirname(FAKE_BIN), "mind-nerve-mcp"));
    expect(SRV.args).toEqual([]);
    expect(SRV.env).toEqual({ TRANSFORMERS_NO_TORCHVISION: "1" });
  });
});

describe("buildMcpSpec — uvx launcher", () => {
  const UVX_SRV = buildMcpSpec(FAKE_BIN, "uvx");

  it("launches via uvx --from mind-nerve mind-nerve-mcp", () => {
    expect(UVX_SRV.command).toBe("uvx");
    expect(UVX_SRV.args).toEqual(["--from", "mind-nerve", "mind-nerve-mcp"]);
  });

  it("keeps the same env-pin shape as the venv spec", () => {
    expect(UVX_SRV.env).toEqual(SRV.env);
  });

  it("writes a valid TOML entry for a TOML CLI (codex)", () => {
    const { updated, changed } = mergeTomlMcp("mcp-toml-codex", "", UVX_SRV, "codex");
    expect(changed).toBe(true);
    expect(updated).toContain("[mcp_servers.mind-nerve]");
    expect(updated).toContain('command = "uvx"');
    expect(updated).toContain('args = ["--from", "mind-nerve", "mind-nerve-mcp"]');
    // The generated section must actually parse as TOML.
    const parsed = TOML.parse(updated) as Record<string, unknown>;
    const servers = parsed["mcp_servers"] as Record<string, unknown>;
    const entry = servers["mind-nerve"] as Record<string, unknown>;
    expect(entry["command"]).toBe("uvx");
    expect(entry["args"]).toEqual(["--from", "mind-nerve", "mind-nerve-mcp"]);
  });

  it("is idempotent for a TOML CLI on second call", () => {
    const { updated: first } = mergeTomlMcp("mcp-toml-codex", "", UVX_SRV, "codex");
    const { changed: second } = mergeTomlMcp("mcp-toml-codex", first, UVX_SRV, "codex");
    expect(second).toBe(false);
  });

  it("writes an argv-array entry for a JSON CLI (generic mcpServers)", () => {
    const { updated, changed } = mergeJsonMcp("mcp-json-servers", {}, UVX_SRV, "test");
    expect(changed).toBe(true);
    const servers = updated["mcpServers"] as Record<string, unknown>;
    const entry = servers["mind-nerve"] as Record<string, unknown>;
    expect(entry["command"]).toBe("uvx");
    expect(entry["args"]).toEqual(["--from", "mind-nerve", "mind-nerve-mcp"]);
    expect(entry["env"]).toEqual(UVX_SRV.env);
    expect(entry["_comment"]).toBe("mind-nerve managed");
  });

  it("is idempotent for a JSON CLI on second call", () => {
    const { updated: first } = mergeJsonMcp("mcp-json-servers", {}, UVX_SRV, "test");
    const { changed: second } = mergeJsonMcp("mcp-json-servers", first, UVX_SRV, "test");
    expect(second).toBe(false);
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

  it("never lands a bare key inside a trailing [[hooks]] element (scoping regression)", () => {
    // Observed live 2026-08-10 on kimi's config.toml: an EOF append after a
    // [[hooks]] block scoped mcp_servers INTO the last hooks element — dead
    // weight kimi never read. The block must go above the first header.
    const existing =
      '[model]\ndefault = "x"\n\n[[hooks]]\nevent = "UserPromptSubmit"\ncommand = "/usr/bin/other"\ntimeout = 8\n';
    const { updated, changed } = mergeTomlMcp("mcp-toml-vibe", existing, SRV, "vibe");
    expect(changed).toBe(true);
    const parsed = TOML.parse(updated) as Record<string, unknown>;
    // Top-level scope: mcp_servers is a root key...
    const servers = parsed["mcp_servers"] as Array<Record<string, unknown>>;
    expect(Array.isArray(servers)).toBe(true);
    expect(servers.some((s) => s["name"] === "mind-nerve")).toBe(true);
    // ...and the hooks element was not polluted.
    const hooks = parsed["hooks"] as Array<Record<string, unknown>>;
    expect(hooks).toHaveLength(1);
    expect(hooks[0]).toEqual({
      event: "UserPromptSubmit",
      command: "/usr/bin/other",
      timeout: 8,
    });
    expect(parsed["model"]).toEqual({ default: "x" });
  });

  it("relocates a previously mis-scoped mcp_servers block to the top level", () => {
    // The exact wreckage the old EOF-append produced: the key sits inside
    // the last [[hooks]] element, marker included. Repair = move it out.
    const existing =
      '[[hooks]]\nevent = "SessionStart"\ncommand = "/usr/bin/other"\n\n' +
      'mcp_servers = [\n  { name = "mind-nerve", command = "/old/bin", args = [], env = {} } # mind-nerve managed\n]\n';
    const { updated, changed } = mergeTomlMcp("mcp-toml-vibe", existing, SRV, "vibe");
    expect(changed).toBe(true);
    const parsed = TOML.parse(updated) as Record<string, unknown>;
    const servers = parsed["mcp_servers"] as Array<Record<string, unknown>>;
    const entry = servers.find((s) => s["name"] === "mind-nerve");
    expect(entry?.["command"]).toBe(SRV.command);
    const hooks = parsed["hooks"] as Array<Record<string, unknown>>;
    expect(hooks).toHaveLength(1);
    expect(hooks[0]).not.toHaveProperty("mcp_servers");
  });

  it("is idempotent after a scoped insert", () => {
    const existing = '[model]\ndefault = "x"\n';
    const first = mergeTomlMcp("mcp-toml-vibe", existing, SRV, "vibe").updated;
    const second = mergeTomlMcp("mcp-toml-vibe", first, SRV, "vibe");
    expect(second.changed).toBe(false);
  });

  it("emits transport = \"stdio\" (codex#16: Vibe 2.9.6 rejects entries without it)", () => {
    const { updated } = mergeTomlMcp("mcp-toml-vibe", "", SRV, "vibe");
    const parsed = TOML.parse(updated) as Record<string, unknown>;
    const servers = parsed["mcp_servers"] as Array<Record<string, unknown>>;
    const entry = servers.find((s) => s["name"] === "mind-nerve");
    expect(entry?.["transport"]).toBe("stdio");
  });

  it("rewrites when args/env drift even though marker+command match (codex#18)", () => {
    const first = mergeTomlMcp("mcp-toml-vibe", "", SRV, "vibe").updated;
    // Same marker + command, drifted env — the old idempotency check waved
    // this through and the stale pin survived.
    const drifted = first.replace(
      'TRANSFORMERS_NO_TORCHVISION = "1"',
      'TRANSFORMERS_NO_TORCHVISION = "0"',
    );
    const second = mergeTomlMcp("mcp-toml-vibe", drifted, SRV, "vibe");
    expect(second.changed).toBe(true);
    expect(second.updated).toContain('TRANSFORMERS_NO_TORCHVISION = "1"');
    expect(second.updated).not.toContain('TRANSFORMERS_NO_TORCHVISION = "0"');
    // Still exactly one managed entry.
    const count = (second.updated.match(/"mind-nerve"/g) ?? []).length;
    expect(count).toBe(1);
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
