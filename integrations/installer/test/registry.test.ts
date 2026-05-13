// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.

import { describe, it, expect } from "vitest";
import { AGENT_REGISTRY, ALL_CLIENT_NAMES, requireSpec } from "../src/registry.js";

const VALID_CONFIG_FMTS = new Set([
  "json-claude-hooks",
  "json-openclaw-hooks",
  "json-gemini",
  "json-continue",
  "json-zed",
  "json-generic",
  "toml-codex",
  "toml-vibe",
  "yaml-aider",
  "text-block",
]);

const VALID_MCP_FMTS = new Set([
  "mcp-json-servers",
  "mcp-json-cursor",
  "mcp-json-windsurf",
  "mcp-json-zed",
  "mcp-toml-codex",
  "mcp-toml-vibe",
  null,
]);

describe("AGENT_REGISTRY", () => {
  it("contains exactly 17 clients", () => {
    expect(AGENT_REGISTRY.size).toBe(17);
  });

  it("has no duplicate names", () => {
    const names = new Set<string>();
    for (const spec of AGENT_REGISTRY.values()) {
      expect(names.has(spec.name), `Duplicate name: ${spec.name}`).toBe(false);
      names.add(spec.name);
    }
  });

  it("ALL_CLIENT_NAMES matches registry keys in order", () => {
    expect(ALL_CLIENT_NAMES).toEqual([...AGENT_REGISTRY.keys()]);
    expect(ALL_CLIENT_NAMES.length).toBe(17);
  });

  it("each spec has a valid configFmt", () => {
    for (const spec of AGENT_REGISTRY.values()) {
      expect(VALID_CONFIG_FMTS.has(spec.configFmt), `${spec.name}: invalid configFmt ${spec.configFmt}`).toBe(true);
    }
  });

  it("each spec has a valid mcpFmt (including null)", () => {
    for (const spec of AGENT_REGISTRY.values()) {
      expect(VALID_MCP_FMTS.has(spec.mcpFmt), `${spec.name}: invalid mcpFmt ${String(spec.mcpFmt)}`).toBe(true);
    }
  });

  it("each spec has at least one detection path or binary, or alwaysOffer", () => {
    for (const spec of AGENT_REGISTRY.values()) {
      const hasSomething =
        spec.alwaysOffer ||
        spec.detectBinaries.length > 0 ||
        spec.detectPaths.length > 0;
      expect(hasSomething, `${spec.name}: no detection mechanism`).toBe(true);
    }
  });

  it("each spec has a non-empty name and description", () => {
    for (const spec of AGENT_REGISTRY.values()) {
      expect(spec.name.length).toBeGreaterThan(0);
      expect(spec.description.length).toBeGreaterThan(0);
    }
  });

  it("clients with MCP surface have mcpPath set", () => {
    for (const spec of AGENT_REGISTRY.values()) {
      if (spec.mcpFmt !== null) {
        expect(spec.mcpPath, `${spec.name}: mcpFmt set but mcpPath is null`).not.toBeNull();
      }
    }
  });

  it("clients without MCP surface have mcpPath null", () => {
    const noMcpClients = ["aider", "copilot", "cody", "qodo"];
    for (const name of noMcpClients) {
      const spec = AGENT_REGISTRY.get(name);
      expect(spec).toBeDefined();
      expect(spec!.mcpFmt).toBeNull();
      expect(spec!.mcpPath).toBeNull();
    }
  });

  it("copilot has alwaysOffer=true", () => {
    const spec = AGENT_REGISTRY.get("copilot");
    expect(spec?.alwaysOffer).toBe(true);
  });

  it("all other clients have alwaysOffer=false", () => {
    for (const [name, spec] of AGENT_REGISTRY) {
      if (name !== "copilot") {
        expect(spec.alwaysOffer, `${name} should have alwaysOffer=false`).toBe(false);
      }
    }
  });

  it("claude-code has projectionDir set", () => {
    const spec = AGENT_REGISTRY.get("claude-code");
    expect(spec?.projectionDir).not.toBeNull();
    expect(spec?.projectionDir).toContain("mind-nerve");
    expect(spec?.projectionDir).toContain("claude-code");
  });

  it("clients without skill surface have projectionDir null", () => {
    const noSkillClients = ["codex", "vibe", "gemini", "cursor", "windsurf",
      "continue", "cline", "roo", "zed", "openclaw", "nanoclaw", "nemoclaw",
      "aider", "copilot", "cody", "qodo"];
    for (const name of noSkillClients) {
      const spec = AGENT_REGISTRY.get(name);
      expect(spec?.projectionDir, `${name} should have projectionDir=null`).toBeNull();
    }
  });

  it("requireSpec returns the spec for a known client", () => {
    const spec = requireSpec("claude-code");
    expect(spec.name).toBe("claude-code");
  });

  it("requireSpec throws for an unknown client", () => {
    expect(() => requireSpec("nonexistent-client-xyz")).toThrow("Unknown client");
  });

  it("workspace-rules clients have instructionFilePath set", () => {
    const wsClients = ["cursor", "windsurf", "aider", "copilot", "cody",
      "qodo", "cline", "roo"];
    for (const name of wsClients) {
      const spec = AGENT_REGISTRY.get(name);
      expect(spec?.instructionFilePath, `${name} should have instructionFilePath`).not.toBeNull();
    }
  });

  it("non-workspace-rules clients with own config format have instructionFilePath null", () => {
    const nonWsClients = ["claude-code", "codex", "vibe", "gemini",
      "continue", "zed", "openclaw", "nanoclaw", "nemoclaw"];
    for (const name of nonWsClients) {
      const spec = AGENT_REGISTRY.get(name);
      expect(spec?.instructionFilePath, `${name} should have instructionFilePath=null`).toBeNull();
    }
  });
});
