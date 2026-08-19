// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.

import { describe, it, expect } from "vitest";
import os from "node:os";
import path from "node:path";
import { AGENT_REGISTRY, ALL_CLIENT_NAMES, requireSpec } from "../src/registry.js";

/** Total clients in the registry: the original 17 + grok, kimi, qwen. */
const TOTAL_CLIENTS = 20;

/**
 * The six CLIs that announce a skills directory, with the shapes verified on
 * disk. Each entry is
 * `[client, skillsDir, hookConfigPath, hookWireFmt, hookEvents]`.
 *
 * gemini is the one outlier: its bundle maps UserPromptSubmit -> BeforeAgent
 * and has no separate UserPromptSubmit event (b10 live-integration).
 */
const SKILL_SURFACE_CLIS = [
  ["claude-code", ".claude/skills", ".claude/settings.json", "json-hooks", ["UserPromptSubmit", "SessionStart"]],
  ["codex", ".codex/skills", ".codex/config.toml", "toml-hooks", ["UserPromptSubmit", "SessionStart"]],
  ["gemini", ".gemini/skills", ".gemini/settings.json", "json-hooks", ["BeforeAgent", "SessionStart"]],
  ["grok", ".grok/skills", ".grok/config.toml", "toml-hooks", ["UserPromptSubmit", "SessionStart"]],
  ["kimi", ".kimi-code/skills", ".kimi-code/config.toml", "toml-hooks-kimi", ["UserPromptSubmit", "SessionStart"]],
  ["qwen", ".qwen/skills", ".qwen/settings.json", "json-hooks", ["UserPromptSubmit", "SessionStart"]],
] as const;

const home = os.homedir();

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
  it("contains exactly 20 clients", () => {
    expect(AGENT_REGISTRY.size).toBe(TOTAL_CLIENTS);
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
    expect(ALL_CLIENT_NAMES.length).toBe(TOTAL_CLIENTS);
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

  it("kimi uses the documented shapes: mcp.json + flat [[hooks]] (docs 2026-08-11)", () => {
    // Official kimi-code docs: MCP servers live in ~/.kimi-code/mcp.json
    // ({"mcpServers": {...}}, Claude-Desktop-compatible); hooks are ONLY
    // [[hooks]] array elements in config.toml. The config.toml [mcp_servers]
    // surface and the Claude [[hooks.<Event>]] shape are never read.
    const spec = AGENT_REGISTRY.get("kimi");
    expect(spec).toBeDefined();
    expect(spec!.mcpPath).toBe(path.join(home, ".kimi-code", "mcp.json"));
    expect(spec!.mcpFmt).toBe("mcp-json-servers");
    expect(spec!.skillSurface?.hookWireFmt).toBe("toml-hooks-kimi");
    expect(spec!.skillSurface?.hookConfigPath).toBe(
      path.join(home, ".kimi-code", "config.toml"),
    );
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

// ---------------------------------------------------------------------------
// Skill surfaces — the six CLIs whose shapes were verified on disk.
// ---------------------------------------------------------------------------

describe("AGENT_REGISTRY skill surfaces", () => {
  it("covers exactly the six skill-surface CLIs", () => {
    const withSurface = [...AGENT_REGISTRY.values()]
      .filter((s) => s.skillSurface !== null)
      .map((s) => s.name)
      .sort();
    expect(withSurface).toEqual(
      SKILL_SURFACE_CLIS.map(([n]) => n)
        .slice()
        .sort(),
    );
  });

  for (const [name, skillsRel, configRel, fmt, events] of SKILL_SURFACE_CLIS) {
    describe(name, () => {
      const spec = AGENT_REGISTRY.get(name);

      it("is registered", () => {
        expect(spec, `${name} missing from registry`).toBeDefined();
        expect(spec!.skillSurface).not.toBeNull();
      });

      it("points at the verified skills directory", () => {
        expect(spec!.skillSurface!.skillsDir).toBe(path.join(home, skillsRel));
      });

      it("points at the verified hook config file", () => {
        expect(spec!.skillSurface!.hookConfigPath).toBe(
          path.join(home, configRel),
        );
      });

      it("uses the verified hook wire format", () => {
        expect(spec!.skillSurface!.hookWireFmt).toBe(fmt);
      });

      it(`registers on ${events.join(" and ")}`, () => {
        expect([...spec!.skillSurface!.hookEvents]).toEqual([...events]);
      });

      it("installs its hook wrapper under the CLI's own directory", () => {
        const { hookScriptPath, hookConfigPath } = spec!.skillSurface!;
        expect(hookScriptPath).toContain("mind-nerve-hook");
        // The wrapper must live inside the same CLI dir as its config, so
        // uninstalling one client never touches another's files.
        expect(hookScriptPath.startsWith(path.dirname(hookConfigPath))).toBe(
          true,
        );
      });

      it("carries a non-empty on-disk verification note", () => {
        expect(spec!.skillSurface!.verified.length).toBeGreaterThan(20);
      });

      it("has a positive hook timeout", () => {
        expect(spec!.skillSurface!.hookTimeoutSecs).toBeGreaterThan(0);
      });
    });
  }

  it("gives every skill-surface client a distinct skills dir", () => {
    const dirs = [...AGENT_REGISTRY.values()]
      .filter((s) => s.skillSurface !== null)
      .map((s) => s.skillSurface!.skillsDir);
    expect(new Set(dirs).size).toBe(dirs.length);
  });

  it("never points a skills dir at the hub itself", () => {
    for (const spec of AGENT_REGISTRY.values()) {
      if (spec.skillSurface === null) continue;
      expect(spec.skillSurface.skillsDir).not.toContain("skills-hub");
    }
  });

  it("leaves non-skill clients without a surface", () => {
    const noSurface = ["vibe", "cursor", "windsurf", "continue", "cline",
      "roo", "zed", "openclaw", "nanoclaw", "nemoclaw", "aider", "copilot",
      "cody", "qodo"];
    for (const name of noSurface) {
      expect(
        AGENT_REGISTRY.get(name)?.skillSurface,
        `${name} should have skillSurface=null`,
      ).toBeNull();
    }
  });
});
