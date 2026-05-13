// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.

import { describe, it, expect, afterEach } from "vitest";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { installClient } from "../src/install.js";
import { AGENT_REGISTRY } from "../src/registry.js";
import { backupPath } from "../src/backup.js";

let tmpDirs: string[] = [];

async function makeTmp(): Promise<string> {
  const d = await fs.mkdtemp(path.join(os.tmpdir(), "mn-install-test-"));
  tmpDirs.push(d);
  return d;
}

afterEach(async () => {
  for (const d of tmpDirs) {
    await fs.rm(d, { recursive: true, force: true });
  }
  tmpDirs = [];
});

const FAKE_BIN = "/usr/local/bin/mind-nerve";

// ---------------------------------------------------------------------------
// Helpers to clone a fixture into a tmpdir.
// ---------------------------------------------------------------------------

async function cloneFixture(fixtureName: string, destPath: string): Promise<void> {
  const fixtureDir = new URL("./fixtures", import.meta.url).pathname;
  const src = path.join(fixtureDir, fixtureName);
  await fs.mkdir(path.dirname(destPath), { recursive: true });
  await fs.copyFile(src, destPath);
}

async function readJson(p: string): Promise<Record<string, unknown>> {
  const raw = await fs.readFile(p, "utf8");
  return JSON.parse(raw) as Record<string, unknown>;
}

async function readText(p: string): Promise<string> {
  return fs.readFile(p, "utf8");
}

// ---------------------------------------------------------------------------
// Tests for clients with JSON MCP surface.
// ---------------------------------------------------------------------------

describe("installClient — claude-code", () => {
  it("merges mind-nerve into mcpServers and creates a .bak", async () => {
    const tmp = await makeTmp();
    const configPath = path.join(tmp, "settings.json");
    await cloneFixture("claude-settings-before.json", configPath);

    // Build a spec pointing to tmpdir.
    const spec = {
      ...AGENT_REGISTRY.get("claude-code")!,
      configPath,
      mcpPath: configPath,
    };

    const result = await installClient(spec, {
      mindNerveBin: FAKE_BIN,
      workspace: tmp,
    });

    expect(result.changed).toBe(true);
    expect(result.idempotentNoop).toBe(false);

    // MCP entry was written.
    const written = await readJson(configPath);
    const servers = written["mcpServers"] as Record<string, unknown>;
    expect(servers).toHaveProperty("mind-nerve");
    // Existing entry preserved.
    expect(servers).toHaveProperty("existing-tool");

    // Backup was created.
    expect(result.backedUp.length).toBeGreaterThan(0);
    const bakExists = await fs.stat(result.backedUp[0]!).then(() => true, () => false);
    expect(bakExists).toBe(true);
  });

  it("is idempotent on second install", async () => {
    const tmp = await makeTmp();
    const configPath = path.join(tmp, "settings.json");
    await cloneFixture("claude-settings-before.json", configPath);

    const spec = {
      ...AGENT_REGISTRY.get("claude-code")!,
      configPath,
      mcpPath: configPath,
    };

    await installClient(spec, { mindNerveBin: FAKE_BIN, workspace: tmp });
    const result2 = await installClient(spec, {
      mindNerveBin: FAKE_BIN,
      workspace: tmp,
      _skipBackup: true,
    });

    expect(result2.idempotentNoop).toBe(true);
    expect(result2.backedUp.length).toBe(0);
  });
});

describe("installClient — codex (TOML)", () => {
  it("appends mcp_servers section to TOML config", async () => {
    const tmp = await makeTmp();
    const configPath = path.join(tmp, ".codex", "config.toml");
    await cloneFixture("codex-config-before.toml", configPath);

    const spec = {
      ...AGENT_REGISTRY.get("codex")!,
      configPath,
      mcpPath: configPath,
    };

    const result = await installClient(spec, {
      mindNerveBin: FAKE_BIN,
      workspace: tmp,
    });

    expect(result.changed).toBe(true);
    const content = await readText(configPath);
    expect(content).toContain("[mcp_servers.mind-nerve]");
    expect(content).toContain("[model]"); // original preserved
    // Backup exists.
    expect(result.backedUp.length).toBeGreaterThan(0);
  });

  it("is idempotent", async () => {
    const tmp = await makeTmp();
    const configPath = path.join(tmp, ".codex", "config.toml");
    await cloneFixture("codex-config-before.toml", configPath);

    const spec = {
      ...AGENT_REGISTRY.get("codex")!,
      configPath,
      mcpPath: configPath,
    };

    await installClient(spec, { mindNerveBin: FAKE_BIN, workspace: tmp });
    const r2 = await installClient(spec, {
      mindNerveBin: FAKE_BIN,
      workspace: tmp,
      _skipBackup: true,
    });
    expect(r2.idempotentNoop).toBe(true);
  });
});

describe("installClient — vibe (TOML array)", () => {
  it("appends mind-nerve to existing mcp_servers array", async () => {
    const tmp = await makeTmp();
    const configPath = path.join(tmp, ".vibe", "config.toml");
    await cloneFixture("vibe-config-before.toml", configPath);

    const spec = {
      ...AGENT_REGISTRY.get("vibe")!,
      configPath,
      mcpPath: configPath,
    };

    const result = await installClient(spec, {
      mindNerveBin: FAKE_BIN,
      workspace: tmp,
    });

    expect(result.changed).toBe(true);
    const content = await readText(configPath);
    expect(content).toContain('"mind-nerve"');
    expect(content).toContain('"other-server"'); // preserved
  });
});

describe("installClient — gemini (JSON)", () => {
  it("adds mcpServers.mind-nerve to gemini settings", async () => {
    const tmp = await makeTmp();
    const configPath = path.join(tmp, ".gemini", "settings.json");
    await cloneFixture("gemini-settings-before.json", configPath);

    const spec = {
      ...AGENT_REGISTRY.get("gemini")!,
      configPath,
      mcpPath: configPath,
    };

    const result = await installClient(spec, {
      mindNerveBin: FAKE_BIN,
      workspace: tmp,
    });
    expect(result.changed).toBe(true);
    const written = await readJson(configPath);
    const servers = written["mcpServers"] as Record<string, unknown>;
    expect(servers).toHaveProperty("mind-nerve");
    expect(written["theme"]).toBe("dark"); // preserved
  });
});

describe("installClient — zed (JSON context_servers)", () => {
  it("adds context_servers.mind-nerve to zed settings", async () => {
    const tmp = await makeTmp();
    const configPath = path.join(tmp, ".config", "zed", "settings.json");
    await cloneFixture("zed-settings-before.json", configPath);

    const spec = {
      ...AGENT_REGISTRY.get("zed")!,
      configPath,
      mcpPath: configPath,
    };

    const result = await installClient(spec, {
      mindNerveBin: FAKE_BIN,
      workspace: tmp,
    });
    expect(result.changed).toBe(true);
    const written = await readJson(configPath);
    const ctx = written["context_servers"] as Record<string, unknown>;
    expect(ctx).toHaveProperty("mind-nerve");
    const entry = ctx["mind-nerve"] as Record<string, unknown>;
    expect(entry["source"]).toBe("custom");
  });
});

describe("installClient — cursor (workspace text-block + mcp)", () => {
  it("appends instruction block to .cursorrules", async () => {
    const tmp = await makeTmp();
    // .cursorrules is workspace-relative.
    const rulesPath = path.join(tmp, ".cursorrules");
    await fs.writeFile(rulesPath, "# My cursor rules\n", "utf8");

    // cursor MCP goes to ~/.cursor/mcp.json
    const mcpPath = path.join(tmp, ".cursor", "mcp.json");
    await cloneFixture("cursor-mcp-before.json", mcpPath);

    const spec = {
      ...AGENT_REGISTRY.get("cursor")!,
      configPath: ".cursorrules",
      mcpPath,
    };

    const result = await installClient(spec, {
      mindNerveBin: FAKE_BIN,
      workspace: tmp,
    });
    expect(result.changed).toBe(true);

    const rules = await readText(rulesPath);
    expect(rules).toContain("mind-nerve managed");

    const mcp = await readJson(mcpPath);
    const servers = mcp["mcpServers"] as Record<string, unknown>;
    expect(servers).toHaveProperty("mind-nerve");
  });
});

describe("installClient — copilot (alwaysOffer, instruction-only)", () => {
  it("creates copilot-instructions.md with instruction block", async () => {
    const tmp = await makeTmp();
    const instrPath = path.join(tmp, ".github", "copilot-instructions.md");
    await cloneFixture("copilot-instructions-before.md", instrPath);

    const spec = {
      ...AGENT_REGISTRY.get("copilot")!,
      configPath: path.join(".github", "copilot-instructions.md"),
      instructionFilePath: path.join(".github", "copilot-instructions.md"),
    };

    const result = await installClient(spec, {
      mindNerveBin: FAKE_BIN,
      workspace: tmp,
    });
    expect(result.changed).toBe(true);
    const content = await readText(instrPath);
    expect(content).toContain("mind-nerve managed");
    expect(content).toContain("GitHub Copilot Instructions"); // original preserved
  });
});

describe("installClient — aider (yaml instruction-only)", () => {
  it("appends mind-nerve block to .aider.conf.yml", async () => {
    const tmp = await makeTmp();
    const aiderPath = path.join(tmp, ".aider.conf.yml");
    await cloneFixture("aider-conf-before.yml", aiderPath);

    const spec = {
      ...AGENT_REGISTRY.get("aider")!,
      configPath: ".aider.conf.yml",
      instructionFilePath: ".aider.conf.yml",
    };

    const result = await installClient(spec, {
      mindNerveBin: FAKE_BIN,
      workspace: tmp,
    });
    expect(result.changed).toBe(true);
    const content = await readText(aiderPath);
    expect(content).toContain("mind-nerve managed");
    expect(content).toContain("auto-commits"); // original preserved
  });
});

describe("installClient — qodo (text-block)", () => {
  it("creates .codium/ai-rules.md with instruction block", async () => {
    const tmp = await makeTmp();
    const qodoPath = path.join(tmp, ".codium", "ai-rules.md");
    await cloneFixture("qodo-airules-before.md", qodoPath);

    const spec = {
      ...AGENT_REGISTRY.get("qodo")!,
      configPath: path.join(".codium", "ai-rules.md"),
      instructionFilePath: path.join(".codium", "ai-rules.md"),
    };

    const result = await installClient(spec, {
      mindNerveBin: FAKE_BIN,
      workspace: tmp,
    });
    expect(result.changed).toBe(true);
    const content = await readText(qodoPath);
    expect(content).toContain("mind-nerve managed");
  });
});

describe("installClient — openclaw / nanoclaw / nemoclaw (json-openclaw-hooks)", () => {
  for (const clientName of ["openclaw", "nanoclaw", "nemoclaw"] as const) {
    it(`${clientName}: adds mcpServers.mind-nerve`, async () => {
      const tmp = await makeTmp();
      const fixtureName = `${clientName}-before.json` as const;
      const configPath = path.join(tmp, `.${clientName}`, `${clientName}.json`);
      await cloneFixture(fixtureName, configPath);

      const spec = {
        ...AGENT_REGISTRY.get(clientName)!,
        configPath,
        mcpPath: configPath,
      };

      const result = await installClient(spec, {
        mindNerveBin: FAKE_BIN,
        workspace: tmp,
      });
      expect(result.changed).toBe(true);
      const written = await readJson(configPath);
      const servers = written["mcpServers"] as Record<string, unknown>;
      expect(servers).toHaveProperty("mind-nerve");
    });
  }
});

describe("installClient — cline / roo / continue (mcp-json-servers)", () => {
  const cases = [
    { name: "cline", fixture: "cline-mcp-before.json" },
    { name: "roo", fixture: "roo-mcp-before.json" },
    { name: "continue", fixture: "continue-config-before.json" },
  ] as const;

  for (const { name, fixture } of cases) {
    it(`${name}: adds mcpServers.mind-nerve`, async () => {
      const tmp = await makeTmp();
      const mcpPath = path.join(tmp, `${name}-mcp.json`);
      await cloneFixture(fixture, mcpPath);

      const spec = {
        ...AGENT_REGISTRY.get(name)!,
        mcpPath,
        instructionFilePath: name !== "continue" ? `.${name}rules` : null,
      };

      const result = await installClient(spec, {
        mindNerveBin: FAKE_BIN,
        workspace: tmp,
      });
      // Only MCP should change (instruction file creation is also change).
      expect(result.changed).toBe(true);
      const written = await readJson(mcpPath);
      const servers = written["mcpServers"] as Record<string, unknown>;
      expect(servers).toHaveProperty("mind-nerve");
    });
  }
});

describe("installClient — windsurf", () => {
  it("adds mind-nerve to windsurf MCP config", async () => {
    const tmp = await makeTmp();
    const mcpPath = path.join(tmp, ".codeium", "windsurf", "mcp_config.json");
    await cloneFixture("windsurf-mcp-before.json", mcpPath);

    const spec = {
      ...AGENT_REGISTRY.get("windsurf")!,
      mcpPath,
      instructionFilePath: ".windsurfrules",
    };

    const result = await installClient(spec, {
      mindNerveBin: FAKE_BIN,
      workspace: tmp,
    });
    expect(result.changed).toBe(true);
    const written = await readJson(mcpPath);
    const servers = written["mcpServers"] as Record<string, unknown>;
    expect(servers).toHaveProperty("mind-nerve");
  });
});

describe("--all install produces backups", () => {
  it("produces at least one .bak file when JSON clients are installed", async () => {
    const tmp = await makeTmp();
    const configPath = path.join(tmp, "settings.json");
    await cloneFixture("claude-settings-before.json", configPath);

    const spec = {
      ...AGENT_REGISTRY.get("claude-code")!,
      configPath,
      mcpPath: configPath,
    };

    const result = await installClient(spec, {
      mindNerveBin: FAKE_BIN,
      workspace: tmp,
    });
    expect(result.backedUp.length).toBe(1);
    const bak = result.backedUp[0]!;
    expect(bak).toMatch(/\.bak-mind-nerve-\d+$/);
    await expect(fs.stat(bak)).resolves.toBeTruthy();
  });
});
