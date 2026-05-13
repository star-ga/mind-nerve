// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.

import { describe, it, expect, afterEach } from "vitest";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { installClient } from "../src/install.js";
import { uninstallClient } from "../src/uninstall.js";
import { AGENT_REGISTRY } from "../src/registry.js";

let tmpDirs: string[] = [];

async function makeTmp(): Promise<string> {
  const d = await fs.mkdtemp(path.join(os.tmpdir(), "mn-uninstall-test-"));
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

async function cloneFixture(fixtureName: string, destPath: string): Promise<void> {
  const fixtureDir = new URL("./fixtures", import.meta.url).pathname;
  const src = path.join(fixtureDir, fixtureName);
  await fs.mkdir(path.dirname(destPath), { recursive: true });
  await fs.copyFile(src, destPath);
}

async function readText(p: string): Promise<string> {
  return fs.readFile(p, "utf8");
}

async function readJson(p: string): Promise<Record<string, unknown>> {
  return JSON.parse(await fs.readFile(p, "utf8")) as Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Round-trip tests: install then uninstall restores original content.
// ---------------------------------------------------------------------------

describe("uninstallClient — claude-code JSON MCP", () => {
  it("restores original settings.json byte-for-byte via backup", async () => {
    const tmp = await makeTmp();
    const configPath = path.join(tmp, "settings.json");
    const fixturePath = new URL("./fixtures/claude-settings-before.json", import.meta.url).pathname;
    await cloneFixture("claude-settings-before.json", configPath);
    const original = await readText(fixturePath);

    const spec = {
      ...AGENT_REGISTRY.get("claude-code")!,
      configPath,
      mcpPath: configPath,
    };

    await installClient(spec, { mindNerveBin: FAKE_BIN, workspace: tmp });
    const afterInstall = await readText(configPath);
    expect(afterInstall).not.toBe(original); // sanity: install changed it

    const result = await uninstallClient(spec, { workspace: tmp });
    expect(result.changed).toBe(true);
    expect(result.restoredPaths).toContain(configPath);

    const afterUninstall = await readText(configPath);
    expect(afterUninstall).toBe(original);
  });
});

describe("uninstallClient — codex TOML", () => {
  it("restores original TOML file via backup", async () => {
    const tmp = await makeTmp();
    const configPath = path.join(tmp, ".codex", "config.toml");
    const fixturePath = new URL("./fixtures/codex-config-before.toml", import.meta.url).pathname;
    await cloneFixture("codex-config-before.toml", configPath);
    const original = await readText(fixturePath);

    const spec = {
      ...AGENT_REGISTRY.get("codex")!,
      configPath,
      mcpPath: configPath,
    };

    await installClient(spec, { mindNerveBin: FAKE_BIN, workspace: tmp });
    const result = await uninstallClient(spec, { workspace: tmp });
    expect(result.changed).toBe(true);

    const afterUninstall = await readText(configPath);
    expect(afterUninstall).toBe(original);
  });
});

describe("uninstallClient — vibe TOML array", () => {
  it("restores original vibe config via backup", async () => {
    const tmp = await makeTmp();
    const configPath = path.join(tmp, ".vibe", "config.toml");
    const fixturePath = new URL("./fixtures/vibe-config-before.toml", import.meta.url).pathname;
    await cloneFixture("vibe-config-before.toml", configPath);
    const original = await readText(fixturePath);

    const spec = {
      ...AGENT_REGISTRY.get("vibe")!,
      configPath,
      mcpPath: configPath,
    };

    await installClient(spec, { mindNerveBin: FAKE_BIN, workspace: tmp });
    const result = await uninstallClient(spec, { workspace: tmp });
    expect(result.changed).toBe(true);

    const afterUninstall = await readText(configPath);
    expect(afterUninstall).toBe(original);
  });
});

describe("uninstallClient — copilot instruction block", () => {
  it("removes mind-nerve block from copilot-instructions.md", async () => {
    const tmp = await makeTmp();
    const instrPath = path.join(tmp, ".github", "copilot-instructions.md");
    const fixturePath = new URL("./fixtures/copilot-instructions-before.md", import.meta.url).pathname;
    await cloneFixture("copilot-instructions-before.md", instrPath);
    const original = await readText(fixturePath);

    const spec = {
      ...AGENT_REGISTRY.get("copilot")!,
      configPath: path.join(".github", "copilot-instructions.md"),
      instructionFilePath: path.join(".github", "copilot-instructions.md"),
    };

    await installClient(spec, { mindNerveBin: FAKE_BIN, workspace: tmp });
    const afterInstall = await readText(instrPath);
    expect(afterInstall).toContain("mind-nerve managed");

    const result = await uninstallClient(spec, { workspace: tmp });
    expect(result.changed).toBe(true);
    expect(result.removedBlocks).toContain(instrPath);

    const afterUninstall = await readText(instrPath);
    expect(afterUninstall).not.toContain("mind-nerve managed");
    // Original content preserved.
    expect(afterUninstall).toContain("GitHub Copilot Instructions");
  });
});

describe("uninstallClient — aider", () => {
  it("removes mind-nerve block from .aider.conf.yml", async () => {
    const tmp = await makeTmp();
    const aiderPath = path.join(tmp, ".aider.conf.yml");
    await cloneFixture("aider-conf-before.yml", aiderPath);

    const spec = {
      ...AGENT_REGISTRY.get("aider")!,
      configPath: ".aider.conf.yml",
      instructionFilePath: ".aider.conf.yml",
    };

    await installClient(spec, { mindNerveBin: FAKE_BIN, workspace: tmp });
    await uninstallClient(spec, { workspace: tmp });

    const content = await readText(aiderPath);
    expect(content).not.toContain("mind-nerve managed");
    expect(content).toContain("auto-commits"); // original preserved
  });
});

describe("uninstallClient — not installed (no-op)", () => {
  it("returns changed=false when nothing was installed", async () => {
    const tmp = await makeTmp();
    const spec = {
      ...AGENT_REGISTRY.get("gemini")!,
      configPath: path.join(tmp, ".gemini", "settings.json"),
      mcpPath: path.join(tmp, ".gemini", "settings.json"),
    };

    const result = await uninstallClient(spec, { workspace: tmp });
    expect(result.changed).toBe(false);
  });
});

describe("uninstallClient — gemini JSON", () => {
  it("restores original gemini settings.json via backup", async () => {
    const tmp = await makeTmp();
    const configPath = path.join(tmp, ".gemini", "settings.json");
    const fixturePath = new URL("./fixtures/gemini-settings-before.json", import.meta.url).pathname;
    await cloneFixture("gemini-settings-before.json", configPath);
    const original = await readText(fixturePath);

    const spec = {
      ...AGENT_REGISTRY.get("gemini")!,
      configPath,
      mcpPath: configPath,
    };

    await installClient(spec, { mindNerveBin: FAKE_BIN, workspace: tmp });
    const result = await uninstallClient(spec, { workspace: tmp });
    expect(result.changed).toBe(true);

    const afterUninstall = await readText(configPath);
    expect(afterUninstall).toBe(original);
  });
});

describe("uninstallClient — zed JSON", () => {
  it("restores original zed settings.json via backup", async () => {
    const tmp = await makeTmp();
    const configPath = path.join(tmp, ".config", "zed", "settings.json");
    const fixturePath = new URL("./fixtures/zed-settings-before.json", import.meta.url).pathname;
    await cloneFixture("zed-settings-before.json", configPath);
    const original = await readText(fixturePath);

    const spec = {
      ...AGENT_REGISTRY.get("zed")!,
      configPath,
      mcpPath: configPath,
    };

    await installClient(spec, { mindNerveBin: FAKE_BIN, workspace: tmp });
    const result = await uninstallClient(spec, { workspace: tmp });
    expect(result.changed).toBe(true);

    const afterUninstall = await readText(configPath);
    expect(afterUninstall).toBe(original);
  });
});
