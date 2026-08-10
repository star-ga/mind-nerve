// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.
//
// End-to-end wiring: install must self-wire a CLI, and uninstall must restore
// what was there exactly — including re-creating the hub symlink.
//
// Runs entirely against a fake home under a temp dir. Nothing here touches a
// real ~/.<cli>/ directory.

import { describe, it, expect, afterEach } from "vitest";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { fileURLToPath } from "node:url";
import { wireClient, unwireClient, sharedHookPath } from "../src/wire.js";
import { inspectSkillsDir } from "../src/skills_dir.js";
import type { AgentSpec, SkillSurface, HookWireFmt } from "../src/registry.js";

const HOOK_SOURCE = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
  "..",
  "hook",
  "mind-nerve-hook",
);
const ROUTER_SKILL = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
  "..",
  "hook",
  "assets",
  "mind-nerve-router.SKILL.md",
);

const tmpDirs: string[] = [];

async function makeTmp(): Promise<string> {
  const d = await fs.mkdtemp(path.join(os.tmpdir(), "mn-wire-"));
  tmpDirs.push(d);
  return d;
}

afterEach(async () => {
  while (tmpDirs.length > 0) {
    const d = tmpDirs.pop();
    if (d !== undefined) await fs.rm(d, { recursive: true, force: true });
  }
});

/** Builds a fake home with a hub and a CLI dir, mirroring the real layout. */
async function makeHome(cliDir: string): Promise<{
  home: string;
  hub: string;
  skillsDir: string;
  configPath: string;
}> {
  const home = await makeTmp();
  const hub = path.join(home, ".agents", "skills-hub");
  for (const n of ["mind-nerve-router", "alpha", "beta"]) {
    await fs.mkdir(path.join(hub, n), { recursive: true });
    await fs.writeFile(path.join(hub, n, "SKILL.md"), `# ${n}\n`);
  }
  await fs.mkdir(path.join(home, cliDir), { recursive: true });
  return {
    home,
    hub,
    skillsDir: path.join(home, cliDir, "skills"),
    configPath: path.join(home, cliDir, "config"),
  };
}

function makeSpec(args: {
  name: string;
  home: string;
  cliDir: string;
  fmt: HookWireFmt;
  configFile: string;
}): AgentSpec {
  const surface: SkillSurface = {
    skillsDir: path.join(args.home, args.cliDir, "skills"),
    hookScriptPath: path.join(
      args.home,
      args.cliDir,
      "hooks",
      "mind-nerve-hook",
    ),
    hookConfigPath: path.join(args.home, args.cliDir, args.configFile),
    hookWireFmt: args.fmt,
    hookEvents: ["UserPromptSubmit", "SessionStart"],
    hookTimeoutSecs: 8,
    verified: "fixture",
  };
  return {
    name: args.name,
    description: "fixture",
    configFmt: "text-block",
    configPath: surface.hookConfigPath,
    mcpPath: null,
    mcpFmt: null,
    detectPaths: [],
    detectBinaries: [],
    alwaysOffer: false,
    projectionDir: null,
    instructionFilePath: null,
    skillSurface: surface,
  };
}

function wireOpts(home: string, hub: string) {
  return {
    hookSourcePath: HOOK_SOURCE,
    routerSkillPath: ROUTER_SKILL,
    hubDir: hub,
    socketPath: path.join(home, "nerve.sock"),
    agentDirs: [path.join(home, ".claude", "agents")],
    homeDir: home,
    now: 1_700_000_000_000,
    _skipBackup: false,
  };
}

// ---------------------------------------------------------------------------
// JSON-configured CLI (claude-code / gemini / qwen shape)
// ---------------------------------------------------------------------------

describe("wireClient — JSON-configured CLI", () => {
  it("installs the shared hook, a wrapper, the registration and the skills dir", async () => {
    const { home, hub, skillsDir } = await makeHome(".fakejson");
    await fs.symlink(hub, skillsDir); // the whole-hub symlink being replaced
    const spec = makeSpec({
      name: "fakejson",
      home,
      cliDir: ".fakejson",
      fmt: "json-hooks",
      configFile: "settings.json",
    });

    const res = await wireClient(spec, wireOpts(home, hub));
    expect(res).not.toBeNull();
    expect(res!.changed).toBe(true);

    // 1. shared implementation, executable
    const shared = sharedHookPath(home);
    expect((await fs.stat(shared)).mode & 0o111).toBeGreaterThan(0);
    expect(await fs.readFile(shared, "utf8")).toBe(
      await fs.readFile(HOOK_SOURCE, "utf8"),
    );

    // 2. per-CLI wrapper, executable, exec'ing the shared hook
    const wrapper = await fs.readFile(spec.skillSurface!.hookScriptPath, "utf8");
    expect(wrapper).toContain(`exec '${shared}'`);
    expect(wrapper).toContain(`export MIND_NERVE_PROJECTED_DIR='${skillsDir}'`);
    expect(wrapper).toContain("export MIND_NERVE_MIN_SCORE='0.35'");
    expect(
      (await fs.stat(spec.skillSurface!.hookScriptPath)).mode & 0o111,
    ).toBeGreaterThan(0);

    // 3. registration on both events
    const cfg = JSON.parse(
      await fs.readFile(spec.skillSurface!.hookConfigPath, "utf8"),
    ) as { hooks: Record<string, unknown[]> };
    expect(Object.keys(cfg.hooks).sort()).toEqual([
      "SessionStart",
      "UserPromptSubmit",
    ]);

    // 4. router-only real skills dir
    expect((await inspectSkillsDir(skillsDir)).kind).toBe("realdir");
    expect((await fs.readdir(skillsDir)).sort()).toEqual([
      "README.md",
      "mind-nerve-router",
    ]);
  });

  it("is idempotent — a second wire changes nothing", async () => {
    const { home, hub, skillsDir } = await makeHome(".fakejson");
    await fs.symlink(hub, skillsDir);
    const spec = makeSpec({
      name: "fakejson",
      home,
      cliDir: ".fakejson",
      fmt: "json-hooks",
      configFile: "settings.json",
    });

    await wireClient(spec, wireOpts(home, hub));
    const second = await wireClient(spec, wireOpts(home, hub));

    expect(second!.changed).toBe(false);
    expect(second!.hookRegistered).toBe(false);
    expect(second!.skillsConverted).toBe(false);
    const cfg = JSON.parse(
      await fs.readFile(spec.skillSurface!.hookConfigPath, "utf8"),
    ) as { hooks: Record<string, unknown[]> };
    expect(cfg.hooks["UserPromptSubmit"]!.length).toBe(1);
  });

  it("backs up an existing config before touching it", async () => {
    const { home, hub } = await makeHome(".fakejson");
    const spec = makeSpec({
      name: "fakejson",
      home,
      cliDir: ".fakejson",
      fmt: "json-hooks",
      configFile: "settings.json",
    });
    await fs.writeFile(
      spec.skillSurface!.hookConfigPath,
      JSON.stringify({ model: "opus" }),
    );

    const res = await wireClient(spec, wireOpts(home, hub));

    expect(res!.backedUp.length).toBe(1);
    expect(JSON.parse(await fs.readFile(res!.backedUp[0]!, "utf8"))).toEqual({
      model: "opus",
    });
    // And the live config keeps the user's setting.
    const cfg = JSON.parse(
      await fs.readFile(spec.skillSurface!.hookConfigPath, "utf8"),
    ) as Record<string, unknown>;
    expect(cfg["model"]).toBe("opus");
  });
});

// ---------------------------------------------------------------------------
// TOML-configured CLI (codex / grok / kimi shape)
// ---------------------------------------------------------------------------

describe("wireClient — TOML-configured CLI", () => {
  it("appends a marked block and preserves the existing config", async () => {
    const { home, hub, skillsDir } = await makeHome(".faketoml");
    await fs.symlink(hub, skillsDir);
    const spec = makeSpec({
      name: "faketoml",
      home,
      cliDir: ".faketoml",
      fmt: "toml-hooks",
      configFile: "config.toml",
    });
    const original = [
      "# generated skills budget — do not lose this comment",
      "skills.config = [",
      '  {path="/hub/a/SKILL.md", enabled=false},',
      "]",
      "",
      "[models]",
      'default = "k3"',
      "",
    ].join("\n");
    await fs.writeFile(spec.skillSurface!.hookConfigPath, original);

    await wireClient(spec, wireOpts(home, hub));

    const text = await fs.readFile(spec.skillSurface!.hookConfigPath, "utf8");
    expect(text).toContain("# generated skills budget — do not lose this comment");
    expect(text).toContain('  {path="/hub/a/SKILL.md", enabled=false},');
    expect(text).toContain('default = "k3"');
    expect(text).toContain("[[hooks.UserPromptSubmit]]");
    expect(text).toContain("[[hooks.SessionStart]]");
  });

  it("is idempotent across repeated installs", async () => {
    const { home, hub, skillsDir } = await makeHome(".faketoml");
    await fs.symlink(hub, skillsDir);
    const spec = makeSpec({
      name: "faketoml",
      home,
      cliDir: ".faketoml",
      fmt: "toml-hooks",
      configFile: "config.toml",
    });

    await wireClient(spec, wireOpts(home, hub));
    const after1 = await fs.readFile(spec.skillSurface!.hookConfigPath, "utf8");
    await wireClient(spec, wireOpts(home, hub));
    await wireClient(spec, wireOpts(home, hub));
    const after3 = await fs.readFile(spec.skillSurface!.hookConfigPath, "utf8");

    expect(after3).toBe(after1);
    expect(after3.split("[[hooks.UserPromptSubmit]]").length - 1).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Round trip
// ---------------------------------------------------------------------------

describe("wire -> unwire round trip", () => {
  it("re-creates the hub symlink exactly", async () => {
    const { home, hub, skillsDir } = await makeHome(".fakejson");
    await fs.symlink(hub, skillsDir);
    const spec = makeSpec({
      name: "fakejson",
      home,
      cliDir: ".fakejson",
      fmt: "json-hooks",
      configFile: "settings.json",
    });

    await wireClient(spec, wireOpts(home, hub));
    expect((await inspectSkillsDir(skillsDir)).kind).toBe("realdir");

    const un = await unwireClient(spec, { homeDir: home });

    expect(un!.changed).toBe(true);
    const after = await inspectSkillsDir(skillsDir);
    expect(after.kind).toBe("symlink");
    expect(after.symlinkTarget).toBe(hub);
    // The hub survived the whole cycle.
    expect((await fs.readdir(hub)).sort()).toEqual([
      "alpha",
      "beta",
      "mind-nerve-router",
    ]);
  });

  it("restores a JSON config to its original content", async () => {
    const { home, hub, skillsDir } = await makeHome(".fakejson");
    await fs.symlink(hub, skillsDir);
    const spec = makeSpec({
      name: "fakejson",
      home,
      cliDir: ".fakejson",
      fmt: "json-hooks",
      configFile: "settings.json",
    });
    const original = {
      model: "opus",
      hooks: {
        Stop: [{ matcher: "", hooks: [{ type: "command", command: "mm status" }] }],
      },
    };
    await fs.writeFile(
      spec.skillSurface!.hookConfigPath,
      JSON.stringify(original, null, 2) + "\n",
    );

    await wireClient(spec, wireOpts(home, hub));
    await unwireClient(spec, { homeDir: home });

    const restored = JSON.parse(
      await fs.readFile(spec.skillSurface!.hookConfigPath, "utf8"),
    ) as unknown;
    expect(restored).toEqual(original);
  });

  it("restores a TOML config byte-for-byte", async () => {
    const { home, hub, skillsDir } = await makeHome(".faketoml");
    await fs.symlink(hub, skillsDir);
    const spec = makeSpec({
      name: "faketoml",
      home,
      cliDir: ".faketoml",
      fmt: "toml-hooks",
      configFile: "config.toml",
    });
    const original = '# keep me\n[models]\ndefault = "k3"\n';
    await fs.writeFile(spec.skillSurface!.hookConfigPath, original);

    await wireClient(spec, wireOpts(home, hub));
    await unwireClient(spec, { homeDir: home });

    expect(await fs.readFile(spec.skillSurface!.hookConfigPath, "utf8")).toBe(
      original,
    );
  });

  it("removes the per-CLI wrapper", async () => {
    const { home, hub, skillsDir } = await makeHome(".fakejson");
    await fs.symlink(hub, skillsDir);
    const spec = makeSpec({
      name: "fakejson",
      home,
      cliDir: ".fakejson",
      fmt: "json-hooks",
      configFile: "settings.json",
    });

    await wireClient(spec, wireOpts(home, hub));
    await unwireClient(spec, { homeDir: home });

    await expect(
      fs.access(spec.skillSurface!.hookScriptPath),
    ).rejects.toThrow();
  });

  it("restores a pre-existing REAL skills directory with its content", async () => {
    const { home, hub, skillsDir } = await makeHome(".fakejson");
    await fs.mkdir(path.join(skillsDir, "my-own-skill"), { recursive: true });
    await fs.writeFile(
      path.join(skillsDir, "my-own-skill", "SKILL.md"),
      "hand written\n",
    );
    const spec = makeSpec({
      name: "fakejson",
      home,
      cliDir: ".fakejson",
      fmt: "json-hooks",
      configFile: "settings.json",
    });

    await wireClient(spec, wireOpts(home, hub));
    await unwireClient(spec, { homeDir: home });

    expect(
      await fs.readFile(path.join(skillsDir, "my-own-skill", "SKILL.md"), "utf8"),
    ).toBe("hand written\n");
  });

  it("a second unwire is a harmless no-op", async () => {
    const { home, hub, skillsDir } = await makeHome(".fakejson");
    await fs.symlink(hub, skillsDir);
    const spec = makeSpec({
      name: "fakejson",
      home,
      cliDir: ".fakejson",
      fmt: "json-hooks",
      configFile: "settings.json",
    });

    await wireClient(spec, wireOpts(home, hub));
    await unwireClient(spec, { homeDir: home });
    const second = await unwireClient(spec, { homeDir: home });

    expect(second!.changed).toBe(false);
    expect((await inspectSkillsDir(skillsDir)).kind).toBe("symlink");
  });

  it("returns null for a client with no skill surface", async () => {
    const { home, hub } = await makeHome(".fakejson");
    const spec = makeSpec({
      name: "nosurface",
      home,
      cliDir: ".fakejson",
      fmt: "json-hooks",
      configFile: "settings.json",
    });
    const bare: AgentSpec = { ...spec, skillSurface: null };
    expect(await wireClient(bare, wireOpts(home, hub))).toBeNull();
    expect(await unwireClient(bare, { homeDir: home })).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// The installed wrapper actually works
// ---------------------------------------------------------------------------

describe("installed wrapper is a working hook", () => {
  it("runs, fails open, and seeds the projection", async () => {
    const { home, hub, skillsDir } = await makeHome(".fakejson");
    await fs.symlink(hub, skillsDir);
    const spec = makeSpec({
      name: "fakejson",
      home,
      cliDir: ".fakejson",
      fmt: "json-hooks",
      configFile: "settings.json",
    });
    await wireClient(spec, wireOpts(home, hub));

    const { spawn } = await import("node:child_process");
    const out = await new Promise<{ code: number | null; stdout: string }>(
      (resolve) => {
        const child = spawn(spec.skillSurface!.hookScriptPath, [], {
          stdio: ["pipe", "pipe", "pipe"],
        });
        let stdout = "";
        child.stdout.on("data", (d: Buffer) => (stdout += d.toString()));
        child.on("close", (code) => resolve({ code, stdout }));
        child.stdin.write(JSON.stringify({ hookEventName: "SessionStart" }));
        child.stdin.end();
      },
    );

    // No daemon is listening — it must still exit 0 with valid JSON.
    expect(out.code).toBe(0);
    expect(() => JSON.parse(out.stdout.trim())).not.toThrow();
    expect((await fs.readdir(skillsDir)).sort()).toEqual([
      "README.md",
      "mind-nerve-router",
    ]);
  });
});
