// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.
//
// Tests for `verify`: self-test of an existing installation per CLI.
// Runs against a fake home under a temp dir, wiring real installs through
// installClient + wireClient exactly like wire.test.ts, then breaking one
// thing at a time to cover every FAIL class.

import { describe, it, expect, afterEach } from "vitest";
import net from "node:net";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { fileURLToPath } from "node:url";
import { installClient } from "../src/install.js";
import {
  verifyClient,
  summarize,
  hasFailures,
  worstStatus,
  type VerifyCheck,
} from "../src/verify.js";
import { AGENT_REGISTRY } from "../src/registry.js";
import type { AgentSpec, SkillSurface, HookWireFmt } from "../src/registry.js";
import { HOOK_BLOCK_BEGIN } from "../src/hook_wiring.js";

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
  const d = await fs.mkdtemp(path.join(os.tmpdir(), "mn-verify-"));
  tmpDirs.push(d);
  return d;
}

afterEach(async () => {
  while (tmpDirs.length > 0) {
    const d = tmpDirs.pop();
    if (d !== undefined) await fs.rm(d, { recursive: true, force: true });
  }
});

/** A fake home with a hub, a CLI dir, and an executable fake mind-nerve bin. */
async function makeHome(cliDir: string): Promise<{
  home: string;
  hub: string;
  fakeBin: string;
  socketPath: string;
}> {
  const home = await makeTmp();
  const hub = path.join(home, ".agents", "skills-hub");
  for (const n of ["mind-nerve-router", "alpha", "beta"]) {
    await fs.mkdir(path.join(hub, n), { recursive: true });
    await fs.writeFile(path.join(hub, n, "SKILL.md"), `# ${n}\n`);
  }
  await fs.mkdir(path.join(home, cliDir), { recursive: true });
  const fakeBin = path.join(home, "bin", "mind-nerve");
  await fs.mkdir(path.dirname(fakeBin), { recursive: true });
  await fs.writeFile(fakeBin, "#!/bin/sh\nexit 0\n", { mode: 0o755 });
  await fs.chmod(fakeBin, 0o755);
  // The venv MCP entry pins the mind-nerve-mcp console script next to the bin.
  const fakeMcpBin = path.join(home, "bin", "mind-nerve-mcp");
  await fs.writeFile(fakeMcpBin, "#!/bin/sh\nexit 0\n", { mode: 0o755 });
  await fs.chmod(fakeMcpBin, 0o755);
  return { home, hub, fakeBin, socketPath: path.join(home, "nerve.sock") };
}

/**
 * Codex-like spec: TOML hooks + TOML MCP in one config file.
 * Gemini-like spec: JSON hooks + JSON MCP in one settings file.
 */
function makeSpec(args: {
  base: string;
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
  const base = AGENT_REGISTRY.get(args.base);
  if (base === undefined) throw new Error(`bad fixture base ${args.base}`);
  return {
    ...base,
    configPath: surface.hookConfigPath,
    mcpPath: surface.hookConfigPath,
    detectPaths: [],
    detectBinaries: [],
    skillSurface: surface,
  };
}

function wireOpts(home: string, hub: string, socketPath: string) {
  return {
    hookSourcePath: HOOK_SOURCE,
    routerSkillPath: ROUTER_SKILL,
    hubDir: hub,
    socketPath,
    agentDirs: [path.join(home, ".claude", "agents")],
    homeDir: home,
    now: 1_700_000_000_000,
    _skipBackup: true as const,
  };
}

/** Installs the full wiring (MCP + skill surface) for a spec. */
async function fullInstall(
  spec: AgentSpec,
  home: string,
  hub: string,
  fakeBin: string,
  socketPath: string,
): Promise<void> {
  const result = await installClient(spec, {
    mindNerveBin: fakeBin,
    workspace: home,
    wire: wireOpts(home, hub, socketPath),
    _skipBackup: true,
  });
  expect(result.changed).toBe(true);
}

/** A minimal daemon: accepts connections on a real UNIX socket. */
async function startDaemon(socketPath: string): Promise<net.Server> {
  const server = net.createServer((conn) => conn.end());
  await new Promise<void>((resolve) => server.listen(socketPath, resolve));
  return server;
}

function checkByName(
  checks: readonly VerifyCheck[],
  name: string,
): VerifyCheck {
  const found = checks.find((c) => c.name === name);
  if (found === undefined) {
    throw new Error(`check '${name}' not present — got: ${checks.map((c) => c.name).join(", ")}`);
  }
  return found;
}

// ---------------------------------------------------------------------------
// Fully wired installs: every check PASS.
// ---------------------------------------------------------------------------

describe("verifyClient — fully wired fake CLIs (all PASS)", () => {
  it("codex-like (TOML hooks + TOML MCP) verifies clean", async () => {
    const { home, hub, fakeBin, socketPath } = await makeHome(".codex");
    const spec = makeSpec({
      base: "codex",
      home,
      cliDir: ".codex",
      fmt: "toml-hooks",
      configFile: "config.toml",
    });
    await fullInstall(spec, home, hub, fakeBin, socketPath);
    const daemon = await startDaemon(socketPath);
    try {
      const report = await verifyClient(spec, {
        workspace: home,
        socketPath,
      });
      expect(report.status).toBe("PASS");
      expect(hasFailures([report])).toBe(false);
      for (const name of [
        "config",
        "hook-config",
        "hook-script",
        "hook-failopen",
        "skills-dir",
        "env-pins",
        "mcp-entry",
        "mcp-command",
        "daemon-socket",
      ]) {
        const check = checkByName(report.checks, name);
        expect(check.status, `${name}: ${check.message}`).toBe("PASS");
      }
    } finally {
      await new Promise<void>((resolve) => daemon.close(() => resolve()));
    }
  });

  it("gemini-like (JSON hooks + JSON MCP) verifies clean", async () => {
    const { home, hub, fakeBin, socketPath } = await makeHome(".gemini");
    const spec = makeSpec({
      base: "gemini",
      home,
      cliDir: ".gemini",
      fmt: "json-hooks",
      configFile: "settings.json",
    });
    await fullInstall(spec, home, hub, fakeBin, socketPath);
    const daemon = await startDaemon(socketPath);
    try {
      const report = await verifyClient(spec, {
        workspace: home,
        socketPath,
      });
      expect(report.status).toBe("PASS");
      expect(checkByName(report.checks, "mcp-entry").status).toBe("PASS");
      expect(checkByName(report.checks, "hook-config").status).toBe("PASS");
    } finally {
      await new Promise<void>((resolve) => daemon.close(() => resolve()));
    }
  });

  it("daemon-not-running is a WARN, not a FAIL", async () => {
    const { home, hub, fakeBin, socketPath } = await makeHome(".codex");
    const spec = makeSpec({
      base: "codex",
      home,
      cliDir: ".codex",
      fmt: "toml-hooks",
      configFile: "config.toml",
    });
    await fullInstall(spec, home, hub, fakeBin, socketPath);
    // No daemon listening, no socket file.
    const report = await verifyClient(spec, {
      workspace: home,
      socketPath,
    });
    const sock = checkByName(report.checks, "daemon-socket");
    expect(sock.status).toBe("WARN");
    expect(sock.hint).toContain("mind-nerve-routed-ensure");
    expect(report.status).toBe("WARN");
    expect(hasFailures([report])).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Broken installs: one FAIL class per test.
// ---------------------------------------------------------------------------

describe("verifyClient — broken installs (FAIL classes)", () => {
  async function wiredCodex(): Promise<{
    home: string;
    spec: AgentSpec;
    socketPath: string;
  }> {
    const { home, hub, fakeBin, socketPath } = await makeHome(".codex");
    const spec = makeSpec({
      base: "codex",
      home,
      cliDir: ".codex",
      fmt: "toml-hooks",
      configFile: "config.toml",
    });
    await fullInstall(spec, home, hub, fakeBin, socketPath);
    return { home, spec, socketPath };
  }

  it("reports FAIL when nothing is installed", async () => {
    const { home } = await makeHome(".codex");
    const spec = makeSpec({
      base: "codex",
      home,
      cliDir: ".codex",
      fmt: "toml-hooks",
      configFile: "config.toml",
    });
    const report = await verifyClient(spec, { workspace: home });
    expect(report.status).toBe("FAIL");
    expect(checkByName(report.checks, "installed").status).toBe("FAIL");
    expect(hasFailures([report])).toBe(true);
  });

  it("config FAIL: config file missing", async () => {
    const { home, spec } = await wiredCodex();
    await fs.rm(spec.configPath);
    const report = await verifyClient(spec, { workspace: home });
    expect(checkByName(report.checks, "config").status).toBe("FAIL");
    // Hook config is the same file — that check must fail too.
    expect(checkByName(report.checks, "hook-config").status).toBe("FAIL");
  });

  it("config FAIL: TOML does not parse", async () => {
    const { home, spec } = await wiredCodex();
    await fs.writeFile(spec.configPath, "[unclosed\n", "utf8");
    const report = await verifyClient(spec, { workspace: home });
    const check = checkByName(report.checks, "config");
    expect(check.status).toBe("FAIL");
    expect(check.hint).toContain(".bak-mind-nerve-");
  });

  it("hook-config FAIL: managed TOML hook block removed", async () => {
    const { home, spec } = await wiredCodex();
    const text = await fs.readFile(spec.configPath, "utf8");
    expect(text).toContain(HOOK_BLOCK_BEGIN);
    const start = text.indexOf(HOOK_BLOCK_BEGIN);
    const end = text.indexOf("# END mind-nerve hook");
    await fs.writeFile(
      spec.configPath,
      text.slice(0, start) + text.slice(end + "# END mind-nerve hook".length),
      "utf8",
    );
    const report = await verifyClient(spec, { workspace: home });
    expect(checkByName(report.checks, "hook-config").status).toBe("FAIL");
  });

  it("hook-script FAIL: wrapper missing", async () => {
    const { home, spec } = await wiredCodex();
    const surface = spec.skillSurface;
    if (surface === null) throw new Error("fixture");
    await fs.rm(surface.hookScriptPath);
    const report = await verifyClient(spec, { workspace: home });
    expect(checkByName(report.checks, "hook-script").status).toBe("FAIL");
  });

  it("hook-script FAIL: wrapper not executable", async () => {
    const { home, spec } = await wiredCodex();
    const surface = spec.skillSurface;
    if (surface === null) throw new Error("fixture");
    await fs.chmod(surface.hookScriptPath, 0o644);
    const report = await verifyClient(spec, { workspace: home });
    const check = checkByName(report.checks, "hook-script");
    expect(check.status).toBe("FAIL");
    expect(check.message).toContain("not executable");
  });

  it("hook-failopen FAIL: hook prints non-JSON", async () => {
    const { home, spec } = await wiredCodex();
    const surface = spec.skillSurface;
    if (surface === null) throw new Error("fixture");
    await fs.writeFile(
      surface.hookScriptPath,
      "#!/bin/sh\necho 'not json at all'\n",
      { mode: 0o755 },
    );
    await fs.chmod(surface.hookScriptPath, 0o755);
    const report = await verifyClient(spec, { workspace: home });
    expect(checkByName(report.checks, "hook-failopen").status).toBe("FAIL");
  });

  it("skills-dir FAIL: skills dir absent", async () => {
    const { home, spec } = await wiredCodex();
    const surface = spec.skillSurface;
    if (surface === null) throw new Error("fixture");
    await fs.rm(surface.skillsDir, { recursive: true, force: true });
    const report = await verifyClient(spec, { workspace: home });
    expect(checkByName(report.checks, "skills-dir").status).toBe("FAIL");
  });

  it("skills-dir FAIL: symlink target lacks mind-nerve-router", async () => {
    const { home, spec } = await wiredCodex();
    const surface = spec.skillSurface;
    if (surface === null) throw new Error("fixture");
    await fs.rm(surface.skillsDir, { recursive: true, force: true });
    const badHub = path.join(home, "bad-hub");
    await fs.mkdir(path.join(badHub, "alpha"), { recursive: true });
    await fs.symlink(badHub, surface.skillsDir);
    const report = await verifyClient(spec, { workspace: home });
    const check = checkByName(report.checks, "skills-dir");
    expect(check.status).toBe("FAIL");
    expect(check.message).toContain("mind-nerve-router");
  });

  it("skills-dir FAIL: dangling symlink", async () => {
    const { home, spec } = await wiredCodex();
    const surface = spec.skillSurface;
    if (surface === null) throw new Error("fixture");
    await fs.rm(surface.skillsDir, { recursive: true, force: true });
    await fs.symlink(path.join(home, "does-not-exist"), surface.skillsDir);
    const report = await verifyClient(spec, { workspace: home });
    expect(checkByName(report.checks, "skills-dir").status).toBe("FAIL");
  });

  it("skills-dir PASS: symlink to a hub containing mind-nerve-router", async () => {
    const { home, hub, spec } = await (async () => {
      const r = await wiredCodex();
      const hubDir = path.join(r.home, ".agents", "skills-hub");
      return { home: r.home, hub: hubDir, spec: r.spec };
    })();
    const surface = spec.skillSurface;
    if (surface === null) throw new Error("fixture");
    await fs.rm(surface.skillsDir, { recursive: true, force: true });
    await fs.symlink(hub, surface.skillsDir);
    const report = await verifyClient(spec, { workspace: home });
    expect(checkByName(report.checks, "skills-dir").status).toBe("PASS");
  });

  it("env-pins FAIL: a pin removed from the wrapper", async () => {
    const { home, spec } = await wiredCodex();
    const surface = spec.skillSurface;
    if (surface === null) throw new Error("fixture");
    const text = await fs.readFile(surface.hookScriptPath, "utf8");
    expect(text).toContain("export MIND_NERVE_SOCKET=");
    const stripped = text
      .split("\n")
      .filter((l) => !l.startsWith("export MIND_NERVE_SOCKET="))
      .join("\n");
    await fs.writeFile(surface.hookScriptPath, stripped, { mode: 0o755 });
    const report = await verifyClient(spec, { workspace: home });
    const check = checkByName(report.checks, "env-pins");
    expect(check.status).toBe("FAIL");
    expect(check.message).toContain("MIND_NERVE_SOCKET");
  });

  it("mcp-entry FAIL: mind-nerve section removed (TOML)", async () => {
    const { home, spec } = await wiredCodex();
    let text = await fs.readFile(spec.configPath, "utf8");
    text = text.replace(
      /# mind-nerve managed\n\[mcp_servers\.mind-nerve\][^\[]*/g,
      "",
    );
    await fs.writeFile(spec.configPath, text, "utf8");
    const report = await verifyClient(spec, { workspace: home });
    expect(checkByName(report.checks, "mcp-entry").status).toBe("FAIL");
  });

  it("mcp-command FAIL: pinned venv binary deleted", async () => {
    const { home, spec, socketPath } = await (async () => {
      const r = await wiredCodex();
      return { home: r.home, spec: r.spec, socketPath: r.socketPath };
    })();
    // The installed entry pins <home>/bin/mind-nerve-mcp — delete it.
    await fs.rm(path.join(home, "bin", "mind-nerve-mcp"));
    const report = await verifyClient(spec, { workspace: home, socketPath });
    const check = checkByName(report.checks, "mcp-command");
    expect(check.status).toBe("FAIL");
    expect(check.hint).toContain("--mcp-launcher uvx");
  });
});

// ---------------------------------------------------------------------------
// JSON CLI broken variants.
// ---------------------------------------------------------------------------

describe("verifyClient — gemini-like JSON failure classes", () => {
  async function wiredGemini(): Promise<{ home: string; spec: AgentSpec }> {
    const { home, hub, fakeBin, socketPath } = await makeHome(".gemini");
    const spec = makeSpec({
      base: "gemini",
      home,
      cliDir: ".gemini",
      fmt: "json-hooks",
      configFile: "settings.json",
    });
    await fullInstall(spec, home, hub, fakeBin, socketPath);
    return { home, spec };
  }

  it("config FAIL: JSON does not parse", async () => {
    const { home, spec } = await wiredGemini();
    await fs.writeFile(spec.configPath, "{ not json", "utf8");
    const report = await verifyClient(spec, { workspace: home });
    expect(checkByName(report.checks, "config").status).toBe("FAIL");
  });

  it("hook-config FAIL: hooks registered for a different command", async () => {
    const { home, spec } = await wiredGemini();
    const parsed = JSON.parse(
      await fs.readFile(spec.configPath, "utf8"),
    ) as Record<string, unknown>;
    const surface = spec.skillSurface;
    if (surface === null) throw new Error("fixture");
    parsed["hooks"] = {
      UserPromptSubmit: [
        { matcher: "", hooks: [{ type: "command", command: "/usr/bin/other" }] },
      ],
      SessionStart: [
        { matcher: "", hooks: [{ type: "command", command: "/usr/bin/other" }] },
      ],
    };
    await fs.writeFile(spec.configPath, JSON.stringify(parsed, null, 2), "utf8");
    const report = await verifyClient(spec, { workspace: home });
    const check = checkByName(report.checks, "hook-config");
    expect(check.status).toBe("FAIL");
    expect(check.message).toContain("UserPromptSubmit");
  });

  it("mcp-entry FAIL: entry present but managed marker stripped", async () => {
    const { home, spec } = await wiredGemini();
    const parsed = JSON.parse(
      await fs.readFile(spec.configPath, "utf8"),
    ) as Record<string, unknown>;
    const servers = parsed["mcpServers"] as Record<string, unknown>;
    const entry = servers["mind-nerve"] as Record<string, unknown>;
    delete entry["_comment"];
    await fs.writeFile(spec.configPath, JSON.stringify(parsed, null, 2), "utf8");
    const report = await verifyClient(spec, { workspace: home });
    const check = checkByName(report.checks, "mcp-entry");
    expect(check.status).toBe("FAIL");
    expect(check.message).toContain("managed marker");
  });

  it("mcp-entry FAIL: entry absent entirely", async () => {
    const { home, spec } = await wiredGemini();
    const parsed = JSON.parse(
      await fs.readFile(spec.configPath, "utf8"),
    ) as Record<string, unknown>;
    delete parsed["mcpServers"];
    await fs.writeFile(spec.configPath, JSON.stringify(parsed, null, 2), "utf8");
    const report = await verifyClient(spec, { workspace: home });
    expect(checkByName(report.checks, "mcp-entry").status).toBe("FAIL");
  });

  it("mcp-entry FAIL: legacy mcp-facade entry shape", async () => {
    const { home, spec } = await wiredGemini();
    const parsed = JSON.parse(
      await fs.readFile(spec.configPath, "utf8"),
    ) as Record<string, unknown>;
    const servers = parsed["mcpServers"] as Record<string, unknown>;
    const entry = servers["mind-nerve"] as Record<string, unknown>;
    // Legacy shape written by older installers — managed marker intact, but
    // the args still invoke the removed `mcp-facade` subcommand.
    entry["args"] = ["mcp-facade", "--config", "/tmp/upstream.json"];
    await fs.writeFile(spec.configPath, JSON.stringify(parsed, null, 2), "utf8");
    const report = await verifyClient(spec, { workspace: home });
    const check = checkByName(report.checks, "mcp-entry");
    expect(check.status).toBe("FAIL");
    expect(check.message).toContain("mcp-facade");
  });
});

// ---------------------------------------------------------------------------
// kimi: docs-derived shapes — flat [[hooks]] in config.toml + mcp.json MCP.
// ---------------------------------------------------------------------------

describe("verifyClient — kimi (flat [[hooks]] + mcp.json)", () => {
  /** kimi spec with all paths redirected into the fake home. */
  function makeKimiSpec(home: string): AgentSpec {
    const surface: SkillSurface = {
      skillsDir: path.join(home, ".kimi-code", "skills"),
      hookScriptPath: path.join(home, ".kimi-code", "hooks", "mind-nerve-hook"),
      hookConfigPath: path.join(home, ".kimi-code", "config.toml"),
      hookWireFmt: "toml-hooks-kimi",
      hookEvents: ["UserPromptSubmit", "SessionStart"],
      hookTimeoutSecs: 8,
      verified: "fixture",
    };
    const base = AGENT_REGISTRY.get("kimi");
    if (base === undefined) throw new Error("bad fixture base kimi");
    return {
      ...base,
      configPath: surface.hookConfigPath,
      mcpPath: path.join(home, ".kimi-code", "mcp.json"),
      detectPaths: [],
      detectBinaries: [],
      skillSurface: surface,
    };
  }

  async function wiredKimi(): Promise<{ home: string; spec: AgentSpec }> {
    const { home, hub, fakeBin, socketPath } = await makeHome(".kimi-code");
    const spec = makeKimiSpec(home);
    await fullInstall(spec, home, hub, fakeBin, socketPath);
    return { home, spec };
  }

  it("install writes the documented shapes and verifies clean", async () => {
    const { home, spec } = await wiredKimi();

    // Hook block: flat [[hooks]] entries, no Claude shape.
    const config = await fs.readFile(spec.configPath, "utf8");
    expect(config).toContain("[[hooks]]");
    expect(config).toContain('event = "UserPromptSubmit"');
    expect(config).not.toContain("[[hooks.");

    // MCP entry: mcp.json mcpServers.mind-nerve, NOT config.toml.
    expect(config).not.toContain("mcp_servers");
    const mcp = JSON.parse(
      await fs.readFile(spec.mcpPath as string, "utf8"),
    ) as Record<string, Record<string, Record<string, unknown>>>;
    expect(mcp["mcpServers"]?.["mind-nerve"]?.["command"]).toBe(
      path.join(home, "bin", "mind-nerve-mcp"),
    );

    const report = await verifyClient(spec, { workspace: home });
    for (const name of [
      "config",
      "hook-config",
      "hook-script",
      "hook-failopen",
      "skills-dir",
      "env-pins",
      "mcp-entry",
      "mcp-command",
    ]) {
      const check = checkByName(report.checks, name);
      expect(check.status, `${name}: ${check.message}`).toBe("PASS");
    }
  });

  it("hook-config FAIL: Claude [[hooks.<Event>]] shape is not what kimi reads", async () => {
    const { home, spec } = await wiredKimi();
    const surface = spec.skillSurface;
    if (surface === null) throw new Error("fixture");
    // Simulate an install written by the OLD code: Claude-shape block.
    const claudeBlock =
      "# BEGIN mind-nerve hook (managed — do not edit)\n" +
      surface.hookEvents
        .map(
          (event) =>
            `[[hooks.${event}]]\nhooks = [\n  { type = "command", command = ` +
            `${JSON.stringify(surface.hookScriptPath)}, timeout = 8 },\n]\n`,
        )
        .join("\n") +
      "# END mind-nerve hook\n";
    await fs.writeFile(spec.configPath, claudeBlock, "utf8");
    const report = await verifyClient(spec, { workspace: home });
    const check = checkByName(report.checks, "hook-config");
    expect(check.status).toBe("FAIL");
    expect(check.message).toContain("kimi-code does not read");
  });

  it("mcp-entry FAIL: entry in config.toml instead of mcp.json is not enough", async () => {
    const { home, spec } = await wiredKimi();
    await fs.rm(spec.mcpPath as string);
    const report = await verifyClient(spec, { workspace: home });
    expect(checkByName(report.checks, "mcp-entry").status).toBe("FAIL");
  });
});

// ---------------------------------------------------------------------------
// Aggregation helpers.
// ---------------------------------------------------------------------------

describe("verify aggregation", () => {
  it("worstStatus ranks FAIL > WARN > PASS", () => {
    const mk = (status: VerifyCheck["status"]): VerifyCheck => ({
      name: "x",
      status,
      message: "m",
      hint: null,
    });
    expect(worstStatus([mk("PASS"), mk("WARN")])).toBe("WARN");
    expect(worstStatus([mk("PASS"), mk("WARN"), mk("FAIL")])).toBe("FAIL");
    expect(worstStatus([mk("PASS")])).toBe("PASS");
  });

  it("summarize counts checks across clients", async () => {
    const { home, hub, fakeBin, socketPath } = await makeHome(".codex");
    const spec = makeSpec({
      base: "codex",
      home,
      cliDir: ".codex",
      fmt: "toml-hooks",
      configFile: "config.toml",
    });
    await fullInstall(spec, home, hub, fakeBin, socketPath);
    const ok = await verifyClient(spec, { workspace: home, socketPath });
    const summary = summarize([ok]);
    expect(summary.clients).toBe(1);
    expect(summary.fail).toBe(0);
    expect(summary.warn).toBe(1); // daemon-socket WARN
    expect(summary.pass).toBeGreaterThan(0);
    expect(summary.pass + summary.warn + summary.fail).toBe(ok.checks.length);
  });
});

// ---------------------------------------------------------------------------
// Vibe transport schema (codex#16) + structured-config parsing (codex#17)
// ---------------------------------------------------------------------------

describe("verifyClient — vibe transport schema", () => {
  async function installVibe(home: string, fakeBin: string): Promise<AgentSpec> {
    const configPath = path.join(home, ".vibe", "config.toml");
    const spec: AgentSpec = {
      ...AGENT_REGISTRY.get("vibe")!,
      configPath,
      mcpPath: configPath,
      detectPaths: [],
      detectBinaries: [],
    };
    const result = await installClient(spec, {
      mindNerveBin: fakeBin,
      workspace: home,
      _skipBackup: true,
    });
    expect(result.changed).toBe(true);
    return spec;
  }

  it("PASS: installed vibe entry carries transport = stdio", async () => {
    const { home, fakeBin } = await makeHome(".vibe");
    const spec = await installVibe(home, fakeBin);
    const report = await verifyClient(spec, { workspace: home, socketPath: null });
    expect(checkByName(report.checks, "mcp-schema").status).toBe("PASS");
  });

  it("FAIL: transport stripped from the entry (Vibe 2.9.6 rejects it)", async () => {
    const { home, fakeBin } = await makeHome(".vibe");
    const spec = await installVibe(home, fakeBin);
    const cfgPath = spec.mcpPath as string;
    const text = await fs.readFile(cfgPath, "utf8");
    expect(text).toContain('transport = "stdio"');
    await fs.writeFile(cfgPath, text.replace('transport = "stdio", ', ""), "utf8");
    const report = await verifyClient(spec, { workspace: home, socketPath: null });
    const schema = checkByName(report.checks, "mcp-schema");
    expect(schema.status).toBe("FAIL");
    expect(schema.message).toContain("transport");
    expect(hasFailures([report])).toBe(true);
  });
});

describe("verifyClient — structured config parsing (codex#17)", () => {
  it("cody: a legacy prose-appended config.json FAILs the parse check", async () => {
    const home = await makeTmp();
    await fs.mkdir(path.join(home, ".cody"), { recursive: true });
    await fs.writeFile(
      path.join(home, ".cody", "config.json"),
      '{"model":"x"}\n\n# mind-nerve managed\nprose prose\n# end mind-nerve managed\n',
    );
    const spec = {
      ...AGENT_REGISTRY.get("cody")!,
      detectPaths: [],
      detectBinaries: [],
    };
    const report = await verifyClient(spec, { workspace: home, socketPath: null });
    expect(checkByName(report.checks, "config").status).toBe("FAIL");
    // The marker is still seen — the failure is the parse, not the block.
    expect(checkByName(report.checks, "instruction-block").status).toBe("PASS");
  });

  it("cody: a format-valid managed config PASSes", async () => {
    const home = await makeTmp();
    const spec = { ...AGENT_REGISTRY.get("cody")!, detectPaths: [], detectBinaries: [] };
    await installClient(spec, {
      mindNerveBin: "/nonexistent/bin/mind-nerve",
      workspace: home,
      _skipBackup: true,
    });
    const report = await verifyClient(spec, { workspace: home, socketPath: null });
    expect(checkByName(report.checks, "config").status).toBe("PASS");
  });

  it("aider: legacy prose in .aider.conf.yml FAILs the parse check", async () => {
    const home = await makeTmp();
    await fs.writeFile(
      path.join(home, ".aider.conf.yml"),
      "read: [AGENTS.md]\n\n# mind-nerve managed\nprose prose\n# end mind-nerve managed\n",
    );
    const spec = {
      ...AGENT_REGISTRY.get("aider")!,
      detectPaths: [],
      detectBinaries: [],
    };
    const report = await verifyClient(spec, { workspace: home, socketPath: null });
    expect(checkByName(report.checks, "config").status).toBe("FAIL");
  });
});

describe("verifyClient — mcp-env-pin", () => {
  it("WARN: populated runtime dir exists but the entry lacks the pin", async () => {
    const { home, hub, fakeBin, socketPath } = await makeHome(".gemini");
    const spec = makeSpec({
      base: "gemini",
      home,
      cliDir: ".gemini",
      fmt: "json-hooks",
      configFile: "settings.json",
    });
    const rt = path.join(home, "rt");
    await fs.mkdir(rt, { recursive: true });
    await fs.writeFile(path.join(rt, "manifest.json"), "{}\n");

    const prev = process.env.MIND_NERVE_RUNTIME_DIR;
    process.env.MIND_NERVE_RUNTIME_DIR = rt;
    try {
      await fullInstall(spec, home, hub, fakeBin, socketPath);
      // Strip the pin the installer wrote — simulating drift.
      const cfgPath = spec.mcpPath as string;
      const cfg = JSON.parse(await fs.readFile(cfgPath, "utf8")) as Record<
        string,
        Record<string, Record<string, Record<string, string>>>
      >;
      delete cfg["mcpServers"]!["mind-nerve"]!["env"]!["MIND_NERVE_RUNTIME_DIR"];
      await fs.writeFile(cfgPath, JSON.stringify(cfg, null, 2) + "\n");

      const report = await verifyClient(spec, { workspace: home, socketPath: null });
      const pin = checkByName(report.checks, "mcp-env-pin");
      expect(pin.status).toBe("WARN");
      expect(pin.message).toContain("does not pin");
      expect(report.status).toBe("WARN");
      expect(hasFailures([report])).toBe(false);
    } finally {
      if (prev === undefined) delete process.env.MIND_NERVE_RUNTIME_DIR;
      else process.env.MIND_NERVE_RUNTIME_DIR = prev;
    }
  });

  it("WARN: pin points at a dir that is no longer populated", async () => {
    const { home, hub, fakeBin, socketPath } = await makeHome(".codex");
    const spec = makeSpec({
      base: "codex",
      home,
      cliDir: ".codex",
      fmt: "toml-hooks",
      configFile: "config.toml",
    });
    const rt = path.join(home, "rt");
    await fs.mkdir(rt, { recursive: true });
    const manifest = path.join(rt, "manifest.json");
    await fs.writeFile(manifest, "{}\n");

    const prev = process.env.MIND_NERVE_RUNTIME_DIR;
    process.env.MIND_NERVE_RUNTIME_DIR = rt;
    try {
      await fullInstall(spec, home, hub, fakeBin, socketPath);
      await fs.rm(manifest); // runtime dir wiped after install -> stale pin

      const report = await verifyClient(spec, { workspace: home, socketPath: null });
      const pin = checkByName(report.checks, "mcp-env-pin");
      expect(pin.status).toBe("WARN");
      expect(pin.message).toContain("stale pin");
    } finally {
      if (prev === undefined) delete process.env.MIND_NERVE_RUNTIME_DIR;
      else process.env.MIND_NERVE_RUNTIME_DIR = prev;
    }
  });

  it("PASS: pin present and the pinned dir is populated", async () => {
    const { home, hub, fakeBin, socketPath } = await makeHome(".codex");
    const spec = makeSpec({
      base: "codex",
      home,
      cliDir: ".codex",
      fmt: "toml-hooks",
      configFile: "config.toml",
    });
    const rt = path.join(home, "rt");
    await fs.mkdir(rt, { recursive: true });
    await fs.writeFile(path.join(rt, "manifest.json"), "{}\n");

    const prev = process.env.MIND_NERVE_RUNTIME_DIR;
    process.env.MIND_NERVE_RUNTIME_DIR = rt;
    try {
      await fullInstall(spec, home, hub, fakeBin, socketPath);
      const report = await verifyClient(spec, { workspace: home, socketPath: null });
      const pin = checkByName(report.checks, "mcp-env-pin");
      expect(pin.status).toBe("PASS");
      expect(pin.message).toContain(rt);
    } finally {
      if (prev === undefined) delete process.env.MIND_NERVE_RUNTIME_DIR;
      else process.env.MIND_NERVE_RUNTIME_DIR = prev;
    }
  });
});
