// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.
//
// `verify` — self-test an existing installation per CLI, the way
// Qwen-MM-Plugins' guided installer self-verifies after driving each
// harness's native install.
//
// Per CLI, as applicable to that CLI's wiring:
//   * the primary config exists, parses (JSON/TOML per CLI), and still shows
//     the mind-nerve managed markers the installer wrote;
//   * the hook wrapper exists, is executable, and answers `{}` on stdin with
//     valid JSON within 5s (the fail-open contract);
//   * the skills dir is consistent (router-only real dir, or a symlink whose
//     target exists and contains mind-nerve-router);
//   * the MCP entry is present and its command binary resolves;
//   * the env pins the installer exported into the hook wrapper survive;
//   * the daemon socket answers — WARN, never FAIL: the daemon is allowed to
//     be on-demand (mind-nerve-routed-ensure starts it on SessionStart).

import fs from "node:fs/promises";
import path from "node:path";
import net from "node:net";
import { spawn } from "node:child_process";
import which from "which";
import TOML from "@iarna/toml";
import YAML from "js-yaml";
import { type AgentSpec, type SkillSurface } from "./registry.js";
import { inspectSkillsDir, ROUTER_SKILL_NAME } from "./skills_dir.js";
import { HOOK_BLOCK_BEGIN } from "./hook_wiring.js";
import { BLOCK_MARKER } from "./instruction_block.js";
import { isJsonMcpFmt, UVX_ENTRYPOINT } from "./mcp_rewire.js";
import { existingRuntimeDir, runtimeDirPopulated } from "./install.js";

export type CheckStatus = "PASS" | "WARN" | "FAIL";

export interface VerifyCheck {
  /** Stable machine-readable check name (e.g. "hook-failopen"). */
  readonly name: string;
  readonly status: CheckStatus;
  readonly message: string;
  /** Actionable next step when status is WARN/FAIL, else null. */
  readonly hint: string | null;
}

export interface ClientReport {
  readonly client: string;
  readonly checks: readonly VerifyCheck[];
  /** Worst status across checks. */
  readonly status: CheckStatus;
}

export interface VerifyOptions {
  /** Workspace directory for workspace-relative config paths. */
  readonly workspace?: string;
  /**
   * Daemon socket to probe. Null/undefined skips the daemon check entirely.
   * A missing or unanswered socket is a WARN, never a FAIL — the daemon is
   * allowed to be on-demand.
   */
  readonly socketPath?: string | null;
  /** Hook fail-open probe timeout. Default 5000 ms. */
  readonly hookTimeoutMs?: number;
  /** Daemon socket connect timeout. Default 2000 ms. */
  readonly socketTimeoutMs?: number;
}

export interface VerifySummary {
  readonly clients: number;
  readonly pass: number;
  readonly warn: number;
  readonly fail: number;
}

const DEFAULT_HOOK_TIMEOUT_MS = 5000;
const DEFAULT_SOCKET_TIMEOUT_MS = 2000;

/** Env pins wireClient exports into every per-CLI hook wrapper. */
const EXPECTED_WRAPPER_ENV: readonly string[] = [
  "MIND_NERVE_SOCKET",
  "MIND_NERVE_SOURCE_DIR",
  "MIND_NERVE_PROJECTED_DIR",
];

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Verifies one client's installation. Never throws for a broken install —
 * breakage is reported as FAIL checks. Throws only for unknown formats.
 */
export async function verifyClient(
  spec: AgentSpec,
  opts: VerifyOptions = {},
): Promise<ClientReport> {
  const ws = opts.workspace ?? process.cwd();
  const checks: VerifyCheck[] = [];
  const installHint = `re-run \`mind-nerve install ${spec.name}\``;

  if (!(await looksInstalled(spec, ws))) {
    checks.push({
      name: "installed",
      status: "FAIL",
      message: "no mind-nerve installation found for this client",
      hint: `run \`mind-nerve install ${spec.name}\``,
    });
    return { client: spec.name, checks, status: "FAIL" };
  }

  // -- Primary config: exists + parses (structured formats only) ------------
  await checkConfig(spec, ws, checks);

  // -- Instruction block (workspace-rules clients) --------------------------
  if (spec.instructionFilePath !== null) {
    await checkInstructionBlock(spec, ws, checks, installHint);
  }

  // -- Skill surface: hook registration, wrapper, skills dir, env pins ------
  if (spec.skillSurface !== null) {
    await checkSkillSurface(spec, checks, opts, installHint);
  }

  // -- MCP entry + command resolution ---------------------------------------
  if (spec.mcpFmt !== null && spec.mcpPath !== null) {
    await checkMcp(spec, ws, checks, installHint);
  }

  // -- Daemon socket (WARN-only) --------------------------------------------
  const socketPath = opts.socketPath ?? null;
  if (socketPath !== null) {
    checks.push(
      await probeDaemonSocket(
        socketPath,
        opts.socketTimeoutMs ?? DEFAULT_SOCKET_TIMEOUT_MS,
      ),
    );
  }

  return { client: spec.name, checks, status: worstStatus(checks) };
}

/** Worst status across a check list (FAIL > WARN > PASS). */
export function worstStatus(checks: readonly VerifyCheck[]): CheckStatus {
  if (checks.some((c) => c.status === "FAIL")) return "FAIL";
  if (checks.some((c) => c.status === "WARN")) return "WARN";
  return "PASS";
}

/** Aggregates per-client reports into pass/warn/fail counts. */
export function summarize(reports: readonly ClientReport[]): VerifySummary {
  let pass = 0;
  let warn = 0;
  let fail = 0;
  for (const r of reports) {
    for (const c of r.checks) {
      if (c.status === "PASS") pass++;
      else if (c.status === "WARN") warn++;
      else fail++;
    }
  }
  return { clients: reports.length, pass, warn, fail };
}

/** True when any check in any report is a FAIL — drives the exit code. */
export function hasFailures(reports: readonly ClientReport[]): boolean {
  return reports.some((r) => r.status === "FAIL");
}

/**
 * True when anything the installer would have written exists on disk.
 * Used by `verify` with no --cli to select clients worth checking.
 */
export async function looksInstalled(
  spec: AgentSpec,
  workspace: string,
): Promise<boolean> {
  const candidates: string[] = [
    resolvePath(spec.configPath, workspace),
  ];
  if (spec.mcpPath !== null) candidates.push(resolvePath(spec.mcpPath, workspace));
  if (spec.instructionFilePath !== null) {
    candidates.push(resolvePath(spec.instructionFilePath, workspace));
  }
  if (spec.skillSurface !== null) {
    candidates.push(spec.skillSurface.hookScriptPath);
    candidates.push(spec.skillSurface.skillsDir);
  }
  for (const p of candidates) {
    if (await pathExists(p)) return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Config check
// ---------------------------------------------------------------------------

async function checkConfig(
  spec: AgentSpec,
  ws: string,
  checks: VerifyCheck[],
): Promise<void> {
  // When the config file doubles as the instruction file (cody, aider), the
  // managed block is serialized FORMAT-VALID (a JSON key / YAML comment
  // lines), so the structured parse still applies — a config the CLI cannot
  // parse must never PASS (codex#17).
  if (spec.configFmt === "text-block") return;

  const configPath = resolvePath(spec.configPath, ws);
  let text: string;
  try {
    text = await fs.readFile(configPath, "utf8");
  } catch {
    checks.push({
      name: "config",
      status: "FAIL",
      message: `config file missing: ${configPath}`,
      hint: `run \`mind-nerve install ${spec.name}\``,
    });
    return;
  }

  const parsed = parseConfig(spec.configFmt, text);
  if (parsed !== null) {
    checks.push({
      name: "config",
      status: "FAIL",
      message: `config file does not parse (${spec.configFmt}): ${parsed}`,
      hint: "fix the syntax error by hand, or restore the latest .bak-mind-nerve-* backup",
    });
    return;
  }

  checks.push({
    name: "config",
    status: "PASS",
    message: `config parses (${spec.configFmt}): ${configPath}`,
    hint: null,
  });
}

/** Returns null on success, else the parse error message. */
function parseConfig(fmt: AgentSpec["configFmt"], text: string): string | null {
  try {
    if (fmt.startsWith("json-")) {
      JSON.parse(text);
    } else if (fmt.startsWith("toml-")) {
      TOML.parse(text);
    } else if (fmt === "yaml-aider") {
      YAML.load(text);
    }
    // text-block is not parse-checked.
    return null;
  } catch (err) {
    return err instanceof Error ? err.message : String(err);
  }
}

// ---------------------------------------------------------------------------
// Instruction block check
// ---------------------------------------------------------------------------

async function checkInstructionBlock(
  spec: AgentSpec,
  ws: string,
  checks: VerifyCheck[],
  installHint: string,
): Promise<void> {
  const filePath = resolvePath(spec.instructionFilePath as string, ws);
  let text: string;
  try {
    text = await fs.readFile(filePath, "utf8");
  } catch {
    checks.push({
      name: "instruction-block",
      status: "FAIL",
      message: `instruction file missing: ${filePath}`,
      hint: installHint,
    });
    return;
  }
  if (!text.includes(BLOCK_MARKER)) {
    checks.push({
      name: "instruction-block",
      status: "FAIL",
      message: `managed block marker not found in ${filePath}`,
      hint: installHint,
    });
    return;
  }
  checks.push({
    name: "instruction-block",
    status: "PASS",
    message: `managed block present in ${filePath}`,
    hint: null,
  });
}

// ---------------------------------------------------------------------------
// Skill surface checks
// ---------------------------------------------------------------------------

async function checkSkillSurface(
  spec: AgentSpec,
  checks: VerifyCheck[],
  opts: VerifyOptions,
  installHint: string,
): Promise<void> {
  const surface = spec.skillSurface as SkillSurface;

  // -- Hook registration in the CLI config ----------------------------------
  checks.push(await checkHookRegistration(surface, spec.name, installHint));

  // -- Wrapper script: exists + executable ----------------------------------
  const scriptOk = await checkHookScript(surface, checks, installHint);

  // -- Skills dir state ------------------------------------------------------
  // Checked BEFORE the fail-open probe on purpose: spawning the wrapper runs
  // the real hook, which re-projects the skills dir as a side effect. A probe
  // first would heal a missing/broken skills dir before we ever looked at it.
  checks.push(await checkSkillsDir(surface));

  // -- Env pins in the wrapper ----------------------------------------------
  if (scriptOk) {
    checks.push(await checkEnvPins(surface, installHint));
  }

  // -- Fail-open probe: `{}` on stdin must yield valid JSON within 5s -------
  // Runs LAST among the skill-surface checks — it is the only mutating one.
  if (scriptOk) {
    checks.push(
      await probeHookFailOpen(
        surface.hookScriptPath,
        opts.hookTimeoutMs ?? DEFAULT_HOOK_TIMEOUT_MS,
      ),
    );
  }
}

async function checkHookRegistration(
  surface: SkillSurface,
  clientName: string,
  installHint: string,
): Promise<VerifyCheck> {
  let text: string;
  try {
    text = await fs.readFile(surface.hookConfigPath, "utf8");
  } catch {
    return {
      name: "hook-config",
      status: "FAIL",
      message: `hook config missing: ${surface.hookConfigPath}`,
      hint: installHint,
    };
  }

  if (surface.hookWireFmt === "json-hooks") {
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch (err) {
      return {
        name: "hook-config",
        status: "FAIL",
        message: `hook config does not parse as JSON: ${surface.hookConfigPath} (${err instanceof Error ? err.message : String(err)})`,
        hint: "fix the syntax error by hand, or restore the latest .bak-mind-nerve-* backup",
      };
    }
    const missing = missingJsonHookEvents(parsed, surface);
    if (missing.length > 0) {
      return {
        name: "hook-config",
        status: "FAIL",
        message: `hook not registered for event(s) ${missing.join(", ")} in ${surface.hookConfigPath}`,
        hint: installHint,
      };
    }
    return {
      name: "hook-config",
      status: "PASS",
      message: `hook registered on ${surface.hookEvents.join(" + ")}`,
      hint: null,
    };
  }

  // TOML formats: validate the managed block in the shape that CLI actually
  // parses. Scope every assertion to the block between the markers — foreign
  // config elsewhere in the file must not satisfy or break the check.
  if (!text.includes(HOOK_BLOCK_BEGIN)) {
    return {
      name: "hook-config",
      status: "FAIL",
      message: `managed hook block not found in ${surface.hookConfigPath}`,
      hint: installHint,
    };
  }
  const begin = text.indexOf(HOOK_BLOCK_BEGIN);
  const endIdx = text.indexOf("# END mind-nerve hook", begin);
  const block = text.slice(begin, endIdx === -1 ? undefined : endIdx);

  if (!block.includes(`command = ${JSON.stringify(surface.hookScriptPath)}`)) {
    return {
      name: "hook-config",
      status: "FAIL",
      message: `managed hook block in ${surface.hookConfigPath} does not point at ${surface.hookScriptPath}`,
      hint: installHint,
    };
  }

  if (surface.hookWireFmt === "toml-hooks-kimi") {
    // kimi reads ONLY flat [[hooks]] elements with event/command/timeout;
    // the Claude [[hooks.<Event>]] shape is silently dead there.
    if (block.includes("[[hooks.")) {
      return {
        name: "hook-config",
        status: "FAIL",
        message: `managed hook block in ${surface.hookConfigPath} uses the Claude [[hooks.<Event>]] shape kimi-code does not read`,
        hint: `${installHint} to rewrite the block as flat [[hooks]] entries`,
      };
    }
    const missing = surface.hookEvents.filter(
      (event) => !block.includes(`event = ${JSON.stringify(event)}`),
    );
    if (missing.length > 0 || !block.includes("[[hooks]]")) {
      return {
        name: "hook-config",
        status: "FAIL",
        message: `managed hook block in ${surface.hookConfigPath} lacks [[hooks]] entries for event(s) ${missing.join(", ") || "(none found)"}`,
        hint: installHint,
      };
    }
    return {
      name: "hook-config",
      status: "PASS",
      message: `managed [[hooks]] block present on ${surface.hookEvents.join(" + ")}`,
      hint: null,
    };
  }

  // toml-hooks (codex, grok): one [[hooks.<Event>]] table per event.
  const missingTables = surface.hookEvents.filter(
    (event) => !block.includes(`[[hooks.${event}]]`),
  );
  if (missingTables.length > 0) {
    return {
      name: "hook-config",
      status: "FAIL",
      message: `managed hook block in ${surface.hookConfigPath} lacks [[hooks.<Event>]] table(s) for ${missingTables.join(", ")}`,
      hint: installHint,
    };
  }
  return {
    name: "hook-config",
    status: "PASS",
    message: `managed hook block present in ${surface.hookConfigPath}`,
    hint: null,
  };
}

/** Events whose matcher groups never invoke the hook command. */
function missingJsonHookEvents(
  parsed: unknown,
  surface: SkillSurface,
): string[] {
  const hooks =
    typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)["hooks"]
      : undefined;
  const missing: string[] = [];
  for (const event of surface.hookEvents) {
    const groups =
      typeof hooks === "object" && hooks !== null && !Array.isArray(hooks)
        ? (hooks as Record<string, unknown>)[event]
        : undefined;
    const found =
      Array.isArray(groups) &&
      groups.some((g: unknown) => {
        if (typeof g !== "object" || g === null) return false;
        const inner = (g as Record<string, unknown>)["hooks"];
        return (
          Array.isArray(inner) &&
          inner.some(
            (hk: unknown) =>
              typeof hk === "object" &&
              hk !== null &&
              (hk as Record<string, unknown>)["command"] ===
                surface.hookScriptPath,
          )
        );
      });
    if (!found) missing.push(event);
  }
  return missing;
}

/** Returns true when the wrapper exists and is executable. */
async function checkHookScript(
  surface: SkillSurface,
  checks: VerifyCheck[],
  installHint: string,
): Promise<boolean> {
  try {
    await fs.access(surface.hookScriptPath, fs.constants.X_OK);
  } catch {
    let exists = false;
    try {
      await fs.access(surface.hookScriptPath);
      exists = true;
    } catch {
      // missing entirely
    }
    checks.push({
      name: "hook-script",
      status: "FAIL",
      message: exists
        ? `hook script is not executable: ${surface.hookScriptPath}`
        : `hook script missing: ${surface.hookScriptPath}`,
      hint: exists
        ? `chmod +x ${surface.hookScriptPath}, or ${installHint}`
        : installHint,
    });
    return false;
  }
  checks.push({
    name: "hook-script",
    status: "PASS",
    message: `hook script present and executable: ${surface.hookScriptPath}`,
    hint: null,
  });
  return true;
}

/**
 * The fail-open contract: `{}` on stdin must produce valid JSON on stdout
 * within the timeout. A broken router must never block a prompt.
 */
async function probeHookFailOpen(
  scriptPath: string,
  timeoutMs: number,
): Promise<VerifyCheck> {
  const run = await runHookOnce(scriptPath, "{}", timeoutMs);
  if (run.timedOut) {
    return {
      name: "hook-failopen",
      status: "FAIL",
      message: `hook did not answer within ${timeoutMs} ms`,
      hint: "check the hook wrapper and the shared hook it execs; the fail-open contract requires an immediate JSON answer",
    };
  }
  if (run.spawnError !== null) {
    return {
      name: "hook-failopen",
      status: "FAIL",
      message: `hook could not be spawned: ${run.spawnError}`,
      hint: "check the wrapper's shebang and exec target",
    };
  }
  try {
    JSON.parse(run.stdout.trim());
  } catch {
    return {
      name: "hook-failopen",
      status: "FAIL",
      message: `hook stdout is not valid JSON (exit ${String(run.code)}): ${run.stdout.slice(0, 120)}`,
      hint: "the fail-open contract requires valid JSON on stdout for any input — re-run install to regenerate the wrapper",
    };
  }
  return {
    name: "hook-failopen",
    status: "PASS",
    message: "hook answered `{}` probe with valid JSON (fail-open intact)",
    hint: null,
  };
}

interface HookRunResult {
  readonly stdout: string;
  readonly code: number | null;
  readonly timedOut: boolean;
  readonly spawnError: string | null;
}

function runHookOnce(
  scriptPath: string,
  stdinText: string,
  timeoutMs: number,
): Promise<HookRunResult> {
  return new Promise((resolve) => {
    let stdout = "";
    let timedOut = false;
    let spawnError: string | null = null;
    let settled = false;

    const finish = (code: number | null): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve({ stdout, code, timedOut, spawnError });
    };

    const child = spawn(scriptPath, [], {
      stdio: ["pipe", "pipe", "pipe"],
    });

    const timer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGKILL");
      finish(null);
    }, timeoutMs);

    child.stdout.on("data", (d: Buffer) => (stdout += d.toString()));
    child.on("error", (err) => {
      spawnError = err.message;
      finish(null);
    });
    child.on("close", (code) => finish(code));
    // A hook that exits before reading stdin makes the write below EPIPE;
    // the close handler already settles the result, so swallow it.
    child.stdin.on("error", () => {});
    child.stdin.write(stdinText);
    child.stdin.end();
  });
}

async function checkSkillsDir(surface: SkillSurface): Promise<VerifyCheck> {
  const state = await inspectSkillsDir(surface.skillsDir);
  switch (state.kind) {
    case "realdir":
      if (state.looksManaged) {
        return {
          name: "skills-dir",
          status: "PASS",
          message: `skills dir is a managed router-only directory: ${surface.skillsDir}`,
          hint: null,
        };
      }
      return {
        name: "skills-dir",
        status: "FAIL",
        message: `skills dir exists but ${ROUTER_SKILL_NAME} is missing: ${surface.skillsDir}`,
        hint: "re-run install to refresh the router-only skills directory",
      };
    case "symlink": {
      const target = state.symlinkTarget;
      if (target === null) {
        return {
          name: "skills-dir",
          status: "FAIL",
          message: `skills path is an unreadable symlink: ${surface.skillsDir}`,
          hint: "repoint or remove the symlink, then re-run install",
        };
      }
      let entries: string[] = [];
      try {
        entries = await fs.readdir(target);
      } catch {
        return {
          name: "skills-dir",
          status: "FAIL",
          message: `skills symlink target does not exist: ${target}`,
          hint: "repoint the symlink at the skills hub, or re-run install",
        };
      }
      if (!entries.includes(ROUTER_SKILL_NAME)) {
        return {
          name: "skills-dir",
          status: "FAIL",
          message: `skills symlink target lacks ${ROUTER_SKILL_NAME}: ${target}`,
          hint: "the router skill must be reachable through the symlink target",
        };
      }
      return {
        name: "skills-dir",
        status: "PASS",
        message: `skills path is a symlink to a hub containing ${ROUTER_SKILL_NAME}: ${target}`,
        hint: null,
      };
    }
    case "absent":
      return {
        name: "skills-dir",
        status: "FAIL",
        message: `skills dir missing: ${surface.skillsDir}`,
        hint: "re-run install to create the router-only skills directory",
      };
    case "file":
      return {
        name: "skills-dir",
        status: "FAIL",
        message: `skills path is a regular file: ${surface.skillsDir}`,
        hint: "move the file aside, then re-run install",
      };
  }
}

async function checkEnvPins(
  surface: SkillSurface,
  installHint: string,
): Promise<VerifyCheck> {
  let text: string;
  try {
    text = await fs.readFile(surface.hookScriptPath, "utf8");
  } catch {
    return {
      name: "env-pins",
      status: "FAIL",
      message: `hook wrapper unreadable: ${surface.hookScriptPath}`,
      hint: installHint,
    };
  }
  const missing = EXPECTED_WRAPPER_ENV.filter(
    (key) => !text.includes(`export ${key}=`),
  );
  if (missing.length > 0) {
    return {
      name: "env-pins",
      status: "FAIL",
      message: `env pin(s) missing from hook wrapper: ${missing.join(", ")}`,
      hint: `${installHint} to regenerate the wrapper`,
    };
  }
  return {
    name: "env-pins",
    status: "PASS",
    message: `env pins present in hook wrapper (${EXPECTED_WRAPPER_ENV.join(", ")})`,
    hint: null,
  };
}

// ---------------------------------------------------------------------------
// MCP checks
// ---------------------------------------------------------------------------

interface McpEntryShape {
  readonly command: string;
  readonly args: readonly string[];
  readonly env: Readonly<Record<string, string>>;
  readonly transport: string | null;
  readonly managed: boolean;
}

async function checkMcp(
  spec: AgentSpec,
  ws: string,
  checks: VerifyCheck[],
  installHint: string,
): Promise<void> {
  const mcpPath = resolvePath(spec.mcpPath as string, ws);
  let text: string;
  try {
    text = await fs.readFile(mcpPath, "utf8");
  } catch {
    checks.push({
      name: "mcp-entry",
      status: "FAIL",
      message: `MCP config missing: ${mcpPath}`,
      hint: installHint,
    });
    return;
  }

  const entry = extractMcpEntry(spec, text);
  if (entry === "parse-error") {
    checks.push({
      name: "mcp-entry",
      status: "FAIL",
      message: `MCP config does not parse: ${mcpPath}`,
      hint: "fix the syntax error by hand, or restore the latest .bak-mind-nerve-* backup",
    });
    return;
  }
  if (entry === null) {
    checks.push({
      name: "mcp-entry",
      status: "FAIL",
      message: `no mind-nerve MCP entry in ${mcpPath}`,
      hint: installHint,
    });
    return;
  }
  if (!entry.managed) {
    checks.push({
      name: "mcp-entry",
      status: "FAIL",
      message: `mind-nerve MCP entry in ${mcpPath} lacks the managed marker (edited by hand?)`,
      hint: installHint,
    });
    return;
  }
  if (entry.args.includes("mcp-facade")) {
    // Legacy shape written by older installers: `mind-nerve mcp-facade
    // --config <path>`. The Python package has no mcp-facade subcommand —
    // the entry is dead weight even though the command still resolves.
    checks.push({
      name: "mcp-entry",
      status: "FAIL",
      message: `mind-nerve MCP entry in ${mcpPath} uses the removed mcp-facade subcommand`,
      hint: `${installHint} to rewrite the entry as the mind-nerve-mcp console script`,
    });
    return;
  }
  checks.push({
    name: "mcp-entry",
    status: "PASS",
    message: `mind-nerve MCP entry present in ${mcpPath}`,
    hint: null,
  });

  // Schema discriminator check (codex#16): Vibe 2.9.6 rejects mcp_servers
  // entries without transport = "stdio" — the entry would be dead weight
  // while every other check PASSes.
  if (spec.mcpRequiresStdioTransport === true) {
    checks.push(
      entry.transport === "stdio"
        ? {
            name: "mcp-schema",
            status: "PASS",
            message: 'mcp_servers entry carries transport = "stdio"',
            hint: null,
          }
        : {
            name: "mcp-schema",
            status: "FAIL",
            message:
              'mind-nerve MCP entry lacks transport = "stdio" — this client rejects it',
            hint: `${installHint} to rewrite the entry`,
          },
    );
  }

  checks.push(await checkMcpCommand(entry.command));

  checks.push(await checkMcpRuntimePin(entry));
}

/**
 * The env pin is what keeps the MCP on the local route table (fleet incident
 * 2026-07-10: without it the server silently loads the OSS catalog). The
 * installer pins MIND_NERVE_RUNTIME_DIR when a populated local runtime dir
 * exists — so a populated dir plus a missing pin, or a pin pointing at a dir
 * that is no longer populated, is a drifted install. WARN, never FAIL: an
 * unpinned entry still runs.
 */
async function checkMcpRuntimePin(entry: McpEntryShape): Promise<VerifyCheck> {
  const pinned = entry.env["MIND_NERVE_RUNTIME_DIR"];
  if (pinned !== undefined) {
    if (!(await runtimeDirPopulated(pinned))) {
      return {
        name: "mcp-env-pin",
        status: "WARN",
        message:
          `MCP entry pins MIND_NERVE_RUNTIME_DIR=${pinned} but that dir has ` +
          "no manifest.json/route_table.jsonl (stale pin)",
        hint: "re-run install to re-pin the populated runtime dir",
      };
    }
    return {
      name: "mcp-env-pin",
      status: "PASS",
      message: `MCP entry pins the local runtime dir (${pinned})`,
      hint: null,
    };
  }
  const local = await existingRuntimeDir();
  if (local !== null) {
    return {
      name: "mcp-env-pin",
      status: "WARN",
      message:
        `populated local runtime dir ${local} exists but the MCP entry ` +
        "does not pin MIND_NERVE_RUNTIME_DIR — the server will load the OSS catalog",
      hint: "re-run install to add the pin",
    };
  }
  return {
    name: "mcp-env-pin",
    status: "PASS",
    message: "no populated local runtime dir; nothing to pin",
    hint: null,
  };
}

/**
 * Pulls the mind-nerve entry out of an MCP config, per format.
 * Returns "parse-error" for unparseable text, null when the entry is absent.
 */
function extractMcpEntry(
  spec: AgentSpec,
  text: string,
): McpEntryShape | "parse-error" | null {
  const fmt = spec.mcpFmt;
  if (fmt === null) return null;

  if (isJsonMcpFmt(fmt)) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      return "parse-error";
    }
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      return "parse-error";
    }
    const root = parsed as Record<string, unknown>;
    const key = fmt === "mcp-json-zed" ? "context_servers" : "mcpServers";
    const servers = root[key];
    if (typeof servers !== "object" || servers === null || Array.isArray(servers)) {
      return null;
    }
    const entry = (servers as Record<string, unknown>)["mind-nerve"];
    if (typeof entry !== "object" || entry === null || Array.isArray(entry)) {
      return null;
    }
    const rec = entry as Record<string, unknown>;
    return {
      command: typeof rec["command"] === "string" ? rec["command"] : "",
      args: stringArray(rec["args"]),
      env: stringRecord(rec["env"]),
      transport: typeof rec["transport"] === "string" ? rec["transport"] : null,
      managed: rec["_comment"] === "mind-nerve managed",
    };
  }

  // TOML formats.
  let parsed: unknown;
  try {
    parsed = TOML.parse(text);
  } catch {
    return "parse-error";
  }
  const managed = text.includes("mind-nerve managed");
  const root = parsed as Record<string, unknown>;
  if (fmt === "mcp-toml-codex") {
    const servers = root["mcp_servers"];
    if (typeof servers !== "object" || servers === null) return null;
    const entry = (servers as Record<string, unknown>)["mind-nerve"];
    if (typeof entry !== "object" || entry === null) return null;
    const rec = entry as Record<string, unknown>;
    return {
      command: typeof rec["command"] === "string" ? rec["command"] : "",
      args: stringArray(rec["args"]),
      env: stringRecord(rec["env"]),
      transport: typeof rec["transport"] === "string" ? rec["transport"] : null,
      managed,
    };
  }
  // mcp-toml-vibe: mcp_servers is an array of inline tables.
  const arr = root["mcp_servers"];
  if (!Array.isArray(arr)) return null;
  for (const item of arr as unknown[]) {
    if (typeof item !== "object" || item === null) continue;
    const rec = item as Record<string, unknown>;
    if (rec["name"] === "mind-nerve") {
      return {
        command: typeof rec["command"] === "string" ? rec["command"] : "",
        args: stringArray(rec["args"]),
        env: stringRecord(rec["env"]),
        transport: typeof rec["transport"] === "string" ? rec["transport"] : null,
        managed,
      };
    }
  }
  return null;
}

/** Reads a TOML/JSON env field as a string record (anything else -> {}). */
function stringRecord(value: unknown): Readonly<Record<string, string>> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return {};
  }
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
    if (typeof v === "string") out[k] = v;
  }
  return out;
}

/** Reads a TOML/JSON args field as a string array (anything else -> []). */
function stringArray(value: unknown): readonly string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((x): x is string => typeof x === "string");
}

/** The entry's command binary must resolve: absolute path +X, or on PATH. */
async function checkMcpCommand(command: string): Promise<VerifyCheck> {
  if (command.length === 0) {
    return {
      name: "mcp-command",
      status: "FAIL",
      message: "mind-nerve MCP entry has no command",
      hint: "re-run install to rewrite the entry",
    };
  }
  if (path.isAbsolute(command)) {
    try {
      await fs.access(command, fs.constants.X_OK);
      return {
        name: "mcp-command",
        status: "PASS",
        message: `MCP command resolves: ${command}`,
        hint: null,
      };
    } catch {
      return {
        name: "mcp-command",
        status: "FAIL",
        message: `MCP command not executable: ${command} (stale venv path?)`,
        hint: "re-run install, or reinstall with `--mcp-launcher uvx` so no pre-built venv is needed",
      };
    }
  }
  try {
    const found = await which(command);
    return {
      name: "mcp-command",
      status: "PASS",
      message: `MCP command resolves on PATH: ${command} -> ${found}`,
      hint: null,
    };
  } catch {
    return {
      name: "mcp-command",
      status: "FAIL",
      message: `MCP command not found on PATH: ${command}`,
      hint:
        command === "uvx"
          ? `install uv (provides uvx), or reinstall with the default venv launcher; uvx resolves the published ${UVX_ENTRYPOINT} package on first launch`
          : "re-run install, or install the missing binary",
    };
  }
}

// ---------------------------------------------------------------------------
// Daemon socket (WARN-only)
// ---------------------------------------------------------------------------

const DAEMON_HINT =
  "the daemon is allowed to be on-demand — start it with `mind-nerve-routed-ensure` " +
  "(idempotent, also used by the SessionStart hook) or `mind-nerve-routed &`";

async function probeDaemonSocket(
  socketPath: string,
  timeoutMs: number,
): Promise<VerifyCheck> {
  let isSocket = false;
  try {
    isSocket = (await fs.stat(socketPath)).isSocket();
  } catch {
    return {
      name: "daemon-socket",
      status: "WARN",
      message: `daemon not running (no socket at ${socketPath})`,
      hint: DAEMON_HINT,
    };
  }
  if (!isSocket) {
    return {
      name: "daemon-socket",
      status: "WARN",
      message: `${socketPath} exists but is not a socket (stale file?)`,
      hint: `remove it and let the daemon rebind: ${DAEMON_HINT}`,
    };
  }

  const answered = await new Promise<boolean>((resolve) => {
    const conn = net.createConnection({ path: socketPath });
    const timer = setTimeout(() => {
      conn.destroy();
      resolve(false);
    }, timeoutMs);
    conn.on("connect", () => {
      clearTimeout(timer);
      conn.end();
      resolve(true);
    });
    conn.on("error", () => {
      clearTimeout(timer);
      resolve(false);
    });
  });

  if (!answered) {
    return {
      name: "daemon-socket",
      status: "WARN",
      message: `daemon socket present but not answering: ${socketPath}`,
      hint: DAEMON_HINT,
    };
  }
  return {
    name: "daemon-socket",
    status: "PASS",
    message: `daemon socket answers: ${socketPath}`,
    hint: null,
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function resolvePath(p: string, ws: string): string {
  if (path.isAbsolute(p)) return p;
  return path.join(ws, p);
}

async function pathExists(p: string): Promise<boolean> {
  try {
    await fs.access(p);
    return true;
  } catch {
    return false;
  }
}
