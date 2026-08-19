// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.

import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { type AgentSpec } from "./registry.js";
import { createBackup } from "./backup.js";
import { ensureProjectionDir } from "./projector.js";
import {
  buildMcpSpec,
  mergeJsonMcp,
  mergeTomlMcp,
  isJsonMcpFmt,
  isTomlMcpFmt,
  type McpLauncher,
} from "./mcp_rewire.js";
import { appendInstructionBlock, BLOCK_MARKER, upsertInstructionJson } from "./instruction_block.js";
import { wireClient, type WireOptions, type WireResult } from "./wire.js";
import { InstallerError } from "./errors.js";

export interface InstallOptions {
  /** Absolute path to the mind-nerve binary. */
  mindNerveBin: string;
  /**
   * Workspace directory for workspace-relative config paths.
   * Defaults to process.cwd().
   */
  workspace?: string;
  /** MCP-only mode: perform ONLY the MCP rewire (skip projection dir,
   * instruction block, and skill-surface wiring). */
  mcpOnly?: boolean;
  /**
   * How MCP server entries are launched. "venv" (default) pins the absolute
   * venv binary path; "uvx" launches via `uvx --from mind-nerve
   * mind-nerve-mcp` so the host needs no pre-built venv.
   */
  mcpLauncher?: McpLauncher;
  /**
   * STARGA power-user shared projection: use this dir instead of per-CLI
   * projection. Only relevant for clients with projectionDir != null.
   */
  sharedProjectionDir?: string;
  /**
   * When present, also performs skill-surface wiring for clients that have one:
   * the router-only skills directory (Part 1) and the automatic hook (Part 2).
   * Omit to install the MCP surface only.
   */
  wire?: WireOptions;
  /** Suppress backup creation. Used in tests only. */
  _skipBackup?: boolean;
}

export interface InstallResult {
  readonly clientName: string;
  readonly configPath: string | null;
  readonly mcpPath: string | null;
  readonly projectionDir: string | null;
  readonly backedUp: readonly string[];
  readonly changed: boolean;
  /** true if a previous install was detected and no changes were made. */
  readonly idempotentNoop: boolean;
  /** Skill-surface wiring result, or null when not requested / not applicable. */
  readonly wire: WireResult | null;
}

/**
 * True when *dir* holds a populated runtime: manifest.json or
 * route_table.jsonl — the same condition Python's _default_user_runtime_dir
 * prefers the STARGA dash dir on.
 */
export async function runtimeDirPopulated(dir: string): Promise<boolean> {
  for (const marker of ["manifest.json", "route_table.jsonl"]) {
    try {
      await fs.access(path.join(dir, marker));
      return true;
    } catch {
      // marker absent — try the next
    }
  }
  return false;
}

/**
 * Runtime dir holding route_table.jsonl + the row-aligned route_table.npy.
 * Returns the dir only when it is POPULATED on this host — an empty or
 * merely-existing dir must not be pinned: the pin would shadow the MCP's
 * Hugging Face auto-seed with nothing, while Python acquire/inference only
 * prefer the dir when manifest.json is present (fleet audit 2026-08).
 * Mirrors runtimeDir() in index.ts. Fleet incident 2026-07-10: without the
 * pin the MCP silently loads the OSS catalog instead of the local route
 * table.
 */
export async function existingRuntimeDir(): Promise<string | null> {
  const dir =
    process.env.MIND_NERVE_RUNTIME_DIR ??
    path.join(os.homedir(), ".local", "share", "mind-nerve-runtime");
  return (await runtimeDirPopulated(dir)) ? dir : null;
}

/**
 * Runs the full 4-step install for a single client:
 *   1. Detection: caller is responsible (call detectClient first).
 *   2. Projection dir setup (if the client has a skill surface).
 *   3. MCP rewire (if the client has an MCP surface).
 *   4. Instruction block injection (workspace-rules clients).
 *
 * Idempotent: re-running on an already-installed client is a no-op.
 */
export async function installClient(
  spec: AgentSpec,
  opts: InstallOptions,
): Promise<InstallResult> {
  const ws = opts.workspace ?? process.cwd();
  const backedUp: string[] = [];
  let changed = false;
  let idempotentNoop = true;

  // -------------------------------------------------------------------------
  // Step 2: Projection dir (skipped in MCP-only mode)
  // -------------------------------------------------------------------------
  const effectiveProjectionDir =
    opts.sharedProjectionDir ??
    spec.projectionDir;

  if (effectiveProjectionDir !== null && opts.mcpOnly !== true) {
    // ensureProjectionDir is idempotent (mkdir -p). Only mark changed if the
    // directory did not exist before.
    let projDirExisted = false;
    try {
      await fs.access(effectiveProjectionDir);
      projDirExisted = true;
    } catch {
      // will be created
    }
    await ensureProjectionDir(effectiveProjectionDir, spec.name);
    if (!projDirExisted) {
      changed = true;
      idempotentNoop = false;
    }
  }

  // -------------------------------------------------------------------------
  // Step 3: MCP rewire — ALWAYS runs when the client has an MCP surface.
  // (--mcp / mcpOnly used to suppress THIS step too, making "MCP-only mode"
  // a no-op; audit finding 2026-08 r4 / codex#14. MCP-only mode skips steps
  // 2, 4 and 5 — never this one.)
  // -------------------------------------------------------------------------
  let effectiveMcpPath: string | null = null;
  if (spec.mcpFmt !== null && spec.mcpPath !== null) {
    effectiveMcpPath = resolvePath(spec.mcpPath, ws);
    const srv = buildMcpSpec(
      opts.mindNerveBin,
      opts.mcpLauncher ?? "venv",
      await existingRuntimeDir(),
    );

    if (isJsonMcpFmt(spec.mcpFmt)) {
      let existing: Record<string, unknown> = {};
      try {
        const raw = await fs.readFile(effectiveMcpPath, "utf8");
        existing = JSON.parse(raw) as Record<string, unknown>;
      } catch {
        // File doesn't exist or is empty — start fresh.
      }

      const { updated, changed: didChange } = mergeJsonMcp(
        spec.mcpFmt,
        existing,
        srv,
        spec.name,
      );

      if (didChange) {
        if (!opts._skipBackup) {
          const bak = await createBackup(effectiveMcpPath, spec.name);
          if (bak !== null) backedUp.push(bak);
        }
        await writeJsonFile(effectiveMcpPath, updated);
        changed = true;
        idempotentNoop = false;
      }
    } else if (isTomlMcpFmt(spec.mcpFmt)) {
      let existingText = "";
      try {
        existingText = await fs.readFile(effectiveMcpPath, "utf8");
      } catch {
        // File doesn't exist — will be created.
      }

      const { updated, changed: didChange } = mergeTomlMcp(
        spec.mcpFmt,
        existingText,
        srv,
        spec.name,
      );

      if (didChange) {
        if (!opts._skipBackup) {
          const bak = await createBackup(effectiveMcpPath, spec.name);
          if (bak !== null) backedUp.push(bak);
        }
        await writeTextFile(effectiveMcpPath, updated);
        changed = true;
        idempotentNoop = false;
      }
    }
  }

  // MCP-only mode: skip step 4.
  if (opts.mcpOnly === true) {
    return {
      clientName: spec.name,
      configPath: null,
      mcpPath: effectiveMcpPath,
      projectionDir: effectiveProjectionDir,
      backedUp,
      changed,
      idempotentNoop: !changed,
      wire: null,
    };
  }

  // -------------------------------------------------------------------------
  // Step 4: Instruction block (workspace-rules clients)
  // -------------------------------------------------------------------------
  let effectiveConfigPath: string | null = null;
  if (spec.instructionFilePath !== null) {
    effectiveConfigPath = resolvePath(spec.instructionFilePath, ws);

    // The block must stay FORMAT-VALID for structured configs (codex#17):
    // cody's .cody/config.json takes a managed JSON key, aider's
    // .aider.conf.yml takes a comment block, plain rules files take prose.
    const didWrite =
      spec.configFmt === "json-generic"
        ? await upsertInstructionJson(effectiveConfigPath, effectiveProjectionDir)
        : await appendInstructionBlock(
            effectiveConfigPath,
            effectiveProjectionDir,
            spec.configFmt === "yaml-aider" ? "yaml-comment" : "prose",
          );
    if (didWrite) {
      changed = true;
      idempotentNoop = false;
    } else {
      // Already installed — this is a no-op.
    }
  }

  // For clients that have no instruction file but have their own JSON config
  // format (claude-code hooks, gemini, continue, zed, cody), we write a
  // managed marker into the config without an instruction block.
  if (spec.instructionFilePath === null && spec.configFmt !== "toml-codex" && spec.configFmt !== "toml-vibe") {
    const configPath = resolvePath(spec.configPath, ws);
    const alreadyManaged = await checkConfigManaged(configPath, spec.configFmt);
    if (!alreadyManaged) {
      changed = true;
      idempotentNoop = false;
    }
    effectiveConfigPath = configPath;
  }

  // -------------------------------------------------------------------------
  // Step 5: Skill-surface wiring — router-only skills dir + automatic hook.
  // -------------------------------------------------------------------------
  let wire: WireResult | null = null;
  if (opts.wire !== undefined && spec.skillSurface !== null) {
    const wireOpts: WireOptions =
      opts._skipBackup === true
        ? { ...opts.wire, _skipBackup: true }
        : opts.wire;
    wire = await wireClient(spec, wireOpts);
    if (wire !== null) {
      for (const b of wire.backedUp) backedUp.push(b);
      if (wire.changed) changed = true;
    }
  }

  return {
    clientName: spec.name,
    configPath: effectiveConfigPath,
    mcpPath: effectiveMcpPath,
    projectionDir: effectiveProjectionDir,
    backedUp,
    changed,
    idempotentNoop: !changed,
    wire,
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Resolves a path that may be workspace-relative (no leading '/') or absolute.
 */
function resolvePath(p: string, ws: string): string {
  if (path.isAbsolute(p)) return p;
  return path.join(ws, p);
}

async function writeJsonFile(filePath: string, data: Record<string, unknown>): Promise<void> {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, JSON.stringify(data, null, 2) + "\n", "utf8");
}

async function writeTextFile(filePath: string, content: string): Promise<void> {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, content, "utf8");
}

/**
 * Checks if a config file already contains a mind-nerve marker (idempotency).
 * gemini's MCP entry carries the marker as an env var instead of the
 * `_comment: "mind-nerve managed"` string every other client uses (its
 * settings.json schema has no room for an unknown top-level key).
 */
async function checkConfigManaged(
  configPath: string,
  _fmt: string,
): Promise<boolean> {
  try {
    const content = await fs.readFile(configPath, "utf8");
    return (
      content.includes(BLOCK_MARKER) ||
      content.includes("mind-nerve managed") ||
      content.includes("MIND_NERVE_MANAGED")
    );
  } catch {
    return false;
  }
}
