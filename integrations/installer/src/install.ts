// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.

import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { type AgentSpec } from "./registry.js";
import { createBackup } from "./backup.js";
import { ensureProjectionDir } from "./projector.js";
import {
  buildMcpSpec,
  mergeJsonMcp,
  mergeTomlMcp,
  isJsonMcpFmt,
  isTomlMcpFmt,
} from "./mcp_rewire.js";
import { appendInstructionBlock, BLOCK_MARKER } from "./instruction_block.js";
import { InstallerError } from "./errors.js";

export interface InstallOptions {
  /** Absolute path to the mind-nerve binary. */
  mindNerveBin: string;
  /**
   * Workspace directory for workspace-relative config paths.
   * Defaults to process.cwd().
   */
  workspace?: string;
  /** Skip MCP rewire even if the client supports MCP. */
  mcpOnly?: boolean;
  /**
   * STARGA power-user shared projection: use this dir instead of per-CLI
   * projection. Only relevant for clients with projectionDir != null.
   */
  sharedProjectionDir?: string;
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
  // Step 2: Projection dir
  // -------------------------------------------------------------------------
  const effectiveProjectionDir =
    opts.sharedProjectionDir ??
    spec.projectionDir;

  if (effectiveProjectionDir !== null) {
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
  // Step 3: MCP rewire
  // -------------------------------------------------------------------------
  let effectiveMcpPath: string | null = null;
  if (spec.mcpFmt !== null && spec.mcpPath !== null && !opts.mcpOnly) {
    effectiveMcpPath = resolvePath(spec.mcpPath, ws);
    const srv = buildMcpSpec(
      opts.mindNerveBin,
      upstreamConfigPath(spec.name),
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
    };
  }

  // -------------------------------------------------------------------------
  // Step 4: Instruction block (workspace-rules clients)
  // -------------------------------------------------------------------------
  let effectiveConfigPath: string | null = null;
  if (spec.instructionFilePath !== null) {
    effectiveConfigPath = resolvePath(spec.instructionFilePath, ws);

    const didWrite = await appendInstructionBlock(
      effectiveConfigPath,
      effectiveProjectionDir,
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

  return {
    clientName: spec.name,
    configPath: effectiveConfigPath,
    mcpPath: effectiveMcpPath,
    projectionDir: effectiveProjectionDir,
    backedUp,
    changed,
    idempotentNoop: !changed,
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

/**
 * Returns the path to the upstream MCP config TOML for this client.
 * These configs live in ~/.config/mind-nerve/mcp/<clientName>.toml.
 */
function upstreamConfigPath(clientName: string): string {
  return path.join(os.homedir(), ".config", "mind-nerve", "mcp", `${clientName}.toml`);
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
 */
async function checkConfigManaged(
  configPath: string,
  _fmt: string,
): Promise<boolean> {
  try {
    const content = await fs.readFile(configPath, "utf8");
    return content.includes(BLOCK_MARKER) || content.includes("mind-nerve managed");
  } catch {
    return false;
  }
}
