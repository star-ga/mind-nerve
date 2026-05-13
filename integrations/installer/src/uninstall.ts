// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.

import fs from "node:fs/promises";
import path from "node:path";
import { type AgentSpec } from "./registry.js";
import { restoreLatestBackup } from "./backup.js";
import { removeInstructionBlock } from "./instruction_block.js";
import { InstallerError } from "./errors.js";

export interface UninstallOptions {
  workspace?: string;
}

export interface UninstallResult {
  readonly clientName: string;
  readonly restoredPaths: readonly string[];
  readonly removedBlocks: readonly string[];
  readonly changed: boolean;
}

/**
 * Reverses the install for a single client:
 *   1. Restores the most recent .bak for each config file that was mutated.
 *   2. Removes mind-nerve instruction blocks from workspace-rules files.
 *   3. Removes the projection directory if present.
 *
 * Does not fail if the client was never installed — partial uninstall is safe.
 */
export async function uninstallClient(
  spec: AgentSpec,
  opts: UninstallOptions,
): Promise<UninstallResult> {
  const ws = opts.workspace ?? process.cwd();
  const restoredPaths: string[] = [];
  const removedBlocks: string[] = [];
  let changed = false;

  // -------------------------------------------------------------------------
  // Restore MCP config backup.
  // -------------------------------------------------------------------------
  if (spec.mcpPath !== null) {
    const mcpPath = resolvePath(spec.mcpPath, ws);
    try {
      const restored = await restoreLatestBackup(mcpPath, spec.name);
      if (restored) {
        restoredPaths.push(mcpPath);
        changed = true;
      }
    } catch (err) {
      if (err instanceof InstallerError) throw err;
      // Non-fatal: log the error in real usage; here re-throw for tests.
      throw err;
    }
  }

  // -------------------------------------------------------------------------
  // Remove instruction blocks from workspace-rules files.
  // -------------------------------------------------------------------------
  if (spec.instructionFilePath !== null) {
    const instrPath = resolvePath(spec.instructionFilePath, ws);
    try {
      const removed = await removeInstructionBlock(instrPath);
      if (removed) {
        removedBlocks.push(instrPath);
        changed = true;
      }
    } catch {
      // File doesn't exist — nothing to remove.
    }
  }

  // -------------------------------------------------------------------------
  // Remove projection directory.
  // -------------------------------------------------------------------------
  if (spec.projectionDir !== null) {
    try {
      await fs.rm(spec.projectionDir, { recursive: true, force: true });
      changed = true;
    } catch {
      // Best-effort — the directory may not exist.
    }
  }

  return {
    clientName: spec.name,
    restoredPaths,
    removedBlocks,
    changed,
  };
}

/**
 * Resolves a path that may be workspace-relative (no leading '/') or absolute.
 */
function resolvePath(p: string, ws: string): string {
  if (path.isAbsolute(p)) return p;
  return path.join(ws, p);
}
