// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.

import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { InstallerError } from "./errors.js";

/**
 * Ensures the per-CLI projection directory exists under ~/.mind-nerve/projections/<clientName>/.
 *
 * The projection dir is a directory that the CLI's skill lookup is redirected to.
 * For Phase 1 it is created empty; the runtime hook populates it per-turn.
 */
export async function ensureProjectionDir(
  projectionDir: string,
  clientName: string,
): Promise<void> {
  try {
    await fs.mkdir(projectionDir, { recursive: true });
  } catch (err) {
    throw new InstallerError(
      "PROJECTION_FAILED",
      clientName,
      `Failed to create projection dir ${projectionDir}: ${String(err)}`,
    );
  }
}

/**
 * Atomically rewrites a projection directory to contain a set of symlinks.
 *
 * Used by the Claude Code hook and extended here for per-CLI projection dirs.
 * Mirror of integrations/claude-code/projector.ts rewriteProjection.
 *
 * @param projectedDir  Target projection directory.
 * @param sources       Map of linkName → target path to symlink inside projectedDir.
 */
export async function rewriteProjectionLinks(
  projectedDir: string,
  sources: ReadonlyMap<string, string>,
  clientName: string,
): Promise<void> {
  const parentDir = path.dirname(projectedDir);
  const suffix = crypto.randomBytes(6).toString("hex");
  const stagingDir = path.join(parentDir, `proj.tmp.${suffix}`);

  try {
    await fs.mkdir(stagingDir, { recursive: true });
    for (const [linkName, target] of sources) {
      await fs.symlink(target, path.join(stagingDir, linkName));
    }

    const oldBackupDir = path.join(parentDir, `proj.old.${suffix}`);
    let existingPresent = false;
    try {
      await fs.access(projectedDir);
      existingPresent = true;
    } catch {
      // projectedDir does not exist yet — first run.
    }

    if (existingPresent) {
      await fs.rename(projectedDir, oldBackupDir);
    }
    await fs.rename(stagingDir, projectedDir);

    if (existingPresent) {
      await removeDirBestEffort(oldBackupDir);
    }
  } catch (err) {
    // Best-effort staging cleanup.
    await removeDirBestEffort(stagingDir);
    if (err instanceof InstallerError) throw err;
    throw new InstallerError(
      "PROJECTION_FAILED",
      clientName,
      `Atomic projection rewrite failed for ${projectedDir}: ${String(err)}`,
    );
  }
}

async function removeDirBestEffort(dirPath: string): Promise<void> {
  try {
    await fs.rm(dirPath, { recursive: true, force: true });
  } catch {
    // Best-effort — stale tmp dirs are harmless.
  }
}
