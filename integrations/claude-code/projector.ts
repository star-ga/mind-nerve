// mind-nerve Claude Code UserPromptSubmit hook. Projector pattern (D1).

import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import crypto from "node:crypto";
import { type SkillEntry } from "./catalog.js";

/**
 * Atomically rewrites the projection directory so that it contains symlinks
 * pointing only to the skill entries in `selected`.
 *
 * Atomicity guarantee:
 *   1. Write symlinks to a temporary staging directory (random suffix).
 *   2. Rename the staging directory over the target with rename(2).
 *      rename(2) is atomic on POSIX when src and dst are on the same
 *      filesystem — ~/.mind-nerve/ is always on the same FS as itself.
 *   3. Remove the old directory (now at tmp path after rename(2) replaced it).
 *
 * If any step fails, the old projection remains intact. Fail-open per D4.
 */
export async function rewriteProjection(
  projectedDir: string,
  selected: SkillEntry[],
): Promise<void> {
  const parentDir = path.dirname(projectedDir);
  const tmpSuffix = crypto.randomBytes(6).toString("hex");
  const stagingDir = path.join(
    parentDir,
    `skills-projected.tmp.${tmpSuffix}`,
  );

  await fs.mkdir(stagingDir, { recursive: true });

  // Populate staging with symlinks to selected skill directories.
  for (const entry of selected) {
    const targetDir = path.dirname(entry.skillPath); // ~/.claude/skills/<id>/
    const linkPath = path.join(stagingDir, entry.id);
    await fs.symlink(targetDir, linkPath);
  }

  // Atomically replace the projection dir.
  // On Linux, rename() over an existing directory fails if the target is
  // non-empty. We rename the existing dir out first (to another tmp name),
  // then rename staging in. Both renames are on the same FS.
  const oldBackup = path.join(parentDir, `skills-projected.old.${tmpSuffix}`);

  let existingDirPresent = false;
  try {
    await fs.access(projectedDir);
    existingDirPresent = true;
  } catch {
    // projectedDir does not exist yet — first run.
  }

  if (existingDirPresent) {
    await fs.rename(projectedDir, oldBackup);
  }

  await fs.rename(stagingDir, projectedDir);

  // Clean up the old directory after successful promotion.
  if (existingDirPresent) {
    await removeDir(oldBackup);
  }
}

/**
 * Rewrites the projection to contain ALL entries (passthrough mode).
 * Used when mind-nerve returns low_confidence / passthrough / is missing.
 */
export async function rewriteProjectionPassthrough(
  projectedDir: string,
  allEntries: SkillEntry[],
): Promise<void> {
  await rewriteProjection(projectedDir, allEntries);
}

async function removeDir(dirPath: string): Promise<void> {
  try {
    await fs.rm(dirPath, { recursive: true, force: true });
  } catch {
    // Best-effort cleanup — stale tmp dirs are harmless.
  }
}

/** Ensures the ~/.mind-nerve/ directory tree exists. */
export async function ensureRuntimeDir(): Promise<void> {
  const runtimeDir = path.join(os.homedir(), ".mind-nerve");
  await fs.mkdir(runtimeDir, { recursive: true });
}
