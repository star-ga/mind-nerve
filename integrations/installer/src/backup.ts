// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.

import fs from "node:fs/promises";
import path from "node:path";
import { InstallerError } from "./errors.js";

/** Returns the backup path for a given source path at a given unix timestamp. */
export function backupPath(sourcePath: string, tsMs: number): string {
  return `${sourcePath}.bak-mind-nerve-${tsMs}`;
}

/**
 * Creates a timestamped backup of sourcePath. Returns the backup path.
 * No-ops if sourcePath does not exist (file may be created fresh).
 * Throws InstallerError(BACKUP_FAILED) on copy failure.
 */
export async function createBackup(
  sourcePath: string,
  clientName: string,
): Promise<string | null> {
  try {
    await fs.access(sourcePath);
  } catch {
    // Source does not exist — nothing to back up.
    return null;
  }

  const bak = backupPath(sourcePath, Date.now());
  try {
    await fs.copyFile(sourcePath, bak);
    return bak;
  } catch (err) {
    throw new InstallerError(
      "BACKUP_FAILED",
      clientName,
      `Failed to create backup at ${bak}: ${String(err)}`,
    );
  }
}

/**
 * Restores the most recent backup for sourcePath.
 * Searches for files matching `<sourcePath>.bak-mind-nerve-*` in the same
 * directory, picks the highest timestamp, and copies it over sourcePath.
 *
 * If no backup is found, returns false. If the restore succeeds, returns true.
 * Throws InstallerError(RESTORE_FAILED) on copy failure.
 */
export async function restoreLatestBackup(
  sourcePath: string,
  clientName: string,
): Promise<boolean> {
  const dir = path.dirname(sourcePath);
  const base = path.basename(sourcePath);
  const prefix = `${base}.bak-mind-nerve-`;

  let entries: string[];
  try {
    const raw = await fs.readdir(dir);
    entries = raw.filter((f) => f.startsWith(prefix));
  } catch {
    return false;
  }

  if (entries.length === 0) return false;

  // Parse timestamps and pick the latest.
  const sorted = entries
    .map((f) => {
      const ts = parseInt(f.slice(prefix.length), 10);
      return { file: f, ts };
    })
    .filter((e) => !isNaN(e.ts))
    .sort((a, b) => b.ts - a.ts);

  const best = sorted[0];
  if (best === undefined) return false;

  const bakPath = path.join(dir, best.file);
  try {
    await fs.copyFile(bakPath, sourcePath);
    return true;
  } catch (err) {
    throw new InstallerError(
      "RESTORE_FAILED",
      clientName,
      `Failed to restore backup ${bakPath} to ${sourcePath}: ${String(err)}`,
    );
  }
}

/**
 * Lists all backup files for a given sourcePath in descending timestamp order.
 */
export async function listBackups(sourcePath: string): Promise<string[]> {
  const dir = path.dirname(sourcePath);
  const base = path.basename(sourcePath);
  const prefix = `${base}.bak-mind-nerve-`;

  try {
    const raw = await fs.readdir(dir);
    return raw
      .filter((f) => f.startsWith(prefix))
      .map((f) => {
        const ts = parseInt(f.slice(prefix.length), 10);
        return { file: path.join(dir, f), ts };
      })
      .filter((e) => !isNaN(e.ts))
      .sort((a, b) => b.ts - a.ts)
      .map((e) => e.file);
  } catch {
    return [];
  }
}
