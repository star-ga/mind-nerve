// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.
//
// Part 1 (STRUCTURAL): replace a whole-hub skills symlink with a REAL directory
// containing only a small router skill.
//
// Measured cost of the whole-hub symlink: 461,782 chars ~= 115k tokens announced
// into every session. The router-only directory announces ~2k. The hub itself is
// never touched and stays reachable by absolute path.
//
// SAFETY — the single most destructive thing this installer could do is delete a
// user's skills hub. The rules are therefore absolute:
//   * a SYMLINK is unlinked (rm -f) — the target is untouched;
//   * a REAL DIRECTORY is never removed, only RENAMED aside to a timestamped
//     backup, so nothing is ever unrecoverable;
//   * the previous state is recorded so `uninstall` can restore it exactly,
//     including re-creating the hub symlink.

import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { InstallerError } from "./errors.js";

/** Name of the router skill projected into every managed skills dir. */
export const ROUTER_SKILL_NAME = "mind-nerve-router";

/** What is currently sitting at a CLI's skills path. */
export type SkillsDirKind = "absent" | "symlink" | "realdir" | "file";

export interface SkillsDirState {
  readonly kind: SkillsDirKind;
  /** Resolved link target when kind === "symlink", else null. */
  readonly symlinkTarget: string | null;
  /** Number of entries when kind === "realdir", else 0. */
  readonly entryCount: number;
  /** True when the directory looks like one we manage (has the router skill). */
  readonly looksManaged: boolean;
}

/** Persisted record of what was there before we touched it. */
export interface PreviousSkillsDir {
  readonly kind: SkillsDirKind;
  /** Link target to re-create on uninstall (kind === "symlink"). */
  readonly symlinkTarget: string | null;
  /** Where the original real directory was moved to (kind === "realdir"). */
  readonly backupPath: string | null;
}

export interface ClientState {
  readonly client: string;
  readonly skillsDir: string;
  readonly previous: PreviousSkillsDir;
  /** Unix millis of the install that produced this record. */
  readonly installedAt: number;
}

export interface ConvertResult {
  readonly skillsDir: string;
  readonly before: SkillsDirState;
  readonly backupPath: string | null;
  readonly changed: boolean;
}

export interface RestoreResult {
  readonly skillsDir: string;
  readonly restoredKind: SkillsDirKind | null;
  readonly changed: boolean;
}

/** Root of the installer's own state, outside any CLI's directory. */
export function stateRoot(homeDir: string = os.homedir()): string {
  return path.join(homeDir, ".mind-nerve", "state");
}

/** Path of the per-client state record. */
export function stateFilePath(
  clientName: string,
  homeDir: string = os.homedir(),
): string {
  return path.join(stateRoot(homeDir), `${clientName}.json`);
}

// ---------------------------------------------------------------------------
// Inspection
// ---------------------------------------------------------------------------

/**
 * Reports what is at `skillsDir` WITHOUT following the symlink for the kind
 * decision — `lstat`, not `stat`. Distinguishing a symlink from a real dir is
 * the whole point: one is safe to unlink, the other must never be removed.
 */
export async function inspectSkillsDir(
  skillsDir: string,
): Promise<SkillsDirState> {
  let st;
  try {
    st = await fs.lstat(skillsDir);
  } catch {
    return {
      kind: "absent",
      symlinkTarget: null,
      entryCount: 0,
      looksManaged: false,
    };
  }

  if (st.isSymbolicLink()) {
    let target: string | null = null;
    try {
      target = await fs.readlink(skillsDir);
    } catch {
      // Dangling or unreadable link — still a symlink, target unknown.
    }
    return {
      kind: "symlink",
      symlinkTarget: target,
      entryCount: 0,
      looksManaged: false,
    };
  }

  if (!st.isDirectory()) {
    return {
      kind: "file",
      symlinkTarget: null,
      entryCount: 0,
      looksManaged: false,
    };
  }

  let entries: string[] = [];
  try {
    entries = await fs.readdir(skillsDir);
  } catch {
    // Unreadable directory — treat as an opaque real dir, i.e. back it up.
  }
  return {
    kind: "realdir",
    symlinkTarget: null,
    entryCount: entries.length,
    looksManaged: entries.includes(ROUTER_SKILL_NAME),
  };
}

// ---------------------------------------------------------------------------
// Conversion
// ---------------------------------------------------------------------------

export interface ConvertOptions {
  /** Client name, used for the state record and error attribution. */
  readonly clientName: string;
  /** Absolute path to the hub, recorded in the generated README. */
  readonly hubDir: string;
  /** Body of the router SKILL.md to install. */
  readonly routerBody: string;
  /** Home directory override (tests). */
  readonly homeDir?: string;
  /** Clock override (tests) — supplies the backup timestamp. */
  readonly now?: number;
}

/**
 * Converts a CLI's skills path into a real, router-only directory.
 *
 * Idempotent: re-running on a directory we already manage refreshes the router
 * body and returns `changed: false` when nothing differed. The pre-existing
 * state is only recorded on the FIRST conversion, so a second run can never
 * overwrite the record of what the user originally had.
 */
export async function convertToRouterDir(
  skillsDir: string,
  opts: ConvertOptions,
): Promise<ConvertResult> {
  const before = await inspectSkillsDir(skillsDir);
  const now = opts.now ?? Date.now();
  const homeDir = opts.homeDir ?? os.homedir();

  const existingState = await readClientState(opts.clientName, homeDir);
  let backupPath: string | null = null;
  let displaced = false;

  if (before.kind === "symlink") {
    // A symlink may be `rm -f`'d: the hub it points at is untouched.
    try {
      await fs.unlink(skillsDir);
      displaced = true;
    } catch (err) {
      throw new InstallerError(
        "PROJECTION_FAILED",
        opts.clientName,
        `Failed to unlink skills symlink ${skillsDir}: ${String(err)}`,
      );
    }
  } else if (before.kind === "realdir" && !before.looksManaged) {
    // NEVER rm -rf a real directory. Rename it aside — always recoverable.
    backupPath = `${skillsDir}.bak-mind-nerve-${now}`;
    try {
      await fs.rename(skillsDir, backupPath);
      displaced = true;
    } catch (err) {
      throw new InstallerError(
        "BACKUP_FAILED",
        opts.clientName,
        `Refusing to replace real directory ${skillsDir}: could not move it to ` +
          `${backupPath}: ${String(err)}`,
      );
    }
  } else if (before.kind === "file") {
    backupPath = `${skillsDir}.bak-mind-nerve-${now}`;
    try {
      await fs.rename(skillsDir, backupPath);
      displaced = true;
    } catch (err) {
      throw new InstallerError(
        "BACKUP_FAILED",
        opts.clientName,
        `Could not move file ${skillsDir} aside to ${backupPath}: ${String(err)}`,
      );
    }
  }

  const wrote = await writeRouterDir(skillsDir, opts);

  // Record the ORIGINAL state once and only once.
  if (existingState === null) {
    await writeClientState(
      {
        client: opts.clientName,
        skillsDir,
        previous: {
          kind: before.kind,
          symlinkTarget: before.symlinkTarget,
          backupPath,
        },
        installedAt: now,
      },
      homeDir,
    );
  }

  return {
    skillsDir,
    before,
    backupPath,
    changed: displaced || wrote,
  };
}

/**
 * Writes the router-only contents. Returns true if anything changed on disk.
 * Kept separate so `install` can refresh the router body without displacing
 * anything.
 */
export async function writeRouterDir(
  skillsDir: string,
  opts: Pick<ConvertOptions, "clientName" | "hubDir" | "routerBody">,
): Promise<boolean> {
  const routerDir = path.join(skillsDir, ROUTER_SKILL_NAME);
  const routerFile = path.join(routerDir, "SKILL.md");
  const readmeFile = path.join(skillsDir, "README.md");
  const readmeBody = buildSkillsReadme(opts.hubDir);

  try {
    await fs.mkdir(routerDir, { recursive: true });
  } catch (err) {
    throw new InstallerError(
      "PROJECTION_FAILED",
      opts.clientName,
      `Failed to create router dir ${routerDir}: ${String(err)}`,
    );
  }

  const routerChanged = await writeIfDifferent(routerFile, opts.routerBody);
  const readmeChanged = await writeIfDifferent(readmeFile, readmeBody);
  return routerChanged || readmeChanged;
}

/** README dropped next to the router so the layout explains itself. */
export function buildSkillsReadme(hubDir: string): string {
  return [
    "# mind-nerve managed skills directory",
    "",
    "This directory intentionally contains only the `mind-nerve-router` skill.",
    "",
    `The full skills hub lives at \`${hubDir}\` and is deliberately NOT announced:`,
    "announcing it costs ~115k tokens per session. mind-nerve routes on demand and",
    "projects only the relevant skills here, per prompt.",
    "",
    "Do not replace this directory with a symlink to the hub.",
    "",
    "To undo: `mind-nerve-installer uninstall <client>`.",
    "",
  ].join("\n");
}

// ---------------------------------------------------------------------------
// Restoration
// ---------------------------------------------------------------------------

/**
 * Restores whatever was at the skills path before install, using the recorded
 * state. Re-creates the hub symlink when that is what was there.
 *
 * The managed directory we created IS safe to remove — but only after we have
 * confirmed from the state record that we are the ones who created it.
 */
export async function restoreSkillsDir(
  clientName: string,
  opts: { readonly homeDir?: string } = {},
): Promise<RestoreResult> {
  const homeDir = opts.homeDir ?? os.homedir();
  const state = await readClientState(clientName, homeDir);
  if (state === null) {
    return { skillsDir: "", restoredKind: null, changed: false };
  }

  const { skillsDir, previous } = state;
  const current = await inspectSkillsDir(skillsDir);

  // Only remove the current path if it is ours (managed) or a symlink.
  if (current.kind === "symlink") {
    await fs.unlink(skillsDir).catch(() => undefined);
  } else if (current.kind === "realdir" && current.looksManaged) {
    await fs.rm(skillsDir, { recursive: true, force: true }).catch(() => undefined);
  } else if (current.kind === "realdir") {
    // Someone replaced our dir with real content. Do not touch it.
    throw new InstallerError(
      "RESTORE_FAILED",
      clientName,
      `${skillsDir} is a real directory that mind-nerve does not manage — ` +
        `refusing to remove it. Move it aside manually, then re-run uninstall.`,
    );
  }

  let restoredKind: SkillsDirKind | null = null;
  if (previous.kind === "symlink" && previous.symlinkTarget !== null) {
    await fs.mkdir(path.dirname(skillsDir), { recursive: true });
    await fs.symlink(previous.symlinkTarget, skillsDir);
    restoredKind = "symlink";
  } else if (previous.kind === "realdir" && previous.backupPath !== null) {
    await fs.rename(previous.backupPath, skillsDir);
    restoredKind = "realdir";
  } else if (previous.kind === "file" && previous.backupPath !== null) {
    await fs.rename(previous.backupPath, skillsDir);
    restoredKind = "file";
  } else {
    // Nothing was there before — leaving it absent IS the exact restore.
    restoredKind = "absent";
  }

  await fs.rm(stateFilePath(clientName, homeDir), { force: true }).catch(
    () => undefined,
  );

  return { skillsDir, restoredKind, changed: true };
}

// ---------------------------------------------------------------------------
// State persistence
// ---------------------------------------------------------------------------

export async function readClientState(
  clientName: string,
  homeDir: string = os.homedir(),
): Promise<ClientState | null> {
  try {
    const raw = await fs.readFile(stateFilePath(clientName, homeDir), "utf8");
    const parsed: unknown = JSON.parse(raw);
    if (
      typeof parsed === "object" &&
      parsed !== null &&
      "skillsDir" in parsed &&
      "previous" in parsed
    ) {
      return parsed as ClientState;
    }
    return null;
  } catch {
    return null;
  }
}

export async function writeClientState(
  state: ClientState,
  homeDir: string = os.homedir(),
): Promise<void> {
  const file = stateFilePath(state.client, homeDir);
  await fs.mkdir(path.dirname(file), { recursive: true });
  await fs.writeFile(file, JSON.stringify(state, null, 2) + "\n", "utf8");
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Writes only when the content differs — keeps `changed` honest. */
async function writeIfDifferent(
  filePath: string,
  content: string,
): Promise<boolean> {
  try {
    const existing = await fs.readFile(filePath, "utf8");
    if (existing === content) return false;
  } catch {
    // Missing — write it.
  }
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, content, "utf8");
  return true;
}
