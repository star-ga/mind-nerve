// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.
//
// Ties Part 1 (structural router-only skills dir) and Part 2 (automatic hook)
// together for one CLI. This is what makes `install` self-wire a detected CLI:
//
//   1. copy the ONE shared CLI-agnostic hook to ~/.mind-nerve/bin/
//   2. write a per-CLI sh wrapper that exports this CLI's env and execs it
//   3. register the wrapper on UserPromptSubmit + SessionStart (backed up)
//   4. replace the whole-hub skills symlink with a router-only real directory
//
// Every step is idempotent and every mutated config is backed up first.

import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { type AgentSpec, type SkillSurface } from "./registry.js";
import { createBackup } from "./backup.js";
import {
  mergeJsonHooks,
  removeJsonHooks,
  mergeTomlHooks,
  removeTomlHooks,
  buildHookEnv,
  buildHookWrapper,
  isJsonHookFmt,
  assertKnownHookFmt,
} from "./hook_wiring.js";
import { convertToRouterDir, restoreSkillsDir } from "./skills_dir.js";
import { InstallerError } from "./errors.js";

/** Defaults chosen from measurement, not taste — see WireOptions docs. */
export const DEFAULT_TOP_K = 8;
/**
 * Measured on the live daemon: for "prove an exact minimum move floor for an
 * ARC-AGI-3 level" the correct hit scored 0.437 while ranks 2-8 scored
 * 0.267-0.282 and were all unrelated (GraphQL depth-limit attacks, SOC tabletop
 * exercises, ...). 7 of 8 injected rows were noise and all 7 got projected.
 * ~0.40+ is a real hit, ~0.30 is noise, 0.74 was a clean hit for
 * "deploy a site to cloudflare pages". 0.35 sits in the gap.
 */
export const DEFAULT_MIN_SCORE = 0.35;
export const DEFAULT_SOCKET_TIMEOUT = 2.0;
export const DEFAULT_CORE_SKILLS: readonly string[] = ["mind-nerve-router"];

export interface WireOptions {
  /** Absolute path to the shared hook implementation in this repo. */
  readonly hookSourcePath: string;
  /** Absolute path to the router SKILL.md body to install. */
  readonly routerSkillPath: string;
  /** Absolute path to the skills hub. */
  readonly hubDir: string;
  /** UNIX socket of mind-nerve-routed. */
  readonly socketPath: string;
  /** Directories holding agent .md definitions. */
  readonly agentDirs: readonly string[];
  readonly topK?: number;
  readonly minScore?: number;
  readonly socketTimeout?: number;
  readonly coreSkills?: readonly string[];
  /** Home directory override (tests). */
  readonly homeDir?: string;
  /** Clock override (tests). */
  readonly now?: number;
  /** Suppress config backups. Tests only. */
  readonly _skipBackup?: boolean;
}

export interface WireResult {
  readonly clientName: string;
  readonly hookScriptPath: string;
  readonly hookConfigPath: string;
  readonly skillsDir: string;
  readonly skillsBackupPath: string | null;
  readonly backedUp: readonly string[];
  readonly hookRegistered: boolean;
  readonly skillsConverted: boolean;
  readonly changed: boolean;
}

/** Where the shared hook lands, so every CLI wrapper execs ONE implementation. */
export function sharedHookPath(homeDir: string = os.homedir()): string {
  return path.join(homeDir, ".mind-nerve", "bin", "mind-nerve-hook");
}

/** Default JSONL log path for the hook. */
export function hookLogPath(
  clientName: string,
  homeDir: string = os.homedir(),
): string {
  return path.join(homeDir, ".mind-nerve", "logs", `${clientName}-hook.log`);
}

// ---------------------------------------------------------------------------
// Install
// ---------------------------------------------------------------------------

/**
 * Wires one CLI. No-op (changed: false) when everything is already in place.
 */
export async function wireClient(
  spec: AgentSpec,
  opts: WireOptions,
): Promise<WireResult | null> {
  const surface = spec.skillSurface;
  if (surface === null) return null;
  assertKnownHookFmt(surface, spec.name);

  const homeDir = opts.homeDir ?? os.homedir();
  const backedUp: string[] = [];

  // -- 1. Shared hook implementation ----------------------------------------
  const shared = sharedHookPath(homeDir);
  await fs.mkdir(path.dirname(shared), { recursive: true });
  const hookBody = await fs.readFile(opts.hookSourcePath, "utf8");
  await fs.writeFile(shared, hookBody, { mode: 0o755 });
  await fs.chmod(shared, 0o755);

  // -- 2. Per-CLI wrapper ----------------------------------------------------
  const env = buildHookEnv({
    surface,
    hubDir: opts.hubDir,
    socketPath: opts.socketPath,
    agentDirs: opts.agentDirs,
    topK: opts.topK ?? DEFAULT_TOP_K,
    minScore: opts.minScore ?? DEFAULT_MIN_SCORE,
    coreSkills: opts.coreSkills ?? DEFAULT_CORE_SKILLS,
    socketTimeout: opts.socketTimeout ?? DEFAULT_SOCKET_TIMEOUT,
    logPath: hookLogPath(spec.name, homeDir),
  });
  const wrapper = buildHookWrapper(shared, env);
  await fs.mkdir(path.dirname(surface.hookScriptPath), { recursive: true });
  const wrapperChanged = await writeIfDifferent(surface.hookScriptPath, wrapper);
  await fs.chmod(surface.hookScriptPath, 0o755);

  // -- 3. Register the hook in the CLI's config ------------------------------
  const hookRegistered = await registerHook(surface, spec.name, {
    skipBackup: opts._skipBackup === true,
    onBackup: (p) => backedUp.push(p),
  });

  // -- 4. Structural skills directory ---------------------------------------
  const routerBody = await fs.readFile(opts.routerSkillPath, "utf8");
  const convert = await convertToRouterDir(surface.skillsDir, {
    clientName: spec.name,
    hubDir: opts.hubDir,
    routerBody,
    homeDir,
    ...(opts.now !== undefined ? { now: opts.now } : {}),
  });

  return {
    clientName: spec.name,
    hookScriptPath: surface.hookScriptPath,
    hookConfigPath: surface.hookConfigPath,
    skillsDir: surface.skillsDir,
    skillsBackupPath: convert.backupPath,
    backedUp,
    hookRegistered,
    skillsConverted: convert.changed,
    changed: wrapperChanged || hookRegistered || convert.changed,
  };
}

async function registerHook(
  surface: SkillSurface,
  clientName: string,
  io: {
    readonly skipBackup: boolean;
    readonly onBackup: (p: string) => void;
  },
): Promise<boolean> {
  if (isJsonHookFmt(surface)) {
    let existing: Record<string, unknown> = {};
    try {
      const raw = await fs.readFile(surface.hookConfigPath, "utf8");
      const parsed: unknown = JSON.parse(raw);
      if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
        existing = parsed as Record<string, unknown>;
      } else {
        throw new InstallerError(
          "INVALID_CONFIG_FORMAT",
          clientName,
          `${surface.hookConfigPath} is not a JSON object — refusing to overwrite`,
        );
      }
    } catch (err) {
      if (err instanceof InstallerError) throw err;
      // Missing or empty file — start fresh.
    }

    const { updated, changed } = mergeJsonHooks(existing, surface);
    if (!changed) return false;
    if (!io.skipBackup) {
      const bak = await createBackup(surface.hookConfigPath, clientName);
      if (bak !== null) io.onBackup(bak);
    }
    await fs.mkdir(path.dirname(surface.hookConfigPath), { recursive: true });
    await fs.writeFile(
      surface.hookConfigPath,
      JSON.stringify(updated, null, 2) + "\n",
      "utf8",
    );
    return true;
  }

  // TOML — edited as text so comments and huge generated blocks survive.
  let text = "";
  try {
    text = await fs.readFile(surface.hookConfigPath, "utf8");
  } catch {
    // Missing file — will be created.
  }
  const { updated, changed } = mergeTomlHooks(text, surface);
  if (!changed) return false;
  if (!io.skipBackup) {
    const bak = await createBackup(surface.hookConfigPath, clientName);
    if (bak !== null) io.onBackup(bak);
  }
  await fs.mkdir(path.dirname(surface.hookConfigPath), { recursive: true });
  await fs.writeFile(surface.hookConfigPath, updated, "utf8");
  return true;
}

// ---------------------------------------------------------------------------
// Uninstall
// ---------------------------------------------------------------------------

export interface UnwireResult {
  readonly clientName: string;
  readonly hookUnregistered: boolean;
  readonly wrapperRemoved: boolean;
  readonly skillsRestored: boolean;
  readonly restoredKind: string | null;
  readonly changed: boolean;
}

/** Reverses wireClient: unregisters the hook, drops the wrapper, restores skills. */
export async function unwireClient(
  spec: AgentSpec,
  opts: { readonly homeDir?: string } = {},
): Promise<UnwireResult | null> {
  const surface = spec.skillSurface;
  if (surface === null) return null;

  const homeDir = opts.homeDir ?? os.homedir();
  let hookUnregistered = false;

  // -- Unregister from config -----------------------------------------------
  if (isJsonHookFmt(surface)) {
    try {
      const raw = await fs.readFile(surface.hookConfigPath, "utf8");
      const parsed: unknown = JSON.parse(raw);
      if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
        const { updated, changed } = removeJsonHooks(
          parsed as Record<string, unknown>,
          surface,
        );
        if (changed) {
          await fs.writeFile(
            surface.hookConfigPath,
            JSON.stringify(updated, null, 2) + "\n",
            "utf8",
          );
          hookUnregistered = true;
        }
      }
    } catch {
      // No config / unreadable — nothing to unregister.
    }
  } else {
    try {
      const text = await fs.readFile(surface.hookConfigPath, "utf8");
      const { updated, changed } = removeTomlHooks(text);
      if (changed) {
        await fs.writeFile(surface.hookConfigPath, updated, "utf8");
        hookUnregistered = true;
      }
    } catch {
      // No config — nothing to unregister.
    }
  }

  // -- Remove the wrapper ---------------------------------------------------
  let wrapperRemoved = false;
  try {
    await fs.unlink(surface.hookScriptPath);
    wrapperRemoved = true;
  } catch {
    // Already gone.
  }

  // -- Restore the skills directory -----------------------------------------
  const restore = await restoreSkillsDir(spec.name, { homeDir });

  return {
    clientName: spec.name,
    hookUnregistered,
    wrapperRemoved,
    skillsRestored: restore.changed,
    restoredKind: restore.restoredKind,
    changed: hookUnregistered || wrapperRemoved || restore.changed,
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function writeIfDifferent(
  filePath: string,
  content: string,
): Promise<boolean> {
  try {
    if ((await fs.readFile(filePath, "utf8")) === content) return false;
  } catch {
    // Missing — write it.
  }
  await fs.writeFile(filePath, content, { mode: 0o755 });
  return true;
}
