// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.
//
// Part 2 (AUTOMATIC): register the generic mind-nerve hook on a CLI's
// UserPromptSubmit / SessionStart events.
//
// All six skill-surface CLIs speak the same hook protocol; only the container
// differs. JSON settings files are parsed/merged/serialised. TOML config files
// are edited as TEXT inside a marked block — codex's live config is 125 KB of
// commented `skills.config`, and round-tripping it through a TOML serialiser
// would silently destroy every comment. Appending an array-of-tables block at
// the end of a TOML file is valid regardless of what precedes it, because an
// `[[table]]` header resets the key context.

import { type SkillSurface } from "./registry.js";
import { InstallerError } from "./errors.js";

export const HOOK_BLOCK_BEGIN = "# BEGIN mind-nerve hook (managed — do not edit)";
export const HOOK_BLOCK_END = "# END mind-nerve hook";

// ---------------------------------------------------------------------------
// JSON hook wiring (claude-code, gemini, qwen)
// ---------------------------------------------------------------------------

interface JsonHookCommand {
  type: "command";
  command: string;
  timeout?: number;
}

interface JsonHookMatcher {
  matcher?: string;
  hooks?: JsonHookCommand[];
}

export interface MergeResult<T> {
  readonly updated: T;
  readonly changed: boolean;
}

/**
 * Merges the hook command into a JSON settings object.
 *
 * Idempotent by command path: if any matcher group for the event already runs
 * this exact command, nothing is added. Re-running install therefore cannot
 * accumulate duplicate hook entries — the failure mode that makes a CLI invoke
 * the router N times per prompt.
 *
 * Foreign hooks on the same event are preserved untouched.
 */
export function mergeJsonHooks(
  existing: Record<string, unknown>,
  surface: SkillSurface,
): MergeResult<Record<string, unknown>> {
  const command = surface.hookScriptPath;
  const priorHooks = isRecord(existing["hooks"]) ? existing["hooks"] : {};
  const nextHooks: Record<string, unknown> = { ...priorHooks };
  let changed = false;

  for (const event of surface.hookEvents) {
    const groupsRaw = nextHooks[event];
    const groups: JsonHookMatcher[] = Array.isArray(groupsRaw)
      ? (groupsRaw as JsonHookMatcher[])
      : [];

    if (groups.some((g) => groupRunsCommand(g, command))) {
      continue; // already wired — no-op
    }

    nextHooks[event] = [
      ...groups,
      {
        matcher: "",
        hooks: [
          {
            type: "command" as const,
            command,
            timeout: hookTimeoutValue(surface),
          },
        ],
      },
    ];
    changed = true;
  }

  if (!changed) return { updated: existing, changed: false };
  return { updated: { ...existing, hooks: nextHooks }, changed: true };
}

/**
 * Removes every mind-nerve hook entry (matched by command path) from a JSON
 * settings object, pruning matcher groups and event keys that become empty.
 * Foreign hooks are preserved.
 */
export function removeJsonHooks(
  existing: Record<string, unknown>,
  surface: SkillSurface,
): MergeResult<Record<string, unknown>> {
  if (!isRecord(existing["hooks"])) return { updated: existing, changed: false };

  const command = surface.hookScriptPath;
  const nextHooks: Record<string, unknown> = {};
  let changed = false;

  for (const [event, groupsRaw] of Object.entries(existing["hooks"])) {
    if (!Array.isArray(groupsRaw)) {
      nextHooks[event] = groupsRaw;
      continue;
    }
    const kept: JsonHookMatcher[] = [];
    for (const group of groupsRaw as JsonHookMatcher[]) {
      const inner = Array.isArray(group.hooks) ? group.hooks : [];
      const survivors = inner.filter((hk) => hk.command !== command);
      if (survivors.length !== inner.length) changed = true;
      if (survivors.length > 0) {
        kept.push({ ...group, hooks: survivors });
      } else if (inner.length === 0) {
        kept.push(group); // group had no hooks to begin with — leave it alone
      }
    }
    if (kept.length > 0) nextHooks[event] = kept;
    else if (groupsRaw.length > 0) changed = true;
  }

  if (!changed) return { updated: existing, changed: false };

  const updated: Record<string, unknown> = { ...existing };
  if (Object.keys(nextHooks).length > 0) updated["hooks"] = nextHooks;
  else delete updated["hooks"];
  return { updated, changed: true };
}

// ---------------------------------------------------------------------------
// TOML hook wiring (codex, grok: "toml-hooks"; kimi: "toml-hooks-kimi")
// ---------------------------------------------------------------------------

/**
 * Renders the managed TOML block. Deterministic — used for idempotency too.
 *
 * Two shapes, keyed off the surface's hookWireFmt:
 *  - "toml-hooks":      `[[hooks.<Event>]]` tables with a `hooks = [...]`
 *                       array — the Claude-native shape codex/grok parse.
 *  - "toml-hooks-kimi": flat `[[hooks]]` array elements with only
 *                       event/command/timeout — the ONLY shape kimi-code
 *                       reads (official hooks docs). Extra fields inside a
 *                       `[[hooks]]` element can fail kimi's whole config
 *                       load, so nothing else is emitted.
 */
export function buildTomlHookBlock(surface: SkillSurface): string {
  const lines: string[] = [HOOK_BLOCK_BEGIN];
  if (surface.hookWireFmt === "toml-hooks-kimi") {
    for (const event of surface.hookEvents) {
      lines.push(
        "[[hooks]]",
        `event = ${JSON.stringify(event)}`,
        `command = ${JSON.stringify(surface.hookScriptPath)}`,
        `timeout = ${hookTimeoutValue(surface)}`,
        "",
      );
    }
  } else {
    for (const event of surface.hookEvents) {
      lines.push(
        `[[hooks.${event}]]`,
        "hooks = [",
        `  { type = "command", command = ${JSON.stringify(surface.hookScriptPath)}, ` +
          `timeout = ${hookTimeoutValue(surface)} },`,
        "]",
        "",
      );
    }
  }
  lines.push(HOOK_BLOCK_END, "");
  return lines.join("\n");
}

/** Matches the whole managed block, including a trailing newline. */
function tomlBlockPattern(): RegExp {
  return new RegExp(
    `\\n*${escapeRegex(HOOK_BLOCK_BEGIN)}[\\s\\S]*?${escapeRegex(HOOK_BLOCK_END)}\\n?`,
    "g",
  );
}

/**
 * Merges the hook block into a TOML config as text.
 *
 * Idempotent: an existing managed block is replaced wholesale, so re-running
 * cannot duplicate it, and a changed hook path or timeout is picked up. Returns
 * `changed: false` when the resulting text is byte-identical.
 */
export function mergeTomlHooks(
  existingText: string,
  surface: SkillSurface,
): MergeResult<string> {
  const block = buildTomlHookBlock(surface);
  const text = existingText ?? "";

  const stripped = text.replace(tomlBlockPattern(), "\n").replace(/\n{3,}/g, "\n\n");
  const base = stripped.trimEnd();
  const separator = base.length > 0 ? "\n\n" : "";
  const updated = base + separator + block;

  return { updated, changed: updated !== text };
}

/** Removes the managed TOML block. Foreign config is preserved byte-for-byte. */
export function removeTomlHooks(existingText: string): MergeResult<string> {
  const text = existingText ?? "";
  if (!text.includes(HOOK_BLOCK_BEGIN)) return { updated: text, changed: false };
  const cleaned = text
    .replace(tomlBlockPattern(), "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trimEnd();
  const updated = cleaned.length > 0 ? cleaned + "\n" : "";
  return { updated, changed: updated !== text };
}

// ---------------------------------------------------------------------------
// Environment block — parameterises the generic hook per CLI
// ---------------------------------------------------------------------------

/**
 * The env the hook needs to behave as THIS CLI's router. Written next to the
 * hook script so the wiring is inspectable, and so a user can override any
 * knob without editing the shared script.
 *
 * ONLY path/socket pins are exported. Tuning knobs (TOP_K, MIN_SCORE,
 * CORE_SKILLS, SOCKET_TIMEOUT, ...) are deliberately NOT pinned: the shared
 * hook is their single source of truth, and a pinned copy goes stale — the
 * wrapper used to pin MIN_SCORE=0.35, silently overriding the hook's
 * recalibrated 0.40 default (2026-08-08).
 *
 * MIND_NERVE_LOG is likewise NOT exported: hook disk logging is opt-in
 * (privacy contract — codex final #30), so the wrapper must not enable it
 * for every prompt.
 */
export function buildHookEnv(args: {
  readonly surface: SkillSurface;
  readonly hubDir: string;
  readonly socketPath: string;
  readonly agentDirs: readonly string[];
}): Record<string, string> {
  return {
    MIND_NERVE_SOCKET: args.socketPath,
    MIND_NERVE_SOURCE_DIR: args.hubDir,
    MIND_NERVE_PROJECTED_DIR: args.surface.skillsDir,
    MIND_NERVE_AGENT_DIRS: args.agentDirs.join(":"),
  };
}

/**
 * Renders a POSIX `sh` wrapper that exports the per-CLI env and execs the
 * shared hook. This is what actually gets installed at `hookScriptPath`, so
 * every CLI runs ONE implementation with different parameters.
 */
export function buildHookWrapper(
  realHookPath: string,
  env: Record<string, string>,
): string {
  const exports = Object.entries(env)
    .map(([k, v]) => `export ${k}=${shQuote(v)}`)
    .join("\n");
  return [
    "#!/bin/sh",
    "# mind-nerve hook wrapper (managed — regenerate with `mind-nerve-installer install`).",
    "# Parameterises the shared CLI-agnostic router for this client.",
    "# Fail-open: if anything here breaks, emit {} and exit 0 so the prompt is never blocked.",
    exports,
    // `exec missing || fallback` NEVER runs the fallback: under POSIX sh a
    // failed exec (command not found) terminates the shell with code 127
    // before the `||` list is reached (qwen Q6). Guard with `[ -x ]` first
    // so a missing/non-executable real hook falls open instead.
    `if [ -x ${shQuote(realHookPath)} ]; then`,
    `  exec ${shQuote(realHookPath)} "$@"`,
    "fi",
    "echo '{}'",
    "exit 0",
    "",
  ].join("\n");
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

export function isJsonHookFmt(surface: SkillSurface): boolean {
  return surface.hookWireFmt === "json-hooks";
}

export function isTomlHookFmt(surface: SkillSurface): boolean {
  return (
    surface.hookWireFmt === "toml-hooks" ||
    surface.hookWireFmt === "toml-hooks-kimi"
  );
}

export function assertKnownHookFmt(
  surface: SkillSurface,
  clientName: string,
): void {
  if (!isJsonHookFmt(surface) && !isTomlHookFmt(surface)) {
    throw new InstallerError(
      "INVALID_CONFIG_FORMAT",
      clientName,
      `Unknown hookWireFmt: ${String(surface.hookWireFmt)}`,
    );
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function groupRunsCommand(group: JsonHookMatcher, command: string): boolean {
  if (!Array.isArray(group.hooks)) return false;
  return group.hooks.some((hk) => hk?.command === command);
}

/**
 * Renders `hookTimeoutSecs` in the unit the CLI actually reads. Every CLI
 * reads seconds except gemini-cli (milliseconds — see `HookTimeoutUnit`).
 */
function hookTimeoutValue(surface: SkillSurface): number {
  return surface.hookTimeoutUnit === "milliseconds"
    ? surface.hookTimeoutSecs * 1000
    : surface.hookTimeoutSecs;
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Single-quote for POSIX sh — the only fully safe quoting for arbitrary paths. */
function shQuote(s: string): string {
  return `'${s.replace(/'/g, `'\\''`)}'`;
}
