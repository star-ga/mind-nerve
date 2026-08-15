// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.
//
// Part 2 config-registration tests. The failure this guards against is a
// re-run that appends a second copy of the hook, making the CLI invoke the
// router twice per prompt.

import { describe, it, expect } from "vitest";
import TOML from "@iarna/toml";
import {
  mergeJsonHooks,
  removeJsonHooks,
  mergeTomlHooks,
  removeTomlHooks,
  buildTomlHookBlock,
  buildHookEnv,
  buildHookWrapper,
  HOOK_BLOCK_BEGIN,
} from "../src/hook_wiring.js";
import { AGENT_REGISTRY, type SkillSurface } from "../src/registry.js";

const SURFACE: SkillSurface = {
  skillsDir: "/home/u/.test/skills",
  hookScriptPath: "/home/u/.test/hooks/mind-nerve-hook",
  hookConfigPath: "/home/u/.test/settings.json",
  hookWireFmt: "json-hooks",
  hookEvents: ["UserPromptSubmit", "SessionStart"],
  hookTimeoutSecs: 8,
  verified: "fixture",
};

const TOML_SURFACE: SkillSurface = { ...SURFACE, hookWireFmt: "toml-hooks" };

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------

describe("mergeJsonHooks", () => {
  it("adds the hook to every configured event", () => {
    const { updated, changed } = mergeJsonHooks({}, SURFACE);
    expect(changed).toBe(true);
    const hooks = updated["hooks"] as Record<string, unknown[]>;
    expect(Object.keys(hooks).sort()).toEqual(["SessionStart", "UserPromptSubmit"]);
  });

  it("is idempotent — a second merge adds nothing", () => {
    const first = mergeJsonHooks({}, SURFACE);
    const second = mergeJsonHooks(first.updated, SURFACE);
    expect(second.changed).toBe(false);
    expect(second.updated).toEqual(first.updated);
  });

  it("never duplicates the hook across many re-runs", () => {
    let cfg: Record<string, unknown> = {};
    for (let i = 0; i < 5; i++) cfg = mergeJsonHooks(cfg, SURFACE).updated;
    const groups = (cfg["hooks"] as Record<string, unknown[]>)[
      "UserPromptSubmit"
    ] as Array<{ hooks: Array<{ command: string }> }>;
    const commands = groups.flatMap((g) => g.hooks.map((h) => h.command));
    expect(commands.filter((c) => c === SURFACE.hookScriptPath).length).toBe(1);
  });

  it("preserves foreign hooks on the same event", () => {
    const existing = {
      hooks: {
        SessionStart: [
          { matcher: "", hooks: [{ type: "command", command: "mm status" }] },
        ],
      },
    };
    const { updated } = mergeJsonHooks(existing, SURFACE);
    const groups = (updated["hooks"] as Record<string, unknown[]>)[
      "SessionStart"
    ] as Array<{ hooks: Array<{ command: string }> }>;
    const commands = groups.flatMap((g) => g.hooks.map((h) => h.command));
    expect(commands).toContain("mm status");
    expect(commands).toContain(SURFACE.hookScriptPath);
  });

  it("preserves unrelated top-level settings", () => {
    const existing = { model: "opus", permissions: { allow: ["Bash"] } };
    const { updated } = mergeJsonHooks(existing, SURFACE);
    expect(updated["model"]).toBe("opus");
    expect(updated["permissions"]).toEqual({ allow: ["Bash"] });
  });

  it("does not mutate the input object", () => {
    const existing = { hooks: { SessionStart: [] } };
    mergeJsonHooks(existing, SURFACE);
    expect(existing).toEqual({ hooks: { SessionStart: [] } });
  });

  it("records the configured timeout", () => {
    const { updated } = mergeJsonHooks({}, SURFACE);
    const groups = (updated["hooks"] as Record<string, unknown[]>)[
      "UserPromptSubmit"
    ] as Array<{ hooks: Array<{ timeout: number }> }>;
    expect(groups[0]!.hooks[0]!.timeout).toBe(8);
  });
});

describe("removeJsonHooks", () => {
  it("removes exactly what merge added", () => {
    const merged = mergeJsonHooks({}, SURFACE).updated;
    const { updated, changed } = removeJsonHooks(merged, SURFACE);
    expect(changed).toBe(true);
    expect(updated["hooks"]).toBeUndefined();
  });

  it("leaves foreign hooks behind", () => {
    const existing = {
      hooks: {
        SessionStart: [
          { matcher: "", hooks: [{ type: "command", command: "mm status" }] },
        ],
      },
    };
    const merged = mergeJsonHooks(existing, SURFACE).updated;
    const { updated } = removeJsonHooks(merged, SURFACE);
    const groups = (updated["hooks"] as Record<string, unknown[]>)[
      "SessionStart"
    ] as Array<{ hooks: Array<{ command: string }> }>;
    expect(groups.flatMap((g) => g.hooks.map((h) => h.command))).toEqual([
      "mm status",
    ]);
  });

  it("is a no-op when the hook is not present", () => {
    const existing = { hooks: { SessionStart: [] } };
    expect(removeJsonHooks(existing, SURFACE).changed).toBe(false);
  });

  it("round-trips: merge then remove restores the original", () => {
    const original = {
      model: "opus",
      hooks: {
        Stop: [{ matcher: "", hooks: [{ type: "command", command: "mm status" }] }],
      },
    };
    const merged = mergeJsonHooks(original, SURFACE).updated;
    const restored = removeJsonHooks(merged, SURFACE).updated;
    expect(restored).toEqual(original);
  });
});

// ---------------------------------------------------------------------------
// TOML
// ---------------------------------------------------------------------------

describe("mergeTomlHooks", () => {
  it("appends a marked block with an array-of-tables per event", () => {
    const { updated, changed } = mergeTomlHooks("", TOML_SURFACE);
    expect(changed).toBe(true);
    expect(updated).toContain(HOOK_BLOCK_BEGIN);
    expect(updated).toContain("[[hooks.UserPromptSubmit]]");
    expect(updated).toContain("[[hooks.SessionStart]]");
    expect(updated).toContain(JSON.stringify(TOML_SURFACE.hookScriptPath));
  });

  it("is idempotent — a second merge is byte-identical", () => {
    const first = mergeTomlHooks("", TOML_SURFACE);
    const second = mergeTomlHooks(first.updated, TOML_SURFACE);
    expect(second.changed).toBe(false);
    expect(second.updated).toBe(first.updated);
  });

  it("never duplicates the block across many re-runs", () => {
    let text = "";
    for (let i = 0; i < 5; i++) text = mergeTomlHooks(text, TOML_SURFACE).updated;
    expect(text.split(HOOK_BLOCK_BEGIN).length - 1).toBe(1);
    expect(text.split("[[hooks.UserPromptSubmit]]").length - 1).toBe(1);
  });

  it("preserves existing config INCLUDING comments", () => {
    // codex's live config is 125 KB of commented `skills.config`. A TOML
    // round-trip through a serialiser would silently drop every comment.
    const existing = [
      "# BEGIN mind-nerve-skills-budget (generated)",
      "# 1306 of 1341 hub skills disabled",
      "skills.config = [",
      '  {path="/hub/a/SKILL.md", enabled=false},',
      "]",
      "",
      "[models]",
      'default = "grok-build"',
      "",
    ].join("\n");

    const { updated } = mergeTomlHooks(existing, TOML_SURFACE);
    expect(updated).toContain("# BEGIN mind-nerve-skills-budget (generated)");
    expect(updated).toContain("# 1306 of 1341 hub skills disabled");
    expect(updated).toContain('  {path="/hub/a/SKILL.md", enabled=false},');
    expect(updated).toContain('default = "grok-build"');
  });

  it("appends AFTER existing tables so no key is captured by them", () => {
    const existing = '[models]\ndefault = "x"\n';
    const { updated } = mergeTomlHooks(existing, TOML_SURFACE);
    expect(updated.indexOf("[models]")).toBeLessThan(
      updated.indexOf(HOOK_BLOCK_BEGIN),
    );
  });

  it("replaces a stale block when the hook path changes", () => {
    const first = mergeTomlHooks("", TOML_SURFACE).updated;
    const moved: SkillSurface = {
      ...TOML_SURFACE,
      hookScriptPath: "/home/u/.test/hooks/mind-nerve-hook-v2",
    };
    const { updated, changed } = mergeTomlHooks(first, moved);
    expect(changed).toBe(true);
    expect(updated).toContain("mind-nerve-hook-v2");
    expect(updated.split(HOOK_BLOCK_BEGIN).length - 1).toBe(1);
    expect(updated).not.toContain('"/home/u/.test/hooks/mind-nerve-hook"');
  });
});

describe("removeTomlHooks", () => {
  it("removes the whole managed block", () => {
    const merged = mergeTomlHooks("", TOML_SURFACE).updated;
    const { updated, changed } = removeTomlHooks(merged);
    expect(changed).toBe(true);
    expect(updated).not.toContain(HOOK_BLOCK_BEGIN);
    expect(updated).not.toContain("[[hooks.UserPromptSubmit]]");
  });

  it("restores the prior config byte-for-byte", () => {
    const existing = '# comment\n[models]\ndefault = "x"\n';
    const merged = mergeTomlHooks(existing, TOML_SURFACE).updated;
    expect(removeTomlHooks(merged).updated).toBe(existing);
  });

  it("is a no-op when no block is present", () => {
    expect(removeTomlHooks('[models]\ndefault = "x"\n').changed).toBe(false);
  });
});

describe("buildTomlHookBlock", () => {
  it("is deterministic", () => {
    expect(buildTomlHookBlock(TOML_SURFACE)).toBe(
      buildTomlHookBlock(TOML_SURFACE),
    );
  });
});

// ---------------------------------------------------------------------------
// kimi TOML shape — flat [[hooks]] elements, the ONLY shape kimi-code reads
// (official hooks docs, fetched 2026-08-11). Extra fields inside a [[hooks]]
// element can fail kimi's whole config load, so the block must be minimal.
// ---------------------------------------------------------------------------

const KIMI_SURFACE: SkillSurface = { ...TOML_SURFACE, hookWireFmt: "toml-hooks-kimi" };

describe("kimi TOML hook shape (toml-hooks-kimi)", () => {
  it("renders flat [[hooks]] entries with event/command/timeout", () => {
    const block = buildTomlHookBlock(KIMI_SURFACE);
    expect(block).toContain(HOOK_BLOCK_BEGIN);
    expect(block).toContain("[[hooks]]");
    expect(block).toContain('event = "UserPromptSubmit"');
    expect(block).toContain('event = "SessionStart"');
    expect(block).toContain(`command = ${JSON.stringify(KIMI_SURFACE.hookScriptPath)}`);
    expect(block).toContain(`timeout = ${KIMI_SURFACE.hookTimeoutSecs}`);
  });

  it("never emits the Claude [[hooks.<Event>]] shape", () => {
    const block = buildTomlHookBlock(KIMI_SURFACE);
    expect(block).not.toContain("[[hooks.");
    expect(block).not.toContain("hooks = [");
  });

  it("parses as TOML with only event/command/timeout per element", () => {
    const existing = '[model]\ndefault = "kimi-for-coding"\n';
    const { updated } = mergeTomlHooks(existing, KIMI_SURFACE);
    const parsed = TOML.parse(updated) as Record<string, unknown>;
    expect(parsed["model"]).toEqual({ default: "kimi-for-coding" });
    const hooks = parsed["hooks"] as Array<Record<string, unknown>>;
    expect(hooks).toHaveLength(2);
    for (const element of hooks) {
      expect(Object.keys(element).sort()).toEqual(["command", "event", "timeout"]);
      expect(element["command"]).toBe(KIMI_SURFACE.hookScriptPath);
    }
    expect(hooks.map((h) => h["event"]).sort()).toEqual([
      "SessionStart",
      "UserPromptSubmit",
    ]);
  });

  it("is idempotent via mergeTomlHooks and removable byte-for-byte", () => {
    const existing = '[model]\ndefault = "x"\n';
    const first = mergeTomlHooks(existing, KIMI_SURFACE).updated;
    expect(mergeTomlHooks(first, KIMI_SURFACE).changed).toBe(false);
    expect(removeTomlHooks(first).updated).toBe(existing);
  });
});

// ---------------------------------------------------------------------------
// Wrapper generation
// ---------------------------------------------------------------------------

describe("buildHookWrapper", () => {
  const env = buildHookEnv({
    surface: SURFACE,
    hubDir: "/home/u/.agents/skills-hub",
    socketPath: "/run/user/1000/mind-nerve.sock",
    agentDirs: ["/home/u/.claude/agents"],
    logPath: "/home/u/.mind-nerve/logs/x.log",
  });

  it("exports only the path/socket pins that parameterise the shared hook", () => {
    expect(Object.keys(env).sort()).toEqual([
      "MIND_NERVE_AGENT_DIRS",
      "MIND_NERVE_LOG",
      "MIND_NERVE_PROJECTED_DIR",
      "MIND_NERVE_SOCKET",
      "MIND_NERVE_SOURCE_DIR",
    ]);
  });

  it("points the projection at THIS CLI's skills dir", () => {
    expect(env["MIND_NERVE_PROJECTED_DIR"]).toBe(SURFACE.skillsDir);
  });

  it("does NOT pin tuning knobs — the hook owns their defaults", () => {
    // Regression: the wrapper used to pin MIND_NERVE_MIN_SCORE=0.35, silently
    // overriding the hook's recalibrated 0.40 default (2026-08-08). The hook
    // source is the single source of truth for every tuning knob.
    for (const knob of [
      "MIND_NERVE_MIN_SCORE",
      "MIND_NERVE_TOP_K",
      "MIND_NERVE_CORE_SKILLS",
      "MIND_NERVE_SOCKET_TIMEOUT",
    ]) {
      expect(env, `${knob} must not be pinned`).not.toHaveProperty(knob);
    }
  });

  it("execs the ONE shared hook implementation", () => {
    const script = buildHookWrapper("/home/u/.mind-nerve/bin/mind-nerve-hook", env);
    expect(script.startsWith("#!/bin/sh")).toBe(true);
    expect(script).toContain("exec '/home/u/.mind-nerve/bin/mind-nerve-hook'");
  });

  it("fails open even if the exec fails", () => {
    const script = buildHookWrapper("/nope", env);
    expect(script).toContain("echo '{}'");
    expect(script).toContain("exit 0");
  });

  it("single-quotes values so a path with spaces cannot break the script", () => {
    const spaced = buildHookWrapper(
      "/home/u/my hooks/mind-nerve-hook",
      { MIND_NERVE_SOURCE_DIR: "/home/u/my skills/hub" },
    );
    expect(spaced).toContain("export MIND_NERVE_SOURCE_DIR='/home/u/my skills/hub'");
    expect(spaced).toContain("exec '/home/u/my hooks/mind-nerve-hook'");
  });

  it("escapes an embedded single quote", () => {
    const script = buildHookWrapper("/bin/true", { X: "it's here" });
    expect(script).toContain(`export X='it'\\''s here'`);
  });
});

// ---------------------------------------------------------------------------
// Every registered surface must be wirable
// ---------------------------------------------------------------------------

describe("registry surfaces are wirable", () => {
  it("merges cleanly for all six CLIs and is idempotent", () => {
    for (const spec of AGENT_REGISTRY.values()) {
      const s = spec.skillSurface;
      if (s === null) continue;

      if (s.hookWireFmt === "json-hooks") {
        const first = mergeJsonHooks({}, s);
        expect(first.changed, spec.name).toBe(true);
        expect(mergeJsonHooks(first.updated, s).changed, spec.name).toBe(false);
        expect(removeJsonHooks(first.updated, s).updated, spec.name).toEqual({});
      } else {
        const first = mergeTomlHooks("", s);
        expect(first.changed, spec.name).toBe(true);
        expect(mergeTomlHooks(first.updated, s).changed, spec.name).toBe(false);
        expect(removeTomlHooks(first.updated).updated, spec.name).toBe("");
      }
    }
  });
});
