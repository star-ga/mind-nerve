// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.
// TODO(Phase 2): port installer to mind-dev (MIND language native).

import os from "node:os";
import path from "node:path";

/** Config file formats the installer knows how to merge. */
export type ConfigFmt =
  | "json-claude-hooks"
  | "json-openclaw-hooks"
  | "json-gemini"
  | "json-continue"
  | "json-zed"
  | "json-generic"
  | "toml-codex"
  | "toml-vibe"
  | "yaml-aider"
  | "text-block";

/** MCP config formats — null means the client has no MCP surface. */
export type McpFmt =
  | "mcp-json-servers"
  | "mcp-json-cursor"
  | "mcp-json-windsurf"
  | "mcp-json-zed"
  | "mcp-toml-codex"
  | "mcp-toml-vibe"
  | null;

/**
 * How a client serialises its hook registration.
 *
 * All six skill-surface CLIs speak the Claude-compatible hook protocol
 * (`UserPromptSubmit` / `SessionStart`, JSON on stdin, `hookSpecificOutput`
 * with `additionalContext` on stdout) — they differ only in the container:
 *
 *  - "json-hooks":      a JSON settings file (claude-code, gemini, qwen).
 *  - "toml-hooks":      TOML `[[hooks.<Event>]]` tables holding a `hooks = [...]`
 *                       array (codex, grok — grok's live config is the working
 *                       reference for this shape).
 *  - "toml-hooks-kimi": TOML `[[hooks]]` array elements with flat
 *                       event/matcher/command/timeout fields — the ONLY shape
 *                       kimi-code parses, per the official docs
 *                       (kimi.com/code/docs/en/kimi-code-cli/customization/hooks.html).
 *                       The Claude `[[hooks.<Event>]]` shape is silently dead
 *                       there, and extra fields inside a `[[hooks]]` element
 *                       can fail the whole config load.
 */
export type HookWireFmt = "json-hooks" | "toml-hooks" | "toml-hooks-kimi";

/**
 * The unit `hookTimeoutSecs` is actually serialised in for a given CLI.
 * Every CLI reads seconds EXCEPT gemini-cli, whose hook timeout field is
 * milliseconds (bundle-verified, b10 live-integration) — writing the raw
 * value there would under-time the hook by 1000x. Optional and defaults to
 * "seconds" so every hand-built fixture surface in the test suite keeps
 * working unchanged.
 */
export type HookTimeoutUnit = "seconds" | "milliseconds";

/**
 * The skill-announce surface of a CLI: where its skills directory lives, and
 * how the generic mind-nerve hook gets registered.
 *
 * Present only for clients that announce a skills directory into the model
 * context. Everything here was confirmed on disk — see `verified`.
 */
export interface SkillSurface {
  /** The CLI's skills directory. This is what gets announced to the model. */
  readonly skillsDir: string;
  /** Where the generic hook script is installed for this CLI. */
  readonly hookScriptPath: string;
  /** Config file the hook registration is merged into. */
  readonly hookConfigPath: string;
  /** Serialisation format of that config file. */
  readonly hookWireFmt: HookWireFmt;
  /** Hook events to register the router on. */
  readonly hookEvents: readonly string[];
  /** Per-invocation hook timeout, in `hookTimeoutUnit` (default seconds). */
  readonly hookTimeoutSecs: number;
  /** Unit `hookTimeoutSecs` is serialised in. Defaults to "seconds". */
  readonly hookTimeoutUnit?: HookTimeoutUnit;
  /** How this shape was verified on disk — kept honest, not guessed. */
  readonly verified: string;
}

export interface AgentSpec {
  /** Canonical client identifier — must be unique across registry. */
  readonly name: string;
  /** Human-readable description. */
  readonly description: string;
  /** How the primary config file is serialised. */
  readonly configFmt: ConfigFmt;
  /** Absolute path template to the primary config file. */
  readonly configPath: string;
  /** Absolute path template to the MCP config file (may equal configPath). */
  readonly mcpPath: string | null;
  /** MCP serialisation format — null if this client has no MCP surface. */
  readonly mcpFmt: McpFmt;
  /**
   * When true, verify FAILs the managed MCP entry unless it carries
   * transport = "stdio" (Vibe 2.9.6 rejects entries without the
   * discriminator — codex#16).
   */
  readonly mcpRequiresStdioTransport?: boolean;
  /** Paths to probe when checking if the client is installed (fs existence). */
  readonly detectPaths: readonly string[];
  /** Binary names to probe on $PATH. */
  readonly detectBinaries: readonly string[];
  /**
   * When true, list-clients always shows this client as a candidate even
   * when it is not detected. Copilot is near-universal.
   */
  readonly alwaysOffer: boolean;
  /**
   * Absolute path to the per-CLI skill projection directory inside
   * ~/.mind-nerve/projections/<name>/. Null for clients without a skill surface.
   */
  readonly projectionDir: string | null;
  /**
   * Workspace-relative path for instruction-block injection (cursor, windsurf,
   * aider, copilot, cody, qodo, cline, roo). Null when not applicable.
   */
  readonly instructionFilePath: string | null;
  /**
   * Skill-announce surface + hook wiring. Null for clients that do not
   * announce a skills directory.
   */
  readonly skillSurface: SkillSurface | null;
}

function h(): string {
  return os.homedir();
}

/** Hook events most skill-surface CLIs are wired on (gemini differs — see below). */
const HOOK_EVENTS: readonly string[] = ["UserPromptSubmit", "SessionStart"];

/**
 * Builds a SkillSurface from the parts that actually differ per CLI.
 * `hookTimeoutSecs` of 8 matches the live, working Grok wiring.
 * `hookEvents`/`hookTimeoutUnit` default to the common shape and are
 * overridable for a CLI whose bundle maps events/units differently (gemini).
 */
function surface(args: {
  readonly skillsDir: string;
  readonly hookScriptPath: string;
  readonly hookConfigPath: string;
  readonly hookWireFmt: HookWireFmt;
  readonly verified: string;
  readonly hookEvents?: readonly string[];
  readonly hookTimeoutUnit?: HookTimeoutUnit;
}): SkillSurface {
  return {
    skillsDir: args.skillsDir,
    hookScriptPath: args.hookScriptPath,
    hookConfigPath: args.hookConfigPath,
    hookWireFmt: args.hookWireFmt,
    hookEvents: args.hookEvents ?? HOOK_EVENTS,
    hookTimeoutSecs: 8,
    hookTimeoutUnit: args.hookTimeoutUnit ?? "seconds",
    verified: args.verified,
  };
}

/**
 * Build the projection directory path for a client that has a skill surface.
 */
function projDir(clientName: string): string {
  return path.join(h(), ".mind-nerve", "projections", clientName);
}

/**
 * AGENT_REGISTRY — 20 AI coding clients.
 *
 * Ported from mind-mem src/mind_mem/hook_installer.py AGENT_REGISTRY (lines 629-808)
 * and extended with mind-nerve-specific fields (projectionDir, instructionFilePath).
 */
export const AGENT_REGISTRY: ReadonlyMap<string, AgentSpec> = new Map<
  string,
  AgentSpec
>([
  [
    "claude-code",
    {
      name: "claude-code",
      description: "Claude Code CLI (Anthropic)",
      configFmt: "json-claude-hooks",
      configPath: path.join(h(), ".claude", "settings.json"),
      mcpPath: path.join(h(), ".claude", "settings.json"),
      mcpFmt: "mcp-json-servers",
      detectPaths: [
        path.join(h(), ".claude"),
        path.join(h(), ".config", "claude"),
      ],
      detectBinaries: ["claude"],
      alwaysOffer: false,
      projectionDir: projDir("claude-code"),
      instructionFilePath: null,
      skillSurface: surface({
        skillsDir: path.join(h(), ".claude", "skills"),
        hookScriptPath: path.join(h(), ".claude", "hooks", "mind-nerve-hook"),
        hookConfigPath: path.join(h(), ".claude", "settings.json"),
        hookWireFmt: "json-hooks",
        verified:
          "~/.claude/settings.json read live (hooks.SessionStart/.Stop present); " +
          "~/.claude/hooks/ exists; ~/.claude/skills is a real dir (2 entries).",
      }),
    },
  ],
  [
    "codex",
    {
      name: "codex",
      description: "OpenAI Codex CLI",
      configFmt: "toml-codex",
      configPath: path.join(h(), ".codex", "config.toml"),
      mcpPath: path.join(h(), ".codex", "config.toml"),
      mcpFmt: "mcp-toml-codex",
      detectPaths: [path.join(h(), ".codex")],
      detectBinaries: ["codex"],
      alwaysOffer: false,
      projectionDir: null,
      instructionFilePath: null,
      skillSurface: surface({
        skillsDir: path.join(h(), ".codex", "skills"),
        hookScriptPath: path.join(h(), ".codex", "hooks", "mind-nerve-hook"),
        hookConfigPath: path.join(h(), ".codex", "config.toml"),
        hookWireFmt: "toml-hooks",
        verified:
          "~/.codex/config.toml read live (skills.config array present); " +
          "~/.codex/skills is a real dir (2 entries); native codex binary " +
          "contains UserPromptSubmit x9, SessionStart x36, SKILL.md x101.",
      }),
    },
  ],
  [
    "vibe",
    {
      name: "vibe",
      description: "Mistral Vibe CLI",
      configFmt: "toml-vibe",
      configPath: path.join(h(), ".vibe", "config.toml"),
      mcpPath: path.join(h(), ".vibe", "config.toml"),
      mcpFmt: "mcp-toml-vibe",
      mcpRequiresStdioTransport: true,
      detectPaths: [path.join(h(), ".vibe")],
      detectBinaries: ["vibe"],
      alwaysOffer: false,
      projectionDir: null,
      instructionFilePath: null,
      skillSurface: null,
    },
  ],
  [
    "gemini",
    {
      name: "gemini",
      description: "Google Gemini CLI",
      configFmt: "json-gemini",
      configPath: path.join(h(), ".gemini", "settings.json"),
      mcpPath: path.join(h(), ".gemini", "settings.json"),
      mcpFmt: "mcp-json-servers",
      detectPaths: [path.join(h(), ".gemini")],
      detectBinaries: ["gemini"],
      alwaysOffer: false,
      projectionDir: null,
      instructionFilePath: null,
      skillSurface: surface({
        skillsDir: path.join(h(), ".gemini", "skills"),
        hookScriptPath: path.join(h(), ".gemini", "hooks", "mind-nerve-hook"),
        hookConfigPath: path.join(h(), ".gemini", "settings.json"),
        hookWireFmt: "json-hooks",
        // gemini-cli maps UserPromptSubmit -> BeforeAgent and has no
        // separate UserPromptSubmit event; its hook timeout field is
        // milliseconds, not seconds, unlike every other json-hooks CLI.
        hookEvents: ["BeforeAgent", "SessionStart"],
        hookTimeoutUnit: "milliseconds",
        verified:
          "~/.gemini/settings.json read live; ~/.gemini/skills is a real dir " +
          "(2 entries); gemini-cli bundle maps UserPromptSubmit -> BeforeAgent " +
          "and ships SKILL.md support (SessionStart in 18 chunks).",
      }),
    },
  ],
  [
    "grok",
    {
      name: "grok",
      description: "xAI Grok CLI",
      configFmt: "toml-vibe",
      configPath: path.join(h(), ".grok", "config.toml"),
      mcpPath: path.join(h(), ".grok", "config.toml"),
      mcpFmt: "mcp-toml-vibe",
      detectPaths: [path.join(h(), ".grok")],
      detectBinaries: ["grok"],
      alwaysOffer: false,
      projectionDir: null,
      instructionFilePath: null,
      skillSurface: surface({
        skillsDir: path.join(h(), ".grok", "skills"),
        hookScriptPath: path.join(h(), ".grok", "hooks", "mind-nerve-hook"),
        hookConfigPath: path.join(h(), ".grok", "config.toml"),
        hookWireFmt: "toml-hooks",
        verified:
          "~/.grok/config.toml read live: [[hooks.UserPromptSubmit]] and " +
          "[[hooks.SessionStart]] already wired to ~/.grok/hooks/, timeout 8. " +
          "This is the working reference implementation.",
      }),
    },
  ],
  [
    "kimi",
    {
      name: "kimi",
      description: "Moonshot Kimi Code CLI",
      configFmt: "toml-vibe",
      configPath: path.join(h(), ".kimi-code", "config.toml"),
      mcpPath: path.join(h(), ".kimi-code", "mcp.json"),
      mcpFmt: "mcp-json-servers",
      detectPaths: [path.join(h(), ".kimi-code")],
      detectBinaries: ["kimi"],
      alwaysOffer: false,
      projectionDir: null,
      instructionFilePath: null,
      skillSurface: surface({
        skillsDir: path.join(h(), ".kimi-code", "skills"),
        hookScriptPath: path.join(h(), ".kimi-code", "hooks", "mind-nerve-hook"),
        hookConfigPath: path.join(h(), ".kimi-code", "config.toml"),
        hookWireFmt: "toml-hooks-kimi",
        verified:
          "Official kimi-code docs (hooks.html + mcp.html, fetched " +
          "2026-08-11): hooks are ONLY [[hooks]] array elements with " +
          "event/matcher/command/timeout in config.toml — the Claude " +
          "[[hooks.<Event>]] shape is never read. MCP servers live in " +
          "~/.kimi-code/mcp.json ({\"mcpServers\": {...}}, " +
          "Claude-Desktop-compatible), NOT config.toml [mcp_servers.*].",
      }),
    },
  ],
  [
    "qwen",
    {
      name: "qwen",
      description: "Alibaba Qwen Code CLI",
      configFmt: "json-gemini",
      configPath: path.join(h(), ".qwen", "settings.json"),
      mcpPath: path.join(h(), ".qwen", "settings.json"),
      mcpFmt: "mcp-json-servers",
      detectPaths: [path.join(h(), ".qwen")],
      detectBinaries: ["qwen"],
      alwaysOffer: false,
      projectionDir: null,
      instructionFilePath: null,
      skillSurface: surface({
        skillsDir: path.join(h(), ".qwen", "skills"),
        hookScriptPath: path.join(h(), ".qwen", "hooks", "mind-nerve-hook"),
        hookConfigPath: path.join(h(), ".qwen", "settings.json"),
        hookWireFmt: "json-hooks",
        verified:
          "~/.qwen/settings.json read live; ~/.qwen/skills is a real dir " +
          "(2 entries); qwen-code 0.21.7 chunks contain hasHooksForEvent" +
          '("UserPromptSubmit") + createHookOutput — Claude-compatible.',
      }),
    },
  ],
  [
    "cursor",
    {
      name: "cursor",
      description: "Cursor editor",
      configFmt: "text-block",
      configPath: ".cursorrules", // workspace-relative
      mcpPath: path.join(h(), ".cursor", "mcp.json"),
      mcpFmt: "mcp-json-cursor",
      detectPaths: [
        path.join(h(), ".cursor"),
        path.join(h(), "Library", "Application Support", "Cursor"),
        path.join(h(), "AppData", "Roaming", "Cursor"),
      ],
      detectBinaries: ["cursor"],
      alwaysOffer: false,
      projectionDir: null,
      instructionFilePath: ".cursorrules",
      skillSurface: null,
    },
  ],
  [
    "windsurf",
    {
      name: "windsurf",
      description: "Windsurf editor (Codeium)",
      configFmt: "text-block",
      configPath: ".windsurfrules", // workspace-relative
      mcpPath: path.join(
        h(),
        ".codeium",
        "windsurf",
        "mcp_config.json",
      ),
      mcpFmt: "mcp-json-windsurf",
      detectPaths: [
        path.join(h(), ".codeium", "windsurf"),
        path.join(h(), ".windsurf"),
        path.join(h(), "Library", "Application Support", "Windsurf"),
      ],
      detectBinaries: ["windsurf"],
      alwaysOffer: false,
      projectionDir: null,
      instructionFilePath: ".windsurfrules",
      skillSurface: null,
    },
  ],
  [
    "continue",
    {
      name: "continue",
      description: "Continue.dev (VS Code / JetBrains extension)",
      configFmt: "json-continue",
      configPath: path.join(h(), ".continue", "config.json"),
      mcpPath: path.join(h(), ".continue", "config.json"),
      mcpFmt: "mcp-json-servers",
      detectPaths: [path.join(h(), ".continue")],
      detectBinaries: [],
      alwaysOffer: false,
      projectionDir: null,
      instructionFilePath: null,
      skillSurface: null,
    },
  ],
  [
    "cline",
    {
      name: "cline",
      description: "Cline (VS Code extension)",
      configFmt: "text-block",
      configPath: ".clinerules", // workspace-relative
      mcpPath: path.join(
        h(),
        ".vscode-server",
        "data",
        "User",
        "globalStorage",
        "saoudrizwan.claude-dev",
        "settings",
        "cline_mcp_settings.json",
      ),
      mcpFmt: "mcp-json-servers",
      detectPaths: [
        path.join(h(), ".vscode", "extensions"),
        path.join(h(), ".vscode-server", "extensions"),
      ],
      detectBinaries: [],
      alwaysOffer: false,
      projectionDir: null,
      instructionFilePath: ".clinerules",
      skillSurface: null,
    },
  ],
  [
    "roo",
    {
      name: "roo",
      description: "Roo Code (VS Code extension)",
      configFmt: "text-block",
      configPath: path.join(".roo", "system-prompt.md"), // workspace-relative
      mcpPath: path.join(
        h(),
        ".vscode-server",
        "data",
        "User",
        "globalStorage",
        "rooveterinaryinc.roo-cline",
        "settings",
        "mcp_settings.json",
      ),
      mcpFmt: "mcp-json-servers",
      detectPaths: [
        path.join(h(), ".roo"),
        path.join(h(), ".vscode", "extensions"),
      ],
      detectBinaries: [],
      alwaysOffer: false,
      projectionDir: null,
      instructionFilePath: path.join(".roo", "system-prompt.md"),
      skillSurface: null,
    },
  ],
  [
    "zed",
    {
      name: "zed",
      description: "Zed editor AI assistant",
      configFmt: "json-zed",
      configPath: path.join(h(), ".config", "zed", "settings.json"),
      mcpPath: path.join(h(), ".config", "zed", "settings.json"),
      mcpFmt: "mcp-json-zed",
      detectPaths: [
        path.join(h(), ".config", "zed"),
        path.join(h(), "Library", "Application Support", "Zed"),
      ],
      detectBinaries: ["zed", "zeditor"],
      alwaysOffer: false,
      projectionDir: null,
      instructionFilePath: null,
      skillSurface: null,
    },
  ],
  [
    "openclaw",
    {
      name: "openclaw",
      description: "OpenClaw (open-source AI assistant)",
      configFmt: "json-openclaw-hooks",
      configPath: path.join(h(), ".openclaw", "openclaw.json"),
      mcpPath: path.join(h(), ".openclaw", "openclaw.json"),
      mcpFmt: "mcp-json-servers",
      detectPaths: [path.join(h(), ".openclaw")],
      detectBinaries: ["openclaw"],
      alwaysOffer: false,
      projectionDir: null,
      instructionFilePath: null,
      skillSurface: null,
    },
  ],
  [
    "nanoclaw",
    {
      name: "nanoclaw",
      description: "NanoClaw (compact claw variant)",
      configFmt: "json-openclaw-hooks",
      configPath: path.join(h(), ".nanoclaw", "nanoclaw.json"),
      mcpPath: path.join(h(), ".nanoclaw", "nanoclaw.json"),
      mcpFmt: "mcp-json-servers",
      detectPaths: [path.join(h(), ".nanoclaw")],
      detectBinaries: ["nanoclaw"],
      alwaysOffer: false,
      projectionDir: null,
      instructionFilePath: null,
      skillSurface: null,
    },
  ],
  [
    "nemoclaw",
    {
      name: "nemoclaw",
      description: "NemoClaw (memory-focused claw variant)",
      configFmt: "json-openclaw-hooks",
      configPath: path.join(h(), ".nemoclaw", "nemoclaw.json"),
      mcpPath: path.join(h(), ".nemoclaw", "nemoclaw.json"),
      mcpFmt: "mcp-json-servers",
      detectPaths: [path.join(h(), ".nemoclaw")],
      detectBinaries: ["nemoclaw"],
      alwaysOffer: false,
      projectionDir: null,
      instructionFilePath: null,
      skillSurface: null,
    },
  ],
  [
    "aider",
    {
      name: "aider",
      description: "aider CLI (paul-gauthier)",
      configFmt: "yaml-aider",
      configPath: ".aider.conf.yml", // workspace-relative
      mcpPath: null,
      mcpFmt: null,
      detectPaths: [],
      detectBinaries: ["aider"],
      alwaysOffer: false,
      projectionDir: null,
      instructionFilePath: ".aider.conf.yml",
      skillSurface: null,
    },
  ],
  [
    "copilot",
    {
      name: "copilot",
      description: "GitHub Copilot (workspace instructions)",
      configFmt: "text-block",
      configPath: path.join(".github", "copilot-instructions.md"), // workspace-relative
      mcpPath: null,
      mcpFmt: null,
      detectPaths: [],
      detectBinaries: [],
      alwaysOffer: true,
      projectionDir: null,
      instructionFilePath: path.join(".github", "copilot-instructions.md"),
      skillSurface: null,
    },
  ],
  [
    "cody",
    {
      name: "cody",
      description: "Sourcegraph Cody",
      configFmt: "json-generic",
      configPath: path.join(".cody", "config.json"), // workspace-relative
      mcpPath: null,
      mcpFmt: null,
      detectPaths: [path.join(h(), ".config", "cody")],
      detectBinaries: ["cody"],
      alwaysOffer: false,
      projectionDir: null,
      instructionFilePath: path.join(".cody", "config.json"),
      skillSurface: null,
    },
  ],
  [
    "qodo",
    {
      name: "qodo",
      description: "Qodo Gen (formerly CodiumAI)",
      configFmt: "text-block",
      configPath: path.join(".codium", "ai-rules.md"), // workspace-relative
      mcpPath: null,
      mcpFmt: null,
      detectPaths: [path.join(h(), ".codium"), path.join(h(), ".qodo")],
      detectBinaries: [],
      alwaysOffer: false,
      projectionDir: null,
      instructionFilePath: path.join(".codium", "ai-rules.md"),
      skillSurface: null,
    },
  ],
]);

/** All 17 client names, ordered as in the registry. */
export const ALL_CLIENT_NAMES: readonly string[] = [
  ...AGENT_REGISTRY.keys(),
];

/**
 * Looks up a spec by name. Throws InstallerError(UNKNOWN_CLIENT) if not found.
 * Callers that import this should handle the error.
 */
export function requireSpec(name: string): AgentSpec {
  const spec = AGENT_REGISTRY.get(name);
  if (spec === undefined) {
    throw new Error(`Unknown client: ${name}`);
  }
  return spec;
}
