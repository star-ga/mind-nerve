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
}

function h(): string {
  return os.homedir();
}

/**
 * Build the projection directory path for a client that has a skill surface.
 */
function projDir(clientName: string): string {
  return path.join(h(), ".mind-nerve", "projections", clientName);
}

/**
 * AGENT_REGISTRY — 17 AI coding clients.
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
      detectPaths: [path.join(h(), ".vibe")],
      detectBinaries: ["vibe"],
      alwaysOffer: false,
      projectionDir: null,
      instructionFilePath: null,
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
