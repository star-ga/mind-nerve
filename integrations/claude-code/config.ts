// mind-nerve Claude Code UserPromptSubmit hook. Projector pattern (D1).

import os from "node:os";
import path from "node:path";
import fs from "node:fs/promises";
import { z } from "zod";

const ConfigSchema = z.object({
  binaryPath: z.string().default("mind-nerve"),
  topK: z.number().int().min(1).max(64).default(5),
  timeoutMs: z.number().int().min(10).max(5000).default(150),
  skillsDir: z.string().default(path.join(os.homedir(), ".claude", "skills")),
  projectedDir: z
    .string()
    .default(path.join(os.homedir(), ".mind-nerve", "skills-projected")),
  cacheFile: z
    .string()
    .default(path.join(os.homedir(), ".mind-nerve", "projector.cache.json")),
  confidenceThreshold: z.number().min(0).max(1).default(0.5),
});

export type Config = z.infer<typeof ConfigSchema>;

// TOML parsing without a dependency: the config is small and well-structured.
// We load only the [hook] table; unknown tables are silently ignored.
// If TOML parsing fails (malformed config), we fall back to defaults + env.
function parseTomlHookTable(raw: string): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  let inHookTable = false;

  for (const rawLine of raw.split("\n")) {
    const line = rawLine.trim();
    if (line === "[hook]") {
      inHookTable = true;
      continue;
    }
    if (line.startsWith("[")) {
      inHookTable = false;
      continue;
    }
    if (!inHookTable || !line || line.startsWith("#")) continue;

    const eqIdx = line.indexOf("=");
    if (eqIdx === -1) continue;

    const key = line.slice(0, eqIdx).trim();
    const rawVal = line.slice(eqIdx + 1).trim();

    // Bare string (unquoted), quoted string, integer, float, boolean
    if (rawVal.startsWith('"') || rawVal.startsWith("'")) {
      result[key] = rawVal.slice(1, -1);
    } else if (rawVal === "true") {
      result[key] = true;
    } else if (rawVal === "false") {
      result[key] = false;
    } else if (/^-?\d+$/.test(rawVal)) {
      result[key] = parseInt(rawVal, 10);
    } else if (/^-?\d*\.\d+$/.test(rawVal)) {
      result[key] = parseFloat(rawVal);
    }
    // anything else: skip (complex TOML not needed here)
  }

  return result;
}

const CONFIG_PATH = path.join(
  os.homedir(),
  ".config",
  "mind-nerve",
  "config.toml",
);

export async function loadConfig(): Promise<Config> {
  // Environment variables take highest precedence.
  const fromEnv: Record<string, unknown> = {};
  if (process.env["MIND_NERVE_BIN"]) {
    fromEnv["binaryPath"] = process.env["MIND_NERVE_BIN"];
  }
  if (process.env["MIND_NERVE_TOP_K"]) {
    fromEnv["topK"] = parseInt(process.env["MIND_NERVE_TOP_K"] ?? "5", 10);
  }
  if (process.env["MIND_NERVE_TIMEOUT_MS"]) {
    fromEnv["timeoutMs"] = parseInt(
      process.env["MIND_NERVE_TIMEOUT_MS"] ?? "150",
      10,
    );
  }
  if (process.env["MIND_NERVE_SKILLS_DIR"]) {
    fromEnv["skillsDir"] = process.env["MIND_NERVE_SKILLS_DIR"];
  }
  if (process.env["MIND_NERVE_PROJECTED_DIR"]) {
    fromEnv["projectedDir"] = process.env["MIND_NERVE_PROJECTED_DIR"];
  }
  if (process.env["MIND_NERVE_THRESHOLD"]) {
    fromEnv["confidenceThreshold"] = parseFloat(
      process.env["MIND_NERVE_THRESHOLD"] ?? "0.5",
    );
  }

  // Load TOML config file if present; fall back silently if not.
  let fromFile: Record<string, unknown> = {};
  try {
    const raw = await fs.readFile(CONFIG_PATH, "utf8");
    fromFile = parseTomlHookTable(raw);
  } catch {
    // Config file is optional — default values suffice.
  }

  // Merge: env > file > defaults.
  const merged = { ...fromFile, ...fromEnv };

  // Parse with Zod — coerces types and fills defaults.
  const parsed = ConfigSchema.safeParse(merged);
  if (!parsed.success) {
    // On invalid config, return defaults. Fail-open per D4.
    return ConfigSchema.parse({});
  }
  return parsed.data;
}
