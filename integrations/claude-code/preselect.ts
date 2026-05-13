// mind-nerve Claude Code UserPromptSubmit hook. Projector pattern (D1).

import fs from "node:fs/promises";
import path from "node:path";
import { z } from "zod";
import { logger } from "./logger.js";
import { loadConfig, type Config } from "./config.js";
import { buildCatalog, buildRegistrySummary, type SkillEntry } from "./catalog.js";
import {
  rewriteProjection,
  rewriteProjectionPassthrough,
  ensureRuntimeDir,
} from "./projector.js";
import {
  isBinaryAvailable,
  callMindNerve,
  BinaryNotFoundError,
  SubprocessTimeoutError,
  type SubprocessInput,
} from "./subprocess.js";

// ---------------------------------------------------------------------------
// Hook contract types
// ---------------------------------------------------------------------------

// Claude Code passes this shape as JSON on stdin to UserPromptSubmit hooks.
const ClaudeHookInputSchema = z.object({
  user_prompt: z.string(),
  transcript_path: z.string().optional(),
});

type ClaudeHookInput = z.infer<typeof ClaudeHookInputSchema>;

// All hooks return {} — we communicate via the projection dir, not hook reply.
type HookResult = Record<string, never>;

// ---------------------------------------------------------------------------
// Cache helpers
// ---------------------------------------------------------------------------

interface ProjectorCache {
  catalogHash: string;
}

async function readCatalogHashCache(
  cacheFile: string,
): Promise<string | null> {
  try {
    const raw = await fs.readFile(cacheFile, "utf8");
    const parsed: unknown = JSON.parse(raw);
    if (
      typeof parsed === "object" &&
      parsed !== null &&
      "catalogHash" in parsed &&
      typeof (parsed as ProjectorCache).catalogHash === "string"
    ) {
      return (parsed as ProjectorCache).catalogHash;
    }
  } catch {
    // Missing or malformed cache — treat as cold start.
  }
  return null;
}

async function writeCatalogHashCache(
  cacheFile: string,
  hash: string,
): Promise<void> {
  try {
    await fs.writeFile(cacheFile, JSON.stringify({ catalogHash: hash }), "utf8");
  } catch {
    // Non-fatal — just means next turn won't benefit from the cache.
  }
}

// ---------------------------------------------------------------------------
// Main hook logic
// ---------------------------------------------------------------------------

/**
 * Selects which skills to project for this turn.
 *
 * Returns all-skills entries for passthrough modes, filtered entries for top_k.
 * On outcome=error, returns null (caller keeps current projection intact).
 */
async function selectEntries(
  input: ClaudeHookInput,
  config: Config,
  allEntries: SkillEntry[],
  catalogHash: string,
): Promise<SkillEntry[] | null> {
  // Binary check.
  if (!isBinaryAvailable(config.binaryPath)) {
    logger.warn(
      { binaryPath: config.binaryPath },
      "mind-nerve binary not available — passthrough",
    );
    return allEntries;
  }

  const subprocessInput: SubprocessInput = {
    current_prompt: input.user_prompt,
    registry_summary: buildRegistrySummary({
      hash: catalogHash,
      entries: allEntries,
    }),
    catalog_hash: catalogHash,
    k: config.topK,
    threshold: config.confidenceThreshold,
  };

  try {
    const reply = await callMindNerve(
      config.binaryPath,
      subprocessInput,
      config.timeoutMs,
    );

    switch (reply.outcome) {
      case "top_k": {
        // Build a lookup from id → entry for O(1) membership test.
        const entryById = new Map(allEntries.map((e) => [e.id, e]));
        const selected: SkillEntry[] = [];
        for (const id of reply.selected) {
          const entry = entryById.get(id);
          if (entry !== undefined) {
            selected.push(entry);
          } else {
            logger.warn(
              { id },
              "mind-nerve returned unknown skill id — skipping",
            );
          }
        }
        logger.info(
          { selected: selected.map((e) => e.id), total: allEntries.length },
          "top_k projection written",
        );
        return selected;
      }

      case "low_confidence":
        logger.info(
          { reason: reply.reason ?? "unspecified" },
          "low_confidence — passthrough",
        );
        return allEntries;

      case "passthrough":
        logger.info(
          { reason: reply.reason ?? "unspecified" },
          "passthrough outcome",
        );
        return allEntries;

      case "error":
        logger.warn(
          { message: reply.message ?? "unspecified" },
          "mind-nerve returned error outcome — keeping current projection",
        );
        return null;
    }
  } catch (err) {
    if (err instanceof BinaryNotFoundError) {
      logger.warn({ binaryPath: config.binaryPath }, "binary not found — passthrough");
      return allEntries;
    }
    if (err instanceof SubprocessTimeoutError) {
      logger.warn({ timeoutMs: config.timeoutMs }, "subprocess timed out — passthrough");
      return allEntries;
    }
    // Unknown error (JSON parse failure, ZodError, etc.) — passthrough.
    logger.warn(
      { err: err instanceof Error ? err.message : String(err) },
      "mind-nerve call failed — passthrough",
    );
    return allEntries;
  }
}

// ---------------------------------------------------------------------------
// Public entrypoint (exported for testing with config injection)
// ---------------------------------------------------------------------------

/**
 * Processes a Claude Code UserPromptSubmit hook call.
 *
 * @param stdinJSON  Raw JSON string from Claude Code (hook stdin).
 * @param configOverride  For testing: skip file/env config loading.
 */
export async function userPromptSubmitHook(
  stdinJSON: string,
  configOverride?: Config,
): Promise<HookResult> {
  // 1. Parse stdin. Fail-open on parse error.
  const parseResult = ClaudeHookInputSchema.safeParse(
    parseJsonSafe(stdinJSON),
  );
  if (!parseResult.success) {
    logger.warn(
      { issues: parseResult.error.issues },
      "stdin parse failed — no-op",
    );
    return {};
  }
  const hookInput = parseResult.data;

  // 2. Load config (or use override in tests).
  let config: Config;
  try {
    config = configOverride ?? (await loadConfig());
  } catch {
    logger.warn("config load failed — no-op");
    return {};
  }

  // 3. Ensure ~/.mind-nerve/ exists.
  try {
    await ensureRuntimeDir();
  } catch {
    logger.warn("could not create runtime dir — no-op");
    return {};
  }

  // 4. Build skill catalog.
  let catalog;
  try {
    catalog = await buildCatalog(config.skillsDir);
  } catch {
    logger.warn({ skillsDir: config.skillsDir }, "catalog build failed — no-op");
    return {};
  }

  // 5. Cache invalidation check. (Phase 1: always rebuild; cache used for
  //    diagnostic logging only. Phase 2: skip subprocess if hash unchanged.)
  const cachedHash = await readCatalogHashCache(config.cacheFile);
  if (cachedHash !== catalog.hash) {
    logger.info(
      { prev: cachedHash ?? "none", curr: catalog.hash },
      "catalog hash changed",
    );
    await writeCatalogHashCache(config.cacheFile, catalog.hash);
  }

  // 6. Decide which entries to project.
  let selectedEntries: SkillEntry[] | null;
  try {
    selectedEntries = await selectEntries(
      hookInput,
      config,
      catalog.entries,
      catalog.hash,
    );
  } catch {
    // selectEntries swallows errors internally, but guard the call site too.
    logger.warn("selectEntries threw unexpectedly — no-op");
    return {};
  }

  // 7. Rewrite projection. null means keep current (error outcome from nerve).
  if (selectedEntries !== null) {
    try {
      await rewriteProjection(config.projectedDir, selectedEntries);
    } catch {
      // Projection rewrite failed. Try passthrough as best-effort recovery.
      logger.warn("projection rewrite failed — attempting passthrough");
      try {
        await rewriteProjectionPassthrough(
          config.projectedDir,
          catalog.entries,
        );
      } catch {
        logger.warn("passthrough fallback also failed — projection may be stale");
      }
    }
  }

  // 8. Always return empty object — hook reply carries no payload.
  return {};
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function parseJsonSafe(raw: string): unknown {
  try {
    return JSON.parse(raw);
  } catch {
    return undefined;
  }
}

// ---------------------------------------------------------------------------
// CLI entrypoint (when run directly as a hook script)
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  let stdinData = "";
  process.stdin.setEncoding("utf8");

  await new Promise<void>((resolve) => {
    process.stdin.on("data", (chunk: string) => (stdinData += chunk));
    process.stdin.on("end", resolve);
  });

  try {
    const result = await userPromptSubmitHook(stdinData);
    process.stdout.write(JSON.stringify(result) + "\n");
  } catch {
    // Absolute last-resort fail-open: log to stderr (structured), exit 0.
    process.stderr.write(
      JSON.stringify({ level: "error", msg: "hook fatal — fail-open" }) + "\n",
    );
    process.stdout.write("{}\n");
  }
}

// Run when invoked directly (e.g., `node preselect.js`).
// ESM: detect if this file is the entrypoint.
const isMain =
  process.argv[1] !== undefined &&
  (await import("node:url")).pathToFileURL(process.argv[1]).href ===
    import.meta.url;

if (isMain) {
  main().catch(() => {
    process.stdout.write("{}\n");
    process.exit(0); // Never exit non-zero — Claude Code must not be blocked.
  });
}
