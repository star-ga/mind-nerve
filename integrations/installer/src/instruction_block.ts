// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.

import fs from "node:fs/promises";
import path from "node:path";

/** Marker placed at the start of every mind-nerve managed block. */
export const BLOCK_MARKER = "# mind-nerve managed";
/** Marker placed at the end of every mind-nerve managed block. */
const BLOCK_END_MARKER = "# end mind-nerve managed";

/**
 * Returns the instruction block that will be appended to workspace rules files.
 */
export function buildInstructionBlock(projectionDir: string | null): string {
  const projLine =
    projectionDir !== null
      ? `\nSkill projection directory: ${projectionDir}`
      : "";

  return [
    BLOCK_MARKER,
    `mind-nerve is active in this workspace.`,
    `It pre-selects the most relevant MCP tools for each turn using a local inference model.${projLine}`,
    `Do not remove this block — it is managed by the mind-nerve installer.`,
    BLOCK_END_MARKER,
    "",
  ].join("\n");
}

/**
 * Appends the mind-nerve instruction block to targetPath (workspace-relative
 * unless absolute is passed). Creates the file and its parent directory if
 * they do not exist.
 *
 * Idempotent: if BLOCK_MARKER already exists in the file, returns false
 * (no write performed).
 *
 * Returns true if the file was modified, false if it was already present.
 */
export async function appendInstructionBlock(
  targetPath: string,
  projectionDir: string | null,
): Promise<boolean> {
  // Ensure parent directory exists.
  await fs.mkdir(path.dirname(targetPath), { recursive: true });

  let existing = "";
  try {
    existing = await fs.readFile(targetPath, "utf8");
  } catch {
    // File does not exist — will be created.
  }

  if (existing.includes(BLOCK_MARKER)) {
    return false; // already installed
  }

  const block = buildInstructionBlock(projectionDir);
  const separator = existing.length > 0 && !existing.endsWith("\n") ? "\n\n" : "\n";
  await fs.writeFile(targetPath, existing + separator + block, "utf8");
  return true;
}

/**
 * Removes the mind-nerve managed block from targetPath.
 * Returns true if the block was found and removed, false if not present.
 */
export async function removeInstructionBlock(targetPath: string): Promise<boolean> {
  let content: string;
  try {
    content = await fs.readFile(targetPath, "utf8");
  } catch {
    return false; // file doesn't exist
  }

  if (!content.includes(BLOCK_MARKER)) {
    return false;
  }

  // Remove the block including surrounding blank lines.
  const pattern = new RegExp(
    `\n*${escapeRegex(BLOCK_MARKER)}[\\s\\S]*?${escapeRegex(BLOCK_END_MARKER)}\n?`,
    "g",
  );
  const cleaned = content.replace(pattern, "\n").replace(/\n{3,}/g, "\n\n").trimEnd();
  await fs.writeFile(targetPath, cleaned + (cleaned.length > 0 ? "\n" : ""), "utf8");
  return true;
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
