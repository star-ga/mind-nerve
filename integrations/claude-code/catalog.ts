// mind-nerve Claude Code UserPromptSubmit hook. Projector pattern (D1).

import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";

export interface SkillEntry {
  /** Skill ID: the directory name under skillsDir. */
  id: string;
  /** Absolute path to the SKILL.md file. */
  skillPath: string;
  /** First 200 bytes of SKILL.md — used in registry summary sent to mind-nerve. */
  excerpt: string;
}

export interface SkillCatalog {
  /** SHA-256 hex digest over the sorted (id, excerptHash) pairs. */
  hash: string;
  entries: SkillEntry[];
}

// Reads the first `limit` bytes of a file as UTF-8, gracefully truncating.
async function readExcerpt(filePath: string, limit: number): Promise<string> {
  let fd: fs.FileHandle | undefined;
  try {
    fd = await fs.open(filePath, "r");
    const buf = Buffer.alloc(limit);
    const { bytesRead } = await fd.read(buf, 0, limit, 0);
    return buf.slice(0, bytesRead).toString("utf8");
  } catch {
    return "";
  } finally {
    await fd?.close();
  }
}

/**
 * Scans skillsDir for subdirectories that contain SKILL.md and returns a
 * deterministic SkillCatalog. The catalog hash is SHA-256 over a canonical
 * serialisation of (skill_id, sha256(excerpt)) pairs, sorted by skill_id.
 *
 * Entries without SKILL.md are silently skipped (fail-open per D4).
 */
export async function buildCatalog(skillsDir: string): Promise<SkillCatalog> {
  let dirEntries: string[];
  try {
    const raw = await fs.readdir(skillsDir, { withFileTypes: true });
    dirEntries = raw
      .filter((e) => e.isDirectory() || e.isSymbolicLink())
      .map((e) => e.name)
      .sort(); // deterministic order
  } catch {
    return { hash: emptyHash(), entries: [] };
  }

  const entries: SkillEntry[] = [];
  for (const name of dirEntries) {
    const skillMdPath = path.join(skillsDir, name, "SKILL.md");
    try {
      await fs.access(skillMdPath);
    } catch {
      continue; // no SKILL.md — skip
    }
    const excerpt = await readExcerpt(skillMdPath, 200);
    entries.push({ id: name, skillPath: skillMdPath, excerpt });
  }

  const hash = computeCatalogHash(entries);
  return { hash, entries };
}

function computeCatalogHash(entries: SkillEntry[]): string {
  if (entries.length === 0) return emptyHash();

  const hasher = crypto.createHash("sha256");
  for (const e of entries) {
    const excerptHash = crypto
      .createHash("sha256")
      .update(e.excerpt, "utf8")
      .digest("hex");
    hasher.update(`${e.id}\0${excerptHash}\n`);
  }
  return hasher.digest("hex");
}

function emptyHash(): string {
  return crypto.createHash("sha256").update("").digest("hex");
}

/** Returns a compact registry summary for the mind-nerve subprocess payload. */
export function buildRegistrySummary(
  catalog: SkillCatalog,
): Array<{ id: string; excerpt: string }> {
  return catalog.entries.map((e) => ({ id: e.id, excerpt: e.excerpt }));
}
