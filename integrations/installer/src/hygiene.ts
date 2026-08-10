// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.
//
// Part 3 (ROUTE-TABLE HYGIENE): keep route_table.jsonl and its ROW-ALIGNED
// route_table.npy embeddings in lockstep.
//
// Dead routes poison ranking: a route whose SKILL.md no longer exists still
// occupies an embedding row and can out-rank a live skill (a phantom `xlsx`
// entry beating `spreadsheet` was the observed case). Pruning is therefore
// necessary — but the .npy is row-aligned to the JSONL, so pruning one without
// the other shifts every subsequent embedding onto the WRONG route and
// silently corrupts all later ranking. That failure is invisible: no error, no
// crash, just quietly worse routing forever.
//
// This module therefore treats row-count agreement as a hard precondition AND
// a hard postcondition, and refuses to write if either fails.

import fs from "node:fs/promises";
import path from "node:path";
import { parseNpy, serializeNpy, selectRows, type NpyArray } from "./npy.js";
import { InstallerError } from "./errors.js";

const CONTEXT = "route-table";

export interface RouteRow {
  /** The parsed JSON object. */
  readonly obj: Record<string, unknown>;
  /** The original line text, so unchanged rows round-trip byte-for-byte. */
  readonly raw: string;
}

export interface HygieneOptions {
  /** Path to route_table.jsonl. */
  readonly tablePath: string;
  /** Path to the row-aligned route_table.npy. Null skips embedding work. */
  readonly embeddingsPath: string | null;
  /** Path rewrites applied to `source_path`, e.g. after a hub rename. */
  readonly repoint?: ReadonlyArray<readonly [string, string]>;
  /** Drop routes whose `source_path` no longer exists. */
  readonly pruneDead?: boolean;
  /** Report only — compute the plan, write nothing. */
  readonly dryRun?: boolean;
  /** Existence probe override (tests). */
  readonly exists?: (p: string) => Promise<boolean>;
  /** Clock override (tests) — supplies the backup suffix. */
  readonly now?: number;
}

export interface HygieneReport {
  readonly rowsBefore: number;
  readonly rowsAfter: number;
  /** Routes whose `source_path` was rewritten. */
  readonly repointed: number;
  /** Routes dropped because their SKILL.md is gone. */
  readonly pruned: number;
  /** Names of pruned routes, for the operator to eyeball. */
  readonly prunedNames: readonly string[];
  /** Embedding rows before/after — must track rowsBefore/rowsAfter exactly. */
  readonly embeddingRowsBefore: number | null;
  readonly embeddingRowsAfter: number | null;
  readonly wrote: boolean;
  readonly backups: readonly string[];
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/**
 * Parses route_table.jsonl. A malformed line is fatal, not skippable: skipping
 * it would silently shift the row alignment against the embeddings.
 */
export function parseRouteTable(text: string): RouteRow[] {
  const rows: RouteRow[] = [];
  const lines = text.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i] as string;
    if (raw.trim().length === 0) continue;
    let obj: unknown;
    try {
      obj = JSON.parse(raw);
    } catch (err) {
      throw new InstallerError(
        "INVALID_CONFIG_FORMAT",
        CONTEXT,
        `route_table.jsonl line ${i + 1} is not valid JSON — refusing to ` +
          `continue, because skipping it would break row alignment: ${String(err)}`,
      );
    }
    if (typeof obj !== "object" || obj === null || Array.isArray(obj)) {
      throw new InstallerError(
        "INVALID_CONFIG_FORMAT",
        CONTEXT,
        `route_table.jsonl line ${i + 1} is not a JSON object`,
      );
    }
    rows.push({ obj: obj as Record<string, unknown>, raw });
  }
  return rows;
}

export function serializeRouteTable(rows: readonly RouteRow[]): string {
  return rows.length === 0 ? "" : rows.map((r) => r.raw).join("\n") + "\n";
}

/** Reads `source_path` if present and a string. */
export function sourcePathOf(row: RouteRow): string | null {
  const v = row.obj["source_path"];
  return typeof v === "string" ? v : null;
}

function nameOf(row: RouteRow): string {
  const v = row.obj["name"];
  return typeof v === "string" ? v : "<unnamed>";
}

/** Applies prefix rewrites to a row, returning a new row (never mutates). */
export function repointRow(
  row: RouteRow,
  rules: ReadonlyArray<readonly [string, string]>,
): { readonly row: RouteRow; readonly changed: boolean } {
  const sp = sourcePathOf(row);
  if (sp === null) return { row, changed: false };

  let next = sp;
  for (const [from, to] of rules) {
    if (next.startsWith(from)) {
      next = to + next.slice(from.length);
    }
  }
  if (next === sp) return { row, changed: false };

  const obj = { ...row.obj, source_path: next };
  return { row: { obj, raw: JSON.stringify(obj) }, changed: true };
}

// ---------------------------------------------------------------------------
// The hygiene pass
// ---------------------------------------------------------------------------

/**
 * Repoints, prunes, and rewrites the route table plus its embeddings.
 *
 * Refuses to write when the JSONL row count and the .npy row count disagree —
 * before OR after. Disagreement means the two files are already out of sync,
 * and any prune computed against one would scramble the other.
 */
export async function runHygiene(opts: HygieneOptions): Promise<HygieneReport> {
  const exists = opts.exists ?? defaultExists;
  const now = opts.now ?? Date.now();

  const text = await fs.readFile(opts.tablePath, "utf8");
  const rows = parseRouteTable(text);
  const rowsBefore = rows.length;

  // ---- Precondition: the two files must already agree. -------------------
  let embeddings: NpyArray | null = null;
  if (opts.embeddingsPath !== null) {
    const buf = await fs.readFile(opts.embeddingsPath);
    embeddings = parseNpy(buf, CONTEXT);
    if (embeddings.rows !== rowsBefore) {
      throw new InstallerError(
        "INVALID_CONFIG_FORMAT",
        CONTEXT,
        `Row-count mismatch BEFORE hygiene: ${opts.tablePath} has ${rowsBefore} ` +
          `routes but ${opts.embeddingsPath} has ${embeddings.rows} embedding rows. ` +
          `These files are row-aligned; refusing to touch either. Re-learn the ` +
          `route table to regenerate both together.`,
      );
    }
  }

  // ---- Repoint --------------------------------------------------------------
  const repointRules = opts.repoint ?? [];
  let repointed = 0;
  const afterRepoint: RouteRow[] = rows.map((r) => {
    if (repointRules.length === 0) return r;
    const res = repointRow(r, repointRules);
    if (res.changed) repointed++;
    return res.row;
  });

  // ---- Prune ----------------------------------------------------------------
  const keepFlags: boolean[] = new Array<boolean>(afterRepoint.length).fill(true);
  const prunedNames: string[] = [];

  if (opts.pruneDead === true) {
    for (let i = 0; i < afterRepoint.length; i++) {
      const row = afterRepoint[i] as RouteRow;
      const sp = sourcePathOf(row);
      // A route with no source_path cannot be proven dead — keep it.
      if (sp === null) continue;
      if (!(await exists(sp))) {
        keepFlags[i] = false;
        prunedNames.push(nameOf(row));
      }
    }
  }

  const keptRows = afterRepoint.filter((_, i) => keepFlags[i] === true);
  const rowsAfter = keptRows.length;
  const pruned = rowsBefore - rowsAfter;

  // ---- Prune the embeddings in LOCKSTEP -------------------------------------
  let nextEmbeddings: NpyArray | null = null;
  if (embeddings !== null) {
    nextEmbeddings = selectRows(embeddings, (i) => keepFlags[i] === true);

    // ---- Postcondition: the result must still agree. ----------------------
    if (nextEmbeddings.rows !== rowsAfter) {
      throw new InstallerError(
        "INVALID_CONFIG_FORMAT",
        CONTEXT,
        `Row-count mismatch AFTER prune: ${rowsAfter} routes vs ` +
          `${nextEmbeddings.rows} embedding rows. Refusing to write.`,
      );
    }
  }

  const report: Omit<HygieneReport, "wrote" | "backups"> = {
    rowsBefore,
    rowsAfter,
    repointed,
    pruned,
    prunedNames,
    embeddingRowsBefore: embeddings?.rows ?? null,
    embeddingRowsAfter: nextEmbeddings?.rows ?? null,
  };

  const nothingToDo = repointed === 0 && pruned === 0;
  if (opts.dryRun === true || nothingToDo) {
    return { ...report, wrote: false, backups: [] };
  }

  // ---- Write, backing both files up first ----------------------------------
  const backups: string[] = [];

  const tableBak = `${opts.tablePath}.bak-mind-nerve-${now}`;
  await fs.copyFile(opts.tablePath, tableBak);
  backups.push(tableBak);

  if (opts.embeddingsPath !== null && nextEmbeddings !== null) {
    const embBak = `${opts.embeddingsPath}.bak-mind-nerve-${now}`;
    await fs.copyFile(opts.embeddingsPath, embBak);
    backups.push(embBak);
    await writeAtomic(opts.embeddingsPath, serializeNpy(nextEmbeddings));
  }

  await writeAtomic(opts.tablePath, Buffer.from(serializeRouteTable(keptRows), "utf8"));

  return { ...report, wrote: true, backups };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function defaultExists(p: string): Promise<boolean> {
  try {
    await fs.access(p);
    return true;
  } catch {
    return false;
  }
}

/**
 * Writes via a temp file + rename. A torn route table is worse than a stale
 * one: the daemon would load a truncated catalog without complaining.
 */
async function writeAtomic(filePath: string, data: Buffer): Promise<void> {
  const tmp = path.join(
    path.dirname(filePath),
    `.${path.basename(filePath)}.tmp-${process.pid}`,
  );
  await fs.writeFile(tmp, data);
  await fs.rename(tmp, filePath);
}
