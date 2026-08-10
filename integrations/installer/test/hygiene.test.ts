// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.
//
// Part 3 tests. The invariant under test is row alignment: route_table.jsonl
// line N corresponds to route_table.npy row N. Breaking it produces no error
// and no crash — just permanently, silently wrong ranking. So the code must
// refuse rather than guess.

import { describe, it, expect, afterEach } from "vitest";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import {
  runHygiene,
  parseRouteTable,
  serializeRouteTable,
  repointRow,
  sourcePathOf,
} from "../src/hygiene.js";
import { parseNpy, serializeNpy, type NpyArray } from "../src/npy.js";
import { InstallerError } from "../src/errors.js";

const tmpDirs: string[] = [];

async function makeTmp(): Promise<string> {
  const d = await fs.mkdtemp(path.join(os.tmpdir(), "mn-hyg-"));
  tmpDirs.push(d);
  return d;
}

afterEach(async () => {
  while (tmpDirs.length > 0) {
    const d = tmpDirs.pop();
    if (d !== undefined) await fs.rm(d, { recursive: true, force: true });
  }
});

interface Fixture {
  readonly dir: string;
  readonly tablePath: string;
  readonly embeddingsPath: string;
}

/** Writes a route table whose row i has a distinguishable embedding value. */
async function makeFixture(
  rows: ReadonlyArray<{ name: string; source_path: string }>,
  embeddingRows = rows.length,
): Promise<Fixture> {
  const dir = await makeTmp();
  const tablePath = path.join(dir, "route_table.jsonl");
  const embeddingsPath = path.join(dir, "route_table.npy");

  await fs.writeFile(
    tablePath,
    rows.map((r) => JSON.stringify({ kind: "skill", ...r })).join("\n") + "\n",
  );

  const cols = 4;
  const data = Buffer.alloc(embeddingRows * cols * 4);
  for (let r = 0; r < embeddingRows; r++) {
    for (let c = 0; c < cols; c++) {
      // Row i is stamped with i so a shifted alignment is detectable.
      data.writeFloatLE(r * 10 + c, (r * cols + c) * 4);
    }
  }
  const arr: NpyArray = {
    descr: "<f4",
    rows: embeddingRows,
    cols,
    oneDimensional: false,
    data,
  };
  await fs.writeFile(embeddingsPath, serializeNpy(arr));
  return { dir, tablePath, embeddingsPath };
}

/** Existence probe: only the listed paths exist. */
function existsIn(live: readonly string[]) {
  const set = new Set(live);
  return (p: string) => Promise.resolve(set.has(p));
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

describe("parseRouteTable", () => {
  it("parses one object per non-empty line", () => {
    const rows = parseRouteTable('{"name":"a"}\n{"name":"b"}\n');
    expect(rows.length).toBe(2);
    expect(rows[0]!.obj["name"]).toBe("a");
  });

  it("REFUSES a malformed line instead of skipping it", () => {
    // Skipping would silently shift every subsequent row against the .npy.
    expect(() => parseRouteTable('{"name":"a"}\nnot json\n{"name":"c"}\n')).toThrow(
      /line 2/,
    );
  });

  it("rejects a non-object line", () => {
    expect(() => parseRouteTable('{"name":"a"}\n[1,2,3]\n')).toThrow(
      /not a JSON object/,
    );
  });

  it("round-trips unchanged rows byte-for-byte", () => {
    const text = '{"name":"a",  "x":1}\n{"name":"b"}\n';
    expect(serializeRouteTable(parseRouteTable(text))).toBe(text);
  });
});

describe("repointRow", () => {
  it("rewrites a matching prefix", () => {
    const [row] = parseRouteTable(
      '{"name":"a","source_path":"/old/hub/a/SKILL.md"}\n',
    );
    const res = repointRow(row!, [["/old/hub", "/new/hub"]]);
    expect(res.changed).toBe(true);
    expect(sourcePathOf(res.row)).toBe("/new/hub/a/SKILL.md");
  });

  it("leaves non-matching paths alone", () => {
    const [row] = parseRouteTable(
      '{"name":"a","source_path":"/other/a/SKILL.md"}\n',
    );
    expect(repointRow(row!, [["/old/hub", "/new/hub"]]).changed).toBe(false);
  });

  it("does not mutate the input row", () => {
    const [row] = parseRouteTable(
      '{"name":"a","source_path":"/old/hub/a/SKILL.md"}\n',
    );
    repointRow(row!, [["/old/hub", "/new/hub"]]);
    expect(sourcePathOf(row!)).toBe("/old/hub/a/SKILL.md");
  });
});

// ---------------------------------------------------------------------------
// Row-alignment refusal
// ---------------------------------------------------------------------------

describe("runHygiene row-alignment guard", () => {
  it("REFUSES when the .npy has more rows than the table", async () => {
    const fx = await makeFixture(
      [
        { name: "a", source_path: "/hub/a/SKILL.md" },
        { name: "b", source_path: "/hub/b/SKILL.md" },
      ],
      5,
    );
    await expect(
      runHygiene({
        tablePath: fx.tablePath,
        embeddingsPath: fx.embeddingsPath,
        pruneDead: true,
        exists: existsIn([]),
      }),
    ).rejects.toThrow(/Row-count mismatch BEFORE hygiene/);
  });

  it("REFUSES when the .npy has fewer rows than the table", async () => {
    const fx = await makeFixture(
      [
        { name: "a", source_path: "/hub/a/SKILL.md" },
        { name: "b", source_path: "/hub/b/SKILL.md" },
        { name: "c", source_path: "/hub/c/SKILL.md" },
      ],
      2,
    );
    await expect(
      runHygiene({
        tablePath: fx.tablePath,
        embeddingsPath: fx.embeddingsPath,
        pruneDead: true,
        exists: existsIn([]),
      }),
    ).rejects.toThrow(InstallerError);
  });

  it("writes NOTHING when it refuses", async () => {
    const fx = await makeFixture(
      [{ name: "a", source_path: "/hub/a/SKILL.md" }],
      3,
    );
    const tableBefore = await fs.readFile(fx.tablePath, "utf8");
    const embBefore = await fs.readFile(fx.embeddingsPath);

    await expect(
      runHygiene({
        tablePath: fx.tablePath,
        embeddingsPath: fx.embeddingsPath,
        pruneDead: true,
        exists: existsIn([]),
      }),
    ).rejects.toThrow();

    expect(await fs.readFile(fx.tablePath, "utf8")).toBe(tableBefore);
    expect((await fs.readFile(fx.embeddingsPath)).equals(embBefore)).toBe(true);
    expect((await fs.readdir(fx.dir)).filter((f) => f.includes(".bak-"))).toEqual(
      [],
    );
  });
});

// ---------------------------------------------------------------------------
// Pruning in lockstep
// ---------------------------------------------------------------------------

describe("runHygiene prune", () => {
  it("drops dead routes and their embedding rows together", async () => {
    const fx = await makeFixture([
      { name: "alive0", source_path: "/hub/alive0/SKILL.md" },
      { name: "dead1", source_path: "/tmp/gone/dead1/SKILL.md" },
      { name: "alive2", source_path: "/hub/alive2/SKILL.md" },
      { name: "dead3", source_path: "/tmp/gone/dead3/SKILL.md" },
    ]);

    const report = await runHygiene({
      tablePath: fx.tablePath,
      embeddingsPath: fx.embeddingsPath,
      pruneDead: true,
      exists: existsIn(["/hub/alive0/SKILL.md", "/hub/alive2/SKILL.md"]),
      now: 1_700_000_000_000,
    });

    expect(report.rowsBefore).toBe(4);
    expect(report.rowsAfter).toBe(2);
    expect(report.pruned).toBe(2);
    expect(report.prunedNames).toEqual(["dead1", "dead3"]);
    expect(report.embeddingRowsAfter).toBe(2);
    expect(report.wrote).toBe(true);
  });

  it("keeps the SURVIVING rows' embeddings — not just the right count", async () => {
    const fx = await makeFixture([
      { name: "alive0", source_path: "/hub/alive0/SKILL.md" },
      { name: "dead1", source_path: "/tmp/gone/dead1/SKILL.md" },
      { name: "alive2", source_path: "/hub/alive2/SKILL.md" },
    ]);

    await runHygiene({
      tablePath: fx.tablePath,
      embeddingsPath: fx.embeddingsPath,
      pruneDead: true,
      exists: existsIn(["/hub/alive0/SKILL.md", "/hub/alive2/SKILL.md"]),
    });

    const arr = parseNpy(await fs.readFile(fx.embeddingsPath), "test");
    // Row 0 kept its stamp (0), and the second row must carry row 2's stamp
    // (20) — not row 1's (10). A count-only check would pass either way.
    expect(arr.rows).toBe(2);
    expect(arr.data.readFloatLE(0)).toBe(0);
    expect(arr.data.readFloatLE(4 * 4)).toBe(20);
  });

  it("keeps the table and the embeddings the same length after writing", async () => {
    const fx = await makeFixture([
      { name: "a", source_path: "/hub/a/SKILL.md" },
      { name: "b", source_path: "/gone/b/SKILL.md" },
      { name: "c", source_path: "/hub/c/SKILL.md" },
    ]);

    await runHygiene({
      tablePath: fx.tablePath,
      embeddingsPath: fx.embeddingsPath,
      pruneDead: true,
      exists: existsIn(["/hub/a/SKILL.md", "/hub/c/SKILL.md"]),
    });

    const lines = (await fs.readFile(fx.tablePath, "utf8"))
      .split("\n")
      .filter((l) => l.trim().length > 0);
    const arr = parseNpy(await fs.readFile(fx.embeddingsPath), "test");
    expect(lines.length).toBe(arr.rows);
  });

  it("keeps routes that have no source_path — they cannot be proven dead", async () => {
    const dir = await makeTmp();
    const tablePath = path.join(dir, "route_table.jsonl");
    await fs.writeFile(
      tablePath,
      '{"name":"no-path"}\n{"name":"dead","source_path":"/gone/x"}\n',
    );
    const report = await runHygiene({
      tablePath,
      embeddingsPath: null,
      pruneDead: true,
      exists: existsIn([]),
    });
    expect(report.rowsAfter).toBe(1);
    expect(report.prunedNames).toEqual(["dead"]);
  });

  it("backs both files up before writing", async () => {
    const fx = await makeFixture([
      { name: "a", source_path: "/hub/a/SKILL.md" },
      { name: "dead", source_path: "/gone/d/SKILL.md" },
    ]);
    const report = await runHygiene({
      tablePath: fx.tablePath,
      embeddingsPath: fx.embeddingsPath,
      pruneDead: true,
      exists: existsIn(["/hub/a/SKILL.md"]),
      now: 42,
    });
    expect(report.backups).toEqual([
      `${fx.tablePath}.bak-mind-nerve-42`,
      `${fx.embeddingsPath}.bak-mind-nerve-42`,
    ]);
    const restored = parseNpy(
      await fs.readFile(`${fx.embeddingsPath}.bak-mind-nerve-42`),
      "test",
    );
    expect(restored.rows).toBe(2);
  });

  it("writes nothing when there is nothing to do", async () => {
    const fx = await makeFixture([
      { name: "a", source_path: "/hub/a/SKILL.md" },
      { name: "b", source_path: "/hub/b/SKILL.md" },
    ]);
    const report = await runHygiene({
      tablePath: fx.tablePath,
      embeddingsPath: fx.embeddingsPath,
      pruneDead: true,
      exists: existsIn(["/hub/a/SKILL.md", "/hub/b/SKILL.md"]),
    });
    expect(report.wrote).toBe(false);
    expect(report.backups).toEqual([]);
  });

  it("dry-run reports the plan without touching disk", async () => {
    const fx = await makeFixture([
      { name: "a", source_path: "/hub/a/SKILL.md" },
      { name: "dead", source_path: "/gone/d/SKILL.md" },
    ]);
    const before = await fs.readFile(fx.tablePath, "utf8");

    const report = await runHygiene({
      tablePath: fx.tablePath,
      embeddingsPath: fx.embeddingsPath,
      pruneDead: true,
      dryRun: true,
      exists: existsIn(["/hub/a/SKILL.md"]),
    });

    expect(report.pruned).toBe(1);
    expect(report.wrote).toBe(false);
    expect(await fs.readFile(fx.tablePath, "utf8")).toBe(before);
  });
});

// ---------------------------------------------------------------------------
// Repointing after a hub move
// ---------------------------------------------------------------------------

describe("runHygiene repoint", () => {
  it("rewrites source paths after a hub rename", async () => {
    const fx = await makeFixture([
      { name: "a", source_path: "/home/u/.agents/skills/a/SKILL.md" },
      { name: "b", source_path: "/home/u/.agents/skills/b/SKILL.md" },
    ]);

    const report = await runHygiene({
      tablePath: fx.tablePath,
      embeddingsPath: fx.embeddingsPath,
      repoint: [["/home/u/.agents/skills", "/home/u/.agents/skills-hub"]],
      pruneDead: false,
    });

    expect(report.repointed).toBe(2);
    expect(report.rowsAfter).toBe(2);
    const text = await fs.readFile(fx.tablePath, "utf8");
    expect(text).toContain("/home/u/.agents/skills-hub/a/SKILL.md");
    expect(text).not.toContain('"/home/u/.agents/skills/a/SKILL.md"');
  });

  it("repoints BEFORE deciding what is dead", async () => {
    // The whole point of repointing during a hub move: the old paths are gone,
    // so pruning first would delete the entire catalog.
    const fx = await makeFixture([
      { name: "a", source_path: "/old/hub/a/SKILL.md" },
      { name: "b", source_path: "/old/hub/b/SKILL.md" },
    ]);

    const report = await runHygiene({
      tablePath: fx.tablePath,
      embeddingsPath: fx.embeddingsPath,
      repoint: [["/old/hub", "/new/hub"]],
      pruneDead: true,
      exists: existsIn(["/new/hub/a/SKILL.md", "/new/hub/b/SKILL.md"]),
    });

    expect(report.repointed).toBe(2);
    expect(report.pruned).toBe(0);
    expect(report.rowsAfter).toBe(2);
  });

  it("repoints and prunes in a single pass, staying aligned", async () => {
    const fx = await makeFixture([
      { name: "a", source_path: "/old/hub/a/SKILL.md" },
      { name: "renamed", source_path: "/old/hub/renamed/SKILL.md" },
      { name: "c", source_path: "/old/hub/c/SKILL.md" },
    ]);

    const report = await runHygiene({
      tablePath: fx.tablePath,
      embeddingsPath: fx.embeddingsPath,
      repoint: [["/old/hub", "/new/hub"]],
      pruneDead: true,
      exists: existsIn(["/new/hub/a/SKILL.md", "/new/hub/c/SKILL.md"]),
    });

    expect(report.repointed).toBe(3);
    expect(report.pruned).toBe(1);
    expect(report.rowsAfter).toBe(2);
    expect(report.embeddingRowsAfter).toBe(2);

    const arr = parseNpy(await fs.readFile(fx.embeddingsPath), "test");
    expect(arr.data.readFloatLE(0)).toBe(0); // row 0
    expect(arr.data.readFloatLE(4 * 4)).toBe(20); // row 2, not row 1
  });
});
