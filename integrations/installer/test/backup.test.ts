// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.

import { describe, it, expect, afterEach } from "vitest";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { createBackup, restoreLatestBackup, listBackups, backupPath } from "../src/backup.js";

let tmpDirs: string[] = [];

async function makeTmp(): Promise<string> {
  const d = await fs.mkdtemp(path.join(os.tmpdir(), "mn-backup-test-"));
  tmpDirs.push(d);
  return d;
}

afterEach(async () => {
  for (const d of tmpDirs) {
    await fs.rm(d, { recursive: true, force: true });
  }
  tmpDirs = [];
});

describe("backupPath", () => {
  it("produces a path with the mind-nerve timestamp suffix", () => {
    const p = backupPath("/home/user/.config/test.json", 1715000000000);
    expect(p).toBe("/home/user/.config/test.json.bak-mind-nerve-1715000000000");
  });
});

describe("createBackup", () => {
  it("creates a .bak copy of the source file", async () => {
    const tmp = await makeTmp();
    const src = path.join(tmp, "config.json");
    await fs.writeFile(src, '{"a":1}', "utf8");

    const bak = await createBackup(src, "test-client");
    expect(bak).not.toBeNull();
    expect(bak!).toMatch(/\.bak-mind-nerve-\d+$/);

    const bakContent = await fs.readFile(bak!, "utf8");
    expect(bakContent).toBe('{"a":1}');
  });

  it("returns null when source does not exist", async () => {
    const tmp = await makeTmp();
    const src = path.join(tmp, "does-not-exist.json");

    const bak = await createBackup(src, "test-client");
    expect(bak).toBeNull();
  });

  it("backup content matches original byte-for-byte", async () => {
    const tmp = await makeTmp();
    const src = path.join(tmp, "data.toml");
    const content = "[section]\nkey = \"value\"\n";
    await fs.writeFile(src, content, "utf8");

    const bak = await createBackup(src, "test-client");
    const bakContent = await fs.readFile(bak!, "utf8");
    expect(bakContent).toBe(content);
  });
});

describe("restoreLatestBackup", () => {
  it("restores the most recent backup and overwrites the source", async () => {
    const tmp = await makeTmp();
    const src = path.join(tmp, "config.json");
    const original = '{"original":true}';
    await fs.writeFile(src, original, "utf8");

    // Create backup.
    await createBackup(src, "test-client");

    // Modify source to simulate install.
    await fs.writeFile(src, '{"modified":true}', "utf8");

    const restored = await restoreLatestBackup(src, "test-client");
    expect(restored).toBe(true);

    const content = await fs.readFile(src, "utf8");
    expect(content).toBe(original);
  });

  it("returns false when no backup exists", async () => {
    const tmp = await makeTmp();
    const src = path.join(tmp, "config.json");

    const restored = await restoreLatestBackup(src, "test-client");
    expect(restored).toBe(false);
  });

  it("restores the LATEST backup when multiple exist", async () => {
    const tmp = await makeTmp();
    const src = path.join(tmp, "config.json");

    await fs.writeFile(src, '"first"', "utf8");
    const bak1 = backupPath(src, Date.now() - 10000);
    await fs.copyFile(src, bak1);

    await fs.writeFile(src, '"second"', "utf8");
    const bak2 = backupPath(src, Date.now());
    await fs.copyFile(src, bak2);

    // Overwrite source with something else.
    await fs.writeFile(src, '"modified"', "utf8");

    const restored = await restoreLatestBackup(src, "test-client");
    expect(restored).toBe(true);

    const content = await fs.readFile(src, "utf8");
    expect(content).toBe('"second"');
  });
});

describe("listBackups", () => {
  it("returns empty list when no backups exist", async () => {
    const tmp = await makeTmp();
    const src = path.join(tmp, "config.json");

    const list = await listBackups(src);
    expect(list).toEqual([]);
  });

  it("returns backups in descending timestamp order", async () => {
    const tmp = await makeTmp();
    const src = path.join(tmp, "config.json");
    await fs.writeFile(src, '{}', "utf8");

    const ts1 = Date.now() - 20000;
    const ts2 = Date.now() - 10000;
    const ts3 = Date.now();

    await fs.copyFile(src, backupPath(src, ts1));
    await fs.copyFile(src, backupPath(src, ts2));
    await fs.copyFile(src, backupPath(src, ts3));

    const list = await listBackups(src);
    expect(list.length).toBe(3);
    // Newest first.
    expect(list[0]).toContain(String(ts3));
    expect(list[1]).toContain(String(ts2));
    expect(list[2]).toContain(String(ts1));
  });
});
