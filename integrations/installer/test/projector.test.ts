// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.

import { describe, it, expect, afterEach } from "vitest";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { ensureProjectionDir, rewriteProjectionLinks } from "../src/projector.js";

let tmpDirs: string[] = [];

async function makeTmp(): Promise<string> {
  const d = await fs.mkdtemp(path.join(os.tmpdir(), "mn-proj-test-"));
  tmpDirs.push(d);
  return d;
}

afterEach(async () => {
  for (const d of tmpDirs) {
    await fs.rm(d, { recursive: true, force: true });
  }
  tmpDirs = [];
});

describe("ensureProjectionDir", () => {
  it("creates the projection directory", async () => {
    const base = await makeTmp();
    const projDir = path.join(base, "projections", "test-client");

    await ensureProjectionDir(projDir, "test-client");

    const stat = await fs.stat(projDir);
    expect(stat.isDirectory()).toBe(true);
  });

  it("is idempotent — does not fail if dir already exists", async () => {
    const base = await makeTmp();
    const projDir = path.join(base, "projections", "test-client");

    await ensureProjectionDir(projDir, "test-client");
    // Second call must not throw.
    await expect(ensureProjectionDir(projDir, "test-client")).resolves.toBeUndefined();
  });

  it("creates nested directories (mkdir -p behaviour)", async () => {
    const base = await makeTmp();
    const projDir = path.join(base, "a", "b", "c", "d");

    await ensureProjectionDir(projDir, "test-client");

    const stat = await fs.stat(projDir);
    expect(stat.isDirectory()).toBe(true);
  });
});

describe("rewriteProjectionLinks", () => {
  it("creates symlinks inside projection dir", async () => {
    const base = await makeTmp();
    const projDir = path.join(base, "proj");
    await fs.mkdir(projDir, { recursive: true });

    // Create a target file to link to.
    const targetDir = path.join(base, "target-skill");
    await fs.mkdir(targetDir);
    const sources = new Map([["my-skill", targetDir]]);

    await rewriteProjectionLinks(projDir, sources, "test-client");

    const link = path.join(projDir, "my-skill");
    const stat = await fs.lstat(link);
    expect(stat.isSymbolicLink()).toBe(true);

    const resolved = await fs.readlink(link);
    expect(resolved).toBe(targetDir);
  });

  it("atomically replaces an existing projection dir", async () => {
    const base = await makeTmp();
    const projDir = path.join(base, "proj");
    await fs.mkdir(projDir, { recursive: true });

    const t1 = path.join(base, "skill-a");
    const t2 = path.join(base, "skill-b");
    await fs.mkdir(t1);
    await fs.mkdir(t2);

    // First write.
    await rewriteProjectionLinks(projDir, new Map([["skill-a", t1]]), "test-client");
    // Second write — replaces, adding skill-b, removing skill-a link.
    await rewriteProjectionLinks(projDir, new Map([["skill-b", t2]]), "test-client");

    const entries = await fs.readdir(projDir);
    expect(entries).toContain("skill-b");
    expect(entries).not.toContain("skill-a");
  });

  it("works with an empty source map (clears the projection dir)", async () => {
    const base = await makeTmp();
    const projDir = path.join(base, "proj");
    await fs.mkdir(projDir, { recursive: true });

    const t = path.join(base, "skill");
    await fs.mkdir(t);
    await rewriteProjectionLinks(projDir, new Map([["skill", t]]), "test-client");

    // Clear.
    await rewriteProjectionLinks(projDir, new Map(), "test-client");

    const entries = await fs.readdir(projDir);
    expect(entries.length).toBe(0);
  });
});
