// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.

import { describe, it, expect, afterEach } from "vitest";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import {
  appendInstructionBlock,
  removeInstructionBlock,
  buildInstructionBlock,
  BLOCK_MARKER,
} from "../src/instruction_block.js";

let tmpDirs: string[] = [];

async function makeTmp(): Promise<string> {
  const d = await fs.mkdtemp(path.join(os.tmpdir(), "mn-instr-test-"));
  tmpDirs.push(d);
  return d;
}

afterEach(async () => {
  for (const d of tmpDirs) {
    await fs.rm(d, { recursive: true, force: true });
  }
  tmpDirs = [];
});

describe("buildInstructionBlock", () => {
  it("contains the managed marker", () => {
    const block = buildInstructionBlock(null);
    expect(block).toContain(BLOCK_MARKER);
  });

  it("includes projection dir when provided", () => {
    const block = buildInstructionBlock("/home/user/.mind-nerve/projections/claude-code");
    expect(block).toContain("/home/user/.mind-nerve/projections/claude-code");
  });

  it("does not include projection dir when null", () => {
    const block = buildInstructionBlock(null);
    expect(block).not.toContain("Skill projection directory");
  });
});

describe("appendInstructionBlock", () => {
  it("appends the block to an existing file", async () => {
    const tmp = await makeTmp();
    const target = path.join(tmp, "rules.md");
    await fs.writeFile(target, "# Existing content\n", "utf8");

    const modified = await appendInstructionBlock(target, null);
    expect(modified).toBe(true);

    const content = await fs.readFile(target, "utf8");
    expect(content).toContain("# Existing content");
    expect(content).toContain(BLOCK_MARKER);
  });

  it("creates the file if it does not exist", async () => {
    const tmp = await makeTmp();
    const target = path.join(tmp, "subdir", "rules.md");

    const modified = await appendInstructionBlock(target, null);
    expect(modified).toBe(true);

    const content = await fs.readFile(target, "utf8");
    expect(content).toContain(BLOCK_MARKER);
  });

  it("is idempotent — returns false on second call", async () => {
    const tmp = await makeTmp();
    const target = path.join(tmp, "rules.md");

    await appendInstructionBlock(target, null);
    const secondResult = await appendInstructionBlock(target, null);
    expect(secondResult).toBe(false);

    // Block appears only once.
    const content = await fs.readFile(target, "utf8");
    const count = (content.match(new RegExp(BLOCK_MARKER.replace(/[#]/g, "\\$&"), "g")) ?? []).length;
    expect(count).toBe(1);
  });

  it("creates parent directories", async () => {
    const tmp = await makeTmp();
    const target = path.join(tmp, ".github", "copilot-instructions.md");

    await appendInstructionBlock(target, null);

    const exists = await fs.stat(target).then(() => true, () => false);
    expect(exists).toBe(true);
  });
});

describe("removeInstructionBlock", () => {
  it("removes the block from a file that contains it", async () => {
    const tmp = await makeTmp();
    const target = path.join(tmp, "rules.md");
    await fs.writeFile(target, "# Before\n", "utf8");

    await appendInstructionBlock(target, null);
    const afterAppend = await fs.readFile(target, "utf8");
    expect(afterAppend).toContain(BLOCK_MARKER);

    const removed = await removeInstructionBlock(target);
    expect(removed).toBe(true);

    const afterRemove = await fs.readFile(target, "utf8");
    expect(afterRemove).not.toContain(BLOCK_MARKER);
    expect(afterRemove).toContain("# Before");
  });

  it("returns false when block is not present", async () => {
    const tmp = await makeTmp();
    const target = path.join(tmp, "rules.md");
    await fs.writeFile(target, "# No block here\n", "utf8");

    const removed = await removeInstructionBlock(target);
    expect(removed).toBe(false);
  });

  it("returns false when file does not exist", async () => {
    const tmp = await makeTmp();
    const target = path.join(tmp, "nonexistent.md");

    const removed = await removeInstructionBlock(target);
    expect(removed).toBe(false);
  });

  it("leaves file clean after round-trip append+remove", async () => {
    const tmp = await makeTmp();
    const target = path.join(tmp, "rules.md");
    const original = "# GitHub Copilot Instructions\n\nUse conventional commits.\n";
    await fs.writeFile(target, original, "utf8");

    await appendInstructionBlock(target, "/some/proj/dir");
    await removeInstructionBlock(target);

    const content = await fs.readFile(target, "utf8");
    // Original content is preserved; no trailing whitespace artefacts.
    expect(content.trim()).toBe(original.trim());
  });
});
