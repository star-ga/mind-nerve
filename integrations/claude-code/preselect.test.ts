// mind-nerve Claude Code UserPromptSubmit hook. Projector pattern (D1).

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import crypto from "node:crypto";

// ---------------------------------------------------------------------------
// Module mocking — must happen before any imports of the module under test.
// ---------------------------------------------------------------------------

vi.mock("./subprocess.js", async (importOriginal) => {
  const original =
    await importOriginal<typeof import("./subprocess.js")>();
  return {
    ...original,
    isBinaryAvailable: vi.fn(() => true),
    callMindNerve: vi.fn(),
  };
});

vi.mock("./logger.js", () => ({
  logger: {
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    debug: vi.fn(),
  },
}));

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

type HookResult = Record<string, never>;

async function makeSkillsDir(
  base: string,
  skillNames: string[],
): Promise<string> {
  const skillsDir = path.join(base, "skills");
  await fs.mkdir(skillsDir, { recursive: true });
  for (const name of skillNames) {
    const skillDir = path.join(skillsDir, name);
    await fs.mkdir(skillDir, { recursive: true });
    await fs.writeFile(
      path.join(skillDir, "SKILL.md"),
      `---\nname: ${name}\n---\nSkill ${name}\n`,
    );
  }
  return skillsDir;
}

async function makeProjectedDir(base: string): Promise<string> {
  const projectedDir = path.join(base, "skills-projected");
  await fs.mkdir(projectedDir, { recursive: true });
  return projectedDir;
}

function makeCacheFile(base: string): string {
  return path.join(base, "projector.cache.json");
}

// ---------------------------------------------------------------------------
// Test suite
// ---------------------------------------------------------------------------

describe("userPromptSubmitHook", () => {
  let tmpDir: string;
  let skillsDir: string;
  let projectedDir: string;
  let cacheFile: string;

  // We import the hook dynamically so mocks are applied before import.
  let runHook: (stdinJSON: string, overrides?: Record<string, unknown>) => Promise<HookResult>;

  beforeEach(async () => {
    vi.resetAllMocks();
    tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "mn-test-"));
    skillsDir = await makeSkillsDir(tmpDir, ["skill-a", "skill-b", "skill-c"]);
    projectedDir = await makeProjectedDir(tmpDir);
    cacheFile = makeCacheFile(tmpDir);

    // Dynamic import after mocks are in place.
    const mod = await import("./preselect.js");
    runHook = async (stdinJSON, overrides = {}) =>
      mod.userPromptSubmitHook(stdinJSON, {
        binaryPath: "mind-nerve",
        topK: 2,
        timeoutMs: 150,
        skillsDir,
        projectedDir,
        cacheFile,
        confidenceThreshold: 0.5,
        ...overrides,
      });
  });

  afterEach(async () => {
    await fs.rm(tmpDir, { recursive: true, force: true });
    vi.restoreAllMocks();
  });

  // -------------------------------------------------------------------------
  // T1: stdin parse failure → returns {}
  // -------------------------------------------------------------------------

  it("returns empty object when stdin is invalid JSON", async () => {
    const result = await runHook("NOT_JSON_AT_ALL");
    expect(result).toEqual({});
  });

  it("returns empty object when stdin is valid JSON but wrong shape", async () => {
    const result = await runHook(JSON.stringify({ wrong_field: 1 }));
    expect(result).toEqual({});
  });

  // -------------------------------------------------------------------------
  // T2: mind-nerve missing binary → passthrough projection
  // -------------------------------------------------------------------------

  it("writes passthrough projection when binary is not available", async () => {
    const { isBinaryAvailable } = await import("./subprocess.js");
    vi.mocked(isBinaryAvailable).mockReturnValue(false);

    const result = await runHook(
      JSON.stringify({ user_prompt: "hello", transcript_path: "/tmp/t.json" }),
    );

    expect(result).toEqual({});

    // Projection dir should contain symlinks to all skills.
    const links = await fs.readdir(projectedDir);
    expect(links.sort()).toEqual(["skill-a", "skill-b", "skill-c"]);
  });

  // -------------------------------------------------------------------------
  // T3: mind-nerve timeout → passthrough projection
  // -------------------------------------------------------------------------

  it("writes passthrough projection on subprocess timeout", async () => {
    const { callMindNerve, isBinaryAvailable, SubprocessTimeoutError } =
      await import("./subprocess.js");
    vi.mocked(isBinaryAvailable).mockReturnValue(true);
    vi.mocked(callMindNerve).mockRejectedValue(
      new SubprocessTimeoutError(150),
    );

    const result = await runHook(
      JSON.stringify({ user_prompt: "timeout test", transcript_path: "/t" }),
    );

    expect(result).toEqual({});

    const links = await fs.readdir(projectedDir);
    expect(links.sort()).toEqual(["skill-a", "skill-b", "skill-c"]);
  });

  // -------------------------------------------------------------------------
  // T4: outcome=top_k → projection contains only K returned skills
  // -------------------------------------------------------------------------

  it("rewrites projection to top-K skills when outcome is top_k", async () => {
    const { callMindNerve, isBinaryAvailable } = await import("./subprocess.js");
    vi.mocked(isBinaryAvailable).mockReturnValue(true);
    vi.mocked(callMindNerve).mockResolvedValue({
      outcome: "top_k",
      selected: ["skill-a", "skill-c"],
      scores: [0.91, 0.78],
    });

    const result = await runHook(
      JSON.stringify({
        user_prompt: "do something with skill-a and skill-c",
        transcript_path: "/t",
      }),
    );

    expect(result).toEqual({});

    const links = await fs.readdir(projectedDir);
    expect(links.sort()).toEqual(["skill-a", "skill-c"]);
  });

  it("filters out skill IDs not present in the catalog", async () => {
    const { callMindNerve, isBinaryAvailable } = await import("./subprocess.js");
    vi.mocked(isBinaryAvailable).mockReturnValue(true);
    vi.mocked(callMindNerve).mockResolvedValue({
      outcome: "top_k",
      selected: ["skill-a", "skill-NONEXISTENT"],
    });

    await runHook(
      JSON.stringify({ user_prompt: "test", transcript_path: "/t" }),
    );

    const links = await fs.readdir(projectedDir);
    expect(links.sort()).toEqual(["skill-a"]);
  });

  // -------------------------------------------------------------------------
  // T5: outcome=low_confidence → projection contains all skills
  // -------------------------------------------------------------------------

  it("writes passthrough projection when outcome is low_confidence", async () => {
    const { callMindNerve, isBinaryAvailable } = await import("./subprocess.js");
    vi.mocked(isBinaryAvailable).mockReturnValue(true);
    vi.mocked(callMindNerve).mockResolvedValue({
      outcome: "low_confidence",
      reason: "confidence below threshold",
    });

    await runHook(
      JSON.stringify({ user_prompt: "unclear request", transcript_path: "/t" }),
    );

    const links = await fs.readdir(projectedDir);
    expect(links.sort()).toEqual(["skill-a", "skill-b", "skill-c"]);
  });

  // -------------------------------------------------------------------------
  // T6: outcome=passthrough → projection contains all skills
  // -------------------------------------------------------------------------

  it("writes passthrough projection when outcome is passthrough", async () => {
    const { callMindNerve, isBinaryAvailable } = await import("./subprocess.js");
    vi.mocked(isBinaryAvailable).mockReturnValue(true);
    vi.mocked(callMindNerve).mockResolvedValue({
      outcome: "passthrough",
    });

    await runHook(
      JSON.stringify({ user_prompt: "pass me through", transcript_path: "/t" }),
    );

    const links = await fs.readdir(projectedDir);
    expect(links.sort()).toEqual(["skill-a", "skill-b", "skill-c"]);
  });

  // -------------------------------------------------------------------------
  // T7: outcome=error → projection unchanged, returns {}
  // -------------------------------------------------------------------------

  it("leaves projection unchanged when outcome is error", async () => {
    const { callMindNerve, isBinaryAvailable } = await import("./subprocess.js");
    vi.mocked(isBinaryAvailable).mockReturnValue(true);
    vi.mocked(callMindNerve).mockResolvedValue({
      outcome: "error",
      message: "inference failed",
    });

    // Pre-populate projection with skill-b only (simulates prior turn).
    const skillBDir = path.join(skillsDir, "skill-b");
    const existingLink = path.join(projectedDir, "skill-b");
    // Remove the existing projectedDir contents first.
    await fs.rm(projectedDir, { recursive: true });
    await fs.mkdir(projectedDir, { recursive: true });
    await fs.symlink(skillBDir, existingLink);

    await runHook(
      JSON.stringify({ user_prompt: "error case", transcript_path: "/t" }),
    );

    // Projection should remain unchanged — only skill-b.
    const links = await fs.readdir(projectedDir);
    expect(links).toEqual(["skill-b"]);
  });

  // -------------------------------------------------------------------------
  // T8: mind-nerve returns invalid JSON → passthrough (fail-open)
  // -------------------------------------------------------------------------

  it("falls back to passthrough when mind-nerve returns invalid JSON", async () => {
    const { callMindNerve, isBinaryAvailable } = await import("./subprocess.js");
    vi.mocked(isBinaryAvailable).mockReturnValue(true);
    vi.mocked(callMindNerve).mockRejectedValue(
      new SyntaxError("Unexpected token"),
    );

    await runHook(
      JSON.stringify({ user_prompt: "bad json", transcript_path: "/t" }),
    );

    const links = await fs.readdir(projectedDir);
    expect(links.sort()).toEqual(["skill-a", "skill-b", "skill-c"]);
  });

  // -------------------------------------------------------------------------
  // T9: Atomic rewrite — old projection survives a kill mid-rewrite
  // -------------------------------------------------------------------------

  it("preserves old projection if staging write fails mid-operation", async () => {
    const { callMindNerve, isBinaryAvailable } = await import("./subprocess.js");
    vi.mocked(isBinaryAvailable).mockReturnValue(true);
    vi.mocked(callMindNerve).mockResolvedValue({
      outcome: "top_k",
      selected: ["skill-a"],
    });

    // Establish an initial projection with all skills.
    const allSkillDirs = ["skill-a", "skill-b", "skill-c"];
    await fs.rm(projectedDir, { recursive: true });
    await fs.mkdir(projectedDir, { recursive: true });
    for (const s of allSkillDirs) {
      await fs.symlink(path.join(skillsDir, s), path.join(projectedDir, s));
    }

    // Simulate a mid-rename failure by monkey-patching fs.rename to throw
    // after the first successful call (staging dir created) on the second
    // call (staging → projectedDir rename). We do this by wrapping rename.
    const originalRename = fs.rename.bind(fs);
    let renameCount = 0;
    const renameStub = vi
      .spyOn(fs, "rename")
      .mockImplementation(async (src, dst) => {
        renameCount++;
        if (renameCount === 2) {
          throw new Error("Simulated rename failure");
        }
        return originalRename(src, dst);
      });

    try {
      // The hook itself should not throw — fail-open.
      const result = await runHook(
        JSON.stringify({ user_prompt: "atomic test", transcript_path: "/t" }),
      );
      expect(result).toEqual({});
    } finally {
      renameStub.mockRestore();
    }

    // Old projection (all three skills) should still be present.
    const links = await fs.readdir(projectedDir);
    expect(links.sort()).toEqual(["skill-a", "skill-b", "skill-c"]);
  });
});
