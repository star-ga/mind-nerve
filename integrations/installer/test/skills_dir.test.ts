// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.
//
// Part 1 (structural) safety tests. The scenario these guard against is the
// worst thing this installer could do: destroying a user's skills hub.

import { describe, it, expect, afterEach } from "vitest";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import {
  inspectSkillsDir,
  convertToRouterDir,
  restoreSkillsDir,
  readClientState,
  stateFilePath,
  ROUTER_SKILL_NAME,
} from "../src/skills_dir.js";
import { InstallerError } from "../src/errors.js";

const tmpDirs: string[] = [];

async function makeTmp(): Promise<string> {
  const d = await fs.mkdtemp(path.join(os.tmpdir(), "mn-skills-"));
  tmpDirs.push(d);
  return d;
}

afterEach(async () => {
  while (tmpDirs.length > 0) {
    const d = tmpDirs.pop();
    if (d !== undefined) await fs.rm(d, { recursive: true, force: true });
  }
});

/** Builds a hub with a couple of skills, standing in for ~/.agents/skills-hub. */
async function makeHub(root: string): Promise<string> {
  const hub = path.join(root, "skills-hub");
  for (const n of ["alpha", "beta"]) {
    await fs.mkdir(path.join(hub, n), { recursive: true });
    await fs.writeFile(path.join(hub, n, "SKILL.md"), `# ${n}\n`);
  }
  return hub;
}

function opts(root: string, hub: string, now = 1_700_000_000_000) {
  return {
    clientName: "test-cli",
    hubDir: hub,
    routerBody: "---\nname: mind-nerve-router\n---\n\n# router\n",
    homeDir: root,
    now,
  };
}

// ---------------------------------------------------------------------------
// Inspection
// ---------------------------------------------------------------------------

describe("inspectSkillsDir", () => {
  it("reports absent for a missing path", async () => {
    const root = await makeTmp();
    const st = await inspectSkillsDir(path.join(root, "nope"));
    expect(st.kind).toBe("absent");
  });

  it("reports symlink WITHOUT following it", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root);
    const link = path.join(root, "skills");
    await fs.symlink(hub, link);

    const st = await inspectSkillsDir(link);
    // The distinction is load-bearing: a symlink may be unlinked, a real dir
    // may not be removed. `stat` would report "directory" for both.
    expect(st.kind).toBe("symlink");
    expect(st.symlinkTarget).toBe(hub);
  });

  it("reports realdir with an entry count", async () => {
    const root = await makeTmp();
    const dir = path.join(root, "skills");
    await fs.mkdir(path.join(dir, "one"), { recursive: true });
    await fs.writeFile(path.join(dir, "README.md"), "x");

    const st = await inspectSkillsDir(dir);
    expect(st.kind).toBe("realdir");
    expect(st.entryCount).toBe(2);
    expect(st.looksManaged).toBe(false);
  });

  it("recognises a directory we already manage", async () => {
    const root = await makeTmp();
    const dir = path.join(root, "skills");
    await fs.mkdir(path.join(dir, ROUTER_SKILL_NAME), { recursive: true });

    const st = await inspectSkillsDir(dir);
    expect(st.looksManaged).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Conversion
// ---------------------------------------------------------------------------

describe("convertToRouterDir", () => {
  it("replaces a hub symlink with a router-only real directory", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root);
    const skills = path.join(root, "skills");
    await fs.symlink(hub, skills);

    const res = await convertToRouterDir(skills, opts(root, hub));

    expect(res.before.kind).toBe("symlink");
    expect(res.changed).toBe(true);
    const after = await inspectSkillsDir(skills);
    expect(after.kind).toBe("realdir");
    expect((await fs.readdir(skills)).sort()).toEqual([
      "README.md",
      ROUTER_SKILL_NAME,
    ]);
  });

  it("NEVER deletes the hub a symlink pointed at", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root);
    const skills = path.join(root, "skills");
    await fs.symlink(hub, skills);

    await convertToRouterDir(skills, opts(root, hub));

    // The hub must be byte-for-byte intact. This is the destructive-failure
    // guard: `rm -rf` on the symlink path would have wiped it.
    expect((await fs.readdir(hub)).sort()).toEqual(["alpha", "beta"]);
    expect(await fs.readFile(path.join(hub, "alpha", "SKILL.md"), "utf8")).toBe(
      "# alpha\n",
    );
  });

  it("NEVER deletes a real directory — it moves it to a timestamped backup", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root);
    const skills = path.join(root, "skills");
    await fs.mkdir(path.join(skills, "users-own-skill"), { recursive: true });
    await fs.writeFile(
      path.join(skills, "users-own-skill", "SKILL.md"),
      "irreplaceable\n",
    );

    const res = await convertToRouterDir(skills, opts(root, hub));

    expect(res.backupPath).toBe(`${skills}.bak-mind-nerve-1700000000000`);
    expect(
      await fs.readFile(
        path.join(res.backupPath as string, "users-own-skill", "SKILL.md"),
        "utf8",
      ),
    ).toBe("irreplaceable\n");
    expect((await fs.readdir(skills)).sort()).toEqual([
      "README.md",
      ROUTER_SKILL_NAME,
    ]);
  });

  it("creates the directory when nothing was there", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root);
    const skills = path.join(root, "skills");

    const res = await convertToRouterDir(skills, opts(root, hub));

    expect(res.before.kind).toBe("absent");
    expect(res.backupPath).toBeNull();
    expect(await fs.readFile(path.join(skills, ROUTER_SKILL_NAME, "SKILL.md"), "utf8"))
      .toContain("router");
  });

  it("is idempotent — a second run changes nothing and adds no backup", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root);
    const skills = path.join(root, "skills");
    await fs.symlink(hub, skills);

    await convertToRouterDir(skills, opts(root, hub));
    const second = await convertToRouterDir(skills, opts(root, hub, 1_800_000_000_000));

    expect(second.changed).toBe(false);
    expect(second.backupPath).toBeNull();
    const siblings = await fs.readdir(root);
    expect(siblings.filter((e) => e.includes(".bak-mind-nerve-"))).toEqual([]);
  });

  it("records the original state only on the first conversion", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root);
    const skills = path.join(root, "skills");
    await fs.symlink(hub, skills);

    await convertToRouterDir(skills, opts(root, hub));
    await convertToRouterDir(skills, opts(root, hub, 1_800_000_000_000));

    const state = await readClientState("test-cli", root);
    // A second run must not overwrite the record with "was already managed" —
    // that would lose the knowledge of the hub symlink and break uninstall.
    expect(state?.previous.kind).toBe("symlink");
    expect(state?.previous.symlinkTarget).toBe(hub);
    expect(state?.installedAt).toBe(1_700_000_000_000);
  });

  it("refreshes a stale router body in place", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root);
    const skills = path.join(root, "skills");
    await convertToRouterDir(skills, opts(root, hub));

    const updated = {
      ...opts(root, hub),
      routerBody: "---\nname: mind-nerve-router\n---\n\n# router v2\n",
    };
    const res = await convertToRouterDir(skills, updated);

    expect(res.changed).toBe(true);
    expect(
      await fs.readFile(path.join(skills, ROUTER_SKILL_NAME, "SKILL.md"), "utf8"),
    ).toContain("router v2");
  });
});

// ---------------------------------------------------------------------------
// Restoration
// ---------------------------------------------------------------------------

describe("restoreSkillsDir", () => {
  it("re-creates the hub symlink exactly", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root);
    const skills = path.join(root, "skills");
    await fs.symlink(hub, skills);

    await convertToRouterDir(skills, opts(root, hub));
    const res = await restoreSkillsDir("test-cli", { homeDir: root });

    expect(res.restoredKind).toBe("symlink");
    const after = await inspectSkillsDir(skills);
    expect(after.kind).toBe("symlink");
    expect(after.symlinkTarget).toBe(hub);
  });

  it("moves a backed-up real directory back", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root);
    const skills = path.join(root, "skills");
    await fs.mkdir(path.join(skills, "users-own-skill"), { recursive: true });
    await fs.writeFile(path.join(skills, "users-own-skill", "SKILL.md"), "keep\n");

    await convertToRouterDir(skills, opts(root, hub));
    const res = await restoreSkillsDir("test-cli", { homeDir: root });

    expect(res.restoredKind).toBe("realdir");
    expect(
      await fs.readFile(path.join(skills, "users-own-skill", "SKILL.md"), "utf8"),
    ).toBe("keep\n");
    expect((await fs.readdir(root)).filter((e) => e.includes(".bak-"))).toEqual([]);
  });

  it("leaves the path absent when nothing was there before", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root);
    const skills = path.join(root, "skills");

    await convertToRouterDir(skills, opts(root, hub));
    const res = await restoreSkillsDir("test-cli", { homeDir: root });

    expect(res.restoredKind).toBe("absent");
    expect((await inspectSkillsDir(skills)).kind).toBe("absent");
  });

  it("clears the state record so a re-install starts fresh", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root);
    const skills = path.join(root, "skills");
    await fs.symlink(hub, skills);

    await convertToRouterDir(skills, opts(root, hub));
    await restoreSkillsDir("test-cli", { homeDir: root });

    expect(await readClientState("test-cli", root)).toBeNull();
    await expect(
      fs.access(stateFilePath("test-cli", root)),
    ).rejects.toThrow();
  });

  it("is a no-op when the client was never installed", async () => {
    const root = await makeTmp();
    const res = await restoreSkillsDir("never-installed", { homeDir: root });
    expect(res.changed).toBe(false);
    expect(res.restoredKind).toBeNull();
  });

  it("REFUSES to remove a real directory it does not manage", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root);
    const skills = path.join(root, "skills");
    await fs.symlink(hub, skills);
    await convertToRouterDir(skills, opts(root, hub));

    // Someone replaced our managed dir with their own real content.
    await fs.rm(skills, { recursive: true, force: true });
    await fs.mkdir(path.join(skills, "not-ours"), { recursive: true });
    await fs.writeFile(path.join(skills, "not-ours", "SKILL.md"), "precious\n");

    await expect(
      restoreSkillsDir("test-cli", { homeDir: root }),
    ).rejects.toThrow(InstallerError);
    // And the content is still there.
    expect(
      await fs.readFile(path.join(skills, "not-ours", "SKILL.md"), "utf8"),
    ).toBe("precious\n");
  });
});
