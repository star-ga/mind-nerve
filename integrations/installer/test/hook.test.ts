// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.
//
// End-to-end tests for the ONE generalised hook (integrations/hook/mind-nerve-hook).
//
// These drive the REAL script as a subprocess against a fake mind-nerve-routed
// daemon on a real UNIX socket. Testing the actual artefact — rather than a
// re-implementation of its logic — is the point: the hook's contract is
// "whatever happens, print JSON and exit 0", and only the real process can
// prove that.

import { describe, it, expect, beforeAll, afterEach } from "vitest";
import { spawn } from "node:child_process";
import net from "node:net";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { fileURLToPath } from "node:url";

const HOOK = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
  "..",
  "hook",
  "mind-nerve-hook",
);

interface HookRun {
  readonly stdout: string;
  readonly stderr: string;
  readonly code: number | null;
}

/** Runs the hook with the given stdin and environment. */
function runHook(
  stdin: string,
  env: Record<string, string>,
): Promise<HookRun> {
  return new Promise((resolve) => {
    const child = spawn("python3", [HOOK], {
      env: { ...process.env, ...env },
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (d: Buffer) => (stdout += d.toString()));
    child.stderr.on("data", (d: Buffer) => (stderr += d.toString()));
    child.on("close", (code) => resolve({ stdout, stderr, code }));
    child.stdin.write(stdin);
    child.stdin.end();
  });
}

/** Extracts the injected additionalContext, or null when the hook emitted {}. */
function contextOf(run: HookRun): string | null {
  const parsed: unknown = JSON.parse(run.stdout.trim());
  if (
    typeof parsed === "object" &&
    parsed !== null &&
    "hookSpecificOutput" in parsed
  ) {
    const hso = (parsed as { hookSpecificOutput: { additionalContext?: string } })
      .hookSpecificOutput;
    return hso.additionalContext ?? null;
  }
  return null;
}

type DaemonBehaviour =
  | { readonly kind: "reply"; readonly body: string }
  | { readonly kind: "hang" };

interface FakeDaemon {
  readonly socketPath: string;
  close(): Promise<void>;
}

async function startDaemon(
  dir: string,
  behaviour: DaemonBehaviour,
): Promise<FakeDaemon> {
  const socketPath = path.join(dir, "nerve.sock");
  const sockets: net.Socket[] = [];
  const server = net.createServer((sock) => {
    sockets.push(sock);
    sock.on("error", () => undefined);
    sock.on("data", () => undefined);
    if (behaviour.kind === "reply") {
      sock.on("end", () => {
        sock.end(behaviour.body);
      });
    }
    // "hang": accept the connection and never respond, forcing a client timeout.
  });
  await new Promise<void>((res) => server.listen(socketPath, res));
  return {
    socketPath,
    close: () =>
      new Promise<void>((res) => {
        for (const s of sockets) s.destroy();
        server.close(() => res());
      }),
  };
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const tmpDirs: string[] = [];

async function makeTmp(): Promise<string> {
  const d = await fs.mkdtemp(path.join(os.tmpdir(), "mn-hook-"));
  tmpDirs.push(d);
  return d;
}

/** Builds a fake skills hub containing the named skills. */
async function makeHub(root: string, names: readonly string[]): Promise<string> {
  const hub = path.join(root, "hub");
  for (const n of names) {
    await fs.mkdir(path.join(hub, n), { recursive: true });
    await fs.writeFile(
      path.join(hub, n, "SKILL.md"),
      `---\nname: ${n}\ndescription: fixture skill ${n}\n---\n\n# ${n}\n`,
    );
  }
  return hub;
}

function baseEnv(args: {
  readonly hub: string;
  readonly projected: string;
  readonly socket: string;
  readonly log: string;
  readonly minScore?: string;
  readonly topK?: string;
  readonly timeout?: string;
}): Record<string, string> {
  return {
    MIND_NERVE_SOURCE_DIR: args.hub,
    MIND_NERVE_PROJECTED_DIR: args.projected,
    MIND_NERVE_SOCKET: args.socket,
    MIND_NERVE_LOG: args.log,
    MIND_NERVE_AGENT_DIRS: "",
    MIND_NERVE_MIN_SCORE: args.minScore ?? "0.35",
    MIND_NERVE_TOP_K: args.topK ?? "8",
    MIND_NERVE_SOCKET_TIMEOUT: args.timeout ?? "2.0",
    MIND_NERVE_CORE_SKILLS: "mind-nerve-router",
    // These tests exercise dedup / thresholding / projection MECHANICS, using
    // short synthetic prompts ("alpha", "beta") as inert carriers — the fake
    // daemon's reply is fixed regardless of prompt text. The intent gate would
    // reject those carriers as non-intent before the daemon is ever consulted,
    // masking the behaviour under test. It is pinned off here and covered at
    // its shipped default (ON) in hook_gates.test.ts, including a test that
    // asserts the default really is ON.
    MIND_NERVE_INTENT_GATE: "0",
  };
}

function route(
  name: string,
  score: number,
  sourceRepo = "starga",
): Record<string, unknown> {
  return { name, score, source_repo: sourceRepo, kind: "skill" };
}

afterEach(async () => {
  while (tmpDirs.length > 0) {
    const d = tmpDirs.pop();
    if (d !== undefined) await fs.rm(d, { recursive: true, force: true });
  }
});

beforeAll(async () => {
  await fs.access(HOOK); // fail loudly if the hook is missing
});

// ---------------------------------------------------------------------------
// Fail-open
// ---------------------------------------------------------------------------

describe("hook fail-open", () => {
  it("exits 0 with valid JSON when the socket is absent", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "alpha"]);
    const run = await runHook(
      JSON.stringify({ hookEventName: "UserPromptSubmit", prompt: "do a thing" }),
      baseEnv({
        hub,
        projected: path.join(root, "skills"),
        socket: path.join(root, "no-such.sock"),
        log: path.join(root, "hook.log"),
      }),
    );
    expect(run.code).toBe(0);
    expect(run.stdout.trim()).toBe("{}");
  });

  it("exits 0 when the daemon accepts but never replies (timeout)", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "alpha"]);
    const daemon = await startDaemon(root, { kind: "hang" });
    try {
      const started = Date.now();
      const run = await runHook(
        JSON.stringify({ hookEventName: "UserPromptSubmit", prompt: "do a thing" }),
        baseEnv({
          hub,
          projected: path.join(root, "skills"),
          socket: daemon.socketPath,
          log: path.join(root, "hook.log"),
          timeout: "0.4",
        }),
      );
      expect(run.code).toBe(0);
      expect(run.stdout.trim()).toBe("{}");
      // The timeout must actually bound the wait — a hook that blocks forever
      // is worse than one that fails.
      expect(Date.now() - started).toBeLessThan(8000);
    } finally {
      await daemon.close();
    }
  });

  it("exits 0 when the daemon returns malformed JSON", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "alpha"]);
    const daemon = await startDaemon(root, {
      kind: "reply",
      body: "{not json at all",
    });
    try {
      const run = await runHook(
        JSON.stringify({ hookEventName: "UserPromptSubmit", prompt: "do a thing" }),
        baseEnv({
          hub,
          projected: path.join(root, "skills"),
          socket: daemon.socketPath,
          log: path.join(root, "hook.log"),
        }),
      );
      expect(run.code).toBe(0);
      expect(run.stdout.trim()).toBe("{}");
    } finally {
      await daemon.close();
    }
  });

  it("exits 0 with valid JSON when stdin itself is malformed", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "alpha"]);
    const run = await runHook(
      "this is not json {{{",
      baseEnv({
        hub,
        projected: path.join(root, "skills"),
        socket: path.join(root, "no-such.sock"),
        log: path.join(root, "hook.log"),
      }),
    );
    expect(run.code).toBe(0);
    expect(() => JSON.parse(run.stdout.trim())).not.toThrow();
    // Garbage in must not produce a ranked route table.
    expect(contextOf(run) ?? "").not.toContain("| rank |");
  });

  it("exits 0 when the daemon reports an error object", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router"]);
    const daemon = await startDaemon(root, {
      kind: "reply",
      body: JSON.stringify({ error: "runtime not loaded" }),
    });
    try {
      const run = await runHook(
        JSON.stringify({ prompt: "anything" }),
        baseEnv({
          hub,
          projected: path.join(root, "skills"),
          socket: daemon.socketPath,
          log: path.join(root, "hook.log"),
        }),
      );
      expect(run.code).toBe(0);
      expect(run.stdout.trim()).toBe("{}");
    } finally {
      await daemon.close();
    }
  });

  it("exits 0 when the hub does not exist", async () => {
    const root = await makeTmp();
    const run = await runHook(
      JSON.stringify({ prompt: "anything" }),
      baseEnv({
        hub: path.join(root, "no-hub"),
        projected: path.join(root, "skills"),
        socket: path.join(root, "no-such.sock"),
        log: path.join(root, "hook.log"),
      }),
    );
    expect(run.code).toBe(0);
    expect(run.stdout.trim()).toBe("{}");
  });
});

// ---------------------------------------------------------------------------
// Dedup across source_repo
// ---------------------------------------------------------------------------

describe("hook dedup", () => {
  it("collapses the same skill carried under starga and local", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, [
      "mind-nerve-router",
      "cloudflare-deploy",
      "deploying-cloudflare-access-for-zero-trust",
    ]);
    const daemon = await startDaemon(root, {
      kind: "reply",
      body: JSON.stringify({
        routes: [
          route("cloudflare-deploy", 0.74, "starga"),
          route("deploying-cloudflare-access-for-zero-trust", 0.61, "starga"),
          route("deploying-cloudflare-access-for-zero-trust", 0.6, "local"),
        ],
      }),
    });
    try {
      const run = await runHook(
        JSON.stringify({ prompt: "deploy a site to cloudflare pages" }),
        baseEnv({
          hub,
          projected: path.join(root, "skills"),
          socket: daemon.socketPath,
          log: path.join(root, "hook.log"),
        }),
      );
      const ctx = contextOf(run) ?? "";
      const rows = ctx
        .split("\n")
        .filter((l) => l.startsWith("| ") && l.includes("`"));
      const zeroTrust = rows.filter((l) =>
        l.includes("deploying-cloudflare-access-for-zero-trust"),
      );
      expect(zeroTrust.length).toBe(1);
      expect(rows.filter((l) => l.includes("cloudflare-deploy")).length).toBe(1);
    } finally {
      await daemon.close();
    }
  });

  it("keeps the higher-scoring duplicate", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "alpha"]);
    const daemon = await startDaemon(root, {
      kind: "reply",
      body: JSON.stringify({
        routes: [route("alpha", 0.5, "starga"), route("alpha", 0.9, "local")],
      }),
    });
    try {
      const run = await runHook(
        JSON.stringify({ prompt: "alpha work" }),
        baseEnv({
          hub,
          projected: path.join(root, "skills"),
          socket: daemon.socketPath,
          log: path.join(root, "hook.log"),
        }),
      );
      const ctx = contextOf(run) ?? "";
      expect(ctx).toContain("0.900");
      expect(ctx).not.toContain("0.500");
    } finally {
      await daemon.close();
    }
  });

  it("dedups case and separator variants onto one row", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "git-workflow"]);
    const daemon = await startDaemon(root, {
      kind: "reply",
      body: JSON.stringify({
        routes: [
          route("git-workflow", 0.8, "starga"),
          route("Git_Workflow", 0.79, "local"),
        ],
      }),
    });
    try {
      const run = await runHook(
        JSON.stringify({ prompt: "branch strategy" }),
        baseEnv({
          hub,
          projected: path.join(root, "skills"),
          socket: daemon.socketPath,
          log: path.join(root, "hook.log"),
        }),
      );
      const ctx = contextOf(run) ?? "";
      const rows = ctx.split("\n").filter((l) => /^\| \d+ \|/.test(l));
      expect(rows.length).toBe(1);
    } finally {
      await daemon.close();
    }
  });
});

// ---------------------------------------------------------------------------
// Score threshold
// ---------------------------------------------------------------------------

describe("hook score threshold", () => {
  /**
   * The measured case: for "prove an exact minimum move floor for an ARC-AGI-3
   * level" the daemon returned agi3-pixel-porter at 0.437 (correct) followed by
   * seven unrelated skills at 0.267-0.282, all of which were injected AND
   * projected. Only the first row should survive the 0.35 floor.
   */
  const MEASURED = [
    route("agi3-pixel-porter", 0.437),
    route("performing-graphql-depth-limit-attack", 0.282),
    route("performing-soc-tabletop-exercise", 0.279),
    route("implementing-siem-correlation-rules-for-apt", 0.276),
    route("containing-active-breach", 0.273),
    route("yeet", 0.271),
    route("using-science-superpowers", 0.269),
    route("ai-truthfulness-enforcer", 0.267),
  ];
  const MEASURED_NAMES = MEASURED.map((r) => r["name"] as string);

  it("injects only the above-threshold route from the measured case", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", ...MEASURED_NAMES]);
    const daemon = await startDaemon(root, {
      kind: "reply",
      body: JSON.stringify({ routes: MEASURED }),
    });
    try {
      const run = await runHook(
        JSON.stringify({
          prompt: "prove an exact minimum move floor for an ARC-AGI-3 level",
        }),
        baseEnv({
          hub,
          projected: path.join(root, "skills"),
          socket: daemon.socketPath,
          log: path.join(root, "hook.log"),
        }),
      );
      const ctx = contextOf(run) ?? "";
      const rows = ctx.split("\n").filter((l) => /^\| \d+ \|/.test(l));
      expect(rows.length).toBe(1);
      expect(rows[0]).toContain("agi3-pixel-porter");
      for (const noise of MEASURED_NAMES.slice(1)) {
        expect(ctx).not.toContain(noise);
      }
    } finally {
      await daemon.close();
    }
  });

  it("projects only the above-threshold skill from the measured case", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", ...MEASURED_NAMES]);
    const projected = path.join(root, "skills");
    const daemon = await startDaemon(root, {
      kind: "reply",
      body: JSON.stringify({ routes: MEASURED }),
    });
    try {
      await runHook(
        JSON.stringify({ prompt: "arc-agi-3 move floor" }),
        baseEnv({
          hub,
          projected,
          socket: daemon.socketPath,
          log: path.join(root, "hook.log"),
        }),
      );
      const entries = (await fs.readdir(projected)).sort();
      // router (real dir) + README + exactly one projected skill.
      expect(entries).toEqual([
        "README.md",
        "agi3-pixel-porter",
        "mind-nerve-router",
      ]);
    } finally {
      await daemon.close();
    }
  });

  it("says 'no strong skill match' when nothing clears the floor", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, [
      "mind-nerve-router",
      "yeet",
      "containing-active-breach",
    ]);
    const daemon = await startDaemon(root, {
      kind: "reply",
      body: JSON.stringify({
        routes: [route("yeet", 0.271), route("containing-active-breach", 0.268)],
      }),
    });
    try {
      const run = await runHook(
        JSON.stringify({ prompt: "something with no matching skill" }),
        baseEnv({
          hub,
          projected: path.join(root, "skills"),
          socket: daemon.socketPath,
          log: path.join(root, "hook.log"),
        }),
      );
      const ctx = contextOf(run) ?? "";
      expect(ctx).toContain("No strong skill match");
      expect(ctx).not.toContain("| rank |");
      expect(ctx).not.toContain("yeet");
    } finally {
      await daemon.close();
    }
  });

  it("honours a custom MIND_NERVE_MIN_SCORE", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "beta"]);
    const daemon = await startDaemon(root, {
      kind: "reply",
      body: JSON.stringify({ routes: [route("beta", 0.30)] }),
    });
    try {
      const strict = await runHook(
        JSON.stringify({ prompt: "beta" }),
        baseEnv({
          hub,
          projected: path.join(root, "s1"),
          socket: daemon.socketPath,
          log: path.join(root, "hook.log"),
          minScore: "0.35",
        }),
      );
      expect(contextOf(strict) ?? "").toContain("No strong skill match");

      const lax = await runHook(
        JSON.stringify({ prompt: "beta" }),
        baseEnv({
          hub,
          projected: path.join(root, "s2"),
          socket: daemon.socketPath,
          log: path.join(root, "hook.log"),
          minScore: "0.25",
        }),
      );
      expect(contextOf(lax) ?? "").toContain("beta");
    } finally {
      await daemon.close();
    }
  });

  it("caps the injected table at TOP_K after dedup and thresholding", async () => {
    const root = await makeTmp();
    const names = ["s0", "s1", "s2", "s3", "s4", "s5"];
    const hub = await makeHub(root, ["mind-nerve-router", ...names]);
    const daemon = await startDaemon(root, {
      kind: "reply",
      body: JSON.stringify({
        routes: names.map((n, i) => route(n, 0.9 - i * 0.01)),
      }),
    });
    try {
      const run = await runHook(
        JSON.stringify({ prompt: "many matches" }),
        baseEnv({
          hub,
          projected: path.join(root, "skills"),
          socket: daemon.socketPath,
          log: path.join(root, "hook.log"),
          topK: "3",
        }),
      );
      const ctx = contextOf(run) ?? "";
      const rows = ctx.split("\n").filter((l) => /^\| \d+ \|/.test(l));
      expect(rows.length).toBe(3);
    } finally {
      await daemon.close();
    }
  });
});

// ---------------------------------------------------------------------------
// Projection atomicity
// ---------------------------------------------------------------------------

describe("hook projection atomicity", () => {
  it("leaves no staging directories behind on success", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "alpha"]);
    const projected = path.join(root, "skills");
    const daemon = await startDaemon(root, {
      kind: "reply",
      body: JSON.stringify({ routes: [route("alpha", 0.8)] }),
    });
    try {
      await runHook(
        JSON.stringify({ prompt: "alpha" }),
        baseEnv({
          hub,
          projected,
          socket: daemon.socketPath,
          log: path.join(root, "hook.log"),
        }),
      );
      const siblings = await fs.readdir(root);
      expect(siblings.filter((e) => e.includes(".tmp."))).toEqual([]);
      expect(siblings.filter((e) => e.includes(".old."))).toEqual([]);
    } finally {
      await daemon.close();
    }
  });

  it("swaps the projection wholesale — never a partially filled dir", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, [
      "mind-nerve-router",
      "alpha",
      "beta",
      "gamma",
    ]);
    const projected = path.join(root, "skills");
    const first = await startDaemon(root, {
      kind: "reply",
      body: JSON.stringify({
        routes: [route("alpha", 0.8), route("beta", 0.75)],
      }),
    });
    let entriesAfterFirst: string[];
    try {
      await runHook(
        JSON.stringify({ prompt: "alpha beta" }),
        baseEnv({
          hub,
          projected,
          socket: first.socketPath,
          log: path.join(root, "hook.log"),
        }),
      );
      entriesAfterFirst = (await fs.readdir(projected)).sort();
    } finally {
      await first.close();
    }
    expect(entriesAfterFirst).toEqual([
      "README.md",
      "alpha",
      "beta",
      "mind-nerve-router",
    ]);

    // A second run with a different result set must fully replace the first —
    // no leftovers from the previous turn.
    const second = await startDaemon(await makeTmp(), {
      kind: "reply",
      body: JSON.stringify({ routes: [route("gamma", 0.9)] }),
    });
    try {
      await runHook(
        JSON.stringify({ prompt: "gamma" }),
        baseEnv({
          hub,
          projected,
          socket: second.socketPath,
          log: path.join(root, "hook.log"),
        }),
      );
      const entries = (await fs.readdir(projected)).sort();
      expect(entries).toEqual(["README.md", "gamma", "mind-nerve-router"]);
    } finally {
      await second.close();
    }
  });

  it("keeps the existing projection intact when the swap cannot happen", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "alpha"]);
    const parent = path.join(root, "locked");
    const projected = path.join(parent, "skills");
    await fs.mkdir(projected, { recursive: true });
    await fs.writeFile(path.join(projected, "SENTINEL"), "pre-existing state");

    const daemon = await startDaemon(root, {
      kind: "reply",
      body: JSON.stringify({ routes: [route("alpha", 0.8)] }),
    });
    // Read+execute only: staging cannot be created, so the swap must not start.
    await fs.chmod(parent, 0o555);
    try {
      const run = await runHook(
        JSON.stringify({ prompt: "alpha" }),
        baseEnv({
          hub,
          projected,
          socket: daemon.socketPath,
          log: path.join(root, "hook.log"),
        }),
      );
      expect(run.code).toBe(0);
      // Original content survives untouched — no torn state.
      const entries = await fs.readdir(projected);
      expect(entries).toContain("SENTINEL");
      expect(await fs.readdir(parent)).toEqual(["skills"]);
    } finally {
      await fs.chmod(parent, 0o755);
      await daemon.close();
    }
  });

  it("always keeps the router skill in the projection", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "alpha"]);
    const projected = path.join(root, "skills");
    const daemon = await startDaemon(root, {
      kind: "reply",
      body: JSON.stringify({ routes: [route("alpha", 0.9)] }),
    });
    try {
      await runHook(
        JSON.stringify({ prompt: "alpha" }),
        baseEnv({
          hub,
          projected,
          socket: daemon.socketPath,
          log: path.join(root, "hook.log"),
        }),
      );
      const body = await fs.readFile(
        path.join(projected, "mind-nerve-router", "SKILL.md"),
        "utf8",
      );
      expect(body).toContain("mind-nerve-router");
    } finally {
      await daemon.close();
    }
  });
});

// ---------------------------------------------------------------------------
// Injected content contract
// ---------------------------------------------------------------------------

describe("hook injected context", () => {
  it("emits absolute SKILL.md paths", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "alpha"]);
    const daemon = await startDaemon(root, {
      kind: "reply",
      body: JSON.stringify({ routes: [route("alpha", 0.9)] }),
    });
    try {
      const run = await runHook(
        JSON.stringify({ prompt: "alpha" }),
        baseEnv({
          hub,
          projected: path.join(root, "skills"),
          socket: daemon.socketPath,
          log: path.join(root, "hook.log"),
        }),
      );
      const ctx = contextOf(run) ?? "";
      expect(ctx).toContain(path.join(hub, "alpha", "SKILL.md"));
    } finally {
      await daemon.close();
    }
  });

  it("routes to agent .md files, not just skills", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router"]);
    const agentsDir = path.join(root, "agents");
    await fs.mkdir(agentsDir, { recursive: true });
    await fs.writeFile(path.join(agentsDir, "code-reviewer.md"), "# reviewer\n");

    const daemon = await startDaemon(root, {
      kind: "reply",
      body: JSON.stringify({ routes: [route("code-reviewer", 0.8)] }),
    });
    try {
      const env = baseEnv({
        hub,
        projected: path.join(root, "skills"),
        socket: daemon.socketPath,
        log: path.join(root, "hook.log"),
      });
      const run = await runHook(JSON.stringify({ prompt: "review my code" }), {
        ...env,
        MIND_NERVE_AGENT_DIRS: agentsDir,
      });
      const ctx = contextOf(run) ?? "";
      expect(ctx).toContain("agent");
      expect(ctx).toContain(path.join(agentsDir, "code-reviewer.md"));
    } finally {
      await daemon.close();
    }
  });

  it("marks the hook event as UserPromptSubmit", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "alpha"]);
    const daemon = await startDaemon(root, {
      kind: "reply",
      body: JSON.stringify({ routes: [route("alpha", 0.9)] }),
    });
    try {
      const run = await runHook(
        JSON.stringify({ hookEventName: "UserPromptSubmit", prompt: "alpha" }),
        baseEnv({
          hub,
          projected: path.join(root, "skills"),
          socket: daemon.socketPath,
          log: path.join(root, "hook.log"),
        }),
      );
      const parsed = JSON.parse(run.stdout.trim()) as {
        hookSpecificOutput: { hookEventName: string };
      };
      expect(parsed.hookSpecificOutput.hookEventName).toBe("UserPromptSubmit");
    } finally {
      await daemon.close();
    }
  });

  it("seeds the projection and announces readiness on SessionStart", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "alpha"]);
    const projected = path.join(root, "skills");
    const run = await runHook(
      JSON.stringify({ hookEventName: "SessionStart" }),
      baseEnv({
        hub,
        projected,
        socket: path.join(root, "no-such.sock"),
        log: path.join(root, "hook.log"),
      }),
    );
    expect(run.code).toBe(0);
    expect(contextOf(run) ?? "").toContain("mind-nerve ready");
    expect((await fs.readdir(projected)).sort()).toEqual([
      "README.md",
      "mind-nerve-router",
    ]);
  });

  it("accepts the alternate stdin prompt keys the CLIs use", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "alpha"]);
    const daemon = await startDaemon(root, {
      kind: "reply",
      body: JSON.stringify({ routes: [route("alpha", 0.9)] }),
    });
    try {
      for (const payload of [
        { user_prompt: "alpha" },
        { userPrompt: "alpha" },
        { message: "alpha" },
        { input: { prompt: "alpha" } },
      ]) {
        const run = await runHook(
          JSON.stringify(payload),
          baseEnv({
            hub,
            projected: path.join(root, "skills"),
            socket: daemon.socketPath,
            log: path.join(root, "hook.log"),
          }),
        );
        expect(contextOf(run) ?? "", JSON.stringify(payload)).toContain("alpha");
      }
    } finally {
      await daemon.close();
    }
  });
});
