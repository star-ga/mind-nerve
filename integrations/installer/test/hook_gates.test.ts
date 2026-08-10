// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.
//
// Gate + projection tests for the ONE generalised hook.
//
// One describe block per shipped decision, so a future reader can see WHICH
// decision a failure belongs to. Like hook.test.ts these drive the REAL script
// as a subprocess against a fake daemon on a real UNIX socket.
//
// The measured fixtures below are the evidence the constants were chosen from.
// They are pinned deliberately: the numbers moved once already when the
// catalog was re-embedded, and an unpinned constant silently changes meaning.

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

/** Rank rows of the injected markdown table. */
function injectedRows(ctx: string | null): string[] {
  if (ctx === null) return [];
  return ctx.split("\n").filter((l) => /^\| \d+ \|/.test(l));
}

interface FakeDaemon {
  readonly socketPath: string;
  close(): Promise<void>;
}

async function startDaemon(dir: string, body: string): Promise<FakeDaemon> {
  const socketPath = path.join(dir, "nerve.sock");
  const sockets: net.Socket[] = [];
  const server = net.createServer((sock) => {
    sockets.push(sock);
    sock.on("error", () => undefined);
    sock.on("data", () => undefined);
    sock.on("end", () => sock.end(body));
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

const tmpDirs: string[] = [];

async function makeTmp(): Promise<string> {
  const d = await fs.mkdtemp(path.join(os.tmpdir(), "mn-gate-"));
  tmpDirs.push(d);
  return d;
}

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

/** Env WITHOUT pinning the gates, so shipped defaults are what gets tested. */
function prodEnv(args: {
  readonly hub: string;
  readonly projected: string;
  readonly socket: string;
  readonly log: string;
  readonly extra?: Record<string, string>;
}): Record<string, string> {
  return {
    MIND_NERVE_SOURCE_DIR: args.hub,
    MIND_NERVE_PROJECTED_DIR: args.projected,
    MIND_NERVE_SOCKET: args.socket,
    MIND_NERVE_LOG: args.log,
    MIND_NERVE_TELEMETRY: path.join(path.dirname(args.log), "telemetry.jsonl"),
    MIND_NERVE_AGENT_DIRS: "",
    MIND_NERVE_SOCKET_TIMEOUT: "2.0",
    MIND_NERVE_CORE_SKILLS: "mind-nerve-router",
    ...(args.extra ?? {}),
  };
}

function route(
  name: string,
  score: number,
  extra: Record<string, unknown> = {},
): Record<string, unknown> {
  return { name, score, source_repo: "starga", kind: "skill", ...extra };
}

afterEach(async () => {
  while (tmpDirs.length > 0) {
    const d = tmpDirs.pop();
    if (d !== undefined) await fs.rm(d, { recursive: true, force: true });
  }
});

beforeAll(async () => {
  await fs.access(HOOK);
});

// ---------------------------------------------------------------------------
// Decision 1 — the absolute score floor is 0.40 (recalibrated 2026-08-08)
// ---------------------------------------------------------------------------
//
// CALIBRATION FIXTURE (n=78, live daemon, 1437-row table):
//   positives (60, frontmatter trigger sentences with the skill name stripped)
//       top1  p05 0.741   p50 0.912   p95 0.959     rank-1 correct 58/60 (97%)
//   negatives (18, real non-intent prompts from the hook log)
//       top1  p50 0.375   p95 0.476   max 0.617
// Retrieval was 97% rank-1 correct -- the GATE was the defect, not the ranker.
//
// WHY THE FLOOR IS 0.40 AND NOT THE NEGATIVE p95 (0.476):
// That n=78 fixture built its positives from each skill's OWN description text,
// which real user queries do not resemble. Measured against real paraphrases the
// bands overlap heavily (paraphrase p50 0.464 vs noise max 0.617), and THREE OF
// SIX real rank-1-correct hits land below 0.476 -- including `diagnose` at 0.456,
// the very case once cited as proof the floor worked. A floor that rejects
// correct top-1 matches is worse than one that admits some noise, so the default
// is 0.40. See the decision record in integrations/hook/mind-nerve-hook.
//
// The deliberate trade-off: at 0.40 two of the five observed noise cases
// (`yeet` 0.45, `check-work` 0.44) DO survive the floor. They are caught by the
// later gates, not this one. These tests encode that honestly rather than
// asserting a floor that would silently drop real hits.

/** The five bad injections actually observed in production, with their scores. */
const OBSERVED_NOISE: ReadonlyArray<readonly [string, number]> = [
  ["apple-reminders", 0.38],
  ["yeet", 0.45],
  ["grill-me", 0.398],
  ["building-patch-tuesday", 0.376],
  ["check-work", 0.44],
];

const CALIBRATION = {
  positiveP05: 0.741,
  positiveP50: 0.912,
  negativeP50: 0.375,
  negativeP95: 0.476,
  negativeMax: 0.617,
  /** The shipped default floor (recalibrated 2026-08-08 — see header). */
  shippedFloor: 0.4,
} as const;

/** Noise cases the shipped 0.40 floor rejects (score < 0.40). */
const NOISE_BELOW_FLOOR = OBSERVED_NOISE.filter(
  ([, s]) => s < CALIBRATION.shippedFloor,
);
/** Noise cases that survive 0.40 and are caught by the LATER gates, not this one. */
const NOISE_ABOVE_FLOOR = OBSERVED_NOISE.filter(
  ([, s]) => s >= CALIBRATION.shippedFloor,
);

describe("decision 1: absolute score floor", () => {
  it("the shipped default rejects the sub-floor noise that 0.35 admitted", async () => {
    const root = await makeTmp();
    const names = OBSERVED_NOISE.map(([n]) => n);
    const hub = await makeHub(root, ["mind-nerve-router", ...names]);
    const daemon = await startDaemon(
      root,
      JSON.stringify({ routes: OBSERVED_NOISE.map(([n, s]) => route(n, s)) }),
    );
    try {
      const projected = path.join(root, "skills");
      const stdin = JSON.stringify({
        prompt: "please investigate the failing deployment pipeline",
      });

      // Shipped default (0.40): every noise case BELOW the floor is rejected.
      // The two above it (yeet 0.45, check-work 0.44) are this floor's known,
      // documented cost -- later gates handle them.
      const strict = await runHook(
        stdin,
        prodEnv({ hub, projected, socket: daemon.socketPath, log: path.join(root, "a.log") }),
      );
      const strictRows = injectedRows(contextOf(strict));
      for (const [name] of NOISE_BELOW_FLOOR) {
        expect(strictRows.join("\n")).not.toContain(name);
      }
      expect(strictRows.length).toBeLessThanOrEqual(NOISE_ABOVE_FLOOR.length);

      // The old 0.35: all five admitted. This is the regression guard that
      // documents WHY the constant changed -- if someone reverts the default,
      // the first assertion above fails and this one still passes.
      const loose = await runHook(
        stdin,
        prodEnv({
          hub,
          projected: path.join(root, "skills2"),
          socket: daemon.socketPath,
          log: path.join(root, "b.log"),
          extra: { MIND_NERVE_MIN_SCORE: "0.35" },
        }),
      );
      expect(injectedRows(contextOf(loose))).toHaveLength(OBSERVED_NOISE.length);
    } finally {
      await daemon.close();
    }
  });

  it("admits the calibrated positive band and rejects the negative band", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "good", "bad"]);
    const daemon = await startDaemon(
      root,
      JSON.stringify({
        routes: [
          route("good", CALIBRATION.positiveP05),
          route("bad", CALIBRATION.negativeP50),
        ],
      }),
    );
    try {
      const ctx = contextOf(
        await runHook(
          JSON.stringify({ prompt: "help me debug this failing integration test" }),
          prodEnv({
            hub,
            projected: path.join(root, "skills"),
            socket: daemon.socketPath,
            log: path.join(root, "h.log"),
          }),
        ),
      );
      // p05 of positives (0.741) clears the floor; the median noise score
      // (0.375) does not. On the calibration set the bands are separated by
      // 0.124 (positives bottom out at 0.741, noise tops out at 0.617).
      expect(ctx).toContain("`good`");
      expect(ctx).not.toContain("`bad`");
      expect(CALIBRATION.positiveP05).toBeGreaterThan(CALIBRATION.negativeMax);
    } finally {
      await daemon.close();
    }
  });

  it("admits the noise tail above p95 — the floor alone is not the filter", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "tail"]);
    // The floor is the noise 95th percentile, so ~5% of noise clears it BY
    // CONSTRUCTION — the calibration's own max noise score is 0.617. This is
    // exactly why the intent gate is load-bearing rather than redundant: the
    // prompts that produce this tail are harness-shaped text the gate rejects
    // before it is ever embedded.
    const daemon = await startDaemon(
      root,
      JSON.stringify({ routes: [route("tail", CALIBRATION.negativeMax)] }),
    );
    try {
      const ctx = contextOf(
        await runHook(
          JSON.stringify({ prompt: "help me debug this failing integration test" }),
          prodEnv({
            hub,
            projected: path.join(root, "skills"),
            socket: daemon.socketPath,
            log: path.join(root, "h.log"),
          }),
        ),
      );
      expect(CALIBRATION.negativeMax).toBeGreaterThan(CALIBRATION.negativeP95);
      expect(ctx).toContain("`tail`");
    } finally {
      await daemon.close();
    }
  });

  it("keeps the floor in one env-overridable constant", async () => {
    const src = await fs.readFile(HOOK, "utf8");
    expect(src).toContain('MIN_SCORE = _env_float("MIND_NERVE_MIN_SCORE", 0.40)');
    // Exactly one definition -- recalibration must be a one-line diff.
    expect(src.match(/^MIN_SCORE = /gm)).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// Decision 2 — NO Lowe's-ratio gate on injection
// ---------------------------------------------------------------------------
//
// Measured over the same n=78:
//     positives  ratio p50 0.782  p95 1.000
//     negatives  ratio p50 0.973  p05 0.857
// The bands overlap almost completely, so ANY cutoff rejects a large slice of
// CORRECT routes. An earlier cut of 0.85 was perfect on n=7 and does not
// survive n=78 -- legitimate near-ties are COMMON, which is exactly the
// margin-gate-versus-near-duplicates conflict that was predicted.

describe("decision 2: no ratio gate on injection", () => {
  it("injects a near-tie (ratio ~1.0) instead of suppressing it", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "first", "second"]);
    // ratio = 0.80/0.81 = 0.988 -- above every cutoff ever proposed (0.85, 0.92).
    const daemon = await startDaemon(
      root,
      JSON.stringify({ routes: [route("first", 0.81), route("second", 0.8)] }),
    );
    try {
      const ctx = contextOf(
        await runHook(
          JSON.stringify({ prompt: "please refactor the payment module for clarity" }),
          prodEnv({
            hub,
            projected: path.join(root, "skills"),
            socket: daemon.socketPath,
            log: path.join(root, "h.log"),
          }),
        ),
      );
      expect(injectedRows(ctx)).toHaveLength(2);
      expect(ctx).toContain("`first`");
      expect(ctx).toContain("`second`");
    } finally {
      await daemon.close();
    }
  });

  it("has no ratio gate wired into the pipeline", async () => {
    const src = await fs.readFile(HOOK, "utf8");
    expect(src).toContain("APPLY_RATIO_GATE = False");
    // The ratio is computed for telemetry, never applied.
    expect(src).toContain("def top2_ratio(");
    expect(src).not.toContain("apply_margin_gate(");
  });

  it("still records the ratio as telemetry", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "first", "second"]);
    const daemon = await startDaemon(
      root,
      JSON.stringify({ routes: [route("first", 0.9), route("second", 0.72)] }),
    );
    try {
      const telemetry = path.join(root, "telemetry.jsonl");
      await runHook(
        JSON.stringify({ prompt: "please optimise the database query plan" }),
        prodEnv({
          hub,
          projected: path.join(root, "skills"),
          socket: daemon.socketPath,
          log: path.join(root, "h.log"),
          extra: { MIND_NERVE_TELEMETRY: telemetry },
        }),
      );
      const line = JSON.parse((await fs.readFile(telemetry, "utf8")).trim());
      expect(line.top2_ratio).toBeCloseTo(0.8, 3);
    } finally {
      await daemon.close();
    }
  });
});

// ---------------------------------------------------------------------------
// Decision 3 — intent gate, BEFORE embedding
// ---------------------------------------------------------------------------

describe("decision 3: intent gate", () => {
  const NON_INTENT = [
    "ok",
    "yes",
    "all 3",
    "proceed",
    "continue",
    "next",
    "thanks",
    "go on",
    "sure",
    "You are a personal assistant running inside OpenClaw. ## Tooling",
    "You are a context summarization assistant. Your task is to read a conversation",
    '[SYSTEM NOTIFICATION - NOT USER INPUT] This is an automated background-task event',
    'Background command "Poll floor agents" completed (exit code 0)',
  ];

  it.each(NON_INTENT)("skips routing for %j", async (prompt) => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "alpha"]);
    const daemon = await startDaemon(
      root,
      JSON.stringify({ routes: [route("alpha", 0.99)] }),
    );
    try {
      const run = await runHook(
        JSON.stringify({ prompt }),
        prodEnv({
          hub,
          projected: path.join(root, "skills"),
          socket: daemon.socketPath,
          log: path.join(root, "h.log"),
        }),
      );
      // Emits {} — no injection at all, even though the daemon would have
      // returned a 0.99 route. The gate runs BEFORE the daemon is consulted.
      expect(run.code).toBe(0);
      expect(run.stdout.trim()).toBe("{}");
    } finally {
      await daemon.close();
    }
  });

  it("routes a genuine request", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "alpha"]);
    const daemon = await startDaemon(
      root,
      JSON.stringify({ routes: [route("alpha", 0.99)] }),
    );
    try {
      const ctx = contextOf(
        await runHook(
          JSON.stringify({
            prompt: "please deploy the alpha service to staging and verify health",
          }),
          prodEnv({
            hub,
            projected: path.join(root, "skills"),
            socket: daemon.socketPath,
            log: path.join(root, "h.log"),
          }),
        ),
      );
      expect(ctx).toContain("`alpha`");
    } finally {
      await daemon.close();
    }
  });

  it("leaves a previous projection intact when it skips", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "alpha"]);
    const projected = path.join(root, "skills");
    const daemon = await startDaemon(
      root,
      JSON.stringify({ routes: [route("alpha", 0.99)] }),
    );
    try {
      const env = prodEnv({
        hub,
        projected,
        socket: daemon.socketPath,
        log: path.join(root, "h.log"),
      });
      await runHook(
        JSON.stringify({ prompt: "please deploy the alpha service to staging" }),
        env,
      );
      const before = (await fs.readdir(projected)).sort();
      expect(before).toContain("alpha");

      await runHook(JSON.stringify({ prompt: "ok" }), env);
      const after = (await fs.readdir(projected)).sort();
      // The user did not ask for anything new, so the skills from their actual
      // last request must survive untouched.
      expect(after).toEqual(before);
    } finally {
      await daemon.close();
    }
  });

  it("evaluates only the newest user turn, not an injected preamble", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "alpha"]);
    const daemon = await startDaemon(
      root,
      JSON.stringify({ routes: [route("alpha", 0.99)] }),
    );
    try {
      const run = await runHook(
        JSON.stringify({
          prompt:
            "You are a helpful assistant running inside a harness.\n\n" +
            "## Tooling\nTool availability is described below.\n\n" +
            "ok",
        }),
        prodEnv({
          hub,
          projected: path.join(root, "skills"),
          socket: daemon.socketPath,
          log: path.join(root, "h.log"),
        }),
      );
      // The trailing real turn is "ok" — a bare ack. The preamble must not be
      // what gets routed on, which is the observed production failure.
      expect(run.stdout.trim()).toBe("{}");
    } finally {
      await daemon.close();
    }
  });

  it("is ON by default and can be disabled explicitly", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "alpha"]);
    const daemon = await startDaemon(
      root,
      JSON.stringify({ routes: [route("alpha", 0.99)] }),
    );
    try {
      const base = {
        hub,
        projected: path.join(root, "skills"),
        socket: daemon.socketPath,
        log: path.join(root, "h.log"),
      };
      const on = await runHook(JSON.stringify({ prompt: "ok" }), prodEnv(base));
      expect(on.stdout.trim()).toBe("{}");

      const off = await runHook(
        JSON.stringify({ prompt: "ok" }),
        prodEnv({ ...base, extra: { MIND_NERVE_INTENT_GATE: "0" } }),
      );
      expect(contextOf(off)).toContain("`alpha`");
    } finally {
      await daemon.close();
    }
  });
});

// ---------------------------------------------------------------------------
// Decision 5 — atomic, generation-keyed projection
// ---------------------------------------------------------------------------

describe("decision 5: generation-keyed atomic projection", () => {
  it("publishes via a symlink to a content-keyed generation dir", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "alpha"]);
    const projected = path.join(root, "skills");
    const daemon = await startDaemon(
      root,
      JSON.stringify({ routes: [route("alpha", 0.99)] }),
    );
    try {
      await runHook(
        JSON.stringify({ prompt: "please deploy the alpha service to staging" }),
        prodEnv({ hub, projected, socket: daemon.socketPath, log: path.join(root, "h.log") }),
      );
      const st = await fs.lstat(projected);
      expect(st.isSymbolicLink()).toBe(true);
      const target = await fs.readlink(projected);
      expect(path.basename(path.dirname(target))).toBe(".mind-nerve-gen");
      expect(path.basename(target)).toMatch(/^[0-9a-f]{16}$/);
    } finally {
      await daemon.close();
    }
  });

  it("an unchanged route set is a zero-work no-op (same generation)", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "alpha"]);
    const projected = path.join(root, "skills");
    const daemon = await startDaemon(
      root,
      JSON.stringify({ routes: [route("alpha", 0.99)] }),
    );
    try {
      const env = prodEnv({
        hub,
        projected,
        socket: daemon.socketPath,
        log: path.join(root, "h.log"),
      });
      const p = JSON.stringify({
        prompt: "please deploy the alpha service to staging",
      });
      await runHook(p, env);
      const first = await fs.readlink(projected);
      const firstIno = (await fs.stat(first)).ino;

      await runHook(p, env);
      const second = await fs.readlink(projected);
      expect(second).toBe(first);
      expect((await fs.stat(second)).ino).toBe(firstIno);
    } finally {
      await daemon.close();
    }
  });

  it("never deletes a pre-existing real directory — it renames it aside", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "alpha"]);
    const projected = path.join(root, "skills");
    // Simulate the pre-upgrade layout: a REAL dir holding user content.
    await fs.mkdir(projected, { recursive: true });
    await fs.writeFile(path.join(projected, "PRECIOUS.md"), "user data");

    const daemon = await startDaemon(
      root,
      JSON.stringify({ routes: [route("alpha", 0.99)] }),
    );
    try {
      await runHook(
        JSON.stringify({ prompt: "please deploy the alpha service to staging" }),
        prodEnv({ hub, projected, socket: daemon.socketPath, log: path.join(root, "h.log") }),
      );
      expect((await fs.lstat(projected)).isSymbolicLink()).toBe(true);
      const siblings = await fs.readdir(root);
      const backup = siblings.find((s) => s.includes("pre-mind-nerve"));
      expect(backup).toBeDefined();
      expect(
        await fs.readFile(path.join(root, backup as string, "PRECIOUS.md"), "utf8"),
      ).toBe("user data");
    } finally {
      await daemon.close();
    }
  });
});

// ---------------------------------------------------------------------------
// Decision 7 — `.system/` is excluded
// ---------------------------------------------------------------------------

describe("decision 7: .system exclusion", () => {
  it("drops a route pointing into .system and keeps the real skill", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "skill-creator"]);
    // The exact production collision: same name, two source paths.
    await fs.mkdir(path.join(hub, ".system", "skill-creator"), { recursive: true });
    await fs.writeFile(
      path.join(hub, ".system", "skill-creator", "SKILL.md"),
      "---\nname: skill-creator\ndescription: hub internal\n---\n",
    );
    const daemon = await startDaemon(
      root,
      JSON.stringify({
        routes: [
          route("skill-creator", 0.95, {
            source_path: path.join(hub, ".system", "skill-creator", "SKILL.md"),
          }),
          route("skill-creator", 0.94, {
            source_path: path.join(hub, "skill-creator", "SKILL.md"),
          }),
        ],
      }),
    );
    try {
      const projected = path.join(root, "skills");
      const ctx = contextOf(
        await runHook(
          JSON.stringify({ prompt: "please create a new skill for the hub" }),
          prodEnv({ hub, projected, socket: daemon.socketPath, log: path.join(root, "h.log") }),
        ),
      );
      // One row, and its path is the REAL skill, never the .system shadow.
      expect(injectedRows(ctx)).toHaveLength(1);
      expect(ctx).toContain(path.join(hub, "skill-creator", "SKILL.md"));
      expect(ctx).not.toContain(path.join(".system", "skill-creator"));

      // The projected symlink resolves to the real skill too.
      const link = await fs.realpath(path.join(projected, "skill-creator"));
      expect(link).toBe(await fs.realpath(path.join(hub, "skill-creator")));
    } finally {
      await daemon.close();
    }
  });

  it("never projects a hub dot-directory", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "alpha"]);
    await fs.mkdir(path.join(hub, ".system", "hidden"), { recursive: true });
    await fs.writeFile(
      path.join(hub, ".system", "hidden", "SKILL.md"),
      "---\nname: hidden\n---\n",
    );
    const daemon = await startDaemon(
      root,
      JSON.stringify({ routes: [route("hidden", 0.99), route("alpha", 0.98)] }),
    );
    try {
      const projected = path.join(root, "skills");
      await runHook(
        JSON.stringify({ prompt: "please deploy the alpha service to staging" }),
        prodEnv({ hub, projected, socket: daemon.socketPath, log: path.join(root, "h.log") }),
      );
      const entries = await fs.readdir(projected);
      expect(entries).not.toContain("hidden");
      expect(entries).toContain("alpha");
    } finally {
      await daemon.close();
    }
  });
});

// ---------------------------------------------------------------------------
// Item 7 — telemetry feedback loop
// ---------------------------------------------------------------------------

describe("telemetry", () => {
  it("records one decision line with gate, scores and generation", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "alpha"]);
    const telemetry = path.join(root, "telemetry.jsonl");
    const daemon = await startDaemon(
      root,
      JSON.stringify({ routes: [route("alpha", 0.99)] }),
    );
    try {
      await runHook(
        JSON.stringify({ prompt: "please deploy the alpha service to staging" }),
        prodEnv({
          hub,
          projected: path.join(root, "skills"),
          socket: daemon.socketPath,
          log: path.join(root, "h.log"),
          extra: { MIND_NERVE_TELEMETRY: telemetry },
        }),
      );
      const line = JSON.parse((await fs.readFile(telemetry, "utf8")).trim());
      expect(line.event).toBe("route_decision");
      expect(line.decision).toBe("routed");
      expect(line.min_score).toBeCloseTo(CALIBRATION.shippedFloor, 3);
      expect(line.top[0].name).toBe("alpha");
      expect(line.generation).toMatch(/^[0-9a-f]{16}$/);
      // The prompt itself is never written — only a digest.
      expect(line.query_sha).toMatch(/^[0-9a-f]{16}$/);
      expect(JSON.stringify(line)).not.toContain("deploy the alpha service");
    } finally {
      await daemon.close();
    }
  });

  it("records a skip decision with the rule that fired", async () => {
    const root = await makeTmp();
    const hub = await makeHub(root, ["mind-nerve-router", "alpha"]);
    const telemetry = path.join(root, "telemetry.jsonl");
    const daemon = await startDaemon(
      root,
      JSON.stringify({ routes: [route("alpha", 0.99)] }),
    );
    try {
      await runHook(
        JSON.stringify({ prompt: "ok" }),
        prodEnv({
          hub,
          projected: path.join(root, "skills"),
          socket: daemon.socketPath,
          log: path.join(root, "h.log"),
          extra: { MIND_NERVE_TELEMETRY: telemetry },
        }),
      );
      const line = JSON.parse((await fs.readFile(telemetry, "utf8")).trim());
      expect(line.decision).toBe("skip");
      expect(line.reason).toBe("bare_ack");
    } finally {
      await daemon.close();
    }
  });
});
