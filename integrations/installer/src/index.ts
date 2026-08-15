// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.
// TODO(Phase 2): port installer to mind-dev (MIND language native).

import os from "node:os";
import path from "node:path";
import fs from "node:fs/promises";
import { existsSync, realpathSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { AGENT_REGISTRY, ALL_CLIENT_NAMES, requireSpec } from "./registry.js";
import { detectClient, detectAll } from "./detect.js";
import { installClient } from "./install.js";
import { uninstallClient } from "./uninstall.js";
import { runHygiene } from "./hygiene.js";
import { type WireOptions } from "./wire.js";
import {
  verifyClient,
  summarize,
  hasFailures,
  looksInstalled,
  type ClientReport,
} from "./verify.js";
import { type McpLauncher } from "./mcp_rewire.js";
import { InstallerError } from "./errors.js";

// ---------------------------------------------------------------------------
// CLI entry point
// ---------------------------------------------------------------------------

const MIND_NERVE_BIN = path.join(os.homedir(), ".local", "bin", "mind-nerve");

/** integrations/installer/src/ -> integrations/ */
const INTEGRATIONS_ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
  "..",
);

// The compiled entry runs from dist/src/ (one level deeper than src/), so the
// naive "../.." lands on integrations/installer instead of integrations/.
// Walk the candidates and take the first root that actually contains the hook.
function findIntegrationsRoot(): string {
  const here = path.dirname(fileURLToPath(import.meta.url));
  for (const up of ["..", "../..", "../../.."]) {
    const candidate = path.resolve(here, up);
    if (existsSync(path.join(candidate, "hook", "mind-nerve-hook"))) {
      return candidate;
    }
  }
  return INTEGRATIONS_ROOT; // preserve the old error path (ENOENT at read time)
}

const EFFECTIVE_ROOT = findIntegrationsRoot();

const HOOK_SOURCE = path.join(EFFECTIVE_ROOT, "hook", "mind-nerve-hook");
const ROUTER_SKILL = path.join(
  EFFECTIVE_ROOT,
  "hook",
  "assets",
  "mind-nerve-router.SKILL.md",
);

function envOr(key: string, fallback: string): string {
  const v = process.env[key];
  return v !== undefined && v.length > 0 ? v : fallback;
}

/** Default socket: XDG runtime dir when present, else /tmp. Mirrors the hook. */
function defaultSocket(): string {
  const uid = typeof process.getuid === "function" ? process.getuid() : 1000;
  return envOr("MIND_NERVE_SOCKET", `/run/user/${uid}/mind-nerve.sock`);
}

function defaultWireOptions(): WireOptions {
  const home = os.homedir();
  return {
    hookSourcePath: HOOK_SOURCE,
    routerSkillPath: ROUTER_SKILL,
    hubDir: envOr(
      "MIND_NERVE_SOURCE_DIR",
      path.join(home, ".agents", "skills-hub"),
    ),
    socketPath: defaultSocket(),
    agentDirs: [
      path.join(home, ".claude", "agents"),
      path.join(home, ".agents", "agents"),
    ],
  };
}

/** Runtime dir holding route_table.jsonl + the row-aligned route_table.npy. */
function runtimeDir(): string {
  return envOr(
    "MIND_NERVE_RUNTIME_DIR",
    path.join(os.homedir(), ".local", "share", "mind-nerve-runtime"),
  );
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const subcommand = args[0];

  switch (subcommand) {
    case "install":
      await cmdInstall(args.slice(1));
      break;
    case "uninstall":
      await cmdUninstall(args.slice(1));
      break;
    case "verify":
      await cmdVerify(args.slice(1));
      break;
    case "list-clients":
      await cmdListClients();
      break;
    case "status":
      await cmdStatus();
      break;
    case "hygiene":
      await cmdHygiene(args.slice(1));
      break;
    default:
      printUsage();
      process.exit(1);
  }
}

// ---------------------------------------------------------------------------
// mind-nerve install <client>
// mind-nerve install --all
// mind-nerve install --mcp <client>
// mind-nerve install --shared a,b,c
// ---------------------------------------------------------------------------

export interface InstallArgs {
  readonly all: boolean;
  readonly mcpOnly: boolean;
  readonly noWire: boolean;
  readonly sharedClients: readonly string[];
  readonly mcpLauncher: McpLauncher;
  readonly clientArg: string | undefined;
}

export type InstallArgsResult = InstallArgs | { readonly error: string };

/**
 * Parses `install` argv. Flags may precede OR follow the client name —
 * `install --mcp-launcher uvx claude-code` and `install claude-code
 * --mcp-launcher uvx` are equivalent (audit finding 2026-08 r3: the
 * documented flag-first order used to hard-fail because args[0] was taken
 * as the client). `--shared` and `--mcp-launcher` consume the next token as
 * their value; the first remaining non-flag token is the client.
 */
export function parseInstallArgs(args: readonly string[]): InstallArgsResult {
  let mcpLauncher: McpLauncher = "venv";
  let sharedClients: string[] = [];
  const positional: string[] = [];
  for (let i = 0; i < args.length; i++) {
    const tok = args[i];
    if (tok === undefined) continue;
    if (tok === "--shared") {
      const value = args[i + 1];
      if (value !== undefined && !value.startsWith("--")) {
        sharedClients = value.split(",");
        i++;
      }
      continue;
    }
    if (tok === "--mcp-launcher") {
      const value = args[i + 1];
      if (value !== "venv" && value !== "uvx") {
        return {
          error: `mind-nerve: bad --mcp-launcher '${String(value)}' (want venv|uvx)`,
        };
      }
      mcpLauncher = value;
      i++;
      continue;
    }
    if (tok.startsWith("--")) continue; // --all / --mcp / --no-wire
    positional.push(tok);
  }
  return {
    all: args.includes("--all"),
    mcpOnly: args.includes("--mcp"),
    noWire: args.includes("--no-wire"),
    sharedClients,
    mcpLauncher,
    clientArg: positional[0],
  };
}

async function cmdInstall(args: string[]): Promise<void> {
  const parsed = parseInstallArgs(args);
  if ("error" in parsed) {
    process.stderr.write(parsed.error + "\n");
    process.exit(1);
  }
  const flagAll = parsed.all;
  const flagMcp = parsed.mcpOnly;
  const flagNoWire = parsed.noWire;
  const sharedClients = [...parsed.sharedClients];
  const mcpLauncher = parsed.mcpLauncher;

  const sharedProjectionDir =
    sharedClients.length > 0
      ? path.join(os.homedir(), ".mind-nerve", "projections", "shared")
      : undefined;

  // Skill-surface wiring is ON by default — a structural install without the
  // automatic hook leaves the router with nothing to route.
  const wire = flagNoWire || flagMcp ? undefined : defaultWireOptions();

  const buildOpts = (useShared: boolean) => ({
    mindNerveBin: MIND_NERVE_BIN,
    mcpOnly: flagMcp,
    mcpLauncher,
    ...(useShared && sharedProjectionDir !== undefined
      ? { sharedProjectionDir }
      : {}),
    ...(wire !== undefined ? { wire } : {}),
  });

  if (flagAll) {
    const specs = [...AGENT_REGISTRY.values()];
    const detections = await detectAll(specs);
    let installedCount = 0;
    for (const det of detections) {
      if (!det.detected) {
        process.stderr.write(`  skip  ${det.name} (not detected)\n`);
        continue;
      }
      const spec = requireSpec(det.name);
      try {
        const result = await installClient(
          spec,
          buildOpts(sharedClients.includes(det.name)),
        );
        if (result.idempotentNoop) {
          process.stderr.write(`  noop  ${det.name} (already installed)\n`);
        } else {
          const w = result.wire;
          const detail =
            w === null
              ? ""
              : ` (skills=${w.skillsDir}${w.skillsBackupPath !== null ? `, backed up -> ${w.skillsBackupPath}` : ""}, hook=${w.hookScriptPath})`;
          process.stderr.write(`  done  ${det.name}${detail}\n`);
          installedCount++;
        }
      } catch (err) {
        process.stderr.write(`  FAIL  ${det.name}: ${String(err)}\n`);
      }
    }
    process.stderr.write(`\nInstalled ${installedCount} client(s).\n`);
    return;
  }

  // Single client install.
  const clientArg = parsed.clientArg;
  if (clientArg === undefined) {
    process.stderr.write("Usage: mind-nerve install [--all] [--mcp] [--shared a,b] <client>\n");
    process.exit(1);
  }

  const spec = resolveSpec(clientArg);
  const result = await installClient(
    spec,
    buildOpts(sharedClients.includes(clientArg)),
  );

  if (result.idempotentNoop) {
    process.stderr.write(`mind-nerve: ${clientArg} already installed (no-op)\n`);
  } else {
    process.stderr.write(`mind-nerve: installed ${clientArg}\n`);
    for (const bak of result.backedUp) {
      process.stderr.write(`  backed up: ${bak}\n`);
    }
    if (result.wire !== null) {
      process.stderr.write(`  skills dir: ${result.wire.skillsDir}\n`);
      if (result.wire.skillsBackupPath !== null) {
        process.stderr.write(
          `  previous skills dir moved to: ${result.wire.skillsBackupPath}\n`,
        );
      }
      process.stderr.write(`  hook: ${result.wire.hookScriptPath}\n`);
    }
  }
}

// ---------------------------------------------------------------------------
// mind-nerve hygiene [--dry-run] [--repoint FROM=TO]... [--no-prune]
// ---------------------------------------------------------------------------
async function cmdHygiene(args: string[]): Promise<void> {
  const dryRun = args.includes("--dry-run");
  const prune = !args.includes("--no-prune");

  const repoint: Array<readonly [string, string]> = [];
  for (let i = 0; i < args.length; i++) {
    if (args[i] !== "--repoint") continue;
    const pair = args[i + 1];
    if (pair === undefined) continue;
    const eq = pair.indexOf("=");
    if (eq <= 0) {
      process.stderr.write(`mind-nerve: bad --repoint '${pair}' (want FROM=TO)\n`);
      process.exit(1);
    }
    repoint.push([pair.slice(0, eq), pair.slice(eq + 1)] as const);
  }

  const dir = runtimeDir();
  const tablePath = path.join(dir, "route_table.jsonl");
  const embeddingsPath = path.join(dir, "route_table.npy");

  let hasEmbeddings = true;
  try {
    await fs.access(embeddingsPath);
  } catch {
    hasEmbeddings = false;
  }

  const report = await runHygiene({
    tablePath,
    embeddingsPath: hasEmbeddings ? embeddingsPath : null,
    repoint,
    pruneDead: prune,
    dryRun,
  });

  process.stdout.write(
    [
      `route table:      ${tablePath}`,
      `embeddings:       ${hasEmbeddings ? embeddingsPath : "(none)"}`,
      `rows before:      ${report.rowsBefore}` +
        (report.embeddingRowsBefore !== null
          ? ` (embeddings ${report.embeddingRowsBefore})`
          : ""),
      `repointed:        ${report.repointed}`,
      `pruned (dead):    ${report.pruned}`,
      `rows after:       ${report.rowsAfter}` +
        (report.embeddingRowsAfter !== null
          ? ` (embeddings ${report.embeddingRowsAfter})`
          : ""),
      `wrote:            ${report.wrote ? "yes" : "no"}`,
      ...report.backups.map((b) => `  backup: ${b}`),
      "",
    ].join("\n"),
  );

  if (report.prunedNames.length > 0) {
    process.stdout.write(
      "pruned routes:\n" +
        report.prunedNames.map((n) => `  ${n}`).join("\n") +
        "\n",
    );
  }
}

// ---------------------------------------------------------------------------
// mind-nerve uninstall <client>
// ---------------------------------------------------------------------------
async function cmdUninstall(args: string[]): Promise<void> {
  const flagAll = args.includes("--all");

  if (flagAll) {
    for (const name of ALL_CLIENT_NAMES) {
      const spec = requireSpec(name);
      try {
        const result = await uninstallClient(spec, {});
        if (result.changed) {
          process.stderr.write(`  done  ${name}\n`);
        } else {
          process.stderr.write(`  skip  ${name} (not installed)\n`);
        }
      } catch (err) {
        process.stderr.write(`  FAIL  ${name}: ${String(err)}\n`);
      }
    }
    return;
  }

  const clientArg = args[0];
  if (clientArg === undefined || clientArg.startsWith("--")) {
    process.stderr.write("Usage: mind-nerve uninstall [--all] <client>\n");
    process.exit(1);
  }

  const spec = resolveSpec(clientArg);
  const result = await uninstallClient(spec, {});
  if (result.changed) {
    process.stderr.write(`mind-nerve: uninstalled ${clientArg}\n`);
    for (const p of result.restoredPaths) {
      process.stderr.write(`  restored: ${p}\n`);
    }
  } else {
    process.stderr.write(`mind-nerve: ${clientArg} was not installed\n`);
  }
}

// ---------------------------------------------------------------------------
// mind-nerve verify [--cli <name|all>] [--json]
// ---------------------------------------------------------------------------
async function cmdVerify(args: string[]): Promise<void> {
  const flagJson = args.includes("--json");
  const cliIdx = args.indexOf("--cli");
  const cliArg = cliIdx !== -1 ? args[cliIdx + 1] : undefined;
  if (cliIdx !== -1 && (cliArg === undefined || cliArg.startsWith("--"))) {
    process.stderr.write("Usage: mind-nerve verify [--cli <name|all>] [--json]\n");
    process.exit(1);
  }

  const specs = [...AGENT_REGISTRY.values()];
  let targets;
  if (cliArg === "all") {
    targets = specs;
  } else if (cliArg !== undefined) {
    targets = [resolveSpec(cliArg)];
  } else {
    // No --cli: verify only clients that have something on disk.
    const installed = await Promise.all(
      specs.map(async (s) => ((await looksInstalled(s, process.cwd())) ? s : null)),
    );
    targets = installed.filter((s) => s !== null);
    if (targets.length === 0) {
      process.stderr.write(
        "mind-nerve: no installs found on this host — nothing to verify\n",
      );
      if (flagJson) {
        process.stdout.write(
          JSON.stringify({ clients: [], summary: summarize([]) }, null, 2) + "\n",
        );
      }
      return;
    }
  }

  const reports: ClientReport[] = [];
  for (const spec of targets) {
    reports.push(await verifyClient(spec, { socketPath: defaultSocket() }));
  }

  const summary = summarize(reports);

  if (flagJson) {
    process.stdout.write(
      JSON.stringify({ clients: reports, summary }, null, 2) + "\n",
    );
  } else {
    for (const report of reports) {
      process.stdout.write(`${report.client}: ${report.status}\n`);
      for (const check of report.checks) {
        process.stdout.write(
          `  ${check.status.padEnd(4)}  ${check.name}: ${check.message}\n`,
        );
        if (check.hint !== null) {
          process.stdout.write(`        hint: ${check.hint}\n`);
        }
      }
    }
    process.stdout.write(
      `\nsummary: ${summary.clients} client(s), ` +
        `${summary.pass} PASS, ${summary.warn} WARN, ${summary.fail} FAIL\n`,
    );
  }

  if (hasFailures(reports)) {
    process.exitCode = 1;
  }
}

// ---------------------------------------------------------------------------
// mind-nerve list-clients
// ---------------------------------------------------------------------------
async function cmdListClients(): Promise<void> {
  const specs = [...AGENT_REGISTRY.values()];
  const detections = await detectAll(specs);

  // ANSI-colour-free output — one line per client.
  const header = `${"CLIENT".padEnd(16)} ${"STATUS".padEnd(12)} DESCRIPTION`;
  process.stdout.write(header + "\n");
  process.stdout.write("-".repeat(header.length) + "\n");

  for (const det of detections) {
    const spec = AGENT_REGISTRY.get(det.name);
    const status = det.alwaysOffer
      ? "always-offer"
      : det.detected
        ? "detected"
        : "not-found";
    const desc = spec?.description ?? "";
    process.stdout.write(
      `${det.name.padEnd(16)} ${status.padEnd(12)} ${desc}\n`,
    );
  }
}

// ---------------------------------------------------------------------------
// mind-nerve status
// ---------------------------------------------------------------------------
async function cmdStatus(): Promise<void> {
  const specs = [...AGENT_REGISTRY.values()];
  const detections = await detectAll(specs);

  const header = `${"CLIENT".padEnd(16)} ${"DETECTED".padEnd(10)} ${"BINARY".padEnd(20)} CONFIG_PATH`;
  process.stdout.write(header + "\n");
  process.stdout.write("-".repeat(header.length + 20) + "\n");

  for (const det of detections) {
    const spec = AGENT_REGISTRY.get(det.name);
    const detected = det.detected ? "yes" : "no";
    const bin = det.foundBinary ?? det.foundPath ?? "-";
    const cfgPath = spec?.configPath ?? "-";
    process.stdout.write(
      `${det.name.padEnd(16)} ${detected.padEnd(10)} ${bin.slice(-20).padEnd(20)} ${cfgPath}\n`,
    );
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function resolveSpec(clientName: string) {
  try {
    return requireSpec(clientName);
  } catch {
    process.stderr.write(`mind-nerve: unknown client '${clientName}'\n`);
    process.stderr.write(`Known clients: ${ALL_CLIENT_NAMES.join(", ")}\n`);
    process.exit(1);
  }
}

function printUsage(): void {
  process.stderr.write(`
mind-nerve installer

Usage:
  mind-nerve install <client>            Install one client
  mind-nerve install --all               Detect + install every CLI present
  mind-nerve install --mcp <client>      MCP-only mode (skip skill projection)
  mind-nerve install --no-wire <client>  Skip the skills dir + hook wiring
  mind-nerve install --shared a,b,c      STARGA power-user: shared projection dir
  mind-nerve install --mcp-launcher venv|uvx <client>
                                         venv (default): pin the absolute venv
                                         binary path — zero first-launch cost,
                                         works offline, needs the venv present.
                                         uvx: write 'uvx --from mind-nerve
                                         mind-nerve-mcp' — no pre-built venv
                                         needed, but first launch downloads the
                                         package; offline hosts stay on venv.
  mind-nerve uninstall <client>          Reverse install, restore backup
  mind-nerve uninstall --all             Uninstall all clients
  mind-nerve verify [--cli <name|all>] [--json]
                                         Self-test existing installs: config
                                         parses + managed markers, hook script
                                         fail-open probe, skills dir state, MCP
                                         entry + command resolution, env pins,
                                         daemon socket (WARN only). Exits
                                         non-zero iff any check FAILs.
  mind-nerve list-clients                Show all clients + detection status
  mind-nerve status                      Show which installs are active
  mind-nerve hygiene [--dry-run]         Repoint + prune the route table and
                                         its ROW-ALIGNED .npy in lockstep
    --repoint FROM=TO                    Rewrite a source_path prefix
    --no-prune                           Repoint only; keep dead routes

Install wires each detected CLI with a skill surface:
  * its skills dir becomes a REAL directory holding only mind-nerve-router
    (announce drops from ~115k tokens to ~2k; the hub is untouched), and
  * the shared CLI-agnostic hook is registered on UserPromptSubmit +
    SessionStart, so relevant skills are projected and injected per prompt.

A skills path that is a symlink is unlinked; a REAL directory is never deleted,
only moved to <dir>.bak-mind-nerve-<ts>. 'uninstall' restores it exactly,
including re-creating the hub symlink.

Migration:
  If you have a STARGA shared ~/.agents/skills/ setup, use --shared to opt
  into one projection directory instead of per-CLI projections.

  Uninstall is always reversible via .bak files. To wipe all state:
    mind-nerve uninstall --all && rm -rf ~/.mind-nerve/

Known clients: ${ALL_CLIENT_NAMES.join(", ")}
`);
}

// Run — only when executed directly, so tests can import parseInstallArgs
// without booting the CLI. realpathSync resolves the npm-bin symlink before
// comparing against this module's URL.
const isDirectRun =
  process.argv[1] !== undefined &&
  import.meta.url === pathToFileURL(realpathSync(process.argv[1])).href;

if (isDirectRun) {
  main().catch((err: unknown) => {
    if (err instanceof InstallerError) {
      process.stderr.write(`mind-nerve installer error [${err.code}] (${err.clientName}): ${err.message}\n`);
    } else {
      process.stderr.write(`mind-nerve installer fatal: ${String(err)}\n`);
    }
    process.exit(1);
  });
}
