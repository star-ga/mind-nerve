// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.
// TODO(Phase 2): port installer to mind-dev (MIND language native).

import os from "node:os";
import path from "node:path";
import { AGENT_REGISTRY, ALL_CLIENT_NAMES, requireSpec } from "./registry.js";
import { detectClient, detectAll } from "./detect.js";
import { installClient } from "./install.js";
import { uninstallClient } from "./uninstall.js";
import { InstallerError } from "./errors.js";

// ---------------------------------------------------------------------------
// CLI entry point
// ---------------------------------------------------------------------------

const MIND_NERVE_BIN = path.join(os.homedir(), ".local", "bin", "mind-nerve");

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
    case "list-clients":
      await cmdListClients();
      break;
    case "status":
      await cmdStatus();
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
async function cmdInstall(args: string[]): Promise<void> {
  const flagAll = args.includes("--all");
  const flagMcp = args.includes("--mcp");
  const sharedIdx = args.indexOf("--shared");
  const sharedArg = sharedIdx !== -1 ? args[sharedIdx + 1] : undefined;
  const sharedClients = sharedArg !== undefined ? sharedArg.split(",") : [];

  const sharedProjectionDir =
    sharedClients.length > 0
      ? path.join(os.homedir(), ".mind-nerve", "projections", "shared")
      : undefined;

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
        const installOpts =
        sharedProjectionDir !== undefined
          ? { mindNerveBin: MIND_NERVE_BIN, mcpOnly: flagMcp, sharedProjectionDir }
          : { mindNerveBin: MIND_NERVE_BIN, mcpOnly: flagMcp };
      const result = await installClient(spec, installOpts);
        if (result.idempotentNoop) {
          process.stderr.write(`  noop  ${det.name} (already installed)\n`);
        } else {
          process.stderr.write(`  done  ${det.name}\n`);
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
  const clientArg = flagMcp ? args[args.indexOf("--mcp") + 1] : args[0];
  if (clientArg === undefined || clientArg.startsWith("--")) {
    process.stderr.write("Usage: mind-nerve install [--all] [--mcp] [--shared a,b] <client>\n");
    process.exit(1);
  }

  const spec = resolveSpec(clientArg);
  const useShared =
    sharedClients.includes(clientArg) && sharedProjectionDir !== undefined;

  const singleOpts =
    useShared && sharedProjectionDir !== undefined
      ? { mindNerveBin: MIND_NERVE_BIN, mcpOnly: flagMcp, sharedProjectionDir }
      : { mindNerveBin: MIND_NERVE_BIN, mcpOnly: flagMcp };
  const result = await installClient(spec, singleOpts);

  if (result.idempotentNoop) {
    process.stderr.write(`mind-nerve: ${clientArg} already installed (no-op)\n`);
  } else {
    process.stderr.write(`mind-nerve: installed ${clientArg}\n`);
    for (const bak of result.backedUp) {
      process.stderr.write(`  backed up: ${bak}\n`);
    }
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
  mind-nerve install --shared a,b,c      STARGA power-user: shared projection dir
  mind-nerve uninstall <client>          Reverse install, restore backup
  mind-nerve uninstall --all             Uninstall all clients
  mind-nerve list-clients                Show all 17 clients + detection status
  mind-nerve status                      Show which installs are active

Migration:
  If you have a STARGA shared ~/.agents/skills/ setup, use --shared to opt
  into one projection directory instead of per-CLI projections.

  Uninstall is always reversible via .bak files. To wipe all state:
    mind-nerve uninstall --all && rm -rf ~/.mind-nerve/

Known clients: ${ALL_CLIENT_NAMES.join(", ")}
`);
}

// Run.
main().catch((err: unknown) => {
  if (err instanceof InstallerError) {
    process.stderr.write(`mind-nerve installer error [${err.code}] (${err.clientName}): ${err.message}\n`);
  } else {
    process.stderr.write(`mind-nerve installer fatal: ${String(err)}\n`);
  }
  process.exit(1);
});
