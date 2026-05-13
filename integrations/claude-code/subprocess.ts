// mind-nerve Claude Code UserPromptSubmit hook. Projector pattern (D1).
//
// Wire protocol aligned to cli/main.mind commit cd8591b:
//   stdin  — mic@2 line-oriented text frame
//   stdout — mic-b fixed-shape little-endian binary frame
//   stderr — mic@2 error frame (code: + detail: keys)

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import crypto from "node:crypto";
import { z } from "zod";

export const SubprocessInputSchema = z.object({
  current_prompt: z.string(),
  registry_summary: z.array(
    z.object({
      id: z.string(),
      excerpt: z.string(),
    }),
  ),
  catalog_hash: z.string(),
  k: z.number().int().min(1),
  threshold: z.number().min(0).max(1),
});

export type SubprocessInput = z.infer<typeof SubprocessInputSchema>;

export const SubprocessOutputSchema = z.discriminatedUnion("outcome", [
  z.object({
    outcome: z.literal("top_k"),
    selected: z.array(z.string()),
    scores: z.array(z.number()).optional(),
    version: z.number().int().optional(),
  }),
  z.object({
    outcome: z.literal("low_confidence"),
    reason: z.string().optional(),
  }),
  z.object({
    outcome: z.literal("passthrough"),
    reason: z.string().optional(),
  }),
  z.object({
    outcome: z.literal("error"),
    message: z.string().optional(),
  }),
]);

export type SubprocessOutput = z.infer<typeof SubprocessOutputSchema>;

// ---------------------------------------------------------------------------
// Binary resolution
// ---------------------------------------------------------------------------

/** Returns true if the mind-nerve binary is reachable. */
export function isBinaryAvailable(binaryPath: string): boolean {
  if (binaryPath.startsWith("/") || binaryPath.includes("/")) {
    return existsSync(binaryPath);
  }
  // Implicit PATH lookup — assume present; let spawn surface ENOENT.
  return true;
}

// ---------------------------------------------------------------------------
// mic@2 stdin encoder
// ---------------------------------------------------------------------------

// Phase 1.2 reconciliation: The CLI binary's mic@2 wire accepts only
// pre-tokenized token IDs in Phase 1.1. BPE tokenization lives inside
// mind-nerve (Phase 1.2). Until then, the TS shim converts the prompt to a
// byte-level token sequence: each UTF-8 byte of current_prompt becomes one
// token ID. This is the byte-level base of the BPE 32k vocab — mind-nerve
// will re-tokenize once Phase 1.2 stdin parsing lands.
function promptToByteTokens(prompt: string): number[] {
  return Array.from(Buffer.from(prompt, "utf8"));
}

/**
 * Builds the mic@2 text frame sent to mind-nerve on stdin.
 *
 * Frame grammar (from cli/main.mind §WIRE PROTOCOL — stdin):
 *   header: "mic@2/mind-nerve/preselect\n"
 *   key:    "<name>: <value>\n"   (no LF in value)
 *   terminator: ".\n"
 *
 * Required keys (any order between header and terminator):
 *   model, catalog, k, tokens
 *
 * model and catalog paths are derived from SubprocessInput.catalog_hash
 * using the canonical runtime paths. The catalog_hash is embedded in the
 * catalog path to let mind-nerve locate the right manifest without a
 * separate handshake.
 */
export function encodeMic2Frame(input: SubprocessInput): string {
  const tokens = promptToByteTokens(input.current_prompt);
  const lines = [
    "mic@2/mind-nerve/preselect",
    // model: canonical weights path; mind-nerve resolves the real path internally.
    "model: /var/lib/mind-nerve/checkpoint.weights",
    // catalog: embed hash so the binary can locate the versioned manifest.
    `catalog: /var/lib/mind-nerve/catalogs/${input.catalog_hash}.catalog`,
    `k: ${String(input.k)}`,
    `tokens: ${tokens.join(",")}`,
    ".",
    "",  // trailing newline after terminator
  ];
  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// RouteId ↔ externalId mapping
// ---------------------------------------------------------------------------

// RouteId = SHA-256(externalId as UTF-8 bytes), 32 bytes raw.
// The CLI binary hashes the external route name identically.
// We build the reverse map here so decodeStdout can look up route names.
export function buildRouteIdMap(
  registrySummary: ReadonlyArray<{ id: string }>,
): Map<string, string> {
  const m = new Map<string, string>();
  for (const { id } of registrySummary) {
    const sha = crypto.createHash("sha256").update(id, "utf8").digest("hex");
    m.set(sha, id);
  }
  return m;
}

// ---------------------------------------------------------------------------
// mic-b stdout decoder
// ---------------------------------------------------------------------------

// mic-b layout (from cli/main.mind §WIRE PROTOCOL — stdout):
//   offset        size       field
//   0             4          magic "MNB1" (0x4D 0x4E 0x42 0x31)
//   4             2          k (u16 LE)
//   6             32*k       k × RouteId (32 bytes each, raw SHA-256)
//   6+32k         4*k        k × score (i32 LE Q16.16)
//   6+36k         212        attestation envelope v2
//   TOTAL         218+36k

const MNB1_MAGIC = Buffer.from([0x4d, 0x4e, 0x42, 0x31]);
const ROUTE_ID_BYTES = 32;
const SCORE_BYTES = 4;
const ENVELOPE_BYTES = 212;

/**
 * Decodes the mic-b binary stdout from mind-nerve into SubprocessOutput.
 *
 * routeIdMap: hex(SHA-256(externalId)) → externalId, built from the
 * registry summary before spawning. RouteIds not present in the map are
 * logged and skipped (unknown skill registered in catalog but not in
 * current registry_summary — treat as a warn-and-skip, consistent with
 * preselect.ts selectEntries behaviour).
 */
export function decodeMicBFrame(
  buf: Buffer,
  routeIdMap: ReadonlyMap<string, string>,
): SubprocessOutput {
  // Validate magic.
  if (buf.length < 6) {
    throw new Error(`mic-b frame too short: ${String(buf.length)} bytes`);
  }
  if (!buf.subarray(0, 4).equals(MNB1_MAGIC)) {
    throw new Error(
      `mic-b bad magic: ${buf.subarray(0, 4).toString("hex")}`,
    );
  }

  const k = buf.readUInt16LE(4);
  const expectedTotal = 6 + ROUTE_ID_BYTES * k + SCORE_BYTES * k + ENVELOPE_BYTES;
  if (buf.length < expectedTotal) {
    throw new Error(
      `mic-b frame truncated: got ${String(buf.length)}, need ${String(expectedTotal)}`,
    );
  }

  // Parse route IDs.
  const selected: string[] = [];
  for (let i = 0; i < k; i++) {
    const offset = 6 + i * ROUTE_ID_BYTES;
    const routeIdHex = buf.subarray(offset, offset + ROUTE_ID_BYTES).toString("hex");
    const externalId = routeIdMap.get(routeIdHex);
    if (externalId !== undefined) {
      selected.push(externalId);
    }
    // Unknown RouteId: skip silently — caller will warn via preselect.ts.
  }

  // Parse scores — i32 LE Q16.16, convert to JS number.
  const scoresOffset = 6 + ROUTE_ID_BYTES * k;
  const scores: number[] = [];
  for (let i = 0; i < k; i++) {
    const raw = buf.readInt32LE(scoresOffset + i * SCORE_BYTES);
    scores.push(raw / 65536.0);
  }

  // Envelope is parsed for integrity by mind-nerve before stdout is written.
  // The TS shim treats it as opaque — no validation at this layer.
  // (attestation envelope v2, bytes [6+36k, 6+36k+212))

  return { outcome: "top_k", selected, scores };
}

// ---------------------------------------------------------------------------
// mic@2 stderr parser
// ---------------------------------------------------------------------------

/**
 * Parses a mic@2 error frame from stderr.
 *
 * Frame format:
 *   mic@2/mind-nerve/error\n
 *   code: <symbol>\n
 *   detail: <free text>\n
 *   .\n
 *
 * Returns the code symbol, or "unknown" if parsing fails.
 */
export function parseStderrFrame(stderr: string): string {
  for (const line of stderr.split("\n")) {
    const trimmed = line.trim();
    if (trimmed.startsWith("code: ")) {
      return trimmed.slice("code: ".length).trim();
    }
  }
  return "unknown";
}

// ---------------------------------------------------------------------------
// Subprocess invocation with AbortController timeout
// ---------------------------------------------------------------------------

export interface SpawnResult {
  stdout: Buffer;
  stderr: string;
  exitCode: number | null;
}

function spawnWithTimeout(
  binaryPath: string,
  args: string[],
  stdinText: string,
  timeoutMs: number,
): { promise: Promise<SpawnResult>; cancel: () => void } {
  const controller = new AbortController();
  let timeoutHandle: ReturnType<typeof setTimeout> | undefined;

  const promise = new Promise<SpawnResult>((resolve, reject) => {
    let proc: ReturnType<typeof spawn>;
    try {
      proc = spawn(binaryPath, args, {
        stdio: ["pipe", "pipe", "pipe"],
        signal: controller.signal,
      });
    } catch (err) {
      reject(err);
      return;
    }

    timeoutHandle = setTimeout(() => {
      controller.abort();
      reject(new Error(`mind-nerve subprocess timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    const stdoutChunks: Buffer[] = [];
    let stderr = "";

    if (proc.stdout === null || proc.stderr === null || proc.stdin === null) {
      clearTimeout(timeoutHandle);
      reject(new Error("subprocess stdio streams are null"));
      return;
    }

    proc.stdout.on("data", (chunk: Buffer) => stdoutChunks.push(chunk));
    proc.stderr.on("data", (chunk: Buffer) => (stderr += chunk.toString("utf8")));

    proc.on("error", (err) => {
      clearTimeout(timeoutHandle);
      reject(err);
    });

    proc.on("close", (code) => {
      clearTimeout(timeoutHandle);
      const stdout = Buffer.concat(stdoutChunks);
      if (code !== 0) {
        reject(
          new Error(
            `mind-nerve exited with code ${String(code)}: ${stderr.trim()}`,
          ),
        );
        return;
      }
      resolve({ stdout, stderr, exitCode: code });
    });

    proc.stdin.write(stdinText, "utf8");
    proc.stdin.end();
  });

  const cancel = () => {
    clearTimeout(timeoutHandle);
    controller.abort();
  };

  return { promise, cancel };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export class SubprocessTimeoutError extends Error {
  constructor(ms: number) {
    super(`mind-nerve subprocess timed out after ${ms}ms`);
    this.name = "SubprocessTimeoutError";
  }
}

export class BinaryNotFoundError extends Error {
  constructor(path: string) {
    super(`mind-nerve binary not found: ${path}`);
    this.name = "BinaryNotFoundError";
  }
}

/**
 * Calls the mind-nerve binary with the given input. Enforces `timeoutMs`.
 *
 * Sends a mic@2 text frame on stdin.
 * Reads a mic-b binary frame from stdout.
 * On non-zero exit, parses the mic@2 error frame from stderr.
 *
 * Throws:
 *   - BinaryNotFoundError — binary missing (ENOENT from spawn)
 *   - SubprocessTimeoutError — timeout exceeded
 *   - Error — non-zero exit code or frame parse failure
 */
export async function callMindNerve(
  binaryPath: string,
  input: SubprocessInput,
  timeoutMs: number,
): Promise<SubprocessOutput> {
  const stdinText = encodeMic2Frame(input);
  const routeIdMap = buildRouteIdMap(input.registry_summary);

  const { promise, cancel: _cancel } = spawnWithTimeout(
    binaryPath,
    ["preselect"],
    stdinText,
    timeoutMs,
  );

  let result: SpawnResult;
  try {
    result = await promise;
  } catch (err) {
    if (
      err instanceof Error &&
      (err.message.includes("ENOENT") || err.message.includes("not found"))
    ) {
      throw new BinaryNotFoundError(binaryPath);
    }
    if (err instanceof Error && err.message.includes("timed out")) {
      throw new SubprocessTimeoutError(timeoutMs);
    }
    throw err;
  }

  return decodeMicBFrame(result.stdout, routeIdMap);
}
