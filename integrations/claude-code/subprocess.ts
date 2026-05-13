// mind-nerve Claude Code UserPromptSubmit hook. Projector pattern (D1).

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { z } from "zod";
import { framePayload, encodeMap, decodeMap, readFrames, type MapFrame } from "@mind/mic-map";

// DONE — using @mind/mic-map@0.1.0
// Wire protocol: MAP frames, length-prefixed for binary-safe stdio.
// Request:  framePayload(encodeMap({kind:"req", op:"preselect", fields:{...}}))
// Response: decodeMap(textDecoder.decode((await readFrames(stream).next()).value))

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
// Subprocess invocation with AbortController timeout
// ---------------------------------------------------------------------------

export interface SpawnResult {
  stdout: string;
  stderr: string;
}

/** Encode SubprocessInput as a MAP-framed binary payload. */
function encodeSubprocessInput(input: SubprocessInput): Uint8Array {
  const fields: Record<string, string | bigint | boolean | string[]> = {
    current_prompt: input.current_prompt,
    catalog_hash: input.catalog_hash,
    k: BigInt(input.k),
    threshold: input.threshold.toString(),
    registry_ids: input.registry_summary.map(e => e.id),
    registry_excerpts: input.registry_summary.map(e => e.excerpt),
  };
  const frame: MapFrame = { kind: "req", op: "preselect", fields };
  return framePayload(encodeMap(frame));
}

/** Decode a MAP response frame to SubprocessOutput. */
function decodeSubprocessOutput(bytes: Uint8Array): SubprocessOutput {
  const text = new TextDecoder().decode(bytes);
  const frame = decodeMap(text);

  if (frame.kind === "ok") {
    const outcome = frame.fields["outcome"];
    if (outcome === "top_k") {
      const selectedRaw = frame.fields["selected"];
      const selected = Array.isArray(selectedRaw)
        ? selectedRaw.map(s => String(s))
        : [String(selectedRaw)];
      return { outcome: "top_k", selected };
    }
    if (outcome === "low_confidence") {
      const reason = frame.fields["reason"];
      return { outcome: "low_confidence", reason: reason !== undefined ? String(reason) : undefined };
    }
    if (outcome === "passthrough") {
      return { outcome: "passthrough" };
    }
  }

  if (frame.kind === "err") {
    const msg = frame.fields["msg"];
    return { outcome: "error", message: msg !== undefined ? String(msg) : frame.code };
  }

  // Fall back to passthrough for any unrecognized frame
  return { outcome: "passthrough" };
}

function spawnWithTimeout(
  binaryPath: string,
  args: string[],
  stdin: Uint8Array,
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

    let stdout = "";
    let stderr = "";

    if (proc.stdout === null || proc.stderr === null || proc.stdin === null) {
      clearTimeout(timeoutHandle);
      reject(new Error("subprocess stdio streams are null"));
      return;
    }

    proc.stdout.on("data", (chunk: Buffer) => (stdout += chunk.toString()));
    proc.stderr.on("data", (chunk: Buffer) => (stderr += chunk.toString()));

    proc.on("error", (err) => {
      clearTimeout(timeoutHandle);
      reject(err);
    });

    proc.on("close", (code) => {
      clearTimeout(timeoutHandle);
      if (code !== 0) {
        reject(
          new Error(
            `mind-nerve exited with code ${String(code)}: ${stderr.trim()}`,
          ),
        );
        return;
      }
      resolve({ stdout, stderr });
    });

    proc.stdin.write(Buffer.from(stdin));
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
 * Throws:
 *   - BinaryNotFoundError — binary missing (ENOENT from spawn)
 *   - SubprocessTimeoutError — timeout exceeded
 *   - Error — non-zero exit code or JSON parse failure
 */
export async function callMindNerve(
  binaryPath: string,
  input: SubprocessInput,
  timeoutMs: number,
): Promise<SubprocessOutput> {
  const framedPayload = encodeSubprocessInput(input);
  const { promise, cancel: _cancel } = spawnWithTimeout(
    binaryPath,
    ["preselect"],
    framedPayload,
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

  // Decode MAP response from stdout bytes.
  const stdoutBytes = new TextEncoder().encode(result.stdout);
  const stream = new ReadableStream<Uint8Array>({
    start(ctrl) { ctrl.enqueue(stdoutBytes); ctrl.close(); },
  });
  const iter = readFrames(stream);
  const { value: frameBytes, done } = await iter.next();
  if (done || frameBytes === undefined) {
    throw new Error("mind-nerve: empty response");
  }
  return decodeSubprocessOutput(frameBytes);
}
