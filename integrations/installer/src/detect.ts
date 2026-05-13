// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.

import fs from "node:fs/promises";
import which from "which";
import { type AgentSpec } from "./registry.js";

export interface DetectionResult {
  readonly name: string;
  readonly detected: boolean;
  readonly alwaysOffer: boolean;
  /** The binary found on PATH, or null if none detected. */
  readonly foundBinary: string | null;
  /** The first config path that exists, or null. */
  readonly foundPath: string | null;
}

/**
 * Probes whether a single client is installed on this machine.
 *
 * Detection succeeds if:
 *   - any detectBinary resolves via PATH, OR
 *   - any detectPath exists on disk, OR
 *   - alwaysOffer is true (copilot).
 */
export async function detectClient(
  spec: AgentSpec,
): Promise<DetectionResult> {
  let foundBinary: string | null = null;
  for (const bin of spec.detectBinaries) {
    try {
      foundBinary = await which(bin);
      break;
    } catch {
      // not found — try next
    }
  }

  let foundPath: string | null = null;
  for (const p of spec.detectPaths) {
    try {
      await fs.access(p);
      foundPath = p;
      break;
    } catch {
      // not found — try next
    }
  }

  const detected = spec.alwaysOffer || foundBinary !== null || foundPath !== null;

  return {
    name: spec.name,
    detected,
    alwaysOffer: spec.alwaysOffer,
    foundBinary,
    foundPath,
  };
}

/**
 * Probes all provided specs in parallel.
 */
export async function detectAll(
  specs: readonly AgentSpec[],
): Promise<DetectionResult[]> {
  return Promise.all(specs.map(detectClient));
}
