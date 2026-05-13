// mind-nerve Claude Code UserPromptSubmit hook. Projector pattern (D1).

import os from "node:os";
import path from "node:path";
import fs from "node:fs";
import { encodeMap } from "@mind/mic-map";

const LOG_PATH = path.join(os.homedir(), ".mind-nerve", "hook.log");

// MAP-framed log records — one MAP frame per line, grep-able text.
// Format: =ok level="<level>" hook=preselect <key>=<value>...
// Each line is a complete, standalone MAP frame parseable by decodeMap().

type LogLevel = "debug" | "info" | "warn" | "error";
type Primitive = string | bigint | boolean | number;

function stringify(v: unknown): string {
  if (v === null || v === undefined) return "";
  if (typeof v === "string") return v;
  if (typeof v === "bigint" || typeof v === "boolean" || typeof v === "number") return String(v);
  try { return JSON.stringify(v); } catch { return String(v); }
}

function toFields(obj: Record<string, unknown>): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(obj)) {
    out[k] = stringify(v);
  }
  return out;
}

function writeLogLine(
  level: LogLevel,
  msgOrObj: string | Record<string, unknown>,
  msgIfObj?: string,
): void {
  const msg = typeof msgOrObj === "string" ? msgOrObj : (msgIfObj ?? "");
  const extra = typeof msgOrObj === "object" ? toFields(msgOrObj) : {};
  const fields: Record<string, string | bigint> = {
    level,
    hook: "preselect",
    t: BigInt(Date.now()),
    msg,
    ...extra,
  };
  const line = encodeMap({ kind: "ok", fields }) + "\n";
  try {
    fs.appendFileSync(LOG_PATH, line, "utf8");
  } catch {
    // Logging must never crash the hook — silently swallow write errors.
  }
}

/** MAP-framed logger. Replaces pino NDJSON for grep-able MAP text log output. */
export const logger = {
  debug: (msgOrObj: string | Record<string, unknown>, msg?: string) =>
    writeLogLine("debug", msgOrObj, msg),
  info: (msgOrObj: string | Record<string, unknown>, msg?: string) =>
    writeLogLine("info", msgOrObj, msg),
  warn: (msgOrObj: string | Record<string, unknown>, msg?: string) =>
    writeLogLine("warn", msgOrObj, msg),
  error: (msgOrObj: string | Record<string, unknown>, msg?: string) =>
    writeLogLine("error", msgOrObj, msg),
};
