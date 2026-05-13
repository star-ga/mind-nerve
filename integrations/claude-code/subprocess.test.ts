// Tests for subprocess.ts wire encoding/decoding.
// Covers the mic@2 text frame (stdin) and mic-b binary frame (stdout)
// as specified in cli/main.mind §WIRE PROTOCOL.

import { describe, it, expect } from "vitest";
import crypto from "node:crypto";
import {
  encodeMic2Frame,
  decodeMicBFrame,
  buildRouteIdMap,
  parseStderrFrame,
  type SubprocessInput,
} from "./subprocess.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeInput(overrides: Partial<SubprocessInput> = {}): SubprocessInput {
  return {
    current_prompt: "hello world",
    registry_summary: [
      { id: "skill-a", excerpt: "Skill A" },
      { id: "skill-b", excerpt: "Skill B" },
    ],
    catalog_hash: "abc123",
    k: 2,
    threshold: 0.5,
    ...overrides,
  };
}

/** Computes SHA-256 of a string as a 32-byte Buffer. */
function sha256Buf(s: string): Buffer {
  return crypto.createHash("sha256").update(s, "utf8").digest();
}

/** Writes a 32-bit signed LE integer at `offset` in `buf`. */
function writeI32LE(buf: Buffer, offset: number, value: number): void {
  buf.writeInt32LE(value, offset);
}

/**
 * Builds a minimal valid mic-b frame for the given skill IDs and Q16.16 scores.
 * Attestation envelope is zero-filled (valid shape, tests don't verify chain).
 */
function buildMicBFrame(
  ids: string[],
  q1616Scores: number[],
): Buffer {
  const k = ids.length;
  const total = 6 + 32 * k + 4 * k + 212;
  const buf = Buffer.alloc(total, 0);

  // magic "MNB1"
  buf[0] = 0x4d;
  buf[1] = 0x4e;
  buf[2] = 0x42;
  buf[3] = 0x31;

  // k (u16 LE)
  buf.writeUInt16LE(k, 4);

  // route IDs (SHA-256 of external id)
  for (let i = 0; i < k; i++) {
    const idHash = sha256Buf(ids[i] ?? "");
    idHash.copy(buf, 6 + i * 32);
  }

  // scores (i32 LE Q16.16)
  const scoresOffset = 6 + 32 * k;
  for (let i = 0; i < k; i++) {
    writeI32LE(buf, scoresOffset + i * 4, q1616Scores[i] ?? 0);
  }

  // attestation envelope: zero-filled (already done by Buffer.alloc)

  return buf;
}

// ---------------------------------------------------------------------------
// mic@2 stdin encoding
// ---------------------------------------------------------------------------

describe("encodeMic2Frame", () => {
  it("starts with the header line", () => {
    const frame = encodeMic2Frame(makeInput());
    expect(frame.split("\n")[0]).toBe("mic@2/mind-nerve/preselect");
  });

  it("ends with terminator line", () => {
    const frame = encodeMic2Frame(makeInput());
    const lines = frame.split("\n").filter((l) => l !== "");
    expect(lines[lines.length - 1]).toBe(".");
  });

  it("includes required keys: model, catalog, k, tokens", () => {
    const frame = encodeMic2Frame(makeInput({ k: 3 }));
    expect(frame).toMatch(/^model: /m);
    expect(frame).toMatch(/^catalog: /m);
    expect(frame).toMatch(/^k: 3$/m);
    expect(frame).toMatch(/^tokens: /m);
  });

  it("embeds catalog_hash in catalog path", () => {
    const frame = encodeMic2Frame(makeInput({ catalog_hash: "deadbeef" }));
    expect(frame).toMatch(/^catalog: .*deadbeef.*$/m);
  });

  it("derives tokens from prompt bytes (byte-level fallback)", () => {
    // "AB" = bytes 65, 66
    const frame = encodeMic2Frame(makeInput({ current_prompt: "AB" }));
    const tokensLine = frame.split("\n").find((l) => l.startsWith("tokens: "));
    expect(tokensLine).toBe("tokens: 65,66");
  });

  it("handles empty prompt with empty tokens", () => {
    const frame = encodeMic2Frame(makeInput({ current_prompt: "" }));
    const tokensLine = frame.split("\n").find((l) => l.startsWith("tokens: "));
    // Empty prompt → zero tokens. The CLI will reject this with RequestTooLong
    // (or ParseError), but the frame is structurally valid.
    expect(tokensLine).toBe("tokens: ");
  });

  it("handles multi-byte UTF-8 correctly (each byte is a token)", () => {
    // "é" is UTF-8 bytes 0xC3 0xA9 = 195, 169
    const frame = encodeMic2Frame(makeInput({ current_prompt: "é" }));
    const tokensLine = frame.split("\n").find((l) => l.startsWith("tokens: "));
    expect(tokensLine).toBe("tokens: 195,169");
  });

  it("uses correct key: value format (colon space, no extra whitespace)", () => {
    const frame = encodeMic2Frame(makeInput({ k: 5 }));
    const kLine = frame.split("\n").find((l) => l.startsWith("k: "));
    expect(kLine).toBe("k: 5");
  });
});

// ---------------------------------------------------------------------------
// RouteId ↔ externalId mapping
// ---------------------------------------------------------------------------

describe("buildRouteIdMap", () => {
  it("maps sha256(id) hex → id for each registry entry", () => {
    const summary = [{ id: "skill-a" }, { id: "skill-b" }];
    const map = buildRouteIdMap(summary);

    const hexA = crypto.createHash("sha256").update("skill-a", "utf8").digest("hex");
    const hexB = crypto.createHash("sha256").update("skill-b", "utf8").digest("hex");

    expect(map.get(hexA)).toBe("skill-a");
    expect(map.get(hexB)).toBe("skill-b");
  });

  it("returns empty map for empty summary", () => {
    const map = buildRouteIdMap([]);
    expect(map.size).toBe(0);
  });

  it("handles duplicate IDs by last-write-wins (should not occur in practice)", () => {
    const map = buildRouteIdMap([{ id: "x" }, { id: "x" }]);
    const hex = crypto.createHash("sha256").update("x", "utf8").digest("hex");
    expect(map.get(hex)).toBe("x");
    expect(map.size).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// mic-b stdout decoding
// ---------------------------------------------------------------------------

describe("decodeMicBFrame", () => {
  it("decodes top_k with correct external IDs", () => {
    const ids = ["skill-a", "skill-b"];
    // 0.95 in Q16.16: Math.round(0.95 * 65536) = 62259
    const frame = buildMicBFrame(ids, [62259, 58982]);
    const routeIdMap = buildRouteIdMap(
      ids.map((id) => ({ id })),
    );

    const result = decodeMicBFrame(frame, routeIdMap);
    expect(result.outcome).toBe("top_k");
    if (result.outcome === "top_k") {
      expect(result.selected).toEqual(["skill-a", "skill-b"]);
    }
  });

  it("converts Q16.16 scores to JS floats", () => {
    const ids = ["skill-a"];
    // 0.5 * 65536 = 32768 exactly
    const frame = buildMicBFrame(ids, [32768]);
    const routeIdMap = buildRouteIdMap(ids.map((id) => ({ id })));

    const result = decodeMicBFrame(frame, routeIdMap);
    if (result.outcome === "top_k") {
      expect(result.scores).toBeDefined();
      expect(result.scores![0]).toBeCloseTo(0.5, 5);
    } else {
      throw new Error("expected top_k");
    }
  });

  it("handles negative Q16.16 scores (i32 signed)", () => {
    const ids = ["skill-a"];
    // -1 in Q16.16 = -65536 as i32
    const frame = buildMicBFrame(ids, [-65536]);
    const routeIdMap = buildRouteIdMap(ids.map((id) => ({ id })));

    const result = decodeMicBFrame(frame, routeIdMap);
    if (result.outcome === "top_k") {
      expect(result.scores![0]).toBeCloseTo(-1.0, 5);
    } else {
      throw new Error("expected top_k");
    }
  });

  it("skips route IDs not present in routeIdMap (unknown skill)", () => {
    // Build frame with "skill-unknown" but don't include it in the map.
    const frame = buildMicBFrame(["skill-unknown", "skill-a"], [65536, 65536]);
    const routeIdMap = buildRouteIdMap([{ id: "skill-a" }]);

    const result = decodeMicBFrame(frame, routeIdMap);
    if (result.outcome === "top_k") {
      // skill-unknown is skipped; only skill-a survives
      expect(result.selected).toEqual(["skill-a"]);
    } else {
      throw new Error("expected top_k");
    }
  });

  it("handles k=0 (empty top_k result)", () => {
    const frame = buildMicBFrame([], []);
    const routeIdMap = buildRouteIdMap([]);

    const result = decodeMicBFrame(frame, routeIdMap);
    expect(result.outcome).toBe("top_k");
    if (result.outcome === "top_k") {
      expect(result.selected).toEqual([]);
      expect(result.scores).toEqual([]);
    }
  });

  it("throws on bad magic bytes", () => {
    const frame = buildMicBFrame(["skill-a"], [65536]);
    // Corrupt the magic.
    frame[0] = 0x00;
    const routeIdMap = buildRouteIdMap([{ id: "skill-a" }]);
    expect(() => decodeMicBFrame(frame, routeIdMap)).toThrow(/magic/);
  });

  it("throws when frame is too short (< 6 bytes)", () => {
    const tiny = Buffer.from([0x4d, 0x4e]);
    expect(() =>
      decodeMicBFrame(tiny, new Map()),
    ).toThrow(/too short/);
  });

  it("throws when frame is truncated after k", () => {
    const buf = Buffer.alloc(7, 0);
    buf[0] = 0x4d; buf[1] = 0x4e; buf[2] = 0x42; buf[3] = 0x31;
    buf.writeUInt16LE(2, 4); // k=2 but only 1 extra byte
    expect(() =>
      decodeMicBFrame(buf, new Map()),
    ).toThrow(/truncated/);
  });

  it("total frame size matches 218+36k formula for k=3", () => {
    const ids = ["a", "b", "c"];
    const frame = buildMicBFrame(ids, [65536, 65536, 65536]);
    const expectedSize = 218 + 36 * 3; // 326
    expect(frame.length).toBe(expectedSize);
  });
});

// ---------------------------------------------------------------------------
// mic@2 stderr parser
// ---------------------------------------------------------------------------

describe("parseStderrFrame", () => {
  it("extracts code from a well-formed error frame", () => {
    const frame = [
      "mic@2/mind-nerve/error",
      "code: RequestTooLong",
      "detail: 1532 tokens exceeded MAX_REQUEST_TOKENS=1024",
      ".",
    ].join("\n");
    expect(parseStderrFrame(frame)).toBe("RequestTooLong");
  });

  it("extracts code when mixed with other output", () => {
    const mixed = "some warning\ncode: InvalidK\nmore text";
    expect(parseStderrFrame(mixed)).toBe("InvalidK");
  });

  it("returns 'unknown' when no code line is present", () => {
    expect(parseStderrFrame("something went wrong")).toBe("unknown");
  });

  it("returns 'unknown' for empty stderr", () => {
    expect(parseStderrFrame("")).toBe("unknown");
  });

  it("handles all documented error codes", () => {
    const codes = [
      "RequestTooLong",
      "InvalidK",
      "EmptyCatalog",
      "ModelMismatch",
      "ParseError",
      "IoError",
    ];
    for (const code of codes) {
      const frame = `mic@2/mind-nerve/error\ncode: ${code}\ndetail: test\n.`;
      expect(parseStderrFrame(frame)).toBe(code);
    }
  });
});
