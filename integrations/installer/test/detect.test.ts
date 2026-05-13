// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.

import { describe, it, expect } from "vitest";
import { detectClient } from "../src/detect.js";
import type { AgentSpec } from "../src/registry.js";

function makeSpec(overrides: Partial<AgentSpec> = {}): AgentSpec {
  return {
    name: "test-client",
    description: "Test client",
    configFmt: "text-block",
    configPath: "/tmp/test.json",
    mcpPath: null,
    mcpFmt: null,
    detectPaths: [],
    detectBinaries: [],
    alwaysOffer: false,
    projectionDir: null,
    instructionFilePath: null,
    ...overrides,
  };
}

describe("detectClient", () => {
  it("returns detected=false when no binary and no path match", async () => {
    const spec = makeSpec({
      detectBinaries: ["this-binary-does-not-exist-xyz-123"],
      detectPaths: ["/this/path/definitely/does/not/exist/xyz"],
    });
    const result = await detectClient(spec);
    expect(result.detected).toBe(false);
    expect(result.foundBinary).toBeNull();
    expect(result.foundPath).toBeNull();
  });

  it("returns detected=true when a detectPath exists on disk", async () => {
    const spec = makeSpec({
      detectPaths: ["/tmp"], // /tmp always exists
    });
    const result = await detectClient(spec);
    expect(result.detected).toBe(true);
    expect(result.foundPath).toBe("/tmp");
  });

  it("returns detected=true when alwaysOffer is true, even with no binary or path", async () => {
    const spec = makeSpec({
      alwaysOffer: true,
      detectBinaries: [],
      detectPaths: [],
    });
    const result = await detectClient(spec);
    expect(result.detected).toBe(true);
    expect(result.alwaysOffer).toBe(true);
    expect(result.foundBinary).toBeNull();
    expect(result.foundPath).toBeNull();
  });

  it("returns alwaysOffer=false when spec.alwaysOffer is false", async () => {
    const spec = makeSpec({ alwaysOffer: false });
    const result = await detectClient(spec);
    expect(result.alwaysOffer).toBe(false);
  });

  it("returns the client name correctly", async () => {
    const spec = makeSpec({ name: "my-cli" });
    const result = await detectClient(spec);
    expect(result.name).toBe("my-cli");
  });

  it("stops after finding the first matching detectPath", async () => {
    const spec = makeSpec({
      detectPaths: ["/tmp", "/usr"], // both exist; should stop at /tmp
    });
    const result = await detectClient(spec);
    expect(result.foundPath).toBe("/tmp");
  });

  it("tries system binaries that are likely to exist (sh)", async () => {
    const spec = makeSpec({
      detectBinaries: ["sh"], // sh is always on PATH
    });
    const result = await detectClient(spec);
    expect(result.detected).toBe(true);
    expect(result.foundBinary).not.toBeNull();
    expect(result.foundBinary).toContain("sh");
  });

  it("foundBinary is null when binary does not exist", async () => {
    const spec = makeSpec({
      detectBinaries: ["this-binary-xyz-does-not-exist-998877"],
    });
    const result = await detectClient(spec);
    expect(result.foundBinary).toBeNull();
  });
});
