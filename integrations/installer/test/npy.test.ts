// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.

import { describe, it, expect } from "vitest";
import { parseNpy, serializeNpy, selectRows, type NpyArray } from "../src/npy.js";
import { InstallerError } from "../src/errors.js";

/** Builds a float32 C-order .npy the way numpy.save does. */
function makeNpy(rows: number, cols: number, fill: (r: number, c: number) => number): Buffer {
  const data = Buffer.alloc(rows * cols * 4);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      data.writeFloatLE(fill(r, c), (r * cols + c) * 4);
    }
  }
  const arr: NpyArray = {
    descr: "<f4",
    rows,
    cols,
    oneDimensional: false,
    data,
  };
  return serializeNpy(arr);
}

describe("parseNpy", () => {
  it("reads the header shape and payload", () => {
    const buf = makeNpy(4, 3, (r, c) => r * 10 + c);
    const arr = parseNpy(buf, "test");
    expect(arr.descr).toBe("<f4");
    expect(arr.rows).toBe(4);
    expect(arr.cols).toBe(3);
    expect(arr.data.length).toBe(4 * 3 * 4);
  });

  it("produces a 64-byte-aligned prelude like numpy", () => {
    const buf = makeNpy(2, 384, () => 1);
    // Matches the live route_table.npy: data begins at offset 128.
    const headerLen = buf.readUInt16LE(8);
    expect((10 + headerLen) % 64).toBe(0);
    expect(buf.subarray(10, 10 + headerLen).toString("latin1")).toMatch(/\n$/);
  });

  it("rejects a non-npy buffer", () => {
    expect(() => parseNpy(Buffer.from("hello world"), "test")).toThrow(
      InstallerError,
    );
  });

  it("rejects Fortran order rather than silently mangling rows", () => {
    const buf = makeNpy(2, 2, () => 1);
    const headerLen = buf.readUInt16LE(8);
    const header = buf
      .subarray(10, 10 + headerLen)
      .toString("latin1")
      .replace("'fortran_order': False", "'fortran_order': True ");
    const patched = Buffer.concat([
      buf.subarray(0, 10),
      Buffer.from(header, "latin1"),
      buf.subarray(10 + headerLen),
    ]);
    expect(() => parseNpy(patched, "test")).toThrow(/Fortran/);
  });

  it("rejects a payload whose size contradicts the declared shape", () => {
    const buf = makeNpy(4, 3, () => 1);
    const truncated = buf.subarray(0, buf.length - 4);
    expect(() => parseNpy(truncated, "test")).toThrow(/payload size mismatch/);
  });

  it("rejects an unsupported dtype", () => {
    const buf = makeNpy(2, 2, () => 1);
    const headerLen = buf.readUInt16LE(8);
    const header = buf
      .subarray(10, 10 + headerLen)
      .toString("latin1")
      .replace("'<f4'", "'|S8'");
    const patched = Buffer.concat([
      buf.subarray(0, 10),
      Buffer.from(header, "latin1"),
      buf.subarray(10 + headerLen),
    ]);
    expect(() => parseNpy(patched, "test")).toThrow(/Unsupported .npy dtype/);
  });
});

describe("serializeNpy", () => {
  it("round-trips byte-for-byte", () => {
    const original = makeNpy(7, 5, (r, c) => r + c / 8);
    const reparsed = parseNpy(original, "test");
    expect(serializeNpy(reparsed).equals(original)).toBe(true);
  });

  it("round-trips at the live embedding width (384 columns)", () => {
    const original = makeNpy(11, 384, (r, c) => (r * 384 + c) / 1024);
    const arr = parseNpy(original, "test");
    expect(arr.cols).toBe(384);
    expect(serializeNpy(arr).equals(original)).toBe(true);
  });
});

describe("selectRows", () => {
  it("keeps exactly the requested rows, in order", () => {
    const arr = parseNpy(makeNpy(5, 2, (r, c) => r * 100 + c), "test");
    const kept = selectRows(arr, (i) => i === 1 || i === 3);

    expect(kept.rows).toBe(2);
    expect(kept.cols).toBe(2);
    expect(kept.data.readFloatLE(0)).toBe(100); // row 1, col 0
    expect(kept.data.readFloatLE(4)).toBe(101); // row 1, col 1
    expect(kept.data.readFloatLE(8)).toBe(300); // row 3, col 0
  });

  it("preserves the exact bytes of every kept row", () => {
    const arr = parseNpy(makeNpy(6, 4, (r, c) => r * 1000 + c), "test");
    const kept = selectRows(arr, (i) => i % 2 === 0);
    const rowBytes = 4 * 4;
    for (let k = 0; k < 3; k++) {
      const src = arr.data.subarray(k * 2 * rowBytes, (k * 2 + 1) * rowBytes);
      const dst = kept.data.subarray(k * rowBytes, (k + 1) * rowBytes);
      expect(dst.equals(src)).toBe(true);
    }
  });

  it("can drop everything", () => {
    const arr = parseNpy(makeNpy(3, 2, () => 1), "test");
    const kept = selectRows(arr, () => false);
    expect(kept.rows).toBe(0);
    expect(kept.data.length).toBe(0);
    expect(parseNpy(serializeNpy(kept), "test").rows).toBe(0);
  });

  it("is a no-op copy when everything is kept", () => {
    const original = makeNpy(4, 3, (r, c) => r - c);
    const arr = parseNpy(original, "test");
    const kept = selectRows(arr, () => true);
    expect(serializeNpy(kept).equals(original)).toBe(true);
  });
});
