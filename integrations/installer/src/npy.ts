// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.
//
// Minimal NumPy .npy v1.0 reader/writer for the route-table embeddings.
//
// Why hand-rolled instead of a dependency: the ONLY thing the installer needs
// is to drop rows from a 2-D C-order float array in lockstep with the JSONL
// route table. That is a byte-slice operation. A full ndarray dependency would
// be a large surface for a 40-line need, and the format is frozen and simple:
//
//   \x93NUMPY  <major:u8> <minor:u8> <header_len:u16le> <header ascii> <raw data>
//
// Verified against the live runtime file (route_table.npy):
//   magic b'\x93NUMPY', version 1.0, header_len 118, data offset 128,
//   header "{'descr': '<f4', 'fortran_order': False, 'shape': (1437, 384), }"
//
// The header is space-padded so that (10 + header_len) is a multiple of 64,
// which is what NumPy itself does for alignment.

import { InstallerError } from "./errors.js";

const MAGIC = Buffer.from([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]); // \x93NUMPY
const ALIGNMENT = 64;

/** Element widths for the dtypes the route table can plausibly use. */
const DTYPE_WIDTH: Readonly<Record<string, number>> = {
  "<f4": 4,
  "<f8": 8,
  "<i4": 4,
  "<i8": 8,
};

export interface NpyArray {
  /** NumPy dtype descriptor, e.g. "<f4". */
  readonly descr: string;
  /** Row count (shape[0]). */
  readonly rows: number;
  /** Column count (shape[1]). 1-D arrays report 1. */
  readonly cols: number;
  /** True when the source declared a 1-D shape. */
  readonly oneDimensional: boolean;
  /** Raw row-major element data, no header. */
  readonly data: Buffer;
}

/**
 * Parses a .npy buffer.
 *
 * Refuses Fortran order outright: row extraction from a column-major buffer is
 * a different operation, and silently getting it wrong would corrupt every
 * embedding. Better to fail loudly than to write a plausible-looking wrong file.
 */
export function parseNpy(buf: Buffer, context: string): NpyArray {
  if (buf.length < 10 || !buf.subarray(0, 6).equals(MAGIC)) {
    throw new InstallerError(
      "INVALID_CONFIG_FORMAT",
      context,
      "Not a .npy file: magic header missing",
    );
  }

  const major = buf[6];
  if (major !== 1) {
    throw new InstallerError(
      "INVALID_CONFIG_FORMAT",
      context,
      `Unsupported .npy major version ${String(major)} (only 1.x is handled)`,
    );
  }

  const headerLen = buf.readUInt16LE(8);
  const dataOffset = 10 + headerLen;
  if (dataOffset > buf.length) {
    throw new InstallerError(
      "INVALID_CONFIG_FORMAT",
      context,
      ".npy header length runs past end of file",
    );
  }

  const header = buf.subarray(10, dataOffset).toString("latin1");

  const descrMatch = /'descr'\s*:\s*'([^']+)'/.exec(header);
  const fortranMatch = /'fortran_order'\s*:\s*(True|False)/.exec(header);
  const shapeMatch = /'shape'\s*:\s*\(([^)]*)\)/.exec(header);
  if (descrMatch === null || fortranMatch === null || shapeMatch === null) {
    throw new InstallerError(
      "INVALID_CONFIG_FORMAT",
      context,
      `Unparseable .npy header: ${header.trim()}`,
    );
  }

  const descr = descrMatch[1] as string;
  const width = DTYPE_WIDTH[descr];
  if (width === undefined) {
    throw new InstallerError(
      "INVALID_CONFIG_FORMAT",
      context,
      `Unsupported .npy dtype '${descr}'`,
    );
  }

  if (fortranMatch[1] === "True") {
    throw new InstallerError(
      "INVALID_CONFIG_FORMAT",
      context,
      "Fortran-ordered .npy is not supported — row pruning would corrupt it",
    );
  }

  const dims = (shapeMatch[1] as string)
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map((s) => Number.parseInt(s, 10));
  if (dims.length === 0 || dims.some((d) => !Number.isFinite(d) || d < 0)) {
    throw new InstallerError(
      "INVALID_CONFIG_FORMAT",
      context,
      `Unsupported .npy shape: (${shapeMatch[1] as string})`,
    );
  }
  if (dims.length > 2) {
    throw new InstallerError(
      "INVALID_CONFIG_FORMAT",
      context,
      `Only 1-D and 2-D .npy arrays are supported, got ${dims.length}-D`,
    );
  }

  const rows = dims[0] as number;
  const cols = dims.length === 2 ? (dims[1] as number) : 1;
  const expected = rows * cols * width;
  const actual = buf.length - dataOffset;
  if (actual !== expected) {
    throw new InstallerError(
      "INVALID_CONFIG_FORMAT",
      context,
      `.npy payload size mismatch: header declares ${rows}x${cols} ` +
        `(${expected} bytes) but ${actual} bytes follow the header`,
    );
  }

  return {
    descr,
    rows,
    cols,
    oneDimensional: dims.length === 1,
    data: buf.subarray(dataOffset),
  };
}

/** Serialises an NpyArray back to .npy v1.0 bytes, NumPy-compatible. */
export function serializeNpy(arr: NpyArray): Buffer {
  const shape = arr.oneDimensional
    ? `(${arr.rows},)`
    : `(${arr.rows}, ${arr.cols})`;
  const dict = `{'descr': '${arr.descr}', 'fortran_order': False, 'shape': ${shape}, }`;

  // Pad with spaces so the total prelude is 64-byte aligned and the header
  // ends with '\n' — byte-for-byte what numpy.save produces.
  const unpadded = 10 + dict.length + 1;
  const padded = Math.ceil(unpadded / ALIGNMENT) * ALIGNMENT;
  const header = dict + " ".repeat(padded - unpadded) + "\n";

  const prelude = Buffer.alloc(10);
  MAGIC.copy(prelude, 0);
  prelude[6] = 1;
  prelude[7] = 0;
  prelude.writeUInt16LE(header.length, 8);

  return Buffer.concat([prelude, Buffer.from(header, "latin1"), arr.data]);
}

/**
 * Returns a copy of `arr` keeping only the rows whose index satisfies `keep`.
 *
 * This is the operation that must stay in lockstep with the JSONL prune: the
 * embedding matrix is ROW-ALIGNED to the route table, so dropping a JSONL line
 * without dropping its matching row shifts every subsequent embedding onto the
 * wrong route and silently corrupts all later ranking.
 */
export function selectRows(
  arr: NpyArray,
  keep: (rowIndex: number) => boolean,
): NpyArray {
  const width = DTYPE_WIDTH[arr.descr];
  if (width === undefined) {
    throw new InstallerError(
      "INVALID_CONFIG_FORMAT",
      "npy",
      `Unsupported dtype '${arr.descr}'`,
    );
  }
  const rowBytes = arr.cols * width;
  const chunks: Buffer[] = [];
  let kept = 0;
  for (let i = 0; i < arr.rows; i++) {
    if (!keep(i)) continue;
    chunks.push(arr.data.subarray(i * rowBytes, (i + 1) * rowBytes));
    kept++;
  }
  return {
    descr: arr.descr,
    rows: kept,
    cols: arr.cols,
    oneDimensional: arr.oneDimensional,
    data: Buffer.concat(chunks, kept * rowBytes),
  };
}
