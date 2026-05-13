// mind-nerve installer — Copyright 2026 STARGA Inc. Apache-2.0.

export type InstallerErrorCode =
  | "CLIENT_NOT_DETECTED"
  | "CONFIG_NOT_FOUND"
  | "BACKUP_FAILED"
  | "RESTORE_FAILED"
  | "MERGE_FAILED"
  | "PROJECTION_FAILED"
  | "IDEMPOTENT_NOOP"
  | "UNKNOWN_CLIENT"
  | "INVALID_CONFIG_FORMAT";

export class InstallerError extends Error {
  readonly code: InstallerErrorCode;
  readonly clientName: string;

  constructor(code: InstallerErrorCode, clientName: string, message: string) {
    super(message);
    this.name = "InstallerError";
    this.code = code;
    this.clientName = clientName;
  }
}
