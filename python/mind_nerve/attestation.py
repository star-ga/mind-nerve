"""Public Python surface for the MindLLM cross-binding handshake.

Mirrors the spec in ``integrations/mindllm_attestation.mind``. This module
exists so external integrators (mind-mem, MindLLM, third-party verifiers)
can produce and verify ``BindingRecord``s without re-implementing the
Ed25519 + SHA-256 plumbing.

Public surface:

  - ``MAGIC``, ``VERSION``, ``RECORD_SIZE`` — wire-format constants.
  - ``binding_message(mind_nerve_hash, mindllm_hash, nonce)``
  - ``sign_binding(private_key, msg)``
  - ``verify_binding(public_key, mind_nerve_hash, mindllm_hash, nonce, signature)``
  - ``application_verify_binding(...)``  — adds the ZeroField guard.
  - ``serialize_binding_record(...)``    — pack to 200 bytes.
  - ``deserialize_binding_record(buf)``  — parse + sanity-check headers.
  - ``BindingRecord``                   — frozen dataclass over the 200-byte
                                          wire format.

All public functions are pure (no I/O, no global state).
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

MAGIC = b"MNBA"
VERSION = 1
RECORD_SIZE = 200
HASH_SIZE = 32
NONCE_SIZE = 32
SIGNATURE_SIZE = 64
PUBKEY_SIZE = 32

VerifyResult = Literal["ok", "ZeroField", "SignatureInvalid", "MagicMismatch", "VersionMismatch"]


@dataclass(frozen=True)
class BindingRecord:
    """Parsed view of a 200-byte BindingRecord (see spec wire format)."""

    mind_nerve_hash: bytes
    mindllm_hash: bytes
    nonce: bytes
    signature: bytes
    signer_pubkey: bytes
    version: int = VERSION

    def to_bytes(self) -> bytes:
        return serialize_binding_record(
            self.mind_nerve_hash,
            self.mindllm_hash,
            self.nonce,
            self.signature,
            self.signer_pubkey,
        )


def binding_message(
    mind_nerve_hash: bytes,
    mindllm_hash: bytes,
    nonce: bytes,
) -> bytes:
    """SHA-256(mind_nerve_hash ++ mindllm_hash ++ nonce) — 32-byte digest.

    Raises ValueError on length mismatch.
    """
    if len(mind_nerve_hash) != HASH_SIZE:
        raise ValueError(f"mind_nerve_hash must be {HASH_SIZE} bytes, got {len(mind_nerve_hash)}")
    if len(mindllm_hash) != HASH_SIZE:
        raise ValueError(f"mindllm_hash must be {HASH_SIZE} bytes, got {len(mindllm_hash)}")
    if len(nonce) != NONCE_SIZE:
        raise ValueError(f"nonce must be {NONCE_SIZE} bytes, got {len(nonce)}")
    return hashlib.sha256(mind_nerve_hash + mindllm_hash + nonce).digest()


def sign_binding(private_key: Ed25519PrivateKey, msg: bytes) -> bytes:
    """Ed25519 sign per RFC 8032 — deterministic. Returns 64-byte signature."""
    return private_key.sign(msg)


def verify_binding(
    public_key: Ed25519PublicKey,
    mind_nerve_hash: bytes,
    mindllm_hash: bytes,
    nonce: bytes,
    signature: bytes,
) -> bool:
    """Cryptographic verification only — no semantic guards.

    Use :func:`application_verify_binding` to additionally enforce the
    ZeroField invariant from the spec.
    """
    msg = binding_message(mind_nerve_hash, mindllm_hash, nonce)
    try:
        public_key.verify(signature, msg)
        return True
    except InvalidSignature:
        return False


def application_verify_binding(
    mind_nerve_hash: bytes,
    mindllm_hash: bytes,
    nonce: bytes,
    signature: bytes,
    public_key: Ed25519PublicKey,
) -> VerifyResult:
    """Full spec verifier: ZeroField guard + Ed25519 verification.

    Returns one of:
      - ``"ok"``               — record is valid and accepted.
      - ``"ZeroField"``        — one of the required 32/64-byte fields is
                                 entirely zero bytes (spec rejection).
      - ``"SignatureInvalid"`` — fields look real but Ed25519 verify failed.
    """
    if mind_nerve_hash == bytes(HASH_SIZE):
        return "ZeroField"
    if mindllm_hash == bytes(HASH_SIZE):
        return "ZeroField"
    if nonce == bytes(NONCE_SIZE):
        return "ZeroField"
    if signature == bytes(SIGNATURE_SIZE):
        return "ZeroField"
    ok = verify_binding(public_key, mind_nerve_hash, mindllm_hash, nonce, signature)
    return "ok" if ok else "SignatureInvalid"


def serialize_binding_record(
    mind_nerve_hash: bytes,
    mindllm_hash: bytes,
    nonce: bytes,
    signature: bytes,
    signer_pubkey: bytes,
) -> bytes:
    """Pack a BindingRecord to the 200-byte wire format.

    Wire layout (offset / size / field):
       0    4  magic "MNBA"
       4    2  version u16 LE = 1
       6    2  reserved = 0
       8   32  mind_nerve_hash
      40   32  mindllm_hash
      72   32  nonce
     104   64  signature
     168   32  signer_pubkey
    """
    if len(mind_nerve_hash) != HASH_SIZE:
        raise ValueError(f"mind_nerve_hash must be {HASH_SIZE} bytes")
    if len(mindllm_hash) != HASH_SIZE:
        raise ValueError(f"mindllm_hash must be {HASH_SIZE} bytes")
    if len(nonce) != NONCE_SIZE:
        raise ValueError(f"nonce must be {NONCE_SIZE} bytes")
    if len(signature) != SIGNATURE_SIZE:
        raise ValueError(f"signature must be {SIGNATURE_SIZE} bytes")
    if len(signer_pubkey) != PUBKEY_SIZE:
        raise ValueError(f"signer_pubkey must be {PUBKEY_SIZE} bytes")

    record = (
        MAGIC
        + struct.pack("<H", VERSION)
        + b"\x00\x00"
        + mind_nerve_hash
        + mindllm_hash
        + nonce
        + signature
        + signer_pubkey
    )
    assert len(record) == RECORD_SIZE, "internal: serialized record is wrong size"
    return record


def deserialize_binding_record(buf: bytes) -> BindingRecord:
    """Parse a 200-byte wire-format record into a ``BindingRecord``.

    Raises ``ValueError`` on length / magic / version mismatch. Does NOT
    verify the signature — call :func:`application_verify_binding` for that.
    """
    if len(buf) != RECORD_SIZE:
        raise ValueError(f"BindingRecord must be {RECORD_SIZE} bytes, got {len(buf)}")
    if buf[:4] != MAGIC:
        raise ValueError(f"magic mismatch: expected {MAGIC!r}, got {buf[:4]!r}")
    (version,) = struct.unpack("<H", buf[4:6])
    if version != VERSION:
        raise ValueError(f"unsupported BindingRecord version: {version}")
    return BindingRecord(
        mind_nerve_hash=buf[8:40],
        mindllm_hash=buf[40:72],
        nonce=buf[72:104],
        signature=buf[104:168],
        signer_pubkey=buf[168:200],
        version=version,
    )


def manifest_export_bytes(
    tensors: list[dict],
    aggregate_hex: str,
) -> bytes:
    """Deterministic Python build of the manifest JSON document.

    Mirrors ``manifest_export()`` from ``loader.mind``. Used for
    cross-binding the ``mind_nerve_hash`` against a reproducible source
    document.

    ``tensors`` items must carry: ``name`` (str), ``rows`` (int),
    ``cols`` (int), ``neuron_hash`` (lowercase hex str). Order is
    preserved as given (callers MUST sort alphabetically by name).
    """
    parts = []
    for t in tensors:
        parts.append(
            '{"name": "'
            + t["name"]
            + '", "shape": ['
            + str(t["rows"])
            + ", "
            + str(t["cols"])
            + '], "neuron_hash": "'
            + t["neuron_hash"]
            + '"}'
        )
    body = (
        '{"version": 1, "tensors": ['
        + ", ".join(parts)
        + '], "aggregate": "'
        + aggregate_hex
        + '"}'
    )
    return body.encode("utf-8")


def neuron_hash_hex(data: bytes) -> str:
    """SHA-256 of ``data`` as 64 lowercase hex chars."""
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "MAGIC",
    "VERSION",
    "RECORD_SIZE",
    "HASH_SIZE",
    "NONCE_SIZE",
    "SIGNATURE_SIZE",
    "PUBKEY_SIZE",
    "BindingRecord",
    "VerifyResult",
    "binding_message",
    "sign_binding",
    "verify_binding",
    "application_verify_binding",
    "serialize_binding_record",
    "deserialize_binding_record",
    "manifest_export_bytes",
    "neuron_hash_hex",
]
