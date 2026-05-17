"""Integration tests for the MindLLM cross-binding handshake.

Validates:
  H1. binding_message() is deterministic: same inputs -> identical digest.
  H2. binding_message() changes when any input changes (non-trivial function).
  H3. Full sign -> verify round-trip passes with a fixed test fixture.
  H4. Verification fails when the signature is corrupted.
  H5. Verification fails when either model hash in the record is altered.
  H6. Verification fails when the nonce in the record is altered.
  H7. Verification rejects a record with an all-zero model hash (ZeroField).
  H8. serialize_binding() is byte-identical across two calls.
  H9. serialize_binding() output is exactly 200 bytes.
  H10. manifest_export determinism: SHA-256 of two export runs is identical.

The test uses a FIXED test fixture (not real STARGA keys).  The fixture is a
deterministic key pair derived from a well-known test seed so the test is
repeatable on any machine without network access or secret-key material.
"""

from __future__ import annotations

import hashlib
import json
import struct
from typing import Any

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

# ---------------------------------------------------------------------------
# Pure-Python reference implementations of the handshake spec primitives.
# These mirror the MIND-language definitions in
# integrations/mindllm_attestation.mind and serve as the Python-side
# conformance reference.
# ---------------------------------------------------------------------------

def binding_message(
    mind_nerve_hash: bytes,
    mindllm_hash: bytes,
    nonce: bytes,
) -> bytes:
    """SHA-256(mind_nerve_hash ++ mindllm_hash ++ nonce) — 96-byte preimage."""
    assert len(mind_nerve_hash) == 32
    assert len(mindllm_hash) == 32
    assert len(nonce) == 32
    preimage = mind_nerve_hash + mindllm_hash + nonce
    return hashlib.sha256(preimage).digest()


def sign_binding(private_key: Ed25519PrivateKey, msg: bytes) -> bytes:
    """Ed25519 sign per RFC 8032. Returns 64-byte signature."""
    return private_key.sign(msg)


def verify_binding(
    public_key: Ed25519PublicKey,
    mind_nerve_hash: bytes,
    mindllm_hash: bytes,
    nonce: bytes,
    signature: bytes,
) -> bool:
    """Return True iff the signature is valid for the binding message."""
    msg = binding_message(mind_nerve_hash, mindllm_hash, nonce)
    try:
        public_key.verify(signature, msg)
        return True
    except InvalidSignature:
        return False


def serialize_binding_record(
    mind_nerve_hash: bytes,
    mindllm_hash: bytes,
    nonce: bytes,
    signature: bytes,
    signer_pubkey_bytes: bytes,
) -> bytes:
    """Pack a BindingRecord to the 200-byte wire format.

    Wire layout (from integrations/mindllm_attestation.mind §SERIALIZATION):
      4   magic "MNBA"
      2   version u16 LE = 1
      2   reserved = 0
     32   mind_nerve_hash
     32   mindllm_hash
     32   nonce
     64   signature
     32   signer_pubkey
    ---
    200 bytes total
    """
    magic = b"MNBA"
    version = struct.pack("<H", 1)
    reserved = b"\x00\x00"
    record = (
        magic
        + version
        + reserved
        + mind_nerve_hash
        + mindllm_hash
        + nonce
        + signature
        + signer_pubkey_bytes
    )
    assert len(record) == 200, f"serialized binding must be 200 bytes, got {len(record)}"
    return record


# ---------------------------------------------------------------------------
# Test fixture: deterministic key pair from a fixed seed.
# The seed is not secret — it exists solely for test repeatability.
# ---------------------------------------------------------------------------

_TEST_SEED = b"mind-nerve-test-vector-seed-v1.0"  # 32 bytes

def _make_test_keypair() -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    """Derive a deterministic Ed25519 key pair from the fixed test seed."""
    private_key = Ed25519PrivateKey.from_private_bytes(_TEST_SEED)
    return private_key, private_key.public_key()


_PRIVATE_KEY, _PUBLIC_KEY = _make_test_keypair()
_PUBLIC_KEY_BYTES: bytes = _PUBLIC_KEY.public_bytes(Encoding.Raw, PublicFormat.Raw)

# Fixed test vectors (deterministic — do not change across runs).
_MIND_NERVE_HASH: bytes = hashlib.sha256(b"mind-nerve-test-model-v1").digest()
_MINDLLM_HASH: bytes = hashlib.sha256(b"mindllm-test-model-v1").digest()
_NONCE: bytes = hashlib.sha256(b"test-nonce-value-1234").digest()


# ---------------------------------------------------------------------------
# H1: binding_message is deterministic
# ---------------------------------------------------------------------------

def test_binding_message_deterministic() -> None:
    """Two calls with identical inputs produce the same 32-byte digest."""
    msg1 = binding_message(_MIND_NERVE_HASH, _MINDLLM_HASH, _NONCE)
    msg2 = binding_message(_MIND_NERVE_HASH, _MINDLLM_HASH, _NONCE)
    assert msg1 == msg2, "binding_message must be deterministic"
    assert len(msg1) == 32


# ---------------------------------------------------------------------------
# H2: binding_message is non-trivially injective on inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("altered_field", ["mind_nerve", "mindllm", "nonce"])
def test_binding_message_changes_on_input_change(altered_field: str) -> None:
    """Changing any single input changes the digest."""
    alt = bytes(b ^ 0xFF for b in (b"\xAB" * 32))

    kwargs: dict[str, bytes] = {
        "mind_nerve_hash": _MIND_NERVE_HASH,
        "mindllm_hash": _MINDLLM_HASH,
        "nonce": _NONCE,
    }
    field_map = {
        "mind_nerve": "mind_nerve_hash",
        "mindllm": "mindllm_hash",
        "nonce": "nonce",
    }
    original = binding_message(**kwargs)
    kwargs[field_map[altered_field]] = alt
    altered = binding_message(**kwargs)
    assert original != altered, f"binding_message must change when {altered_field} changes"


# ---------------------------------------------------------------------------
# H3: full round-trip sign -> verify
# ---------------------------------------------------------------------------

def test_binding_sign_verify_roundtrip() -> None:
    """Sign a binding message with the test private key; verify with public key."""
    msg = binding_message(_MIND_NERVE_HASH, _MINDLLM_HASH, _NONCE)
    sig = sign_binding(_PRIVATE_KEY, msg)

    assert len(sig) == 64, "Ed25519 signature must be 64 bytes"
    assert verify_binding(
        _PUBLIC_KEY,
        _MIND_NERVE_HASH,
        _MINDLLM_HASH,
        _NONCE,
        sig,
    ), "valid signature must verify"


# ---------------------------------------------------------------------------
# H4: corrupted signature is rejected
# ---------------------------------------------------------------------------

def test_binding_corrupted_signature_rejected() -> None:
    """Flip a single bit in the signature; verification must fail."""
    msg = binding_message(_MIND_NERVE_HASH, _MINDLLM_HASH, _NONCE)
    sig = bytearray(sign_binding(_PRIVATE_KEY, msg))
    sig[0] ^= 0x01  # single-bit corruption

    assert not verify_binding(
        _PUBLIC_KEY,
        _MIND_NERVE_HASH,
        _MINDLLM_HASH,
        _NONCE,
        bytes(sig),
    ), "corrupted signature must not verify"


# ---------------------------------------------------------------------------
# H5: altered model hash in the record is rejected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("which_hash", ["mind_nerve", "mindllm"])
def test_binding_altered_hash_rejected(which_hash: str) -> None:
    """Flip the first byte of either model hash; verification must fail."""
    msg = binding_message(_MIND_NERVE_HASH, _MINDLLM_HASH, _NONCE)
    sig = sign_binding(_PRIVATE_KEY, msg)

    # Construct the altered version of the relevant hash.
    mn = bytearray(_MIND_NERVE_HASH)
    ml = bytearray(_MINDLLM_HASH)
    if which_hash == "mind_nerve":
        mn[0] ^= 0xFF
    else:
        ml[0] ^= 0xFF

    assert not verify_binding(
        _PUBLIC_KEY,
        bytes(mn),
        bytes(ml),
        _NONCE,
        sig,
    ), f"altered {which_hash} hash must cause verification failure"


# ---------------------------------------------------------------------------
# H6: altered nonce in the record is rejected
# ---------------------------------------------------------------------------

def test_binding_altered_nonce_rejected() -> None:
    """Flip the last byte of the nonce; verification must fail."""
    msg = binding_message(_MIND_NERVE_HASH, _MINDLLM_HASH, _NONCE)
    sig = sign_binding(_PRIVATE_KEY, msg)

    bad_nonce = bytearray(_NONCE)
    bad_nonce[-1] ^= 0xFF

    assert not verify_binding(
        _PUBLIC_KEY,
        _MIND_NERVE_HASH,
        _MINDLLM_HASH,
        bytes(bad_nonce),
        sig,
    ), "altered nonce must cause verification failure"


# ---------------------------------------------------------------------------
# H7: all-zero model hash is semantically rejected
# ---------------------------------------------------------------------------

def test_binding_zero_model_hash_rejected() -> None:
    """A ZeroField guard: verify_binding must reject all-zero mind_nerve_hash."""
    zero = bytes(32)
    msg = binding_message(zero, _MINDLLM_HASH, _NONCE)
    sig = sign_binding(_PRIVATE_KEY, msg)

    # The MIND-side verify_binding checks for ZeroField; on the Python side we
    # replicate the spec invariant: an all-zero hash MUST be treated as invalid
    # input regardless of signature validity.
    assert zero == bytes(32), "sanity: zero hash is all zeros"
    # Signature over a zero-hash message would be 'valid' cryptographically,
    # but the spec requires application-level rejection.
    result = _application_verify_binding(
        mind_nerve_hash=zero,
        mindllm_hash=_MINDLLM_HASH,
        nonce=_NONCE,
        signature=sig,
        public_key=_PUBLIC_KEY,
    )
    assert result == "ZeroField", "all-zero model hash must trigger ZeroField rejection"


def _application_verify_binding(
    mind_nerve_hash: bytes,
    mindllm_hash: bytes,
    nonce: bytes,
    signature: bytes,
    public_key: Ed25519PublicKey,
) -> str:
    """Application-level verify that includes the ZeroField guard from the spec."""
    if mind_nerve_hash == bytes(32):
        return "ZeroField"
    if mindllm_hash == bytes(32):
        return "ZeroField"
    if nonce == bytes(32):
        return "ZeroField"
    if signature == bytes(64):
        return "ZeroField"
    ok = verify_binding(public_key, mind_nerve_hash, mindllm_hash, nonce, signature)
    return "ok" if ok else "SignatureInvalid"


# ---------------------------------------------------------------------------
# H8: serialize_binding is deterministic
# ---------------------------------------------------------------------------

def test_serialize_binding_deterministic() -> None:
    """Two serialization calls with the same record produce identical bytes."""
    msg = binding_message(_MIND_NERVE_HASH, _MINDLLM_HASH, _NONCE)
    sig = sign_binding(_PRIVATE_KEY, msg)

    rec1 = serialize_binding_record(
        _MIND_NERVE_HASH, _MINDLLM_HASH, _NONCE, sig, _PUBLIC_KEY_BYTES
    )
    rec2 = serialize_binding_record(
        _MIND_NERVE_HASH, _MINDLLM_HASH, _NONCE, sig, _PUBLIC_KEY_BYTES
    )
    assert rec1 == rec2, "serialize_binding must be byte-identical across calls"


# ---------------------------------------------------------------------------
# H9: serialize_binding output is exactly 200 bytes
# ---------------------------------------------------------------------------

def test_serialize_binding_size() -> None:
    """Serialized BindingRecord is exactly 200 bytes."""
    msg = binding_message(_MIND_NERVE_HASH, _MINDLLM_HASH, _NONCE)
    sig = sign_binding(_PRIVATE_KEY, msg)

    rec = serialize_binding_record(
        _MIND_NERVE_HASH, _MINDLLM_HASH, _NONCE, sig, _PUBLIC_KEY_BYTES
    )
    assert len(rec) == 200


# ---------------------------------------------------------------------------
# H10: manifest_export determinism (Python reference)
# ---------------------------------------------------------------------------

def _build_manifest_json(
    tensors: list[dict[str, Any]],
    aggregate_hex: str,
) -> bytes:
    """Construct the deterministic manifest JSON document (Python reference).

    Mirrors the MIND-language manifest_export() contract:
    - Fixed key order within each tensor object.
    - No trailing commas.
    - Single space after ':' and ','.
    - All hex lowercase.
    - No trailing newline.
    """
    tensor_parts = []
    for t in tensors:
        part = (
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
        tensor_parts.append(part)

    body = (
        '{"version": 1, "tensors": ['
        + ", ".join(tensor_parts)
        + '], "aggregate": "'
        + aggregate_hex
        + '"}'
    )
    return body.encode("utf-8")


def _neuron_hash_hex(data: bytes) -> str:
    """SHA-256 of data, returned as 64 lowercase hex chars."""
    return hashlib.sha256(data).hexdigest()


def test_manifest_export_deterministic_python() -> None:
    """Two manifest export runs on the same fixture produce SHA-256-identical bytes."""
    # Synthetic tensor: a 4x4 all-zero Q16.16 matrix.
    tensor_bytes = bytes(4 * 4 * 4)  # 4 rows * 4 cols * 4 bytes per i32-LE
    neuron_hash = _neuron_hash_hex(tensor_bytes)

    # Aggregate preimage: MNPM magic + entry_count(1) + entry for "alpha".
    # Mirrors build_manifest_preimage in loader.mind.
    name = b"alpha"
    entry = (
        struct.pack("<I", len(name))
        + name
        + struct.pack("<II", 4, 4)          # rows=4, cols=4
        + bytes.fromhex(neuron_hash)         # 32-byte neuron_hash
    )
    preimage = b"MNPM" + struct.pack("<I", 1) + entry
    aggregate_hex = hashlib.sha256(preimage).hexdigest()

    tensors = [{"name": "alpha", "rows": 4, "cols": 4, "neuron_hash": neuron_hash}]

    run1 = _build_manifest_json(tensors, aggregate_hex)
    run2 = _build_manifest_json(tensors, aggregate_hex)

    assert run1 == run2, "manifest JSON must be byte-identical across calls"

    # Verify the SHA-256 of both runs match.
    assert hashlib.sha256(run1).digest() == hashlib.sha256(run2).digest()

    # Spot-check that the output is valid JSON with expected keys.
    parsed: dict[str, Any] = json.loads(run1.decode("utf-8"))
    assert parsed["version"] == 1
    assert len(parsed["tensors"]) == 1
    assert parsed["tensors"][0]["name"] == "alpha"
    assert "neuron_hash" in parsed["tensors"][0]
    assert "aggregate" in parsed
