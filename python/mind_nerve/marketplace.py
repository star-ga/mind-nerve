"""Skill marketplace adapter — Phase 3 stub.

Design: spec/skill_marketplace.md

All public functions raise NotImplementedError until Phase 2 (native MIND
inference, mindc 0.3.0 cdylib emit) completes and the functional
implementation replaces this stub.

Consumers may import and type-check against these interfaces today.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillProvider:
    """A registered skill marketplace provider.

    ``signature`` is the ed25519 signature (RFC 8032) of
    ``name_bytes || url_bytes || nonce_bytes`` produced by the provider's
    private key. ``nonce`` is a 32-byte random value chosen by the caller
    at registration time and included in the JSON-RPC delta request so
    replayed signatures are detected.

    ``license`` MUST be an SPDX identifier accepted by ``discovery.PUBLIC_LICENSES``
    for any skills emitted by this provider to pass the license gate.
    """

    name: str
    url: str
    signature: bytes  # ed25519 signature of (name + url + nonce); 64 bytes
    license: str


@dataclass(frozen=True)
class ProviderDelta:
    """Incremental route-table update returned by a provider's delta endpoint.

    ``added_count`` and ``removed_count`` reflect the post-license-gate
    numbers — entries that fail signature verification or the license gate
    are not counted.

    ``as_of_version`` is the provider's catalog version cursor; callers
    pass this back as ``since_version`` in the next delta request.
    """

    provider: SkillProvider
    added_count: int
    removed_count: int
    as_of_version: str


# ---------------------------------------------------------------------------
# Stub functions
# ---------------------------------------------------------------------------

_PHASE2_BLOCKED = "functional ship requires Phase 2 completion"


def register_skill_provider(url: str) -> SkillProvider:
    """Fetch provider metadata, pin the ed25519 public key, and persist the
    provider record.

    Calls ``marketplace.describe`` (JSON-RPC 2.0 over HTTPS) to retrieve:
    - ``name``, ``version``, ``license``
    - ``pubkey_ed25519_b64`` — the 32-byte ed25519 public key, base64-encoded
    - ``min_fetch_interval_s`` — rate-limit floor the provider declares

    On first call, the public key is pinned in the provider store. Subsequent
    calls with the same URL detect key rotations and re-verify the full catalog.

    Args:
        url: HTTPS base URL of the provider endpoint. HTTP is rejected.

    Returns:
        A ``SkillProvider`` populated from the ``describe`` response.

    Raises:
        ValueError: if ``url`` is not HTTPS.
        NotImplementedError: always (stub — Phase 2 gate).
    """
    raise NotImplementedError(_PHASE2_BLOCKED)


def list_providers() -> list[SkillProvider]:
    """Return all currently registered skill providers.

    Reads the provider store persisted by previous ``register_skill_provider``
    calls. The list reflects only providers whose public keys remain valid.

    Returns:
        Ordered list of registered ``SkillProvider`` instances.

    Raises:
        NotImplementedError: always (stub — Phase 2 gate).
    """
    raise NotImplementedError(_PHASE2_BLOCKED)


def query_route_delta(provider: SkillProvider) -> bytes:
    """Fetch a route-table delta from ``provider`` and apply it atomically.

    Steps:
    1. Generate a 32-byte random nonce.
    2. Call ``marketplace.delta`` (JSON-RPC 2.0) with the stored
       ``since_version`` cursor and the nonce.
    3. Verify each returned skill entry's ed25519 signature against
       ``name || url || nonce``.
    4. Apply the license gate (``discovery.PUBLIC_LICENSES`` /
       ``discovery.COMMERCIAL_MARKERS``).
    5. Embed passing entries and apply removals via
       ``discovery._save_table_atomic``.
    6. Emit a ``CatalogLoad`` attestation envelope with the updated catalog
       hash.
    7. Return the raw JSON-RPC response body for caller logging.

    Rate limiting: raises ``RateLimitError`` if the provider's
    ``min_fetch_interval_s`` has not elapsed since the last successful fetch.

    Args:
        provider: A registered ``SkillProvider`` (from ``register_skill_provider``
            or ``list_providers``).

    Returns:
        Raw JSON-RPC response bytes from the delta endpoint.

    Raises:
        NotImplementedError: always (stub — Phase 2 gate).
    """
    raise NotImplementedError(_PHASE2_BLOCKED)


def apply_delta(provider: SkillProvider, delta_bytes: bytes) -> ProviderDelta:
    """Parse, verify, and apply a raw delta payload to the local route table.

    Separated from ``query_route_delta`` to allow offline testing with
    pre-fetched payloads. The same verification and license-gate logic applies.

    Args:
        provider: The provider whose public key is used for signature
            verification.
        delta_bytes: Raw JSON-RPC response body from a ``marketplace.delta``
            call.

    Returns:
        A ``ProviderDelta`` summarising the applied changes.

    Raises:
        NotImplementedError: always (stub — Phase 2 gate).
    """
    raise NotImplementedError(_PHASE2_BLOCKED)
