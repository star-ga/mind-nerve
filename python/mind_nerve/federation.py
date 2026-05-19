"""Federated routing — Phase 3 stub.

Design: spec/federated_routing.md

All public functions raise NotImplementedError until Phase 2 (native MIND
inference) completes and the typed-edges composition layer is available;
the functional implementation replaces this stub at that point.

Consumers may import and type-check against these interfaces today.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Peer:
    """A remote mind-nerve instance participating in the federation mesh.

    ``pubkey`` is the peer's 32-byte ed25519 public key. It is pinned on first
    contact (TOFU) and used to verify every subsequent message from this peer.

    ``chain_tip`` is the 32-byte SHA-256 of the peer's most recently emitted
    attestation envelope, as defined in spec/architecture.md §"Attestation
    envelope". It is NOT stored in the envelope itself; it is computed as
    SHA-256(212-byte envelope serialization) by the verifier.
    """

    host: str
    port: int
    pubkey: bytes  # 32-byte ed25519 public key
    chain_tip: bytes  # 32-byte SHA-256 of the peer's latest envelope


@dataclass(frozen=True)
class ReconciledChain:
    """Result of a chain-reconciliation pass across a set of peers.

    ``tip`` is the SHA-256 of the last envelope in the merged chain after
    reconciliation. All ``contributing_peers`` have adopted this tip.

    ``contributing_peers`` lists every peer whose envelope sequence was
    merged into the reconciled chain. A peer that could not be reached or
    whose envelopes failed verification is excluded from this list.
    """

    tip: bytes  # 32-byte SHA-256
    contributing_peers: list[Peer] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stub functions
# ---------------------------------------------------------------------------

_PHASE2_BLOCKED = "functional ship requires Phase 2 completion"


def reconcile(peers: list[Peer]) -> ReconciledChain:
    """Reconcile the attestation chains of the given peers.

    Algorithm (spec/federated_routing.md §"Chain-reconciliation algorithm"):

    1. Exchange sync messages with each reachable peer to obtain their
       current ``chain_tip`` and vector clock.
    2. Detect divergence by walking ``chain_prev`` links from each tip
       back to a common ancestor (depth limit: 1024 envelopes).
    3. Collect envelope sequences between the common ancestor and each tip.
    4. Verify all received envelopes (signature, ``chain_prev`` linkage,
       reserved-byte check, architecture-enum validity).
    5. Merge passing envelopes in vector-clock order (Lamport 1978),
       tie-broken by SHA-256(envelope_bytes) ascending.
    6. Return the reconciled ``ReconciledChain`` with the new tip and the
       list of peers whose sequences were successfully merged.

    Peers whose envelopes fail verification or who are unreachable are
    excluded from ``contributing_peers`` in the result.

    Args:
        peers: The set of remote instances to reconcile against. May be
            discovered via mDNS or from explicit config.

    Returns:
        A ``ReconciledChain`` with the merged tip and contributing peers.

    Raises:
        NotImplementedError: always (stub — Phase 2 gate).
    """
    raise NotImplementedError(_PHASE2_BLOCKED)


def discover_peers(timeout_s: float = 3.0) -> list[Peer]:
    """Discover federation peers via mDNS (RFC 6762 / RFC 6763).

    Listens on the ``_mind-nerve._tcp.local.`` service type for
    ``timeout_s`` seconds and returns the set of peers found.

    Each discovered peer's public key is extracted from the TXT record
    ``pubkey=<64 hex chars>`` and stored as 32 bytes.

    Args:
        timeout_s: How long to listen for mDNS announcements.

    Returns:
        List of discovered ``Peer`` instances, possibly empty.

    Raises:
        NotImplementedError: always (stub — Phase 2 gate).
    """
    raise NotImplementedError(_PHASE2_BLOCKED)


def connect_peer(host: str, port: int, pubkey: bytes) -> Peer:
    """Explicitly connect to a peer by address and pin its public key.

    Performs a handshake (exchanges sync message), verifies the peer's
    reported chain tip signature, and returns a populated ``Peer``.

    Args:
        host: Hostname or IP address of the remote instance.
        port: Federation listener port on the remote instance.
        pubkey: Expected 32-byte ed25519 public key. Connection is aborted
            if the peer presents a different key during the handshake.

    Returns:
        A ``Peer`` with the current ``chain_tip`` from the handshake.

    Raises:
        ValueError: if ``pubkey`` is not exactly 32 bytes.
        NotImplementedError: always (stub — Phase 2 gate).
    """
    raise NotImplementedError(_PHASE2_BLOCKED)


def broadcast_route_delta(
    peers: list[Peer],
    catalog_hash_after: bytes,
    added_route_ids: list[str],
    removed_route_ids: list[str],
    provider_url: str,
) -> dict[str, bool]:
    """Notify peers of a local route-table delta.

    Sends a compact propagation message (spec/federated_routing.md
    §"Route-table propagation") to each peer in ``peers``. The message is
    signed with the local instance's ed25519 private key.

    Receiving peers will fetch missing routes from ``provider_url``
    directly — not from the local instance — to preserve the marketplace
    chain of custody.

    Args:
        peers: Peers to notify.
        catalog_hash_after: 32-byte SHA-256 of the local catalog after the
            delta was applied.
        added_route_ids: Route IDs added in this delta (16 hex chars each).
        removed_route_ids: Route IDs removed in this delta.
        provider_url: The marketplace URL the receiving peers should fetch
            the new routes from.

    Returns:
        Dict mapping peer host:port strings to ``True`` (notified) or
        ``False`` (unreachable / signature rejected).

    Raises:
        NotImplementedError: always (stub — Phase 2 gate).
    """
    raise NotImplementedError(_PHASE2_BLOCKED)
