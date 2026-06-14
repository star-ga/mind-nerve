"""Federated routing — Phase 3 stub.

Design: spec/federated_routing.md

All public functions raise NotImplementedError until Phase 2 (native MIND
inference) completes and the typed-edges composition layer is available;
the functional implementation replaces this stub at that point.

Consumers may import and type-check against these interfaces today.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Federated route table — deterministic merge of per-node manifests
# ---------------------------------------------------------------------------
#
# This is the data-plane primitive that lets one mind-nerve instance route a
# query against the agents + skills of EVERY naestro node in the federation,
# not just its own. It is deliberately separate from the evidence-chain
# reconciliation below (`reconcile`/`discover_peers`/...), which is gated on
# Phase 2. A table merge is a pure function of its inputs and needs no native
# inference, so it ships now.
#
# Wedge guardrails (identical contract to scan_repo / route):
#   * The merge is a pure function of the input manifests — no network, no
#     clock, no env. Two nodes fed the same manifest set produce a
#     byte-identical federated table (same `table_hash`).
#   * Each manifest is the node's OWN governed route table, signed by that
#     node. Routes are never trusted because a peer relayed them; the owning
#     node's signature is authoritative and an artifact is fetched from its
#     origin (see `broadcast_route_delta` chain-of-custody rule).
#   * Collisions on `id` keep the higher-score entry; ties broken by
#     SHA-256(node_id || id) ascending — bit-stable, order-independent.


@dataclass(frozen=True)
class FederatedRoute:
    """One route in the merged federated table, tagged with its owner node."""

    id: str
    name: str
    kind: str
    source_repo: str
    sha256: str
    score: float
    node_id: str  # SHA-256(pubkey)[:16] of the node that owns this route

    def _tiebreak_key(self) -> str:
        return hashlib.sha256((self.node_id + self.id).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NodeManifest:
    """A single node's contribution to the federation.

    `routes` is that node's full governed route table (the same JSONL records
    `route_table.jsonl` holds: id, name, kind, source_repo, sha256, ...).
    `table_hash` is SHA-256 over the canonical serialization of `routes`;
    `sig` is the node's ed25519 signature over (node_id || table_hash).
    """

    node_id: str
    table_hash: str
    routes: list[dict[str, Any]]
    sig: bytes = b""


@dataclass(frozen=True)
class FederatedTable:
    """The deterministic merge of a set of node manifests."""

    routes: list[FederatedRoute]
    table_hash: str  # SHA-256 over the canonical merged serialization
    contributing_node_ids: list[str]


def compute_table_hash(routes: list[dict[str, Any]]) -> str:
    """SHA-256 over a node's route list, canonicalized for cross-host stability.

    Routes are sorted by `id` and serialized with sorted keys and no
    whitespace so the digest is independent of on-disk ordering.
    """
    canon = [
        json.dumps(r, sort_keys=True, separators=(",", ":"))
        for r in sorted(routes, key=lambda r: r["id"])
    ]
    blob = "\n".join(canon).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def merge_manifests(manifests: list[NodeManifest]) -> FederatedTable:
    """Merge per-node manifests into one deterministic federated table.

    Pure function: the result depends only on the manifest contents, never on
    the order they are passed or the host doing the merge.

    Collision policy on duplicate route `id`: keep the higher `score`; on a
    score tie keep the lower SHA-256(node_id || id). This mirrors the
    cross-arch tie-break used by `route()` and `scan_repo`, so the federated
    table is bit-stable.

    Manifests whose `table_hash` does not match a recomputation over their
    `routes` are dropped (a node that misreports its table is excluded, not
    trusted).
    """
    best: dict[str, FederatedRoute] = {}
    contributing: set[str] = set()

    for m in sorted(manifests, key=lambda m: m.node_id):
        if compute_table_hash(m.routes) != m.table_hash:
            continue  # node misreported its own table — drop it
        contributing.add(m.node_id)
        for r in m.routes:
            fr = FederatedRoute(
                id=r["id"],
                name=r["name"],
                kind=r["kind"],
                source_repo=r.get("source_repo", ""),
                sha256=r.get("sha256", r["id"]),
                score=float(r.get("score", 0.0)),
                node_id=m.node_id,
            )
            prev = best.get(fr.id)
            if prev is None or _better(fr, prev):
                best[fr.id] = fr

    merged = sorted(
        best.values(),
        key=lambda fr: (-fr.score, fr._tiebreak_key()),
    )
    canon = "\n".join(f"{fr.id}|{fr.node_id}|{fr.score:.6f}|{fr.sha256}" for fr in merged).encode(
        "utf-8"
    )
    return FederatedTable(
        routes=merged,
        table_hash=hashlib.sha256(canon).hexdigest(),
        contributing_node_ids=sorted(contributing),
    )


def _better(a: FederatedRoute, b: FederatedRoute) -> bool:
    """True if `a` should win a route-id collision against `b`."""
    if a.score != b.score:
        return a.score > b.score
    return a._tiebreak_key() < b._tiebreak_key()


def load_local_manifest(route_table_path: str, node_id: str) -> NodeManifest:
    """Build this node's manifest from its on-disk `route_table.jsonl`.

    The resulting `table_hash` is a pure function of the table's contents, so
    two nodes with the same catalog publish the same hash.
    """
    routes: list[dict[str, Any]] = []
    with open(route_table_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                routes.append(json.loads(line))
    return NodeManifest(
        node_id=node_id,
        table_hash=compute_table_hash(routes),
        routes=routes,
    )


def manifest_from_json(obj: dict[str, Any]) -> NodeManifest:
    """Parse a peer manifest emitted by `to_json` back into a NodeManifest."""
    return NodeManifest(
        node_id=obj["node_id"],
        table_hash=obj["table_hash"],
        routes=obj["routes"],
    )


def to_json(m: NodeManifest) -> dict[str, Any]:
    """Serialize a node manifest for publication to peers."""
    return {
        "protocol": "mind-nerve-federation/1",
        "node_id": m.node_id,
        "table_hash": m.table_hash,
        "routes": m.routes,
    }


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
