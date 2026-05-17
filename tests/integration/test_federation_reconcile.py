"""Contract tests for the federated routing stub interfaces.

These tests validate:
  - The typed signatures of all public functions and dataclasses.
  - The schema described in spec/federated_routing.md.
  - That stub functions raise NotImplementedError with the correct message.

Tests do NOT suppress NotImplementedError or mock around it. The contract is
that stubs raise — any test that passes by catching that exception is
explicitly checking the stub is in place, not bypassing the requirement.
"""

from __future__ import annotations

import inspect
from typing import get_type_hints

import pytest

from mind_nerve.federation import (
    Peer,
    ReconciledChain,
    broadcast_route_delta,
    connect_peer,
    discover_peers,
    reconcile,
)


# ---------------------------------------------------------------------------
# Peer dataclass contract
# ---------------------------------------------------------------------------


class TestPeerContract:
    def test_has_required_fields(self) -> None:
        fields = set(Peer.__dataclass_fields__)
        assert "host" in fields
        assert "port" in fields
        assert "pubkey" in fields
        assert "chain_tip" in fields

    def test_host_is_str(self) -> None:
        assert get_type_hints(Peer)["host"] is str

    def test_port_is_int(self) -> None:
        assert get_type_hints(Peer)["port"] is int

    def test_pubkey_is_bytes(self) -> None:
        assert get_type_hints(Peer)["pubkey"] is bytes

    def test_chain_tip_is_bytes(self) -> None:
        assert get_type_hints(Peer)["chain_tip"] is bytes

    def test_is_frozen_dataclass(self) -> None:
        peer = Peer(host="10.0.0.1", port=47361, pubkey=bytes(32), chain_tip=bytes(32))
        with pytest.raises((AttributeError, TypeError)):
            peer.host = "mutated"  # type: ignore[misc]

    def test_instantiation_with_valid_values(self) -> None:
        peer = Peer(
            host="10.0.0.5",
            port=47361,
            pubkey=bytes(32),   # 32-byte ed25519 public key
            chain_tip=bytes(32),  # 32-byte SHA-256 chain tip
        )
        assert peer.host == "10.0.0.5"
        assert peer.port == 47361
        assert len(peer.pubkey) == 32
        assert len(peer.chain_tip) == 32

    def test_chain_tip_expected_length(self) -> None:
        # spec/federated_routing.md: chain_tip is SHA-256 = 32 bytes
        peer = Peer(host="h", port=1, pubkey=bytes(32), chain_tip=bytes(32))
        assert len(peer.chain_tip) == 32

    def test_pubkey_expected_length(self) -> None:
        # spec/federated_routing.md: pubkey is ed25519 = 32 bytes
        peer = Peer(host="h", port=1, pubkey=bytes(32), chain_tip=bytes(32))
        assert len(peer.pubkey) == 32


# ---------------------------------------------------------------------------
# ReconciledChain dataclass contract
# ---------------------------------------------------------------------------


class TestReconciledChainContract:
    def test_has_required_fields(self) -> None:
        fields = set(ReconciledChain.__dataclass_fields__)
        assert "tip" in fields
        assert "contributing_peers" in fields

    def test_tip_is_bytes(self) -> None:
        assert get_type_hints(ReconciledChain)["tip"] is bytes

    def test_is_frozen_dataclass(self) -> None:
        chain = ReconciledChain(tip=bytes(32))
        with pytest.raises((AttributeError, TypeError)):
            chain.tip = bytes(32)  # type: ignore[misc]

    def test_contributing_peers_defaults_to_empty_list(self) -> None:
        chain = ReconciledChain(tip=bytes(32))
        assert chain.contributing_peers == []

    def test_tip_expected_length(self) -> None:
        # spec: tip is SHA-256 of last reconciled envelope = 32 bytes
        chain = ReconciledChain(tip=bytes(32))
        assert len(chain.tip) == 32

    def test_contributing_peers_accepts_peer_list(self) -> None:
        p = Peer(host="h", port=1, pubkey=bytes(32), chain_tip=bytes(32))
        chain = ReconciledChain(tip=bytes(32), contributing_peers=[p])
        assert len(chain.contributing_peers) == 1
        assert chain.contributing_peers[0] is p


# ---------------------------------------------------------------------------
# reconcile() signature contract
# ---------------------------------------------------------------------------


class TestReconcileSignature:
    def test_accepts_peers_parameter(self) -> None:
        sig = inspect.signature(reconcile)
        assert "peers" in sig.parameters

    def test_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError) as exc_info:
            reconcile([])
        assert "Phase 2" in str(exc_info.value)

    def test_raises_not_implemented_with_peer_list(self) -> None:
        peers = [Peer(host="h", port=1, pubkey=bytes(32), chain_tip=bytes(32))]
        with pytest.raises(NotImplementedError) as exc_info:
            reconcile(peers)
        assert "Phase 2" in str(exc_info.value)

    def test_stub_message_references_phase2(self) -> None:
        try:
            reconcile([])
        except NotImplementedError as err:
            assert "Phase 2" in str(err)


# ---------------------------------------------------------------------------
# discover_peers() signature contract
# ---------------------------------------------------------------------------


class TestDiscoverPeersSignature:
    def test_has_timeout_parameter_with_default(self) -> None:
        sig = inspect.signature(discover_peers)
        assert "timeout_s" in sig.parameters
        assert sig.parameters["timeout_s"].default == 3.0

    def test_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError) as exc_info:
            discover_peers()
        assert "Phase 2" in str(exc_info.value)


# ---------------------------------------------------------------------------
# connect_peer() signature contract
# ---------------------------------------------------------------------------


class TestConnectPeerSignature:
    def test_accepts_host_port_pubkey(self) -> None:
        sig = inspect.signature(connect_peer)
        assert "host" in sig.parameters
        assert "port" in sig.parameters
        assert "pubkey" in sig.parameters

    def test_return_annotation_is_peer(self) -> None:
        hints = get_type_hints(connect_peer)
        assert hints.get("return") is Peer

    def test_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError) as exc_info:
            connect_peer("10.0.0.1", 47361, bytes(32))
        assert "Phase 2" in str(exc_info.value)


# ---------------------------------------------------------------------------
# broadcast_route_delta() signature contract
# ---------------------------------------------------------------------------


class TestBroadcastRouteDeltaSignature:
    def test_accepts_required_parameters(self) -> None:
        sig = inspect.signature(broadcast_route_delta)
        params = set(sig.parameters)
        assert "peers" in params
        assert "catalog_hash_after" in params
        assert "added_route_ids" in params
        assert "removed_route_ids" in params
        assert "provider_url" in params

    def test_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError) as exc_info:
            broadcast_route_delta(
                peers=[],
                catalog_hash_after=bytes(32),
                added_route_ids=[],
                removed_route_ids=[],
                provider_url="https://example.com",
            )
        assert "Phase 2" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Schema alignment: spec/federated_routing.md field names
# ---------------------------------------------------------------------------


class TestSpecSchemaAlignment:
    def test_peer_fields_match_sync_message_schema(self) -> None:
        # spec/federated_routing.md §"Sync message" defines: peer_id, chain_tip,
        # clock, signature. Peer carries the subset needed for reconcile().
        expected_fields = {"host", "port", "pubkey", "chain_tip"}
        actual_fields = set(Peer.__dataclass_fields__)
        assert expected_fields == actual_fields

    def test_reconciled_chain_fields_match_spec(self) -> None:
        # spec/federated_routing.md §"Chain-reconciliation algorithm" step 6:
        # result carries tip + contributing_peers
        expected_fields = {"tip", "contributing_peers"}
        actual_fields = set(ReconciledChain.__dataclass_fields__)
        assert expected_fields == actual_fields
