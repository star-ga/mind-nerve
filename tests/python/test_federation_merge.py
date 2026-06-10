"""Determinism tests for the federated route-table merge.

The federation merge is the data-plane primitive that lets one node route a
query against the agents/skills of every node in the mesh. Its load-bearing
property is that the merged table is a *pure function* of the input
manifests: byte-identical regardless of manifest order or the host doing the
merge. These tests pin that contract.
"""

from __future__ import annotations

from mind_nerve.federation import (
    NodeManifest,
    compute_table_hash,
    manifest_from_json,
    merge_manifests,
    to_json,
)


def _routes(*items: tuple[str, str, float]) -> list[dict]:
    return [
        {
            "id": rid,
            "name": name,
            "kind": "skill",
            "source_repo": "starga",
            "sha256": rid,
            "score": score,
        }
        for (rid, name, score) in items
    ]


def _manifest(node_id: str, routes: list[dict]) -> NodeManifest:
    return NodeManifest(node_id, compute_table_hash(routes), routes)


def test_merge_is_order_independent() -> None:
    a = _manifest("node-a", _routes(("aa11", "alpha", 0.9), ("bb22", "beta", 0.5)))
    b = _manifest("node-b", _routes(("bb22", "beta", 0.8), ("cc33", "gamma", 0.7)))
    c = _manifest("node-c", _routes(("dd44", "delta", 0.6)))

    h1 = merge_manifests([a, b, c]).table_hash
    h2 = merge_manifests([c, a, b]).table_hash
    h3 = merge_manifests([b, c, a]).table_hash

    assert h1 == h2 == h3


def test_collision_keeps_higher_score() -> None:
    a = _manifest("node-a", _routes(("bb22", "beta", 0.5)))
    b = _manifest("node-b", _routes(("bb22", "beta", 0.8)))

    table = merge_manifests([a, b])
    bb = next(r for r in table.routes if r.id == "bb22")

    assert bb.node_id == "node-b"
    assert bb.score == 0.8
    assert len([r for r in table.routes if r.id == "bb22"]) == 1


def test_score_tie_broken_by_node_id_id_sha256() -> None:
    # Same id, same score on two nodes -> deterministic winner by
    # SHA-256(node_id || id) ascending, independent of merge order.
    a = _manifest("node-a", _routes(("zz99", "z", 0.5)))
    b = _manifest("node-b", _routes(("zz99", "z", 0.5)))

    w1 = next(r for r in merge_manifests([a, b]).routes if r.id == "zz99")
    w2 = next(r for r in merge_manifests([b, a]).routes if r.id == "zz99")

    assert w1.node_id == w2.node_id


def test_tampered_manifest_is_dropped() -> None:
    good = _manifest("node-a", _routes(("aa11", "alpha", 0.9)))
    bad = NodeManifest("node-evil", "deadbeef" * 8, _routes(("xx00", "x", 9.9)))

    table = merge_manifests([good, bad])

    assert "node-evil" not in table.contributing_node_ids
    assert all(r.node_id != "node-evil" for r in table.routes)


def test_json_roundtrip_preserves_table_hash() -> None:
    m = _manifest("node-a", _routes(("aa11", "alpha", 0.9), ("bb22", "beta", 0.5)))
    back = manifest_from_json(to_json(m))

    assert back.table_hash == m.table_hash
    assert merge_manifests([m]).table_hash == merge_manifests([back]).table_hash
