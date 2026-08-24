"""Tests for cosine similarity, top-k and the label-scoped scanner."""

import math

import pytest

from graphdb import Edge, GraphStore, Node
from graphdb.vector import cosine_similarity, top_k_similar
from graphdb.similarity import SimilarityScanner


def test_cosine_identical():
    assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)


def test_cosine_orthogonal():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_opposite():
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_cosine_scaled_is_one():
    # magnitude should not matter after normalisation
    assert cosine_similarity([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)


def test_cosine_known_value():
    a = [1, 2, 3]
    b = [4, 5, 6]
    dot = 1 * 4 + 2 * 5 + 3 * 6
    na = math.sqrt(1 + 4 + 9)
    nb = math.sqrt(16 + 25 + 36)
    assert cosine_similarity(a, b) == pytest.approx(dot / (na * nb))


def test_cosine_zero_vector():
    assert cosine_similarity([0, 0], [1, 1]) == 0.0
    assert cosine_similarity([], [1, 1]) == 0.0


def test_top_k():
    query = [1, 0, 0]
    candidates = [
        ("a", [1, 0, 0]),
        ("b", [0, 1, 0]),
        ("c", [0.9, 0.1, 0]),
    ]
    res = top_k_similar(query, candidates, k=2)
    assert res[0][0] == "a"
    assert res[1][0] == "c"
    assert len(res) == 2


def _make_store_with_labels():
    store = GraphStore()
    # Users
    u1 = store.add_node(Node(label="User", embedding=[1.0, 0.0, 0.0]))
    u2 = store.add_node(Node(label="User", embedding=[0.99, 0.01, 0.0]))
    # Posts
    p1 = store.add_node(Node(label="Post", embedding=[0.0, 1.0, 0.0]))
    p2 = store.add_node(Node(label="Post", embedding=[0.0, 0.98, 0.02]))
    return store, u1, u2, p1, p2


def test_scan_finds_similar_edge():
    store, u1, u2, p1, p2 = _make_store_with_labels()
    # Existing edge u1 -> p1
    store.add_edge(Edge(src_id=u1.id, dst_id=p1.id, label="LIKES"))

    scanner = SimilarityScanner(store)
    # Proposed edge u2 -> p2; u2 ~ u1 and p2 ~ p1
    matches = scanner.scan_before_add(u2, p2, "LIKES", threshold=0.85)
    assert len(matches) == 1
    assert matches[0].existing_edge_id
    assert matches[0].combined_score >= 0.85


def test_scan_respects_edge_label():
    store, u1, u2, p1, p2 = _make_store_with_labels()
    store.add_edge(Edge(src_id=u1.id, dst_id=p1.id, label="LIKES"))
    scanner = SimilarityScanner(store)
    # Different edge label => no match
    matches = scanner.scan_before_add(u2, p2, "SHARES", threshold=0.85)
    assert matches == []


def test_scan_is_label_scoped():
    store, u1, u2, p1, p2 = _make_store_with_labels()
    # Existing edge from a Post to a User (reversed endpoint labels)
    store.add_edge(Edge(src_id=p1.id, dst_id=u1.id, label="LIKES"))
    scanner = SimilarityScanner(store)
    # Proposed edge User -> Post. Endpoint text labels differ from existing,
    # so it must NOT be considered similar.
    matches = scanner.scan_before_add(u2, p2, "LIKES", threshold=0.5)
    assert matches == []


def test_scan_below_threshold():
    store = GraphStore()
    u1 = store.add_node(Node(label="User", embedding=[1.0, 0.0]))
    u2 = store.add_node(Node(label="User", embedding=[0.0, 1.0]))
    p1 = store.add_node(Node(label="Post", embedding=[1.0, 0.0]))
    p2 = store.add_node(Node(label="Post", embedding=[0.0, 1.0]))
    store.add_edge(Edge(src_id=u1.id, dst_id=p1.id, label="LIKES"))
    scanner = SimilarityScanner(store)
    matches = scanner.scan_before_add(u2, p2, "LIKES", threshold=0.85)
    assert matches == []


def test_add_edge_reports_similarity():
    store, u1, u2, p1, p2 = _make_store_with_labels()
    store.add_edge(Edge(src_id=u1.id, dst_id=p1.id, label="LIKES"))
    res = store.add_edge(
        Edge(src_id=u2.id, dst_id=p2.id, label="LIKES"),
        similarity_threshold=0.85,
    )
    # The edge is still added, but similarity is reported.
    assert res.was_deduplicated is True
    assert len(res.similar_edges) == 1
    assert store.get_edge(res.edge.id)


def test_search_similar_nodes_label_scoped():
    store, u1, u2, p1, p2 = _make_store_with_labels()
    results = store.search_similar_nodes([1.0, 0.0, 0.0], label="User", k=5)
    ids = [n.id for n, _ in results]
    assert u1.id in ids and u2.id in ids
    assert p1.id not in ids  # Post nodes excluded by label scoping
