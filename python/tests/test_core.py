"""Tests for core model, store CRUD and persistence."""

import os

import pytest

from graphdb import Edge, GraphStore, Node
from graphdb.exceptions import (
    DuplicateIdError,
    EdgeNotFoundError,
    InvalidPropertyError,
    NodeNotFoundError,
)


def test_node_auto_id_and_types():
    n = Node(label="User", properties={"name": "alice", "age": 30, "vip": True})
    assert n.id
    assert n.label == "User"
    assert n.properties["age"] == 30
    assert n.properties["vip"] is True


def test_node_invalid_property():
    with pytest.raises(InvalidPropertyError):
        Node(label="User", properties={"bad": {"nested": 1}})


def test_edge_defaults():
    e = Edge(src_id="a", dst_id="b", label="FOLLOWS")
    assert e.id
    assert e.weight == 1.0


def test_add_get_delete_node():
    store = GraphStore()
    n = store.add_node(Node(label="User", properties={"name": "bob"}))
    assert store.get_node(n.id).properties["name"] == "bob"
    store.delete_node(n.id)
    with pytest.raises(NodeNotFoundError):
        store.get_node(n.id)


def test_duplicate_node_id():
    store = GraphStore()
    n = Node(label="User")
    store.add_node(n)
    with pytest.raises(DuplicateIdError):
        store.add_node(n)


def test_update_node():
    store = GraphStore()
    n = store.add_node(Node(label="User", properties={"name": "x"}))
    store.update_node(n.id, {"name": "y", "age": 5})
    assert store.get_node(n.id).properties == {"name": "y", "age": 5}


def test_add_edge_and_lookup():
    store = GraphStore()
    a = store.add_node(Node(label="User"))
    b = store.add_node(Node(label="User"))
    res = store.add_edge(Edge(src_id=a.id, dst_id=b.id, label="FOLLOWS"))
    assert res.edge.id
    assert store.edges_from(a.id)[0].dst_id == b.id
    assert store.edges_to(b.id)[0].src_id == a.id
    assert store.edges_between(a.id, b.id)


def test_add_edge_missing_node():
    store = GraphStore()
    a = store.add_node(Node(label="User"))
    with pytest.raises(NodeNotFoundError):
        store.add_edge(Edge(src_id=a.id, dst_id="missing", label="X"))


def test_delete_node_cascades_edges():
    store = GraphStore()
    a = store.add_node(Node(label="User"))
    b = store.add_node(Node(label="User"))
    res = store.add_edge(Edge(src_id=a.id, dst_id=b.id, label="FOLLOWS"))
    store.delete_node(a.id)
    with pytest.raises(EdgeNotFoundError):
        store.get_edge(res.edge.id)


def test_nodes_by_label_index():
    store = GraphStore()
    store.add_node(Node(label="User"))
    store.add_node(Node(label="User"))
    store.add_node(Node(label="Post"))
    assert len(store.nodes_by_label("User")) == 2
    assert len(store.nodes_by_label("Post")) == 1


def test_persistence_round_trip(tmp_path):
    path = os.path.join(tmp_path, "graph.json")
    store = GraphStore(path=path)
    a = store.add_node(
        Node(label="User", properties={"name": "alice"}, embedding=[0.1, 0.2, 0.3])
    )
    b = store.add_node(Node(label="User", properties={"name": "bob"}))
    store.add_edge(Edge(src_id=a.id, dst_id=b.id, label="FOLLOWS", weight=2.0))
    store.save()

    store2 = GraphStore(path=path)
    assert len(store2.all_nodes()) == 2
    assert len(store2.all_edges()) == 1
    loaded = store2.get_node(a.id)
    assert loaded.properties["name"] == "alice"
    assert loaded.embedding == [0.1, 0.2, 0.3]


def test_load_missing_file_is_empty(tmp_path):
    path = os.path.join(tmp_path, "does_not_exist.json")
    store = GraphStore(path=path)
    assert store.all_nodes() == []


def test_load_corrupt_file(tmp_path):
    from graphdb.exceptions import PersistenceError

    path = os.path.join(tmp_path, "bad.json")
    with open(path, "w") as fh:
        fh.write("{not valid json")
    with pytest.raises(PersistenceError):
        GraphStore(path=path)
