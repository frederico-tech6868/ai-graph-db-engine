"""Tests for traversal and the query API."""

from graphdb import Edge, GraphQuery, GraphStore, Node
from graphdb.query import bfs, dfs, find_path


def _build_chain():
    """a -> b -> c -> d, plus a -> d shortcut of label SHORTCUT."""
    store = GraphStore()
    a = store.add_node(Node(label="User", properties={"name": "a"}))
    b = store.add_node(Node(label="User", properties={"name": "b"}))
    c = store.add_node(Node(label="User", properties={"name": "c"}))
    d = store.add_node(Node(label="User", properties={"name": "d"}))
    store.add_edge(Edge(src_id=a.id, dst_id=b.id, label="KNOWS"))
    store.add_edge(Edge(src_id=b.id, dst_id=c.id, label="KNOWS"))
    store.add_edge(Edge(src_id=c.id, dst_id=d.id, label="KNOWS"))
    return store, a, b, c, d


def test_bfs_order_and_depth():
    store, a, b, c, d = _build_chain()
    result = bfs(store, a.id, max_depth=5)
    names = [n.properties["name"] for n in result]
    assert names == ["b", "c", "d"]

    shallow = bfs(store, a.id, max_depth=1)
    assert [n.properties["name"] for n in shallow] == ["b"]


def test_dfs_reaches_all():
    store, a, b, c, d = _build_chain()
    result = dfs(store, a.id, max_depth=5)
    names = {n.properties["name"] for n in result}
    assert names == {"b", "c", "d"}


def test_bfs_edge_label_filter():
    store, a, b, c, d = _build_chain()
    store.add_edge(Edge(src_id=a.id, dst_id=d.id, label="SHORTCUT"))
    result = bfs(store, a.id, edge_label="SHORTCUT", max_depth=5)
    assert [n.properties["name"] for n in result] == ["d"]


def test_find_path_shortest():
    store, a, b, c, d = _build_chain()
    store.add_edge(Edge(src_id=a.id, dst_id=d.id, label="SHORTCUT"))
    path = find_path(store, a.id, d.id)
    assert path is not None
    names = [n.properties["name"] for n in path]
    # shortest path is direct a -> d via shortcut
    assert names == ["a", "d"]


def test_find_path_none():
    store, a, b, c, d = _build_chain()
    isolated = store.add_node(Node(label="User", properties={"name": "z"}))
    assert find_path(store, a.id, isolated.id) is None


def test_find_path_same_node():
    store, a, b, c, d = _build_chain()
    path = find_path(store, a.id, a.id)
    assert path == [a] or (path and path[0].id == a.id)


def test_query_match_by_property():
    store, a, b, c, d = _build_chain()
    result = GraphQuery(store).match(label="User", name="b").result()
    assert len(result) == 1
    assert result[0].properties["name"] == "b"


def test_query_match_and_traverse():
    store, a, b, c, d = _build_chain()
    result = (
        GraphQuery(store)
        .match(label="User", name="a")
        .traverse(edge_label="KNOWS", direction="out", max_depth=2)
        .result()
    )
    names = {n.properties["name"] for n in result}
    assert names == {"b", "c"}


def test_query_match_all_by_label():
    store, a, b, c, d = _build_chain()
    result = GraphQuery(store).match(label="User").result()
    assert len(result) == 4


def test_traverse_both_directions():
    store, a, b, c, d = _build_chain()
    result = bfs(store, c.id, direction="both", max_depth=5)
    names = {n.properties["name"] for n in result}
    assert names == {"a", "b", "d"}
