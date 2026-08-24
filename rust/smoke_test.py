"""Quick smoke test of the compiled Rust extension via Python."""
import graphdb_rs
from graphdb_rs import PyGraphStore, bfs, dfs, find_path

print("graphdb_rs version:", graphdb_rs.__version__)

store = PyGraphStore()

# Nodes with embeddings
alice = store.add_node("User", {"name": "Alice", "age": 30}, [1.0, 0.0, 0.0])
post1 = store.add_node("Post", {"title": "Hello"}, [0.0, 1.0, 0.0])
bob = store.add_node("User", {"name": "Bob"}, [0.99, 0.01, 0.0])
post2 = store.add_node("Post", {"title": "Hi"}, [0.01, 0.99, 0.0])

print("node_count:", store.node_count())
assert store.node_count() == 4
assert alice.properties["name"] == "Alice"
assert alice.properties["age"] == 30
assert alice.embedding == [1.0, 0.0, 0.0]

# First edge — not flagged
r1 = store.add_edge(alice.id, post1.id, "LIKES")
assert not r1.was_flagged, "first edge should not be flagged"

# Second, similar edge — flagged
r2 = store.add_edge(bob.id, post2.id, "LIKES")
print("was_flagged:", r2.was_flagged, "similar:", len(r2.similar_edges))
assert r2.was_flagged
assert r2.similar_edges[0].existing_edge_id == r1.edge.id
assert r2.similar_edges[0].combined_score >= 0.85

# Label-scoped similarity search
results = store.search_similar_nodes([1.0, 0.0, 0.0], "User", 5)
print("search_similar_nodes(User):", results)
assert results[0][0] == alice.id
assert abs(results[0][1] - 1.0) < 1e-6
assert all(store.get_node(nid).label == "User" for nid, _ in results)

# Traversal
c = store.add_node("N")
d = store.add_node("N")
e = store.add_node("N")
store.add_edge(c.id, d.id, "E")
store.add_edge(d.id, e.id, "E")
assert bfs(store, c.id) == [d.id, e.id]
assert dfs(store, c.id) == [d.id, e.id]
assert find_path(store, c.id, e.id) == [c.id, d.id, e.id]
assert find_path(store, e.id, c.id) is None

# Error mapping: missing node -> KeyError; bad edge endpoint -> KeyError
try:
    store.get_node("does-not-exist")
    raise AssertionError("expected KeyError")
except KeyError:
    pass

# Persistence round-trip
import tempfile, os
path = os.path.join(tempfile.gettempdir(), "graphdb_rs_smoke.json")
s = PyGraphStore(path)
n = s.add_node("User", {"name": "Zoe"}, [0.5, 0.5])
s.save()
s2 = PyGraphStore(path)
s2.load()
assert s2.node_count() == 1
assert s2.get_node(n.id).properties["name"] == "Zoe"
os.remove(path)

print("\nALL SMOKE TESTS PASSED ✔")
