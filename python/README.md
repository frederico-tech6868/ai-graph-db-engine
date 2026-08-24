# graphdb — a from-scratch graph database engine

`graphdb` is a lightweight, dependency-free **in-memory property graph
database** written in pure Python. It supports typed nodes and edges, vector
embeddings, **label-scoped cosine-similarity search**, graph traversal
(BFS/DFS/shortest-path), automatic indexing, JSON persistence, thread safety,
and an interactive REPL CLI.

> No external graph-DB libraries are used (no `networkx`, no `neo4j`). `numpy`
> is *optional* — the engine falls back to a pure-Python cosine implementation
> when it is not installed.

---

## Architecture overview

```
graphdb/
  core.py          Node, Edge dataclasses + property typing/validation
  store.py         GraphStore — the main API (CRUD, adjacency, vector search)
  vector.py        cosine_similarity + top_k_similar (numpy optional)
  similarity.py    SimilarityScanner — label-scoped edge similarity scan
  index.py         LabelIndex + PropertyIndex (kept in sync automatically)
  persistence.py   JSON (de)serialization, atomic + corruption-safe
  query.py         GraphQuery fluent API + bfs / dfs / find_path
  exceptions.py    Custom exception hierarchy
cli.py             Interactive REPL (Python `cmd` module)
tests/             pytest suite
```

### Data model
- **Node**: `id` (UUID), `label` (text type), `properties` (typed dict),
  optional `embedding` (list of floats).
- **Edge**: `id` (UUID), `src_id`, `dst_id`, `label`, `properties`, `weight`.
- Property values may be `str`, `int`, `float`, `bool`, or `list` of primitives.
  Everything else is rejected with `InvalidPropertyError`.

### Indexing
- `LabelIndex` maps `label -> {node_id}` for O(1) label lookups.
- `PropertyIndex` maps `(label, key, value) -> {node_id}` for O(1) property
  lookups. Both are updated automatically on add/update/delete.
- The store also maintains **adjacency maps** (`out_edges` / `in_edges`) so edge
  lookups by endpoint are O(1).

### Thread safety
Every public `GraphStore` mutation and read is guarded by a
`threading.RLock`, so a store can be shared safely across threads.

---

## How the label-scoped similarity scan works

The signature feature. Before adding a new edge `src --[label]--> dst`, the
`SimilarityScanner` looks for **near-duplicate relationships** — but it only
ever compares nodes of the *same text label*.

`scan_before_add(src_node, dst_node, label, threshold=0.85)`:

1. Uses the **label index** to fetch the id sets for `src_node.label` and
   `dst_node.label`. This immediately skips every node of a different type.
2. Iterates existing edges, keeping only those where:
   - the edge label equals `label`, **and**
   - the edge's source is in the `src_node.label` set, **and**
   - the edge's destination is in the `dst_node.label` set.
3. For each qualifying edge it computes:
   - `src_similarity = cosine(src_node.embedding, existing_src.embedding)`
   - `dst_similarity = cosine(dst_node.embedding, existing_dst.embedding)`
   - `combined_score = (src_similarity + dst_similarity) / 2`
4. Returns every `SimilarMatch` whose `combined_score >= threshold`, sorted
   descending.

This guarantees a `User` embedding is never compared against a `Post`
embedding, and the label index keeps the scan efficient by pruning
non-matching endpoints up front.

`GraphStore.add_edge` runs this scan automatically and returns an
`AddEdgeResult(edge, similar_edges, was_deduplicated)`. The edge is **always
added** — the caller decides whether to deduplicate or skip based on the
reported matches.

---

## Install

```bash
cd /home/ubuntu/graphdb
pip install -e .          # core engine (pure python)
pip install -e ".[fast]"  # + numpy acceleration
pip install -e ".[test]"  # + pytest
```

---

## API examples

### Nodes and edges

```python
from graphdb import GraphStore, Node, Edge

store = GraphStore(path="graph.json")   # path optional

alice = store.add_node(Node(label="User", properties={"name": "alice"},
                            embedding=[0.10, 0.20, 0.30]))
bob   = store.add_node(Node(label="User", properties={"name": "bob"},
                            embedding=[0.11, 0.19, 0.29]))
post  = store.add_node(Node(label="Post", properties={"title": "hello"},
                            embedding=[0.90, 0.10, 0.00]))

result = store.add_edge(Edge(src_id=alice.id, dst_id=post.id, label="LIKES"))
print(result.edge.id, result.was_deduplicated, result.similar_edges)
```

### Label-scoped vector search

```python
# only searches User nodes
hits = store.search_similar_nodes([0.10, 0.20, 0.30], label="User", k=5)
for node, score in hits:
    print(score, node.properties)
```

### Traversal & queries

```python
from graphdb import GraphQuery
from graphdb.query import bfs, dfs, find_path

# fluent API
users = (GraphQuery(store)
         .match(label="User", name="alice")
         .traverse(edge_label="LIKES", direction="out", max_depth=2)
         .result())

# standalone helpers
reachable = bfs(store, alice.id, max_depth=3)
path = find_path(store, alice.id, post.id)   # shortest path or None
```

### Persistence

```python
store.save()                 # writes to the configured path (atomic)
store.load()                 # reloads, rebuilding all indexes
store2 = GraphStore(path="graph.json")   # auto-loads on construction
```

---

## CLI usage

```bash
python cli.py graph.json      # or: graphdb-cli graph.json
```

```
Commands:
  add-node <label> [key=value ...]                     Add a node
  add-edge <src_id> <dst_id> <label> [key=value ...]   Add an edge (similarity scan)
  get-node <id>                                        Show node details
  get-edge <id>                                        Show edge details
  list-nodes [label]                                   List nodes (optionally by label)
  list-edges                                           List all edges
  search <label> <query_text>                          Similarity search (dummy embedding)
  traverse <node_id> [depth]                           BFS traverse from a node
  path <src_id> <dst_id>                               Find shortest path
  save                                                 Save graph to disk
  load                                                 Load graph from disk
  stats                                                Show graph statistics
  help                                                 Show help
  exit                                                 Exit
```

Nodes added via the CLI get a deterministic **dummy embedding** derived from
their `name`/`text` property (a bag-of-hashes vector), so `search` and the
similarity scan produce meaningful results without a real embedding model.

Example session:

```
graphdb> add-node User name=alice
  added node 2f1c... (label=User)
graphdb> add-node User name=alicia
  added node 8ab3... (label=User)
graphdb> search User alice
  top 2 matches for 'alice' in label 'User':
    1.0000  2f1c...  {'name': 'alice'}
    0.7421  8ab3...  {'name': 'alicia'}
```

---

## Running the tests

```bash
cd /home/ubuntu/graphdb
pip install -e ".[test]"
python -m pytest tests/ -v
```

The suite covers node/edge CRUD, cosine correctness on known vectors,
label-scoped similarity scanning, BFS/DFS/shortest-path traversal, the query
API, and a persistence save/load round-trip.

---

## Design constraints honoured

1. No external graph-DB libraries.
2. Pure-Python core; `numpy` optional with graceful fallback.
3. All similarity operations are **label-scoped**.
4. The pre-add similarity scan uses the label index to skip non-matching nodes.
5. Thread-safe store via `threading.RLock`.
6. All ids are UUID strings.
```
