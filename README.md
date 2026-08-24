# Graph DB Engine

A fully functional **graph database engine written from scratch**, with label-scoped
vector (embedding) similarity search, an AI agent memory layer, and a Rust port.

## Repository layout

| Path | Description |
|------|-------------|
| [`python/`](./python) | **Phase 1 & 2** — Pure-Python graph DB engine + AI agent memory layer |
| [`rust/`](./rust) | **Phase 3** — Rust port of the core engine with a PyO3 Python FFI bridge |

## Phase 1 — Python Graph DB Engine (`python/graphdb/`)
- `Node` / `Edge` property-graph model with UUID IDs and typed properties
- Cosine-similarity vector search (`vector.py`)
- **Label-scoped** `SimilarityScanner` — before an edge is added, only compares
  node pairs that share the same text label (e.g. `User` vs `User` only)
- `LabelIndex` + `PropertyIndex` for O(1) lookups
- Thread-safe `GraphStore`, BFS/DFS/shortest-path query API, JSON persistence
- Interactive REPL CLI (`cli.py`)

## Phase 2 — AI Agent Memory (`python/ai_memory/`)
- `LocalEmbedder` (deterministic, no API key) + optional `OpenAIEmbedder`
- `AgentMemory` — `remember()` / `recall()` / `reflect()`, entity & session graph
- Semantic deduplication scoped strictly to `Memory` nodes
- `GraphAgent` chat pipeline + OpenAI function-calling tool schemas

## Phase 3 — Rust Engine (`rust/`)
- Production-quality Rust re-implementation of the core engine
- `parking_lot::RwLock`, `serde` persistence, min-heap top-k search
- PyO3 bindings (`PyGraphStore`, `PyNode`, `PyEdge`, …) with a Python
  fallback bridge (`python_bridge/bridge.py`)

## Quick start

### Python
```bash
cd python
pip install -e .
python -m pytest tests/ -v
python ai_memory/demo.py
```

### Rust
```bash
cd rust
cargo test
./build.sh          # build the Python extension via maturin
python smoke_test.py
```
