# graphdb_rs — Rust port of the graph DB core (Phase 3)

`graphdb_rs` is a production-quality Rust re-implementation of the same in-memory
property-graph core built in Phase 1 (the pure-Python `graphdb` package), plus a
**Python FFI bridge** built with [PyO3](https://pyo3.rs) so the Phase-2
`ai_memory` layer can *optionally* call the faster Rust engine.

The public behaviour is intentionally identical to the Python engine — most
importantly the **label-scoped similarity scan**, which only ever compares nodes
that share the same text label.

---

## Why Rust?

| Concern | Python (Phase 1) | Rust (Phase 3) |
| --- | --- | --- |
| **Performance** | Interpreted; per-op overhead on every dict/list access | Compiled, monomorphized, `--release` optimized. Tight loops over `Vec<f32>` for cosine similarity run at native speed. |
| **Concurrency** | Bound by the GIL; `threading.RLock` serializes even CPU work | `parking_lot::RwLock` gives real multi-reader / single-writer access; the borrow checker statically guarantees no data races. |
| **Memory safety** | Runtime errors, `None` surprises | Ownership + `Option<T>` make null/aliasing bugs compile errors, not crashes. |
| **Error handling** | Exceptions can escape anywhere | Every fallible API returns `Result<T, GraphError>`; the library never panics on recoverable errors. |
| **Zero-cost abstractions** | Iterators/generators have overhead | Iterator chains compile down to the same code as hand-written loops. |

The trade-off is a compile step and stricter code, which is exactly why the
Python bridge falls back gracefully when the compiled extension is absent.

---

## Architecture: Python vs Rust

| Python module (`graphdb/`) | Rust module (`src/`) | Notes |
| --- | --- | --- |
| `exceptions.py` | `error.rs` | `GraphError` enum via `thiserror`; `Result<T>` alias. |
| `core.py` (`Node`, `Edge`) | `core.rs` | `Node`, `Edge`, `PropertyValue` enum (typed, `serde`-derived). |
| `vector.py` | `vector.rs` | `cosine_similarity`, `normalize`, `top_k_similar` (bounded min-heap). |
| `index.py` | `index.rs` | `LabelIndex`, `PropertyIndex` over `HashMap`/`HashSet`. |
| `similarity.py` | `similarity.rs` | `SimilarityScanner` — label-scoped, identical logic. |
| `store.py` | `store.rs` | `GraphStore` with adjacency maps + `parking_lot::RwLock`. |
| `query.py` | `query.rs` | `bfs`, `dfs`, `find_path`, fluent `GraphQuery`. |
| `persistence.py` | `persistence.rs` | `serde_json` (de)serialization; atomic write via temp-file + rename. |
| — | `lib.rs` | PyO3 bindings (`Py*` wrapper classes + module functions). |

---

## Build instructions

Rust (stable) is required. Install it with:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
```

### Pure-Rust build & tests

```bash
cd graphdb_rs
cargo build            # compile the rlib + cdylib
cargo test             # run tests/integration_test.rs (11 tests)
cargo clippy           # lint
```

### Build the Python extension

The recommended way (installs into the active virtualenv/conda env):

```bash
./build.sh             # pip install maturin && maturin develop --release
```

If you are **not** in a virtualenv, build a wheel and install it directly:

```bash
pip install maturin
maturin build --release
pip install --force-reinstall target/wheels/graphdb_rs-*.whl
```

---

## Python FFI bridge usage

The `python_bridge` package prefers the compiled Rust extension and falls back
to the pure-Python `graphdb` package automatically:

```python
from python_bridge.bridge import PyGraphStore, bfs, dfs, find_path, get_backend

print(get_backend())            # "rust" if the .so is built, else "python"

store = PyGraphStore()
alice = store.add_node("User", {"name": "Alice"}, [1.0, 0.0, 0.0])
post  = store.add_node("Post", {"title": "Hello"}, [0.0, 1.0, 0.0])

result = store.add_edge(alice.id, post.id, "LIKES", similarity_threshold=0.85)
print(result.was_flagged, len(result.similar_edges))

hits = store.search_similar_nodes([1.0, 0.0, 0.0], label="User", k=5)
print(hits)                     # [(node_id, score), ...]

print(bfs(store, alice.id))     # list of node ids in BFS order
```

Because `bridge.py` degrades to the pure-Python backend, the `ai_memory` layer
keeps working whether or not the Rust `.so` has been compiled.

### Exception mapping

| Rust `GraphError` | Python exception |
| --- | --- |
| `NodeNotFound` / `EdgeNotFound` | `KeyError` |
| `DimensionMismatch` | `ValueError` |
| `Io` | `IOError` / `OSError` |
| `Serde` | `ValueError` |

---

## Performance notes

- **`RwLock` vs the GIL.** Python's GIL means even a `threading.RLock`-guarded
  store cannot run two reads truly in parallel. `parking_lot::RwLock` allows
  many concurrent readers and a single writer with far lower lock overhead than
  `std::sync::RwLock`. In the Rust API, `&self`/`&mut self` already enforce the
  reader/writer discipline at compile time.
- **`Vec<f32>` vs Python `list`.** Embeddings are contiguous `Vec<f32>`, so the
  cosine-similarity inner loop is cache-friendly and auto-vectorizable, versus
  boxed Python floats scattered across the heap.
- **Bounded min-heap for top-k.** `top_k_similar` keeps a `BinaryHeap` of size
  `k`, giving `O(n log k)` selection instead of sorting all `n` candidates.
- **Atomic persistence.** Saves write to a temp file then `rename` over the
  target, so a crash never leaves a half-written graph.

---

## Rust-specific design decisions

- **Ownership over the borrow checker.** `GraphStore` methods take `&self` for
  reads and `&mut self` for writes; the compiler guarantees no aliasing. Internal
  index/adjacency mutations are done through free functions that borrow *disjoint*
  fields, so the `RwLock` guard field never conflicts with data mutation.
- **`Result`-based error handling.** No library function panics on recoverable
  input. Callers (and the PyO3 layer) decide how to react.
- **`Option<Vec<f32>>` for embeddings.** A node either has an embedding or it
  doesn't — encoded in the type, not by a sentinel value. The similarity scanner
  simply short-circuits when either endpoint lacks an embedding.
- **`PropertyValue` enum.** A closed set of typed values (`Str`, `Int`, `Float`,
  `Bool`, `List`) mirrors the Python engine's allowed property types while giving
  exhaustiveness checking and `serde` support for free.
- **`crate-type = ["cdylib", "rlib"]`.** The `cdylib` is the Python extension;
  the `rlib` lets `cargo test` and other Rust crates link the same code.
