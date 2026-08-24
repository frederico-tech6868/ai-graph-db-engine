"""Python wrapper that prefers the compiled Rust extension.

This module tries to import the compiled Rust extension (``graphdb_rs``, built
with maturin/PyO3). If it is not available — e.g. the Rust ``.so`` has not been
built — it transparently falls back to the pure-Python ``graphdb`` package from
Phase 1, so the ``ai_memory`` layer keeps working either way.

Usage::

    from python_bridge.bridge import PyGraphStore, bfs, dfs, find_path, get_backend
    print(get_backend())   # "rust" or "python"
"""

try:
    from graphdb_rs import PyGraphStore, bfs, dfs, find_path  # type: ignore

    BACKEND = "rust"
except ImportError:  # pragma: no cover - exercised when the .so is absent
    from graphdb.store import GraphStore as PyGraphStore  # type: ignore
    from graphdb.query import bfs, dfs, find_path  # type: ignore

    BACKEND = "python"


def get_backend() -> str:
    """Return the active backend: ``"rust"`` or ``"python"``."""
    return BACKEND


__all__ = ["PyGraphStore", "bfs", "dfs", "find_path", "get_backend", "BACKEND"]
