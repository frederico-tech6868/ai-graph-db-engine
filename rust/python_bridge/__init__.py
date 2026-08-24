"""python_bridge: FFI bridge exposing the Rust graphdb core to Python.

Re-exports the backend-selecting wrappers from :mod:`python_bridge.bridge`.
"""

from .bridge import BACKEND, PyGraphStore, bfs, dfs, find_path, get_backend

__all__ = ["PyGraphStore", "bfs", "dfs", "find_path", "get_backend", "BACKEND"]
