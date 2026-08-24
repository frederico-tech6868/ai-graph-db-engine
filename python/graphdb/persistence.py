"""JSON serialization / deserialization of the full graph."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List, Tuple

from .core import Edge, Node
from .exceptions import PersistenceError

SCHEMA_VERSION = 1


def serialize_graph(nodes: List[Node], edges: List[Edge]) -> Dict[str, Any]:
    """Build a JSON-serialisable dict from nodes + edges."""
    return {
        "version": SCHEMA_VERSION,
        "nodes": [n.to_dict() for n in nodes],
        "edges": [e.to_dict() for e in edges],
    }


def deserialize_graph(data: Dict[str, Any]) -> Tuple[List[Node], List[Edge]]:
    """Rebuild ``(nodes, edges)`` from a parsed JSON dict."""
    if not isinstance(data, dict):
        raise PersistenceError("graph data must be a JSON object")
    nodes = [Node.from_dict(d) for d in data.get("nodes", [])]
    edges = [Edge.from_dict(d) for d in data.get("edges", [])]
    return nodes, edges


def save_graph(path: str, nodes: List[Node], edges: List[Edge]) -> None:
    """Atomically write the graph to ``path`` as JSON."""
    if not path:
        raise PersistenceError("no path configured for save")
    payload = serialize_graph(nodes, edges)
    try:
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        # Atomic write via temp file + rename.
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    except PersistenceError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise PersistenceError(f"failed to save graph to {path}: {exc}") from exc


def load_graph(path: str) -> Tuple[List[Node], List[Edge]]:
    """Load the graph from ``path``.

    Missing files return an empty graph. Corrupt files raise
    :class:`PersistenceError`.
    """
    if not path or not os.path.exists(path):
        return [], []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read().strip()
        if not content:
            return [], []
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise PersistenceError(f"corrupt graph file {path}: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise PersistenceError(f"failed to load graph from {path}: {exc}") from exc
    return deserialize_graph(data)
