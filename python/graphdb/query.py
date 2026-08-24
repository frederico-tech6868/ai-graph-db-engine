"""Graph traversal & query API.

Provides a fluent :class:`GraphQuery` plus standalone ``bfs``, ``dfs`` and
``find_path`` helpers.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Set, TYPE_CHECKING

from .core import Node

if TYPE_CHECKING:  # pragma: no cover
    from .store import GraphStore


def _neighbors(
    store: "GraphStore",
    node_id: str,
    edge_label: Optional[str],
    direction: str,
) -> List[str]:
    """Return neighbour node ids for the given direction/edge label."""
    result: List[str] = []
    if direction in ("out", "both"):
        for edge in store.edges_from(node_id):
            if edge_label is None or edge.label == edge_label:
                result.append(edge.dst_id)
    if direction in ("in", "both"):
        for edge in store.edges_to(node_id):
            if edge_label is None or edge.label == edge_label:
                result.append(edge.src_id)
    return result


def bfs(
    store: "GraphStore",
    start_id: str,
    edge_label: Optional[str] = None,
    max_depth: int = 5,
    direction: str = "out",
) -> List[Node]:
    """Breadth-first traversal from ``start_id`` (excludes the start node)."""
    visited: Set[str] = {start_id}
    order: List[Node] = []
    queue: Deque = deque([(start_id, 0)])
    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for nb in _neighbors(store, current, edge_label, direction):
            if nb not in visited:
                visited.add(nb)
                node = store.get_node_or_none(nb)
                if node is not None:
                    order.append(node)
                    queue.append((nb, depth + 1))
    return order


def dfs(
    store: "GraphStore",
    start_id: str,
    edge_label: Optional[str] = None,
    max_depth: int = 5,
    direction: str = "out",
) -> List[Node]:
    """Depth-first traversal from ``start_id`` (excludes the start node)."""
    visited: Set[str] = {start_id}
    order: List[Node] = []

    def _visit(node_id: str, depth: int) -> None:
        if depth >= max_depth:
            return
        for nb in _neighbors(store, node_id, edge_label, direction):
            if nb not in visited:
                visited.add(nb)
                node = store.get_node_or_none(nb)
                if node is not None:
                    order.append(node)
                    _visit(nb, depth + 1)

    _visit(start_id, 0)
    return order


def find_path(
    store: "GraphStore",
    src_id: str,
    dst_id: str,
    edge_label: Optional[str] = None,
    direction: str = "out",
) -> Optional[List[Node]]:
    """Return the shortest path (list of nodes) from ``src`` to ``dst``.

    Uses BFS. Returns ``None`` if no path exists. The returned list includes
    both endpoints.
    """
    if store.get_node_or_none(src_id) is None:
        return None
    if store.get_node_or_none(dst_id) is None:
        return None
    if src_id == dst_id:
        return [store.get_node(src_id)]

    visited: Set[str] = {src_id}
    prev: dict = {}
    queue: Deque[str] = deque([src_id])
    found = False
    while queue:
        current = queue.popleft()
        if current == dst_id:
            found = True
            break
        for nb in _neighbors(store, current, edge_label, direction):
            if nb not in visited:
                visited.add(nb)
                prev[nb] = current
                queue.append(nb)
    if not found and dst_id not in prev:
        return None

    # Reconstruct path.
    path_ids: List[str] = [dst_id]
    while path_ids[-1] != src_id:
        parent = prev.get(path_ids[-1])
        if parent is None:
            return None
        path_ids.append(parent)
    path_ids.reverse()
    return [store.get_node(i) for i in path_ids]


class GraphQuery:
    """A small fluent query/traversal builder over a :class:`GraphStore`."""

    def __init__(self, store: "GraphStore") -> None:
        self._store = store
        self._current: List[Node] = []
        self._matched = False

    def match(self, label: Optional[str] = None, **props) -> "GraphQuery":
        """Filter nodes by label and/or exact property values."""
        self._current = self._store.find_nodes(label=label, **props)
        self._matched = True
        return self

    def traverse(
        self,
        edge_label: Optional[str] = None,
        direction: str = "out",
        max_depth: int = 1,
    ) -> "GraphQuery":
        """Traverse from currently matched nodes, collecting reachable nodes."""
        if not self._matched:
            self._current = self._store.all_nodes()
            self._matched = True
        seen: Set[str] = set()
        collected: List[Node] = []
        for start in self._current:
            for node in bfs(
                self._store,
                start.id,
                edge_label=edge_label,
                max_depth=max_depth,
                direction=direction,
            ):
                if node.id not in seen:
                    seen.add(node.id)
                    collected.append(node)
        self._current = collected
        return self

    def result(self) -> List[Node]:
        """Return the current list of matched/traversed nodes."""
        return list(self._current)
