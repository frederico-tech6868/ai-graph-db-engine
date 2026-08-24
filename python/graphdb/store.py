"""GraphStore: the main in-memory graph store with persistence.

The store is thread-safe (guarded by a ``threading.RLock``) and keeps a
:class:`~graphdb.index.LabelIndex` and :class:`~graphdb.index.PropertyIndex`
in sync automatically. It also maintains adjacency maps for fast edge lookup.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .core import Edge, Node
from .exceptions import (
    DuplicateIdError,
    EdgeNotFoundError,
    NodeNotFoundError,
)
from .index import LabelIndex, PropertyIndex
from .persistence import load_graph, save_graph
from .similarity import SimilarMatch, SimilarityScanner
from .vector import top_k_similar


@dataclass
class AddEdgeResult:
    """Result of :meth:`GraphStore.add_edge`."""

    edge: Edge
    similar_edges: List[SimilarMatch] = field(default_factory=list)
    was_deduplicated: bool = False


class GraphStore:
    """An in-memory property graph with optional JSON persistence."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._nodes: Dict[str, Node] = {}
        self._edges: Dict[str, Edge] = {}
        self._label_index = LabelIndex()
        self._property_index = PropertyIndex()
        # adjacency: node_id -> set(edge_id)
        self._out_edges: Dict[str, Set[str]] = {}
        self._in_edges: Dict[str, Set[str]] = {}
        self._scanner = SimilarityScanner(self)
        if self.path:
            self.load()

    # ------------------------------------------------------------------ nodes
    def add_node(self, node: Node) -> Node:
        with self._lock:
            if node.id in self._nodes:
                raise DuplicateIdError(f"node id already exists: {node.id}")
            self._nodes[node.id] = node
            self._label_index.add(node)
            self._property_index.add(node)
            self._out_edges.setdefault(node.id, set())
            self._in_edges.setdefault(node.id, set())
            return node

    def get_node(self, node_id: str) -> Node:
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                raise NodeNotFoundError(f"no such node: {node_id}")
            return node

    def get_node_or_none(self, node_id: str) -> Optional[Node]:
        with self._lock:
            return self._nodes.get(node_id)

    def delete_node(self, node_id: str) -> None:
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                raise NodeNotFoundError(f"no such node: {node_id}")
            # Remove connected edges first.
            for edge_id in list(self._out_edges.get(node_id, set())):
                self.delete_edge(edge_id)
            for edge_id in list(self._in_edges.get(node_id, set())):
                self.delete_edge(edge_id)
            self._label_index.remove(node)
            self._property_index.remove(node)
            self._out_edges.pop(node_id, None)
            self._in_edges.pop(node_id, None)
            del self._nodes[node_id]

    def update_node(self, node_id: str, properties: dict) -> Node:
        with self._lock:
            node = self.get_node(node_id)
            # Re-index: remove old property keys, update, re-add.
            self._property_index.remove(node)
            merged = dict(node.properties)
            merged.update(properties)
            # Validate via a throwaway assignment path.
            from .core import validate_properties

            node.properties = validate_properties(merged)
            self._property_index.add(node)
            return node

    def nodes_by_label(self, label: str) -> List[Node]:
        with self._lock:
            return [self._nodes[i] for i in self._label_index.get(label)]

    def all_nodes(self) -> List[Node]:
        with self._lock:
            return list(self._nodes.values())

    def find_nodes(self, label: Optional[str] = None, **props) -> List[Node]:
        """Find nodes by label and/or exact property matches (index-assisted)."""
        with self._lock:
            candidate_ids: Optional[Set[str]] = None
            if label is not None:
                candidate_ids = self._label_index.get(label)
                # Narrow using the property index where possible.
                for key, value in props.items():
                    ids = self._property_index.get(label, key, value)
                    candidate_ids = (
                        ids if candidate_ids is None else candidate_ids & ids
                    )
                nodes = [self._nodes[i] for i in candidate_ids]
            else:
                nodes = list(self._nodes.values())
            # Final filter (handles non-hashable props / no label case).
            result = []
            for n in nodes:
                if all(n.properties.get(k) == v for k, v in props.items()):
                    result.append(n)
            return result

    # ------------------------------------------------------------------ edges
    def add_edge(self, edge: Edge, similarity_threshold: float = 0.85) -> AddEdgeResult:
        with self._lock:
            if edge.id in self._edges:
                raise DuplicateIdError(f"edge id already exists: {edge.id}")
            src = self._nodes.get(edge.src_id)
            dst = self._nodes.get(edge.dst_id)
            if src is None:
                raise NodeNotFoundError(f"edge src node not found: {edge.src_id}")
            if dst is None:
                raise NodeNotFoundError(f"edge dst node not found: {edge.dst_id}")

            similar = self._scanner.scan_before_add(
                src, dst, edge.label, threshold=similarity_threshold
            )

            self._edges[edge.id] = edge
            self._out_edges.setdefault(edge.src_id, set()).add(edge.id)
            self._in_edges.setdefault(edge.dst_id, set()).add(edge.id)

            return AddEdgeResult(
                edge=edge,
                similar_edges=similar,
                was_deduplicated=bool(similar),
            )

    def get_edge(self, edge_id: str) -> Edge:
        with self._lock:
            edge = self._edges.get(edge_id)
            if edge is None:
                raise EdgeNotFoundError(f"no such edge: {edge_id}")
            return edge

    def delete_edge(self, edge_id: str) -> None:
        with self._lock:
            edge = self._edges.get(edge_id)
            if edge is None:
                raise EdgeNotFoundError(f"no such edge: {edge_id}")
            self._out_edges.get(edge.src_id, set()).discard(edge_id)
            self._in_edges.get(edge.dst_id, set()).discard(edge_id)
            del self._edges[edge_id]

    def edges_from(self, node_id: str) -> List[Edge]:
        with self._lock:
            return [self._edges[i] for i in self._out_edges.get(node_id, set())]

    def edges_to(self, node_id: str) -> List[Edge]:
        with self._lock:
            return [self._edges[i] for i in self._in_edges.get(node_id, set())]

    def edges_between(self, src_id: str, dst_id: str) -> List[Edge]:
        with self._lock:
            return [
                self._edges[i]
                for i in self._out_edges.get(src_id, set())
                if self._edges[i].dst_id == dst_id
            ]

    def all_edges(self) -> List[Edge]:
        with self._lock:
            return list(self._edges.values())

    # --------------------------------------------------------------- vector
    def search_similar_nodes(
        self,
        query_vec: List[float],
        label: Optional[str] = None,
        k: int = 5,
    ) -> List[Tuple[Node, float]]:
        """Return the ``k`` nodes most similar to ``query_vec``.

        When ``label`` is provided the search is scoped to nodes with that text
        label only (using the label index).
        """
        with self._lock:
            if label is not None:
                node_ids = self._label_index.get(label)
                nodes = [self._nodes[i] for i in node_ids]
            else:
                nodes = list(self._nodes.values())
            candidates = [
                (n.id, n.embedding) for n in nodes if n.embedding is not None
            ]
            scored = top_k_similar(query_vec, candidates, k=k)
            return [(self._nodes[cid], score) for cid, score in scored]

    # ------------------------------------------------------------ persistence
    def save(self, path: Optional[str] = None) -> None:
        target = path or self.path
        with self._lock:
            save_graph(target, self.all_nodes(), self.all_edges())

    def load(self, path: Optional[str] = None) -> None:
        target = path or self.path
        with self._lock:
            nodes, edges = load_graph(target)
            self._reset()
            for node in nodes:
                # Bypass duplicate check during bulk load.
                self._nodes[node.id] = node
                self._label_index.add(node)
                self._property_index.add(node)
                self._out_edges.setdefault(node.id, set())
                self._in_edges.setdefault(node.id, set())
            for edge in edges:
                self._edges[edge.id] = edge
                self._out_edges.setdefault(edge.src_id, set()).add(edge.id)
                self._in_edges.setdefault(edge.dst_id, set()).add(edge.id)

    def _reset(self) -> None:
        self._nodes.clear()
        self._edges.clear()
        self._label_index.clear()
        self._property_index.clear()
        self._out_edges.clear()
        self._in_edges.clear()

    # ---------------------------------------------------------------- stats
    def stats(self) -> Dict[str, object]:
        with self._lock:
            return {
                "node_count": len(self._nodes),
                "edge_count": len(self._edges),
                "labels": self._label_index.labels(),
                "path": self.path,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._nodes)
