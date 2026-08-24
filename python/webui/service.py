"""Service layer: wraps GraphStore + AgentMemory for the web API.

Keeps the FastAPI handlers thin. All graph mutations go through here so the
web layer never touches the engine internals directly.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from graphdb.core import Edge, Node
from graphdb.query import bfs, find_path
from graphdb.store import GraphStore

from ai_memory.embedder import get_embedder
from ai_memory.memory import AgentMemory
from ai_memory.schema import ENTITY, MEMORY, MemoryType

DEFAULT_PATH = os.environ.get(
    "GRAPHDB_WEB_PATH", "/home/ubuntu/graphdb/webui_graph.json"
)
AGENT_ID = "web-agent"


def _memory_type(value: Optional[str]) -> Optional[MemoryType]:
    if not value:
        return None
    try:
        return MemoryType(value.lower())
    except ValueError:
        return MemoryType.OBSERVATION


class GraphService:
    """Holds a single GraphStore + AgentMemory + embedder for the app."""

    def __init__(self, path: str = DEFAULT_PATH) -> None:
        self.path = path
        self.embedder = get_embedder()
        self._init_store(load=os.path.exists(path))

    # ------------------------------------------------------------- lifecycle
    def _init_store(self, load: bool) -> None:
        # path=None avoids auto-load in the constructor; we load explicitly.
        self.store = GraphStore(path=None)
        self.store.path = self.path
        self.memory = AgentMemory(AGENT_ID, self.store, self.embedder)
        if load:
            try:
                self.store.load(self.path)
                # Re-bind the agent node after loading.
                self.memory = AgentMemory(AGENT_ID, self.store, self.embedder)
            except Exception:
                pass

    def reset(self) -> None:
        if os.path.exists(self.path):
            try:
                os.remove(self.path)
            except OSError:
                pass
        self._init_store(load=False)

    def save(self) -> str:
        self.store.save(self.path)
        return self.path

    def load(self) -> None:
        self._init_store(load=os.path.exists(self.path))

    # --------------------------------------------------------- serialization
    @staticmethod
    def node_dict(node: Node, with_embedding: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": node.id,
            "label": node.label,
            "properties": dict(node.properties),
            "has_embedding": node.embedding is not None,
        }
        if with_embedding:
            data["embedding"] = list(node.embedding) if node.embedding else None
        return data

    @staticmethod
    def edge_dict(edge: Edge) -> Dict[str, Any]:
        return {
            "id": edge.id,
            "src_id": edge.src_id,
            "dst_id": edge.dst_id,
            "label": edge.label,
            "properties": dict(edge.properties),
            "weight": edge.weight,
        }

    @staticmethod
    def match_dict(match) -> Dict[str, Any]:
        return {
            "existing_edge_id": match.existing_edge_id,
            "src_similarity": round(match.src_similarity, 4),
            "dst_similarity": round(match.dst_similarity, 4),
            "combined_score": round(match.combined_score, 4),
        }

    # --------------------------------------------------------------- stats
    def stats(self) -> Dict[str, Any]:
        label_counts: Dict[str, int] = {}
        for label in self.store._label_index.labels():
            label_counts[label] = len(self.store.nodes_by_label(label))
        return {
            "node_count": len(self.store.all_nodes()),
            "edge_count": len(self.store.all_edges()),
            "labels": [
                {"label": k, "count": v}
                for k, v in sorted(label_counts.items(), key=lambda kv: -kv[1])
            ],
            "backend": "python",
            "path": self.path,
            "memory": self.memory.stats(),
        }

    # --------------------------------------------------------------- graph
    def graph(self, label: Optional[str] = None, limit: int = 200) -> Dict[str, Any]:
        nodes = (
            self.store.nodes_by_label(label) if label else self.store.all_nodes()
        )
        nodes = nodes[:limit]
        node_ids = {n.id for n in nodes}
        edges = [
            self.edge_dict(e)
            for e in self.store.all_edges()
            if e.src_id in node_ids and e.dst_id in node_ids
        ]
        return {
            "nodes": [self.node_dict(n) for n in nodes],
            "edges": edges,
        }

    # --------------------------------------------------------------- nodes
    def list_nodes(self, label: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        nodes = self.store.nodes_by_label(label) if label else self.store.all_nodes()
        return [self.node_dict(n) for n in nodes[:limit]]

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        node = self.store.get_node_or_none(node_id)
        if node is None:
            return None
        out = [self.edge_dict(e) for e in self.store.edges_from(node_id)]
        inc = [self.edge_dict(e) for e in self.store.edges_to(node_id)]
        data = self.node_dict(node, with_embedding=False)
        data["out_edges"] = out
        data["in_edges"] = inc
        return data

    def create_node(
        self, label: str, properties: Dict[str, Any], text: Optional[str] = None
    ) -> Dict[str, Any]:
        props = dict(properties or {})
        embedding = None
        if text:
            props.setdefault("text", text)
            embedding = self.embedder.embed(text)
        node = Node(label=label, properties=props, embedding=embedding)
        self.store.add_node(node)
        return self.node_dict(node)

    def update_node(self, node_id: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        node = self.store.update_node(node_id, properties or {})
        return self.node_dict(node)

    def delete_node(self, node_id: str) -> None:
        self.store.delete_node(node_id)

    # --------------------------------------------------------------- edges
    def list_edges(self) -> List[Dict[str, Any]]:
        return [self.edge_dict(e) for e in self.store.all_edges()]

    def create_edge(
        self,
        src_id: str,
        dst_id: str,
        label: str,
        properties: Dict[str, Any],
        weight: float,
        similarity_threshold: float,
    ) -> Dict[str, Any]:
        edge = Edge(
            src_id=src_id,
            dst_id=dst_id,
            label=label,
            properties=properties or {},
            weight=weight,
        )
        result = self.store.add_edge(edge, similarity_threshold=similarity_threshold)
        return {
            "edge": self.edge_dict(result.edge),
            "similar_edges": [self.match_dict(m) for m in result.similar_edges],
            "was_flagged": bool(result.similar_edges),
        }

    def delete_edge(self, edge_id: str) -> None:
        self.store.delete_edge(edge_id)

    # --------------------------------------------------------------- search
    def search(self, text: str, label: Optional[str], k: int) -> List[Dict[str, Any]]:
        query_vec = self.embedder.embed(text)
        results = self.store.search_similar_nodes(query_vec, label=label, k=k)
        return [
            {"node": self.node_dict(n), "score": round(score, 4)}
            for n, score in results
        ]

    # ------------------------------------------------------------- traversal
    def traverse(self, node_id: str, edge_label: Optional[str], max_depth: int):
        nodes = bfs(self.store, node_id, edge_label=edge_label, max_depth=max_depth)
        return [self.node_dict(n) for n in nodes]

    def path(self, src_id: str, dst_id: str):
        result = find_path(self.store, src_id, dst_id)
        if result is None:
            return None
        return [self.node_dict(n) for n in result]

    # --------------------------------------------------------------- memory
    def memory_stats(self) -> Dict[str, Any]:
        return self.memory.stats()

    def remember(
        self,
        text: str,
        memory_type: str,
        entities: List[str],
        session_id: Optional[str],
    ) -> Dict[str, Any]:
        result = self.memory.remember(
            text,
            memory_type=_memory_type(memory_type) or MemoryType.OBSERVATION,
            entities=entities or None,
            session_id=session_id,
        )
        return {
            "memory_node": self.node_dict(result.memory_node),
            "similar_existing": [
                {"node": self.node_dict(s.node), "score": round(s.score, 4)}
                for s in result.similar_existing
            ],
            "was_duplicate": result.was_duplicate,
        }

    def recall(self, query: str, k: int, memory_type: Optional[str]) -> List[Dict[str, Any]]:
        results = self.memory.recall(query, k=k, memory_type=_memory_type(memory_type))
        return [
            {
                "node": self.node_dict(r.node),
                "score": round(r.score, 4),
                "context_snippet": r.context_snippet,
            }
            for r in results
        ]

    def reflect(self) -> str:
        return self.memory.reflect()

    def entities(self) -> List[Dict[str, Any]]:
        return [self.node_dict(n) for n in self.store.nodes_by_label(ENTITY)]

    # --------------------------------------------------------------- seed
    def seed(self) -> Dict[str, Any]:
        """Load a small demo dataset so the UI has something to show."""
        self.reset()
        emb = self.embedder.embed

        def user(name: str, role: str) -> Node:
            n = Node(
                label="User",
                properties={"name": name, "role": role, "text": f"{name} {role}"},
                embedding=emb(f"{name} {role}"),
            )
            self.store.add_node(n)
            return n

        def company(name: str, sector: str) -> Node:
            n = Node(
                label="Company",
                properties={"name": name, "sector": sector, "text": f"{name} {sector}"},
                embedding=emb(f"{name} {sector}"),
            )
            self.store.add_node(n)
            return n

        alice = user("Alice", "ML Engineer")
        bob = user("Bob", "NLP Researcher")
        carol = user("Carol", "Data Scientist")
        techcorp = company("TechCorp", "artificial intelligence software")
        dataworks = company("DataWorks", "data analytics platform")

        self.store.add_edge(Edge(src_id=alice.id, dst_id=techcorp.id, label="WORKS_AT"))
        self.store.add_edge(Edge(src_id=bob.id, dst_id=techcorp.id, label="WORKS_AT"))
        self.store.add_edge(Edge(src_id=carol.id, dst_id=dataworks.id, label="WORKS_AT"))
        self.store.add_edge(Edge(src_id=alice.id, dst_id=bob.id, label="KNOWS"))

        # A couple of memories.
        self.memory.remember(
            "Alice works at TechCorp on machine learning using PyTorch.",
            memory_type=MemoryType.FACT,
            entities=["Alice", "TechCorp", "PyTorch"],
        )
        self.memory.remember(
            "Bob is Alice's colleague at TechCorp working on NLP.",
            memory_type=MemoryType.FACT,
            entities=["Bob", "Alice", "TechCorp"],
        )
        return self.stats()
