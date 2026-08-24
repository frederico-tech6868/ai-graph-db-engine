"""AgentMemory: the main long-term memory API for an agent.

``AgentMemory`` turns the generic graph store into a semantic memory:

* :meth:`remember` stores a new memory node (with embedding), deduplicates it
  against existing memories, links it to entities/sessions and chains it to the
  previous memory.
* :meth:`recall` / :meth:`get_context` retrieve relevant memories.
* :meth:`reflect` synthesises a higher-level reflection from recent memories.
* entity and session management helpers.

Design notes / constraints honoured:

* Semantic deduplication (:meth:`_check_duplicate`) uses **label-scoped**
  vector search over :data:`~ai_memory.schema.MEMORY` nodes only, and the
  :class:`~graphdb.similarity.SimilarityScanner` is used when adding
  ``SIMILAR_TO`` edges between memory nodes -- so we never compare memory
  embeddings against entity or session embeddings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, TYPE_CHECKING

from graphdb.core import Edge, Node
from graphdb.similarity import SimilarityScanner

from .recall import RecallEngine, RecalledMemory
from .schema import (
    AGENT,
    ENTITY,
    FOLLOWS,
    KNOWS,
    MEMORY,
    OCCURRED_IN,
    RELATES_TO,
    REMEMBERS,
    SESSION,
    SIMILAR_TO,
    MemoryType,
)

if TYPE_CHECKING:  # pragma: no cover
    from graphdb.store import GraphStore
    from .embedder import Embedder


@dataclass
class SimilarMemory:
    """An existing memory node found to be similar to a new one."""

    node: Node
    score: float


@dataclass
class MemoryResult:
    """Result of :meth:`AgentMemory.remember`."""

    memory_node: Node
    similar_existing: List[SimilarMemory] = field(default_factory=list)
    was_duplicate: bool = False


def _now() -> float:
    return time.time()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


class AgentMemory:
    """Long-term semantic memory for a single agent."""

    def __init__(
        self,
        agent_id: str,
        store: "GraphStore",
        embedder: "Embedder",
        similarity_threshold: float = 0.85,
    ) -> None:
        self.agent_id = agent_id
        self.store = store
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.recall_engine = RecallEngine(store, embedder)
        # Used when adding SIMILAR_TO edges between Memory nodes (label-scoped).
        self._scanner = SimilarityScanner(store)
        self._agent_node = self._ensure_agent_node()

    # ---------------------------------------------------------- agent node
    def _ensure_agent_node(self) -> Node:
        existing = self.store.find_nodes(label=AGENT, agent_id=self.agent_id)
        if existing:
            return existing[0]
        node = Node(label=AGENT, properties={"agent_id": self.agent_id})
        self.store.add_node(node)
        return node

    # -------------------------------------------------------------- remember
    def remember(
        self,
        text: str,
        memory_type: MemoryType = MemoryType.OBSERVATION,
        entities: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> MemoryResult:
        """Store a new memory and wire it into the graph.

        Steps: embed -> check for near-duplicates (label-scoped) -> create the
        Memory node -> link to the agent (``REMEMBERS``), session
        (``OCCURRED_IN``), entities (``RELATES_TO``) -> chain to the previous
        memory (``FOLLOWS``) -> connect to similar memories (``SIMILAR_TO``).
        """
        embedding = self.embedder.embed(text)

        # 1) Semantic deduplication against existing Memory nodes only.
        similar = self._check_duplicate(embedding)
        was_duplicate = bool(similar)

        # 2) Create the memory node.
        ts = _now()
        props: Dict[str, object] = {
            "text": text,
            "memory_type": str(memory_type),
            "timestamp": ts,
            "created_at": _iso(ts),
            "agent_id": self.agent_id,
        }
        if session_id:
            props["session_id"] = session_id
        if metadata:
            for key, value in metadata.items():
                # Only keep primitive-typed metadata (graphdb requirement).
                if isinstance(value, (str, int, float, bool, list)):
                    props[key] = value
        mem = Node(label=MEMORY, properties=props, embedding=embedding)
        self.store.add_node(mem)

        # 3) Agent REMEMBERS memory.
        self.store.add_edge(Edge(src_id=self._agent_node.id, dst_id=mem.id, label=REMEMBERS))

        # 4) Memory OCCURRED_IN session.
        if session_id:
            session_node = self._get_session_node(session_id)
            if session_node is not None:
                self.store.add_edge(
                    Edge(src_id=mem.id, dst_id=session_node.id, label=OCCURRED_IN)
                )

        # 5) Entities.
        for name in entities or []:
            self.relate_memory_to_entity(mem.id, name)

        # 6) Temporal chain.
        self._chain_memory(mem, session_id)

        # 7) Link to similar existing memories (SIMILAR_TO, uses the scanner).
        for sim in similar:
            self._link_similar(mem, sim.node)

        return MemoryResult(
            memory_node=mem,
            similar_existing=similar,
            was_duplicate=was_duplicate,
        )

    # --------------------------------------------------------------- recall
    def recall(
        self,
        query: str,
        k: int = 5,
        memory_type: Optional[MemoryType] = None,
        session_id: Optional[str] = None,
    ) -> List[RecalledMemory]:
        """Retrieve the ``k`` memories most relevant to ``query``."""
        filters: Dict[str, object] = {}
        if memory_type is not None:
            filters["memory_type"] = str(memory_type)
        if session_id is not None:
            filters["session_id"] = session_id
        return self.recall_engine.search(query, label=MEMORY, k=k, filters=filters)

    def get_context(self, query: str, k: int = 10) -> str:
        """Return a formatted context window for ``query`` (for an LLM prompt)."""
        recalled = self.recall(query, k=k)
        return self.recall_engine.build_context_window(recalled)

    # -------------------------------------------------------------- reflect
    def reflect(self, recent_k: int = 20) -> str:
        """Synthesise a higher-level reflection from recent memories.

        Offline-friendly: it summarises the most recent memories and the most
        frequently referenced entities into a short reflection string, then
        stores that reflection as a ``REFLECTION`` memory.
        """
        recent = self._recent_memories(recent_k)
        if not recent:
            return "No memories to reflect on yet."

        # Tally referenced entities across the recent memories.
        entity_counts: Dict[str, int] = {}
        for mem in recent:
            for ent in self._entities_of_memory(mem.id):
                name = str(ent.properties.get("name", ""))
                if name:
                    entity_counts[name] = entity_counts.get(name, 0) + 1

        top_entities = sorted(entity_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
        entity_str = ", ".join(f"{name} ({count})" for name, count in top_entities)

        latest = recent[-1].properties.get("text", "")
        reflection = (
            f"Reflection over the last {len(recent)} memories. "
            f"Key entities: {entity_str or 'none identified'}. "
            f"Most recent focus: {latest}"
        )

        # Persist the reflection itself as a memory (without re-reflecting).
        self.remember(reflection, memory_type=MemoryType.REFLECTION)
        return reflection

    # ------------------------------------------------------------- entities
    def add_entity(
        self,
        name: str,
        entity_type: str = "concept",
        properties: Optional[Dict[str, object]] = None,
    ) -> Node:
        """Create (or return existing) entity node and link the agent to it."""
        existing = self.get_entity(name)
        if existing is not None:
            return existing

        props: Dict[str, object] = {"name": name, "entity_type": entity_type}
        if properties:
            for key, value in properties.items():
                if isinstance(value, (str, int, float, bool, list)):
                    props[key] = value
        node = Node(label=ENTITY, properties=props, embedding=self.embedder.embed(name))
        self.store.add_node(node)
        # Agent KNOWS entity.
        self.store.add_edge(Edge(src_id=self._agent_node.id, dst_id=node.id, label=KNOWS))
        return node

    def get_entity(self, name: str) -> Optional[Node]:
        matches = self.store.find_nodes(label=ENTITY, name=name)
        return matches[0] if matches else None

    def relate_memory_to_entity(self, memory_id: str, entity_name: str) -> None:
        """Ensure the entity exists and add a ``RELATES_TO`` edge to it."""
        entity = self.add_entity(entity_name)
        self.store.add_edge(Edge(src_id=memory_id, dst_id=entity.id, label=RELATES_TO))

    # -------------------------------------------------------------- sessions
    def start_session(self, metadata: Optional[Dict[str, object]] = None) -> str:
        """Create a session node and return its id."""
        ts = _now()
        props: Dict[str, object] = {
            "started_at": _iso(ts),
            "started_ts": ts,
            "active": True,
            "agent_id": self.agent_id,
        }
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool, list)):
                    props[key] = value
        node = Node(label=SESSION, properties=props)
        node.properties["session_id"] = node.id
        self.store.add_node(node)
        return node.id

    def end_session(self, session_id: str) -> None:
        node = self._get_session_node(session_id)
        if node is None:
            return
        self.store.update_node(
            node.id, {"active": False, "ended_at": _iso(_now())}
        )

    # ------------------------------------------------------- internal helpers
    def _get_session_node(self, session_id: str) -> Optional[Node]:
        node = self.store.get_node_or_none(session_id)
        if node is not None and node.label == SESSION:
            return node
        matches = self.store.find_nodes(label=SESSION, session_id=session_id)
        return matches[0] if matches else None

    def _chain_memory(self, new_memory_node: Node, session_id: Optional[str] = None) -> None:
        """Link the new memory to the most recent prior memory via ``FOLLOWS``."""
        candidates = [
            n
            for n in self.store.nodes_by_label(MEMORY)
            if n.id != new_memory_node.id
            and n.properties.get("agent_id") == self.agent_id
        ]
        if session_id is not None:
            scoped = [n for n in candidates if n.properties.get("session_id") == session_id]
            # Fall back to global chain if this is the first memory of a session.
            candidates = scoped or candidates
        if not candidates:
            return
        new_ts = new_memory_node.properties.get("timestamp", 0.0)
        prior = [n for n in candidates if n.properties.get("timestamp", 0.0) <= new_ts]
        prior = prior or candidates
        previous = max(prior, key=lambda n: n.properties.get("timestamp", 0.0))
        self.store.add_edge(
            Edge(src_id=new_memory_node.id, dst_id=previous.id, label=FOLLOWS)
        )

    def _check_duplicate(self, embedding: List[float]) -> List[SimilarMemory]:
        """Return existing Memory nodes similar to ``embedding``.

        Uses **label-scoped** vector search (``label="Memory"``) so memory
        embeddings are never compared against entity or session embeddings.
        Only matches at/above ``similarity_threshold`` are returned.
        """
        scored = self.store.search_similar_nodes(embedding, label=MEMORY, k=5)
        return [
            SimilarMemory(node=node, score=score)
            for node, score in scored
            if score >= self.similarity_threshold
        ]

    def _link_similar(self, new_memory: Node, existing_memory: Node) -> None:
        """Add a ``SIMILAR_TO`` edge, using the label-scoped SimilarityScanner.

        ``store.add_edge`` internally runs the :class:`SimilarityScanner`; both
        endpoints are ``Memory`` nodes, so the scan stays scoped to memories.
        """
        self.store.add_edge(
            Edge(src_id=new_memory.id, dst_id=existing_memory.id, label=SIMILAR_TO),
            similarity_threshold=self.similarity_threshold,
        )

    def _recent_memories(self, k: int) -> List[Node]:
        mems = [
            n
            for n in self.store.nodes_by_label(MEMORY)
            if n.properties.get("agent_id") == self.agent_id
            and n.properties.get("memory_type") != str(MemoryType.REFLECTION)
        ]
        mems.sort(key=lambda n: n.properties.get("timestamp", 0.0))
        return mems[-k:]

    def _entities_of_memory(self, memory_id: str) -> List[Node]:
        result: List[Node] = []
        for edge in self.store.edges_from(memory_id):
            if edge.label == RELATES_TO:
                node = self.store.get_node_or_none(edge.dst_id)
                if node is not None:
                    result.append(node)
        return result

    # ----------------------------------------------------------------- stats
    def stats(self) -> Dict[str, object]:
        memories = self.store.nodes_by_label(MEMORY)
        type_counts: Dict[str, int] = {}
        for m in memories:
            mt = str(m.properties.get("memory_type", "observation"))
            type_counts[mt] = type_counts.get(mt, 0) + 1
        return {
            "agent_id": self.agent_id,
            "total_memories": len(memories),
            "total_entities": len(self.store.nodes_by_label(ENTITY)),
            "total_sessions": len(self.store.nodes_by_label(SESSION)),
            "total_edges": len(self.store.all_edges()),
            "memory_types": type_counts,
        }


__all__ = ["AgentMemory", "MemoryResult", "SimilarMemory"]
