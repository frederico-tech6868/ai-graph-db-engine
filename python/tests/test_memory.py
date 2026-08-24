"""Tests for the AI memory layer (ai_memory.memory.AgentMemory).

These tests use a deterministic mock embedder with known vectors so no API key
or ML model is required.
"""

import math

import pytest

from graphdb import GraphStore
from ai_memory.memory import AgentMemory
from ai_memory.schema import (
    AGENT,
    ENTITY,
    FOLLOWS,
    MEMORY,
    SESSION,
    MemoryType,
)


class MockEmbedder:
    """Deterministic embedder driven by an explicit text->vector mapping.

    Unknown texts get a stable pseudo-random unit vector derived from the text,
    so they are near-orthogonal to the known vectors.
    """

    dim = 4

    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def _fallback(self, text):
        # Deterministic small vector, unlikely to collide with mapped ones.
        h = abs(hash(text)) if False else sum(ord(c) for c in text)
        # spread across dims deterministically
        vec = [((h * (i + 1)) % 7) + 1 for i in range(self.dim)]
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec]

    def embed(self, text):
        if text in self.mapping:
            v = self.mapping[text]
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            return [x / norm for x in v]
        return self._fallback(text)

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


def _make_memory(mapping=None, threshold=0.85):
    store = GraphStore()
    embedder = MockEmbedder(mapping)
    mem = AgentMemory("agent-x", store, embedder, similarity_threshold=threshold)
    return store, mem


# --------------------------------------------------------------------- remember
def test_remember_creates_memory_node():
    store, mem = _make_memory()
    result = mem.remember("the sky is blue", memory_type=MemoryType.FACT)
    node = result.memory_node
    assert node.label == MEMORY
    assert node.properties["text"] == "the sky is blue"
    assert node.properties["memory_type"] == MemoryType.FACT.value
    assert node.embedding is not None
    # Stored in the graph.
    assert store.get_node(node.id).id == node.id
    assert not result.was_duplicate


def test_agent_node_created_once():
    store, mem = _make_memory()
    mem.remember("hello one")
    mem.remember("hello two")
    agents = store.nodes_by_label(AGENT)
    assert len(agents) == 1
    assert agents[0].properties["agent_id"] == "agent-x"


def test_remember_detects_near_duplicate():
    # Two texts map to almost-identical vectors -> high cosine similarity.
    mapping = {
        "cats are great": [1.0, 0.0, 0.0, 0.0],
        "cats are wonderful": [0.99, 0.01, 0.0, 0.0],
        "totally different topic": [0.0, 0.0, 0.0, 1.0],
    }
    store, mem = _make_memory(mapping, threshold=0.9)

    r1 = mem.remember("cats are great")
    assert not r1.was_duplicate

    r2 = mem.remember("cats are wonderful")
    assert r2.was_duplicate
    assert len(r2.similar_existing) >= 1
    assert r2.similar_existing[0].node.id == r1.memory_node.id
    assert r2.similar_existing[0].score >= 0.9

    r3 = mem.remember("totally different topic")
    assert not r3.was_duplicate


def test_dedup_is_label_scoped_to_memory_only():
    # An Entity with a colliding embedding must NOT count as a duplicate memory.
    mapping = {
        "quantum computing": [1.0, 0.0, 0.0, 0.0],
        "Quantum": [1.0, 0.0, 0.0, 0.0],  # entity name, identical vector
    }
    store, mem = _make_memory(mapping, threshold=0.9)
    mem.add_entity("Quantum", "concept")  # creates an Entity node w/ same vector
    result = mem.remember("quantum computing")
    # No Memory node existed before, so despite the identical Entity vector,
    # this must not be flagged as a duplicate memory.
    assert not result.was_duplicate
    assert result.similar_existing == []


# ----------------------------------------------------------------------- recall
def test_recall_returns_most_relevant():
    mapping = {
        "I love pizza": [1.0, 0.0, 0.0, 0.0],
        "pasta is tasty": [0.0, 1.0, 0.0, 0.0],
        "the weather is cold": [0.0, 0.0, 1.0, 0.0],
        "pizza query": [0.95, 0.05, 0.0, 0.0],
    }
    store, mem = _make_memory(mapping, threshold=0.99)
    mem.remember("I love pizza")
    mem.remember("pasta is tasty")
    mem.remember("the weather is cold")

    recalled = mem.recall("pizza query", k=2)
    assert len(recalled) == 2
    assert recalled[0].node.properties["text"] == "I love pizza"
    assert recalled[0].score >= recalled[1].score


def test_recall_filters_by_memory_type():
    store, mem = _make_memory(threshold=0.99)
    mem.remember("a plain observation", memory_type=MemoryType.OBSERVATION)
    mem.remember("an important fact", memory_type=MemoryType.FACT)

    facts = mem.recall("anything", k=10, memory_type=MemoryType.FACT)
    assert all(r.node.properties["memory_type"] == MemoryType.FACT.value for r in facts)
    assert any(r.node.properties["text"] == "an important fact" for r in facts)


# --------------------------------------------------------------------- entities
def test_entity_creation_and_linking():
    store, mem = _make_memory(threshold=0.99)
    result = mem.remember("Alice works here", entities=["Alice", "TechCorp"])
    ent_alice = mem.get_entity("Alice")
    ent_corp = mem.get_entity("TechCorp")
    assert ent_alice is not None and ent_alice.label == ENTITY
    assert ent_corp is not None

    # Memory -> Entity RELATES_TO edges exist.
    related = mem._entities_of_memory(result.memory_node.id)
    names = {n.properties["name"] for n in related}
    assert names == {"Alice", "TechCorp"}


def test_add_entity_idempotent():
    store, mem = _make_memory()
    e1 = mem.add_entity("TechCorp", "organisation")
    e2 = mem.add_entity("TechCorp", "organisation")
    assert e1.id == e2.id
    assert len(store.nodes_by_label(ENTITY)) == 1


# --------------------------------------------------------------------- sessions
def test_session_start_and_end():
    store, mem = _make_memory()
    sid = mem.start_session({"channel": "web"})
    sessions = store.nodes_by_label(SESSION)
    assert len(sessions) == 1
    assert sessions[0].properties["active"] is True
    assert sessions[0].properties["channel"] == "web"

    mem.end_session(sid)
    assert store.get_node(sid).properties["active"] is False
    assert "ended_at" in store.get_node(sid).properties


def test_memory_linked_to_session():
    store, mem = _make_memory()
    sid = mem.start_session()
    result = mem.remember("in-session memory", session_id=sid)
    # OCCURRED_IN edge from memory to session.
    from ai_memory.schema import OCCURRED_IN

    edges = [e for e in store.edges_from(result.memory_node.id) if e.label == OCCURRED_IN]
    assert len(edges) == 1
    assert edges[0].dst_id == sid


# ------------------------------------------------------------------- temporal
def test_temporal_chaining_creates_follows_edges():
    store, mem = _make_memory(threshold=0.99)
    r1 = mem.remember("first")
    r2 = mem.remember("second")
    r3 = mem.remember("third")

    # r2 FOLLOWS r1, r3 FOLLOWS r2.
    def follows_target(mem_id):
        edges = [e for e in store.edges_from(mem_id) if e.label == FOLLOWS]
        return edges[0].dst_id if edges else None

    assert follows_target(r1.memory_node.id) is None
    assert follows_target(r2.memory_node.id) == r1.memory_node.id
    assert follows_target(r3.memory_node.id) == r2.memory_node.id


# ----------------------------------------------------------------------- stats
def test_stats_counts():
    store, mem = _make_memory(threshold=0.99)
    sid = mem.start_session()
    mem.remember("m1", entities=["EntA"], session_id=sid)
    mem.remember("m2", entities=["EntB"], session_id=sid, memory_type=MemoryType.FACT)

    stats = mem.stats()
    assert stats["total_memories"] == 2
    assert stats["total_entities"] == 2
    assert stats["total_sessions"] == 1
    assert stats["memory_types"][MemoryType.OBSERVATION.value] == 1
    assert stats["memory_types"][MemoryType.FACT.value] == 1


def test_reflect_returns_text_and_stores_reflection():
    store, mem = _make_memory(threshold=0.99)
    mem.remember("worked on project X", entities=["ProjectX"])
    mem.remember("met with ProjectX team", entities=["ProjectX"])
    reflection = mem.reflect(recent_k=10)
    assert isinstance(reflection, str)
    assert "ProjectX" in reflection
    # A REFLECTION memory was stored.
    reflections = [
        n
        for n in store.nodes_by_label(MEMORY)
        if n.properties["memory_type"] == MemoryType.REFLECTION.value
    ]
    assert len(reflections) == 1
