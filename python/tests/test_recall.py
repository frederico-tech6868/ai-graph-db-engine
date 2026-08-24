"""Tests for the recall engine (ai_memory.recall.RecallEngine)."""

import math

from graphdb import GraphStore
from ai_memory.memory import AgentMemory
from ai_memory.recall import RecallEngine, RecalledMemory
from ai_memory.schema import MEMORY


class MockEmbedder:
    dim = 4

    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def _fallback(self, text):
        h = sum(ord(c) for c in text)
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


def _setup(threshold=0.99, mapping=None):
    store = GraphStore()
    embedder = MockEmbedder(mapping)
    mem = AgentMemory("agent-r", store, embedder, similarity_threshold=threshold)
    engine = RecallEngine(store, embedder)
    return store, mem, engine


# ------------------------------------------------------ build_context_window
def test_build_context_window_format():
    store, mem, engine = _setup()
    mem.remember("hello world")
    recalled = engine.search("hello world", k=5)
    ctx = engine.build_context_window(recalled)
    assert "MEMORY [" in ctx
    assert "hello world" in ctx


def test_build_context_window_empty():
    store, mem, engine = _setup()
    ctx = engine.build_context_window([])
    assert ctx == "(no relevant memories)"


def test_build_context_window_truncates():
    store, mem, engine = _setup()
    # Build many recalled memories with long text.
    recalled = []
    for i in range(50):
        node = list(store.nodes_by_label(MEMORY))  # placeholder to avoid empty
        from graphdb.core import Node

        n = Node(
            label=MEMORY,
            properties={
                "text": "x" * 200,
                "memory_type": "fact",
                "timestamp": float(i),
            },
        )
        recalled.append(RecalledMemory(node=n, score=1.0, context_snippet="x"))

    # Very small budget: only a few lines should fit.
    ctx = engine.build_context_window(recalled, max_tokens=100)  # ~400 chars
    lines = ctx.splitlines()
    assert len(lines) < 50
    assert len(ctx) <= 400 + 250  # roughly within budget (+ one overflow line)


# ------------------------------------------------------------ temporal_window
def test_temporal_window_returns_last_n_in_order():
    store, mem, engine = _setup()
    sid = mem.start_session()
    for i in range(6):
        mem.remember(f"msg {i}", session_id=sid)

    window = engine.temporal_window(sid, last_n=3)
    texts = [n.properties["text"] for n in window]
    assert texts == ["msg 3", "msg 4", "msg 5"]


def test_temporal_window_scoped_to_session():
    store, mem, engine = _setup()
    sid1 = mem.start_session()
    sid2 = mem.start_session()
    mem.remember("s1 a", session_id=sid1)
    mem.remember("s2 a", session_id=sid2)
    mem.remember("s1 b", session_id=sid1)

    window = engine.temporal_window(sid1, last_n=10)
    texts = [n.properties["text"] for n in window]
    assert texts == ["s1 a", "s1 b"]


# --------------------------------------------------------- entity neighborhood
def test_entity_neighborhood_bfs():
    store, mem, engine = _setup()
    r1 = mem.remember("Alice joined TechCorp", entities=["Alice", "TechCorp"])
    r2 = mem.remember("Bob joined TechCorp", entities=["Bob", "TechCorp"])
    mem.remember("unrelated memory", entities=["Zeta"])

    neighborhood = engine.get_entity_neighborhood("TechCorp", depth=2)
    ids = {n.id for n in neighborhood}
    # Both memories relating to TechCorp should be reachable.
    assert r1.memory_node.id in ids
    assert r2.memory_node.id in ids
    # All returned nodes are Memory nodes.
    assert all(n.label == MEMORY for n in neighborhood)


def test_entity_neighborhood_missing_entity():
    store, mem, engine = _setup()
    assert engine.get_entity_neighborhood("DoesNotExist") == []


# --------------------------------------------------------------------- search
def test_search_is_label_scoped():
    # Entity nodes must never appear in Memory search results.
    mapping = {
        "target": [1.0, 0.0, 0.0, 0.0],
        "SomeEntity": [1.0, 0.0, 0.0, 0.0],
    }
    store, mem, engine = _setup(mapping=mapping)
    mem.add_entity("SomeEntity", "concept")
    mem.remember("target")

    results = engine.search("target", label=MEMORY, k=10)
    assert all(r.node.label == MEMORY for r in results)
    assert any(r.node.properties["text"] == "target" for r in results)
