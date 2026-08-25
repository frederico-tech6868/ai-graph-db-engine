"""Tests for the token-budgeted ContextManager (Ollama context safety)."""

from __future__ import annotations

from ai_memory.context_window import ContextBudget, ContextManager
from ai_memory.embedder import LocalEmbedder
from ai_memory.memory import AgentMemory
from ai_memory.schema import MemoryType
from graphdb.store import GraphStore


def _mgr(context_limit=4096, memory_fraction=0.5):
    store = GraphStore()
    memory = AgentMemory("t", store=store, embedder=LocalEmbedder())
    budget = ContextBudget(
        context_limit=context_limit,
        reserve_response=256,
        reserve_system=128,
        memory_fraction=memory_fraction,
    )
    return ContextManager(memory, budget=budget), memory, store


def test_budget_math():
    b = ContextBudget(context_limit=1000, reserve_response=100, reserve_system=50,
                      memory_fraction=0.5)
    assert b.working_tokens() == 850
    assert b.memory_tokens() == 425
    assert b.history_tokens() == 425


def test_never_exceeds_limit_under_huge_history():
    mgr, memory, _ = _mgr(context_limit=2048)
    for i in range(20):
        memory.remember(f"seed fact number {i} about databases and vectors")
    history = []
    for i in range(200):
        history.append({"role": "user", "content": f"turn {i} " + ("filler " * 50)})
        history.append({"role": "assistant", "content": f"reply {i} " + ("words " * 50)})

    for msg in ["what about databases?", "and vectors?", "recap please"]:
        assembled, history = mgr.assemble(msg, "system", history)
        total_with_reply = assembled.total_tokens + assembled.breakdown["reserved_for_response"]
        assert total_with_reply <= mgr.budget.context_limit
        assert assembled.within_limit


def test_overflow_turns_summarised_into_graph():
    mgr, memory, store = _mgr(context_limit=1024)
    before = len(store.all_nodes())
    history = [
        {"role": "user", "content": "big turn " + ("x " * 400)}
        for _ in range(10)
    ]
    mgr.assemble("hi", "system", history)
    after = len(store.all_nodes())
    # At least one summary memory node should have been written back.
    assert after > before
    reflections = [
        n for n in store.nodes_by_label("Memory")
        if n.properties.get("memory_type") == str(MemoryType.REFLECTION)
    ]
    assert reflections


def test_relevant_memories_are_retrieved_into_prompt():
    mgr, memory, _ = _mgr(context_limit=8192)
    memory.remember("The capital of the project is label-scoped vector search.")
    memory.remember("Deployment uses a local Ollama runtime.")
    assembled, _ = mgr.assemble("Tell me about vector search", "system", [])
    assert assembled.memory_block  # non-empty
    assert "vector search" in assembled.memory_block.lower()


def test_custom_token_counter_is_used():
    store = GraphStore()
    memory = AgentMemory("t", store=store, embedder=LocalEmbedder())
    # Counter that counts words instead of chars.
    mgr = ContextManager(
        memory,
        budget=ContextBudget(context_limit=1000, reserve_response=50, reserve_system=50),
        token_counter=lambda s: len(s.split()),
    )
    assembled, _ = mgr.assemble("one two three", "sys words here", [])
    assert assembled.breakdown["user"] == 3
    assert assembled.breakdown["system"] == 3


def test_to_messages_folds_memory_into_system():
    mgr, memory, _ = _mgr(context_limit=8192)
    memory.remember("Fred prefers Rust for hot paths.")
    assembled, _ = mgr.assemble("what language do I prefer?", "You are helpful.", [])
    msgs = assembled.to_messages()
    assert msgs[0]["role"] == "system"
    assert "long-term memory" in msgs[0]["content"].lower()
    assert msgs[-1]["role"] == "user"
