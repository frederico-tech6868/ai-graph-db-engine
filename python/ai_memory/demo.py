"""Full offline demo of the AI agent memory system.

Run with::

    python ai_memory/demo.py

Requires no API keys -- it uses the deterministic :class:`LocalEmbedder` and a
mock LLM, so it runs anywhere.
"""

from __future__ import annotations

import os
import sys

# Allow running as a plain script (``python ai_memory/demo.py``) by making the
# repository root importable so ``import ai_memory`` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphdb import GraphStore

from ai_memory import AgentMemory, GraphAgent, LocalEmbedder

SAVE_PATH = "/tmp/demo_graph.json"


def _rule(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _clean_memory_lines(context):
    """Return just the human text of each recalled memory line.

    Turns ``MEMORY [ts] (type): <text>`` into ``<text>`` and drops any lines
    that are themselves prior assistant summaries (to avoid echo build-up).
    """
    texts = []
    for line in context.splitlines():
        if "): " in line:
            body = line.split("): ", 1)[1].strip()
        else:
            body = line.strip()
        if not body or body.startswith(("Here's what I remember", "ML frameworks", "Understood")):
            continue
        texts.append(body)
    return texts


def _mock_llm(messages):
    """A tiny deterministic 'LLM' that answers using the injected memory context."""
    system = messages[0]["content"]
    user = messages[-1]["content"].lower()
    # Pull the memory block out of the system prompt.
    context = ""
    if "=== RELEVANT LONG-TERM MEMORIES ===" in system:
        context = system.split("=== RELEVANT LONG-TERM MEMORIES ===", 1)[1]
        context = context.split("=== END MEMORIES ===", 1)[0].strip()
    facts = _clean_memory_lines(context)

    if "techcorp" in user:
        relevant = [f for f in facts if "techcorp" in f.lower()]
        body = " | ".join(relevant) if relevant else "I don't have details about TechCorp yet."
        return f"Here's what I remember about TechCorp: {body}"
    if "framework" in user or " ml " in f" {user} ":
        relevant = [
            f
            for f in facts
            if any(fw in f.lower() for fw in ("pytorch", "tensorflow", "machine learning"))
        ]
        body = " | ".join(relevant) if relevant else "No ML frameworks mentioned yet."
        return f"ML frameworks mentioned so far: {body}"
    return "Understood. I've made a note of that."


def run_demo() -> None:
    _rule("1. Setting up GraphStore + AgentMemory + GraphAgent (LocalEmbedder)")
    store = GraphStore()
    embedder = LocalEmbedder()
    memory = AgentMemory("assistant-1", store, embedder, similarity_threshold=0.9)
    agent = GraphAgent("Aria", memory, llm_fn=_mock_llm)
    session_id = memory.start_session({"channel": "demo"})
    print(f"Created agent 'Aria' with session {session_id[:8]}...")

    conversation = [
        "My name is Alice and I work at TechCorp.",
        "I'm working on a machine learning project using PyTorch.",
        "Alice's colleague Bob also works at TechCorp on NLP.",
        "What do you know about TechCorp?",
        "What ML frameworks have been mentioned?",
    ]

    for turn, user_msg in enumerate(conversation, start=1):
        _rule(f"Turn {turn} — User: {user_msg}")
        # Show what will be recalled *before* this turn stores new memories.
        recalled = memory.recall(user_msg, k=5)
        response = agent.chat(user_msg, session_id=session_id)

        print(f"\nAssistant: {response}")

        print("\nRelevant memories recalled for this turn:")
        if recalled:
            for rm in recalled:
                print(f"  - ({rm.score:.3f}) {rm.context_snippet}")
        else:
            print("  (none yet)")

        # Entities stored in the graph so far.
        entities = [n.properties.get("name") for n in store.nodes_by_label("Entity")]
        print(f"\nEntities known so far: {sorted(e for e in entities if e)}")

    memory.end_session(session_id)

    _rule("2. Reflection over recent memories")
    reflection = memory.reflect(recent_k=20)
    print(reflection)

    _rule("3. Final graph stats")
    stats = agent.memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    _rule("4. Memory summary (all stored memories)")
    for node in sorted(
        store.nodes_by_label("Memory"),
        key=lambda n: n.properties.get("timestamp", 0.0),
    ):
        mtype = node.properties.get("memory_type")
        text = node.properties.get("text", "")
        print(f"  [{mtype}] {text[:80]}")

    _rule("5. Persistence: save + reload")
    agent.save_memory(SAVE_PATH)
    size = os.path.getsize(SAVE_PATH)
    print(f"Saved graph to {SAVE_PATH} ({size} bytes)")

    store2 = GraphStore()
    store2.load(SAVE_PATH)
    memory2 = AgentMemory("assistant-1", store2, embedder, similarity_threshold=0.9)
    print("Reloaded graph stats:")
    for key, value in memory2.stats().items():
        print(f"  {key}: {value}")

    # Verify recall works after reload.
    recalled = memory2.recall("Where does Alice work?", k=3)
    print("\nRecall after reload — 'Where does Alice work?':")
    for rm in recalled:
        print(f"  - ({rm.score:.3f}) {rm.context_snippet}")

    assert memory2.stats()["total_memories"] == memory.stats()["total_memories"], (
        "persistence mismatch!"
    )
    print("\nPersistence verified: memory counts match after reload. ✅")


if __name__ == "__main__":
    run_demo()
