"""Using **local Ollama models** with the graph memory + orchestration layer.

This example shows how to swap the deterministic ``LocalEmbedder`` for real
neural embeddings served by a **local Ollama** instance -- and, optionally, how
to use a local Ollama chat model to write agent reflections. Nothing here talks
to a cloud API.

Prerequisites (one-time, on the machine running this script)::

    # 1. install Ollama from https://ollama.com
    # 2. pull an embedding model (and optionally a chat model)
    ollama pull nomic-embed-text
    ollama pull llama3.2
    # 3. Ollama serves on http://localhost:11434 by default

Configuration (all optional, sensible defaults shown)::

    export OLLAMA_HOST=http://localhost:11434
    export OLLAMA_EMBED_MODEL=nomic-embed-text
    export OLLAMA_CHAT_MODEL=llama3.2

Run it::

    python examples/example_ollama.py

If no Ollama server is reachable, the script explains how to start one and
falls back to the offline ``LocalEmbedder`` so it still runs end-to-end.
"""

from __future__ import annotations

import os
import sys

# --- make the repo importable when run as a plain script from any cwd --------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ai_memory.embedder import (  # noqa: E402
    LocalEmbedder,
    OllamaEmbedder,
    ollama_chat,
)
from ai_memory.schema import MemoryType  # noqa: E402
from graphdb.store import GraphStore  # noqa: E402
from protocols import Orchestrator  # noqa: E402


def pick_embedder():
    """Return an OllamaEmbedder if a local server is reachable, else LocalEmbedder."""
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    try:
        emb = OllamaEmbedder(model=model, host=host)
        print(f"[ollama] connected to {host}")
        print(f"[ollama] embedding model '{model}', dim={emb.dim}")
        return emb, True
    except RuntimeError as exc:
        print(f"[ollama] not available: {exc}")
        print("[ollama] falling back to offline LocalEmbedder.")
        print("         To use Ollama:  ollama serve  &&  ollama pull nomic-embed-text")
        return LocalEmbedder(), False


def main() -> None:
    print("=" * 70)
    print("  Local Ollama models with graph memory + orchestration")
    print("=" * 70)

    embedder, using_ollama = pick_embedder()

    # The Orchestrator (and everything under it) simply uses whatever embedder
    # you hand it -- the rest of the system is embedder-agnostic.
    orch = Orchestrator(store=GraphStore(), embedder=embedder, interest_threshold=0.35)

    orch.create_agent(
        "researcher",
        name="Researcher",
        description="Gathers findings.",
        skills=["search"],
        interests=["databases", "vectors"],
    )
    orch.create_agent(
        "engineer",
        name="Engineer",
        description="Builds features.",
        skills=["code"],
        interests=["databases", "performance"],
    )

    # 1. Record a fact through the researcher's MCP tool (embedded via Ollama).
    print("\n1) MCP remember_fact (embedded with the active model)")
    call = orch.mcp_call(
        "researcher",
        "remember_fact",
        {
            "text": "Graph databases with label-scoped vector search avoid cross-type false matches.",
            "memory_type": MemoryType.FACT.value,
        },
    )
    print("   ->", call["content"][0]["text"])

    # 2. Share it over A2A; routing uses the same embeddings for similarity.
    print("\n2) A2A share -> interest-routed to peers")
    result = orch.a2a_share(
        "researcher",
        "Benchmark: label-scoped ANN search cut recall errors ~40% on our graph.",
        topics=["databases", "vectors"],
    )
    print("   delivered to:", [d["agent_id"] for d in result["delivered_to"]])

    # 3. Recall with the engineer's MCP tool (vector search over Ollama vectors).
    print("\n3) MCP recall_memories on the engineer")
    recall = orch.mcp_call("engineer", "recall_memories", {"query": "vector search", "k": 3})
    print("   ->", recall["content"][0]["text"])

    # 4. Optional: use a local Ollama *chat* model for a richer reflection.
    print("\n4) Local Ollama chat model (optional)")
    if using_ollama:
        try:
            summary = ollama_chat(
                "In one sentence, summarise why label-scoped vector search helps a graph DB.",
                system="You are a concise engineering assistant.",
            )
            print("   ollama_chat ->", summary.strip() or "(empty response)")
        except Exception as exc:
            print(f"   ollama_chat unavailable ({exc}); "
                  "pull a chat model with: ollama pull llama3.2")
    else:
        print("   skipped (no Ollama server). Try: ollama pull llama3.2")

    print("\nDone. The whole stack ran on", "Ollama" if using_ollama else "the offline fallback",
          "with zero code changes beyond the embedder choice.")


if __name__ == "__main__":
    main()
