"""Keep a local Ollama model *in context* using the graph engine as memory.

Problem
-------
An Ollama model has a fixed context window (``num_ctx`` -- commonly 64k, up to
256k). A long conversation, or a large knowledge base, will overflow it and the
model "forgets" or errors out.

Solution
--------
Use the graph engine as **unbounded long-term memory** and put a
:class:`ai_memory.context_window.ContextManager` in front of the model. Every
turn it:

* reserves space for the system prompt and the model's reply,
* retrieves only the most relevant memories from the graph (label-scoped vector
  search) and packs them to a memory sub-budget,
* keeps the most recent transcript turns that fit, and **summarises older turns
  back into the graph** so they stay retrievable but leave the window,

so the assembled prompt is *guaranteed* to fit ``num_ctx`` no matter how long
you chat.

Run it (works fully offline; uses Ollama if available)::

    python examples/example_ollama_context.py

To actually generate with Ollama::

    ollama pull llama3.1          # or any chat model
    ollama pull nomic-embed-text  # embeddings for retrieval
    USE_OLLAMA=1 python examples/example_ollama_context.py
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ai_memory.context_window import ContextBudget, ContextManager  # noqa: E402
from ai_memory.embedder import LocalEmbedder, OllamaEmbedder, ollama_chat  # noqa: E402
from ai_memory.memory import AgentMemory  # noqa: E402
from graphdb.store import GraphStore  # noqa: E402


# --------------------------------------------------------------------------
# Pick embedder + a summariser/generator, preferring local Ollama.
# --------------------------------------------------------------------------
def pick_backend():
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        emb = OllamaEmbedder(host=host)
        chat_model = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.1")

        def generate(messages):
            # Flatten chat messages into a single prompt for ollama_chat.
            system = next((m["content"] for m in messages if m["role"] == "system"), "")
            user = messages[-1]["content"]
            return ollama_chat(user, system=system, model=chat_model, host=host)

        def summarize(raw: str) -> str:
            return ollama_chat(
                f"Summarise the following conversation in 2-3 sentences, keeping "
                f"names, decisions and facts:\n\n{raw}",
                system="You are a precise note-taker.",
                model=chat_model,
                host=host,
            )

        print(f"[backend] Ollama at {host} (embeddings + chat)")
        return emb, generate, summarize
    except Exception as exc:
        print(f"[backend] Ollama unavailable ({exc}); running offline.")
        print("          Retrieval + budgeting still work; generation is stubbed.")

        def generate(messages):
            return "(offline stub reply - set USE_OLLAMA=1 with Ollama running to generate)"

        return LocalEmbedder(), generate, None  # None -> extractive summariser


def human(n: int) -> str:
    return f"{n:,}"


def main() -> None:
    print("=" * 74)
    print("  Ollama + graph memory: staying within a 64k / 256k context window")
    print("=" * 74)

    embedder, generate, summarize = pick_backend()

    store = GraphStore()
    memory = AgentMemory(agent_id="assistant", store=store, embedder=embedder)

    # Choose your model's real context window here.
    #   64k  -> context_limit=65_536
    #   128k -> context_limit=131_072
    #   256k -> context_limit=262_144
    budget = ContextBudget(
        context_limit=65_536,     # <-- set to your Ollama num_ctx
        reserve_response=2_048,   # room for the model's answer
        reserve_system=1_024,
        memory_fraction=0.5,      # half the working area for retrieved memory
    )
    ctx = ContextManager(memory, budget=budget, summarizer=summarize)

    print(f"\nModel context window : {human(budget.context_limit)} tokens")
    print(f"  reserved (system)  : {human(budget.reserve_system)}")
    print(f"  reserved (reply)   : {human(budget.reserve_response)}")
    print(f"  working area       : {human(budget.working_tokens())}")
    print(f"    - memories       : {human(budget.memory_tokens())}")
    print(f"    - live transcript: {human(budget.history_tokens())}")

    system_prompt = (
        "You are a helpful engineering assistant with a long-term memory graph. "
        "Use the retrieved memories when relevant."
    )

    # -------------------------------------------------------------------
    # Seed some long-term memory (imagine these were stored days ago).
    # -------------------------------------------------------------------
    for fact in [
        "The project is a graph database engine with label-scoped vector search.",
        "Deployment target is a local Ollama runtime, no cloud APIs.",
        "The user's name is Fred and he prefers Rust for hot paths.",
        "The WebUI runs on port 3000 and persists to webui_graph.json.",
        "A2A routing uses topic overlap first, embedding similarity second.",
    ]:
        memory.remember(fact)

    # -------------------------------------------------------------------
    # Simulate a LONG conversation to prove the window never overflows.
    # -------------------------------------------------------------------
    history: list = []
    turns = [
        "Remind me what database technology this project uses.",
        "What's my name and which language do I prefer for performance-critical code?",
        "Where does the web UI run and how is state persisted?",
        "Summarise how agents decide what memories to share with each other.",
        "Given all of the above, propose a deployment plan for a 256k-context model.",
    ]

    # Pad history with lots of large prior turns to FORCE summarisation/eviction
    # (each turn is deliberately big so the transcript exceeds its sub-budget).
    for i in range(60):
        history.append({"role": "user", "content": f"[old turn {i}] " + ("context filler " * 220)})
        history.append({"role": "assistant", "content": f"[old reply {i}] " + ("noted detail " * 220)})

    print("\n" + "-" * 74)
    print(f"Starting with {len(history)} padded prior turns to force eviction.")
    print("-" * 74)

    max_seen = 0
    for user_msg in turns:
        assembled, history = ctx.assemble(user_msg, system_prompt, history)
        b = assembled.breakdown
        max_seen = max(max_seen, assembled.total_tokens + b["reserved_for_response"])

        print(f"\nUSER: {user_msg}")
        print(
            f"  budget -> sys={b['system']} mem={b['memories']} "
            f"hist={b['history']} user={b['user']} "
            f"| total+reply={assembled.total_tokens + b['reserved_for_response']} "
            f"/ {b['context_limit']}  within_limit={assembled.within_limit}"
        )
        print(f"  kept {len(assembled.history)} live turns; "
              f"memories retrieved into prompt: "
              f"{'yes' if assembled.memory_block else 'no'}")

        # Generate the reply from the assembled (budget-safe) messages.
        reply = generate(assembled.to_messages())
        print(f"  ASSISTANT: {reply[:160]}{'...' if len(reply) > 160 else ''}")

        # Persist the exchange and continue the live transcript.
        ctx.ingest_turn(user_msg, reply)
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": reply})

    print("\n" + "=" * 74)
    print("RESULT")
    print("=" * 74)
    print(f"  Peak prompt+reply tokens observed : {human(max_seen)}")
    print(f"  Hard context limit                : {human(budget.context_limit)}")
    print(f"  Overflowed the window?            : "
          f"{'YES (bug!)' if max_seen > budget.context_limit else 'NO - stayed in context'}")
    print(f"  Total memories now in graph       : {store_node_count(store)}")
    print("\nOlder turns were summarised back into the graph and remain")
    print("retrievable via vector search - nothing was lost, nothing overflowed.")


def store_node_count(store) -> int:
    return len(store.all_nodes())


if __name__ == "__main__":
    main()
