"""Example: A2A (Agent -> Agent) shared memories of interest.

Run:  python examples/example_a2a.py

Three agents declare different interests. When one publishes a memory, the A2A
bus routes it only to peers whose interests match — by explicit topic overlap or
by embedding similarity — and records the provenance in the shared graph.
Everything runs fully offline with the deterministic ``LocalEmbedder``.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from ai_memory.embedder import LocalEmbedder
from ai_memory.memory import AgentMemory
from graphdb.store import GraphStore
from protocols import A2AAgent, A2ABus, AgentCard
from protocols.schema import PUBLISHED, SHARED_WITH, TAGGED


def make_agent(store, embedder, agent_id, name, interests):
    return A2AAgent(
        memory=AgentMemory(agent_id, store, embedder),
        card=AgentCard(agent_id=agent_id, name=name, interests=interests),
    )


def main() -> None:
    # All agents share ONE graph store, so shared memories live in one place.
    store = GraphStore()
    embedder = LocalEmbedder()
    bus = A2ABus(embedder)

    researcher = make_agent(store, embedder, "researcher", "Researcher",
                            ["databases", "machine learning"])
    engineer = make_agent(store, embedder, "engineer", "Engineer",
                          ["databases", "deployment"])
    chef = make_agent(store, embedder, "chef", "Chef", ["cooking"])
    for agent in (researcher, engineer, chef):
        bus.register(agent)

    print("== agents & interests ==")
    for agent in bus.agents():
        print(f"  {agent.card.name}: {agent.card.interests}")

    # 1. Preview routing before publishing (no side effects).
    print("\n== preview: who is interested in 'databases'? ==")
    for hit in bus.interested_agents(["databases"], "graph database indexing"):
        print(f"  {hit['agent_id']} via {hit['reason']} ({hit['score']})")

    # 2. Researcher shares a memory tagged 'databases'.
    print("\n== researcher shares a 'databases' memory ==")
    result = bus.share_memory(
        "researcher",
        "A new graph database index accelerates vector similarity search.",
        topics=["databases"],
    )
    for d in result["delivered_to"]:
        print(f"  -> {d['agent_id']} ({d['reason']}, {d['score']})")

    # 3. Chef shares a cooking memory — researcher/engineer are NOT interested.
    print("\n== chef shares a 'cooking' memory ==")
    result = bus.share_memory(
        "chef", "Slow-roasting tomatoes deepens their umami flavour.", topics=["cooking"]
    )
    print(f"  delivered_to: {result['delivered_to'] or 'nobody'}")

    # 4. Inspect each agent's inbox.
    print("\n== inboxes ==")
    for agent in bus.agents():
        for msg in agent.inbox:
            c = msg.content
            print(f"  {agent.card.name} <- {msg.sender_id}: \"{c['text'][:50]}...\""
                  f" ({c['match_reason']})")

    # 5. The shared memory carries provenance edges in the graph.
    print("\n== provenance of the shared memory ==")
    mem_id = result["memory_id"]  # last shared (chef's); show researcher's instead
    # find the databases memory (published by researcher)
    from ai_memory.schema import MEMORY
    db_mem = next(
        n for n in store.nodes_by_label(MEMORY)
        if "graph database index" in n.properties.get("text", "")
    )
    tagged = [e for e in store.edges_from(db_mem.id) if e.label == TAGGED]
    shared = [e for e in store.edges_from(db_mem.id) if e.label == SHARED_WITH]
    published = [e for e in store.edges_to(db_mem.id) if e.label == PUBLISHED]
    print(f"  PUBLISHED by:   {len(published)} agent")
    print(f"  TAGGED topics:  {len(tagged)}")
    print(f"  SHARED_WITH:    {len(shared)} recipient(s)")


if __name__ == "__main__":
    main()
