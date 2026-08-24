"""Example: full orchestration combining MCP + A2A.

Run:  python examples/example_orchestration.py

The :class:`Orchestrator` is the single facade the web UI uses. It gives each
agent a graph-backed memory, an MCP server (tools + resources) and an A2A card,
all over one shared graph. This script shows a realistic loop:

  1. an agent uses an MCP *tool* to record what it learned,
  2. it shares the finding over A2A to interested peers,
  3. a peer uses its own MCP *tool* to recall what it received.

Everything runs fully offline with the deterministic ``LocalEmbedder``.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from protocols import Orchestrator


def main() -> None:
    orc = Orchestrator()

    # 1. Register a small team of agents with overlapping interests.
    orc.create_agent("researcher", name="Researcher",
                     skills=["research"], interests=["databases", "machine learning"])
    orc.create_agent("engineer", name="Engineer",
                     skills=["deployment"], interests=["databases", "deployment"])
    orc.create_agent("chef", name="Chef", skills=["cooking"], interests=["cooking"])

    print("== registered agents ==")
    for a in orc.agents():
        print(f"  {a['name']}: interests={a['interests']}")

    # 2. Researcher records a finding through its MCP tool.
    print("\n== researcher uses MCP tool remember_fact ==")
    env = orc.mcp_call("researcher", "remember_fact", {
        "text": "Label-scoped vector indexes cut query latency by 10x.",
        "memory_type": "fact",
    })
    print(f"  isError={env['isError']} -> {env['structuredContent']}")

    # 3. Researcher shares the finding over A2A (interest-routed).
    print("\n== researcher shares it over A2A ==")
    result = orc.a2a_share(
        "researcher",
        "Recommendation: adopt label-scoped vector indexes for faster queries.",
        topics=["databases"],
    )
    for d in result["delivered_to"]:
        print(f"  routed -> {d['agent_id']} ({d['reason']})")

    # 4. Engineer received it; it now shows up in the engineer's inbox...
    print("\n== engineer inbox ==")
    for msg in orc.inbox("engineer"):
        print(f"  from {msg['sender_id']}: \"{msg['content']['text']}\"")

    # 5. ...and because the memory is shared in the graph, the engineer can
    #    recall it through its OWN MCP tool.
    print("\n== engineer uses MCP tool recall_memories ==")
    env = orc.mcp_call("engineer", "recall_memories", {"query": "vector index recommendation", "k": 3})
    for hit in env["structuredContent"]:
        print(f"  [{hit['score']}] {hit['text']}")

    # 6. Read a resource through the orchestrator.
    print("\n== read resource memory://stats (engineer) ==")
    env = orc.read_resource("engineer", "memory://stats")
    print(f"  {env['structuredContent']}")

    # 7. Observability: the A2A message feed and the MCP call log.
    print(f"\n== A2A message feed ({len(orc.messages())}) ==")
    for m in orc.messages():
        print(f"  {m['type']}: {m['sender_id']} -> {m['recipient_id']}")

    print(f"\n== MCP call log ({len(orc.call_log())}) ==")
    for c in orc.call_log():
        print(f"  {c['server']}::{c['tool']} error={c['is_error']}")


if __name__ == "__main__":
    main()
