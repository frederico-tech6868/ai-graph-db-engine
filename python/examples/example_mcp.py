"""Example: MCP (Agent -> Tools & Data) over graph memory.

Run:  python examples/example_mcp.py

Demonstrates how an agent's memory is exposed as MCP *tools* (callable) and
*resources* (readable data), and how an MCP client discovers and invokes them.
Everything runs fully offline with the deterministic ``LocalEmbedder``.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from ai_memory.embedder import LocalEmbedder
from ai_memory.memory import AgentMemory
from graphdb.store import GraphStore
from protocols import MCPClient, build_memory_mcp_server


def main() -> None:
    # 1. An agent with graph-backed memory.
    store = GraphStore()
    memory = AgentMemory("assistant", store, LocalEmbedder())

    # 2. Expose that memory as an MCP server (tools + resources).
    server = build_memory_mcp_server(memory, name="memory::assistant")

    # 3. A client connects and discovers what's available.
    client = MCPClient()
    client.connect(server)

    print("== MCP tools/list ==")
    for tool in client.list_tools():
        print(f"  - {tool['name']}: {tool['description']}")

    print("\n== MCP resources/list ==")
    for res in client.list_resources():
        print(f"  - {res['uri']}: {res['description']}")

    # 4. Call a tool (tools/call) to store some facts.
    print("\n== tools/call remember_fact ==")
    for text in [
        "Graph databases model data as nodes and edges.",
        "Vector search finds semantically similar nodes.",
        "MCP lets an agent expose tools and data to other agents.",
    ]:
        env = client.call_tool("remember_fact", {"text": text, "memory_type": "fact"})
        print(f"  stored (isError={env['isError']}): {env['structuredContent']}")

    # 5. Call a tool that returns structured content.
    print("\n== tools/call recall_memories ==")
    env = client.call_tool("recall_memories", {"query": "similarity search", "k": 3})
    for hit in env["structuredContent"]:
        print(f"  [{hit['score']}] {hit['text']}")

    # 6. Read a resource (resources/read).
    print("\n== resources/read memory://stats ==")
    env = client.read_resource("memory://stats")
    print(f"  {env['structuredContent']}")

    # 7. Errors are surfaced in the MCP envelope, not raised.
    print("\n== error handling ==")
    env = client.call_tool("does_not_exist", {})
    print(f"  isError={env['isError']}: {env['content'][0]['text']}")

    # 8. The client keeps a call log for observability.
    print(f"\n== call log ({len(client.call_log)} calls) ==")
    for call in client.call_log:
        print(f"  {call.server}::{call.tool} error={call.is_error}")


if __name__ == "__main__":
    main()
