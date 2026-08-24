# Examples — MCP & A2A protocols

Runnable, fully-offline examples for the agent-interoperability layer built on
top of the graph database engine. No API key is required — everything uses the
deterministic `LocalEmbedder`.

## The two protocols

| Protocol | Meaning | What it does here |
|----------|---------|-------------------|
| **MCP** — *Model Context Protocol* | Agent → **Tools & Data** | Exposes an agent's graph memory as callable **tools** (`remember_fact`, `recall_memories`, `search_nodes`, `get_entity_info`, `reflect`) and readable **resources** (`memory://stats`, `memory://entities`, `memory://recent`). |
| **A2A** — *Agent-to-Agent* | Agent → **Agent** | Agents advertise **interests** and share **memories of interest**; the bus routes each shared memory to peers by topic overlap or embedding similarity, recording provenance in the shared graph. |

Both live in `protocols/` and share a single `GraphStore`, so all activity is
visible in the web UI and searchable with the label-scoped vector engine.

## Running

From the repository root:

```bash
python examples/example_mcp.py            # MCP tools & resources
python examples/example_a2a.py            # A2A interest-routed memory sharing
python examples/example_orchestration.py  # both, via the Orchestrator facade
```

Each script is self-contained and prints a narrated walkthrough.

## What each example shows

### `example_mcp.py`
- Build an MCP server from an `AgentMemory` (`build_memory_mcp_server`).
- Discover tools/resources (`tools/list`, `resources/list`).
- Invoke tools (`tools/call`) and read resources (`resources/read`).
- MCP-shaped result envelopes and error handling.
- The client-side call log for observability.

### `example_a2a.py`
- Give three agents different interests via `AgentCard`.
- Preview routing before publishing (`interested_agents`).
- Publish a memory and watch it route only to interested peers.
- Show that unrelated memories are *not* delivered.
- Inspect the provenance edges written into the shared graph
  (`PUBLISHED`, `TAGGED`, `SHARED_WITH`).

### `example_orchestration.py`
- Use the `Orchestrator` facade (the same one the web UI uses).
- A realistic loop: an agent records a finding with an MCP **tool**, shares it
  over **A2A**, and a peer recalls it through its *own* MCP tool.
- Read resources and inspect the A2A message feed + MCP call log.

## In the web UI

The same capabilities are available interactively in the **Orchestration** tab
of the web console (`python -m webui.run`, then open the app):

- register agents with skills & interests,
- browse and invoke each agent's MCP tools, read its resources,
- publish a memory and watch the A2A bus route it (with a live message feed),
- see agents, topics and shared-memory edges appear in the graph view.
