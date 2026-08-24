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
python examples/example_mcp.py             # MCP tools & resources
python examples/example_a2a.py             # A2A interest-routed memory sharing
python examples/example_orchestration.py   # both, via the Orchestrator facade
python examples/example_project_layers.py  # one agent per AI layer, full PIV run
python examples/example_ollama.py          # use local Ollama models for embeddings/chat
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

### `example_project_layers.py`
A full, end-to-end **project-orchestration** program: it turns each layer of the
"AI Layer" workflow (see `aiLayers.md`) into its own project agent and runs a
complete **Plan → Implement → Validate** loop for a sample feature.

- Registers **one agent per AI layer** — Context & Priming, Build the Layer,
  Slicing & Parallelism, PIV: Plan / Implement / Validate, Review, Commit, and
  System Evolution — each with its own skills and interests.
- Each layer does its work through its **MCP tools** (records output with
  `remember_fact`) and publishes it as a **memory of interest** over **A2A**.
- The bus routes each layer's output to the downstream layers that registered
  matching interests, so the pipeline self-assembles from `plan → implement →
  validate → review → commit → evolve` with no hard-wired calls.
- Ends with the emergent orchestration state: per-layer memory ownership, the
  MCP tool/resource surface, `recall`/`reflect` calls, the full A2A message
  feed, and the single shared graph that ties every layer together.

### `example_ollama.py`
Runs the **same** MCP + A2A + orchestration flow but backed by **local Ollama
models** instead of the deterministic offline embedder — real neural embeddings
with no cloud API.

- `OllamaEmbedder` (standard-library only) talks to a local Ollama server and
  auto-discovers the embedding dimensionality.
- Optional `ollama_chat()` helper drives a local chat model (e.g. for richer
  reflections).
- Everything else is unchanged: the whole stack is embedder-agnostic, so you
  swap models by swapping one object.
- If no Ollama server is running, the script explains how to start one and
  falls back to the offline `LocalEmbedder` so it still completes.

#### Using Ollama

```bash
# install Ollama from https://ollama.com, then:
ollama pull nomic-embed-text        # embedding model
ollama pull llama3.2                # optional chat model
# Ollama serves on http://localhost:11434 by default
```

Configuration (all optional):

| Env var | Default | Purpose |
|---------|---------|---------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | embedding model |
| `OLLAMA_CHAT_MODEL` | `llama3.2` | chat model for `ollama_chat()` |
| `USE_OLLAMA` | *(unset)* | if set, `get_embedder()` prefers Ollama |

Wiring it into your own code:

```python
from ai_memory.embedder import OllamaEmbedder, get_embedder
from protocols import Orchestrator

# explicit
orch = Orchestrator(embedder=OllamaEmbedder(model="nomic-embed-text"))

# or auto-select (Ollama -> OpenAI -> LocalEmbedder)
orch = Orchestrator(embedder=get_embedder(prefer_ollama=True))
```

## In the web UI

The same capabilities are available interactively in the **Orchestration** tab
of the web console (`python -m webui.run`, then open the app):

- register agents with skills & interests,
- browse and invoke each agent's MCP tools, read its resources,
- publish a memory and watch the A2A bus route it (with a live message feed),
- see agents, topics and shared-memory edges appear in the graph view.
