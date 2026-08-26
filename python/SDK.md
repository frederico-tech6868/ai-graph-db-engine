# AI-GraphDB-Engine SDK Documentation

**Version:** 1.0  
**For:** LLMs, AI agents, and developers building graph-based memory systems

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [Quick Start](#quick-start)
5. [API Reference](#api-reference)
   - [GraphStore](#graphstore)
   - [AgentMemory](#agentmemory)
   - [Embedders](#embedders)
   - [MCP Protocol](#mcp-protocol)
   - [A2A Protocol](#a2a-protocol)
   - [Orchestrator](#orchestrator)
   - [ContextManager](#contextmanager)
6. [Usage Patterns](#usage-patterns)
7. [Examples](#examples)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The **AI-GraphDB-Engine** is a graph-based long-term memory system designed for AI agents and LLMs. It provides:

- **Unbounded memory** — store unlimited facts, conversations, and observations in a persistent graph
- **Label-scoped vector search** — retrieve only memories of the right *type* (avoids cross-type false matches)
- **Agent interoperability** — MCP (Agent→Tools&Data) and A2A (Agent→Agent) protocols
- **Context management** — stay within 16k–256k LLM context windows via smart retrieval + rolling summarization
- **Offline-first** — works with local Ollama models or cloud APIs; fully functional without internet

### What Problem Does It Solve?

LLMs have a **fixed context window** (16k–256k tokens). Long conversations or large knowledge bases **overflow** and cause:
- Lost context ("the model forgot")
- Errors (context limit exceeded)
- Expensive re-processing

This engine solves it by acting as **external long-term memory**: the graph holds *everything*, and only the most relevant memories are retrieved into the prompt each turn.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  LLM / Agent (16k-256k context window)                  │
└───────────────────┬─────────────────────────────────────┘
                    │ assembled prompt (guaranteed ≤ limit)
                    ↓
┌─────────────────────────────────────────────────────────┐
│  ContextManager (budgeting + retrieval + summarization) │
└───────────────────┬─────────────────────────────────────┘
                    │ label-scoped vector search
                    ↓
┌─────────────────────────────────────────────────────────┐
│  AgentMemory (high-level memory interface)              │
└───────────────────┬─────────────────────────────────────┘
                    │
┌───────────────────┴─────────────────────────────────────┐
│  GraphStore (core property graph + embeddings)          │
│  • Nodes (facts, entities, sessions, agents)            │
│  • Edges (relationships, temporal chains)               │
│  • Persistent (JSON on disk)                            │
└─────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.8+
- Optional: [Ollama](https://ollama.com) for local models

### Install

```bash
# Clone the repository
git clone https://github.com/frederico-tech6868/ai-graph-db-engine.git
cd ai-graph-db-engine/python

# Install in editable mode
pip install -e .

# Optional: install the faster Rust backend
cd ../rust
./build.sh   # requires Rust toolchain; falls back to Python if not built
```

### Verify

```python
from graphdb.store import GraphStore
from ai_memory.memory import AgentMemory
from ai_memory.embedder import LocalEmbedder

store = GraphStore()
memory = AgentMemory("test-agent", store=store, embedder=LocalEmbedder())
print("✅ AI-GraphDB-Engine is ready")
```

---

## Core Concepts

### 1. **Nodes**

A node is an entity in the graph. Every node has:
- `id` (UUID, auto-generated)
- `label` (e.g., `"Memory"`, `"Entity"`, `"Agent"`)
- `properties` (dict, e.g., `{"text": "...", "memory_type": "fact"}`)
- `embedding` (optional vector for similarity search)
- `created_at` (timestamp)

**Node labels in this system:**
- `Memory` — a stored fact, observation, plan, reflection, or emotion
- `Entity` — a named entity (person, place, concept)
- `Session` — a conversation session
- `Agent` — an AI agent node
- `TOOL`, `RESOURCE`, `TOPIC` — MCP/A2A protocol nodes

### 2. **Edges**

An edge is a directed relationship between two nodes:
- `id` (UUID)
- `src_id` → `dst_id`
- `label` (e.g., `"RELATES_TO"`, `"REMEMBERS"`)
- `weight` (float, default 1.0)
- `properties` (dict)

**Edge labels in this system:**
- `RELATES_TO` — Memory → Entity
- `OCCURRED_IN` — Memory → Session
- `FOLLOWS` — Memory → Memory (temporal chain)
- `SIMILAR_TO` — Memory → Memory (semantic similarity)
- `KNOWS` — Agent → Entity
- `REMEMBERS` — Agent → Memory
- `EXPOSES`, `PROVIDES`, `INTERESTED_IN`, `PUBLISHED`, `SHARED_WITH`, `TAGGED` — MCP/A2A protocol edges

### 3. **Label-Scoped Vector Search**

The engine indexes embeddings **per label**. When you search for similar memories, it only compares against nodes with the **same label**, avoiding cross-type pollution (e.g., a query about "databases" won't match an Entity named "Database Inc.").

### 4. **Memory Types**

Memories are categorized by `memory_type` property:
- `observation` — raw input (user said X, assistant replied Y)
- `fact` — extracted knowledge
- `reflection` — higher-level summary
- `plan` — future intent
- `emotion` — affective state

### 5. **Persistence**

The graph is saved to disk as JSON. Load/save is explicit:

```python
store = GraphStore(path="my_graph.json")  # auto-loads if exists
# ... work ...
store.save()  # persist to disk
```

---

## Quick Start

### Minimal Working Example

```python
from graphdb.store import GraphStore
from ai_memory.memory import AgentMemory
from ai_memory.embedder import LocalEmbedder

# 1. Create a graph store (in-memory or persistent)
store = GraphStore()
embedder = LocalEmbedder()  # offline, deterministic embedder

# 2. Create an agent memory
memory = AgentMemory(agent_id="my-agent", store=store, embedder=embedder)

# 3. Store some facts
memory.remember("The project uses a graph database with label-scoped vector search.")
memory.remember("Fred prefers Rust for performance-critical code.")
memory.remember("Deployment target is local Ollama, no cloud dependencies.")

# 4. Retrieve relevant memories
results = memory.recall("What database technology does the project use?", k=3)
for r in results:
    print(f"[{r.score:.3f}] {r.node.properties['text']}")

# 5. Persist to disk
store.save("agent_memory.json")
```

**Output:**
```
[0.712] The project uses a graph database with label-scoped vector search.
[0.301] Deployment target is local Ollama, no cloud dependencies.
[0.098] Fred prefers Rust for performance-critical code.
```

---

## API Reference

### GraphStore

**Path:** `graphdb.store.GraphStore`

The core property graph engine. Manages nodes, edges, indexes, and persistence.

#### Constructor

```python
GraphStore(path: Optional[str] = None)
```
- `path`: If provided, auto-loads from disk on construction.

#### Methods

##### Node Operations

```python
add_node(node: Node) -> Node
```
Add a node. Returns the same node (or existing if duplicate ID).

```python
get_node(node_id: str) -> Optional[Node]
```
Retrieve a node by ID. Returns `None` if not found.

```python
delete_node(node_id: str) -> None
```
Delete a node and **all connected edges** (cascade).

```python
nodes_by_label(label: str) -> List[Node]
```
Get all nodes with a specific label.

```python
all_nodes() -> List[Node]
```
Get all nodes in the graph.

##### Edge Operations

```python
add_edge(edge: Edge) -> Edge
```
Add an edge. Both `src_id` and `dst_id` must exist.

```python
get_edge(edge_id: str) -> Optional[Edge]
```
Retrieve an edge by ID.

```python
delete_edge(edge_id: str) -> None
```
Delete an edge.

```python
edges_from(node_id: str) -> List[Edge]
```
Get all outgoing edges from a node.

```python
edges_to(node_id: str) -> List[Edge]
```
Get all incoming edges to a node.

```python
edges_between(src_id: str, dst_id: str) -> List[Edge]
```
Get all edges from `src_id` to `dst_id`.

```python
all_edges() -> List[Edge]
```
Get all edges in the graph.

##### Search

```python
search_similar_nodes(
    query_vec: List[float],
    label: Optional[str] = None,
    k: int = 5
) -> List[Tuple[Node, float]]
```
Vector similarity search. If `label` is provided, search is **scoped** to that label.

Returns `[(node, score), ...]` sorted by score descending.

##### Persistence

```python
save(path: Optional[str] = None) -> None
```
Save the graph to disk (JSON).

```python
load(path: Optional[str] = None) -> None
```
Load the graph from disk.

##### Stats

```python
stats() -> Dict[str, Any]
```
Returns node count, edge count, and label distribution.

---

### AgentMemory

**Path:** `ai_memory.memory.AgentMemory`

High-level memory interface for a single agent. Wraps a `GraphStore` and provides semantic operations.

#### Constructor

```python
AgentMemory(
    agent_id: str,
    store: GraphStore,
    embedder: Embedder,
    session_id: Optional[str] = None
)
```
- `agent_id`: Unique identifier for this agent
- `store`: The underlying graph store (shared across agents)
- `embedder`: Text→vector embedder
- `session_id`: Optional session scope

#### Methods

##### Store Memories

```python
remember(
    text: str,
    memory_type: MemoryType = MemoryType.OBSERVATION,
    entities: Optional[List[str]] = None,
    session_id: Optional[str] = None
) -> RememberResult
```
Store a memory. Automatically:
- Embeds the text
- Checks for duplicates (cosine similarity > 0.95)
- Links to entities (creates them if needed)
- Chains to previous memories in session

**Returns:** `RememberResult(memory_node, similar_existing, was_duplicate)`

**Memory types:**
```python
from ai_memory.schema import MemoryType

MemoryType.OBSERVATION   # raw input
MemoryType.FACT          # extracted knowledge
MemoryType.REFLECTION    # higher-level summary
MemoryType.PLAN          # future intent
MemoryType.EMOTION       # affective state
```

##### Retrieve Memories

```python
recall(
    query: str,
    k: int = 5,
    memory_type: Optional[MemoryType] = None,
    session_id: Optional[str] = None
) -> List[RecalledMemory]
```
Label-scoped vector search over `Memory` nodes. Returns the `k` most relevant.

**Returns:** `[RecalledMemory(node, score, context_snippet), ...]`

```python
get_context(query: str, k: int = 10) -> str
```
Formatted context string for LLM prompts (calls `recall` + formats).

##### Reflection

```python
reflect(recent_k: int = 20) -> str
```
Synthesize a higher-level reflection from recent memories. Stores the reflection as a `REFLECTION` memory. Works offline (no LLM needed).

##### Entities

```python
add_entity(
    name: str,
    entity_type: str = "concept",
    properties: Optional[Dict] = None
) -> Node
```
Create or retrieve an entity node.

```python
get_entity(name: str) -> Optional[Node]
```
Get an entity by name.

##### Stats

```python
stats() -> Dict[str, Any]
```
Returns counts by memory type, total entities, sessions, edges.

---

### Embedders

**Path:** `ai_memory.embedder`

Text → vector transformations.

#### LocalEmbedder (default)

```python
from ai_memory.embedder import LocalEmbedder

embedder = LocalEmbedder(dim=128)
vec = embedder.embed("hello world")  # List[float] of length 128
```
- Deterministic, offline, no dependencies
- Hash-based character n-grams + tf-idf weighting
- Good for testing; weaker semantic understanding than neural models

#### OllamaEmbedder

```python
from ai_memory.embedder import OllamaEmbedder

embedder = OllamaEmbedder(
    model="nomic-embed-text",
    host="http://localhost:11434"
)
vec = embedder.embed("hello world")  # neural embedding
```
- Requires Ollama running locally: `ollama pull nomic-embed-text`
- Standard library only (uses `urllib`)
- Auto-discovers vector dimensionality on construction

**Env vars:**
- `OLLAMA_HOST` — default `http://localhost:11434`
- `OLLAMA_EMBED_MODEL` — default `nomic-embed-text`

#### OpenAIEmbedder

```python
from ai_memory.embedder import OpenAIEmbedder

embedder = OpenAIEmbedder(model="text-embedding-3-small", api_key="sk-...")
```
- Requires `openai` package + API key
- Best semantic quality, but needs internet + costs money

#### Auto-Select

```python
from ai_memory.embedder import get_embedder

# Order: Ollama → OpenAI → LocalEmbedder
embedder = get_embedder(prefer_ollama=True)

# Or via env var
# USE_OLLAMA=1 python script.py
embedder = get_embedder()  # checks USE_OLLAMA
```

#### ollama_chat Helper

```python
from ai_memory.embedder import ollama_chat

reply = ollama_chat(
    "Summarize the conversation",
    system="You are a note-taker.",
    model="llama3.1",
    host="http://localhost:11434"
)
```
Standard-library one-shot chat with local Ollama models. Useful for summarization.

---

### MCP Protocol

**Path:** `protocols.mcp`

**MCP** = Model Context Protocol (Agent → Tools & Data). Exposes an agent's memory as callable tools and readable resources.

#### Build an MCP Server

```python
from ai_memory.memory import AgentMemory
from protocols.mcp import build_memory_mcp_server

server = build_memory_mcp_server(memory, name="agent-mcp")
```

**Tools exposed:**
- `remember_fact(text, memory_type?, entities?)` → stores a memory
- `recall_memories(query, k?)` → retrieves relevant memories
- `search_nodes(text, label?, k?)` → vector search across any label
- `get_entity_info(name)` → entity + related memories
- `reflect(recent_k?)` → synthesize a reflection

**Resources exposed:**
- `memory://stats` — memory counts by type
- `memory://entities` — list of known entities
- `memory://recent` — most recent memories

#### MCP Client

```python
from protocols.mcp import MCPClient

client = MCPClient()
client.connect(server)

# Call a tool
result = client.call_tool("remember_fact", {"text": "Paris is the capital of France."})
print(result)  # MCP envelope: {"content": [...], "isError": False}

# Read a resource
stats = client.read_resource("memory://stats")
print(stats)
```

#### MCPTool / MCPResource

Define custom tools/resources:

```python
from protocols.mcp import MCPServer, MCPTool, MCPResource

server = MCPServer(name="custom")

def my_tool_handler(args):
    return {"result": f"You said: {args['input']}"}

server.register_tool(
    name="echo",
    description="Echoes input",
    input_schema={"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]},
    handler=my_tool_handler
)

def my_resource_reader():
    return {"status": "online"}

server.register_resource(
    uri="custom://status",
    name="System Status",
    mime_type="application/json",
    reader=my_resource_reader
)
```

---

### A2A Protocol

**Path:** `protocols.a2a`

**A2A** = Agent-to-Agent. Interest-based memory routing. Agents advertise **interests** (topics), share memories, and the bus routes them to peers whose interests match.

#### Agent Card

```python
from protocols.a2a import AgentCard, A2AAgent

card = AgentCard(
    agent_id="researcher",
    name="Research Agent",
    description="Gathers findings.",
    skills=["search", "summarize"],
    interests=["databases", "vector search", "AI"]
)

agent = A2AAgent(memory=memory, card=card)
```

#### A2A Bus

```python
from protocols.a2a import A2ABus

bus = A2ABus(embedder=embedder, interest_threshold=0.35)
bus.register(agent)
```

#### Share Memory

```python
result = bus.share_memory(
    sender_id="researcher",
    text="Label-scoped ANN search cut recall errors ~40%.",
    topics=["databases", "vector search"],
    memory_type=MemoryType.FACT
)

print(result["delivered_to"])  # [{"agent_id": "engineer", "reason": "topic:databases", "score": 1.0}]
```

**Routing logic:**
1. **Topic overlap** — if any topic in `topics` matches any interest → deliver (score 1.0)
2. **Embedding similarity** — cosine(memory_vec, interest_vec) ≥ `interest_threshold` → deliver

Writes graph edges:
- `sender PUBLISHED memory_node`
- `memory_node TAGGED topic_node`
- `memory_node SHARED_WITH recipient`
- `recipient REMEMBERS memory_node`

#### Preview Interest

```python
preview = bus.interested_agents(
    topics=["databases", "performance"],
    text="Benchmark results"
)
# [{"agent_id": "...", "reason": "...", "score": ...}, ...]
```

#### Direct Send

```python
msg = bus.send(
    sender_id="agent-a",
    recipient_id="agent-b",
    content={"note": "Check this out"},
    type="text"
)
agent_b.receive(msg)  # adds to inbox
```

#### Inbox

```python
messages = agent.inbox  # List[A2AMessage]
```

---

### Orchestrator

**Path:** `protocols.orchestrator.Orchestrator`

Unified façade over a **shared** graph store + embedder. Manages multiple agents, their MCP servers, and the A2A bus.

#### Constructor

```python
from protocols import Orchestrator

orch = Orchestrator(
    store=GraphStore(),
    embedder=LocalEmbedder(),
    interest_threshold=0.35
)
```

#### Create Agents

```python
agent = orch.create_agent(
    agent_id="researcher",
    name="Research Agent",
    description="Gathers findings.",
    skills=["search"],
    interests=["databases", "AI"]
)
```

This creates:
- An `AgentMemory` with its own AGENT node
- An MCP server exposing that memory
- A2A registration

#### List Agents

```python
agents = orch.agents()
# [{"agent_id": "...", "name": "...", "inbox": 0, "owned_memories": 5, "memory": {...}}, ...]
```

#### MCP Operations

```python
# List tools
tools = orch.tools("researcher")

# Call a tool
result = orch.mcp_call("researcher", "remember_fact", {"text": "Rust is fast."})

# List resources
resources = orch.resources("researcher")

# Read a resource
data = orch.read_resource("researcher", "memory://stats")

# Call log
log = orch.call_log()
```

#### A2A Operations

```python
# Share memory
result = orch.a2a_share(
    sender_id="researcher",
    text="Found a new optimization.",
    topics=["performance"]
)

# Direct send
msg = orch.a2a_send("agent-a", "agent-b", {"content": "hello"})

# Preview routing
preview = orch.preview_interest(["AI"], "Large language models")

# Message feed
feed = orch.messages(limit=50)

# Agent inbox
inbox = orch.inbox("researcher")
```

---

### ContextManager

**Path:** `ai_memory.context_window.ContextManager`

Keeps a local LLM **within its context window** (16k–256k) by budgeting the prompt, retrieving only relevant memories, and rolling old turns into the graph.

#### Problem

Ollama models have a fixed `num_ctx` (16k–256k tokens). Long conversations overflow → the model "forgets" or errors.

#### Solution

The `ContextManager` sits between your model and the graph:
1. **Budgets** — reserves space for system prompt + model's reply, splits the rest between memories and live transcript
2. **Retrieves** — only the top memories relevant to the current message (label-scoped vector search)
3. **Rolls up overflow** — when the transcript exceeds its sub-budget, oldest turns are summarized and written back into the graph as memories, then dropped from the prompt

**Guarantee:** assembled prompt ≤ `num_ctx`, always.

#### Constructor

```python
from ai_memory.context_window import ContextBudget, ContextManager

budget = ContextBudget(
    context_limit=65_536,       # 64k; use 262_144 for 256k
    reserve_response=2_048,     # tokens for model's reply
    reserve_system=1_024,       # tokens for system prompt
    memory_fraction=0.5,        # 50% of working area → memories, 50% → transcript
    chars_per_token=4.0         # heuristic for default token counter
)

ctx = ContextManager(
    memory=memory,
    budget=budget,
    token_counter=None,         # optional: pass your own tokenizer
    summarizer=None             # optional: pass ollama_chat callback for better summaries
)
```

#### Usage Loop

```python
system_prompt = "You are a helpful assistant with long-term memory."
history = []  # List[Dict[str, str]] with {"role": "user"/"assistant", "content": "..."}

while True:
    user_message = input("User: ")
    
    # Assemble a budget-safe prompt
    assembled, history = ctx.assemble(
        user_message,
        system_prompt,
        history,
        recall_k=12
    )
    
    # Check it fits
    print(f"Tokens: {assembled.total_tokens} / {budget.context_limit}")
    assert assembled.within_limit
    
    # Generate
    reply = my_ollama_generate(assembled.to_messages())
    print(f"Assistant: {reply}")
    
    # Persist the exchange
    ctx.ingest_turn(user_message, reply)
    
    # Continue the live transcript
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": reply})
```

#### Breakdown

```python
assembled.breakdown
# {
#   "system": 28,
#   "memories": 3200,
#   "history": 58000,
#   "user": 19,
#   "reserved_for_response": 2048,
#   "context_limit": 65536
# }
```

#### Custom Token Counter

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
ctx = ContextManager(
    memory=memory,
    budget=budget,
    token_counter=lambda text: len(enc.encode(text))
)
```

#### Custom Summarizer

```python
from ai_memory.embedder import ollama_chat

def my_summarizer(raw_text: str) -> str:
    return ollama_chat(
        f"Summarize in 2-3 sentences:\n\n{raw_text}",
        system="You are a precise note-taker.",
        model="llama3.1"
    )

ctx = ContextManager(memory, budget, summarizer=my_summarizer)
```

---

## Usage Patterns

### Pattern 1: Single-Agent Long-Term Memory

```python
from graphdb.store import GraphStore
from ai_memory.memory import AgentMemory
from ai_memory.embedder import get_embedder

store = GraphStore(path="agent.json")
memory = AgentMemory("assistant", store=store, embedder=get_embedder(prefer_ollama=True))

# Store facts as you learn them
memory.remember("User's name is Alice.")
memory.remember("Alice prefers Python over JavaScript.")

# Later: retrieve relevant context
context = memory.get_context("What programming language does the user prefer?", k=5)
print(context)
```

### Pattern 2: Multi-Agent Collaboration via A2A

```python
from protocols import Orchestrator

orch = Orchestrator()

# Create specialist agents
orch.create_agent("researcher", interests=["research", "data"])
orch.create_agent("engineer", interests=["code", "data"])
orch.create_agent("writer", interests=["content", "docs"])

# Researcher shares a finding
orch.a2a_share(
    "researcher",
    "Found benchmark data: 10x speedup with Rust.",
    topics=["data", "research"]
)
# → routed to "engineer" (has "data" interest)

# Engineer shares implementation
orch.a2a_share(
    "engineer",
    "Implemented the optimization in Rust.",
    topics=["code"]
)
# → routed to "writer" (no match) and self

# Check engineer's inbox
inbox = orch.inbox("engineer")
print(len(inbox), "messages")
```

### Pattern 3: Budget-Safe Ollama Chat

```python
from ai_memory.context_window import ContextBudget, ContextManager
from ai_memory.embedder import ollama_chat

budget = ContextBudget(context_limit=16_384)  # 16k model
ctx = ContextManager(memory, budget)

history = []
for user_msg in ["Tell me about Python", "Now about Rust", "Compare them"]:
    assembled, history = ctx.assemble(user_msg, "You are helpful.", history)
    
    # Always fits
    assert assembled.within_limit
    
    # Generate
    reply = ollama_chat(assembled.to_prompt(), model="llama3.2:3b")
    
    # Persist + continue
    ctx.ingest_turn(user_msg, reply)
    history.extend([
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": reply}
    ])
```

### Pattern 4: Project Knowledge Base

```python
# Build a knowledge graph for a codebase
memory.remember("The API uses FastAPI and returns JSON.", memory_type=MemoryType.FACT)
memory.remember("Database is PostgreSQL, accessed via SQLAlchemy.", memory_type=MemoryType.FACT)
memory.remember("Auth is JWT-based, tokens expire in 1 hour.", memory_type=MemoryType.FACT)

# Add entities
memory.add_entity("FastAPI", entity_type="library")
memory.add_entity("PostgreSQL", entity_type="database")

# Query the knowledge base
results = memory.recall("What database do we use?", k=3)
for r in results:
    print(r.node.properties["text"])
```

### Pattern 5: Session-Scoped Memories

```python
# Start a session
session_id = "conv-2024-01-15"

memory.remember("User asked about deployment.", session_id=session_id)
memory.remember("I suggested Docker Compose.", session_id=session_id)

# Later: recall from this session only
results = memory.recall("What did we discuss?", session_id=session_id, k=5)
```

---

## Examples

The repository includes working examples in `/examples/`:

| File | Description |
|------|-------------|
| `example_mcp.py` | MCP tools & resources over graph memory |
| `example_a2a.py` | A2A interest-based memory routing |
| `example_orchestration.py` | Full MCP+A2A orchestration |
| `example_project_layers.py` | One agent per AI-layer workflow (Plan→Implement→Validate) |
| `example_ollama.py` | Use local Ollama models for embeddings + chat |
| `example_ollama_context.py` | Keep Ollama within 64k–256k context window |

Run any example:
```bash
cd python
python examples/example_mcp.py
```

---

## Best Practices

### 1. Choose the Right Embedder

| Scenario | Embedder | Why |
|----------|----------|-----|
| Offline development | `LocalEmbedder` | No dependencies, deterministic |
| Local deployment | `OllamaEmbedder` | Neural quality, no cloud |
| Production (cloud OK) | `OpenAIEmbedder` | Best quality |

### 2. Use Memory Types Intentionally

- `OBSERVATION` — raw turn-level I/O
- `FACT` — extracted, deduplicated knowledge
- `REFLECTION` — periodically synthesize these for high-level summaries
- `PLAN` — for future intent
- `EMOTION` — for affective state (optional)

### 3. Persist Regularly

```python
# Auto-load on startup
store = GraphStore(path="agent.json")

# Auto-save after critical operations
memory.remember("Important fact")
store.save()
```

### 4. Label-Scope Your Searches

Always pass `label` when you know the node type:
```python
# Good: scoped
store.search_similar_nodes(vec, label="Memory", k=5)

# Bad: unscoped (searches across ALL node types)
store.search_similar_nodes(vec, k=5)
```

### 5. Tune `memory_fraction` for Your Context Window

| Window | Recommended `memory_fraction` |
|--------|-------------------------------|
| 16k | 0.3–0.4 (favor live transcript) |
| 64k | 0.5 (balanced) |
| 128k+ | 0.6–0.7 (favor retrieval) |

### 6. Use `reflect()` Periodically

Every 20–50 turns, synthesize a reflection:
```python
summary = memory.reflect(recent_k=30)
print(summary)
```
This keeps high-level context available for retrieval.

### 7. Monitor Duplicate Detections

```python
result = memory.remember("Paris is the capital of France.")
if result.was_duplicate:
    print("Already knew this fact.")
```

### 8. Delete Stale Sessions

```python
# Delete old session nodes + their memories
session_nodes = store.nodes_by_label("Session")
for s in session_nodes:
    if s.properties.get("created_at") < cutoff_timestamp:
        store.delete_node(s.id)  # cascades to connected memories
```

---

## Troubleshooting

### "Connection refused" when using OllamaEmbedder

**Cause:** Ollama server not running.

**Fix:**
```bash
ollama serve &
ollama pull nomic-embed-text
```

### "GraphError: DimensionMismatch"

**Cause:** Embeddings have different dimensions (e.g., switched embedders mid-session).

**Fix:** Use the same embedder consistently, or clear the graph and rebuild.

### Context still overflows with ContextManager

**Cause:** Your actual `num_ctx` is smaller than `context_limit`, or you're using a real tokenizer and the heuristic (`chars/4`) underestimated.

**Fix:**
1. Check Ollama's actual `num_ctx`: `ollama show <model> | grep num_ctx`
2. Pass a real tokenizer as `token_counter`
3. Lower `context_limit` by 10–20% as a safety margin

### A2A routing doesn't deliver messages

**Cause:** No topic overlap and embedding similarity below `interest_threshold`.

**Fix:**
1. Check agent interests: `agent.card.interests`
2. Lower `interest_threshold` (default 0.35, try 0.25)
3. Use explicit topic overlap (more reliable than embedding similarity with `LocalEmbedder`)

### Graph file corrupted after crash

**Cause:** Interrupted write.

**Fix:** The Rust backend uses atomic writes (`.tmp` → rename), so corruption is rare. The Python backend doesn't — use the Rust backend for production, or implement your own backup strategy.

### Memory search returns irrelevant results

**Cause:** `LocalEmbedder` is weak for semantic similarity.

**Fix:** Switch to `OllamaEmbedder` or `OpenAIEmbedder`:
```python
memory = AgentMemory("agent", store, embedder=OllamaEmbedder())
```

### WebUI shows 0 agents after restart

**Cause:** Agents are not persisted separately; only the graph is.

**Fix:** The webui's `service.py` calls `_rehydrate_agents()` which recreates agents from `AGENT` nodes in the graph. Make sure you're using the same `webui_graph.json` path.

---

## Summary

You now have everything you need to integrate the **AI-GraphDB-Engine** into your LLM or agent system:

1. **Install** — `pip install -e .` in `python/`
2. **Store** — `memory.remember(text)`
3. **Retrieve** — `memory.recall(query, k=5)`
4. **Stay in context** — `ContextManager` for budget-safe prompts
5. **Collaborate** — `Orchestrator` + A2A for multi-agent systems
6. **Expose tools** — MCP servers for tool-calling LLMs
7. **Persist** — `store.save(path)`

The graph holds **everything**; retrieval surfaces **only what's relevant**. Your LLM never overflows, and nothing is lost.

For more details, see:
- `examples/` — working code samples
- `protocols/` — MCP/A2A source
- `ai_memory/` — memory + context management source
- `graphdb/` — core graph engine

---

**License:** MIT  
**Repository:** https://github.com/frederico-tech6868/ai-graph-db-engine  
**Issues:** https://github.com/frederico-tech6868/ai-graph-db-engine/issues

Happy building! 🚀
