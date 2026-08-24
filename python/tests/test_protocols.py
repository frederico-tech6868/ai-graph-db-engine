"""Tests for the MCP + A2A protocol layer (protocols package).

All tests run fully offline with the deterministic ``LocalEmbedder``.
"""

import pytest

from ai_memory.embedder import LocalEmbedder
from ai_memory.memory import AgentMemory
from ai_memory.schema import MemoryType, REMEMBERS
from graphdb import GraphStore
from protocols import (
    A2AAgent,
    A2ABus,
    AgentCard,
    MCPClient,
    Orchestrator,
    build_memory_mcp_server,
)
from protocols.schema import INTERESTED_IN, PUBLISHED, SHARED_WITH, TAGGED, TOPIC


# ------------------------------------------------------------------- fixtures
@pytest.fixture
def memory():
    return AgentMemory("agent-1", GraphStore(), LocalEmbedder())


# ------------------------------------------------------------------ MCP tests
def test_mcp_list_and_call_tools(memory):
    server = build_memory_mcp_server(memory)
    client = MCPClient()
    client.connect(server)

    names = {t["name"] for t in client.list_tools()}
    assert {"remember_fact", "recall_memories", "search_nodes"} <= names

    env = client.call_tool("remember_fact", {"text": "Postgres uses MVCC", "memory_type": "fact"})
    assert env["isError"] is False
    assert env["structuredContent"]["was_duplicate"] is False

    recall = client.call_tool("recall_memories", {"query": "Postgres MVCC", "k": 3})
    assert recall["isError"] is False
    assert any("Postgres" in r["text"] for r in recall["structuredContent"])


def test_mcp_resources(memory):
    server = build_memory_mcp_server(memory)
    client = MCPClient()
    client.connect(server)
    memory.remember("A first memory", memory_type=MemoryType.OBSERVATION)

    uris = {r["uri"] for r in client.list_resources()}
    assert uris == {"memory://stats", "memory://entities", "memory://recent"}

    res = client.read_resource("memory://stats")
    assert res["isError"] is False
    assert res["structuredContent"]["total_memories"] >= 1


def test_mcp_unknown_tool_is_error(memory):
    server = build_memory_mcp_server(memory)
    assert server.call_tool("does_not_exist", {})["isError"] is True


def test_mcp_call_log(memory):
    server = build_memory_mcp_server(memory)
    client = MCPClient()
    client.connect(server)
    client.call_tool("remember_fact", {"text": "logged fact"})
    assert len(client.call_log) == 1
    assert client.call_log[0].tool == "remember_fact"


# ------------------------------------------------------------------ A2A tests
def test_a2a_topic_routing():
    store, emb = GraphStore(), LocalEmbedder()
    bus = A2ABus(emb)
    a = A2AAgent(AgentMemory("a", store, emb), AgentCard("a", "A", interests=["databases"]))
    b = A2AAgent(AgentMemory("b", store, emb), AgentCard("b", "B", interests=["databases"]))
    c = A2AAgent(AgentMemory("c", store, emb), AgentCard("c", "C", interests=["cooking"]))
    for ag in (a, b, c):
        bus.register(ag)

    out = bus.share_memory("a", "Graph databases scale horizontally", topics=["databases"])
    got = {d["agent_id"] for d in out["delivered_to"]}
    assert got == {"b"}  # b interested, c not, sender excluded
    assert len(b.inbox) == 1
    assert b.inbox[0].content["match_reason"] == "topic:databases"
    assert len(c.inbox) == 0


def test_a2a_explicit_recipients():
    store, emb = GraphStore(), LocalEmbedder()
    bus = A2ABus(emb)
    a = A2AAgent(AgentMemory("a", store, emb), AgentCard("a", "A"))
    b = A2AAgent(AgentMemory("b", store, emb), AgentCard("b", "B"))
    bus.register(a)
    bus.register(b)
    out = bus.share_memory("a", "direct note", topics=[], recipients=["b"])
    assert [d["agent_id"] for d in out["delivered_to"]] == ["b"]


def test_a2a_provenance_edges():
    store, emb = GraphStore(), LocalEmbedder()
    bus = A2ABus(emb)
    a = A2AAgent(AgentMemory("a", store, emb), AgentCard("a", "A", interests=["x"]))
    b = A2AAgent(AgentMemory("b", store, emb), AgentCard("b", "B", interests=["databases"]))
    bus.register(a)
    bus.register(b)

    out = bus.share_memory("a", "shared", topics=["databases"])
    mem_id = out["memory_id"]

    labels_from_mem = {e.label for e in store.edges_from(mem_id)}
    assert TAGGED in labels_from_mem
    assert SHARED_WITH in labels_from_mem

    # sender PUBLISHED, recipient REMEMBERS
    assert any(e.label == PUBLISHED for e in store.edges_from(a.node_id))
    assert any(e.label == REMEMBERS and e.dst_id == mem_id for e in store.edges_from(b.node_id))
    # topic node exists
    assert store.find_nodes(label=TOPIC, name="databases")
    # interest edge materialised on register
    assert any(e.label == INTERESTED_IN for e in store.edges_from(b.node_id))


def test_a2a_direct_send():
    store, emb = GraphStore(), LocalEmbedder()
    bus = A2ABus(emb)
    a = A2AAgent(AgentMemory("a", store, emb), AgentCard("a", "A"))
    b = A2AAgent(AgentMemory("b", store, emb), AgentCard("b", "B"))
    bus.register(a)
    bus.register(b)
    msg = bus.send("a", "b", {"hello": "world"}, type="text")
    assert b.inbox[0].id == msg.id
    assert len(bus.history) == 1


# --------------------------------------------------------- orchestrator tests
def test_orchestrator_end_to_end():
    orc = Orchestrator()
    orc.create_agent("alice", interests=["databases", "machine learning"])
    bob = orc.create_agent("bob", interests=["databases"])

    # MCP through the orchestrator
    r = orc.mcp_call("alice", "remember_fact", {"text": "B-trees index data"})
    assert r["isError"] is False
    assert {t["name"] for t in orc.tools("alice")}
    assert {x["uri"] for x in orc.resources("alice")}

    # A2A through the orchestrator
    out = orc.a2a_share("alice", "New vector index for databases", topics=["databases"])
    assert [d["agent_id"] for d in out["delivered_to"]] == ["bob"]
    assert len(orc.inbox("bob")) == 1
    assert len(orc.messages()) >= 1
    assert len(orc.call_log()) >= 1

    summary = {a["agent_id"]: a for a in orc.agents()}
    assert summary["bob"]["inbox"] == 1
    assert summary["alice"]["memory"]["total_memories"] >= 1


def test_orchestrator_shared_store():
    orc = Orchestrator()
    orc.create_agent("a")
    orc.create_agent("b")
    # both agents live in the same store
    assert orc.get_agent("a").memory.store is orc.get_agent("b").memory.store


def test_orchestrator_preview_interest():
    orc = Orchestrator()
    orc.create_agent("a", interests=["databases"])
    orc.create_agent("b", interests=["cooking"])
    preview = orc.preview_interest(["databases"], "some db text")
    assert [p["agent_id"] for p in preview] == ["a"]
