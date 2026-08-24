"""High-level orchestrator that ties MCP + A2A together over one shared graph.

The :class:`Orchestrator` is the single entry point used by the web UI and the
examples. It owns:

* one shared :class:`~graphdb.store.GraphStore` and embedder,
* an :class:`~ai_memory.memory.AgentMemory` per agent,
* an MCP server per agent (built with :func:`build_memory_mcp_server`) plus a
  shared :class:`~protocols.mcp.MCPClient` for discovery/invocation,
* an :class:`~protocols.a2a.A2ABus` for interest-routed memory sharing.

Everything runs in-process and offline (default embedder is the deterministic
``LocalEmbedder``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ai_memory.embedder import LocalEmbedder
from ai_memory.memory import AgentMemory

from .a2a import A2AAgent, A2ABus, AgentCard
from .mcp import MCPClient, MCPServer, build_memory_mcp_server


class Orchestrator:
    """Coordinate multiple agents, their MCP tools/data, and A2A sharing."""

    def __init__(self, store=None, embedder=None, interest_threshold: float = 0.35) -> None:
        if store is None:
            from graphdb.store import GraphStore

            store = GraphStore()
        self.store = store
        self.embedder = embedder or LocalEmbedder()
        self.bus = A2ABus(self.embedder, interest_threshold=interest_threshold)
        self.mcp_client = MCPClient()
        self._agents: Dict[str, A2AAgent] = {}
        self._servers: Dict[str, MCPServer] = {}

    # ------------------------------------------------------------- agents
    def create_agent(
        self,
        agent_id: str,
        name: Optional[str] = None,
        description: str = "",
        skills: Optional[List[str]] = None,
        interests: Optional[List[str]] = None,
    ) -> A2AAgent:
        """Create (or return existing) agent with memory, MCP server and A2A card."""
        if agent_id in self._agents:
            return self._agents[agent_id]

        memory = AgentMemory(agent_id=agent_id, store=self.store, embedder=self.embedder)
        card = AgentCard(
            agent_id=agent_id,
            name=name or agent_id,
            description=description,
            skills=list(skills or []),
            interests=list(interests or []),
        )
        agent = A2AAgent(memory=memory, card=card)
        self.bus.register(agent)

        server = build_memory_mcp_server(memory, name=f"mcp::{agent_id}")
        self.mcp_client.connect(server)

        self._agents[agent_id] = agent
        self._servers[agent_id] = server
        return agent

    def agents(self) -> List[Dict[str, Any]]:
        from ai_memory.schema import REMEMBERS

        out = []
        for a in self._agents.values():
            owned = sum(
                1 for e in self.store.edges_from(a.node_id) if e.label == REMEMBERS
            )
            card = a.card.to_dict()
            card["inbox"] = len(a.inbox)
            card["owned_memories"] = owned
            card["memory"] = a.memory.stats()  # store-wide totals
            out.append(card)
        return out

    def get_agent(self, agent_id: str) -> A2AAgent:
        if agent_id not in self._agents:
            raise KeyError(f"unknown agent: {agent_id}")
        return self._agents[agent_id]

    # ---------------------------------------------------------------- MCP
    def server_name(self, agent_id: str) -> str:
        self.get_agent(agent_id)
        return self._servers[agent_id].name

    def tools(self, agent_id: str) -> List[Dict[str, Any]]:
        return self.mcp_client.list_tools(self.server_name(agent_id))

    def resources(self, agent_id: str) -> List[Dict[str, Any]]:
        return self.mcp_client.list_resources(self.server_name(agent_id))

    def mcp_call(self, agent_id: str, tool: str, arguments: Optional[Dict[str, Any]] = None):
        return self.mcp_client.call_tool(
            tool, arguments or {}, server=self.server_name(agent_id)
        )

    def read_resource(self, agent_id: str, uri: str):
        return self.mcp_client.read_resource(uri, server=self.server_name(agent_id))

    def call_log(self) -> List[Dict[str, Any]]:
        return [
            {
                "server": c.server,
                "tool": c.tool,
                "arguments": c.arguments,
                "is_error": c.is_error,
                "ts": c.ts,
            }
            for c in self.mcp_client.call_log
        ]

    # ---------------------------------------------------------------- A2A
    def a2a_share(
        self,
        sender_id: str,
        text: str,
        topics: Optional[List[str]] = None,
        memory_type=None,
        recipients: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return self.bus.share_memory(
            sender_id=sender_id,
            text=text,
            topics=topics,
            memory_type=memory_type,
            recipients=recipients,
        )

    def a2a_send(
        self, sender_id: str, recipient_id: str, content: Dict[str, Any], type: str = "text"
    ) -> Dict[str, Any]:
        return self.bus.send(sender_id, recipient_id, content, type=type).to_dict()

    def preview_interest(self, topics: List[str], text: str) -> List[Dict[str, Any]]:
        return self.bus.interested_agents(topics, text)

    def messages(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self.bus.history[-limit:]]

    def inbox(self, agent_id: str) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self.get_agent(agent_id).inbox]


__all__ = ["Orchestrator"]
