"""MCP (Agent -> Tools & Data) and A2A (Agent -> Agent) protocol layer.

This package adds two agent-interoperability protocols on top of the graph
memory engine, all sharing a single :class:`~graphdb.store.GraphStore`:

* **MCP** (:mod:`protocols.mcp`) — expose an agent's memory as callable *tools*
  and readable *resources*.
* **A2A** (:mod:`protocols.a2a`) — let agents advertise interests and share
  *memories of interest*, interest-routed across a shared graph.
* :class:`~protocols.orchestrator.Orchestrator` — a single facade the web UI and
  examples use to drive both protocols.
"""

from __future__ import annotations

from .a2a import A2AAgent, A2ABus, A2AMessage, AgentCard
from .mcp import (
    MCPClient,
    MCPResource,
    MCPServer,
    MCPTool,
    ToolCall,
    build_memory_mcp_server,
)
from .orchestrator import Orchestrator

__all__ = [
    # MCP
    "MCPTool",
    "MCPResource",
    "ToolCall",
    "MCPServer",
    "MCPClient",
    "build_memory_mcp_server",
    # A2A
    "AgentCard",
    "A2AMessage",
    "A2AAgent",
    "A2ABus",
    # Orchestration
    "Orchestrator",
]
