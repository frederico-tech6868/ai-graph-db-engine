"""A lightweight, in-process MCP (Model Context Protocol) implementation.

MCP is the "Agent -> Tools and Data" protocol: a *server* exposes **tools**
(callable functions with a JSON input schema) and **resources** (readable data
identified by a URI), and a *client* discovers and invokes them.

This module implements the MCP conceptual surface faithfully (``list_tools`` /
``call_tool`` / ``list_resources`` / ``read_resource`` with MCP-shaped
envelopes) without requiring a network transport, so it runs fully in-process
and offline. :func:`build_memory_mcp_server` wires an
:class:`~ai_memory.memory.AgentMemory` up as an MCP server, turning the graph
memory into a set of tools + data resources any agent can use.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ai_memory.memory import AgentMemory


# --------------------------------------------------------------------- models
@dataclass
class MCPTool:
    """A callable tool exposed over MCP."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Any]

    def describe(self) -> Dict[str, Any]:
        """Return the MCP ``tools/list`` descriptor (no handler)."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


@dataclass
class MCPResource:
    """A readable data resource exposed over MCP, addressed by a URI."""

    uri: str
    name: str
    description: str
    reader: Callable[[], Any]
    mime_type: str = "application/json"

    def describe(self) -> Dict[str, Any]:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }


@dataclass
class ToolCall:
    """A record of a single tool invocation (for observability/orchestration)."""

    server: str
    tool: str
    arguments: Dict[str, Any]
    result: Any
    is_error: bool
    ts: float = field(default_factory=time.time)


# --------------------------------------------------------------------- server
class MCPServer:
    """An in-process MCP server exposing tools and resources."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._tools: Dict[str, MCPTool] = {}
        self._resources: Dict[str, MCPResource] = {}

    # ------------------------------------------------------------ registration
    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable[[Dict[str, Any]], Any],
    ) -> None:
        self._tools[name] = MCPTool(name, description, input_schema, handler)

    def register_resource(
        self,
        uri: str,
        name: str,
        description: str,
        reader: Callable[[], Any],
        mime_type: str = "application/json",
    ) -> None:
        self._resources[uri] = MCPResource(uri, name, description, reader, mime_type)

    # ---------------------------------------------------------------- protocol
    def list_tools(self) -> List[Dict[str, Any]]:
        """MCP ``tools/list``."""
        return [t.describe() for t in self._tools.values()]

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """MCP ``tools/call``.

        Returns an MCP-shaped result envelope:
        ``{"content": [{"type": "text", "text": ...}], "isError": bool}``.
        """
        tool = self._tools.get(name)
        if tool is None:
            return self._error(f"unknown tool: {name}")
        try:
            result = tool.handler(arguments or {})
            return {
                "content": [{"type": "text", "text": _as_text(result)}],
                "structuredContent": result,
                "isError": False,
            }
        except Exception as exc:  # surfaces tool failures to the caller
            return self._error(f"{type(exc).__name__}: {exc}")

    def list_resources(self) -> List[Dict[str, Any]]:
        """MCP ``resources/list``."""
        return [r.describe() for r in self._resources.values()]

    def read_resource(self, uri: str) -> Dict[str, Any]:
        """MCP ``resources/read``."""
        resource = self._resources.get(uri)
        if resource is None:
            return self._error(f"unknown resource: {uri}")
        try:
            data = resource.reader()
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": resource.mime_type,
                        "text": _as_text(data),
                    }
                ],
                "structuredContent": data,
                "isError": False,
            }
        except Exception as exc:
            return self._error(f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _error(message: str) -> Dict[str, Any]:
        return {"content": [{"type": "text", "text": message}], "isError": True}


# --------------------------------------------------------------------- client
class MCPClient:
    """A thin MCP client bound to one or more :class:`MCPServer` instances."""

    def __init__(self) -> None:
        self._servers: Dict[str, MCPServer] = {}
        self.call_log: List[ToolCall] = []

    def connect(self, server: MCPServer) -> None:
        self._servers[server.name] = server

    def servers(self) -> List[str]:
        return list(self._servers.keys())

    def _server(self, name: Optional[str]) -> MCPServer:
        if name is not None:
            if name not in self._servers:
                raise KeyError(f"no MCP server named {name!r}")
            return self._servers[name]
        if len(self._servers) != 1:
            raise ValueError("server name required when multiple servers are connected")
        return next(iter(self._servers.values()))

    # ---------------------------------------------------------------- discovery
    def list_tools(self, server: Optional[str] = None) -> List[Dict[str, Any]]:
        if server is None:
            out: List[Dict[str, Any]] = []
            for srv in self._servers.values():
                for desc in srv.list_tools():
                    out.append({"server": srv.name, **desc})
            return out
        return [{"server": server, **d} for d in self._server(server).list_tools()]

    def list_resources(self, server: Optional[str] = None) -> List[Dict[str, Any]]:
        if server is None:
            out: List[Dict[str, Any]] = []
            for srv in self._servers.values():
                for desc in srv.list_resources():
                    out.append({"server": srv.name, **desc})
            return out
        return [{"server": server, **d} for d in self._server(server).list_resources()]

    # ---------------------------------------------------------------- invocation
    def call_tool(
        self,
        tool: str,
        arguments: Optional[Dict[str, Any]] = None,
        server: Optional[str] = None,
    ) -> Dict[str, Any]:
        srv = self._server(server)
        envelope = srv.call_tool(tool, arguments)
        self.call_log.append(
            ToolCall(
                server=srv.name,
                tool=tool,
                arguments=arguments or {},
                result=envelope.get("structuredContent"),
                is_error=bool(envelope.get("isError")),
            )
        )
        return envelope

    def read_resource(self, uri: str, server: Optional[str] = None) -> Dict[str, Any]:
        return self._server(server).read_resource(uri)


# ------------------------------------------------------- memory MCP server
def build_memory_mcp_server(memory: "AgentMemory", name: Optional[str] = None) -> MCPServer:
    """Expose an :class:`AgentMemory` as an MCP server (tools + resources)."""
    from ai_memory.schema import ENTITY, MEMORY, MemoryType

    server = MCPServer(name or f"memory::{memory.agent_id}")

    # ------------------------------------------------------------------ tools
    def _remember(args: Dict[str, Any]) -> Dict[str, Any]:
        mtype = args.get("memory_type", "fact")
        try:
            mt = MemoryType(mtype)
        except ValueError:
            mt = MemoryType.FACT
        res = memory.remember(
            text=args["text"],
            memory_type=mt,
            entities=args.get("entities"),
        )
        return {
            "memory_id": res.memory_node.id,
            "was_duplicate": res.was_duplicate,
            "similar_count": len(res.similar_existing),
        }

    def _recall(args: Dict[str, Any]) -> List[Dict[str, Any]]:
        recalled = memory.recall(args["query"], k=int(args.get("k", 5)))
        return [
            {
                "text": rm.node.properties.get("text", ""),
                "score": round(rm.score, 4),
                "memory_type": rm.node.properties.get("memory_type"),
            }
            for rm in recalled
        ]

    def _search(args: Dict[str, Any]) -> List[Dict[str, Any]]:
        vec = memory.embedder.embed(args["text"])
        label = args.get("label")
        results = memory.store.search_similar_nodes(vec, label=label, k=int(args.get("k", 5)))
        return [
            {
                "id": n.id,
                "label": n.label,
                "name": n.properties.get("name") or n.properties.get("text", "")[:60],
                "score": round(score, 4),
            }
            for n, score in results
        ]

    def _get_entity(args: Dict[str, Any]) -> Dict[str, Any]:
        name_ = args["name"]
        entity = memory.get_entity(name_)
        if entity is None:
            return {"found": False, "name": name_}
        related = memory.recall_engine.get_entity_neighborhood(name_, depth=2)
        return {
            "found": True,
            "name": name_,
            "entity_type": entity.properties.get("entity_type"),
            "related_memories": [n.properties.get("text", "") for n in related],
        }

    def _reflect(args: Dict[str, Any]) -> Dict[str, Any]:
        return {"reflection": memory.reflect(recent_k=int(args.get("recent_k", 20)))}

    server.register_tool(
        "remember_fact",
        "Store a fact or observation in long-term graph memory.",
        {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "memory_type": {"type": "string", "enum": [t.value for t in MemoryType]},
                "entities": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["text"],
        },
        _remember,
    )
    server.register_tool(
        "recall_memories",
        "Retrieve memories relevant to a query (label-scoped vector search).",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer"},
            },
            "required": ["query"],
        },
        _recall,
    )
    server.register_tool(
        "search_nodes",
        "Vector-search nodes by text, optionally scoped to a single label.",
        {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "label": {"type": "string"},
                "k": {"type": "integer"},
            },
            "required": ["text"],
        },
        _search,
    )
    server.register_tool(
        "get_entity_info",
        "Get an entity and its related memories.",
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        _get_entity,
    )
    server.register_tool(
        "reflect",
        "Synthesise a higher-level reflection from recent memories.",
        {
            "type": "object",
            "properties": {"recent_k": {"type": "integer"}},
            "required": [],
        },
        _reflect,
    )

    # -------------------------------------------------------------- resources
    server.register_resource(
        "memory://stats",
        "Memory statistics",
        "Counts of memories, entities, sessions and edges.",
        memory.stats,
    )
    server.register_resource(
        "memory://entities",
        "Known entities",
        "All entities the agent knows about.",
        lambda: [
            {"name": n.properties.get("name"), "entity_type": n.properties.get("entity_type")}
            for n in memory.store.nodes_by_label(ENTITY)
        ],
    )
    server.register_resource(
        "memory://recent",
        "Recent memories",
        "The most recent memories (chronological).",
        lambda: [
            {"text": n.properties.get("text", ""), "type": n.properties.get("memory_type")}
            for n in sorted(
                memory.store.nodes_by_label(MEMORY),
                key=lambda x: x.properties.get("timestamp", 0.0),
            )[-10:]
        ],
    )
    return server


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


__all__ = [
    "MCPTool",
    "MCPResource",
    "ToolCall",
    "MCPServer",
    "MCPClient",
    "build_memory_mcp_server",
]
