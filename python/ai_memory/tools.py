"""LLM tool (function-calling) definitions for the memory system.

``MEMORY_TOOLS`` is a list of JSON schemas in the OpenAI function-calling
format. :class:`MemoryToolExecutor` dispatches a tool call (name + arguments)
to the corresponding :class:`~ai_memory.memory.AgentMemory` method and returns
a JSON-serialisable result.
"""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

from .schema import MemoryType

if TYPE_CHECKING:  # pragma: no cover
    from .memory import AgentMemory


def _tool(name: str, description: str, properties: dict, required: list) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


MEMORY_TOOLS: List[dict] = [
    _tool(
        "remember_fact",
        "Store a fact or observation in long-term memory.",
        {
            "text": {"type": "string", "description": "The fact to remember."},
            "memory_type": {
                "type": "string",
                "enum": [t.value for t in MemoryType],
                "description": "The kind of memory (default 'fact').",
            },
            "entities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Named entities mentioned in the fact.",
            },
        },
        ["text"],
    ),
    _tool(
        "recall_memories",
        "Retrieve memories relevant to a query.",
        {
            "query": {"type": "string", "description": "What to recall."},
            "k": {"type": "integer", "description": "Max number of memories (default 5)."},
        },
        ["query"],
    ),
    _tool(
        "get_entity_info",
        "Get stored information and related memories about a named entity.",
        {"name": {"type": "string", "description": "The entity name."}},
        ["name"],
    ),
    _tool(
        "reflect",
        "Synthesise a higher-level reflection from recent memories.",
        {
            "recent_k": {
                "type": "integer",
                "description": "How many recent memories to consider (default 20).",
            }
        },
        [],
    ),
    _tool(
        "list_entities",
        "List all entities the agent knows about.",
        {},
        [],
    ),
]


class MemoryToolExecutor:
    """Maps tool calls to :class:`AgentMemory` method invocations."""

    def __init__(self, memory: "AgentMemory") -> None:
        self.memory = memory

    def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Dispatch a tool call by name; returns a JSON-serialisable result."""
        arguments = arguments or {}
        handler = getattr(self, f"_do_{name}", None)
        if handler is None:
            return {"error": f"unknown tool: {name}"}
        return handler(arguments)

    # ---------------------------------------------------------------- tools
    def _do_remember_fact(self, args: Dict[str, Any]) -> Dict[str, Any]:
        mtype = MemoryType(args.get("memory_type", "fact")) if args.get("memory_type") else MemoryType.FACT
        result = self.memory.remember(
            text=args["text"],
            memory_type=mtype,
            entities=args.get("entities"),
        )
        return {
            "memory_id": result.memory_node.id,
            "was_duplicate": result.was_duplicate,
            "similar_count": len(result.similar_existing),
        }

    def _do_recall_memories(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        recalled = self.memory.recall(args["query"], k=int(args.get("k", 5)))
        return [
            {
                "text": rm.node.properties.get("text", ""),
                "score": round(rm.score, 4),
                "memory_type": rm.node.properties.get("memory_type"),
            }
            for rm in recalled
        ]

    def _do_get_entity_info(self, args: Dict[str, Any]) -> Dict[str, Any]:
        name = args["name"]
        entity = self.memory.get_entity(name)
        if entity is None:
            return {"found": False, "name": name}
        related = self.memory.recall_engine.get_entity_neighborhood(name, depth=2)
        return {
            "found": True,
            "name": name,
            "entity_type": entity.properties.get("entity_type"),
            "related_memories": [n.properties.get("text", "") for n in related],
        }

    def _do_reflect(self, args: Dict[str, Any]) -> Dict[str, Any]:
        text = self.memory.reflect(recent_k=int(args.get("recent_k", 20)))
        return {"reflection": text}

    def _do_list_entities(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        from .schema import ENTITY

        return [
            {
                "name": n.properties.get("name"),
                "entity_type": n.properties.get("entity_type"),
            }
            for n in self.memory.store.nodes_by_label(ENTITY)
        ]


__all__ = ["MEMORY_TOOLS", "MemoryToolExecutor"]
