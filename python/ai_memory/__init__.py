"""ai_memory: long-term semantic memory for AI agents, built on graphdb.

Phase 2 of the graphdb project. Turns the in-memory property graph into a
persistent semantic memory system for LLM agents.

Quick start (fully offline, no API key)::

    from graphdb import GraphStore
    from ai_memory import AgentMemory, GraphAgent, LocalEmbedder

    store = GraphStore()
    memory = AgentMemory("agent-1", store, LocalEmbedder())
    agent = GraphAgent("agent-1", memory)
    print(agent.chat("My name is Alice and I work at TechCorp."))
"""

from __future__ import annotations

from .agent import GraphAgent
from .embedder import Embedder, LocalEmbedder, OpenAIEmbedder, get_embedder
from .memory import AgentMemory, MemoryResult, SimilarMemory
from .prompts import (
    ENTITY_EXTRACTION_PROMPT,
    REFLECTION_PROMPT,
    SYSTEM_PROMPT_WITH_MEMORY,
)
from .needle_agent import (
    GRAPHDB_TOOL_SCHEMAS,
    NeedleAgentGroup,
    NeedleOrchestrator,
)
from .recall import RecallEngine, RecalledMemory
from .schema import (
    AGENT,
    ENTITY,
    FOLLOWS,
    KNOWS,
    MEMORY,
    OCCURRED_IN,
    RELATES_TO,
    REMEMBERS,
    SESSION,
    SIMILAR_TO,
    MemoryType,
)
from .tools import MEMORY_TOOLS, MemoryToolExecutor

__version__ = "0.2.0"

__all__ = [
    # embedder
    "Embedder",
    "LocalEmbedder",
    "OpenAIEmbedder",
    "get_embedder",
    # memory
    "AgentMemory",
    "MemoryResult",
    "SimilarMemory",
    # recall
    "RecallEngine",
    "RecalledMemory",
    # agent
    "GraphAgent",
    # needle
    "GRAPHDB_TOOL_SCHEMAS",
    "NeedleAgentGroup",
    "NeedleOrchestrator",
    # tools
    "MEMORY_TOOLS",
    "MemoryToolExecutor",
    # prompts
    "SYSTEM_PROMPT_WITH_MEMORY",
    "REFLECTION_PROMPT",
    "ENTITY_EXTRACTION_PROMPT",
    # schema
    "MemoryType",
    "MEMORY",
    "ENTITY",
    "SESSION",
    "AGENT",
    "RELATES_TO",
    "OCCURRED_IN",
    "FOLLOWS",
    "SIMILAR_TO",
    "KNOWS",
    "REMEMBERS",
]
