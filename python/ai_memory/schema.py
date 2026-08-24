"""Semantic schema for the AI agent memory graph.

This module defines the *vocabulary* of the memory graph: the node labels,
edge labels and the ``MemoryType`` enum. Everything else in the ``ai_memory``
package refers back to these constants so the schema stays consistent.

The memory graph is layered on top of the generic :mod:`graphdb` engine:

* **Nodes** carry a text ``label`` (one of the constants below) plus a typed
  property dict and an optional embedding vector.
* **Edges** carry a text ``label`` (one of the edge constants below) that
  encodes the semantics of the relationship.
"""

from __future__ import annotations

from enum import Enum

# --------------------------------------------------------------------- nodes
MEMORY = "Memory"     # A single memory / fact / observation.
ENTITY = "Entity"     # A named entity (person, place, concept, object).
SESSION = "Session"   # A conversation session.
AGENT = "Agent"       # The agent itself.

# --------------------------------------------------------------------- edges
RELATES_TO = "RELATES_TO"     # Memory  -> Entity
OCCURRED_IN = "OCCURRED_IN"   # Memory  -> Session
FOLLOWS = "FOLLOWS"           # Memory  -> Memory  (temporal chain)
SIMILAR_TO = "SIMILAR_TO"     # Memory  -> Memory  (semantic similarity)
KNOWS = "KNOWS"               # Agent   -> Entity
REMEMBERS = "REMEMBERS"       # Agent   -> Memory


class MemoryType(str, Enum):
    """The kind of information a :data:`MEMORY` node holds.

    Inherits from ``str`` so the value can be stored directly as a node
    property (the :mod:`graphdb` engine only accepts primitive property types).
    """

    OBSERVATION = "observation"
    FACT = "fact"
    REFLECTION = "reflection"
    PLAN = "plan"
    EMOTION = "emotion"

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.value


# Convenience collections -------------------------------------------------
NODE_LABELS = (MEMORY, ENTITY, SESSION, AGENT)
EDGE_LABELS = (RELATES_TO, OCCURRED_IN, FOLLOWS, SIMILAR_TO, KNOWS, REMEMBERS)

__all__ = [
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
    "MemoryType",
    "NODE_LABELS",
    "EDGE_LABELS",
]
