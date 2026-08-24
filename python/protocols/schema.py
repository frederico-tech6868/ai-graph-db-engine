"""Schema additions for the MCP + A2A protocol layer.

These labels extend the ``ai_memory`` schema so that agents, the tools/data
they expose (MCP) and the memories they share with one another (A2A) all live
in the *same* graph and are therefore visible in the web UI and searchable with
the same label-scoped vector engine.
"""

from __future__ import annotations

# --------------------------------------------------------------------- nodes
# ``Agent`` already exists in ai_memory.schema; we reuse it.
TOOL = "Tool"          # An MCP tool exposed by an agent/server.
RESOURCE = "Resource"  # An MCP data resource exposed by an agent/server.
TOPIC = "Topic"        # A topic of interest for A2A routing.

# --------------------------------------------------------------------- edges
EXPOSES = "EXPOSES"            # Agent   -> Tool      (MCP)
PROVIDES = "PROVIDES"          # Agent   -> Resource  (MCP)
INTERESTED_IN = "INTERESTED_IN"  # Agent -> Topic     (A2A subscription)
PUBLISHED = "PUBLISHED"        # Agent   -> Memory    (A2A: sender published it)
SHARED_WITH = "SHARED_WITH"    # Memory  -> Agent     (A2A: routed to recipient)
TAGGED = "TAGGED"              # Memory  -> Topic     (A2A: topic tag)

NODE_LABELS = (TOOL, RESOURCE, TOPIC)
EDGE_LABELS = (EXPOSES, PROVIDES, INTERESTED_IN, PUBLISHED, SHARED_WITH, TAGGED)

__all__ = [
    "TOOL",
    "RESOURCE",
    "TOPIC",
    "EXPOSES",
    "PROVIDES",
    "INTERESTED_IN",
    "PUBLISHED",
    "SHARED_WITH",
    "TAGGED",
    "NODE_LABELS",
    "EDGE_LABELS",
]
