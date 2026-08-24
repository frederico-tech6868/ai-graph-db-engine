"""Semantic recall engine: search + context building.

The :class:`RecallEngine` wraps the graph store's label-scoped vector search
and adds:

* property filtering (memory type, session);
* a human-/LLM-readable context window builder with token budgeting;
* entity-neighbourhood BFS;
* temporal windows over a session's ``FOLLOWS`` chain.

All vector search is scoped to :data:`~ai_memory.schema.MEMORY` nodes so that
memory embeddings are never compared against entity or session embeddings.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

from graphdb.core import Node
from graphdb.query import bfs

from .schema import MEMORY, FOLLOWS

if TYPE_CHECKING:  # pragma: no cover
    from graphdb.store import GraphStore
    from .embedder import Embedder


@dataclass
class RecalledMemory:
    """A memory node returned from a recall query."""

    node: Node
    score: float
    context_snippet: str


def _snippet(text: str, limit: int = 160) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "\u2026"


def _fmt_timestamp(ts: object) -> str:
    """Format a float epoch or iso string as a compact readable timestamp."""
    if isinstance(ts, (int, float)):
        try:
            return _dt.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
        except (OverflowError, OSError, ValueError):
            return str(ts)
    return str(ts) if ts else "unknown"


class RecallEngine:
    """Semantic search + context window construction over memory nodes."""

    def __init__(self, store: "GraphStore", embedder: "Embedder") -> None:
        self.store = store
        self.embedder = embedder

    # ------------------------------------------------------------- search
    def search(
        self,
        query: str,
        label: str = MEMORY,
        k: int = 10,
        filters: Optional[Dict[str, object]] = None,
    ) -> List[RecalledMemory]:
        """Return the ``k`` most relevant nodes for ``query``.

        The vector search is label-scoped (default :data:`MEMORY`). Optional
        ``filters`` are applied as exact-match property constraints *after* the
        vector search, so we over-fetch to keep ``k`` results after filtering.
        """
        filters = filters or {}
        query_vec = self.embedder.embed(query)

        # Over-fetch when filtering so we still return up to ``k`` matches.
        fetch_k = k * 5 if filters else k
        scored = self.store.search_similar_nodes(query_vec, label=label, k=fetch_k)

        results: List[RecalledMemory] = []
        for node, score in scored:
            if not self._passes(node, filters):
                continue
            text = str(node.properties.get("text", ""))
            results.append(
                RecalledMemory(node=node, score=score, context_snippet=_snippet(text))
            )
            if len(results) >= k:
                break
        return results

    @staticmethod
    def _passes(node: Node, filters: Dict[str, object]) -> bool:
        for key, value in filters.items():
            if value is None:
                continue
            if node.properties.get(key) != value:
                return False
        return True

    # ------------------------------------------------- context window
    def build_context_window(
        self,
        memories: List[RecalledMemory],
        max_tokens: int = 2000,
    ) -> str:
        """Format recalled memories into a context string.

        Each line looks like ``MEMORY [2024-01-01 10:00:00] (fact): <text>``.
        Lines are added until the approximate token budget (``~4 chars/token``)
        is exhausted.
        """
        if not memories:
            return "(no relevant memories)"

        char_budget = max_tokens * 4
        lines: List[str] = []
        used = 0
        for rm in memories:
            props = rm.node.properties
            ts = _fmt_timestamp(props.get("timestamp") or props.get("created_at"))
            mtype = props.get("memory_type", "observation")
            text = str(props.get("text", ""))
            line = f"MEMORY [{ts}] ({mtype}): {text}"
            if used + len(line) + 1 > char_budget and lines:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)

    # --------------------------------------------------- graph helpers
    def get_entity_neighborhood(self, entity_name: str, depth: int = 2) -> List[Node]:
        """BFS out from an entity node; return connected memory nodes.

        Traverses edges in both directions (memories point *to* entities via
        ``RELATES_TO``) up to ``depth`` hops and returns the reachable
        :data:`MEMORY` nodes.
        """
        entity = self._find_entity(entity_name)
        if entity is None:
            return []
        reachable = bfs(
            self.store, entity.id, edge_label=None, max_depth=depth, direction="both"
        )
        return [n for n in reachable if n.label == MEMORY]

    def temporal_window(self, session_id: str, last_n: int = 10) -> List[Node]:
        """Return the last ``last_n`` memories of a session in time order.

        Memories are chained together with ``FOLLOWS`` edges (newest ->
        previous). We collect the session's memory nodes and order them by
        their stored ``timestamp`` (which matches the ``FOLLOWS`` chain order),
        returning the most recent ``last_n`` in chronological order.
        """
        mems = [
            n
            for n in self.store.nodes_by_label(MEMORY)
            if n.properties.get("session_id") == session_id
        ]
        mems.sort(key=lambda n: n.properties.get("timestamp", 0.0))
        return mems[-last_n:]

    # ------------------------------------------------------------ internal
    def _find_entity(self, entity_name: str) -> Optional[Node]:
        from .schema import ENTITY

        matches = self.store.find_nodes(label=ENTITY, name=entity_name)
        return matches[0] if matches else None


__all__ = ["RecallEngine", "RecalledMemory"]
