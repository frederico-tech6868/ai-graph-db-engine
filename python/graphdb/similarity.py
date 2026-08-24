"""Label-scoped edge similarity scanning.

Before a new edge is added between ``src`` and ``dst``, we scan the existing
edges that share the same edge label *and* whose endpoints share the same text
labels as ``src`` / ``dst``. We then compare embeddings to find near-duplicate
relationships. All scans are label-scoped so we never compare a node against a
node of a different type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

from .core import Node
from .vector import cosine_similarity

if TYPE_CHECKING:  # pragma: no cover
    from .store import GraphStore


@dataclass
class SimilarMatch:
    """A single existing edge that is similar to a proposed edge."""

    existing_edge_id: str
    src_similarity: float
    dst_similarity: float
    combined_score: float


class SimilarityScanner:
    """Scans existing edges for near-duplicates before an edge is added."""

    def __init__(self, store: "GraphStore") -> None:
        self._store = store

    def scan_before_add(
        self,
        src_node: Node,
        dst_node: Node,
        label: str,
        threshold: float = 0.85,
    ) -> List[SimilarMatch]:
        """Return existing edges similar to a proposed ``src -> dst`` edge.

        Only edges are considered when:

        * the edge label matches ``label``;
        * the edge's source node has the same *text label* as ``src_node``;
        * the edge's destination node has the same *text label* as ``dst_node``.

        For each qualifying edge the cosine similarity between the source
        embeddings and between the destination embeddings is computed. The edge
        is returned when the combined score ``(src_sim + dst_sim) / 2`` is at or
        above ``threshold``.
        """
        matches: List[SimilarMatch] = []

        if src_node.embedding is None or dst_node.embedding is None:
            # Without embeddings on the proposed endpoints there is nothing to
            # compare against; nothing is considered similar.
            return matches

        # Use the label index to restrict the candidate node sets, so we skip
        # any nodes whose text label differs from the proposed endpoints.
        src_label_ids = self._store._label_index.get(src_node.label)
        dst_label_ids = self._store._label_index.get(dst_node.label)

        for edge in self._store.all_edges():
            if edge.label != label:
                continue
            if edge.src_id not in src_label_ids:
                continue
            if edge.dst_id not in dst_label_ids:
                continue

            existing_src = self._store.get_node_or_none(edge.src_id)
            existing_dst = self._store.get_node_or_none(edge.dst_id)
            if existing_src is None or existing_dst is None:
                continue
            if existing_src.embedding is None or existing_dst.embedding is None:
                continue

            src_sim = cosine_similarity(src_node.embedding, existing_src.embedding)
            dst_sim = cosine_similarity(dst_node.embedding, existing_dst.embedding)
            combined = (src_sim + dst_sim) / 2.0

            if combined >= threshold:
                matches.append(
                    SimilarMatch(
                        existing_edge_id=edge.id,
                        src_similarity=src_sim,
                        dst_similarity=dst_sim,
                        combined_score=combined,
                    )
                )

        matches.sort(key=lambda m: m.combined_score, reverse=True)
        return matches
