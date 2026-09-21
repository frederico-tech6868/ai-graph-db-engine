"""JEPA-Needle integration for the AI-GraphDB-Engine.

This module bridges two subsystems that already live in :mod:`ai_memory`:

* :mod:`ai_memory.needle_agent` — Needle2 (``cactus-needle``) embedded
  function-calling agents (:class:`NeedleAgentGroup`,
  :class:`NeedleOrchestrator`).
* :mod:`ai_memory.jepa_graphrag` — the JEPA-GraphRAG retrieval stack
  (:class:`JEPAGraphRAG`) with community detection, local / global / latent
  search and a Graph-JEPA world model trained via VICReg.

It provides:

* :data:`JEPA_TOOL_SCHEMAS` — Needle-compatible tool schemas whose
  ``search_knowledge_base`` is upgraded with a ``mode`` parameter
  (``local`` / ``global`` / ``latent`` / ``hybrid``), plus two new
  community-aware tools (``search_communities``, ``get_community_summary``).
* :class:`JEPANeedleAgent` — a :class:`NeedleAgentGroup` whose graph-backed
  tools route through a shared :class:`JEPAGraphRAG` instance, giving the
  Needle agent multi-mode retrieval instead of plain similarity search.
* :class:`JEPAOrchestrator` — a :class:`NeedleOrchestrator` that routes
  queries by comparing them in the JEPA latent space (energy-based) instead
  of raw text-embedding cosine similarity, with a graceful fallback to the
  text-embedding router.

Like the rest of the Needle surface, everything except *live inference*
(:meth:`JEPANeedleAgent.run`) works **without** ``cactus-needle`` installed.
The JEPA machinery only needs numpy (torch is optional).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from .needle_agent import GRAPHDB_TOOL_SCHEMAS, NeedleAgentGroup, NeedleOrchestrator


# --------------------------------------------------------------------------
# JEPA-enhanced Needle tool schemas
# --------------------------------------------------------------------------
def _build_jepa_tool_schemas() -> List[Dict[str, Any]]:
    """Return JEPA-enhanced copies of the built-in GraphDB tool schemas.

    ``search_knowledge_base`` gains a ``mode`` enum selecting the JEPA-GraphRAG
    retrieval strategy; ``list_documents`` and ``get_document_chunks`` are
    carried over unchanged; and two community-aware tools are appended.
    Needle allows at most 5 tool schemas, which is exactly what we return.
    """
    schemas: List[Dict[str, Any]] = [copy.deepcopy(s) for s in GRAPHDB_TOOL_SCHEMAS]

    for schema in schemas:
        if schema["name"] == "search_knowledge_base":
            schema["description"] = (
                "Search the knowledge base for passages relevant to a query "
                "using JEPA-GraphRAG multi-mode retrieval. 'local' expands the "
                "k-hop neighbourhood of the best-matching nodes, 'global' ranks "
                "graph communities, 'latent' uses the Graph-JEPA world model's "
                "energy-based latent space, and 'hybrid' fuses all modes."
            )
            schema["parameters"]["properties"]["mode"] = {
                "type": "string",
                "description": (
                    "retrieval strategy: local (k-hop), global (community), "
                    "latent (JEPA energy), or hybrid (fused)"
                ),
                "enum": ["local", "global", "latent", "hybrid"],
            }

    # community-aware tools (unique to the JEPA integration)
    schemas.append(
        {
            "name": "search_communities",
            "description": (
                "Find the graph communities most relevant to a query. Returns "
                "community ids, sizes, and human-readable summaries — useful for "
                "broad, thematic questions rather than pinpoint lookups."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "natural language topic to search communities for",
                        "minLength": 1,
                        "maxLength": 500,
                    },
                    "k": {
                        "type": "integer",
                        "description": "number of communities to return",
                        "minimum": 1,
                        "maximum": 20,
                    },
                },
                "required": ["query"],
            },
        }
    )
    schemas.append(
        {
            "name": "get_community_summary",
            "description": (
                "Return a human-readable summary of a specific graph community "
                "by its integer community id, as returned by search_communities."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "community_id": {
                        "type": "integer",
                        "description": "integer community id from search_communities",
                    },
                },
                "required": ["community_id"],
            },
        }
    )
    return schemas


JEPA_TOOL_SCHEMAS: List[Dict[str, Any]] = _build_jepa_tool_schemas()


# --------------------------------------------------------------------------
def _node_to_result(node: Any, score: float) -> Dict[str, Any]:
    """Map a graph :class:`Node` + score into a Needle-friendly result dict."""
    props = getattr(node, "properties", {}) or {}
    return {
        "text": props.get("text", ""),
        "source": props.get("source_path", ""),
        "doc_type": props.get("doc_type", ""),
        "node_id": getattr(node, "id", None),
        "score": float(score),
    }


# --------------------------------------------------------------------------
class JEPANeedleAgent(NeedleAgentGroup):
    """A :class:`NeedleAgentGroup` backed by JEPA-GraphRAG retrieval.

    Behaves exactly like :class:`NeedleAgentGroup` (same ingest / export /
    train / load / run workflow) but its graph-backed tools route through a
    shared :class:`~ai_memory.jepa_graphrag.JEPAGraphRAG` instance, giving the
    Needle agent local / global / latent / hybrid retrieval and
    community-level tools.

    Parameters
    ----------
    Same as :class:`NeedleAgentGroup`, plus:

    latent_dim:
        Dimensionality of the Graph-JEPA latent space (default 128).
    use_torch:
        Force the torch backend for JEPA (``None`` = auto-detect).
    default_search_mode:
        Retrieval mode used by ``search_knowledge_base`` when the model does
        not specify one. One of ``local`` / ``global`` / ``latent`` /
        ``hybrid`` (default ``hybrid``).
    """

    def __init__(
        self,
        name: str,
        store=None,
        embedder=None,
        loader=None,
        system: Optional[str] = None,
        weights: Optional[str] = None,
        tool_schemas: Optional[List[Dict[str, Any]]] = None,
        latent_dim: int = 128,
        use_torch: Optional[bool] = None,
        default_search_mode: str = "hybrid",
    ) -> None:
        # Default to the JEPA-enhanced schemas unless the caller overrides.
        if tool_schemas is None:
            tool_schemas = list(JEPA_TOOL_SCHEMAS)
        super().__init__(
            name=name,
            store=store,
            embedder=embedder,
            loader=loader,
            system=system,
            weights=weights,
            tool_schemas=tool_schemas,
        )
        self.latent_dim = latent_dim
        self.use_torch = use_torch
        self.default_search_mode = default_search_mode
        self._jepa: Optional[Any] = None

    # ---------------------------------------------------------------- jepa
    @property
    def jepa(self):
        """Return the shared :class:`JEPAGraphRAG`, building it on first use.

        The instance is bound to this group's :class:`GraphStore` and
        embedder, so it always reflects the currently-ingested documents.
        """
        if self._jepa is None:
            from .jepa_graphrag import JEPAGraphRAG

            self._jepa = JEPAGraphRAG(
                store=self.store,
                embedder=self.embedder,
                latent_dim=self.latent_dim,
                use_torch=self.use_torch,
            )
        return self._jepa

    def rebuild_index(self) -> None:
        """Invalidate cached communities / latent embeddings after new ingest.

        Call this after :meth:`ingest` if you have already used the JEPA
        retriever and want it to pick up the new nodes.
        """
        if self._jepa is not None:
            self._jepa.retriever.rebuild_communities()
            self._jepa._community_embeddings = {}

    def train_world_model(
        self,
        epochs: int = 1,
        learning_rate: float = 0.001,
        ema_alpha: float = 0.99,
    ) -> List[Dict[str, float]]:
        """Self-supervised training of the Graph-JEPA world model.

        Uses each node's text as context and its neighbours' text as targets,
        which teaches the predictor to map a node into its graph
        neighbourhood in latent space. Returns the per-epoch loss dicts.

        Requires ingested documents and an embedder; safe no-op (returns an
        empty list) on an empty graph.
        """
        contexts, targets = self._build_training_pairs()
        if not contexts:
            return []

        history: List[Dict[str, float]] = []
        for _ in range(max(1, epochs)):
            loss = self.jepa.train_step(
                contexts,
                targets,
                learning_rate=learning_rate,
                ema_alpha=ema_alpha,
            )
            history.append(loss)
        # refresh cached latent community embeddings after training
        self.jepa.precompute_community_embeddings()
        return history

    def _build_training_pairs(self) -> Tuple[List[str], List[str]]:
        """Build (context_text, target_text) pairs from graph adjacency."""
        contexts: List[str] = []
        targets: List[str] = []
        for node in self.store.all_nodes():
            ctx_text = (node.properties or {}).get("text")
            if not ctx_text:
                continue
            # collect neighbour texts (out + in edges)
            neighbour_texts: List[str] = []
            for edge in self.store.edges_from(node.id):
                nb = self.store.get_node(edge.dst_id)
                if nb and (nb.properties or {}).get("text"):
                    neighbour_texts.append(nb.properties["text"])
            for edge in self.store.edges_to(node.id):
                nb = self.store.get_node(edge.src_id)
                if nb and (nb.properties or {}).get("text"):
                    neighbour_texts.append(nb.properties["text"])
            # self-target fallback keeps isolated nodes in the training set
            tgt_text = neighbour_texts[0] if neighbour_texts else ctx_text
            contexts.append(ctx_text)
            targets.append(tgt_text)
        return contexts, targets

    # ----------------------------------------------------- retrieval helpers
    def search(
        self,
        query: str,
        k: int = 5,
        mode: Optional[str] = None,
        doc_type: str = "",
    ) -> Dict[str, Any]:
        """JEPA-GraphRAG search callable directly (outside the Needle loop).

        This is the exact function the Needle ``search_knowledge_base`` tool
        wraps, exposed for programmatic use and testing.
        """
        mode = mode or self.default_search_mode
        jepa = self.jepa
        results: List[Dict[str, Any]] = []

        if mode == "local":
            for node, score in jepa.retriever.local_search(query, k=k):
                results.append(_node_to_result(node, score))
        elif mode == "global":
            for comm_id, nodes, score in jepa.retriever.global_search(query, k=k):
                for node in nodes:
                    if not (node.properties or {}).get("text"):
                        continue  # skip structural nodes (e.g. Document) w/o text
                    results.append(_node_to_result(node, score))
        elif mode == "latent":
            for node, dist in jepa.latent_search(query, k=k, mode="node"):
                # convert distance to a similarity-like score (higher = closer)
                results.append(_node_to_result(node, 1.0 / (1.0 + float(dist))))
        else:  # hybrid
            seen: set = set()
            hybrid = jepa.hybrid_search(query, k=k)
            for node, score in hybrid.get("local", []):
                if node.id not in seen:
                    seen.add(node.id)
                    results.append(_node_to_result(node, score))
            for node, dist in hybrid.get("latent_node", []):
                if node.id not in seen:
                    seen.add(node.id)
                    results.append(_node_to_result(node, 1.0 / (1.0 + float(dist))))

        if doc_type:
            results = [r for r in results if r.get("doc_type") == doc_type]
        results.sort(key=lambda r: r["score"], reverse=True)
        results = results[:k]
        return {"results": results, "count": len(results), "mode": mode}

    def search_communities(self, query: str, k: int = 3) -> Dict[str, Any]:
        """Return the top-k communities for a query with summaries."""
        jepa = self.jepa
        out: List[Dict[str, Any]] = []
        for comm_id, nodes, score in jepa.retriever.global_search(query, k=k):
            out.append(
                {
                    "community_id": int(comm_id),
                    "size": len(nodes),
                    "score": float(score),
                    "summary": jepa.retriever.get_community_summary(comm_id),
                }
            )
        return {"communities": out, "count": len(out)}

    def community_summary(self, community_id: int) -> Dict[str, Any]:
        """Return a summary string for a specific community id."""
        return {
            "community_id": int(community_id),
            "summary": self.jepa.retriever.get_community_summary(int(community_id)),
        }

    # --------------------------------------------------------- agent access
    @property
    def agent(self):
        """Build the Needle agent with JEPA-GraphRAG-backed tools.

        Raises ``ImportError`` if ``cactus-needle`` is not installed.
        """
        if self._agent is not None:
            return self._agent

        try:
            import needle  # type: ignore
        except ImportError:
            raise ImportError(
                "Needle2 support requires: pip install cactus-needle\n"
                "For GPU: pip install 'cactus-needle[train,gpu]'\n"
                "For Apple Metal: pip install 'cactus-needle[train,metal]'"
            )

        this = self
        builder = self.builder

        def search_knowledge_base(
            query: str, k: int = 5, mode: str = "", doc_type: str = ""
        ) -> Dict[str, Any]:
            """Search the knowledge base with JEPA-GraphRAG multi-mode retrieval.

            Args:
                query: natural language question or topic to search for
                k: number of results to return (1-20)
                mode: retrieval strategy: local, global, latent, or hybrid
                doc_type: restrict to a document type: pdf, csv, txt, md, docx, excel, or empty for all
            """
            return this.search(
                query, k=k, mode=(mode or None), doc_type=doc_type
            )

        def search_communities(query: str, k: int = 3) -> Dict[str, Any]:
            """Find graph communities most relevant to a query.

            Args:
                query: natural language topic to search communities for
                k: number of communities to return (1-20)
            """
            return this.search_communities(query, k=k)

        def get_community_summary(community_id: int) -> Dict[str, Any]:
            """Return a human-readable summary of a specific community.

            Args:
                community_id: integer community id from search_communities
            """
            return this.community_summary(community_id)

        def list_documents() -> Dict[str, Any]:
            """List all documents ingested into this knowledge group."""
            return {"documents": builder.list_documents()}

        def get_document_chunks(source_path: str) -> Dict[str, Any]:
            """Retrieve all text chunks from a specific document.

            Args:
                source_path: absolute file path of the document
            """
            results = builder.build_dataset(
                format="raw", source_paths=[source_path], k=1000
            )
            return {"chunks": results, "count": len(results)}

        try:
            fns = [
                needle.tool(search_knowledge_base),
                needle.tool(search_communities),
                needle.tool(get_community_summary),
                needle.tool(list_documents),
                needle.tool(get_document_chunks),
            ]
            tools = fns
        except Exception:
            tools = self.tool_schemas

        kwargs: Dict[str, Any] = {"tools": tools}
        if self.system:
            kwargs["system"] = self.system
        if self.weights:
            kwargs["weights"] = self.weights

        self._agent = needle.Needle(**kwargs)
        return self._agent

    def stats(self) -> Dict[str, Any]:
        """Return ingestion/graph stats plus JEPA configuration."""
        st = super().stats()
        st["latent_dim"] = self.latent_dim
        st["default_search_mode"] = self.default_search_mode
        st["backend"] = "torch" if self.jepa.use_torch else "numpy"
        return st

    def __repr__(self) -> str:
        s = self.builder.stats()
        return (
            f"JEPANeedleAgent(name={self.name!r}, "
            f"documents={s['documents']}, chunks={s['chunks']}, "
            f"mode={self.default_search_mode!r}, weights={self.weights!r})"
        )


# --------------------------------------------------------------------------
class JEPAOrchestrator(NeedleOrchestrator):
    """Route queries across :class:`JEPANeedleAgent` groups via JEPA-GraphRAG.

    Instead of comparing the query text-embedding against each group's *name*
    (as :class:`NeedleOrchestrator` does), this orchestrator scores every
    group by how well its **actual ingested content** matches the query,
    using each group's JEPA-GraphRAG retriever. Concretely, each group's
    routing score is the best (highest) similarity returned by
    :meth:`GraphRAGRetriever.local_search` over that group's graph — i.e. it
    routes to the group that genuinely knows the most about the query, not to
    the one whose label happens to match.

    When ``use_latent`` is enabled and two groups are within ``latent_margin``
    of each other on content score, the Graph-JEPA latent energy is used as a
    tie-breaker (smaller latent distance wins). Latent energy is only a
    tie-breaker, never the primary signal, because an under-trained world
    model carries little discriminative signal and a per-group distance bias.

    If no group returns any content match, the orchestrator falls back to the
    base-class text-embedding router.

    Parameters
    ----------
    groups:
        List of :class:`JEPANeedleAgent` instances.
    embedder:
        Shared embedder for the fallback router. Defaults to LocalEmbedder.
    use_latent:
        Enable JEPA latent energy as a tie-breaker (default ``True``).
    latent_margin:
        Content-score gap below which the latent tie-breaker kicks in
        (default ``0.05``).
    """

    def __init__(
        self,
        groups: Optional[List["JEPANeedleAgent"]] = None,
        embedder=None,
        use_latent: bool = True,
        latent_margin: float = 0.05,
    ) -> None:
        super().__init__(groups=groups, embedder=embedder)
        self.use_latent = use_latent
        self.latent_margin = latent_margin

    def _content_score(self, group: NeedleAgentGroup, query: str) -> float:
        """Best JEPA-GraphRAG content-relevance score for a query in a group."""
        if not isinstance(group, JEPANeedleAgent):
            return -1.0
        try:
            results = group.jepa.retriever.local_search(query, k=1)
        except Exception:
            return -1.0
        return float(results[0][1]) if results else -1.0

    def _latent_distance(self, group: NeedleAgentGroup, query: str) -> float:
        """Best Graph-JEPA latent distance for a query in a group (smaller=closer)."""
        if not isinstance(group, JEPANeedleAgent):
            return float("inf")
        try:
            results = group.jepa.latent_search(query, k=1, mode="node")
        except Exception:
            return float("inf")
        return float(results[0][1]) if results else float("inf")

    def route(self, query: str) -> Optional[NeedleAgentGroup]:
        """Return the best group for ``query`` by JEPA-GraphRAG content relevance.

        Uses latent energy only to break near-ties, and falls back to the
        base-class text-embedding router when no group has matching content.
        """
        if not self.groups:
            return None
        if len(self.groups) == 1:
            return self.groups[0]

        scored: List[Tuple[NeedleAgentGroup, float]] = [
            (g, self._content_score(g, query)) for g in self.groups
        ]
        # keep only groups that returned a real content match
        usable = [(g, s) for g, s in scored if s >= 0.0]
        if not usable:
            return super().route(query)

        usable.sort(key=lambda gs: gs[1], reverse=True)
        best_group, best_score = usable[0]

        if self.use_latent:
            # gather groups within latent_margin of the top content score
            contenders = [
                g for g, s in usable if (best_score - s) <= self.latent_margin
            ]
            if len(contenders) > 1:
                best_group = min(
                    contenders, key=lambda g: self._latent_distance(g, query)
                )

        return best_group

    def run(self, query: str, max_steps: int = 8) -> Dict[str, Any]:
        """Route ``query`` to the best group and run its Needle agent.

        Adds a ``"routed_to"`` key with the selected group name.
        """
        group = self.route(query)
        if group is None:
            return {"type": "error", "error": "No groups registered"}
        result = dict(group.run(query, max_steps=max_steps))
        result["routed_to"] = group.name
        return result


__all__ = [
    "JEPA_TOOL_SCHEMAS",
    "JEPANeedleAgent",
    "JEPAOrchestrator",
]
