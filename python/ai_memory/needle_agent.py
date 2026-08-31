"""Needle2 (cactus-needle) integration for the AI-GraphDB-Engine.

`Needle2 <https://pypi.org/project/cactus-needle/>`_ is a compact embedded
function-calling model — not a chat LLM.  Every response is a structured JSON
tool call (or ``[]`` for off-topic queries).  It is fine-tuned via LoRA
adapters on JSONL data and exported to a ``.cact`` archive.

This module bridges Needle2 with the graph knowledge base and provides:

* :data:`GRAPHDB_TOOL_SCHEMAS` — three Needle-compatible tool schemas backed
  by the graph (``search_knowledge_base``, ``list_documents``,
  ``get_document_chunks``).
* :class:`NeedleAgentGroup` — a named, trainable knowledge group with its own
  graph partition, :class:`~ai_memory.dataset_builder.DatasetBuilder`, and
  (after weights are loaded) a ``needle.Needle`` agent whose tools call
  directly back into the graph.
* :class:`NeedleOrchestrator` — routes queries across multiple
  :class:`NeedleAgentGroup` instances, each specialising in its own document
  domain.

The ingest / export / routing / stats surface works **without**
``cactus-needle`` installed.  Only live inference (:attr:`NeedleAgentGroup.agent`,
:meth:`NeedleAgentGroup.run`) requires ``pip install cactus-needle``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------
# Built-in GraphDB-backed Needle tool schemas
# --------------------------------------------------------------------------
GRAPHDB_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the knowledge base for passages relevant to a query. "
            "Returns the top-k most semantically similar text chunks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "natural language question or topic to search for",
                    "minLength": 1,
                    "maxLength": 500,
                },
                "k": {
                    "type": "integer",
                    "description": "number of results to return",
                    "minimum": 1,
                    "maximum": 20,
                },
                "doc_type": {
                    "type": "string",
                    "description": (
                        "restrict results to a document type: pdf, csv, txt, "
                        "md, docx, excel, or empty string for all"
                    ),
                    "enum": ["", "pdf", "csv", "txt", "md", "docx", "excel"],
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_documents",
        "description": (
            "List all documents ingested into this knowledge group, including "
            "filename, document type, and chunk count."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_document_chunks",
        "description": (
            "Retrieve all text chunks from a specific document by its source path."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source_path": {
                    "type": "string",
                    "description": (
                        "absolute file path of the document as returned by "
                        "list_documents"
                    ),
                },
            },
            "required": ["source_path"],
        },
    },
]


# --------------------------------------------------------------------------
class NeedleAgentGroup:
    """A named, trainable knowledge group backed by the graph and Needle2.

    Each group has its own ``GraphStore`` partition, ``DatasetBuilder``, and
    (after weights are loaded) a ``needle.Needle`` agent whose tools call
    directly back into the graph.

    Typical workflow
    ----------------
    1. Ingest documents::

           group = NeedleAgentGroup("legal")
           group.ingest(["contract.pdf", "policy.docx"])

    2. Export Needle training data::

           group.export_training_data("legal_train.jsonl")

    3. Fine-tune Needle outside Python::

           needle finetune legal_train.jsonl --epochs 20 --out adapter.pkl
           needle build checkpoints/needle2.pkl --lora adapter.pkl --out legal.cact

    4. Load weights and query::

           group.load_weights("legal.cact")
           result = group.run("What are the termination clauses?")

    Parameters
    ----------
    name:
        Logical name for this knowledge group (e.g. ``"legal"``, ``"tech"``).
    store:
        GraphStore instance. Defaults to a fresh in-memory store.
    embedder:
        Text embedder. Defaults to LocalEmbedder.
    loader:
        DocumentLoader. Defaults to DocumentLoader().
    system:
        System string injected into Needle at construction and into every
        training example, e.g. ``"knowledge_group: legal; date: 2026-08-31"``.
    weights:
        Path to a pre-built ``.cact`` archive to load immediately.
    tool_schemas:
        Override the default ``GRAPHDB_TOOL_SCHEMAS`` with custom ones.
        Must be Needle-compatible JSON schema dicts (max 5).
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
    ) -> None:
        from graphdb.store import GraphStore
        from .embedder import LocalEmbedder
        from .document_loader import DocumentLoader
        from .dataset_builder import DatasetBuilder

        self.name = name
        self.store = store if store is not None else GraphStore()
        self.embedder = embedder if embedder is not None else LocalEmbedder()
        self.loader = loader if loader is not None else DocumentLoader()
        self.builder = DatasetBuilder(
            store=self.store,
            embedder=self.embedder,
            loader=self.loader,
        )
        self.system = system or f"knowledge_group: {name}"
        self.weights = str(weights) if weights else None
        self.tool_schemas = (
            tool_schemas if tool_schemas is not None else list(GRAPHDB_TOOL_SCHEMAS)
        )
        self._agent = None  # built lazily

    # ---------------------------------------------------------------- ingest
    def ingest(self, paths: List[str]):
        """Ingest documents into this group's graph store.

        Returns the IngestResult from DatasetBuilder.ingest().
        """
        return self.builder.ingest(paths)

    # -------------------------------------------------------- training data
    def export_training_data(
        self,
        output_path: str,
        k: int = 500,
        off_topic_ratio: float = 0.125,
    ) -> str:
        """Export Needle2 fine-tuning JSONL for this group.

        Parameters
        ----------
        output_path:
            Where to write the JSONL file.
        k:
            Maximum number of positive training examples.
        off_topic_ratio:
            Fraction of total examples that are off-topic (no tool call).
            Needle requires ~1-in-8 (0.125) to prevent the model from calling
            a tool on everything.

        Returns
        -------
        Absolute path of the written file.
        """
        dataset = self.builder.build_needle_dataset(
            tools=self.tool_schemas,
            k=k,
            off_topic_ratio=off_topic_ratio,
            system=self.system,
        )
        return self.builder.export(dataset, output_path, file_format="jsonl")

    # ------------------------------------------------------------ weights
    def load_weights(self, weights_path: str) -> None:
        """Load a fine-tuned ``.cact`` archive.

        Resets the cached agent so it is rebuilt with the new weights on
        the next call to :attr:`agent` or :meth:`run`.

        Parameters
        ----------
        weights_path:
            Path to the ``.cact`` file produced by ``needle build``.
        """
        self.weights = str(weights_path) if weights_path else None
        self._agent = None  # invalidate cached agent

    # --------------------------------------------------------- agent access
    @property
    def agent(self):
        """Return the ``needle.Needle`` agent, constructing it on first access.

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

        builder = self.builder

        # ---- define GraphDB-backed tool functions ----

        def search_knowledge_base(
            query: str, k: int = 5, doc_type: str = ""
        ) -> Dict[str, Any]:
            """Search the knowledge base for passages relevant to a query.

            Args:
                query: natural language question or topic to search for
                k: number of results to return (1-20)
                doc_type: restrict results to a document type: pdf, csv, txt, md, docx, excel, or empty string for all
            """
            results = builder.build_dataset(format="raw", query=query, k=k)
            if doc_type:
                results = [r for r in results if r.get("doc_type") == doc_type]
            return {"results": results[:k], "count": len(results[:k])}

        def list_documents() -> Dict[str, Any]:
            """List all documents ingested into this knowledge group."""
            return {"documents": builder.list_documents()}

        def get_document_chunks(source_path: str) -> Dict[str, Any]:
            """Retrieve all text chunks from a specific document by its source path.

            Args:
                source_path: absolute file path of the document as returned by list_documents
            """
            results = builder.build_dataset(
                format="raw",
                source_paths=[source_path],
                k=1000,
            )
            return {"chunks": results, "count": len(results)}

        # Decorate with needle.tool if the decorator is available
        try:
            search_knowledge_base = needle.tool(search_knowledge_base)
            list_documents = needle.tool(list_documents)
            get_document_chunks = needle.tool(get_document_chunks)
            tools = [search_knowledge_base, list_documents, get_document_chunks]
        except Exception:
            # Fall back to raw schemas if decorator fails (e.g. version mismatch)
            tools = self.tool_schemas

        kwargs: Dict[str, Any] = {"tools": tools}
        if self.system:
            kwargs["system"] = self.system
        if self.weights:
            kwargs["weights"] = self.weights

        self._agent = needle.Needle(**kwargs)
        return self._agent

    def run(self, query: str, max_steps: int = 8) -> Dict[str, Any]:
        """Run a query through the Needle agent's full agentic loop.

        Parameters
        ----------
        query:
            Natural language question to answer using the knowledge base.
        max_steps:
            Maximum number of tool-call / feed-back cycles.

        Returns
        -------
        The raw Needle response dict (``type``, ``function_calls``,
        ``reasoning``, ``confidence``, ``results``, etc.).
        """
        return self.agent.run(query, max_steps=max_steps)

    def complete(self, text: str) -> Dict[str, Any]:
        """Single-turn completion — returns the raw Needle response.

        Use this when you want to drive the tool-call loop yourself instead
        of letting :meth:`run` handle it end-to-end.
        """
        return self.agent.complete(text)

    def reset(self) -> None:
        """Rewind the agent's conversation history, keeping tools loaded."""
        if self._agent is not None:
            self._agent.reset()

    # ------------------------------------------------------------- stats
    def stats(self) -> Dict[str, Any]:
        """Return ingestion and graph statistics for this group."""
        st = self.builder.stats()
        st["group_name"] = self.name
        st["system"] = self.system
        st["weights"] = self.weights
        st["tool_count"] = len(self.tool_schemas)
        return st

    def __repr__(self) -> str:
        st = self.builder.stats()
        return (
            f"NeedleAgentGroup(name={self.name!r}, "
            f"documents={st['documents']}, chunks={st['chunks']}, "
            f"weights={self.weights!r})"
        )


# --------------------------------------------------------------------------
class NeedleOrchestrator:
    """Route queries across multiple NeedleAgentGroups.

    Each group specialises in its own document domain. The orchestrator
    picks the best group for a query by embedding the query and comparing
    it to each group's name/system string, then delegates to that group's
    Needle agent.

    Parameters
    ----------
    groups:
        List of :class:`NeedleAgentGroup` instances.
    embedder:
        Shared embedder for routing. Defaults to LocalEmbedder.
    """

    def __init__(
        self,
        groups: Optional[List["NeedleAgentGroup"]] = None,
        embedder=None,
    ) -> None:
        from .embedder import LocalEmbedder

        self.groups: List[NeedleAgentGroup] = list(groups or [])
        self.embedder = embedder if embedder is not None else LocalEmbedder()

    def add_group(self, group: "NeedleAgentGroup") -> None:
        """Register a new knowledge group."""
        self.groups.append(group)

    def route(self, query: str) -> Optional["NeedleAgentGroup"]:
        """Return the best group for ``query`` by cosine similarity to group names.

        Falls back to the first group if routing fails.
        """
        if not self.groups:
            return None
        if len(self.groups) == 1:
            return self.groups[0]

        import numpy as np  # type: ignore

        q_vec = self.embedder.embed(query)
        best_group = self.groups[0]
        best_score = -1.0

        for group in self.groups:
            label = f"{group.name} {group.system}"
            g_vec = self.embedder.embed(label)
            q_arr = np.array(q_vec)
            g_arr = np.array(g_vec)
            denom = np.linalg.norm(q_arr) * np.linalg.norm(g_arr)
            score = float(np.dot(q_arr, g_arr) / denom) if denom > 0 else 0.0
            if score > best_score:
                best_score = score
                best_group = group

        return best_group

    def run(self, query: str, max_steps: int = 8) -> Dict[str, Any]:
        """Route ``query`` to the best group and run its Needle agent.

        Returns the Needle response dict plus a ``"routed_to"`` key with
        the group name.
        """
        group = self.route(query)
        if group is None:
            return {"type": "error", "error": "No groups registered"}
        result = group.run(query, max_steps=max_steps)
        result = dict(result)
        result["routed_to"] = group.name
        return result

    def export_all_training_data(
        self,
        output_dir: str,
        k: int = 500,
        off_topic_ratio: float = 0.125,
    ) -> Dict[str, str]:
        """Export Needle training JSONL for every group.

        Parameters
        ----------
        output_dir:
            Directory to write ``<group_name>_train.jsonl`` files into.
        k:
            Max positive examples per group.
        off_topic_ratio:
            Off-topic ratio for each group.

        Returns
        -------
        Dict mapping group name → written file path.
        """
        from pathlib import Path

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        paths: Dict[str, str] = {}
        for group in self.groups:
            out = str(Path(output_dir) / f"{group.name}_train.jsonl")
            paths[group.name] = group.export_training_data(
                out, k=k, off_topic_ratio=off_topic_ratio
            )
        return paths

    def stats(self) -> Dict[str, Any]:
        """Return stats for all groups."""
        return {g.name: g.stats() for g in self.groups}

    def __repr__(self) -> str:
        names = [g.name for g in self.groups]
        return f"NeedleOrchestrator(groups={names})"


__all__ = [
    "GRAPHDB_TOOL_SCHEMAS",
    "NeedleAgentGroup",
    "NeedleOrchestrator",
]
