"""Training dataset generator backed by the AI-GraphDB-Engine.

Pipeline
--------
1. **Ingest** local documents (PDF, DOCX, TXT, MD, CSV, XLSX) via
   :class:`~ai_memory.document_loader.DocumentLoader`.
   Each document becomes a ``Document`` graph node; each text chunk becomes a
   ``Chunk`` node with a vector embedding; the two are linked by a
   ``CHUNK_OF`` edge.  Adjacent chunks are linked by ``NEXT_CHUNK`` edges so
   the ordering is preserved and retrievable.

2. **Build a dataset** from the stored chunks. Chunks are retrieved either by
   vector search (``query`` param) or by scanning all ``Chunk`` nodes.
   Each chunk is turned into one or more training examples in the requested
   format.

3. **Export** the dataset to disk as JSONL, CSV, or (optionally) Parquet.

Supported output formats
------------------------
* ``"raw"``        — ``{text, source, chunk_index, page, section, doc_type}``
* ``"completion"`` — ``{prompt, completion}``  (classic fine-tuning)
* ``"qa"``         — ``{question, answer, context, source}``
* ``"alpaca"``     — ``{instruction, input, output}``  (Alpaca/Llama)
* ``"openai"``     — ``{messages: [{role, content}, ...]}``  (OpenAI chat)

Q&A pair generation
-------------------
Questions are generated either:

* **Offline** (default): a fast, deterministic heuristic — extracts key noun
  phrases from the first sentence and produces template questions.  No model
  needed; runs in pure Python.
* **Ollama** (optional): pass ``qa_generator=ollama_qa_generator()`` to
  generate natural questions with a local LLM.

Usage example
-------------
::

    from ai_memory.dataset_builder import DatasetBuilder
    from ai_memory.embedder import get_embedder
    from graphdb.store import GraphStore

    store = GraphStore(path="docs.json")
    builder = DatasetBuilder(store=store, embedder=get_embedder(prefer_ollama=True))

    result = builder.ingest(["paper.pdf", "notes.md", "data.csv"])
    print(result)

    dataset = builder.build_dataset(format="alpaca", k=500)
    builder.export(dataset, "train.jsonl")
"""

from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from graphdb.store import GraphStore, Node, Edge

from .document_loader import Chunk, DocumentLoader
from .embedder import Embedder, LocalEmbedder


# -------------------------------------------------------------------- labels
DOCUMENT = "Document"
CHUNK = "Chunk"
CHUNK_OF = "CHUNK_OF"
NEXT_CHUNK = "NEXT_CHUNK"


# ----------------------------------------------------------------- dataclasses
@dataclass
class IngestResult:
    """Summary returned after ingesting one or more documents."""

    documents: int = 0
    chunks_stored: int = 0
    chunks_skipped: int = 0
    files: List[str] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        ok = ", ".join(Path(f).name for f in self.files) or "(none)"
        err = (
            "; ".join(f"{Path(k).name}: {v}" for k, v in self.errors.items())
            or "(none)"
        )
        return (
            f"IngestResult(docs={self.documents}, chunks={self.chunks_stored}, "
            f"skipped={self.chunks_skipped})\n"
            f"  files  : {ok}\n"
            f"  errors : {err}"
        )


@dataclass
class DatasetEntry:
    """A single training example in a format-agnostic container."""

    text: str                              # raw chunk text (always available)
    source: str                            # source file path
    chunk_index: int
    page: Optional[int] = None
    section: Optional[str] = None
    doc_type: str = ""
    question: Optional[str] = None        # populated for qa / alpaca / openai
    answer: Optional[str] = None          # populated for qa / alpaca / openai


# ---------------------------------------------------------------- Q&A helpers
def _offline_question(text: str) -> str:
    """Deterministic heuristic to generate a question from a text chunk.

    Strategy:
    1. Take the first sentence (up to the first period / newline).
    2. Strip trailing words to form a cloze stem.
    3. Build a "What does the document say about <topic>?" question.
    """
    # First sentence.
    first = re.split(r"[.\n]", text.strip())[0].strip()
    if not first:
        first = text[:100].strip()

    # Extract the first noun phrase — capitalised run or after a verb.
    caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", first)
    loww = re.findall(r"\b(?:the|a|an)\s+(\w+(?:\s+\w+){0,2})", first, re.I)

    if caps:
        topic = caps[0]
    elif loww:
        topic = loww[0]
    else:
        # Fall back: use the first 4 words.
        words = first.split()[:4]
        topic = " ".join(words)

    topic = topic.strip().rstrip(".,;:")
    return f"What does the document say about {topic}?"


def ollama_qa_generator(
    model: str = "llama3.1",
    host: str = "http://localhost:11434",
) -> Callable[[str], str]:
    """Return a Q&A generator backed by a local Ollama chat model.

    The returned callable accepts a chunk of text and returns a question whose
    answer is contained in that chunk.

    Usage::

        gen = ollama_qa_generator(model="llama3.2:3b")
        builder = DatasetBuilder(store, embedder, qa_generator=gen)
    """
    from .embedder import ollama_chat  # imported lazily

    def generate(text: str) -> str:
        prompt = (
            "Generate one clear, specific question whose answer is directly "
            "contained in the following text. Output ONLY the question, "
            "nothing else.\n\nText:\n" + text[:1200]
        )
        q = ollama_chat(
            prompt,
            system=(
                "You are a training-data expert. "
                "Generate precise questions for fine-tuning LLMs."
            ),
            model=model,
            host=host,
        )
        return q.strip().strip('"').strip("'")

    return generate


# --------------------------------------------------------------- main class
class DatasetBuilder:
    """Ingest documents into the graph and generate training datasets.

    Parameters
    ----------
    store:
        The :class:`~graphdb.store.GraphStore` to use for persistence.
    embedder:
        Text-to-vector embedder (defaults to :class:`~ai_memory.embedder.LocalEmbedder`).
    loader:
        :class:`~ai_memory.document_loader.DocumentLoader` instance.
        Defaults to one with ``chunk_size=800``.
    qa_generator:
        Callable ``(chunk_text) -> question_string``.
        Defaults to the offline heuristic.  Pass :func:`ollama_qa_generator`
        for LLM-generated questions.
    system_prompt:
        System prompt injected into OpenAI-format entries.
    """

    def __init__(
        self,
        store: Optional[GraphStore] = None,
        embedder: Optional[Embedder] = None,
        loader: Optional[DocumentLoader] = None,
        qa_generator: Optional[Callable[[str], str]] = None,
        system_prompt: str = (
            "You are a knowledgeable assistant. "
            "Answer based on the provided context."
        ),
    ) -> None:
        self.store = store if store is not None else GraphStore()
        self.embedder = embedder if embedder is not None else LocalEmbedder()
        self.loader = loader if loader is not None else DocumentLoader()
        self.qa_generator = qa_generator or _offline_question
        self.system_prompt = system_prompt

    # ---------------------------------------------------------------- ingest
    def ingest(self, paths: List[str]) -> IngestResult:
        """Parse documents and store them as ``Document`` + ``Chunk`` nodes.

        Already-ingested documents (same absolute path) are **skipped** to
        avoid duplicates.  Delete the ``Document`` node first if you want to
        re-ingest.

        Parameters
        ----------
        paths:
            List of file paths to ingest.

        Returns
        -------
        :class:`IngestResult`
        """
        result = IngestResult()
        for path in paths:
            abs_path = str(Path(path).resolve())
            try:
                self._ingest_one(abs_path, result)
            except Exception as exc:
                result.errors[abs_path] = str(exc)
        return result

    def _ingest_one(self, path: str, result: IngestResult) -> None:
        # Skip if already ingested.
        existing = self.store.find_nodes(DOCUMENT, source_path=path)
        if existing:
            result.chunks_skipped += len(
                self.store.edges_from(existing[0].id)
            )
            return

        chunks = self.loader.load(path)
        if not chunks:
            return

        # Create document node.
        doc_node = self.store.add_node(
            Node(
                label=DOCUMENT,
                properties={
                    "source_path": path,
                    "filename": Path(path).name,
                    "doc_type": chunks[0].doc_type if chunks else "",
                    "num_chunks": len(chunks),
                    "ingested_at": time.time(),
                },
            )
        )

        prev_chunk_id: Optional[str] = None
        for chunk in chunks:
            vec = self.embedder.embed(chunk.text)
            props: Dict[str, Any] = {
                "text": chunk.text,
                "chunk_index": chunk.chunk_index,
                "source_path": path,
                "doc_type": chunk.doc_type,
            }
            if chunk.page is not None:
                props["page"] = chunk.page
            if chunk.section is not None:
                props["section"] = chunk.section
            if chunk.row is not None:
                props["row"] = chunk.row

            chunk_node = self.store.add_node(
                Node(label=CHUNK, properties=props, embedding=vec)
            )

            # Link chunk → document.
            self.store.add_edge(
                Edge(src_id=chunk_node.id, dst_id=doc_node.id, label=CHUNK_OF)
            )

            # Link to previous chunk for ordering.
            if prev_chunk_id:
                self.store.add_edge(
                    Edge(
                        src_id=prev_chunk_id,
                        dst_id=chunk_node.id,
                        label=NEXT_CHUNK,
                        weight=1.0,
                    )
                )
            prev_chunk_id = chunk_node.id
            result.chunks_stored += 1

        result.documents += 1
        result.files.append(path)

    # --------------------------------------------------------------- build
    def build_dataset(
        self,
        format: str = "qa",
        source_paths: Optional[List[str]] = None,
        query: Optional[str] = None,
        k: int = 200,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate a training dataset from ingested chunks.

        Parameters
        ----------
        format:
            One of ``"raw"``, ``"completion"``, ``"qa"``, ``"alpaca"``,
            ``"openai"``.
        source_paths:
            If set, restrict to chunks from these source files.
        query:
            If set, retrieve the ``k`` most relevant chunks via vector search.
            If ``None``, scan all ``Chunk`` nodes (up to ``k``).
        k:
            Maximum number of training examples to produce.
        system_prompt:
            Override the default system prompt (``"openai"`` format only).

        Returns
        -------
        List[dict] — one dict per training example, ready to serialise.
        """
        sys_prompt = system_prompt or self.system_prompt
        entries = self._gather_entries(source_paths, query, k)
        fmt = format.lower().strip()
        formatters = {
            "raw": self._fmt_raw,
            "completion": self._fmt_completion,
            "qa": self._fmt_qa,
            "alpaca": self._fmt_alpaca,
            "openai": lambda e: self._fmt_openai(e, sys_prompt),
        }
        if fmt not in formatters:
            raise ValueError(
                f"Unknown format {fmt!r}. Choose one of: {list(formatters)}"
            )
        fn = formatters[fmt]
        return [fn(e) for e in entries]

    # --------------------------------------------------------------- export
    def export(
        self,
        dataset: List[Dict[str, Any]],
        output_path: str,
        file_format: str = "jsonl",
    ) -> str:
        """Write ``dataset`` to disk.

        Parameters
        ----------
        dataset:
            Output of :meth:`build_dataset`.
        output_path:
            Destination file path.
        file_format:
            ``"jsonl"`` (default), ``"csv"``, or ``"parquet"``.

        Returns
        -------
        Absolute path of the written file.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fmt = file_format.lower().strip()

        if fmt == "jsonl":
            with open(output_path, "w", encoding="utf-8") as fh:
                for row in dataset:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        elif fmt == "csv":
            if not dataset:
                Path(output_path).write_text("", encoding="utf-8")
                return str(Path(output_path).resolve())
            # Flatten nested structures (e.g. openai messages list).
            flat = [_flatten(row) for row in dataset]
            keys = list(flat[0].keys())
            with open(output_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(flat)

        elif fmt == "parquet":
            try:
                import pandas as pd  # type: ignore
            except ImportError:
                raise ImportError(
                    "Parquet export requires pandas: pip install pandas pyarrow"
                )
            flat = [_flatten(row) for row in dataset]
            pd.DataFrame(flat).to_parquet(output_path, index=False)

        else:
            raise ValueError(f"Unknown file_format {fmt!r}. Use jsonl / csv / parquet.")

        return str(Path(output_path).resolve())

    # --------------------------------------------------------- stats helpers
    def stats(self) -> Dict[str, Any]:
        """Return ingestion statistics from the graph."""
        docs = self.store.nodes_by_label(DOCUMENT)
        chunks = self.store.nodes_by_label(CHUNK)
        by_type: Dict[str, int] = {}
        by_doc: Dict[str, int] = {}
        for c in chunks:
            dt = str(c.properties.get("doc_type", "unknown"))
            by_type[dt] = by_type.get(dt, 0) + 1
            src = str(Path(c.properties.get("source_path", "?")).name)
            by_doc[src] = by_doc.get(src, 0) + 1
        return {
            "documents": len(docs),
            "chunks": len(chunks),
            "chunks_by_type": by_type,
            "chunks_by_document": by_doc,
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all ingested documents with metadata."""
        return [
            {
                "filename": n.properties.get("filename"),
                "source_path": n.properties.get("source_path"),
                "doc_type": n.properties.get("doc_type"),
                "num_chunks": n.properties.get("num_chunks"),
                "ingested_at": n.properties.get("ingested_at"),
            }
            for n in self.store.nodes_by_label(DOCUMENT)
        ]

    # --------------------------------------------------------------- private
    def _gather_entries(
        self,
        source_paths: Optional[List[str]],
        query: Optional[str],
        k: int,
    ) -> List[DatasetEntry]:
        abs_paths = (
            {str(Path(p).resolve()) for p in source_paths}
            if source_paths
            else None
        )

        if query:
            vec = self.embedder.embed(query)
            hits = self.store.search_similar_nodes(vec, label=CHUNK, k=k * 3)
            nodes = [n for n, _ in hits]
        else:
            nodes = self.store.nodes_by_label(CHUNK)

        entries: List[DatasetEntry] = []
        for node in nodes:
            if abs_paths and node.properties.get("source_path") not in abs_paths:
                continue
            text = str(node.properties.get("text", "")).strip()
            if not text:
                continue
            q = self.qa_generator(text)
            entries.append(
                DatasetEntry(
                    text=text,
                    source=str(node.properties.get("source_path", "")),
                    chunk_index=int(node.properties.get("chunk_index", 0)),
                    page=node.properties.get("page"),  # type: ignore[arg-type]
                    section=node.properties.get("section"),  # type: ignore[arg-type]
                    doc_type=str(node.properties.get("doc_type", "")),
                    question=q,
                    answer=text,
                )
            )
            if len(entries) >= k:
                break

        return entries

    # ---------------------------------------------------------- formatters
    @staticmethod
    def _fmt_raw(e: DatasetEntry) -> Dict[str, Any]:
        return {
            "text": e.text,
            "source": e.source,
            "chunk_index": e.chunk_index,
            "page": e.page,
            "section": e.section,
            "doc_type": e.doc_type,
        }

    @staticmethod
    def _fmt_completion(e: DatasetEntry) -> Dict[str, Any]:
        prompt = (
            f"Based on the following document excerpt, "
            f"continue or summarise the content:\n\n{e.text[:400]}"
        )
        return {"prompt": prompt, "completion": e.text}

    @staticmethod
    def _fmt_qa(e: DatasetEntry) -> Dict[str, Any]:
        return {
            "question": e.question,
            "answer": e.answer,
            "context": e.text,
            "source": e.source,
            "page": e.page,
            "section": e.section,
        }

    @staticmethod
    def _fmt_alpaca(e: DatasetEntry) -> Dict[str, Any]:
        return {
            "instruction": e.question or "Summarise the following excerpt.",
            "input": e.text,
            "output": e.answer or e.text,
        }

    @staticmethod
    def _fmt_openai(e: DatasetEntry, system_prompt: str) -> Dict[str, Any]:
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": e.question or e.text},
                {"role": "assistant", "content": e.answer or e.text},
            ]
        }


# --------------------------------------------------------------- utilities
def _flatten(d: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested dicts/lists to strings for CSV export."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (dict, list)):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = v
    return out


__all__ = [
    "DOCUMENT",
    "CHUNK",
    "CHUNK_OF",
    "NEXT_CHUNK",
    "IngestResult",
    "DatasetEntry",
    "DatasetBuilder",
    "ollama_qa_generator",
]
