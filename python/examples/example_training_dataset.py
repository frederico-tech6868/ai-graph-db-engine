"""Full pipeline: local documents → graph → training dataset.

What this example does
----------------------
1. Creates synthetic local documents (TXT, MD, CSV) so the example is
   self-contained and runnable without any real files.
2. Ingests them into the graph via :class:`~ai_memory.dataset_builder.DatasetBuilder`.
3. Builds training datasets in all five supported formats:
   raw / completion / qa / alpaca / openai.
4. Exports each format to JSONL and shows a sample entry.
5. Demonstrates query-scoped retrieval (vector search over chunks).
6. Prints a full pipeline summary.

Run it
------
::

    python examples/example_training_dataset.py

With Ollama (better Q&A quality)::

    USE_OLLAMA=1 python examples/example_training_dataset.py

Point it at your own files::

    from ai_memory.dataset_builder import DatasetBuilder, ollama_qa_generator
    from ai_memory.embedder import get_embedder
    from graphdb.store import GraphStore

    builder = DatasetBuilder(
        store=GraphStore(path="my_docs.json"),
        embedder=get_embedder(prefer_ollama=True),
        qa_generator=ollama_qa_generator(),   # optional: LLM questions
    )
    result = builder.ingest(["docs/paper.pdf", "docs/notes.md", "data/metrics.csv"])
    print(result)

    dataset = builder.build_dataset(format="alpaca", k=1000)
    builder.export(dataset, "train_alpaca.jsonl")
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ai_memory.dataset_builder import DatasetBuilder, ollama_qa_generator  # noqa: E402
from ai_memory.embedder import LocalEmbedder, OllamaEmbedder              # noqa: E402
from graphdb.store import GraphStore                                        # noqa: E402


# --------------------------------------------------------------------------
# 0. Synthetic documents
# --------------------------------------------------------------------------
TXT_CONTENT = """\
Introduction to Graph Databases

A graph database stores data as nodes and edges rather than rows and tables.
Nodes represent entities such as people, places, and concepts.
Edges represent relationships between those entities.

Graph databases excel at traversing complex relationships quickly.
Traditional relational databases require expensive JOIN operations for the same task.
Graph queries run in constant time regardless of the total dataset size.

Vector Search in Graphs

Label-scoped vector search restricts similarity lookups to a specific node type.
This prevents cross-type false matches between, for example, memory nodes and entity nodes.
Each node stores an embedding vector alongside its properties.
Cosine similarity is computed against stored vectors to rank results.

Persistence and Storage

The graph engine persists all nodes and edges as JSON on disk.
On startup the engine loads the graph file and rebuilds all indexes automatically.
Atomic writes (write to .tmp then rename) prevent data corruption on crash.
Embeddings are stored as JSON float arrays alongside each node.
"""

MD_CONTENT = """\
# Agent Memory SDK

## AgentMemory

`AgentMemory` is the high-level interface for storing and retrieving facts.

### remember()

Call `remember(text, memory_type)` to store a fact as a `Memory` node.
The text is embedded and stored in the graph with a timestamp.
Duplicate detection uses cosine similarity with a threshold of 0.95.

### recall()

Call `recall(query, k=5)` to retrieve the most relevant memories.
Results are ranked by embedding cosine similarity to the query vector.
Pass `memory_type` to scope the search to a specific memory kind.

### reflect()

Call `reflect(recent_k=20)` to synthesise a higher-level summary.
The reflection is stored back into the graph as a `REFLECTION` memory.
Key entities mentioned in recent memories are counted and ranked.

## ContextManager

`ContextManager` keeps an LLM within its context window.
It budgets the window across system prompt, retrieved memories, and live transcript.
Overflow turns are summarised and written back to the graph as memories.
The assembled prompt is guaranteed to be within the model's `num_ctx` limit.
"""

CSV_CONTENT = """\
model,context_window,open_source,recommended_use
llama3.2:1b,16384,yes,fast on-device inference
llama3.2:3b,32768,yes,balanced local model
llama3.1:8b,65536,yes,default Ollama choice
llama3.1:70b,131072,yes,high-quality local generation
phi3:mini,4096,yes,ultra-lightweight embedding host
nomic-embed-text,2048,yes,best local embedding model
mistral:7b,32768,yes,European open weights model
gemma2:9b,8192,yes,Google open weights model
"""


def create_synthetic_docs(tmp_dir: str) -> list[str]:
    paths = []
    for name, content in [
        ("graphdb_overview.txt", TXT_CONTENT),
        ("sdk_reference.md", MD_CONTENT),
        ("ollama_models.csv", CSV_CONTENT),
    ]:
        p = Path(tmp_dir) / name
        p.write_text(content, encoding="utf-8")
        paths.append(str(p))
    return paths


# --------------------------------------------------------------------------
# 1. Pick embedder
# --------------------------------------------------------------------------
def pick_embedder():
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        emb = OllamaEmbedder(host=host)
        print(f"[embedder] Ollama at {host}")
        return emb
    except Exception:
        print("[embedder] Offline — using LocalEmbedder")
        return LocalEmbedder()


# --------------------------------------------------------------------------
# 2. Main
# --------------------------------------------------------------------------
def main() -> None:
    print("=" * 72)
    print("  AI-GraphDB-Engine  ·  Local Documents → Training Dataset")
    print("=" * 72)

    embedder = pick_embedder()

    with tempfile.TemporaryDirectory() as tmp_dir:
        doc_paths = create_synthetic_docs(tmp_dir)
        out_dir = Path(tmp_dir) / "datasets"
        out_dir.mkdir()

        store = GraphStore()
        builder = DatasetBuilder(store=store, embedder=embedder)

        # ------------------------------------------------------------------
        # Ingest
        # ------------------------------------------------------------------
        print(f"\n{'─'*72}")
        print("STEP 1 — Ingest documents")
        print(f"{'─'*72}")
        result = builder.ingest(doc_paths)
        print(result)

        st = builder.stats()
        print(f"\nGraph after ingestion:")
        print(f"  Documents  : {st['documents']}")
        print(f"  Chunks     : {st['chunks']}")
        print(f"  By type    : {st['chunks_by_type']}")
        print(f"  By file    : {st['chunks_by_document']}")

        # ------------------------------------------------------------------
        # Build datasets in every format
        # ------------------------------------------------------------------
        FORMATS = ["raw", "completion", "qa", "alpaca", "openai"]
        print(f"\n{'─'*72}")
        print("STEP 2 — Build training datasets (all 5 formats)")
        print(f"{'─'*72}")

        produced: dict[str, str] = {}
        for fmt in FORMATS:
            dataset = builder.build_dataset(format=fmt, k=100)
            out_file = str(out_dir / f"dataset_{fmt}.jsonl")
            builder.export(dataset, out_file, file_format="jsonl")
            size_kb = Path(out_file).stat().st_size / 1024
            produced[fmt] = out_file
            print(f"\n  [{fmt:10s}] {len(dataset):3d} examples  →  {size_kb:.1f} KB")
            # Show one sample entry.
            sample = dataset[0] if dataset else {}
            sample_str = json.dumps(sample, indent=4, ensure_ascii=False)
            # Trim for readability.
            if len(sample_str) > 600:
                sample_str = sample_str[:600] + "\n    ... (truncated)"
            for line in sample_str.splitlines():
                print(f"    {line}")

        # ------------------------------------------------------------------
        # CSV export demo
        # ------------------------------------------------------------------
        print(f"\n{'─'*72}")
        print("STEP 3 — Export QA dataset as CSV")
        print(f"{'─'*72}")
        qa_dataset = builder.build_dataset(format="qa", k=20)
        csv_path = str(out_dir / "dataset_qa.csv")
        builder.export(qa_dataset, csv_path, file_format="csv")
        size_kb = Path(csv_path).stat().st_size / 1024
        print(f"  Wrote {len(qa_dataset)} rows to dataset_qa.csv ({size_kb:.1f} KB)")

        # ------------------------------------------------------------------
        # Query-scoped retrieval demo
        # ------------------------------------------------------------------
        print(f"\n{'─'*72}")
        print("STEP 4 — Query-scoped dataset (vector search over chunks)")
        print(f"{'─'*72}")
        queries = [
            "How does label-scoped vector search work?",
            "Which Ollama models have large context windows?",
            "How does ContextManager avoid overflow?",
        ]
        for q in queries:
            scoped = builder.build_dataset(format="qa", query=q, k=3)
            print(f"\n  Query : {q}")
            for i, entry in enumerate(scoped, 1):
                print(f"  [{i}] Q: {entry['question']}")
                print(f"      A: {entry['answer'][:140].replace(chr(10), ' ')}...")

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        print(f"\n{'═'*72}")
        print("RESULT SUMMARY")
        print(f"{'═'*72}")
        total_chunks = st["chunks"]
        print(f"  Documents ingested  : {st['documents']}")
        print(f"  Total chunks stored : {total_chunks}")
        print(f"  Formats generated   : {len(FORMATS)}")
        total_examples = sum(
            len(builder.build_dataset(format=f, k=1000)) for f in FORMATS
        )
        print(f"  Total training examples (all formats) : {total_examples}")
        print()
        print("  Use with your own files:")
        print("    builder.ingest(['paper.pdf', 'notes.md', 'data.csv'])")
        print("    dataset = builder.build_dataset(format='alpaca', k=5000)")
        print("    builder.export(dataset, 'train.jsonl')")
        print()
        print("  With Ollama for better Q&A quality:")
        print("    from ai_memory.dataset_builder import ollama_qa_generator")
        print("    builder = DatasetBuilder(..., qa_generator=ollama_qa_generator())")


if __name__ == "__main__":
    main()
