#!/usr/bin/env python3
"""Example: Needle2 Trainable Agent Groups backed by AI-GraphDB-Engine.

This example requires:
  pip install cactus-needle          # for inference
  pip install 'cactus-needle[train]' # for fine-tuning
  pip install 'cactus-needle[train,gpu]'  # NVIDIA GPU training

The ingest, export, routing, and stats sections run without needle installed.
Only the agent.run() section is skipped if needle is not available.
"""

import json
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# --- Import the new Needle integration ----
from ai_memory.needle_agent import (  # noqa: E402
    NeedleAgentGroup,
    NeedleOrchestrator,
    GRAPHDB_TOOL_SCHEMAS,
)
from ai_memory.embedder import LocalEmbedder  # noqa: E402
from ai_memory.document_loader import DocumentLoader  # noqa: E402
from graphdb.store import GraphStore  # noqa: E402


def main():
    print("=" * 60)
    print("  Needle2 Trainable Agent Groups — AI-GraphDB-Engine")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # -----------------------------------------------------------
        # 1. Create sample documents for two knowledge domains
        # -----------------------------------------------------------
        tech_doc = tmp / "tech_overview.md"
        tech_doc.write_text("""# Graph Database Technical Reference

## Node Storage

Nodes are stored in-memory as a dictionary keyed by UUID.
Each node has a label, properties dict, and an optional embedding vector.

## Edge Storage

Edges link two nodes by source and destination ID.
An edge has a label and an optional weight.

## Vector Search

Cosine similarity search is label-scoped to avoid cross-type false matches.
The search returns the top-k most similar nodes for a given embedding vector.

## Persistence

The graph is serialised as JSON to disk using atomic writes (temp file rename).
The Rust backend uses memory-mapped files for faster load times.
""")

        research_doc = tmp / "research_notes.md"
        research_doc.write_text("""# Research Notes — LLM Fine-tuning

## LoRA Adapters

Low-Rank Adaptation (LoRA) freezes the base model and trains small adapter matrices.
Rank 16 on the five attention projections covers most downstream tasks.

## Training Data Quality

Tool selection improves with a few hundred clean examples.
Argument grounding requires thousands of examples with varied phrasings.
Off-topic examples (1-in-8) prevent the model from calling a tool on everything.

## Dataset Formats

The Alpaca format uses instruction, input, output fields.
The OpenAI chat format uses a messages list with role/content pairs.
The Needle format adds tools, answers, and reasoning fields.
""")

        # -----------------------------------------------------------
        # 2. Create two NeedleAgentGroups
        # -----------------------------------------------------------
        print("\n[1] Creating knowledge groups...")

        # A small chunk_size splits each section into its own chunk, giving
        # enough positive examples to interleave off-topic ones (1-in-8).
        tech_group = NeedleAgentGroup(
            name="tech_docs",
            store=GraphStore(),
            embedder=LocalEmbedder(),
            loader=DocumentLoader(chunk_size=200, min_chunk_len=20),
            system="knowledge_group: tech_docs; domain: graph-database-engine",
        )

        research_group = NeedleAgentGroup(
            name="research_docs",
            store=GraphStore(),
            embedder=LocalEmbedder(),
            loader=DocumentLoader(chunk_size=200, min_chunk_len=20),
            system="knowledge_group: research_docs; domain: llm-fine-tuning",
        )

        # -----------------------------------------------------------
        # 3. Ingest documents
        # -----------------------------------------------------------
        print("\n[2] Ingesting documents...")

        r1 = tech_group.ingest([str(tech_doc)])
        print(f"  tech_docs:     {r1.documents} document, {r1.chunks_stored} chunks")

        r2 = research_group.ingest([str(research_doc)])
        print(f"  research_docs: {r2.documents} document, {r2.chunks_stored} chunks")

        # -----------------------------------------------------------
        # 4. Export Needle training data
        # -----------------------------------------------------------
        print("\n[3] Exporting Needle training JSONL...")

        out_dir = tmp / "needle_training"
        out_dir.mkdir()

        p1 = tech_group.export_training_data(str(out_dir / "tech_train.jsonl"), k=50)
        p2 = research_group.export_training_data(
            str(out_dir / "research_train.jsonl"), k=50
        )

        tech_lines = Path(p1).read_text().strip().splitlines()
        res_lines = Path(p2).read_text().strip().splitlines()
        print(f"  tech_docs:     {len(tech_lines)} examples  →  {p1}")
        print(f"  research_docs: {len(res_lines)} examples  →  {p2}")

        # Show one training example
        print("\n[4] Sample training example (tech_docs):")
        sample = json.loads(tech_lines[0])
        print(json.dumps(sample, indent=2))

        # Show an off-topic example
        off_topic = next(
            (json.loads(l) for l in tech_lines if json.loads(l)["answers"] == []),
            None,
        )
        if off_topic:
            print("\n  Off-topic example (answers=[]):")
            print(json.dumps(off_topic, indent=2))

        # -----------------------------------------------------------
        # 5. Orchestrator routing
        # -----------------------------------------------------------
        print("\n[5] NeedleOrchestrator routing...")

        orchestrator = NeedleOrchestrator(
            groups=[tech_group, research_group],
            embedder=LocalEmbedder(),
        )

        test_queries = [
            "How does cosine similarity work in vector search?",
            "What is LoRA and how does it help with fine-tuning?",
            "Explain the graph persistence mechanism.",
            "How many off-topic examples should I include in training data?",
        ]

        for q in test_queries:
            chosen = orchestrator.route(q)
            print(f"  Query: {q[:55]!r}...")
            print(f"  → Routed to: {chosen.name!r}\n")

        # -----------------------------------------------------------
        # 6. Stats
        # -----------------------------------------------------------
        print("[6] Group stats:")
        for name, st in orchestrator.stats().items():
            print(
                f"  {name}: {st['documents']} docs, {st['chunks']} chunks, "
                f"weights={st['weights']!r}"
            )

        # -----------------------------------------------------------
        # 7. Training instructions
        # -----------------------------------------------------------
        print("\n[7] Fine-tuning commands (run these in your terminal):")
        print("""
  # Install training deps:
  pip install 'cactus-needle[train]'          # CPU
  pip install 'cactus-needle[train,gpu]'       # NVIDIA GPU
  pip install 'cactus-needle[train,metal]'     # Apple Silicon

  # Fine-tune each group:
  needle finetune tech_train.jsonl --epochs 20 --out tech_adapter.pkl
  needle build checkpoints/needle2.pkl --lora tech_adapter.pkl --out tech.cact

  needle finetune research_train.jsonl --epochs 20 --out research_adapter.pkl
  needle build checkpoints/needle2.pkl --lora research_adapter.pkl --out research.cact

  # Then load weights:
  tech_group.load_weights("tech.cact")
  result = tech_group.run("How does vector search work?")
""")

        # -----------------------------------------------------------
        # 8. Live inference (requires needle installed)
        # -----------------------------------------------------------
        print("[8] Live inference (requires: pip install cactus-needle)...")
        try:
            import needle  # type: ignore  # noqa: F401

            print("  needle available — running query...")
            tech_group.load_weights(None)  # reset, no weights = base model
            result = tech_group.run("What is the graph persistence mechanism?")
            print(f"  type: {result.get('type')}")
            print(f"  calls: {result.get('function_calls')}")
            print(f"  confidence: {result.get('confidence')}")
        except ImportError:
            print("  needle not installed — skipping live inference.")
            print("  Install with: pip install cactus-needle")

    print("\nDone! ✓")


if __name__ == "__main__":
    main()
