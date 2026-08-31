"""Tests for DocumentLoader and DatasetBuilder."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

import pytest

from ai_memory.dataset_builder import (
    CHUNK,
    CHUNK_OF,
    DOCUMENT,
    NEXT_CHUNK,
    DatasetBuilder,
    IngestResult,
)
from ai_memory.document_loader import Chunk, DocumentLoader
from ai_memory.embedder import LocalEmbedder
from graphdb.store import GraphStore


# ------------------------------------------------------------------ fixtures
@pytest.fixture()
def tmp(tmp_path):
    return tmp_path


@pytest.fixture()
def txt_file(tmp):
    p = tmp / "doc.txt"
    p.write_text(
        "Graph databases store data as nodes and edges.\n\n"
        "Vector search finds similar nodes by cosine similarity.\n\n"
        "Label-scoped search restricts lookups to a specific node type.\n\n"
        "Persistence is handled via JSON serialisation on disk.",
        encoding="utf-8",
    )
    return str(p)


@pytest.fixture()
def md_file(tmp):
    p = tmp / "doc.md"
    p.write_text(
        "# SDK Reference\n\n"
        "## AgentMemory\n\n"
        "The `remember()` method stores a fact as a Memory node.\n\n"
        "## ContextManager\n\n"
        "The `assemble()` method returns a budget-safe prompt.",
        encoding="utf-8",
    )
    return str(p)


@pytest.fixture()
def csv_file(tmp):
    p = tmp / "models.csv"
    p.write_text(
        "model,context,open_source\n"
        "llama3.1:8b,65536,yes\n"
        "llama3.2:3b,32768,yes\n"
        "phi3:mini,4096,yes\n",
        encoding="utf-8",
    )
    return str(p)


@pytest.fixture()
def builder(txt_file, md_file, csv_file):
    from ai_memory.document_loader import DocumentLoader
    store = GraphStore()
    # chunk_size=100 forces multiple chunks per document so NEXT_CHUNK
    # edges are created; min_chunk_len=20 avoids discarding short paragraphs.
    b = DatasetBuilder(
        store=store,
        embedder=LocalEmbedder(),
        loader=DocumentLoader(chunk_size=100, min_chunk_len=20),
    )
    b.ingest([txt_file, md_file, csv_file])
    return b


# ====================================================================== loader
class TestDocumentLoader:
    def test_loads_txt_returns_chunks(self, txt_file):
        loader = DocumentLoader(chunk_size=200, min_chunk_len=20)
        chunks = loader.load(txt_file)
        assert chunks
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.doc_type == "txt" for c in chunks)
        assert all(c.source_path == txt_file for c in chunks)

    def test_loads_md_returns_chunks(self, md_file):
        loader = DocumentLoader(chunk_size=200, min_chunk_len=20)
        chunks = loader.load(md_file)
        assert chunks
        assert all(c.doc_type == "md" for c in chunks)

    def test_loads_csv_one_chunk_per_row(self, csv_file):
        # CSV rows are always produced regardless of min_chunk_len (structured data)
        loader = DocumentLoader()
        chunks = loader.load(csv_file)
        # 3 data rows → 3 chunks (no filtering by min_chunk_len for CSV)
        assert len(chunks) == 3
        assert all(c.doc_type == "csv" for c in chunks)
        assert all("|" in c.text for c in chunks)

    def test_chunk_indices_are_sequential(self, txt_file):
        loader = DocumentLoader(chunk_size=100, min_chunk_len=10)
        chunks = loader.load(txt_file)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunks_respect_max_size(self, tmp):
        long_para = "word " * 500  # 2500 chars — well above default 800
        p = tmp / "long.txt"
        p.write_text(long_para, encoding="utf-8")
        loader = DocumentLoader(chunk_size=200, chunk_overlap=20, min_chunk_len=10)
        chunks = loader.load(str(p))
        assert all(len(c.text) <= 220 for c in chunks)  # size + tiny overlap margin

    def test_short_noise_filtered_out(self, tmp):
        p = tmp / "noise.txt"
        p.write_text("ok\n\n" + ("substantial content here. " * 30), encoding="utf-8")
        loader = DocumentLoader(min_chunk_len=60)
        chunks = loader.load(str(p))
        assert all(len(c.text) >= 60 for c in chunks)

    def test_unknown_extension_treated_as_text(self, tmp):
        p = tmp / "notes.log"
        p.write_text("System started successfully.\n\nAll services online.", encoding="utf-8")
        loader = DocumentLoader(min_chunk_len=5)
        chunks = loader.load(str(p))
        assert chunks  # falls back to text loader


# =================================================================== ingest
class TestIngest:
    def test_creates_document_nodes(self, builder):
        docs = builder.store.nodes_by_label(DOCUMENT)
        assert len(docs) == 3

    def test_document_node_has_metadata(self, txt_file):
        store = GraphStore()
        b = DatasetBuilder(store=store, embedder=LocalEmbedder())
        b.ingest([txt_file])
        doc = store.nodes_by_label(DOCUMENT)[0]
        assert doc.properties["filename"] == Path(txt_file).name
        assert doc.properties["source_path"] == str(Path(txt_file).resolve())
        assert doc.properties["num_chunks"] > 0

    def test_creates_chunk_nodes(self, builder):
        chunks = builder.store.nodes_by_label(CHUNK)
        assert len(chunks) > 0

    def test_chunk_of_edges_link_chunk_to_document(self, builder):
        chunks = builder.store.nodes_by_label(CHUNK)
        for chunk in chunks:
            edges = builder.store.edges_from(chunk.id)
            labels = [e.label for e in edges]
            assert CHUNK_OF in labels

    def test_next_chunk_edges_preserve_order(self, builder):
        all_next = [
            e for e in builder.store.all_edges() if e.label == NEXT_CHUNK
        ]
        assert len(all_next) > 0

    def test_chunks_have_embeddings(self, builder):
        for node in builder.store.nodes_by_label(CHUNK):
            assert node.embedding is not None
            assert len(node.embedding) > 0

    def test_duplicate_ingest_is_skipped(self, txt_file):
        store = GraphStore()
        b = DatasetBuilder(store=store, embedder=LocalEmbedder())
        r1 = b.ingest([txt_file])
        r2 = b.ingest([txt_file])
        assert r1.documents == 1
        assert r2.documents == 0  # skipped
        assert len(store.nodes_by_label(DOCUMENT)) == 1  # still only one

    def test_ingest_result_reports_counts(self, txt_file, csv_file):
        store = GraphStore()
        b = DatasetBuilder(store=store, embedder=LocalEmbedder())
        result = b.ingest([txt_file, csv_file])
        assert result.documents == 2
        assert result.chunks_stored > 0
        assert len(result.files) == 2
        assert not result.errors

    def test_ingest_bad_path_reported_in_errors(self):
        store = GraphStore()
        b = DatasetBuilder(store=store, embedder=LocalEmbedder())
        result = b.ingest(["/nonexistent/path/doc.txt"])
        assert result.errors


# ================================================================ dataset build
class TestDatasetBuild:
    def test_build_raw_format(self, builder):
        ds = builder.build_dataset(format="raw", k=50)
        assert ds
        for entry in ds:
            assert "text" in entry
            assert "source" in entry
            assert "chunk_index" in entry
            assert "doc_type" in entry

    def test_build_completion_format(self, builder):
        ds = builder.build_dataset(format="completion", k=50)
        for entry in ds:
            assert "prompt" in entry
            assert "completion" in entry
            assert entry["completion"] in entry["prompt"] or entry["completion"]

    def test_build_qa_format(self, builder):
        ds = builder.build_dataset(format="qa", k=50)
        for entry in ds:
            assert "question" in entry
            assert "answer" in entry
            assert "context" in entry
            assert "source" in entry
            assert entry["question"].endswith("?")

    def test_build_alpaca_format(self, builder):
        ds = builder.build_dataset(format="alpaca", k=50)
        for entry in ds:
            assert "instruction" in entry
            assert "input" in entry
            assert "output" in entry

    def test_build_openai_format(self, builder):
        ds = builder.build_dataset(format="openai", k=50)
        for entry in ds:
            assert "messages" in entry
            roles = [m["role"] for m in entry["messages"]]
            assert roles == ["system", "user", "assistant"]

    def test_invalid_format_raises(self, builder):
        with pytest.raises(ValueError, match="Unknown format"):
            builder.build_dataset(format="invalid_xyz")

    def test_k_limits_output(self, builder):
        ds = builder.build_dataset(format="raw", k=2)
        assert len(ds) <= 2

    def test_query_scoped_retrieval(self, builder):
        ds = builder.build_dataset(format="qa", query="vector search similarity", k=3)
        assert len(ds) <= 3
        assert all("source" in e for e in ds)

    def test_source_path_filter(self, txt_file, md_file, csv_file):
        store = GraphStore()
        b = DatasetBuilder(store=store, embedder=LocalEmbedder())
        b.ingest([txt_file, md_file, csv_file])
        ds = b.build_dataset(format="raw", source_paths=[txt_file], k=100)
        for entry in ds:
            assert entry["source"] == str(Path(txt_file).resolve())

    def test_custom_qa_generator(self, builder):
        def my_gen(text):
            return f"CUSTOM: {text[:20]}?"

        builder.qa_generator = my_gen
        ds = builder.build_dataset(format="qa", k=5)
        for entry in ds:
            assert entry["question"].startswith("CUSTOM:")


# ================================================================== export
class TestExport:
    def test_export_jsonl(self, builder, tmp):
        ds = builder.build_dataset(format="qa", k=10)
        out = str(tmp / "out.jsonl")
        builder.export(ds, out, file_format="jsonl")
        lines = Path(out).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == len(ds)
        for line in lines:
            obj = json.loads(line)
            assert "question" in obj

    def test_export_csv(self, builder, tmp):
        ds = builder.build_dataset(format="qa", k=10)
        out = str(tmp / "out.csv")
        builder.export(ds, out, file_format="csv")
        with open(out, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == len(ds)
        assert "question" in rows[0]

    def test_export_openai_jsonl_messages_is_valid_json(self, builder, tmp):
        ds = builder.build_dataset(format="openai", k=5)
        out = str(tmp / "out.jsonl")
        builder.export(ds, out, file_format="jsonl")
        for line in Path(out).read_text().strip().splitlines():
            obj = json.loads(line)
            assert isinstance(obj["messages"], list)

    def test_export_invalid_format_raises(self, builder, tmp):
        ds = builder.build_dataset(format="raw", k=5)
        with pytest.raises(ValueError, match="Unknown file_format"):
            builder.export(ds, str(tmp / "out.xyz"), file_format="xyz")

    def test_stats_reflects_ingested_docs(self, builder):
        st = builder.stats()
        assert st["documents"] == 3
        assert st["chunks"] > 0
        assert "csv" in st["chunks_by_type"]

    def test_list_documents(self, builder):
        docs = builder.list_documents()
        assert len(docs) == 3
        for d in docs:
            assert "filename" in d
            assert "doc_type" in d
            assert "num_chunks" in d
