"""Tests for the Needle2 integration (needle_agent + build_needle_dataset).

``cactus-needle`` is NOT assumed to be installed. Any test that touches the
live ``needle.Needle`` agent either mocks the import or asserts the graceful
ImportError, so the whole suite runs green in a bare environment.
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest

from ai_memory.dataset_builder import DatasetBuilder
from ai_memory.document_loader import DocumentLoader
from ai_memory.embedder import LocalEmbedder
from ai_memory.needle_agent import (
    GRAPHDB_TOOL_SCHEMAS,
    NeedleAgentGroup,
    NeedleOrchestrator,
)
from graphdb.store import GraphStore


# ------------------------------------------------------------------ fixtures
@pytest.fixture()
def tech_txt(tmp_path):
    p = tmp_path / "tech.txt"
    p.write_text(
        "Graph databases store data as nodes and edges.\n\n"
        "Vector search finds similar nodes by cosine similarity.\n\n"
        "Label-scoped search restricts lookups to a specific node type.\n\n"
        "Persistence uses JSON serialisation with atomic writes.\n\n"
        "Indexes accelerate label and property lookups in memory.\n\n"
        "Traversal supports BFS, DFS, and shortest-path queries.\n\n"
        "Thread safety is provided by a reentrant lock around the store.\n\n"
        "Edges can carry a weight and arbitrary typed properties.",
        encoding="utf-8",
    )
    return str(p)


@pytest.fixture()
def research_txt(tmp_path):
    p = tmp_path / "research.txt"
    p.write_text(
        "LoRA freezes the base model and trains small adapter matrices.\n\n"
        "Off-topic examples prevent the model from calling a tool on everything.\n\n"
        "Alpaca format uses instruction, input, and output fields.\n\n"
        "The OpenAI chat format uses a messages list of role and content.\n\n"
        "Argument grounding needs many examples with varied phrasings.",
        encoding="utf-8",
    )
    return str(p)


@pytest.fixture()
def small_loader():
    # Small chunk_size forces multiple chunks per document.
    return DocumentLoader(chunk_size=100, min_chunk_len=20)


@pytest.fixture()
def populated_builder(tech_txt, small_loader):
    store = GraphStore()
    b = DatasetBuilder(store=store, embedder=LocalEmbedder(), loader=small_loader)
    b.ingest([tech_txt])
    return b


@pytest.fixture()
def group(tech_txt, small_loader):
    g = NeedleAgentGroup(
        name="tech",
        store=GraphStore(),
        embedder=LocalEmbedder(),
        loader=small_loader,
        system="knowledge_group: tech; domain: graph-db",
    )
    g.ingest([tech_txt])
    return g


# ==================================================== build_needle_dataset
class TestBuildNeedleDataset:
    def test_needle_format_returns_list(self, populated_builder):
        ds = populated_builder.build_needle_dataset(tools=GRAPHDB_TOOL_SCHEMAS, k=50)
        assert isinstance(ds, list)
        assert ds

    def test_needle_examples_have_required_keys(self, populated_builder):
        ds = populated_builder.build_needle_dataset(tools=GRAPHDB_TOOL_SCHEMAS, k=50)
        for e in ds:
            assert "query" in e
            assert "tools" in e
            assert "answers" in e
            assert "reasoning" in e

    def test_needle_positive_examples_have_calls(self, populated_builder):
        ds = populated_builder.build_needle_dataset(tools=GRAPHDB_TOOL_SCHEMAS, k=50)
        positives = [e for e in ds if e["answers"]]
        assert positives
        primary = GRAPHDB_TOOL_SCHEMAS[0]["name"]
        for e in positives:
            assert e["answers"][0]["name"] == primary
            assert "query" in e["answers"][0]["arguments"]

    def test_needle_off_topic_examples_have_empty_answers(self, populated_builder):
        ds = populated_builder.build_needle_dataset(
            tools=GRAPHDB_TOOL_SCHEMAS, k=50, off_topic_ratio=0.5
        )
        off = [e for e in ds if e["answers"] == []]
        assert off  # with a high ratio and enough chunks there must be some

    def test_needle_off_topic_ratio_approx(self, populated_builder):
        ds = populated_builder.build_needle_dataset(
            tools=GRAPHDB_TOOL_SCHEMAS, k=50, off_topic_ratio=0.25
        )
        off = [e for e in ds if e["answers"] == []]
        total = len(ds)
        if total >= 8:  # only meaningful with enough examples
            observed = len(off) / total
            # within 50% of expected 0.25 → roughly 0.1–0.4
            assert 0.05 <= observed <= 0.45

    def test_needle_tools_embedded_in_every_example(self, populated_builder):
        ds = populated_builder.build_needle_dataset(tools=GRAPHDB_TOOL_SCHEMAS, k=50)
        for e in ds:
            assert e["tools"] == GRAPHDB_TOOL_SCHEMAS

    def test_needle_reasoning_is_string(self, populated_builder):
        ds = populated_builder.build_needle_dataset(tools=GRAPHDB_TOOL_SCHEMAS, k=50)
        for e in ds:
            assert isinstance(e["reasoning"], str)
            assert e["reasoning"].strip()

    def test_needle_system_injected_when_provided(self, populated_builder):
        ds = populated_builder.build_needle_dataset(
            tools=GRAPHDB_TOOL_SCHEMAS, k=50, system="knowledge_group: x"
        )
        for e in ds:
            assert e.get("system") == "knowledge_group: x"

    def test_needle_system_absent_when_not_provided(self, populated_builder):
        ds = populated_builder.build_needle_dataset(
            tools=GRAPHDB_TOOL_SCHEMAS, k=50, system=None
        )
        for e in ds:
            assert "system" not in e

    def test_needle_k_limits_positive_examples(self, populated_builder):
        ds = populated_builder.build_needle_dataset(
            tools=GRAPHDB_TOOL_SCHEMAS, k=2, off_topic_ratio=0.0
        )
        positives = [e for e in ds if e["answers"]]
        assert len(positives) <= 2

    def test_needle_search_query_is_not_empty(self, populated_builder):
        ds = populated_builder.build_needle_dataset(tools=GRAPHDB_TOOL_SCHEMAS, k=50)
        for e in ds:
            if e["answers"]:
                assert e["answers"][0]["arguments"]["query"].strip() != ""

    def test_needle_k_zero_returns_empty(self, populated_builder):
        ds = populated_builder.build_needle_dataset(tools=GRAPHDB_TOOL_SCHEMAS, k=0)
        assert ds == []

    def test_needle_off_topic_ratio_zero_no_off_topic(self, populated_builder):
        ds = populated_builder.build_needle_dataset(
            tools=GRAPHDB_TOOL_SCHEMAS, k=50, off_topic_ratio=0.0
        )
        assert all(e["answers"] for e in ds)


# ==================================================== NeedleAgentGroup
class TestNeedleAgentGroup:
    def test_init_default_store(self):
        g = NeedleAgentGroup("g")
        assert g.store is not None
        assert isinstance(g.store, GraphStore)

    def test_init_custom_store(self):
        store = GraphStore()
        g = NeedleAgentGroup("g", store=store)
        assert g.store is store

    def test_init_name(self):
        g = NeedleAgentGroup("legal")
        assert g.name == "legal"

    def test_init_system_default(self):
        g = NeedleAgentGroup("legal")
        assert "legal" in g.system

    def test_init_system_custom(self):
        g = NeedleAgentGroup("legal", system="custom system string")
        assert g.system == "custom system string"

    def test_init_weights_none(self):
        g = NeedleAgentGroup("g")
        assert g.weights is None

    def test_init_default_tool_schemas(self):
        g = NeedleAgentGroup("g")
        assert g.tool_schemas == GRAPHDB_TOOL_SCHEMAS

    def test_ingest_returns_result(self, tech_txt, small_loader):
        g = NeedleAgentGroup("g", loader=small_loader)
        r = g.ingest([tech_txt])
        assert r.documents > 0

    def test_ingest_populates_store(self, tech_txt, small_loader):
        g = NeedleAgentGroup("g", loader=small_loader)
        g.ingest([tech_txt])
        assert len(g.store.nodes_by_label("Document")) == 1

    def test_ingest_multiple_files(self, tech_txt, research_txt, small_loader):
        g = NeedleAgentGroup("g", loader=small_loader)
        r = g.ingest([tech_txt, research_txt])
        assert r.documents == 2

    def test_export_training_data_creates_file(self, group, tmp_path):
        out = str(tmp_path / "train.jsonl")
        path = group.export_training_data(out, k=20)
        assert Path(path).exists()

    def test_export_training_data_valid_jsonl(self, group, tmp_path):
        out = str(tmp_path / "train.jsonl")
        path = group.export_training_data(out, k=20)
        for line in Path(path).read_text().strip().splitlines():
            obj = json.loads(line)
            assert isinstance(obj, dict)

    def test_export_training_data_has_query_key(self, group, tmp_path):
        out = str(tmp_path / "train.jsonl")
        path = group.export_training_data(out, k=20)
        for line in Path(path).read_text().strip().splitlines():
            assert "query" in json.loads(line)

    def test_export_training_data_off_topic_examples_present(self, group, tmp_path):
        out = str(tmp_path / "train.jsonl")
        path = group.export_training_data(out, k=50, off_topic_ratio=0.5)
        objs = [json.loads(l) for l in Path(path).read_text().strip().splitlines()]
        assert any(o["answers"] == [] for o in objs)

    def test_load_weights_updates_path(self, group):
        group.load_weights("model.cact")
        assert group.weights == "model.cact"

    def test_load_weights_invalidates_cache(self, group):
        group._agent = "sentinel"
        group.load_weights("model.cact")
        assert group._agent is None

    def test_load_weights_none_resets(self, group):
        group.load_weights("model.cact")
        group.load_weights(None)
        assert group.weights is None

    def test_stats_includes_group_name(self, group):
        assert group.stats()["group_name"] == "tech"

    def test_stats_includes_weights(self, group):
        assert "weights" in group.stats()

    def test_stats_includes_tool_count(self, group):
        assert group.stats()["tool_count"] == len(GRAPHDB_TOOL_SCHEMAS)

    def test_repr_contains_name(self, group):
        r = repr(group)
        assert "tech" in r
        assert "documents=" in r

    def test_agent_raises_without_needle(self, group, monkeypatch):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "needle":
                raise ImportError("no needle")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="cactus-needle"):
            _ = group.agent


# ==================================================== NeedleOrchestrator
class TestNeedleOrchestrator:
    def test_init_empty(self):
        orch = NeedleOrchestrator()
        assert orch.groups == []

    def test_add_group(self, group):
        orch = NeedleOrchestrator()
        orch.add_group(group)
        assert group in orch.groups

    def test_route_single_group(self, group):
        orch = NeedleOrchestrator(groups=[group])
        assert orch.route("anything at all") is group

    def test_route_empty_returns_none(self):
        orch = NeedleOrchestrator()
        assert orch.route("q") is None

    def test_route_two_groups_returns_a_group(
        self, tech_txt, research_txt, small_loader
    ):
        g1 = NeedleAgentGroup(
            "tech", loader=small_loader, system="graph database engine"
        )
        g1.ingest([tech_txt])
        g2 = NeedleAgentGroup(
            "research", loader=small_loader, system="llm fine-tuning research"
        )
        g2.ingest([research_txt])
        orch = NeedleOrchestrator(groups=[g1, g2])
        chosen = orch.route("graph database vector search")
        assert chosen in (g1, g2)

    def test_export_all_creates_files(
        self, tech_txt, research_txt, small_loader, tmp_path
    ):
        g1 = NeedleAgentGroup("tech", loader=small_loader)
        g1.ingest([tech_txt])
        g2 = NeedleAgentGroup("research", loader=small_loader)
        g2.ingest([research_txt])
        orch = NeedleOrchestrator(groups=[g1, g2])
        paths = orch.export_all_training_data(str(tmp_path / "out"), k=20)
        assert set(paths.keys()) == {"tech", "research"}
        for p in paths.values():
            assert Path(p).exists()

    def test_stats_returns_dict_per_group(self, tech_txt, small_loader):
        g1 = NeedleAgentGroup("tech", loader=small_loader)
        g1.ingest([tech_txt])
        g2 = NeedleAgentGroup("research", loader=small_loader)
        g2.ingest([tech_txt])
        orch = NeedleOrchestrator(groups=[g1, g2])
        st = orch.stats()
        assert set(st.keys()) == {"tech", "research"}

    def test_repr_contains_group_names(self, group):
        orch = NeedleOrchestrator(groups=[group])
        assert "tech" in repr(orch)

    def test_run_no_groups_returns_error(self):
        orch = NeedleOrchestrator()
        result = orch.run("q")
        assert result["type"] == "error"
