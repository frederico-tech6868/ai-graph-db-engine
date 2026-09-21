"""Tests for the JEPA-Needle integration (ai_memory.jepa_needle).

These cover everything that works *without* ``cactus-needle`` installed:
the JEPA-enhanced tool schemas, JEPA-GraphRAG-backed search on a
:class:`JEPANeedleAgent`, community tools, world-model training, stats, and
content-relevance routing in :class:`JEPAOrchestrator`. Live Needle inference
(``agent`` / ``run``) is not exercised here because it requires the optional
``cactus-needle`` dependency.
"""

import os
import tempfile

import pytest

from ai_memory.document_loader import DocumentLoader
from ai_memory.needle_agent import GRAPHDB_TOOL_SCHEMAS
from ai_memory.jepa_needle import (
    JEPA_TOOL_SCHEMAS,
    JEPANeedleAgent,
    JEPAOrchestrator,
)


# ============================================================================
# Fixtures
# ============================================================================

LEGAL_TEXT = (
    "The termination clause allows either party to end the contract with "
    "notice. Liability is limited under the indemnification section. The "
    "governing law is the state of California for this agreement.\n"
)

TECH_TEXT = (
    "The API server exposes REST endpoints over HTTP. Authentication uses "
    "JWT bearer tokens in the request header. The database layer caches "
    "queries for low latency responses.\n"
)


def _loader():
    # chunk_overlap must be < chunk_size or the sliding window never advances
    return DocumentLoader(chunk_size=200, chunk_overlap=40, min_chunk_len=20)


def _write(tmp_path, name, text):
    p = os.path.join(tmp_path, name)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)
    return p


@pytest.fixture
def legal_group(tmp_path):
    g = JEPANeedleAgent("legal", loader=_loader())
    g.ingest([_write(tmp_path, "legal.txt", LEGAL_TEXT)])
    return g


@pytest.fixture
def tech_group(tmp_path):
    g = JEPANeedleAgent("tech", loader=_loader())
    g.ingest([_write(tmp_path, "tech.txt", TECH_TEXT)])
    return g


# ============================================================================
# Tool schemas
# ============================================================================

class TestJepaToolSchemas:
    def test_schema_count_within_needle_limit(self):
        # Needle allows at most 5 tool schemas.
        assert 1 <= len(JEPA_TOOL_SCHEMAS) <= 5

    def test_schema_names(self):
        names = {s["name"] for s in JEPA_TOOL_SCHEMAS}
        assert "search_knowledge_base" in names
        assert "search_communities" in names
        assert "get_community_summary" in names
        assert "list_documents" in names
        assert "get_document_chunks" in names

    def test_search_has_mode_enum(self):
        kb = next(s for s in JEPA_TOOL_SCHEMAS if s["name"] == "search_knowledge_base")
        mode = kb["parameters"]["properties"]["mode"]
        assert set(mode["enum"]) == {"local", "global", "latent", "hybrid"}

    def test_schemas_are_independent_copies(self):
        # Editing JEPA schemas must not mutate the base needle schemas.
        base_kb = next(
            s for s in GRAPHDB_TOOL_SCHEMAS if s["name"] == "search_knowledge_base"
        )
        assert "mode" not in base_kb["parameters"]["properties"]

    def test_community_schema_shapes(self):
        sc = next(s for s in JEPA_TOOL_SCHEMAS if s["name"] == "search_communities")
        assert "query" in sc["parameters"]["properties"]
        gs = next(
            s for s in JEPA_TOOL_SCHEMAS if s["name"] == "get_community_summary"
        )
        assert "community_id" in gs["parameters"]["properties"]


# ============================================================================
# JEPANeedleAgent construction & config
# ============================================================================

class TestJepaNeedleAgentBasics:
    def test_defaults_to_jepa_schemas(self):
        g = JEPANeedleAgent("x")
        names = {s["name"] for s in g.tool_schemas}
        assert "search_communities" in names

    def test_custom_schemas_respected(self):
        custom = [GRAPHDB_TOOL_SCHEMAS[0]]
        g = JEPANeedleAgent("x", tool_schemas=custom)
        assert len(g.tool_schemas) == 1

    def test_lazy_jepa_built_once(self, legal_group):
        j1 = legal_group.jepa
        j2 = legal_group.jepa
        assert j1 is j2
        # bound to the same store/embedder as the group
        assert j1.store is legal_group.store
        assert j1.embedder is legal_group.embedder

    def test_default_search_mode(self):
        g = JEPANeedleAgent("x", default_search_mode="local")
        assert g.default_search_mode == "local"

    def test_repr(self, legal_group):
        assert "JEPANeedleAgent" in repr(legal_group)
        assert "legal" in repr(legal_group)

    def test_stats_includes_jepa_fields(self, legal_group):
        st = legal_group.stats()
        assert st["group_name"] == "legal"
        assert "latent_dim" in st
        assert st["default_search_mode"] == "hybrid"
        assert st["backend"] in ("numpy", "torch")


# ============================================================================
# Search modes
# ============================================================================

class TestJepaNeedleSearch:
    @pytest.mark.parametrize("mode", ["local", "global", "latent", "hybrid"])
    def test_all_modes_return_results(self, legal_group, mode):
        out = legal_group.search("termination clause", k=3, mode=mode)
        assert out["mode"] == mode
        assert out["count"] >= 1
        assert len(out["results"]) == out["count"]

    def test_result_shape(self, legal_group):
        out = legal_group.search("indemnification", k=2, mode="local")
        r = out["results"][0]
        assert set(["text", "source", "doc_type", "node_id", "score"]).issubset(r)

    def test_default_mode_used_when_none(self, tmp_path):
        g = JEPANeedleAgent("legal", loader=_loader(), default_search_mode="local")
        g.ingest([_write(tmp_path, "legal.txt", LEGAL_TEXT)])
        out = g.search("contract")
        assert out["mode"] == "local"

    def test_results_capped_at_k(self, legal_group):
        out = legal_group.search("contract", k=1, mode="hybrid")
        assert len(out["results"]) <= 1

    def test_doc_type_filter_removes_nonmatching(self, legal_group):
        out = legal_group.search("contract", k=5, mode="local", doc_type="pdf")
        # our fixture is a .txt document, so a pdf filter yields nothing
        assert out["count"] == 0

    def test_results_sorted_desc(self, legal_group):
        out = legal_group.search("contract liability", k=5, mode="hybrid")
        scores = [r["score"] for r in out["results"]]
        assert scores == sorted(scores, reverse=True)


# ============================================================================
# Community tools
# ============================================================================

class TestCommunityTools:
    def test_search_communities(self, legal_group):
        out = legal_group.search_communities("contract", k=2)
        assert out["count"] >= 1
        c = out["communities"][0]
        assert "community_id" in c and "summary" in c and "size" in c

    def test_community_summary(self, legal_group):
        comms = legal_group.search_communities("contract", k=1)["communities"]
        cid = comms[0]["community_id"]
        out = legal_group.community_summary(cid)
        assert out["community_id"] == cid
        assert isinstance(out["summary"], str) and out["summary"]


# ============================================================================
# World-model training
# ============================================================================

class TestWorldModelTraining:
    def test_train_returns_loss_history(self, legal_group):
        hist = legal_group.train_world_model(epochs=2)
        assert len(hist) == 2
        assert "total" in hist[0]

    def test_train_empty_graph_is_noop(self):
        g = JEPANeedleAgent("empty")
        assert g.train_world_model(epochs=3) == []

    def test_rebuild_index_safe(self, legal_group):
        _ = legal_group.jepa  # build it
        legal_group.rebuild_index()  # should not raise
        assert legal_group.jepa._community_embeddings == {}


# ============================================================================
# Orchestrator routing
# ============================================================================

class TestJepaOrchestrator:
    def test_empty_returns_none(self):
        assert JEPAOrchestrator([]).route("anything") is None

    def test_single_group_shortcut(self, legal_group):
        orch = JEPAOrchestrator([legal_group])
        assert orch.route("anything") is legal_group

    def test_routes_by_content_relevance(self, legal_group, tech_group):
        orch = JEPAOrchestrator([legal_group, tech_group])
        assert orch.route("JWT bearer token authentication").name == "tech"
        assert orch.route("contract termination liability").name == "legal"
        assert orch.route("REST API database caching").name == "tech"
        assert orch.route("governing law indemnification clause").name == "legal"

    def test_add_group(self, legal_group, tech_group):
        orch = JEPAOrchestrator([legal_group])
        orch.add_group(tech_group)
        assert len(orch.groups) == 2

    def test_latent_can_be_disabled(self, legal_group, tech_group):
        orch = JEPAOrchestrator([legal_group, tech_group], use_latent=False)
        # still routes correctly by content alone
        assert orch.route("JWT token").name == "tech"

    def test_stats_all_groups(self, legal_group, tech_group):
        orch = JEPAOrchestrator([legal_group, tech_group])
        st = orch.stats()
        assert set(st.keys()) == {"legal", "tech"}

    def test_export_all_training_data(self, legal_group, tech_group, tmp_path):
        orch = JEPAOrchestrator([legal_group, tech_group])
        out_dir = os.path.join(tmp_path, "train")
        paths = orch.export_all_training_data(out_dir, k=10)
        assert set(paths.keys()) == {"legal", "tech"}
        for p in paths.values():
            assert os.path.exists(p)
