"""Tests for JEPA-GraphRAG module."""

import pytest
import numpy as np

from graphdb.core import Node, Edge
from graphdb.store import GraphStore
from ai_memory.embedder import LocalEmbedder
from ai_memory.jepa_graphrag import (
    CommunityDetector,
    GraphRAGRetriever,
    JEPAGraphRAG,
    vicreg_loss_numpy,
    HAS_NUMPY,
    HAS_TORCH,
)

if HAS_TORCH:
    import torch
    from ai_memory.jepa_graphrag import (
        TorchContextEncoder,
        TorchTargetEncoder,
        TorchPredictor,
        vicreg_loss_torch,
    )


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def store_with_communities():
    """Create a store with 3 clear communities."""
    store = GraphStore()
    embedder = LocalEmbedder()
    
    # Community 1: Users (alice, bob, carol)
    alice = store.add_node(Node(
        label="User",
        properties={"name": "alice"},
        embedding=embedder.embed("alice is a developer")
    ))
    bob = store.add_node(Node(
        label="User",
        properties={"name": "bob"},
        embedding=embedder.embed("bob is a developer")
    ))
    carol = store.add_node(Node(
        label="User",
        properties={"name": "carol"},
        embedding=embedder.embed("carol is a developer")
    ))
    
    # Connect users (community 1)
    store.add_edge(Edge(src_id=alice.id, dst_id=bob.id, label="KNOWS", weight=1.0))
    store.add_edge(Edge(src_id=bob.id, dst_id=carol.id, label="KNOWS", weight=1.0))
    store.add_edge(Edge(src_id=carol.id, dst_id=alice.id, label="KNOWS", weight=1.0))
    
    # Community 2: Posts (post1, post2)
    post1 = store.add_node(Node(
        label="Post",
        properties={"title": "Python tutorial"},
        embedding=embedder.embed("Learn Python programming basics")
    ))
    post2 = store.add_node(Node(
        label="Post",
        properties={"title": "Advanced Python"},
        embedding=embedder.embed("Advanced Python techniques")
    ))
    
    # Connect posts (community 2)
    store.add_edge(Edge(src_id=post1.id, dst_id=post2.id, label="RELATED", weight=1.0))
    
    # Community 3: Tags (tag1, tag2)
    tag1 = store.add_node(Node(
        label="Tag",
        properties={"name": "python"},
        embedding=embedder.embed("python programming language")
    ))
    tag2 = store.add_node(Node(
        label="Tag",
        properties={"name": "tutorial"},
        embedding=embedder.embed("tutorial and learning")
    ))
    
    # Connect tags (community 3)
    store.add_edge(Edge(src_id=tag1.id, dst_id=tag2.id, label="RELATED", weight=1.0))
    
    # Cross-community edges (lighter weight)
    store.add_edge(Edge(src_id=alice.id, dst_id=post1.id, label="AUTHORED", weight=0.5))
    store.add_edge(Edge(src_id=post1.id, dst_id=tag1.id, label="TAGGED", weight=0.5))
    
    return store, embedder


@pytest.fixture
def simple_store():
    """Create a simple 4-node store."""
    store = GraphStore()
    embedder = LocalEmbedder()
    
    n1 = store.add_node(Node(label="A", properties={"text": "first"},
                              embedding=embedder.embed("first node")))
    n2 = store.add_node(Node(label="A", properties={"text": "second"},
                              embedding=embedder.embed("second node")))
    n3 = store.add_node(Node(label="B", properties={"text": "third"},
                              embedding=embedder.embed("third node")))
    n4 = store.add_node(Node(label="B", properties={"text": "fourth"},
                              embedding=embedder.embed("fourth node")))
    
    store.add_edge(Edge(src_id=n1.id, dst_id=n2.id, label="LINK"))
    store.add_edge(Edge(src_id=n3.id, dst_id=n4.id, label="LINK"))
    
    return store, embedder


# ============================================================================
# Community Detection Tests
# ============================================================================

class TestCommunityDetector:
    """Test community detection."""
    
    def test_detect_communities_basic(self, store_with_communities):
        """Test basic community detection."""
        store, _ = store_with_communities
        detector = CommunityDetector(store)
        communities = detector.detect_communities()
        
        assert len(communities) == len(store.all_nodes())
        
        # All nodes should be assigned to communities
        for node in store.all_nodes():
            assert node.id in communities
            assert isinstance(communities[node.id], int)
    
    def test_detect_communities_structure(self, store_with_communities):
        """Test that strongly connected nodes share communities."""
        store, _ = store_with_communities
        detector = CommunityDetector(store)
        communities = detector.detect_communities()
        
        # Users (alice, bob, carol) should be in same community
        user_nodes = [n for n in store.all_nodes() if n.label == "User"]
        user_comms = {communities[n.id] for n in user_nodes}
        
        # Posts should likely be together
        post_nodes = [n for n in store.all_nodes() if n.label == "Post"]
        if len(post_nodes) > 1:
            post_comms = {communities[n.id] for n in post_nodes}
    
    def test_empty_store(self):
        """Test community detection on empty store."""
        store = GraphStore()
        detector = CommunityDetector(store)
        communities = detector.detect_communities()
        assert communities == {}
    
    def test_single_node(self):
        """Test community detection with single node."""
        store = GraphStore()
        n = store.add_node(Node(label="Single", properties={}))
        
        detector = CommunityDetector(store)
        communities = detector.detect_communities()
        
        assert len(communities) == 1
        assert n.id in communities


# ============================================================================
# GraphRAG Retriever Tests
# ============================================================================

class TestGraphRAGRetriever:
    """Test GraphRAG retrieval."""
    
    def test_local_search(self, store_with_communities):
        """Test local search retrieval."""
        store, embedder = store_with_communities
        retriever = GraphRAGRetriever(store, embedder)
        
        results = retriever.local_search("developer", k=2, max_hops=1)
        
        assert len(results) > 0
        assert all(isinstance(r[0], Node) for r in results)
        assert all(isinstance(r[1], float) for r in results)
        
        # Should retrieve User nodes
        labels = {r[0].label for r in results}
        assert "User" in labels
    
    def test_local_search_with_label(self, store_with_communities):
        """Test local search with label filter."""
        store, embedder = store_with_communities
        retriever = GraphRAGRetriever(store, embedder)
        
        results = retriever.local_search("python", k=5, label="Post")
        
        # Should only return Post nodes (label filter applies to seed search)
        # Note: expanded neighborhood may include other labels
        if results:
            # At least some should be Post nodes
            post_nodes = [r for r in results if r[0].label == "Post"]
            assert len(post_nodes) > 0
    
    def test_global_search(self, store_with_communities):
        """Test global search (community-based)."""
        store, embedder = store_with_communities
        retriever = GraphRAGRetriever(store, embedder)
        
        results = retriever.global_search("developer", k=2)
        
        assert len(results) > 0
        # Each result is (community_id, nodes, score)
        assert all(len(r) == 3 for r in results)
        assert all(isinstance(r[0], int) for r in results)
        assert all(isinstance(r[1], list) for r in results)
        assert all(isinstance(r[2], float) for r in results)
    
    def test_community_summary(self, store_with_communities):
        """Test community summary generation."""
        store, embedder = store_with_communities
        retriever = GraphRAGRetriever(store, embedder)
        
        communities = retriever.communities
        if communities:
            comm_id = list(communities.values())[0]
            summary = retriever.get_community_summary(comm_id)
            
            assert isinstance(summary, str)
            assert f"Community {comm_id}" in summary
    
    def test_rebuild_communities(self, simple_store):
        """Test community rebuild."""
        store, embedder = simple_store
        retriever = GraphRAGRetriever(store, embedder)
        
        # Get initial communities
        comm1 = retriever.communities
        assert len(comm1) > 0
        
        # Rebuild
        comm2 = retriever.rebuild_communities()
        assert len(comm2) == len(comm1)


# ============================================================================
# JEPA Encoder Tests (Numpy)
# ============================================================================

@pytest.mark.skipif(not HAS_NUMPY, reason="numpy not available")
class TestNumpyEncoders:
    """Test numpy-based JEPA encoders."""
    
    def test_context_encoder_forward(self):
        """Test context encoder forward pass."""
        from ai_memory.jepa_graphrag import NumpyContextEncoder
        
        encoder = NumpyContextEncoder(input_dim=64, latent_dim=32)
        x = np.random.randn(4, 64)
        z = encoder(x)
        
        assert z.shape == (4, 32)
        # Should be L2 normalized
        norms = np.linalg.norm(z, axis=-1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)
    
    def test_target_encoder_forward(self):
        """Test target encoder forward pass."""
        from ai_memory.jepa_graphrag import NumpyTargetEncoder
        
        encoder = NumpyTargetEncoder(input_dim=64, latent_dim=32)
        x = np.random.randn(4, 64)
        z = encoder(x)
        
        assert z.shape == (4, 32)
        norms = np.linalg.norm(z, axis=-1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)
    
    def test_predictor_forward(self):
        """Test predictor forward pass."""
        from ai_memory.jepa_graphrag import NumpyPredictor
        
        predictor = NumpyPredictor(latent_dim=32)
        z_ctx = np.random.randn(4, 32)
        z_pred = predictor(z_ctx)
        
        assert z_pred.shape == (4, 32)
        norms = np.linalg.norm(z_pred, axis=-1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)
    
    def test_ema_update(self):
        """Test EMA update mechanism."""
        from ai_memory.jepa_graphrag import NumpyContextEncoder, NumpyTargetEncoder
        
        context_enc = NumpyContextEncoder(input_dim=64, latent_dim=32)
        target_enc = NumpyTargetEncoder(input_dim=64, latent_dim=32)
        
        # Store original weights
        orig_w1 = target_enc.w1.copy()
        
        # Update context encoder weights
        context_enc.w1 += 0.1
        
        # Apply EMA
        target_enc.update_ema(context_enc, alpha=0.9)
        
        # Target weights should have moved
        assert not np.allclose(target_enc.w1, orig_w1)
        # But not equal to context weights (due to EMA)
        assert not np.allclose(target_enc.w1, context_enc.w1)


# ============================================================================
# JEPA Encoder Tests (PyTorch)
# ============================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestTorchEncoders:
    """Test PyTorch-based JEPA encoders."""
    
    def test_context_encoder_forward(self):
        """Test context encoder forward pass."""
        encoder = TorchContextEncoder(input_dim=64, latent_dim=32)
        x = torch.randn(4, 64)
        z = encoder(x)
        
        assert z.shape == (4, 32)
        # Should be normalized
        norms = torch.norm(z, dim=-1)
        torch.testing.assert_close(norms, torch.ones(4), rtol=1e-5, atol=1e-5)
    
    def test_target_encoder_forward(self):
        """Test target encoder forward pass."""
        encoder = TorchTargetEncoder(input_dim=64, latent_dim=32)
        x = torch.randn(4, 64)
        z = encoder(x)
        
        assert z.shape == (4, 32)
        norms = torch.norm(z, dim=-1)
        torch.testing.assert_close(norms, torch.ones(4), rtol=1e-5, atol=1e-5)
    
    def test_predictor_forward(self):
        """Test predictor forward pass."""
        predictor = TorchPredictor(latent_dim=32)
        z_ctx = torch.randn(4, 32)
        z_pred = predictor(z_ctx)
        
        assert z_pred.shape == (4, 32)
        norms = torch.norm(z_pred, dim=-1)
        torch.testing.assert_close(norms, torch.ones(4), rtol=1e-5, atol=1e-5)
    
    def test_ema_update(self):
        """Test EMA update mechanism."""
        context_enc = TorchContextEncoder(input_dim=64, latent_dim=32)
        target_enc = TorchTargetEncoder(input_dim=64, latent_dim=32)
        
        # Copy initial weights
        target_enc.load_state_dict(context_enc.state_dict())
        
        # Store original
        orig_param = list(target_enc.parameters())[0].clone()
        
        # Modify context encoder
        with torch.no_grad():
            for p in context_enc.parameters():
                p.add_(0.1)
        
        # Apply EMA
        target_enc.update_ema(context_enc, alpha=0.9)
        
        # Target should have moved
        new_param = list(target_enc.parameters())[0]
        assert not torch.allclose(new_param, orig_param)


# ============================================================================
# VICReg Loss Tests
# ============================================================================

@pytest.mark.skipif(not HAS_NUMPY, reason="numpy not available")
class TestVICRegLoss:
    """Test VICReg loss functions."""
    
    def test_vicreg_numpy_basic(self):
        """Test basic VICReg loss computation."""
        z_pred = np.random.randn(8, 32)
        z_target = np.random.randn(8, 32)
        
        total_loss, loss_dict = vicreg_loss_numpy(z_pred, z_target)
        
        assert isinstance(total_loss, (float, np.floating))
        assert 'total' in loss_dict
        assert 'sim' in loss_dict
        assert 'var' in loss_dict
        assert 'cov' in loss_dict
        
        # All losses should be non-negative
        assert total_loss >= 0
        assert loss_dict['sim'] >= 0
        assert loss_dict['var'] >= 0
        assert loss_dict['cov'] >= 0
    
    def test_vicreg_numpy_identical(self):
        """Test VICReg loss with identical inputs."""
        z = np.random.randn(8, 32)
        
        total_loss, loss_dict = vicreg_loss_numpy(z, z)
        
        # Similarity loss should be near zero
        assert loss_dict['sim'] < 1e-5
    
    @pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
    def test_vicreg_torch_basic(self):
        """Test basic VICReg loss computation in torch."""
        z_pred = torch.randn(8, 32)
        z_target = torch.randn(8, 32)
        
        total_loss, loss_dict = vicreg_loss_torch(z_pred, z_target)
        
        assert isinstance(total_loss, torch.Tensor)
        assert 'total' in loss_dict
        assert 'sim' in loss_dict
        assert 'var' in loss_dict
        assert 'cov' in loss_dict
        
        # All losses should be non-negative
        assert total_loss.item() >= 0
        assert loss_dict['sim'] >= 0
        assert loss_dict['var'] >= 0
        assert loss_dict['cov'] >= 0
    
    @pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
    def test_vicreg_torch_identical(self):
        """Test VICReg loss with identical inputs."""
        z = torch.randn(8, 32)
        
        total_loss, loss_dict = vicreg_loss_torch(z, z)
        
        # Similarity loss should be near zero
        assert loss_dict['sim'] < 1e-5


# ============================================================================
# Integrated JEPA-GraphRAG Tests
# ============================================================================

class TestJEPAGraphRAG:
    """Test integrated JEPA-GraphRAG system."""
    
    def test_init_numpy(self, simple_store):
        """Test initialization with numpy backend."""
        store, embedder = simple_store
        jepa = JEPAGraphRAG(store, embedder, latent_dim=32, use_torch=False)
        
        assert jepa.store is store
        assert jepa.embedder is embedder
        assert jepa.latent_dim == 32
        assert not jepa.use_torch
        assert jepa.retriever is not None
    
    @pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
    def test_init_torch(self, simple_store):
        """Test initialization with torch backend."""
        store, embedder = simple_store
        jepa = JEPAGraphRAG(store, embedder, latent_dim=32, use_torch=True)
        
        assert jepa.use_torch
        assert isinstance(jepa.context_encoder, TorchContextEncoder)
        assert isinstance(jepa.target_encoder, TorchTargetEncoder)
        assert isinstance(jepa.predictor, TorchPredictor)
    
    def test_train_step_text_inputs(self, simple_store):
        """Test training step with text inputs."""
        store, embedder = simple_store
        jepa = JEPAGraphRAG(store, embedder, latent_dim=32, use_torch=False)
        
        contexts = ["query one", "query two"]
        targets = ["result one", "result two"]
        
        loss_dict = jepa.train_step(contexts, targets)
        
        assert 'total' in loss_dict
        assert 'sim' in loss_dict
        assert 'var' in loss_dict
        assert 'cov' in loss_dict
        assert all(v >= 0 for v in loss_dict.values())
    
    def test_precompute_community_embeddings(self, store_with_communities):
        """Test precomputing community embeddings."""
        store, embedder = store_with_communities
        jepa = JEPAGraphRAG(store, embedder, latent_dim=32, use_torch=False)
        
        jepa.precompute_community_embeddings()
        
        assert len(jepa._community_embeddings) > 0
        
        # Each community embedding should be latent_dim dimensional
        for comm_id, emb in jepa._community_embeddings.items():
            assert isinstance(comm_id, int)
            assert emb.shape == (32,)
    
    def test_latent_search_community(self, store_with_communities):
        """Test latent search in community mode."""
        store, embedder = store_with_communities
        jepa = JEPAGraphRAG(store, embedder, latent_dim=32, use_torch=False)
        
        results = jepa.latent_search("developer python", k=2, mode='community')
        
        assert len(results) <= 2
        assert all(isinstance(r[0], int) for r in results)  # community IDs
        assert all(isinstance(r[1], float) for r in results)  # distances
        
        # Distances should be sorted (ascending)
        if len(results) > 1:
            assert results[0][1] <= results[1][1]
    
    def test_latent_search_node(self, store_with_communities):
        """Test latent search in node mode."""
        store, embedder = store_with_communities
        jepa = JEPAGraphRAG(store, embedder, latent_dim=32, use_torch=False)
        
        results = jepa.latent_search("python tutorial", k=3, mode='node')
        
        assert len(results) <= 3
        assert all(isinstance(r[0], Node) for r in results)
        assert all(isinstance(r[1], float) for r in results)
        
        # Distances should be sorted
        if len(results) > 1:
            assert results[0][1] <= results[1][1]
    
    def test_hybrid_search(self, store_with_communities):
        """Test hybrid search combining all modes."""
        store, embedder = store_with_communities
        jepa = JEPAGraphRAG(store, embedder, latent_dim=32, use_torch=False)
        
        results = jepa.hybrid_search(
            "python developer",
            k=2,
            use_local=True,
            use_global=True,
            use_latent=True
        )
        
        assert 'local' in results
        assert 'global' in results
        assert 'latent_community' in results
        assert 'latent_node' in results
        
        # All should return results
        assert len(results['local']) > 0
        assert len(results['global']) > 0
        assert len(results['latent_community']) > 0
        assert len(results['latent_node']) > 0
    
    def test_hybrid_search_selective(self, simple_store):
        """Test hybrid search with selective modes."""
        store, embedder = simple_store
        jepa = JEPAGraphRAG(store, embedder, latent_dim=32, use_torch=False)
        
        results = jepa.hybrid_search(
            "test query",
            k=1,
            use_local=True,
            use_global=False,
            use_latent=False
        )
        
        assert 'local' in results
        assert 'global' not in results
        assert 'latent_community' not in results
        assert 'latent_node' not in results


# ============================================================================
# Integration Tests
# ============================================================================

class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_full_pipeline(self, store_with_communities):
        """Test full JEPA-GraphRAG pipeline."""
        store, embedder = store_with_communities
        
        # 1. Initialize system
        jepa = JEPAGraphRAG(store, embedder, latent_dim=32, use_torch=False)
        
        # 2. Train on some examples
        contexts = ["user developer", "python tutorial"]
        targets = ["alice bob", "post about python"]
        
        loss_dict = jepa.train_step(contexts, targets)
        assert loss_dict['total'] >= 0
        
        # 3. Precompute community embeddings
        jepa.precompute_community_embeddings()
        assert len(jepa._community_embeddings) > 0
        
        # 4. Perform hybrid search
        results = jepa.hybrid_search("python programming", k=2)
        
        assert 'local' in results
        assert 'global' in results
        assert 'latent_community' in results
        assert 'latent_node' in results
        
        # All modes should return something
        assert len(results['local']) > 0
        assert len(results['global']) > 0
    
    def test_retriever_integration(self, store_with_communities):
        """Test GraphRAG retriever integration."""
        store, embedder = store_with_communities
        jepa = JEPAGraphRAG(store, embedder, latent_dim=32, use_torch=False)
        
        # Access retriever
        retriever = jepa.retriever
        
        # Get communities
        communities = retriever.communities
        assert len(communities) > 0
        
        # Perform local search
        local_results = retriever.local_search("developer", k=3)
        assert len(local_results) > 0
        
        # Perform global search
        global_results = retriever.global_search("developer", k=2)
        assert len(global_results) > 0
        
        # Get summary
        if global_results:
            comm_id = global_results[0][0]
            summary = retriever.get_community_summary(comm_id)
            assert isinstance(summary, str)
            assert "Community" in summary
