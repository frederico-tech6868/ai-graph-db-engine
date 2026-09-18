"""
JEPA-GraphRAG: Joint-Embedding Predictive Architecture with GraphRAG retrieval.

Implements:
1. GraphRAG-style retrieval (community detection, local/global search)
2. Graph-JEPA world model (Context/Target encoders, Predictor, VICReg loss)
3. Energy-based latent retrieval

Pure Python + numpy (optional torch acceleration), no external graph DBs.
"""

from __future__ import annotations

import copy
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from graphdb.core import Node, Edge
from graphdb.store import GraphStore


# ============================================================================
# Community Detection (Louvain Algorithm)
# ============================================================================

class CommunityDetector:
    """
    Louvain algorithm for hierarchical community detection.
    Pure Python implementation, works with GraphStore.
    """
    
    def __init__(self, store: GraphStore, resolution: float = 1.0):
        """
        Args:
            store: GraphStore instance
            resolution: Resolution parameter (higher = more communities)
        """
        self.store = store
        self.resolution = resolution
        
    def detect_communities(self) -> Dict[str, int]:
        """
        Run Louvain algorithm to detect communities.
        
        Returns:
            Dict mapping node_id -> community_id
        """
        # Build adjacency structure
        nodes = self.store.all_nodes()
        node_ids = [n.id for n in nodes]
        
        if not node_ids:
            return {}
        
        # Initialize: each node in its own community
        node_to_comm = {nid: i for i, nid in enumerate(node_ids)}
        
        # Build weighted degree and edge weight maps
        weighted_degree = defaultdict(float)
        edge_weights = {}
        
        for edge in self.store.all_edges():
            w = edge.weight
            weighted_degree[edge.src_id] += w
            weighted_degree[edge.dst_id] += w
            edge_weights[(edge.src_id, edge.dst_id)] = w
            edge_weights[(edge.dst_id, edge.src_id)] = w  # undirected
        
        total_weight = sum(weighted_degree.values()) / 2.0
        
        if total_weight == 0:
            return node_to_comm
        
        # Iterative optimization
        improved = True
        iteration = 0
        max_iterations = 100
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Shuffle nodes for randomization
            random.shuffle(node_ids)
            
            for node_id in node_ids:
                # Get neighbors
                neighbors = set()
                for edge in self.store.edges_from(node_id):
                    neighbors.add(edge.dst_id)
                for edge in self.store.edges_to(node_id):
                    neighbors.add(edge.src_id)
                
                if not neighbors:
                    continue
                
                # Current community
                curr_comm = node_to_comm[node_id]
                
                # Calculate modularity gain for each neighbor community
                neighbor_comms = {node_to_comm[n] for n in neighbors}
                
                best_comm = curr_comm
                best_gain = 0.0
                
                for comm in neighbor_comms:
                    gain = self._modularity_gain(
                        node_id, comm, node_to_comm, 
                        weighted_degree, edge_weights, total_weight
                    )
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_comm = comm
                
                # Move to best community
                if best_comm != curr_comm and best_gain > 1e-10:
                    node_to_comm[node_id] = best_comm
                    improved = True
        
        return node_to_comm
    
    def _modularity_gain(
        self,
        node_id: str,
        target_comm: int,
        node_to_comm: Dict[str, int],
        weighted_degree: Dict[str, float],
        edge_weights: Dict[Tuple[str, str], float],
        total_weight: float
    ) -> float:
        """Calculate modularity gain for moving node to target community."""
        curr_comm = node_to_comm[node_id]
        
        if curr_comm == target_comm:
            return 0.0
        
        # Sum of weights from node to target community
        ki_in = 0.0
        for other_id, comm in node_to_comm.items():
            if comm == target_comm and other_id != node_id:
                ki_in += edge_weights.get((node_id, other_id), 0.0)
        
        # Node's weighted degree
        ki = weighted_degree.get(node_id, 0.0)
        
        # Total degree of target community
        sigma_tot = sum(
            weighted_degree.get(nid, 0.0)
            for nid, c in node_to_comm.items()
            if c == target_comm
        )
        
        # Modularity gain
        gain = (ki_in - self.resolution * sigma_tot * ki / (2.0 * total_weight))
        return gain


# ============================================================================
# GraphRAG Retrieval
# ============================================================================

class GraphRAGRetriever:
    """
    GraphRAG-style retrieval over GraphStore.
    Supports local search (k-hop neighborhoods) and global search (communities).
    """
    
    def __init__(
        self,
        store: GraphStore,
        embedder: Optional[Any] = None,
        community_cache: Optional[Dict[str, int]] = None
    ):
        """
        Args:
            store: GraphStore instance
            embedder: Embedder for text encoding (optional)
            community_cache: Pre-computed community assignments (optional)
        """
        self.store = store
        self.embedder = embedder
        self._community_cache = community_cache
    
    @property
    def communities(self) -> Dict[str, int]:
        """Get or compute community assignments."""
        if self._community_cache is None:
            detector = CommunityDetector(self.store)
            self._community_cache = detector.detect_communities()
        return self._community_cache
    
    def rebuild_communities(self):
        """Force rebuild of community structure."""
        self._community_cache = None
        return self.communities
    
    def local_search(
        self,
        query: str,
        k: int = 5,
        max_hops: int = 2,
        label: Optional[str] = None
    ) -> List[Tuple[Node, float]]:
        """
        Local search: find relevant nodes and expand k-hop neighborhood.
        
        Args:
            query: Search query text
            k: Number of seed nodes to retrieve
            max_hops: Maximum hop distance for expansion
            label: Optional label filter
            
        Returns:
            List of (node, score) tuples
        """
        if not self.embedder:
            raise ValueError("Embedder required for local search")
        
        # Get query embedding
        query_emb = self.embedder.embed(query)
        
        # Find top-k seed nodes via vector search
        seed_nodes = self.store.search_similar_nodes(
            query_emb, label=label, k=k
        )
        
        if not seed_nodes:
            return []
        
        # Expand neighborhoods
        expanded = set()
        for node, score in seed_nodes:
            expanded.add(node.id)
            self._expand_neighborhood(node.id, max_hops, expanded)
        
        # Return all expanded nodes with scores
        results = []
        for node_id in expanded:
            node = self.store.get_node(node_id)
            if node and node.embedding:
                # Re-score against query
                from graphdb.vector import cosine_similarity
                score = cosine_similarity(query_emb, node.embedding)
                results.append((node, score))
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _expand_neighborhood(
        self, node_id: str, max_hops: int, visited: set
    ):
        """BFS expansion of neighborhood."""
        if max_hops <= 0:
            return
        
        # Get neighbors
        neighbors = set()
        for edge in self.store.edges_from(node_id):
            neighbors.add(edge.dst_id)
        for edge in self.store.edges_to(node_id):
            neighbors.add(edge.src_id)
        
        for neighbor_id in neighbors:
            if neighbor_id not in visited:
                visited.add(neighbor_id)
                self._expand_neighborhood(neighbor_id, max_hops - 1, visited)
    
    def global_search(
        self,
        query: str,
        k: int = 3,
        label: Optional[str] = None
    ) -> List[Tuple[int, List[Node], float]]:
        """
        Global search: retrieve top-k communities based on query.
        
        Args:
            query: Search query text
            k: Number of communities to retrieve
            label: Optional label filter
            
        Returns:
            List of (community_id, nodes, score) tuples
        """
        if not self.embedder:
            raise ValueError("Embedder required for global search")
        
        query_emb = self.embedder.embed(query)
        communities = self.communities
        
        # Group nodes by community
        comm_nodes = defaultdict(list)
        for node in self.store.all_nodes():
            if label and node.label != label:
                continue
            comm_id = communities.get(node.id)
            if comm_id is not None:
                comm_nodes[comm_id].append(node)
        
        # Score each community (mean similarity of its nodes)
        from graphdb.vector import cosine_similarity
        comm_scores = []
        
        for comm_id, nodes in comm_nodes.items():
            # Only score nodes with embeddings
            embedded_nodes = [n for n in nodes if n.embedding]
            if not embedded_nodes:
                continue
            
            # Compute mean similarity
            scores = [
                cosine_similarity(query_emb, n.embedding)
                for n in embedded_nodes
            ]
            mean_score = sum(scores) / len(scores) if scores else 0.0
            comm_scores.append((comm_id, nodes, mean_score))
        
        # Sort and return top-k
        comm_scores.sort(key=lambda x: x[2], reverse=True)
        return comm_scores[:k]
    
    def get_community_summary(self, community_id: int) -> str:
        """Generate text summary for a community."""
        communities = self.communities
        nodes = [
            n for n in self.store.all_nodes()
            if communities.get(n.id) == community_id
        ]
        
        if not nodes:
            return f"Community {community_id}: (empty)"
        
        # Aggregate labels and properties
        labels = defaultdict(int)
        for node in nodes:
            labels[node.label] += 1
        
        summary_parts = [f"Community {community_id}:"]
        summary_parts.append(f"  {len(nodes)} nodes")
        for label, count in labels.items():
            summary_parts.append(f"  - {count} {label} nodes")
        
        return "\n".join(summary_parts)


# ============================================================================
# Graph-JEPA Neural Components
# ============================================================================

if HAS_NUMPY:
    class NumpyContextEncoder:
        """Context encoder using pure numpy."""
        
        def __init__(self, input_dim: int, latent_dim: int = 128):
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            
            # Simple MLP: input -> hidden -> latent
            self.w1 = np.random.randn(input_dim, 256) * 0.01
            self.b1 = np.zeros(256)
            self.w2 = np.random.randn(256, latent_dim) * 0.01
            self.b2 = np.zeros(latent_dim)
        
        def __call__(self, x: np.ndarray) -> np.ndarray:
            """Forward pass."""
            # Layer 1
            h = np.maximum(0, x @ self.w1 + self.b1)  # ReLU
            # Layer 2
            z = h @ self.w2 + self.b2
            # L2 normalize
            z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
            return z
    
    class NumpyTargetEncoder:
        """Target encoder with EMA (Exponential Moving Average)."""
        
        def __init__(self, input_dim: int, latent_dim: int = 128):
            self.input_dim = input_dim
            self.latent_dim = latent_dim
            
            # Same structure as context encoder
            self.w1 = np.random.randn(input_dim, 256) * 0.01
            self.b1 = np.zeros(256)
            self.w2 = np.random.randn(256, latent_dim) * 0.01
            self.b2 = np.zeros(latent_dim)
        
        def __call__(self, x: np.ndarray) -> np.ndarray:
            """Forward pass."""
            h = np.maximum(0, x @ self.w1 + self.b1)
            z = h @ self.w2 + self.b2
            z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
            return z
        
        def update_ema(self, source_encoder: 'NumpyContextEncoder', alpha: float = 0.99):
            """Update weights via EMA from source encoder."""
            self.w1 = alpha * self.w1 + (1 - alpha) * source_encoder.w1
            self.b1 = alpha * self.b1 + (1 - alpha) * source_encoder.b1
            self.w2 = alpha * self.w2 + (1 - alpha) * source_encoder.w2
            self.b2 = alpha * self.b2 + (1 - alpha) * source_encoder.b2
    
    class NumpyPredictor:
        """Predictor network in latent space."""
        
        def __init__(self, latent_dim: int = 128):
            self.latent_dim = latent_dim
            
            # MLP: latent -> hidden -> latent
            self.w1 = np.random.randn(latent_dim, latent_dim) * 0.01
            self.b1 = np.zeros(latent_dim)
            self.w2 = np.random.randn(latent_dim, latent_dim) * 0.01
            self.b2 = np.zeros(latent_dim)
        
        def __call__(self, z_ctx: np.ndarray) -> np.ndarray:
            """Predict target state from context."""
            h = np.maximum(0, z_ctx @ self.w1 + self.b1)
            z_pred = h @ self.w2 + self.b2
            z_pred = z_pred / (np.linalg.norm(z_pred, axis=-1, keepdims=True) + 1e-8)
            return z_pred


if HAS_TORCH:
    class TorchContextEncoder(nn.Module):
        """Context encoder using PyTorch."""
        
        def __init__(self, input_dim: int, latent_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, latent_dim)
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            z = self.net(x)
            z = F.normalize(z, dim=-1)
            return z
    
    class TorchTargetEncoder(nn.Module):
        """Target encoder with EMA."""
        
        def __init__(self, input_dim: int, latent_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, latent_dim)
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            z = self.net(x)
            z = F.normalize(z, dim=-1)
            return z
        
        @torch.no_grad()
        def update_ema(self, source_encoder: nn.Module, alpha: float = 0.99):
            """Update weights via EMA from source encoder."""
            for param_t, param_s in zip(self.parameters(), source_encoder.parameters()):
                param_t.data.mul_(alpha).add_(param_s.data, alpha=1 - alpha)
    
    class TorchPredictor(nn.Module):
        """Predictor network in latent space."""
        
        def __init__(self, latent_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim)
            )
        
        def forward(self, z_ctx: torch.Tensor) -> torch.Tensor:
            z_pred = self.net(z_ctx)
            z_pred = F.normalize(z_pred, dim=-1)
            return z_pred


# ============================================================================
# VICReg Loss (Variance-Covariance Regularization)
# ============================================================================

def vicreg_loss_numpy(
    z_pred: np.ndarray,
    z_target: np.ndarray,
    sim_coeff: float = 25.0,
    var_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    eps: float = 1e-4
) -> Tuple[float, Dict[str, float]]:
    """
    VICReg loss in numpy.
    
    Returns:
        (total_loss, loss_dict)
    """
    # 1. Invariance (MSE)
    sim_loss = np.mean((z_pred - z_target) ** 2)
    
    # Center
    z_pred_centered = z_pred - z_pred.mean(axis=0, keepdims=True)
    z_target_centered = z_target - z_target.mean(axis=0, keepdims=True)
    
    # 2. Variance
    std_pred = np.sqrt(z_pred_centered.var(axis=0) + eps)
    std_target = np.sqrt(z_target_centered.var(axis=0) + eps)
    var_loss = (
        np.mean(np.maximum(0, 1.0 - std_pred)) +
        np.mean(np.maximum(0, 1.0 - std_target))
    ) / 2.0
    
    # 3. Covariance
    batch_size = z_pred.shape[0]
    if batch_size > 1:
        cov_pred = (z_pred_centered.T @ z_pred_centered) / (batch_size - 1)
        cov_target = (z_target_centered.T @ z_target_centered) / (batch_size - 1)
        
        # Zero out diagonal
        np.fill_diagonal(cov_pred, 0)
        np.fill_diagonal(cov_target, 0)
        
        cov_loss = (np.sum(cov_pred ** 2) + np.sum(cov_target ** 2)) / (2.0 * z_pred.shape[1])
    else:
        cov_loss = 0.0
    
    total_loss = sim_coeff * sim_loss + var_coeff * var_loss + cov_coeff * cov_loss
    
    return total_loss, {
        'total': total_loss,
        'sim': sim_loss,
        'var': var_loss,
        'cov': cov_loss
    }


if HAS_TORCH:
    def vicreg_loss_torch(
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
        sim_coeff: float = 25.0,
        var_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        eps: float = 1e-4
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """VICReg loss in PyTorch."""
        # 1. Invariance
        sim_loss = F.mse_loss(z_pred, z_target)
        
        # Center
        z_pred_centered = z_pred - z_pred.mean(dim=0, keepdim=True)
        z_target_centered = z_target - z_target.mean(dim=0, keepdim=True)
        
        # 2. Variance
        std_pred = torch.sqrt(z_pred_centered.var(dim=0) + eps)
        std_target = torch.sqrt(z_target_centered.var(dim=0) + eps)
        var_loss = (
            torch.mean(F.relu(1.0 - std_pred)) +
            torch.mean(F.relu(1.0 - std_target))
        ) / 2.0
        
        # 3. Covariance
        batch_size = z_pred.size(0)
        if batch_size > 1:
            cov_pred = (z_pred_centered.T @ z_pred_centered) / (batch_size - 1)
            cov_target = (z_target_centered.T @ z_target_centered) / (batch_size - 1)
            
            # Zero out diagonal
            cov_pred.fill_diagonal_(0)
            cov_target.fill_diagonal_(0)
            
            cov_loss = (cov_pred.pow(2).sum() + cov_target.pow(2).sum()) / (2.0 * z_pred.size(1))
        else:
            cov_loss = torch.tensor(0.0, device=z_pred.device)
        
        total_loss = sim_coeff * sim_loss + var_coeff * var_loss + cov_coeff * cov_loss
        
        return total_loss, {
            'total': total_loss.item(),
            'sim': sim_loss.item(),
            'var': var_loss.item(),
            'cov': cov_loss.item() if isinstance(cov_loss, torch.Tensor) else cov_loss
        }


# ============================================================================
# Unified JEPA-GraphRAG System
# ============================================================================

class JEPAGraphRAG:
    """
    Unified JEPA-GraphRAG system.
    
    Combines:
    - GraphRAG retrieval (community detection, local/global search)
    - Graph-JEPA world model (latent prediction + energy-based retrieval)
    """
    
    def __init__(
        self,
        store: GraphStore,
        embedder: Optional[Any] = None,
        latent_dim: int = 128,
        use_torch: bool = None
    ):
        """
        Args:
            store: GraphStore instance
            embedder: Embedder for text encoding
            latent_dim: Dimensionality of latent space
            use_torch: Force torch backend (None = auto-detect)
        """
        self.store = store
        self.embedder = embedder
        self.latent_dim = latent_dim
        
        # Determine backend
        if use_torch is None:
            use_torch = HAS_TORCH
        elif use_torch and not HAS_TORCH:
            raise ValueError("PyTorch not available but use_torch=True")
        
        self.use_torch = use_torch
        
        # GraphRAG retriever
        self.retriever = GraphRAGRetriever(store, embedder)
        
        # JEPA components
        if embedder:
            input_dim = len(embedder.embed("test"))
        else:
            input_dim = 128  # default
        
        if use_torch:
            self.context_encoder = TorchContextEncoder(input_dim, latent_dim)
            self.target_encoder = TorchTargetEncoder(input_dim, latent_dim)
            self.predictor = TorchPredictor(latent_dim)
            
            # Copy initial weights to target encoder
            self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        else:
            if not HAS_NUMPY:
                raise ValueError("Numpy required for non-torch backend")
            self.context_encoder = NumpyContextEncoder(input_dim, latent_dim)
            self.target_encoder = NumpyTargetEncoder(input_dim, latent_dim)
            self.predictor = NumpyPredictor(latent_dim)
        
        # Cache for precomputed community embeddings
        self._community_embeddings: Dict[int, Any] = {}
    
    def train_step(
        self,
        contexts: Union[List[str], np.ndarray, 'torch.Tensor'],
        targets: Union[List[str], np.ndarray, 'torch.Tensor'],
        learning_rate: float = 0.001,
        ema_alpha: float = 0.99
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            contexts: Context inputs (queries or node texts)
            targets: Target inputs (subgraph nodes)
            learning_rate: Learning rate for gradient update
            ema_alpha: EMA momentum for target encoder
            
        Returns:
            Loss dictionary
        """
        # Encode inputs
        if isinstance(contexts, list):
            if not self.embedder:
                raise ValueError("Embedder required for text inputs")
            x_ctx = np.array([self.embedder.embed(c) for c in contexts])
            x_tgt = np.array([self.embedder.embed(t) for t in targets])
        else:
            x_ctx = contexts
            x_tgt = targets
        
        if self.use_torch:
            if not isinstance(x_ctx, torch.Tensor):
                x_ctx = torch.from_numpy(x_ctx).float()
            if not isinstance(x_tgt, torch.Tensor):
                x_tgt = torch.from_numpy(x_tgt).float()
            
            # Forward
            z_ctx = self.context_encoder(x_ctx)
            z_pred = self.predictor(z_ctx)
            
            with torch.no_grad():
                z_tgt = self.target_encoder(x_tgt)
            
            # Loss
            loss, loss_dict = vicreg_loss_torch(z_pred, z_tgt)
            
            # Backward (simplified - in practice use optimizer)
            loss.backward()
            
            # Update target encoder via EMA
            self.target_encoder.update_ema(self.context_encoder, ema_alpha)
            
            return loss_dict
        else:
            # Numpy backend
            z_ctx = self.context_encoder(x_ctx)
            z_pred = self.predictor(z_ctx)
            z_tgt = self.target_encoder(x_tgt)
            
            # Loss
            total_loss, loss_dict = vicreg_loss_numpy(z_pred, z_tgt)
            
            # Simplified gradient update (would need proper backprop in production)
            # For now, just update EMA
            self.target_encoder.update_ema(self.context_encoder, ema_alpha)
            
            return loss_dict
    
    def precompute_community_embeddings(self):
        """Precompute and cache community embeddings for fast retrieval."""
        communities = self.retriever.communities
        
        # Group nodes by community
        comm_nodes = defaultdict(list)
        for node in self.store.all_nodes():
            comm_id = communities.get(node.id)
            if comm_id is not None and node.embedding:
                comm_nodes[comm_id].append(node)
        
        # Encode each community
        self._community_embeddings = {}
        
        for comm_id, nodes in comm_nodes.items():
            # Stack embeddings
            embeddings = np.array([n.embedding for n in nodes])
            
            if self.use_torch:
                embeddings = torch.from_numpy(embeddings).float()
                with torch.no_grad():
                    z = self.target_encoder(embeddings)
                    # Mean pool
                    z_comm = z.mean(dim=0).cpu().numpy()
            else:
                z = self.target_encoder(embeddings)
                z_comm = z.mean(axis=0)
            
            self._community_embeddings[comm_id] = z_comm
    
    def latent_search(
        self,
        query: str,
        k: int = 3,
        mode: str = 'community'
    ) -> List[Tuple[Union[int, Node], float]]:
        """
        Energy-based latent search.
        
        Args:
            query: Search query
            k: Number of results
            mode: 'community' or 'node'
            
        Returns:
            List of (community_id or node, distance) tuples
        """
        if not self.embedder:
            raise ValueError("Embedder required for latent search")
        
        # Encode query
        query_emb = self.embedder.embed(query)
        
        if self.use_torch:
            query_emb_t = torch.from_numpy(np.array([query_emb])).float()
            with torch.no_grad():
                z_ctx = self.context_encoder(query_emb_t)
                z_pred = self.predictor(z_ctx).squeeze(0).cpu().numpy()
        else:
            z_ctx = self.context_encoder(np.array([query_emb]))
            z_pred = self.predictor(z_ctx).squeeze(0)
        
        if mode == 'community':
            # Search communities
            if not self._community_embeddings:
                self.precompute_community_embeddings()
            
            results = []
            for comm_id, z_comm in self._community_embeddings.items():
                # L2 distance
                dist = np.linalg.norm(z_pred - z_comm)
                results.append((comm_id, dist))
            
            results.sort(key=lambda x: x[1])
            return results[:k]
        
        else:  # node mode
            # Search all nodes
            results = []
            for node in self.store.all_nodes():
                if not node.embedding:
                    continue
                
                if self.use_torch:
                    node_emb_t = torch.from_numpy(np.array([node.embedding])).float()
                    with torch.no_grad():
                        z_node = self.target_encoder(node_emb_t).squeeze(0).cpu().numpy()
                else:
                    z_node = self.target_encoder(np.array([node.embedding])).squeeze(0)
                
                dist = np.linalg.norm(z_pred - z_node)
                results.append((node, dist))
            
            results.sort(key=lambda x: x[1])
            return results[:k]
    
    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        use_local: bool = True,
        use_global: bool = True,
        use_latent: bool = True
    ) -> Dict[str, Any]:
        """
        Hybrid search combining all retrieval modes.
        
        Returns:
            Dictionary with results from each mode
        """
        results = {}
        
        if use_local:
            results['local'] = self.retriever.local_search(query, k=k)
        
        if use_global:
            results['global'] = self.retriever.global_search(query, k=k)
        
        if use_latent:
            results['latent_community'] = self.latent_search(query, k=k, mode='community')
            results['latent_node'] = self.latent_search(query, k=k, mode='node')
        
        return results
