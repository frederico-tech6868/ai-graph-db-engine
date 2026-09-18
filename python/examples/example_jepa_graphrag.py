"""
Example: JEPA-GraphRAG - Joint-Embedding Predictive Architecture with GraphRAG.

Demonstrates:
1. GraphRAG-style retrieval (community detection, local/global search)
2. Graph-JEPA world model (latent prediction + energy-based retrieval)
3. Hybrid search combining all retrieval modes
"""

import sys
from pathlib import Path

# Bootstrap path
root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from graphdb.core import Node, Edge
from graphdb.store import GraphStore
from ai_memory.embedder import LocalEmbedder
from ai_memory.jepa_graphrag import (
    JEPAGraphRAG,
    CommunityDetector,
    GraphRAGRetriever,
)


def create_knowledge_graph():
    """Create a sample knowledge graph with multiple communities."""
    print("📚 Building knowledge graph...")
    
    store = GraphStore()
    embedder = LocalEmbedder()
    
    # Community 1: Programming Languages
    python = store.add_node(Node(
        label="Language",
        properties={"name": "Python", "type": "interpreted"},
        embedding=embedder.embed("Python is a high-level interpreted programming language")
    ))
    
    rust = store.add_node(Node(
        label="Language",
        properties={"name": "Rust", "type": "compiled"},
        embedding=embedder.embed("Rust is a systems programming language focused on safety")
    ))
    
    javascript = store.add_node(Node(
        label="Language",
        properties={"name": "JavaScript", "type": "interpreted"},
        embedding=embedder.embed("JavaScript is a web programming language for browsers")
    ))
    
    # Connect languages
    store.add_edge(Edge(src_id=python.id, dst_id=rust.id, label="INFLUENCED_BY", weight=0.3))
    store.add_edge(Edge(src_id=javascript.id, dst_id=python.id, label="SIMILAR_TO", weight=0.6))
    
    # Community 2: AI/ML Frameworks
    pytorch = store.add_node(Node(
        label="Framework",
        properties={"name": "PyTorch", "domain": "ML"},
        embedding=embedder.embed("PyTorch is a deep learning framework for Python")
    ))
    
    tensorflow = store.add_node(Node(
        label="Framework",
        properties={"name": "TensorFlow", "domain": "ML"},
        embedding=embedder.embed("TensorFlow is a machine learning framework by Google")
    ))
    
    # Connect frameworks
    store.add_edge(Edge(src_id=pytorch.id, dst_id=tensorflow.id, label="COMPETES_WITH", weight=0.8))
    
    # Cross-community connections
    store.add_edge(Edge(src_id=python.id, dst_id=pytorch.id, label="SUPPORTS", weight=1.0))
    store.add_edge(Edge(src_id=python.id, dst_id=tensorflow.id, label="SUPPORTS", weight=1.0))
    
    # Community 3: Developers
    alice = store.add_node(Node(
        label="Developer",
        properties={"name": "Alice", "role": "ML Engineer"},
        embedding=embedder.embed("Alice is a machine learning engineer specializing in PyTorch")
    ))
    
    bob = store.add_node(Node(
        label="Developer",
        properties={"name": "Bob", "role": "Systems Engineer"},
        embedding=embedder.embed("Bob is a systems engineer working with Rust and C++")
    ))
    
    # Connect developers to tools
    store.add_edge(Edge(src_id=alice.id, dst_id=pytorch.id, label="USES", weight=1.0))
    store.add_edge(Edge(src_id=alice.id, dst_id=python.id, label="USES", weight=1.0))
    store.add_edge(Edge(src_id=bob.id, dst_id=rust.id, label="USES", weight=1.0))
    
    print(f"✅ Created graph with {len(store.all_nodes())} nodes and {len(store.all_edges())} edges\n")
    
    return store, embedder


def demonstrate_community_detection(store):
    """Demonstrate Louvain community detection."""
    print("🔍 === Community Detection ===")
    
    detector = CommunityDetector(store)
    communities = detector.detect_communities()
    
    # Group nodes by community
    from collections import defaultdict
    comm_groups = defaultdict(list)
    for node in store.all_nodes():
        comm_id = communities.get(node.id)
        if comm_id is not None:
            comm_groups[comm_id].append(node)
    
    print(f"Found {len(comm_groups)} communities:")
    for comm_id, nodes in comm_groups.items():
        labels = [n.label for n in nodes]
        names = [n.properties.get('name', 'unknown') for n in nodes]
        print(f"  Community {comm_id}: {len(nodes)} nodes")
        print(f"    Labels: {', '.join(set(labels))}")
        print(f"    Members: {', '.join(names)}")
    print()


def demonstrate_graphrag_retrieval(store, embedder):
    """Demonstrate GraphRAG local and global search."""
    print("🔎 === GraphRAG Retrieval ===")
    
    retriever = GraphRAGRetriever(store, embedder)
    
    # Local search (k-hop expansion)
    query = "deep learning framework"
    print(f"Local Search: '{query}'")
    local_results = retriever.local_search(query, k=3, max_hops=2)
    
    print(f"  Found {len(local_results)} results:")
    for node, score in local_results[:5]:
        print(f"    {score:.3f} - {node.label}: {node.properties.get('name', 'N/A')}")
    print()
    
    # Global search (community-based)
    print(f"Global Search: '{query}'")
    global_results = retriever.global_search(query, k=2)
    
    print(f"  Found {len(global_results)} communities:")
    for comm_id, nodes, score in global_results:
        print(f"    Community {comm_id} (score: {score:.3f})")
        for node in nodes[:3]:
            print(f"      - {node.label}: {node.properties.get('name', 'N/A')}")
    print()


def demonstrate_jepa_world_model(store, embedder):
    """Demonstrate Graph-JEPA latent prediction and retrieval."""
    print("🧠 === Graph-JEPA World Model ===")
    
    # Initialize JEPA system (numpy backend for lightweight demo)
    jepa = JEPAGraphRAG(store, embedder, latent_dim=64, use_torch=False)
    
    # Train the world model with a few examples
    print("Training JEPA world model...")
    contexts = [
        "machine learning engineer",
        "systems programming",
        "web development"
    ]
    targets = [
        "PyTorch deep learning",
        "Rust safety performance",
        "JavaScript browser"
    ]
    
    for i in range(3):
        loss_dict = jepa.train_step(
            contexts, targets,
            learning_rate=0.001,
            ema_alpha=0.95
        )
        if i % 1 == 0:
            print(f"  Epoch {i+1}: loss={loss_dict['total']:.4f} "
                  f"(sim={loss_dict['sim']:.4f}, var={loss_dict['var']:.4f}, "
                  f"cov={loss_dict['cov']:.4f})")
    print()
    
    # Precompute community embeddings for fast retrieval
    print("Precomputing community embeddings...")
    jepa.precompute_community_embeddings()
    print(f"  Cached {len(jepa._community_embeddings)} community embeddings\n")
    
    # Latent search (energy-based)
    query = "machine learning with Python"
    print(f"Latent Search (community mode): '{query}'")
    latent_results = jepa.latent_search(query, k=3, mode='community')
    
    print(f"  Top communities by latent distance:")
    for comm_id, distance in latent_results:
        print(f"    Community {comm_id}: distance={distance:.4f}")
    print()
    
    # Node-level latent search
    print(f"Latent Search (node mode): '{query}'")
    node_results = jepa.latent_search(query, k=5, mode='node')
    
    print(f"  Top nodes by latent distance:")
    for node, distance in node_results:
        print(f"    {node.label}: {node.properties.get('name', 'N/A')} (distance={distance:.4f})")
    print()


def demonstrate_hybrid_search(store, embedder):
    """Demonstrate hybrid search combining all retrieval modes."""
    print("🔀 === Hybrid Search ===")
    
    jepa = JEPAGraphRAG(store, embedder, latent_dim=64, use_torch=False)
    
    # Precompute embeddings
    jepa.precompute_community_embeddings()
    
    # Hybrid search
    query = "Python deep learning engineer"
    print(f"Query: '{query}'\n")
    
    results = jepa.hybrid_search(
        query,
        k=3,
        use_local=True,
        use_global=True,
        use_latent=True
    )
    
    print("Local Search Results:")
    for node, score in results['local'][:3]:
        print(f"  {score:.3f} - {node.label}: {node.properties.get('name', 'N/A')}")
    print()
    
    print("Global Search Results:")
    for comm_id, nodes, score in results['global'][:2]:
        print(f"  Community {comm_id} (score: {score:.3f})")
    print()
    
    print("Latent Community Results:")
    for comm_id, distance in results['latent_community'][:3]:
        print(f"  Community {comm_id}: distance={distance:.4f}")
    print()
    
    print("Latent Node Results:")
    for node, distance in results['latent_node'][:3]:
        print(f"  {node.label}: {node.properties.get('name', 'N/A')} (distance={distance:.4f})")
    print()


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("JEPA-GraphRAG Demo")
    print("Joint-Embedding Predictive Architecture + GraphRAG")
    print("=" * 60)
    print()
    
    # Create knowledge graph
    store, embedder = create_knowledge_graph()
    
    # 1. Community Detection
    demonstrate_community_detection(store)
    
    # 2. GraphRAG Retrieval
    demonstrate_graphrag_retrieval(store, embedder)
    
    # 3. JEPA World Model
    demonstrate_jepa_world_model(store, embedder)
    
    # 4. Hybrid Search
    demonstrate_hybrid_search(store, embedder)
    
    print("=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
