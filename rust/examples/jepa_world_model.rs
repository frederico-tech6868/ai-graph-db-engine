//! # JEPA world-model example
//!
//! A **Joint-Embedding Predictive Architecture (JEPA)** for graphs. Instead of
//! reconstructing raw features, JEPA learns to *predict the latent
//! representation* of a target from the latent representation of a context,
//! trained with the **VICReg** loss (invariance + variance + covariance) to
//! avoid collapse. Here it is combined with GraphRAG so you can retrieve over
//! communities and nodes in the learned latent space.
//!
//! What it demonstrates, end to end:
//! 1. building a small property graph in a [`GraphStore`],
//! 2. running JEPA training steps and watching the VICReg loss terms,
//! 3. precomputing latent community embeddings,
//! 4. latent search over communities and nodes,
//! 5. a hybrid search combining local + global (GraphRAG) + latent retrieval.
//!
//! Run it:
//! ```bash
//! cargo run -p graphdb_rs --example jepa_world_model
//! ```

use graphdb_rs::core::{Edge, Node};
use graphdb_rs::jepa::JEPAGraphRAG;
use graphdb_rs::store::GraphStore;
use ndarray::Array2;

/// Embedding dimension for node features in this demo.
const INPUT_DIM: usize = 8;
/// JEPA latent dimension.
const LATENT_DIM: usize = 16;

fn main() {
    // ---------------------------------------------------------------------
    // 1. Build a small graph with two clusters of related nodes.
    // ---------------------------------------------------------------------
    let mut store = GraphStore::new();

    // Cluster A ("systems") embeddings point roughly one way,
    // cluster B ("audio") the other, so communities are separable.
    let cluster_a = [
        ("Rust", basis(INPUT_DIM, 0)),
        ("Concurrency", basis(INPUT_DIM, 1)),
        ("Memory", basis(INPUT_DIM, 2)),
    ];
    let cluster_b = [
        ("Whisper", basis(INPUT_DIM, 4)),
        ("Audio", basis(INPUT_DIM, 5)),
        ("Speech", basis(INPUT_DIM, 6)),
    ];

    let mut ids = Vec::new();
    for (label, emb) in cluster_a.iter().chain(cluster_b.iter()) {
        let node = Node::new(*label).with_embedding(emb.clone());
        let stored = store.add_node(node).expect("add node");
        ids.push(stored.id);
    }

    // Connect nodes within each cluster (dense) to form communities.
    connect(&mut store, &ids[0..3]);
    connect(&mut store, &ids[3..6]);

    println!(
        "graph: {} nodes, {} edges\n",
        store.node_count(),
        store.edge_count()
    );

    // ---------------------------------------------------------------------
    // 2. Train the JEPA world model for a few steps.
    // ---------------------------------------------------------------------
    let mut jepa = JEPAGraphRAG::new(INPUT_DIM, LATENT_DIM, 1.0);

    // Use node embeddings as both context and (shifted) target batches.
    let all: Vec<Vec<f32>> = store
        .all_nodes()
        .iter()
        .filter_map(|n| n.embedding.clone())
        .collect();
    let batch = all.len();
    let ctx = Array2::from_shape_vec((batch, INPUT_DIM), all.concat()).expect("ctx shape");
    // Target = context rotated by one row (predict a neighbour's latent).
    let mut rotated = all.clone();
    rotated.rotate_left(1);
    let tgt = Array2::from_shape_vec((batch, INPUT_DIM), rotated.concat()).expect("tgt shape");

    // Note: this pure-Rust port computes the forward pass + VICReg loss and
    // EMA-updates the target encoder each step, but does not run a gradient
    // optimizer on the context encoder / predictor. The reported loss therefore
    // reflects the (fixed, deterministically-initialised) weights rather than a
    // decreasing training curve — it is shown to illustrate the four VICReg
    // terms. Wire in an optimizer to actually minimise it.
    println!("== JEPA training (VICReg loss terms) ==");
    for step in 1..=5 {
        let loss = jepa.train_step(ctx.clone(), tgt.clone(), 0.99);
        println!(
            "  step {step}: total={:.4}  sim={:.4}  var={:.4}  cov={:.4}",
            loss.total, loss.sim, loss.var, loss.cov
        );
    }
    println!();

    // ---------------------------------------------------------------------
    // 3. Precompute latent community embeddings.
    // ---------------------------------------------------------------------
    jepa.precompute_community_embeddings(&store);

    // ---------------------------------------------------------------------
    // 4. Latent search: query with a "systems"-like vector.
    // ---------------------------------------------------------------------
    let query = basis(INPUT_DIM, 0); // aligned with cluster A
    println!("== latent community search ==");
    for (comm_id, dist) in jepa.latent_search_communities(&query, 3) {
        println!("  community {comm_id}: L2={dist:.4}");
    }

    println!("\n== latent node search ==");
    for (node_id, dist) in jepa.latent_search_nodes(&store, &query, 3) {
        let label = store.get_node(&node_id).map(|n| n.label.clone()).unwrap_or_default();
        println!("  {label:<12} L2={dist:.4}");
    }

    // ---------------------------------------------------------------------
    // 5. Hybrid search: local + global (GraphRAG) + latent together.
    // ---------------------------------------------------------------------
    println!("\n== hybrid search (local + global + latent) ==");
    let result = jepa
        .hybrid_search(&store, &query, 3, true, true, true)
        .expect("hybrid search");

    if let Some(local) = &result.local {
        println!("  local:");
        for (id, score) in local {
            let label = store.get_node(id).map(|n| n.label.clone()).unwrap_or_default();
            println!("    {label:<12} score={score:.4}");
        }
    }
    if let Some(latent_nodes) = &result.latent_nodes {
        println!("  latent nodes:");
        for (id, dist) in latent_nodes {
            let label = store.get_node(id).map(|n| n.label.clone()).unwrap_or_default();
            println!("    {label:<12} L2={dist:.4}");
        }
    }
    if let Some(latent_comms) = &result.latent_communities {
        println!("  latent communities: {latent_comms:?}");
    }
}

/// A unit basis vector of length `dim` with `1.0` at position `k`.
fn basis(dim: usize, k: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    v[k % dim] = 1.0;
    v
}

/// Add undirected "related" edges between every pair in `ids`.
fn connect(store: &mut GraphStore, ids: &[String]) {
    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            store
                .add_edge(Edge::new(&ids[i], &ids[j], "related"), 0.0)
                .expect("add edge");
            store
                .add_edge(Edge::new(&ids[j], &ids[i], "related"), 0.0)
                .expect("add edge");
        }
    }
}
