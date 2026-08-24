//! Integration tests for the pure-Rust graphdb_rs engine.

use graphdb_rs::core::{Edge, Node, PropertyValue};
use graphdb_rs::persistence;
use graphdb_rs::query::{bfs, dfs, find_path};
use graphdb_rs::store::GraphStore;
use graphdb_rs::vector::{cosine_similarity, top_k_similar};

fn node_with_emb(label: &str, emb: Vec<f32>) -> Node {
    Node::new(label).with_embedding(emb)
}

#[test]
fn test_label_index_and_nodes_by_label() {
    let mut store = GraphStore::new();
    let u1 = store.add_node(Node::new("User")).unwrap();
    let u2 = store.add_node(Node::new("User")).unwrap();
    let _p1 = store.add_node(Node::new("Post")).unwrap();

    let users = store.nodes_by_label("User");
    assert_eq!(users.len(), 2);
    let ids: Vec<&str> = users.iter().map(|n| n.id.as_str()).collect();
    assert!(ids.contains(&u1.id.as_str()));
    assert!(ids.contains(&u2.id.as_str()));

    assert_eq!(store.nodes_by_label("Post").len(), 1);
    assert_eq!(store.nodes_by_label("Missing").len(), 0);
    assert_eq!(store.node_count(), 3);
}

#[test]
fn test_similarity_scanner_finds_match_above_threshold() {
    let mut store = GraphStore::new();
    // Two users, two posts with near-identical embeddings.
    let ua = store.add_node(node_with_emb("User", vec![1.0, 0.0, 0.0])).unwrap();
    let pa = store.add_node(node_with_emb("Post", vec![0.0, 1.0, 0.0])).unwrap();
    let ub = store
        .add_node(node_with_emb("User", vec![0.99, 0.01, 0.0]))
        .unwrap();
    let pb = store
        .add_node(node_with_emb("Post", vec![0.01, 0.99, 0.0]))
        .unwrap();

    // First edge: no similar edges yet.
    let r1 = store
        .add_edge(Edge::new(&ua.id, &pa.id, "LIKES"), 0.85)
        .unwrap();
    assert!(!r1.was_flagged);
    assert!(r1.similar_edges.is_empty());

    // Second edge between very similar endpoints -> flagged.
    let r2 = store
        .add_edge(Edge::new(&ub.id, &pb.id, "LIKES"), 0.85)
        .unwrap();
    assert!(r2.was_flagged);
    assert_eq!(r2.similar_edges.len(), 1);
    assert_eq!(r2.similar_edges[0].existing_edge_id, r1.edge.id);
    assert!(r2.similar_edges[0].combined_score >= 0.85);
}

#[test]
fn test_similarity_is_label_scoped() {
    // Nodes with different text labels must NEVER be compared even if their
    // embeddings are identical.
    let mut store = GraphStore::new();
    let ua = store.add_node(node_with_emb("User", vec![1.0, 0.0])).unwrap();
    let pa = store.add_node(node_with_emb("Post", vec![0.0, 1.0])).unwrap();
    let _e1 = store
        .add_edge(Edge::new(&ua.id, &pa.id, "LIKES"), 0.85)
        .unwrap();

    // New endpoints have identical embeddings but DIFFERENT labels.
    let ca = store
        .add_node(node_with_emb("Company", vec![1.0, 0.0]))
        .unwrap();
    let da = store
        .add_node(node_with_emb("Document", vec![0.0, 1.0]))
        .unwrap();
    let r = store
        .add_edge(Edge::new(&ca.id, &da.id, "LIKES"), 0.85)
        .unwrap();
    // Label mismatch => no matches.
    assert!(!r.was_flagged);
    assert!(r.similar_edges.is_empty());
}

#[test]
fn test_similarity_respects_edge_label() {
    let mut store = GraphStore::new();
    let ua = store.add_node(node_with_emb("User", vec![1.0, 0.0])).unwrap();
    let pa = store.add_node(node_with_emb("Post", vec![0.0, 1.0])).unwrap();
    store
        .add_edge(Edge::new(&ua.id, &pa.id, "LIKES"), 0.85)
        .unwrap();

    let ub = store.add_node(node_with_emb("User", vec![1.0, 0.0])).unwrap();
    let pb = store.add_node(node_with_emb("Post", vec![0.0, 1.0])).unwrap();
    // Different edge label -> not compared.
    let r = store
        .add_edge(Edge::new(&ub.id, &pb.id, "SHARES"), 0.85)
        .unwrap();
    assert!(!r.was_flagged);
}

#[test]
fn test_bfs_dfs_order() {
    // a -> b -> c -> d
    let mut store = GraphStore::new();
    let a = store.add_node(Node::new("N")).unwrap();
    let b = store.add_node(Node::new("N")).unwrap();
    let c = store.add_node(Node::new("N")).unwrap();
    let d = store.add_node(Node::new("N")).unwrap();
    store.add_edge(Edge::new(&a.id, &b.id, "E"), 1.0).unwrap();
    store.add_edge(Edge::new(&b.id, &c.id, "E"), 1.0).unwrap();
    store.add_edge(Edge::new(&c.id, &d.id, "E"), 1.0).unwrap();

    let order = bfs(&store, &a.id, None, 5).unwrap();
    assert_eq!(order, vec![b.id.clone(), c.id.clone(), d.id.clone()]);

    // Depth limit.
    let limited = bfs(&store, &a.id, None, 2).unwrap();
    assert_eq!(limited, vec![b.id.clone(), c.id.clone()]);

    let dfs_order = dfs(&store, &a.id, None, 5).unwrap();
    assert_eq!(dfs_order, vec![b.id.clone(), c.id.clone(), d.id.clone()]);
}

#[test]
fn test_find_path() {
    let mut store = GraphStore::new();
    let a = store.add_node(Node::new("N")).unwrap();
    let b = store.add_node(Node::new("N")).unwrap();
    let c = store.add_node(Node::new("N")).unwrap();
    let d = store.add_node(Node::new("N")).unwrap();
    // a->b->c->d and a->d direct? Make shortest a->b->d
    store.add_edge(Edge::new(&a.id, &b.id, "E"), 1.0).unwrap();
    store.add_edge(Edge::new(&b.id, &c.id, "E"), 1.0).unwrap();
    store.add_edge(Edge::new(&c.id, &d.id, "E"), 1.0).unwrap();

    let path = find_path(&store, &a.id, &d.id).unwrap().unwrap();
    assert_eq!(path, vec![a.id.clone(), b.id.clone(), c.id.clone(), d.id.clone()]);

    // Same node.
    let self_path = find_path(&store, &a.id, &a.id).unwrap().unwrap();
    assert_eq!(self_path, vec![a.id.clone()]);

    // Unreachable (d -> a).
    assert!(find_path(&store, &d.id, &a.id).unwrap().is_none());
}

#[test]
fn test_save_load_round_trip() {
    let dir = std::env::temp_dir();
    let path = dir.join(format!("graphdb_rs_test_{}.json", std::process::id()));
    let path_str = path.to_str().unwrap().to_string();

    let mut store = GraphStore::with_path(&path_str);
    let mut n1 = Node::new("User").with_embedding(vec![0.1, 0.2, 0.3]);
    n1.set_property("name", PropertyValue::Str("Alice".into()));
    n1.set_property("age", PropertyValue::Int(30));
    let n1 = store.add_node(n1).unwrap();
    let n2 = store.add_node(Node::new("Post")).unwrap();
    store.add_edge(Edge::new(&n1.id, &n2.id, "WROTE"), 1.0).unwrap();

    store.save().unwrap();

    let mut store2 = GraphStore::with_path(&path_str);
    store2.load().unwrap();
    assert_eq!(store2.node_count(), 2);
    assert_eq!(store2.edge_count(), 1);
    let loaded = store2.get_node(&n1.id).unwrap();
    assert_eq!(loaded.label, "User");
    assert_eq!(loaded.embedding, Some(vec![0.1, 0.2, 0.3]));
    assert_eq!(
        loaded.properties.get("name"),
        Some(&PropertyValue::Str("Alice".into()))
    );
    // Adjacency preserved.
    assert_eq!(store2.edges_from(&n1.id).len(), 1);

    // Direct persistence helpers.
    let (nodes, edges) = persistence::load_from_file(&path_str).unwrap();
    assert_eq!(nodes.len(), 2);
    assert_eq!(edges.len(), 1);

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_top_k_similar_ordering() {
    let query = vec![1.0, 0.0, 0.0];
    let a = vec![1.0, 0.0, 0.0]; // score 1.0
    let b = vec![0.0, 1.0, 0.0]; // score 0.0
    let c = vec![0.7, 0.7, 0.0]; // score ~0.707
    let candidates: Vec<(&str, &[f32])> = vec![
        ("a", a.as_slice()),
        ("b", b.as_slice()),
        ("c", c.as_slice()),
    ];
    let top = top_k_similar(&query, &candidates, 2).unwrap();
    assert_eq!(top.len(), 2);
    assert_eq!(top[0].0, "a");
    assert_eq!(top[1].0, "c");
    assert!(top[0].1 > top[1].1);
}

#[test]
fn test_cosine_similarity_correctness() {
    // Parallel vectors -> 1.0
    let s = cosine_similarity(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]).unwrap();
    assert!((s - 1.0).abs() < 1e-6);

    // Orthogonal vectors -> 0.0
    let o = cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).unwrap();
    assert!(o.abs() < 1e-6);

    // Opposite vectors -> -1.0
    let n = cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]).unwrap();
    assert!((n + 1.0).abs() < 1e-6);

    // Zero vector -> 0.0
    let z = cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]).unwrap();
    assert_eq!(z, 0.0);

    // Dimension mismatch -> error
    assert!(cosine_similarity(&[1.0], &[1.0, 2.0]).is_err());
}

#[test]
fn test_search_similar_nodes_label_scoped() {
    let mut store = GraphStore::new();
    let u1 = store.add_node(node_with_emb("User", vec![1.0, 0.0])).unwrap();
    let _u2 = store.add_node(node_with_emb("User", vec![0.0, 1.0])).unwrap();
    let _p1 = store.add_node(node_with_emb("Post", vec![1.0, 0.0])).unwrap();

    let results = store
        .search_similar_nodes(&[1.0, 0.0], Some("User"), 5)
        .unwrap();
    // Only User nodes returned.
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, u1.id);
    assert!((results[0].1 - 1.0).abs() < 1e-6);
}

#[test]
fn test_delete_node_cascades_edges() {
    let mut store = GraphStore::new();
    let a = store.add_node(Node::new("N")).unwrap();
    let b = store.add_node(Node::new("N")).unwrap();
    store.add_edge(Edge::new(&a.id, &b.id, "E"), 1.0).unwrap();
    assert_eq!(store.edge_count(), 1);
    store.delete_node(&a.id).unwrap();
    assert_eq!(store.node_count(), 1);
    assert_eq!(store.edge_count(), 0);
    assert!(store.get_node(&a.id).is_err());
}
