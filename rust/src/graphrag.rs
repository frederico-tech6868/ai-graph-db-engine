//! GraphRAG-style retrieval over a [`GraphStore`].
//!
//! Pure-Rust port of the Python `GraphRAGRetriever`. Provides:
//! * `local_search` — vector-seed + k-hop neighbourhood expansion, re-scored by
//!   cosine similarity.
//! * `global_search` — community-level retrieval scored by mean member
//!   similarity.
//! * `get_community_summary` — a short text summary of a community.
//!
//! Unlike the Python version (which embeds text via an embedder), this Rust API
//! takes a pre-computed query embedding directly, matching the crate's
//! embedding-first design.

use std::collections::{HashMap, HashSet};

use crate::community::CommunityDetector;
use crate::core::Node;
use crate::error::Result;
use crate::store::GraphStore;
use crate::vector::cosine_similarity;

/// GraphRAG retriever holding a cached community partition.
pub struct GraphRAGRetriever {
    /// Resolution used when detecting communities.
    pub resolution: f64,
    community_cache: Option<HashMap<String, usize>>,
}

impl Default for GraphRAGRetriever {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl GraphRAGRetriever {
    /// Create a retriever with the given community-detection resolution.
    pub fn new(resolution: f64) -> Self {
        Self {
            resolution,
            community_cache: None,
        }
    }

    /// Get the (lazily computed & cached) community partition.
    pub fn communities(&mut self, store: &GraphStore) -> &HashMap<String, usize> {
        if self.community_cache.is_none() {
            let partition = CommunityDetector::new(self.resolution).detect_communities(store);
            self.community_cache = Some(partition);
        }
        self.community_cache.as_ref().unwrap()
    }

    /// Force a rebuild of the community partition.
    pub fn rebuild_communities(&mut self, store: &GraphStore) {
        let partition = CommunityDetector::new(self.resolution).detect_communities(store);
        self.community_cache = Some(partition);
    }

    /// Local search: seed via vector search, expand k-hop, re-score by cosine.
    ///
    /// Returns `(node_id, score)` pairs sorted by descending similarity.
    pub fn local_search(
        &mut self,
        store: &GraphStore,
        query_emb: &[f32],
        k: usize,
        max_hops: usize,
        label: Option<&str>,
    ) -> Result<Vec<(String, f32)>> {
        let seeds = store.search_similar_nodes(query_emb, label, k)?;
        if seeds.is_empty() {
            return Ok(Vec::new());
        }

        // Expand neighbourhoods around every seed.
        let mut expanded: HashSet<String> = HashSet::new();
        for (node_id, _score) in seeds.iter() {
            expanded.insert(node_id.clone());
            self.expand_neighborhood(store, node_id, max_hops, &mut expanded);
        }

        // Re-score every expanded node against the query.
        let mut results: Vec<(String, f32)> = Vec::new();
        for node_id in expanded.iter() {
            if let Ok(node) = store.get_node(node_id) {
                if let Some(emb) = &node.embedding {
                    let score = cosine_similarity(query_emb, emb)?;
                    results.push((node_id.clone(), score));
                }
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }

    /// BFS expansion of a node's neighbourhood up to `max_hops`.
    fn expand_neighborhood(
        &self,
        store: &GraphStore,
        node_id: &str,
        max_hops: usize,
        visited: &mut HashSet<String>,
    ) {
        if max_hops == 0 {
            return;
        }

        let mut neighbors: HashSet<String> = HashSet::new();
        for edge in store.edges_from(node_id) {
            neighbors.insert(edge.dst_id.clone());
        }
        for edge in store.edges_to(node_id) {
            neighbors.insert(edge.src_id.clone());
        }

        for neighbor_id in neighbors {
            if !visited.contains(&neighbor_id) {
                visited.insert(neighbor_id.clone());
                self.expand_neighborhood(store, &neighbor_id, max_hops - 1, visited);
            }
        }
    }

    /// Global search: score each community by the mean similarity of its
    /// embedded members. Returns `(community_id, node_ids, mean_score)` sorted
    /// by descending score, truncated to `k`.
    pub fn global_search(
        &mut self,
        store: &GraphStore,
        query_emb: &[f32],
        k: usize,
        label: Option<&str>,
    ) -> Result<Vec<(usize, Vec<String>, f32)>> {
        // Clone the cached partition to avoid holding a borrow of `self`.
        let partition: HashMap<String, usize> = {
            if self.community_cache.is_none() {
                self.rebuild_communities(store);
            }
            self.community_cache.as_ref().unwrap().clone()
        };

        // Group nodes by community (respecting the optional label filter).
        let mut comm_nodes: HashMap<usize, Vec<String>> = HashMap::new();
        for node in store.all_nodes() {
            if let Some(l) = label {
                if node.label != l {
                    continue;
                }
            }
            if let Some(&comm_id) = partition.get(&node.id) {
                comm_nodes.entry(comm_id).or_default().push(node.id.clone());
            }
        }

        // Score each community.
        let mut comm_scores: Vec<(usize, Vec<String>, f32)> = Vec::new();
        for (comm_id, node_ids) in comm_nodes.iter() {
            let mut scores: Vec<f32> = Vec::new();
            for nid in node_ids.iter() {
                if let Ok(node) = store.get_node(nid) {
                    if let Some(emb) = &node.embedding {
                        scores.push(cosine_similarity(query_emb, emb)?);
                    }
                }
            }
            if scores.is_empty() {
                continue;
            }
            let mean_score = scores.iter().sum::<f32>() / scores.len() as f32;
            comm_scores.push((*comm_id, node_ids.clone(), mean_score));
        }

        comm_scores.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        comm_scores.truncate(k);
        Ok(comm_scores)
    }

    /// Produce a short text summary for a community.
    pub fn get_community_summary(&mut self, store: &GraphStore, community_id: usize) -> String {
        let partition: HashMap<String, usize> = {
            if self.community_cache.is_none() {
                self.rebuild_communities(store);
            }
            self.community_cache.as_ref().unwrap().clone()
        };

        let nodes: Vec<&Node> = store
            .all_nodes()
            .into_iter()
            .filter(|n| partition.get(&n.id) == Some(&community_id))
            .collect();

        if nodes.is_empty() {
            return format!("Community {}: (empty)", community_id);
        }

        // Aggregate labels.
        let mut labels: HashMap<String, usize> = HashMap::new();
        for node in nodes.iter() {
            *labels.entry(node.label.clone()).or_insert(0) += 1;
        }

        let mut parts: Vec<String> = Vec::new();
        parts.push(format!("Community {}:", community_id));
        parts.push(format!("  {} nodes", nodes.len()));
        for (label, count) in labels.iter() {
            parts.push(format!("  - {} {} nodes", count, label));
        }
        parts.join("\n")
    }
}
