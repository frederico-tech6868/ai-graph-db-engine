//! Louvain community detection over a [`GraphStore`].
//!
//! Pure-Rust port of the Python `CommunityDetector` in
//! `ai_memory/jepa_graphrag.py`. No extra dependencies. The algorithm assigns
//! every node an integer community id by greedily maximising modularity.

use std::collections::{HashMap, HashSet};

use crate::store::GraphStore;

/// Greedy Louvain community detector.
pub struct CommunityDetector {
    /// Resolution parameter (higher => more, smaller communities).
    pub resolution: f64,
}

impl CommunityDetector {
    /// Create a detector with the given resolution.
    pub fn new(resolution: f64) -> Self {
        Self { resolution }
    }

    /// Run the Louvain algorithm, returning a `node_id -> community_id` map.
    ///
    /// Each node starts in its own (index-based) community. Edges are treated
    /// as undirected. Returns an empty map for an empty graph, and the initial
    /// singleton assignment when the graph has no weighted edges.
    pub fn detect_communities(&self, store: &GraphStore) -> HashMap<String, usize> {
        let nodes = store.all_nodes();
        let node_ids: Vec<String> = nodes.iter().map(|n| n.id.clone()).collect();

        if node_ids.is_empty() {
            return HashMap::new();
        }

        // Initialize: each node in its own community.
        let mut node_to_comm: HashMap<String, usize> = HashMap::new();
        for (i, nid) in node_ids.iter().enumerate() {
            node_to_comm.insert(nid.clone(), i);
        }

        // Build weighted degree and undirected edge-weight maps.
        let mut weighted_degree: HashMap<String, f64> = HashMap::new();
        let mut edge_weights: HashMap<(String, String), f64> = HashMap::new();

        for edge in store.all_edges() {
            let w = edge.weight as f64;
            *weighted_degree.entry(edge.src_id.clone()).or_insert(0.0) += w;
            *weighted_degree.entry(edge.dst_id.clone()).or_insert(0.0) += w;
            edge_weights.insert((edge.src_id.clone(), edge.dst_id.clone()), w);
            edge_weights.insert((edge.dst_id.clone(), edge.src_id.clone()), w);
        }

        let total_weight: f64 = weighted_degree.values().sum::<f64>() / 2.0;

        if total_weight == 0.0 {
            return node_to_comm;
        }

        let mut improved = true;
        let mut iteration = 0usize;
        let max_iterations = 100usize;

        while improved && iteration < max_iterations {
            improved = false;
            iteration += 1;

            for node_id in node_ids.iter() {
                // Collect neighbours (undirected).
                let mut neighbors: HashSet<String> = HashSet::new();
                for edge in store.edges_from(node_id) {
                    neighbors.insert(edge.dst_id.clone());
                }
                for edge in store.edges_to(node_id) {
                    neighbors.insert(edge.src_id.clone());
                }

                if neighbors.is_empty() {
                    continue;
                }

                let curr_comm = *node_to_comm.get(node_id).unwrap();

                // Candidate communities among neighbours.
                let neighbor_comms: HashSet<usize> = neighbors
                    .iter()
                    .filter_map(|n| node_to_comm.get(n).copied())
                    .collect();

                let mut best_comm = curr_comm;
                let mut best_gain = 0.0_f64;

                for &comm in neighbor_comms.iter() {
                    let gain = self.modularity_gain(
                        node_id,
                        comm,
                        &node_to_comm,
                        &weighted_degree,
                        &edge_weights,
                        total_weight,
                    );
                    if gain > best_gain {
                        best_gain = gain;
                        best_comm = comm;
                    }
                }

                if best_comm != curr_comm && best_gain > 1e-10 {
                    node_to_comm.insert(node_id.clone(), best_comm);
                    improved = true;
                }
            }
        }

        node_to_comm
    }

    /// Modularity gain from moving `node_id` into `target_comm`.
    fn modularity_gain(
        &self,
        node_id: &str,
        target_comm: usize,
        node_to_comm: &HashMap<String, usize>,
        weighted_degree: &HashMap<String, f64>,
        edge_weights: &HashMap<(String, String), f64>,
        total_weight: f64,
    ) -> f64 {
        let curr_comm = *node_to_comm.get(node_id).unwrap_or(&usize::MAX);
        if curr_comm == target_comm {
            return 0.0;
        }

        // Sum of edge weights from node to members of the target community.
        let mut ki_in = 0.0_f64;
        for (other_id, comm) in node_to_comm.iter() {
            if *comm == target_comm && other_id != node_id {
                ki_in += edge_weights
                    .get(&(node_id.to_string(), other_id.clone()))
                    .copied()
                    .unwrap_or(0.0);
            }
        }

        // Node's weighted degree.
        let ki = weighted_degree.get(node_id).copied().unwrap_or(0.0);

        // Total degree of the target community.
        let sigma_tot: f64 = node_to_comm
            .iter()
            .filter(|(_, c)| **c == target_comm)
            .map(|(nid, _)| weighted_degree.get(nid).copied().unwrap_or(0.0))
            .sum();

        ki_in - self.resolution * sigma_tot * ki / (2.0 * total_weight)
    }
}

/// Invert a `node_id -> community_id` partition into `community_id -> node_ids`.
pub fn group_by_community(partition: &HashMap<String, usize>) -> HashMap<usize, Vec<String>> {
    let mut groups: HashMap<usize, Vec<String>> = HashMap::new();
    for (node_id, comm) in partition.iter() {
        groups.entry(*comm).or_default().push(node_id.clone());
    }
    groups
}
