//! [`GraphStore`]: the central in-memory property-graph engine.
//!
//! Provides CRUD for nodes and edges, adjacency maps, label/property indexes,
//! label-scoped vector search and JSON persistence. A `parking_lot::RwLock`
//! guards concurrent access (see the module note in `README_RUST.md`).

use std::collections::HashMap;

use parking_lot::RwLock;

use crate::core::{Edge, Node, PropertyValue};
use crate::error::{GraphError, Result};
use crate::index::{LabelIndex, PropertyIndex};
use crate::persistence;
use crate::similarity::{SimilarMatch, SimilarityScanner};
use crate::vector::top_k_similar;

/// Result of an [`GraphStore::add_edge`] call.
pub struct AddEdgeResult {
    pub edge: Edge,
    pub similar_edges: Vec<SimilarMatch>,
    /// `true` when at least one similar existing edge was found.
    pub was_flagged: bool,
}

/// The main in-memory graph store.
pub struct GraphStore {
    nodes: HashMap<String, Node>,
    edges: HashMap<String, Edge>,
    adjacency_out: HashMap<String, Vec<String>>,
    adjacency_in: HashMap<String, Vec<String>>,
    label_index: LabelIndex,
    property_index: PropertyIndex,
    /// Concurrency guard. Held for the duration of higher-level operations by
    /// callers that need atomicity across multiple stores; individual methods
    /// take `&self`/`&mut self` and rely on Rust's borrow rules.
    lock: RwLock<()>,
    persistence_path: Option<String>,
}

impl Default for GraphStore {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphStore {
    /// Create an empty store with no persistence path.
    pub fn new() -> Self {
        GraphStore {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            adjacency_out: HashMap::new(),
            adjacency_in: HashMap::new(),
            label_index: LabelIndex::new(),
            property_index: PropertyIndex::new(),
            lock: RwLock::new(()),
            persistence_path: None,
        }
    }

    /// Create an empty store bound to a JSON persistence path.
    pub fn with_path(path: impl Into<String>) -> Self {
        let mut s = Self::new();
        s.persistence_path = Some(path.into());
        s
    }

    /// Access to the label index (used by the similarity scanner).
    pub fn label_index(&self) -> &LabelIndex {
        &self.label_index
    }

    // ----- node operations -------------------------------------------------

    /// Insert a node, returning a clone of the stored node.
    pub fn add_node(&mut self, node: Node) -> Result<Node> {
        let _guard = self.lock.write();
        // Operate on disjoint fields directly so the lock guard (which borrows
        // only `self.lock`) does not conflict with mutation of other fields.
        index_node(&mut self.label_index, &mut self.property_index, &node);
        let id = node.id.clone();
        self.adjacency_out.entry(id.clone()).or_default();
        self.adjacency_in.entry(id.clone()).or_default();
        self.nodes.insert(id.clone(), node);
        Ok(self.nodes.get(&id).unwrap().clone())
    }

    /// Get a reference to a node by id.
    pub fn get_node(&self, id: &str) -> Result<&Node> {
        self.nodes
            .get(id)
            .ok_or_else(|| GraphError::NodeNotFound(id.to_string()))
    }

    /// Get a mutable reference to a node by id.
    ///
    /// Note: mutating properties directly bypasses the property index. Prefer
    /// deleting and re-adding a node if indexed properties change.
    pub fn get_node_mut(&mut self, id: &str) -> Result<&mut Node> {
        self.nodes
            .get_mut(id)
            .ok_or_else(|| GraphError::NodeNotFound(id.to_string()))
    }

    /// Delete a node and cascade-delete all connected edges.
    pub fn delete_node(&mut self, id: &str) -> Result<()> {
        let _guard = self.lock.write();
        let node = self
            .nodes
            .remove(id)
            .ok_or_else(|| GraphError::NodeNotFound(id.to_string()))?;
        deindex_node(&mut self.label_index, &mut self.property_index, &node);

        // Collect all edges touching this node.
        let mut edge_ids: Vec<String> = Vec::new();
        if let Some(out) = self.adjacency_out.get(id) {
            edge_ids.extend(out.iter().cloned());
        }
        if let Some(inc) = self.adjacency_in.get(id) {
            edge_ids.extend(inc.iter().cloned());
        }
        for eid in edge_ids {
            remove_edge_internal(
                &mut self.edges,
                &mut self.adjacency_out,
                &mut self.adjacency_in,
                &eid,
            );
        }
        self.adjacency_out.remove(id);
        self.adjacency_in.remove(id);
        Ok(())
    }

    /// Return all nodes carrying the given label.
    pub fn nodes_by_label(&self, label: &str) -> Vec<&Node> {
        match self.label_index.get(label) {
            Some(ids) => ids.iter().filter_map(|i| self.nodes.get(i)).collect(),
            None => Vec::new(),
        }
    }

    /// Return all nodes.
    pub fn all_nodes(&self) -> Vec<&Node> {
        self.nodes.values().collect()
    }

    // ----- edge operations -------------------------------------------------

    /// Add an edge, running the [`SimilarityScanner`] first.
    ///
    /// Returns an [`AddEdgeResult`] carrying the inserted edge plus any similar
    /// existing edges found at or above `similarity_threshold`.
    pub fn add_edge(&mut self, edge: Edge, similarity_threshold: f32) -> Result<AddEdgeResult> {
        let _guard = self.lock.write();

        // Both endpoints must exist.
        let src = self
            .nodes
            .get(&edge.src_id)
            .ok_or_else(|| GraphError::NodeNotFound(edge.src_id.clone()))?
            .clone();
        let dst = self
            .nodes
            .get(&edge.dst_id)
            .ok_or_else(|| GraphError::NodeNotFound(edge.dst_id.clone()))?
            .clone();

        // Run the label-scoped similarity scan before mutating.
        let similar_edges = {
            let scanner = SimilarityScanner::new(self);
            scanner.scan_before_add(&src, &dst, &edge.label, similarity_threshold)?
        };
        let was_flagged = !similar_edges.is_empty();

        // Insert into adjacency maps and edge table.
        self.adjacency_out
            .entry(edge.src_id.clone())
            .or_default()
            .push(edge.id.clone());
        self.adjacency_in
            .entry(edge.dst_id.clone())
            .or_default()
            .push(edge.id.clone());
        let stored = edge.clone();
        self.edges.insert(edge.id.clone(), edge);

        Ok(AddEdgeResult {
            edge: stored,
            similar_edges,
            was_flagged,
        })
    }

    /// Get an edge by id.
    pub fn get_edge(&self, id: &str) -> Result<&Edge> {
        self.edges
            .get(id)
            .ok_or_else(|| GraphError::EdgeNotFound(id.to_string()))
    }

    /// Delete an edge by id.
    pub fn delete_edge(&mut self, id: &str) -> Result<()> {
        let _guard = self.lock.write();
        if !self.edges.contains_key(id) {
            return Err(GraphError::EdgeNotFound(id.to_string()));
        }
        remove_edge_internal(
            &mut self.edges,
            &mut self.adjacency_out,
            &mut self.adjacency_in,
            id,
        );
        Ok(())
    }

    /// Outgoing edges from a node.
    pub fn edges_from(&self, node_id: &str) -> Vec<&Edge> {
        match self.adjacency_out.get(node_id) {
            Some(ids) => ids.iter().filter_map(|i| self.edges.get(i)).collect(),
            None => Vec::new(),
        }
    }

    /// Incoming edges to a node.
    pub fn edges_to(&self, node_id: &str) -> Vec<&Edge> {
        match self.adjacency_in.get(node_id) {
            Some(ids) => ids.iter().filter_map(|i| self.edges.get(i)).collect(),
            None => Vec::new(),
        }
    }

    /// Edges directly connecting `src_id -> dst_id`.
    pub fn edges_between(&self, src_id: &str, dst_id: &str) -> Vec<&Edge> {
        self.edges_from(src_id)
            .into_iter()
            .filter(|e| e.dst_id == dst_id)
            .collect()
    }

    /// All edges.
    pub fn all_edges(&self) -> Vec<&Edge> {
        self.edges.values().collect()
    }

    // ----- vector search ---------------------------------------------------

    /// Label-scoped top-k similarity search over node embeddings.
    ///
    /// When `label` is `Some`, only nodes with that label are searched;
    /// otherwise all nodes with embeddings are considered.
    pub fn search_similar_nodes(
        &self,
        query: &[f32],
        label: Option<&str>,
        k: usize,
    ) -> Result<Vec<(String, f32)>> {
        let candidate_nodes: Vec<&Node> = match label {
            Some(l) => self.nodes_by_label(l),
            None => self.all_nodes(),
        };

        let candidates: Vec<(&str, &[f32])> = candidate_nodes
            .iter()
            .filter_map(|n| n.embedding.as_ref().map(|e| (n.id.as_str(), e.as_slice())))
            .collect();

        top_k_similar(query, &candidates, k)
    }

    // ----- persistence -----------------------------------------------------

    /// Save the store to its configured persistence path.
    pub fn save(&self) -> Result<()> {
        let _guard = self.lock.read();
        match &self.persistence_path {
            Some(path) => persistence::save_to_file(self, path),
            None => Err(GraphError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "no persistence path configured",
            ))),
        }
    }

    /// Load the store from its configured persistence path, replacing contents.
    pub fn load(&mut self) -> Result<()> {
        let path = match &self.persistence_path {
            Some(p) => p.clone(),
            None => {
                return Err(GraphError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "no persistence path configured",
                )))
            }
        };
        let (nodes, edges) = persistence::load_from_file(&path)?;
        self.reset();
        for node in nodes {
            self.add_node(node)?;
        }
        for edge in edges {
            // Re-populate adjacency + edge table without re-running similarity.
            self.adjacency_out
                .entry(edge.src_id.clone())
                .or_default()
                .push(edge.id.clone());
            self.adjacency_in
                .entry(edge.dst_id.clone())
                .or_default()
                .push(edge.id.clone());
            self.edges.insert(edge.id.clone(), edge);
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.adjacency_out.clear();
        self.adjacency_in.clear();
        self.label_index.clear();
        self.property_index.clear();
    }

    // ----- stats -----------------------------------------------------------

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Find nodes by label and an exact property match (uses the property index
    /// when possible). Exposed as a convenience for the query layer.
    pub fn find_nodes(
        &self,
        label: Option<&str>,
        prop: Option<(&str, &PropertyValue)>,
    ) -> Vec<&Node> {
        match (label, prop) {
            (Some(l), Some((k, v))) => match self.property_index.get(l, k, v) {
                Some(ids) => ids.iter().filter_map(|i| self.nodes.get(i)).collect(),
                None => Vec::new(),
            },
            (Some(l), None) => self.nodes_by_label(l),
            (None, Some((k, v))) => self
                .all_nodes()
                .into_iter()
                .filter(|n| n.properties.get(k).map(|pv| pv == v).unwrap_or(false))
                .collect(),
            (None, None) => self.all_nodes(),
        }
    }
}

// ----- module-private helpers operating on disjoint fields -----------------
//
// These are free functions (not `&mut self` methods) so they can be called
// while the `RwLock` guard holds a borrow of `self.lock`. Borrowing individual
// fields is disjoint from borrowing `self.lock`, which keeps the borrow checker
// satisfied.

fn index_node(label_index: &mut LabelIndex, property_index: &mut PropertyIndex, node: &Node) {
    label_index.insert(&node.label, &node.id);
    for (key, value) in &node.properties {
        property_index.insert(&node.label, key, value, &node.id);
    }
}

fn deindex_node(label_index: &mut LabelIndex, property_index: &mut PropertyIndex, node: &Node) {
    label_index.remove(&node.label, &node.id);
    for (key, value) in &node.properties {
        property_index.remove(&node.label, key, value, &node.id);
    }
}

fn remove_edge_internal(
    edges: &mut HashMap<String, Edge>,
    adjacency_out: &mut HashMap<String, Vec<String>>,
    adjacency_in: &mut HashMap<String, Vec<String>>,
    id: &str,
) {
    if let Some(edge) = edges.remove(id) {
        if let Some(out) = adjacency_out.get_mut(&edge.src_id) {
            out.retain(|e| e != id);
        }
        if let Some(inc) = adjacency_in.get_mut(&edge.dst_id) {
            inc.retain(|e| e != id);
        }
    }
}
