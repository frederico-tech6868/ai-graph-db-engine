//! Graph traversal: BFS, DFS, shortest path and a fluent [`GraphQuery`].
//!
//! Traversals return node ids (matching the Phase-3 spec) and mirror the
//! Python `graphdb.query` semantics: the start node is excluded from BFS/DFS
//! results, `max_depth` bounds the walk, and `find_path` returns the shortest
//! path (including both endpoints).

use std::collections::{HashSet, VecDeque};

use crate::core::{Node, PropertyValue};
use crate::error::Result;
use crate::store::GraphStore;

/// Outgoing neighbours of `node_id`, optionally filtered by edge label.
fn neighbors(store: &GraphStore, node_id: &str, edge_label: Option<&str>) -> Vec<String> {
    store
        .edges_from(node_id)
        .into_iter()
        .filter(|e| edge_label.map(|l| e.label == l).unwrap_or(true))
        .map(|e| e.dst_id.clone())
        .collect()
}

/// Breadth-first traversal from `start_id` (excludes the start node).
pub fn bfs(
    store: &GraphStore,
    start_id: &str,
    edge_label: Option<&str>,
    max_depth: usize,
) -> Result<Vec<String>> {
    // Validate the start node exists.
    store.get_node(start_id)?;

    let mut visited: HashSet<String> = HashSet::new();
    visited.insert(start_id.to_string());
    let mut order: Vec<String> = Vec::new();
    let mut queue: VecDeque<(String, usize)> = VecDeque::new();
    queue.push_back((start_id.to_string(), 0));

    while let Some((current, depth)) = queue.pop_front() {
        if depth >= max_depth {
            continue;
        }
        for nb in neighbors(store, &current, edge_label) {
            if visited.insert(nb.clone()) {
                order.push(nb.clone());
                queue.push_back((nb, depth + 1));
            }
        }
    }
    Ok(order)
}

/// Depth-first traversal from `start_id` (excludes the start node).
pub fn dfs(
    store: &GraphStore,
    start_id: &str,
    edge_label: Option<&str>,
    max_depth: usize,
) -> Result<Vec<String>> {
    store.get_node(start_id)?;

    let mut visited: HashSet<String> = HashSet::new();
    visited.insert(start_id.to_string());
    let mut order: Vec<String> = Vec::new();

    fn visit(
        store: &GraphStore,
        node_id: &str,
        edge_label: Option<&str>,
        depth: usize,
        max_depth: usize,
        visited: &mut HashSet<String>,
        order: &mut Vec<String>,
    ) {
        if depth >= max_depth {
            return;
        }
        for nb in neighbors(store, node_id, edge_label) {
            if visited.insert(nb.clone()) {
                order.push(nb.clone());
                visit(store, &nb, edge_label, depth + 1, max_depth, visited, order);
            }
        }
    }

    visit(
        store,
        start_id,
        edge_label,
        0,
        max_depth,
        &mut visited,
        &mut order,
    );
    Ok(order)
}

/// Shortest path (list of node ids, endpoints included) from `src_id` to
/// `dst_id`, or `None` if unreachable.
pub fn find_path(store: &GraphStore, src_id: &str, dst_id: &str) -> Result<Option<Vec<String>>> {
    if store.get_node(src_id).is_err() || store.get_node(dst_id).is_err() {
        return Ok(None);
    }
    if src_id == dst_id {
        return Ok(Some(vec![src_id.to_string()]));
    }

    let mut visited: HashSet<String> = HashSet::new();
    visited.insert(src_id.to_string());
    let mut prev: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut queue: VecDeque<String> = VecDeque::new();
    queue.push_back(src_id.to_string());
    let mut found = false;

    while let Some(current) = queue.pop_front() {
        if current == dst_id {
            found = true;
            break;
        }
        for nb in neighbors(store, &current, None) {
            if visited.insert(nb.clone()) {
                prev.insert(nb.clone(), current.clone());
                queue.push_back(nb);
            }
        }
    }

    if !found && !prev.contains_key(dst_id) {
        return Ok(None);
    }

    // Reconstruct the path.
    let mut path: Vec<String> = vec![dst_id.to_string()];
    while path.last().map(|s| s.as_str()) != Some(src_id) {
        match prev.get(path.last().unwrap()) {
            Some(parent) => path.push(parent.clone()),
            None => return Ok(None),
        }
    }
    path.reverse();
    Ok(Some(path))
}

/// A small fluent query builder over a [`GraphStore`].
pub struct GraphQuery<'a> {
    store: &'a GraphStore,
    label_filter: Option<String>,
    prop_filters: Vec<(String, PropertyValue)>,
}

impl<'a> GraphQuery<'a> {
    pub fn new(store: &'a GraphStore) -> Self {
        GraphQuery {
            store,
            label_filter: None,
            prop_filters: Vec::new(),
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label_filter = Some(label.into());
        self
    }

    pub fn with_property(mut self, key: impl Into<String>, value: PropertyValue) -> Self {
        self.prop_filters.push((key.into(), value));
        self
    }

    /// Execute the query, returning matching nodes.
    pub fn execute(self) -> Vec<&'a Node> {
        let label = self.label_filter.as_deref();
        // Start from the label-scoped set (or all nodes) then apply property
        // filters in memory.
        let mut nodes: Vec<&Node> = match label {
            Some(l) => self.store.nodes_by_label(l),
            None => self.store.all_nodes(),
        };
        for (key, value) in &self.prop_filters {
            nodes.retain(|n| n.properties.get(key).map(|pv| pv == value).unwrap_or(false));
        }
        nodes
    }
}
