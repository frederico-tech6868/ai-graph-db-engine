//! In-memory indexes kept in sync with the [`crate::store::GraphStore`].
//!
//! [`LabelIndex`] maps a text label to the set of node ids carrying it.
//! [`PropertyIndex`] maps `(label, key, value)` to the set of node ids, enabling
//! O(1) exact-match property lookups scoped to a label.

use std::collections::{HashMap, HashSet};

use crate::core::PropertyValue;

/// Maps `label -> set of node ids`.
#[derive(Debug, Default)]
pub struct LabelIndex {
    inner: HashMap<String, HashSet<String>>,
}

impl LabelIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node id under a label.
    pub fn insert(&mut self, label: &str, node_id: &str) {
        self.inner
            .entry(label.to_string())
            .or_default()
            .insert(node_id.to_string());
    }

    /// Remove a node id from a label bucket, cleaning up empty buckets.
    pub fn remove(&mut self, label: &str, node_id: &str) {
        if let Some(set) = self.inner.get_mut(label) {
            set.remove(node_id);
            if set.is_empty() {
                self.inner.remove(label);
            }
        }
    }

    /// Get the set of node ids for a label, if any.
    pub fn get(&self, label: &str) -> Option<&HashSet<String>> {
        self.inner.get(label)
    }

    /// Return all known labels.
    pub fn all_labels(&self) -> Vec<String> {
        self.inner.keys().cloned().collect()
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

/// Maps `(label, key, value_hash) -> set of node ids`.
#[derive(Debug, Default)]
pub struct PropertyIndex {
    inner: HashMap<(String, String, String), HashSet<String>>,
}

impl PropertyIndex {
    pub fn new() -> Self {
        Self::default()
    }

    fn key(label: &str, key: &str, value: &PropertyValue) -> (String, String, String) {
        (label.to_string(), key.to_string(), value.index_key())
    }

    /// Index a scalar property value for a node. Lists are ignored.
    pub fn insert(&mut self, label: &str, key: &str, value: &PropertyValue, node_id: &str) {
        if !value.is_indexable() {
            return;
        }
        self.inner
            .entry(Self::key(label, key, value))
            .or_default()
            .insert(node_id.to_string());
    }

    /// Remove a node id from an indexed property value.
    pub fn remove(&mut self, label: &str, key: &str, value: &PropertyValue, node_id: &str) {
        if !value.is_indexable() {
            return;
        }
        let k = Self::key(label, key, value);
        if let Some(set) = self.inner.get_mut(&k) {
            set.remove(node_id);
            if set.is_empty() {
                self.inner.remove(&k);
            }
        }
    }

    /// Look up node ids matching a label/key/value triple.
    pub fn get(&self, label: &str, key: &str, value: &PropertyValue) -> Option<&HashSet<String>> {
        self.inner.get(&Self::key(label, key, value))
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }
}
