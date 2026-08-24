//! Core data models: [`Node`], [`Edge`] and [`PropertyValue`].

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A typed property value that can be stored on a node or edge.
///
/// Mirrors the Python engine which supports `str`, `int`, `float`, `bool` and
/// lists of those values.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PropertyValue {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    List(Vec<PropertyValue>),
}

impl PropertyValue {
    /// Produce a stable string key for property indexing.
    ///
    /// Only scalar values participate in the property index (matching the
    /// Python `PropertyIndex`, which only indexes hashable scalars). Lists
    /// return a deterministic representation but are generally not indexed.
    pub fn index_key(&self) -> String {
        match self {
            PropertyValue::Str(s) => format!("s:{s}"),
            PropertyValue::Int(i) => format!("i:{i}"),
            PropertyValue::Float(f) => format!("f:{f}"),
            PropertyValue::Bool(b) => format!("b:{b}"),
            PropertyValue::List(items) => {
                let inner: Vec<String> = items.iter().map(|v| v.index_key()).collect();
                format!("l:[{}]", inner.join(","))
            }
        }
    }

    /// Whether this value is a scalar that should be placed in the property index.
    pub fn is_indexable(&self) -> bool {
        !matches!(self, PropertyValue::List(_))
    }
}

/// Current Unix time in milliseconds.
pub fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// A graph node: a labelled entity with typed properties and an optional
/// embedding used for vector similarity search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    pub label: String,
    pub properties: HashMap<String, PropertyValue>,
    pub embedding: Option<Vec<f32>>,
    /// Unix timestamp in milliseconds.
    pub created_at: u64,
}

impl Node {
    /// Create a new node with an auto-generated UUID and the given label.
    pub fn new(label: impl Into<String>) -> Self {
        Node {
            id: Uuid::new_v4().to_string(),
            label: label.into(),
            properties: HashMap::new(),
            embedding: None,
            created_at: now_millis(),
        }
    }

    /// Builder helper attaching an embedding vector.
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Set a single property.
    pub fn set_property(&mut self, key: impl Into<String>, value: PropertyValue) {
        self.properties.insert(key.into(), value);
    }
}

/// A directed, labelled, weighted edge between two nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub id: String,
    pub src_id: String,
    pub dst_id: String,
    pub label: String,
    pub properties: HashMap<String, PropertyValue>,
    pub weight: f32,
    /// Unix timestamp in milliseconds.
    pub created_at: u64,
}

impl Edge {
    /// Create a new edge with an auto-generated UUID.
    pub fn new(
        src_id: impl Into<String>,
        dst_id: impl Into<String>,
        label: impl Into<String>,
    ) -> Self {
        Edge {
            id: Uuid::new_v4().to_string(),
            src_id: src_id.into(),
            dst_id: dst_id.into(),
            label: label.into(),
            properties: HashMap::new(),
            weight: 1.0,
            created_at: now_millis(),
        }
    }

    /// Builder helper setting the edge weight.
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    /// Set a single property.
    pub fn set_property(&mut self, key: impl Into<String>, value: PropertyValue) {
        self.properties.insert(key.into(), value);
    }
}
