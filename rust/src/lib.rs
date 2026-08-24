//! `graphdb_rs` — a Rust port of the graphdb core engine.
//!
//! This crate exposes two surfaces:
//! * A pure-Rust API (`core`, `store`, `query`, `vector`, ...), each function
//!   returning [`error::Result`] and never panicking on recoverable errors.
//! * A Python extension module (built with PyO3 + maturin) that wraps the Rust
//!   API. Rust errors are mapped to idiomatic Python exceptions.

pub mod core;
pub mod error;
pub mod index;
pub mod persistence;
pub mod query;
pub mod similarity;
pub mod store;
pub mod vector;

use std::collections::HashMap;

use pyo3::exceptions::{PyIOError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::core::{Edge, Node, PropertyValue};
use crate::error::GraphError;
use crate::store::GraphStore;

// --------------------------------------------------------------------------
// Error mapping
// --------------------------------------------------------------------------

/// Map a [`GraphError`] onto an appropriate Python exception.
fn to_pyerr(err: GraphError) -> PyErr {
    match err {
        GraphError::NodeNotFound(_) | GraphError::EdgeNotFound(_) => {
            PyKeyError::new_err(err.to_string())
        }
        GraphError::DimensionMismatch { .. } => PyValueError::new_err(err.to_string()),
        GraphError::Io(_) => PyIOError::new_err(err.to_string()),
        GraphError::Serde(_) => PyValueError::new_err(err.to_string()),
    }
}

// --------------------------------------------------------------------------
// PropertyValue <-> Python conversion
// --------------------------------------------------------------------------

fn py_to_property(obj: &Bound<'_, PyAny>) -> PyResult<PropertyValue> {
    // Order matters: bool is a subclass of int in Python.
    if let Ok(b) = obj.extract::<bool>() {
        // Guard: only treat as bool if it is actually a Python bool.
        if obj.is_instance_of::<pyo3::types::PyBool>() {
            return Ok(PropertyValue::Bool(b));
        }
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(PropertyValue::Int(i));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(PropertyValue::Float(f));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(PropertyValue::Str(s));
    }
    if let Ok(list) = obj.downcast::<PyList>() {
        let mut items = Vec::with_capacity(list.len());
        for item in list.iter() {
            items.push(py_to_property(&item)?);
        }
        return Ok(PropertyValue::List(items));
    }
    Err(PyValueError::new_err(
        "unsupported property type (allowed: str, int, float, bool, list)",
    ))
}

fn property_to_py(py: Python<'_>, value: &PropertyValue) -> PyObject {
    match value {
        PropertyValue::Str(s) => s.into_py(py),
        PropertyValue::Int(i) => i.into_py(py),
        PropertyValue::Float(f) => f.into_py(py),
        PropertyValue::Bool(b) => b.into_py(py),
        PropertyValue::List(items) => {
            let list = PyList::empty_bound(py);
            for it in items {
                list.append(property_to_py(py, it)).ok();
            }
            list.into_py(py)
        }
    }
}

fn dict_to_properties(dict: Option<&Bound<'_, PyDict>>) -> PyResult<HashMap<String, PropertyValue>> {
    let mut map = HashMap::new();
    if let Some(d) = dict {
        for (k, v) in d.iter() {
            let key: String = k.extract()?;
            map.insert(key, py_to_property(&v)?);
        }
    }
    Ok(map)
}

fn properties_to_dict(py: Python<'_>, props: &HashMap<String, PropertyValue>) -> PyObject {
    let dict = PyDict::new_bound(py);
    for (k, v) in props {
        dict.set_item(k, property_to_py(py, v)).ok();
    }
    dict.into_py(py)
}

// --------------------------------------------------------------------------
// Python-facing wrapper types
// --------------------------------------------------------------------------

/// Python view of a [`Node`].
#[pyclass(name = "PyNode")]
#[derive(Clone)]
pub struct PyNode {
    inner: Node,
}

#[pymethods]
impl PyNode {
    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }
    #[getter]
    fn label(&self) -> String {
        self.inner.label.clone()
    }
    #[getter]
    fn properties(&self, py: Python<'_>) -> PyObject {
        properties_to_dict(py, &self.inner.properties)
    }
    #[getter]
    fn embedding(&self) -> Option<Vec<f32>> {
        self.inner.embedding.clone()
    }
    #[getter]
    fn created_at(&self) -> u64 {
        self.inner.created_at
    }
    fn __repr__(&self) -> String {
        format!("PyNode(id={}, label={})", self.inner.id, self.inner.label)
    }
}

/// Python view of an [`Edge`].
#[pyclass(name = "PyEdge")]
#[derive(Clone)]
pub struct PyEdge {
    inner: Edge,
}

#[pymethods]
impl PyEdge {
    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }
    #[getter]
    fn src_id(&self) -> String {
        self.inner.src_id.clone()
    }
    #[getter]
    fn dst_id(&self) -> String {
        self.inner.dst_id.clone()
    }
    #[getter]
    fn label(&self) -> String {
        self.inner.label.clone()
    }
    #[getter]
    fn properties(&self, py: Python<'_>) -> PyObject {
        properties_to_dict(py, &self.inner.properties)
    }
    #[getter]
    fn weight(&self) -> f32 {
        self.inner.weight
    }
    fn __repr__(&self) -> String {
        format!(
            "PyEdge(id={}, {} -[{}]-> {})",
            self.inner.id, self.inner.src_id, self.inner.label, self.inner.dst_id
        )
    }
}

/// Python view of a `SimilarMatch`.
#[pyclass(name = "PySimilarMatch")]
#[derive(Clone)]
pub struct PySimilarMatch {
    #[pyo3(get)]
    existing_edge_id: String,
    #[pyo3(get)]
    src_similarity: f32,
    #[pyo3(get)]
    dst_similarity: f32,
    #[pyo3(get)]
    combined_score: f32,
}

/// Python view of an `AddEdgeResult`.
#[pyclass(name = "PyAddEdgeResult")]
pub struct PyAddEdgeResult {
    #[pyo3(get)]
    edge: PyEdge,
    #[pyo3(get)]
    similar_edges: Vec<PySimilarMatch>,
    #[pyo3(get)]
    was_flagged: bool,
}

/// Python-facing graph store.
#[pyclass(name = "PyGraphStore")]
pub struct PyGraphStore {
    inner: GraphStore,
}

#[pymethods]
impl PyGraphStore {
    #[new]
    #[pyo3(signature = (path=None))]
    fn new(path: Option<String>) -> Self {
        let inner = match path {
            Some(p) => GraphStore::with_path(p),
            None => GraphStore::new(),
        };
        PyGraphStore { inner }
    }

    #[pyo3(signature = (label, properties=None, embedding=None))]
    fn add_node(
        &mut self,
        label: &str,
        properties: Option<&Bound<'_, PyDict>>,
        embedding: Option<Vec<f32>>,
    ) -> PyResult<PyNode> {
        let mut node = Node::new(label);
        node.properties = dict_to_properties(properties)?;
        node.embedding = embedding;
        let stored = self.inner.add_node(node).map_err(to_pyerr)?;
        Ok(PyNode { inner: stored })
    }

    fn get_node(&self, id: &str) -> PyResult<PyNode> {
        let node = self.inner.get_node(id).map_err(to_pyerr)?;
        Ok(PyNode {
            inner: node.clone(),
        })
    }

    fn delete_node(&mut self, id: &str) -> PyResult<()> {
        self.inner.delete_node(id).map_err(to_pyerr)
    }

    fn nodes_by_label(&self, label: &str) -> Vec<PyNode> {
        self.inner
            .nodes_by_label(label)
            .into_iter()
            .map(|n| PyNode { inner: n.clone() })
            .collect()
    }

    fn all_nodes(&self) -> Vec<PyNode> {
        self.inner
            .all_nodes()
            .into_iter()
            .map(|n| PyNode { inner: n.clone() })
            .collect()
    }

    #[pyo3(signature = (src_id, dst_id, label, properties=None, weight=1.0, similarity_threshold=0.85))]
    fn add_edge(
        &mut self,
        src_id: &str,
        dst_id: &str,
        label: &str,
        properties: Option<&Bound<'_, PyDict>>,
        weight: f32,
        similarity_threshold: f32,
    ) -> PyResult<PyAddEdgeResult> {
        let mut edge = Edge::new(src_id, dst_id, label);
        edge.properties = dict_to_properties(properties)?;
        edge.weight = weight;
        let result = self
            .inner
            .add_edge(edge, similarity_threshold)
            .map_err(to_pyerr)?;
        let similar_edges = result
            .similar_edges
            .into_iter()
            .map(|m| PySimilarMatch {
                existing_edge_id: m.existing_edge_id,
                src_similarity: m.src_similarity,
                dst_similarity: m.dst_similarity,
                combined_score: m.combined_score,
            })
            .collect();
        Ok(PyAddEdgeResult {
            edge: PyEdge { inner: result.edge },
            similar_edges,
            was_flagged: result.was_flagged,
        })
    }

    fn get_edge(&self, id: &str) -> PyResult<PyEdge> {
        let edge = self.inner.get_edge(id).map_err(to_pyerr)?;
        Ok(PyEdge {
            inner: edge.clone(),
        })
    }

    fn delete_edge(&mut self, id: &str) -> PyResult<()> {
        self.inner.delete_edge(id).map_err(to_pyerr)
    }

    fn edges_from(&self, node_id: &str) -> Vec<PyEdge> {
        self.inner
            .edges_from(node_id)
            .into_iter()
            .map(|e| PyEdge { inner: e.clone() })
            .collect()
    }

    fn edges_to(&self, node_id: &str) -> Vec<PyEdge> {
        self.inner
            .edges_to(node_id)
            .into_iter()
            .map(|e| PyEdge { inner: e.clone() })
            .collect()
    }

    #[pyo3(signature = (query, label=None, k=5))]
    fn search_similar_nodes(
        &self,
        query: Vec<f32>,
        label: Option<&str>,
        k: usize,
    ) -> PyResult<Vec<(String, f32)>> {
        self.inner
            .search_similar_nodes(&query, label, k)
            .map_err(to_pyerr)
    }

    fn save(&self) -> PyResult<()> {
        self.inner.save().map_err(to_pyerr)
    }

    fn load(&mut self) -> PyResult<()> {
        self.inner.load().map_err(to_pyerr)
    }

    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }
}

// --------------------------------------------------------------------------
// Module-level traversal functions
// --------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (store, start_id, edge_label=None, max_depth=5))]
fn bfs(
    store: &PyGraphStore,
    start_id: &str,
    edge_label: Option<&str>,
    max_depth: usize,
) -> PyResult<Vec<String>> {
    query::bfs(&store.inner, start_id, edge_label, max_depth).map_err(to_pyerr)
}

#[pyfunction]
#[pyo3(signature = (store, start_id, edge_label=None, max_depth=5))]
fn dfs(
    store: &PyGraphStore,
    start_id: &str,
    edge_label: Option<&str>,
    max_depth: usize,
) -> PyResult<Vec<String>> {
    query::dfs(&store.inner, start_id, edge_label, max_depth).map_err(to_pyerr)
}

#[pyfunction]
fn find_path(
    store: &PyGraphStore,
    src_id: &str,
    dst_id: &str,
) -> PyResult<Option<Vec<String>>> {
    query::find_path(&store.inner, src_id, dst_id).map_err(to_pyerr)
}

/// The Python extension module.
#[pymodule]
fn graphdb_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNode>()?;
    m.add_class::<PyEdge>()?;
    m.add_class::<PySimilarMatch>()?;
    m.add_class::<PyAddEdgeResult>()?;
    m.add_class::<PyGraphStore>()?;
    m.add_function(wrap_pyfunction!(bfs, m)?)?;
    m.add_function(wrap_pyfunction!(dfs, m)?)?;
    m.add_function(wrap_pyfunction!(find_path, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
