//! JSON persistence for the [`GraphStore`] using `serde_json`.
//!
//! The full store (nodes + edges) is serialized to a single JSON document with
//! a schema `version` field. Writes are atomic: data is written to a temporary
//! file which is then renamed over the destination.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::core::{Edge, Node};
use crate::error::Result;
use crate::store::GraphStore;

/// Serializable on-disk representation of the whole graph.
#[derive(Serialize, Deserialize)]
struct SerializableStore {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    version: String,
}

/// Current on-disk schema version.
pub const SCHEMA_VERSION: &str = "1";

/// Serialize the full store to a JSON file (atomic write).
pub fn save_to_file(store: &GraphStore, path: &str) -> Result<()> {
    let payload = SerializableStore {
        nodes: store.all_nodes().into_iter().cloned().collect(),
        edges: store.all_edges().into_iter().cloned().collect(),
        version: SCHEMA_VERSION.to_string(),
    };
    let json = serde_json::to_string_pretty(&payload)?;

    // Atomic write via a temporary file in the same directory.
    let dst = Path::new(path);
    let tmp = match dst.parent() {
        Some(dir) => dir.join(format!(
            ".{}.tmp",
            dst.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("graph.json")
        )),
        None => Path::new(".graph.json.tmp").to_path_buf(),
    };
    fs::write(&tmp, json)?;
    fs::rename(&tmp, dst)?;
    Ok(())
}

/// Load nodes + edges from a JSON file.
///
/// A missing or empty file yields an empty graph (matching the Python engine).
pub fn load_from_file(path: &str) -> Result<(Vec<Node>, Vec<Edge>)> {
    let p = Path::new(path);
    if !p.exists() {
        return Ok((Vec::new(), Vec::new()));
    }
    let raw = fs::read_to_string(p)?;
    if raw.trim().is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let parsed: SerializableStore = serde_json::from_str(&raw)?;
    Ok((parsed.nodes, parsed.edges))
}
