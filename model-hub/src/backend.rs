//! Graph backend abstraction.
//!
//! `model-hub` deliberately does **not** depend on the `graphdb_rs` crate
//! (which links against PyO3 as a `cdylib`). Instead it defines the
//! [`GraphBackend`] trait, which any storage engine can implement to plug into
//! the RAG / tool-calling pipelines. This keeps the AI pipeline decoupled from
//! the concrete graph engine and avoids native-link conflicts.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Metadata describing a single ingested document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DocumentInfo {
    /// Absolute source path (unique identifier for the document).
    pub source_path: String,
    /// Human-readable title.
    pub title: String,
    /// Document type/category (e.g. "pdf", "markdown", "code").
    pub doc_type: String,
    /// Number of chunks the document was split into.
    pub chunk_count: usize,
}

/// A single chunk of a document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChunkInfo {
    /// Identifier of the chunk (stable within a document).
    pub id: String,
    /// Source document path.
    pub source_path: String,
    /// Ordinal index of the chunk within the document.
    pub index: usize,
    /// Chunk text content.
    pub text: String,
}

/// A search hit returned from the knowledge base.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchResult {
    /// The matching chunk.
    pub chunk: ChunkInfo,
    /// Similarity / relevance score (higher is better).
    pub score: f32,
}

/// Abstraction over a graph-backed knowledge base.
///
/// Implementors provide semantic search, document listing, and chunk retrieval.
/// The pipeline layer relies only on this trait, never on a concrete engine.
#[async_trait]
pub trait GraphBackend: Send + Sync {
    /// Semantic search over the knowledge base.
    ///
    /// * `query`    - natural-language query.
    /// * `k`        - maximum number of results.
    /// * `doc_type` - optional filter restricting results to a document type.
    async fn search(
        &self,
        query: &str,
        k: usize,
        doc_type: Option<&str>,
    ) -> Result<Vec<SearchResult>>;

    /// List all documents that have been ingested.
    async fn list_documents(&self) -> Result<Vec<DocumentInfo>>;

    /// Retrieve all chunks for a specific document.
    async fn get_document_chunks(&self, source_path: &str) -> Result<Vec<ChunkInfo>>;
}
