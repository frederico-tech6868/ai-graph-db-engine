//! In-memory reference implementation of [`GraphBackend`].
//!
//! This is a lightweight, dependency-free knowledge base intended for offline
//! development, testing, and demos. Documents are chunked on ingest and scored
//! at query time with a simple bag-of-words cosine similarity. Production
//! deployments should implement [`GraphBackend`] on top of the real graph
//! engine instead.

use std::collections::HashMap;
use std::sync::RwLock;

use async_trait::async_trait;

use crate::backend::{ChunkInfo, DocumentInfo, GraphBackend, SearchResult};
use crate::error::Result;

/// A simple in-memory knowledge base.
#[derive(Default)]
pub struct InMemoryGraph {
    inner: RwLock<GraphState>,
}

#[derive(Default)]
struct GraphState {
    docs: Vec<DocumentInfo>,
    chunks: Vec<ChunkInfo>,
}

impl InMemoryGraph {
    /// Create an empty in-memory graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingest a document, splitting `content` into word-window chunks.
    ///
    /// Returns the number of chunks created.
    pub fn ingest(
        &self,
        source_path: &str,
        title: &str,
        doc_type: &str,
        content: &str,
    ) -> usize {
        let words: Vec<&str> = content.split_whitespace().collect();
        let window = 120usize; // words per chunk
        let mut state = self.inner.write().expect("graph lock poisoned");

        // Remove any existing document with the same path.
        state.chunks.retain(|c| c.source_path != source_path);
        state.docs.retain(|d| d.source_path != source_path);

        let mut chunk_count = 0usize;
        if words.is_empty() {
            let id = format!("{source_path}#0");
            state.chunks.push(ChunkInfo {
                id,
                source_path: source_path.to_string(),
                index: 0,
                text: content.to_string(),
            });
            chunk_count = 1;
        } else {
            for (idx, window_words) in words.chunks(window).enumerate() {
                let text = window_words.join(" ");
                let id = format!("{source_path}#{idx}");
                state.chunks.push(ChunkInfo {
                    id,
                    source_path: source_path.to_string(),
                    index: idx,
                    text,
                });
                chunk_count += 1;
            }
        }

        state.docs.push(DocumentInfo {
            source_path: source_path.to_string(),
            title: title.to_string(),
            doc_type: doc_type.to_string(),
            chunk_count,
        });
        chunk_count
    }

    /// Number of documents currently ingested.
    pub fn document_count(&self) -> usize {
        self.inner.read().expect("graph lock poisoned").docs.len()
    }

    /// Number of chunks currently stored.
    pub fn chunk_count(&self) -> usize {
        self.inner.read().expect("graph lock poisoned").chunks.len()
    }
}

/// Tokenize into lowercase alphanumeric terms.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

/// Build a term-frequency vector.
fn term_freq(tokens: &[String]) -> HashMap<String, f32> {
    let mut tf: HashMap<String, f32> = HashMap::new();
    for t in tokens {
        *tf.entry(t.clone()).or_insert(0.0) += 1.0;
    }
    tf
}

/// Cosine similarity between two sparse term-frequency vectors.
fn cosine(a: &HashMap<String, f32>, b: &HashMap<String, f32>) -> f32 {
    let mut dot = 0.0f32;
    for (k, va) in a {
        if let Some(vb) = b.get(k) {
            dot += va * vb;
        }
    }
    let na: f32 = a.values().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.values().map(|v| v * v).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

#[async_trait]
impl GraphBackend for InMemoryGraph {
    async fn search(
        &self,
        query: &str,
        k: usize,
        doc_type: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let state = self.inner.read().expect("graph lock poisoned");
        let q_tf = term_freq(&tokenize(query));

        // Map source_path -> doc_type for filtering.
        let type_of: HashMap<&str, &str> = state
            .docs
            .iter()
            .map(|d| (d.source_path.as_str(), d.doc_type.as_str()))
            .collect();

        let mut scored: Vec<SearchResult> = state
            .chunks
            .iter()
            .filter(|c| match doc_type {
                Some(dt) => type_of.get(c.source_path.as_str()) == Some(&dt),
                None => true,
            })
            .map(|c| {
                let c_tf = term_freq(&tokenize(&c.text));
                SearchResult {
                    chunk: c.clone(),
                    score: cosine(&q_tf, &c_tf),
                }
            })
            .collect();

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored)
    }

    async fn list_documents(&self) -> Result<Vec<DocumentInfo>> {
        Ok(self.inner.read().expect("graph lock poisoned").docs.clone())
    }

    async fn get_document_chunks(&self, source_path: &str) -> Result<Vec<ChunkInfo>> {
        let state = self.inner.read().expect("graph lock poisoned");
        let mut chunks: Vec<ChunkInfo> = state
            .chunks
            .iter()
            .filter(|c| c.source_path == source_path)
            .cloned()
            .collect();
        chunks.sort_by_key(|c| c.index);
        Ok(chunks)
    }
}
