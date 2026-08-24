//! Label-scoped edge similarity scanning.
//!
//! Before a new edge is added between `src` and `dst`, we scan existing edges
//! that share the same edge label *and* whose endpoints share the same text
//! labels as `src` / `dst`. Embeddings are then compared to surface
//! near-duplicate relationships. Every scan is label-scoped so a node is never
//! compared against a node of a different type — this exactly mirrors the
//! Python `graphdb.similarity.SimilarityScanner`.

use crate::core::Node;
use crate::error::Result;
use crate::store::GraphStore;
use crate::vector::cosine_similarity;

/// A single existing edge found to be similar to a proposed edge.
#[derive(Debug, Clone)]
pub struct SimilarMatch {
    pub existing_edge_id: String,
    pub src_similarity: f32,
    pub dst_similarity: f32,
    pub combined_score: f32,
}

/// Scans existing edges for near-duplicates before an edge is added.
pub struct SimilarityScanner<'a> {
    store: &'a GraphStore,
}

impl<'a> SimilarityScanner<'a> {
    pub fn new(store: &'a GraphStore) -> Self {
        SimilarityScanner { store }
    }

    /// Return existing edges similar to a proposed `src -> dst` edge.
    ///
    /// An edge qualifies only when:
    /// * its edge label equals `edge_label`;
    /// * its source node has the same text label as `src_node`;
    /// * its destination node has the same text label as `dst_node`.
    ///
    /// For each qualifying edge, cosine similarity is computed between the
    /// source embeddings and between the destination embeddings. The edge is
    /// returned when `(src_sim + dst_sim) / 2 >= threshold`. Results are sorted
    /// by combined score, descending.
    pub fn scan_before_add(
        &self,
        src_node: &Node,
        dst_node: &Node,
        edge_label: &str,
        threshold: f32,
    ) -> Result<Vec<SimilarMatch>> {
        let mut matches: Vec<SimilarMatch> = Vec::new();

        // Without embeddings on the proposed endpoints there is nothing to
        // compare against.
        let (src_emb, dst_emb) = match (&src_node.embedding, &dst_node.embedding) {
            (Some(s), Some(d)) => (s, d),
            _ => return Ok(matches),
        };

        // Restrict candidate node sets via the label index so we skip any node
        // whose text label differs from the proposed endpoints.
        let src_label_ids = self.store.label_index().get(&src_node.label);
        let dst_label_ids = self.store.label_index().get(&dst_node.label);
        let (src_label_ids, dst_label_ids) = match (src_label_ids, dst_label_ids) {
            (Some(s), Some(d)) => (s, d),
            _ => return Ok(matches),
        };

        for edge in self.store.all_edges() {
            if edge.label != edge_label {
                continue;
            }
            if !src_label_ids.contains(&edge.src_id) {
                continue;
            }
            if !dst_label_ids.contains(&edge.dst_id) {
                continue;
            }

            let existing_src = match self.store.get_node(&edge.src_id) {
                Ok(n) => n,
                Err(_) => continue,
            };
            let existing_dst = match self.store.get_node(&edge.dst_id) {
                Ok(n) => n,
                Err(_) => continue,
            };
            let (esrc_emb, edst_emb) = match (&existing_src.embedding, &existing_dst.embedding) {
                (Some(s), Some(d)) => (s, d),
                _ => continue,
            };

            let src_sim = cosine_similarity(src_emb, esrc_emb)?;
            let dst_sim = cosine_similarity(dst_emb, edst_emb)?;
            let combined = (src_sim + dst_sim) / 2.0;

            if combined >= threshold {
                matches.push(SimilarMatch {
                    existing_edge_id: edge.id.clone(),
                    src_similarity: src_sim,
                    dst_similarity: dst_sim,
                    combined_score: combined,
                });
            }
        }

        matches.sort_by(|a, b| {
            b.combined_score
                .partial_cmp(&a.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(matches)
    }
}
