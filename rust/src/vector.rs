//! Vector utilities: cosine similarity, L2 normalization and top-k search.
//!
//! The pure-Rust implementations mirror the semantics of the Python
//! `graphdb.vector` module: cosine similarity returns `0.0` when either vector
//! is empty or has zero magnitude, and raises a dimension-mismatch error when
//! the lengths differ.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::error::{GraphError, Result};

/// Cosine similarity between two vectors: `dot(a, b) / (||a|| * ||b||)`.
///
/// Returns `0.0` if either vector is empty or has zero magnitude. Returns a
/// [`GraphError::DimensionMismatch`] if the lengths differ.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(GraphError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.is_empty() {
        return Ok(0.0);
    }

    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return Ok(0.0);
    }
    Ok(dot / (na.sqrt() * nb.sqrt()))
}

/// L2-normalize a vector. Returns a zero vector (a copy) if the norm is zero.
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

/// A scored candidate used inside the min-heap for top-k selection.
///
/// The heap is a max-heap by default, so we invert the ordering to keep the
/// *smallest* score at the top (a min-heap), allowing O(n log k) selection.
struct Scored {
    id: String,
    score: f32,
}

impl PartialEq for Scored {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl Eq for Scored {}

impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse so the min score is at the top of the BinaryHeap (max-heap).
        // NaN is treated as the smallest so it gets evicted first.
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Less)
    }
}

/// Return the top-`k` `(id, score)` pairs sorted by descending cosine similarity.
///
/// Uses a bounded min-heap ([`BinaryHeap`]) for efficient O(n log k) selection.
/// Candidates whose dimensions do not match the query are skipped.
pub fn top_k_similar(
    query: &[f32],
    candidates: &[(&str, &[f32])],
    k: usize,
) -> Result<Vec<(String, f32)>> {
    if k == 0 {
        return Ok(Vec::new());
    }

    let mut heap: BinaryHeap<Scored> = BinaryHeap::with_capacity(k + 1);
    for (id, vec) in candidates {
        let score = match cosine_similarity(query, vec) {
            Ok(s) => s,
            Err(GraphError::DimensionMismatch { .. }) => continue,
            Err(e) => return Err(e),
        };
        heap.push(Scored {
            id: (*id).to_string(),
            score,
        });
        if heap.len() > k {
            // Evict the current minimum.
            heap.pop();
        }
    }

    // Drain and sort descending by score.
    let mut result: Vec<(String, f32)> = heap.into_iter().map(|s| (s.id, s.score)).collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    Ok(result)
}
