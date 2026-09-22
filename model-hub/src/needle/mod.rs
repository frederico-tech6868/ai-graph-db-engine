//! The **Needle** engine: a lightweight, deterministic, fully-offline
//! alternative to an LLM for three capabilities — tool selection, structured
//! extraction, and text embeddings.
//!
//! Needle uses heuristics and regular expressions instead of a neural model, so
//! it runs instantly with no weights. It is Needle-rs-compatible in spirit: an
//! [`NeedleAgent`] exposes intent detection and extraction, and
//! [`NeedleOrchestrator`] drives a full tool-calling turn against a
//! [`GraphBackend`](crate::backend::GraphBackend).

pub mod agent;
pub mod orchestrator;

pub use agent::NeedleAgent;
pub use orchestrator::NeedleOrchestrator;

/// Fixed embedding dimension for the Needle text embedder.
pub const NEEDLE_EMBED_DIM: usize = 384;

/// Deterministic hashed bag-of-words embedding used by the Needle engine.
pub fn needle_embed(text: &str) -> Vec<f32> {
    let mut v = vec![0f32; NEEDLE_EMBED_DIM];
    for token in text.split(|c: char| !c.is_alphanumeric()).filter(|s| !s.is_empty()) {
        let lower = token.to_lowercase();
        let mut h: u64 = 1469598103934665603;
        for b in lower.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(1099511628211);
        }
        let bucket = (h as usize) % NEEDLE_EMBED_DIM;
        let sign = if (h >> 40) & 1 == 0 { 1.0 } else { -1.0 };
        v[bucket] += sign;
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    v
}
