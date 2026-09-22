//! Model abstractions and implementations.
//!
//! Three modalities are supported, each behind a trait so that real
//! (candle-transformers backed) and stub (offline) implementations are
//! interchangeable:
//!
//! * [`TextModel`]  — text generation & embeddings (Llama GGUF / stub).
//! * [`AudioModel`] — speech-to-text (Whisper / stub).
//! * [`ImageModel`] — image & text embeddings (CLIP / stub).

pub mod audio;
pub mod config;
pub mod image;
pub mod text;

use candle_core::Device;

pub use config::{DeviceKind, GenerationConfig, ModelSource};

use crate::error::Result;

/// Resolve a [`DeviceKind`] into a concrete candle [`Device`].
///
/// GPU variants require the corresponding cargo feature to be enabled; when it
/// is not, this falls back to CPU with a warning.
pub fn resolve_device(kind: DeviceKind) -> Result<Device> {
    match kind {
        DeviceKind::Cpu => Ok(Device::Cpu),
        DeviceKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0).map_err(crate::error::ModelHubError::from)
            }
            #[cfg(not(feature = "cuda"))]
            {
                tracing::warn!("cuda feature not enabled; falling back to CPU");
                Ok(Device::Cpu)
            }
        }
        DeviceKind::Metal => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0).map_err(crate::error::ModelHubError::from)
            }
            #[cfg(not(feature = "metal"))]
            {
                tracing::warn!("metal feature not enabled; falling back to CPU");
                Ok(Device::Cpu)
            }
        }
    }
}

/// A text-generation / embedding model.
pub trait TextModel: Send {
    /// Generate a completion for `prompt`.
    fn generate(&mut self, prompt: &str, config: &GenerationConfig) -> Result<String>;

    /// Produce a fixed-size embedding vector for `text`.
    fn embed(&mut self, text: &str) -> Result<Vec<f32>>;

    /// A short human-readable identifier for the loaded model.
    fn name(&self) -> &str;
}

/// A speech-to-text model.
pub trait AudioModel: Send {
    /// Transcribe 16 kHz mono PCM samples into text.
    fn transcribe(&mut self, samples: &[f32]) -> Result<String>;

    /// Produce an embedding for the given audio samples.
    fn embed(&mut self, samples: &[f32]) -> Result<Vec<f32>>;

    /// A short human-readable identifier for the loaded model.
    fn name(&self) -> &str;
}

/// An image (and paired text) embedding model.
pub trait ImageModel: Send {
    /// Produce an embedding for raw RGB image pixels (`height` x `width` x 3).
    fn embed_image(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<Vec<f32>>;

    /// Produce an embedding for a text caption in the shared CLIP space.
    fn embed_text(&mut self, text: &str) -> Result<Vec<f32>>;

    /// A short human-readable identifier for the loaded model.
    fn name(&self) -> &str;
}

/// Cosine similarity between two equal-length vectors. Returns 0 for mismatched
/// or empty inputs.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}
