//! Error types for the model-hub crate.

use thiserror::Error;

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, ModelHubError>;

/// Unified error type for model loading, inference, and pipeline operations.
#[derive(Debug, Error)]
pub enum ModelHubError {
    /// A model file or weight could not be found or downloaded.
    #[error("model not found: {0}")]
    ModelNotFound(String),

    /// A model failed to load (bad format, missing metadata, etc.).
    #[error("failed to load model: {0}")]
    ModelLoad(String),

    /// Inference (forward pass / generation) failed.
    #[error("inference error: {0}")]
    Inference(String),

    /// Tokenizer construction or encode/decode failed.
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// The graph backend returned an error.
    #[error("graph backend error: {0}")]
    Graph(String),

    /// Structured extraction / JSON parsing failed.
    #[error("extraction error: {0}")]
    Extraction(String),

    /// A tool call was malformed or referenced an unknown tool.
    #[error("tool error: {0}")]
    Tool(String),

    /// Configuration was invalid.
    #[error("configuration error: {0}")]
    Config(String),

    /// Wrapper around candle tensor errors.
    #[error("tensor error: {0}")]
    Tensor(#[from] candle_core::Error),

    /// Wrapper around IO errors.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Wrapper around JSON (de)serialization errors.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// Any other error.
    #[error("{0}")]
    Other(String),
}

impl From<anyhow::Error> for ModelHubError {
    fn from(e: anyhow::Error) -> Self {
        ModelHubError::Other(e.to_string())
    }
}

impl From<Box<dyn std::error::Error + Send + Sync>> for ModelHubError {
    fn from(e: Box<dyn std::error::Error + Send + Sync>) -> Self {
        ModelHubError::Other(e.to_string())
    }
}
