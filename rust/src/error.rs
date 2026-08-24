//! Error types for the graph database engine.
//!
//! Every public library API returns [`Result<T>`] so that the library never
//! panics on recoverable errors. The PyO3 layer (see `lib.rs`) maps each
//! variant onto an appropriate Python exception.

/// The error type used throughout the crate.
#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    /// A node with the given id does not exist.
    #[error("Node not found: {0}")]
    NodeNotFound(String),

    /// An edge with the given id does not exist.
    #[error("Edge not found: {0}")]
    EdgeNotFound(String),

    /// Two vectors of differing lengths were compared.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// An underlying I/O error (persistence).
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// A (de)serialization error from `serde_json`.
    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

/// Convenience alias for `Result<T, GraphError>`.
pub type Result<T> = std::result::Result<T, GraphError>;
