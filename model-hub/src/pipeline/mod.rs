//! High-level AI pipelines: RAG, tool calling, structured extraction, and an
//! orchestrator that combines them.
//!
//! Several capabilities can be powered by **either** an LLM or the lightweight,
//! offline **Needle** engine. The choice is made per-capability through
//! [`EngineKind`], so a user can, for example, run tool-calling via Needle while
//! generating final answers with an LLM.

pub mod extraction;
pub mod orchestrator;
pub mod rag;
pub mod tools;

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

pub use extraction::StructuredExtractor;
pub use orchestrator::Orchestrator;
pub use rag::RagPipeline;
pub use tools::{graphdb_tool_schemas, ToolCall, ToolExecutor, ToolParam, ToolSchema};

/// Selects which engine powers a given capability (tool calling, structured
/// extraction, or text embeddings).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum EngineKind {
    /// Use the loaded LLM (text model) to perform the task.
    #[default]
    Llm,
    /// Use the lightweight, deterministic, offline Needle engine.
    Needle,
}

impl fmt::Display for EngineKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineKind::Llm => write!(f, "llm"),
            EngineKind::Needle => write!(f, "needle"),
        }
    }
}

impl FromStr for EngineKind {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "llm" | "model" => Ok(EngineKind::Llm),
            "needle" | "rules" | "heuristic" => Ok(EngineKind::Needle),
            other => Err(format!("unknown engine '{other}' (expected 'llm' or 'needle')")),
        }
    }
}

/// Per-capability engine selection plus generation parameters.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PipelineConfig {
    /// Which engine performs tool-call selection.
    pub tool_engine: EngineKind,
    /// Which engine performs structured extraction.
    pub extraction_engine: EngineKind,
    /// Which engine produces text embeddings.
    pub embedding_engine: EngineKind,
}

impl PipelineConfig {
    /// Set all three capabilities to the same engine.
    pub fn all(engine: EngineKind) -> Self {
        Self {
            tool_engine: engine,
            extraction_engine: engine,
            embedding_engine: engine,
        }
    }
}
