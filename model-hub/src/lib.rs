//! # model-hub
//!
//! A pure-Rust AI model hub and pipeline library for the graphdb engine.
//!
//! Features:
//! * **Multi-modal models** via [`candle`](candle_core): quantized Llama (text,
//!   GGUF), Whisper (audio), and CLIP (image), each with an offline **stub**
//!   fallback so the whole system runs without downloading weights.
//! * **GPU feature flags** (`cuda`, `rocm`, `metal`, `mkl`, `flash-attn`), all
//!   **off by default** — the crate compiles and runs CPU-only out of the box.
//! * **Pipelines**: retrieval-augmented generation, tool calling
//!   (Needle-compatible schemas), and structured extraction.
//! * **Selectable engines**: tool calling, structured extraction, and text
//!   embeddings can each be powered by the **LLM** or the lightweight **Needle**
//!   engine — see [`pipeline::EngineKind`].
//! * **Decoupled storage** through the [`backend::GraphBackend`] trait, so the AI
//!   layer never links against the PyO3-based graph engine directly.

#![warn(missing_docs)]

pub mod backend;
pub mod error;
pub mod graph;
pub mod hub;
pub mod models;
pub mod needle;
pub mod pipeline;

pub use backend::{ChunkInfo, DocumentInfo, GraphBackend, SearchResult};
pub use error::{ModelHubError, Result};
pub use graph::InMemoryGraph;
pub use hub::ModelHub;
pub use models::{
    cosine_similarity, resolve_device, AudioModel, DeviceKind, GenerationConfig, ImageModel,
    ModelSource, TextModel,
};
pub use needle::{NeedleAgent, NeedleOrchestrator};
pub use pipeline::{
    graphdb_tool_schemas, EngineKind, Orchestrator, PipelineConfig, RagPipeline,
    StructuredExtractor, ToolCall, ToolExecutor, ToolSchema,
};
