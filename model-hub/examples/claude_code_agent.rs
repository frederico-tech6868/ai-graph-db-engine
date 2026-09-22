//! # Claude-Code-style multi-agent coding assistant
//!
//! A coding-assistant example (in the spirit of "Claude Code") that answers
//! questions **grounded in the graph-db-engine codebase** and is driven by a
//! **main orchestrator that delegates by task difficulty** across two engines:
//!
//! * **Needle** — a lightweight, deterministic, fully-offline engine (regex /
//!   heuristics, no weights). Handles *simple* tasks instantly: listing files,
//!   fetching a document's chunks, direct lookups.
//! * **Local LLM** — a text model ([`TextModel`]) that reasons and writes prose.
//!   Handles *complex* tasks: explanations, design, synthesis, comparisons.
//!   By default this is the offline [`StubTextModel`]; point it at a local GGUF
//!   file to run a real quantized Llama (see "Selecting a local model" below).
//!
//! ## The difficulty router (the "main orchestrator")
//!
//! Every task is first classified into a [`Difficulty`] by
//! [`DifficultyRouter::classify`]. The router then delegates:
//!
//! | Difficulty | Engine(s)             | How it runs                                             |
//! |------------|-----------------------|--------------------------------------------------------|
//! | `Simple`   | Needle only           | [`NeedleOrchestrator`] picks + runs a tool, no LLM.     |
//! | `Moderate` | Needle **+** LLM      | Needle routes the tool; the LLM composes the answer.    |
//! | `Complex`  | LLM (with tools)      | The LLM routes the tool **and** composes the answer.    |
//!
//! This is the "multi-agent combination between Needle and a local LLM": the
//! cheap engine handles what it can, and only genuinely hard tasks pay for LLM
//! inference. The classification is transparent — each turn prints the chosen
//! difficulty, the reason, and which engine(s) ran.
//!
//! ## Run it
//! ```bash
//! # fully offline (stub LLM):
//! cargo run -p model-hub --example claude_code_agent
//!
//! # with a real local model (quantized Llama GGUF):
//! GRAPHDB_MODEL=/path/to/model.gguf \
//! GRAPHDB_TOKENIZER=/path/to/tokenizer.json \
//! cargo run -p model-hub --example claude_code_agent
//! ```

use std::env;

use model_hub::models::text::{QuantizedLlama, StubTextModel};
use model_hub::{
    resolve_device, DeviceKind, EngineKind, GenerationConfig, InMemoryGraph, NeedleOrchestrator,
    Orchestrator, PipelineConfig, TextModel,
};

/// How hard a task is, and therefore which engine should handle it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Difficulty {
    /// Deterministic lookup — the Needle engine alone can answer.
    Simple,
    /// Needs retrieval + a short synthesized answer — Needle routes, LLM writes.
    Moderate,
    /// Open-ended reasoning / generation — the LLM drives end to end.
    Complex,
}

impl Difficulty {
    /// A short label for display.
    fn label(self) -> &'static str {
        match self {
            Difficulty::Simple => "SIMPLE",
            Difficulty::Moderate => "MODERATE",
            Difficulty::Complex => "COMPLEX",
        }
    }
}

/// The main orchestrator: classifies a task and delegates to the right engine.
struct DifficultyRouter {
    /// Generation parameters passed to the LLM when it is used.
    gen_config: GenerationConfig,
}

impl DifficultyRouter {
    fn new() -> Self {
        Self {
            gen_config: GenerationConfig::deterministic().with_max_tokens(160),
        }
    }

    /// Classify a task into a [`Difficulty`], returning the reason too.
    ///
    /// Heuristics (cheap and transparent, no model needed):
    /// * "list / show all documents" or "chunks of <path>" → **Simple** (a pure
    ///   tool call Needle can resolve).
    /// * reasoning verbs ("explain", "why", "how", "design", "refactor",
    ///   "compare", "implement", "write", "summarize"…) or long/multi-part
    ///   requests → **Complex**.
    /// * everything else (a focused factual lookup) → **Moderate**.
    fn classify(&self, task: &str) -> (Difficulty, &'static str) {
        let q = task.to_lowercase();
        let words = q.split_whitespace().count();

        // Simple: deterministic tool intents.
        let is_list = (q.contains("list") || q.contains("show all") || q.contains("what documents"))
            && q.contains("document");
        let is_chunks = q.contains("chunk") || q.contains("sections of");
        if is_list || is_chunks {
            return (Difficulty::Simple, "deterministic tool intent (list/chunks)");
        }

        // Complex: reasoning / generation verbs, or a long multi-part request.
        const COMPLEX_CUES: [&str; 14] = [
            "explain", "why", "how ", "design", "refactor", "compare", "implement",
            "write", "generate", "summarize", "architecture", "trade-off", "tradeoff",
            "walk me through",
        ];
        if COMPLEX_CUES.iter().any(|c| q.contains(c)) {
            return (Difficulty::Complex, "reasoning/generation cue detected");
        }
        if words > 18 || q.matches('?').count() > 1 {
            return (Difficulty::Complex, "long or multi-part request");
        }

        // Otherwise: a focused factual lookup.
        (Difficulty::Moderate, "focused factual lookup")
    }

    /// Handle one task end to end, delegating by difficulty. Returns the answer
    /// and a description of which engine(s) ran.
    async fn handle(
        &self,
        graph: &InMemoryGraph,
        model: &mut dyn TextModel,
        task: &str,
    ) -> model_hub::Result<(Difficulty, &'static str, String, String)> {
        let (difficulty, reason) = self.classify(task);
        let (engine_desc, answer) = match difficulty {
            // Needle only — no LLM inference at all.
            Difficulty::Simple => {
                let needle = NeedleOrchestrator::new();
                let out = needle.run(graph, task).await?;
                ("needle (tool only)".to_string(), out)
            }
            // Needle routes the tool; the local LLM composes the final answer.
            Difficulty::Moderate => {
                let orch = Orchestrator::new(PipelineConfig {
                    tool_engine: EngineKind::Needle,
                    extraction_engine: EngineKind::Needle,
                    embedding_engine: EngineKind::Needle,
                });
                let turn = orch.run(graph, model, task, &self.gen_config).await?;
                (
                    format!(
                        "needle (route: {}) + {} (compose)",
                        turn.tool_name.as_deref().unwrap_or("none"),
                        model.name()
                    ),
                    turn.answer,
                )
            }
            // LLM drives tool routing and answer composition.
            Difficulty::Complex => {
                let orch = Orchestrator::new(PipelineConfig::all(EngineKind::Llm));
                let turn = orch.run(graph, model, task, &self.gen_config).await?;
                (
                    format!(
                        "{} (route: {} + compose)",
                        model.name(),
                        turn.tool_name.as_deref().unwrap_or("none")
                    ),
                    turn.answer,
                )
            }
        };
        Ok((difficulty, reason, engine_desc, answer))
    }
}

/// Load the local LLM: a real GGUF quantized Llama if `GRAPHDB_MODEL` (and
/// `GRAPHDB_TOKENIZER`) are set, otherwise the offline stub.
fn load_model() -> model_hub::Result<Box<dyn TextModel>> {
    match (env::var("GRAPHDB_MODEL"), env::var("GRAPHDB_TOKENIZER")) {
        (Ok(weights), Ok(tokenizer)) => {
            let device = resolve_device(DeviceKind::Cpu)?;
            println!("loading local GGUF model: {weights}");
            let model = QuantizedLlama::load(&weights, &tokenizer, device)?;
            Ok(Box::new(model))
        }
        _ => {
            println!("no GRAPHDB_MODEL set — using offline stub LLM");
            Ok(Box::new(StubTextModel::new()))
        }
    }
}

/// Seed the knowledge base with a compact description of the graph-db-engine
/// codebase, so the coding assistant has real project context to work from.
fn build_code_context() -> InMemoryGraph {
    let g = InMemoryGraph::new();
    g.ingest(
        "/model-hub/src/pipeline/orchestrator.rs",
        "Orchestrator",
        "code",
        "The Orchestrator runs a full agentic turn: it selects a tool via the configured engine \
         (LLM or Needle), executes it against a GraphBackend, then has the text model compose a \
         final answer grounded in the tool result. Engine selection is per-capability through \
         PipelineConfig.",
    );
    g.ingest(
        "/model-hub/src/needle/mod.rs",
        "Needle Engine",
        "code",
        "Needle is a lightweight, deterministic, fully-offline engine that replaces an LLM for \
         tool selection, structured extraction, and text embeddings. It uses regex and heuristics \
         instead of neural weights, so it runs instantly. NeedleOrchestrator drives a complete \
         tool-calling turn with no LLM.",
    );
    g.ingest(
        "/model-hub/src/models/mod.rs",
        "Model Traits",
        "code",
        "Three modality traits decouple pipelines from implementations: TextModel (generate and \
         embed), AudioModel (transcribe), and ImageModel (embed_image/embed_text). Each has a real \
         candle-transformers implementation and an offline stub, selected at load time.",
    );
    g.ingest(
        "/model-hub/src/pipeline/rag.rs",
        "RAG Pipeline",
        "code",
        "RagPipeline retrieves top-k chunks from a GraphBackend, optionally re-ranks them by \
         embedding similarity, builds a grounded prompt, and generates an answer with a TextModel. \
         The embedding engine (LLM or Needle) is configurable.",
    );
    g.ingest(
        "/rust/src/jepa.rs",
        "JEPA World Model",
        "code",
        "The Graph-JEPA world model has context/target encoders and a predictor trained with the \
         VICReg loss (invariance, variance, covariance). JEPAGraphRAG combines it with GraphRAG for \
         latent search over communities and nodes plus hybrid retrieval.",
    );
    g
}

#[tokio::main]
async fn main() -> model_hub::Result<()> {
    println!("=== Claude-Code-style multi-agent coding assistant ===\n");

    let graph = build_code_context();
    println!(
        "code context: {} files / {} chunks ingested\n",
        graph.document_count(),
        graph.chunk_count()
    );

    let mut model = load_model()?;
    let router = DifficultyRouter::new();
    println!("main orchestrator: difficulty router  |  simple->needle  moderate->needle+llm  complex->llm\n");

    // A batch of coding tasks spanning all three difficulty levels.
    let tasks = [
        "list all documents in the codebase",
        "show me the chunks of /rust/src/jepa.rs",
        "what does the Needle engine do",
        "which file defines the RAG pipeline",
        "explain how the Orchestrator delegates work between the LLM and Needle engines",
        "compare the trade-offs of using Needle versus a local LLM for tool routing",
    ];

    for (i, task) in tasks.iter().enumerate() {
        let (difficulty, reason, engine_desc, answer) =
            router.handle(&graph, model.as_mut(), task).await?;
        println!("────────────────────────────────────────────────────────");
        println!("task {}: {task}", i + 1);
        println!("  routed as : {} ({reason})", difficulty.label());
        println!("  handled by: {engine_desc}");
        println!("  answer    : {}", truncate(&answer, 320));
        println!();
    }

    println!("Tip: set GRAPHDB_MODEL and GRAPHDB_TOKENIZER to run a real local GGUF model.");
    Ok(())
}

/// Truncate a string for tidy console output.
fn truncate(s: &str, n: usize) -> String {
    let t: String = s.chars().take(n).collect();
    if s.chars().count() > n {
        format!("{t}…")
    } else {
        t
    }
}
