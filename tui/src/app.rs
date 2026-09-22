//! TUI application state and business logic.

use model_hub::models::text::StubTextModel;
use model_hub::{
    EngineKind, GenerationConfig, InMemoryGraph, Orchestrator, PipelineConfig, RagPipeline,
    StructuredExtractor, TextModel,
};

/// The interactive modes offered by the TUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Free-form chat / text generation.
    Chat,
    /// Retrieval-augmented generation over the demo knowledge base.
    Rag,
    /// Structured extraction from the input text.
    Extract,
    /// System / build information.
    Info,
}

impl Mode {
    /// All modes in tab order.
    pub const ALL: [Mode; 4] = [Mode::Chat, Mode::Rag, Mode::Extract, Mode::Info];

    /// Human-readable title.
    pub fn title(&self) -> &'static str {
        match self {
            Mode::Chat => "Chat",
            Mode::Rag => "RAG",
            Mode::Extract => "Extract",
            Mode::Info => "System",
        }
    }

    /// Index of this mode within [`Mode::ALL`].
    pub fn index(&self) -> usize {
        Mode::ALL.iter().position(|m| m == self).unwrap_or(0)
    }
}

/// The TUI application.
pub struct App {
    /// Currently active mode.
    pub mode: Mode,
    /// The current input buffer.
    pub input: String,
    /// The latest output to display.
    pub output: String,
    /// Engine used for tool calling / extraction / embeddings.
    pub engine: EngineKind,
    /// Whether the app should exit.
    pub should_quit: bool,
    graph: InMemoryGraph,
    model: Box<dyn TextModel>,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    /// Create a new application seeded with a demo knowledge base and a stub
    /// text model (fully offline).
    pub fn new() -> Self {
        let graph = InMemoryGraph::new();
        graph.ingest(
            "/kb/rust.md",
            "The Rust Programming Language",
            "markdown",
            "Rust is a systems programming language focused on safety, speed, and concurrency. \
             Ownership and borrowing guarantee memory safety without a garbage collector. Cargo is \
             the Rust build system and package manager.",
        );
        graph.ingest(
            "/kb/graphdb.md",
            "Graph Database Engine",
            "markdown",
            "The graphdb engine stores documents as nodes and chunks connected by edges. It \
             supports semantic search, retrieval-augmented generation, and tool calling.",
        );
        graph.ingest(
            "/kb/candle.md",
            "Candle ML Framework",
            "markdown",
            "Candle is a minimalist ML framework for Rust with CPU and GPU backends, powering \
             quantized Llama, Whisper, and CLIP models in this project.",
        );

        Self {
            mode: Mode::Chat,
            input: String::new(),
            output: "Welcome to graphdb-tui. Type a message and press Enter.\n\
                     Tab: switch mode  |  Ctrl+E: toggle engine  |  Esc/Ctrl+Q: quit"
                .to_string(),
            engine: EngineKind::Needle,
            should_quit: false,
            graph,
            model: Box::new(StubTextModel::new()),
        }
    }

    /// Advance to the next mode.
    pub fn next_mode(&mut self) {
        let idx = (self.mode.index() + 1) % Mode::ALL.len();
        self.mode = Mode::ALL[idx];
    }

    /// Toggle the engine between LLM and Needle.
    pub fn toggle_engine(&mut self) {
        self.engine = match self.engine {
            EngineKind::Llm => EngineKind::Needle,
            EngineKind::Needle => EngineKind::Llm,
        };
    }

    /// Handle a submitted input line according to the active mode.
    pub async fn submit(&mut self) {
        let input = self.input.trim().to_string();
        if input.is_empty() && self.mode != Mode::Info {
            return;
        }
        let result = match self.mode {
            Mode::Chat => self.run_chat(&input),
            Mode::Rag => self.run_rag(&input).await,
            Mode::Extract => self.run_extract(&input),
            Mode::Info => Ok(self.render_info()),
        };
        self.output = match result {
            Ok(s) => s,
            Err(e) => format!("Error: {e}"),
        };
        self.input.clear();
    }

    fn run_chat(&mut self, input: &str) -> model_hub::Result<String> {
        self.model.generate(input, &GenerationConfig::default())
    }

    async fn run_rag(&mut self, query: &str) -> model_hub::Result<String> {
        let rag = RagPipeline::new(4, self.engine).with_rerank(true);
        let ans = rag
            .answer(
                &self.graph,
                self.model.as_mut(),
                query,
                &GenerationConfig::default(),
            )
            .await?;
        let mut out = format!("{}\n\nSources:\n", ans.answer);
        for (i, r) in ans.context.iter().enumerate() {
            out.push_str(&format!(
                "  {}. {} (score {:.3})\n",
                i + 1,
                r.chunk.source_path,
                r.score
            ));
        }
        Ok(out)
    }

    fn run_extract(&mut self, text: &str) -> model_hub::Result<String> {
        let schema = r#"{"type":"object","properties":{"name":{"type":"string"},"organization":{"type":"string"},"date":{"type":"string"}}}"#;
        let extractor = StructuredExtractor::new(self.engine);
        let value = extractor.extract(text, schema, Some(self.model.as_mut()))?;
        Ok(serde_json::to_string_pretty(&value)?)
    }

    fn render_info(&self) -> String {
        let _ = Orchestrator::default();
        let _ = PipelineConfig::default();
        format!(
            "graphdb model hub TUI\n\
             engine (tools/extract/embed): {}\n\
             text model: {}\n\
             documents in demo KB: {}\n\
             chunks in demo KB: {}\n\
             gpu features: cuda={} metal={} mkl={}\n\n\
             Modes: Chat, RAG, Extract, System\n\
             Keys: Tab switch mode, Ctrl+E toggle engine, Enter submit, Esc quit",
            self.engine,
            self.model.name(),
            self.graph.document_count(),
            self.graph.chunk_count(),
            cfg!(feature = "cuda"),
            cfg!(feature = "metal"),
            cfg!(feature = "mkl"),
        )
    }
}
