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
    /// Human-readable source of the settings that were loaded.
    pub settings_source: String,
    graph: InMemoryGraph,
    /// The active text model (Chat / RAG).
    model: Box<dyn TextModel>,
    /// The alternate text model, available when Ctrl+M is pressed.
    /// `None` means no toggle is possible (footer hides the hint).
    alt_model: Option<Box<dyn TextModel>>,
    /// Generation parameters derived from settings.
    gen: GenerationConfig,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    /// Seed and return a fresh knowledge-base graph.
    fn make_graph() -> InMemoryGraph {
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
        graph
    }

    /// Create a new application with explicit model(s) and generation config.
    ///
    /// * `primary`         — the text model used on startup.
    /// * `alt`             — optional alternate model reachable via Ctrl+M.
    /// * `gen`             — generation parameters from settings.
    /// * `settings_source` — human-readable label of where settings came from.
    pub fn with_models(
        primary: Box<dyn TextModel>,
        alt: Option<Box<dyn TextModel>>,
        gen: GenerationConfig,
        settings_source: String,
    ) -> Self {
        let welcome = if alt.is_some() {
            "Welcome to graphdb-tui. Type a message and press Enter.\n\
             Tab: switch mode  |  Ctrl+E: toggle engine  |  Ctrl+M: switch model  |  Esc/Ctrl+Q: quit"
        } else {
            "Welcome to graphdb-tui. Type a message and press Enter.\n\
             Tab: switch mode  |  Ctrl+E: toggle engine  |  Esc/Ctrl+Q: quit"
        };
        Self {
            mode: Mode::Chat,
            input: String::new(),
            output: welcome.to_string(),
            engine: EngineKind::Needle,
            should_quit: false,
            settings_source,
            graph: Self::make_graph(),
            model: primary,
            alt_model: alt,
            gen,
        }
    }

    /// Create a new application seeded with a demo knowledge base and a stub
    /// text model (fully offline). Equivalent to calling `with_models` with
    /// defaults — used when no settings file is found.
    pub fn new() -> Self {
        Self::with_models(
            Box::new(StubTextModel::new()),
            None,
            GenerationConfig::default(),
            "built-in defaults".to_string(),
        )
    }

    // ── model toggle ──────────────────────────────────────────────────────────

    /// Whether a second model is available to toggle to with Ctrl+M.
    pub fn can_toggle_model(&self) -> bool {
        self.alt_model.is_some()
    }

    /// The name of the currently active text model.
    pub fn active_model_name(&self) -> &str {
        self.model.name()
    }

    /// Swap the active model with the alternate (no-op if no alternate exists).
    pub fn toggle_chat_model(&mut self) {
        if let Some(alt) = self.alt_model.as_mut() {
            std::mem::swap(&mut self.model, alt);
            self.output = format!(
                "Switched to model: {}\n(Ctrl+M to switch back)",
                self.model.name()
            );
        }
    }

    // ── mode / engine ─────────────────────────────────────────────────────────

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

    // ── submit ────────────────────────────────────────────────────────────────

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
        self.model.generate(input, &self.gen)
    }

    async fn run_rag(&mut self, query: &str) -> model_hub::Result<String> {
        let rag = RagPipeline::new(4, self.engine).with_rerank(true);
        let ans = rag
            .answer(
                &self.graph,
                self.model.as_mut(),
                query,
                &self.gen,
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
        let alt_name = self
            .alt_model
            .as_ref()
            .map(|m| m.name().to_string())
            .unwrap_or_else(|| "none (Ctrl+M unavailable)".to_string());
        format!(
            "graphdb model hub TUI\n\
             settings source:              {}\n\
             engine (tools/extract/embed): {}\n\
             active text model:            {}\n\
             alt text model:               {}\n\
             documents in demo KB:         {}\n\
             chunks in demo KB:            {}\n\
             gpu features: cuda={} metal={} mkl={}\n\n\
             Modes: Chat, RAG, Extract, System\n\
             Keys: Tab switch mode, Ctrl+E toggle engine, Ctrl+M switch model, Enter submit, Esc quit",
            self.settings_source,
            self.engine,
            self.model.name(),
            alt_name,
            self.graph.document_count(),
            self.graph.chunk_count(),
            cfg!(feature = "cuda"),
            cfg!(feature = "metal"),
            cfg!(feature = "mkl"),
        )
    }
}
