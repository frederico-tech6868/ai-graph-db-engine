//! `graphdb-cli` — command-line interface to the model-hub AI pipelines.
//!
//! By default all commands run with offline **stub** models and can use either
//! the **LLM** or the **Needle** engine for tool calling, structured
//! extraction, and text embeddings (via `--engine`). Point `--model`/`--tokenizer`
//! at real GGUF weights to run a quantized Llama instead of the stub.
//!
//! The `agent` sub-command launches a Claude-Code-style interactive REPL. See
//! [`agent`] for the full REPL implementation.

mod agent;
mod settings;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use colored::Colorize;

use model_hub::models::text::{QuantizedLlama, StubTextModel};
use model_hub::{
    graphdb_tool_schemas, resolve_device, DeviceKind, EngineKind, GenerationConfig, InMemoryGraph,
    Orchestrator, PipelineConfig, RagPipeline, StructuredExtractor, TextModel, ToolExecutor,
};

/// CLI engine choice mirroring [`model_hub::EngineKind`].
#[derive(Debug, Clone, Copy, ValueEnum)]
enum Engine {
    /// Use the loaded LLM.
    Llm,
    /// Use the offline Needle engine.
    Needle,
}

impl From<Engine> for EngineKind {
    fn from(e: Engine) -> Self {
        match e {
            Engine::Llm => EngineKind::Llm,
            Engine::Needle => EngineKind::Needle,
        }
    }
}

/// Compute device selection.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum Device {
    /// CPU (default, always available).
    Cpu,
    /// CUDA GPU (requires the `cuda` build feature).
    Cuda,
    /// Metal GPU (requires the `metal` build feature).
    Metal,
}

impl From<Device> for DeviceKind {
    fn from(d: Device) -> Self {
        match d {
            Device::Cpu => DeviceKind::Cpu,
            Device::Cuda => DeviceKind::Cuda,
            Device::Metal => DeviceKind::Metal,
        }
    }
}

/// graphdb model hub CLI.
#[derive(Parser)]
#[command(name = "graphdb-cli", version, about = "Pure-Rust AI pipelines for graphdb", long_about = None)]
struct Cli {
    /// Path to a GGUF text model. If omitted, an offline stub model is used.
    #[arg(long, global = true)]
    model: Option<PathBuf>,

    /// Path to the tokenizer.json for the GGUF model.
    #[arg(long, global = true)]
    tokenizer: Option<PathBuf>,

    /// Compute device. If omitted, falls back to settings.json then CPU.
    #[arg(long, global = true, value_enum)]
    device: Option<Device>,

    /// Path to a settings.json (agent sub-command). Overrides auto-discovery.
    #[arg(long, global = true, value_name = "FILE")]
    settings: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

impl Cli {
    /// The effective device: the CLI flag if given, else CPU. (For the `agent`
    /// sub-command, settings.json can supply the device before this default.)
    fn device_or_default(&self) -> Device {
        self.device.unwrap_or(Device::Cpu)
    }
}

/// Parse a device string from settings.json.
fn parse_device(s: &str) -> Result<Device> {
    match s.trim().to_lowercase().as_str() {
        "cpu" => Ok(Device::Cpu),
        "cuda" | "gpu" => Ok(Device::Cuda),
        "metal" => Ok(Device::Metal),
        other => anyhow::bail!("unknown device {other:?} (expected cpu, cuda, or metal)"),
    }
}

#[derive(Subcommand)]
enum Commands {
    /// Generate text from a prompt.
    Generate {
        /// The prompt.
        prompt: String,
        /// Maximum new tokens.
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
        /// Sampling temperature (0 = greedy).
        #[arg(long, default_value_t = 0.7)]
        temperature: f64,
    },
    /// Produce a text embedding vector.
    Embed {
        /// Text to embed.
        text: String,
        /// Engine used to compute the embedding.
        #[arg(long, value_enum, default_value_t = Engine::Needle)]
        engine: Engine,
    },
    /// Retrieval-augmented generation over a small demo knowledge base.
    Rag {
        /// The question to answer.
        query: String,
        /// Number of chunks to retrieve.
        #[arg(long, default_value_t = 4)]
        k: usize,
        /// Engine used for embedding-based re-ranking.
        #[arg(long, value_enum, default_value_t = Engine::Needle)]
        engine: Engine,
    },
    /// Extract structured JSON from text according to a schema.
    Extract {
        /// The text to extract from.
        text: String,
        /// A named built-in schema (`entities`) or an inline JSON schema string.
        #[arg(long, default_value = "entities")]
        schema: String,
        /// Engine used for extraction.
        #[arg(long, value_enum, default_value_t = Engine::Needle)]
        engine: Engine,
    },
    /// Run the agentic orchestrator (tool calling + answer) on the demo KB.
    Ask {
        /// The user question.
        query: String,
        /// Engine used to route/select tools.
        #[arg(long, value_enum, default_value_t = Engine::Needle)]
        engine: Engine,
    },
    /// Transcribe a mono f32 PCM WAV-less raw file (stub unless a model is wired).
    Transcribe {
        /// Path to a file of little-endian f32 samples at 16 kHz.
        path: PathBuf,
    },
    /// Print system / build information.
    Info,
    /// Launch the interactive Claude-Code-style agent REPL.
    ///
    /// The REPL classifies each task (Simple / Moderate / Complex), picks the
    /// best matching skill, runs a visible tool-call trace, and prints the
    /// composed answer. Slash commands: /help /settings /skills /tools /context
    /// /ingest /clear /quit.
    ///
    /// Configuration is read from a settings.json (see --settings) with
    /// precedence: CLI flag > settings.json > default.
    Agent {
        /// Directory of .rs / .md / .txt files to ingest into the knowledge
        /// base at startup (in addition to the built-in demo context).
        #[arg(long, value_name = "DIR")]
        context_dir: Option<PathBuf>,
    },
}

/// Build the text model: real GGUF Llama if paths are given, else the stub.
fn build_model(cli: &Cli) -> Result<Box<dyn TextModel>> {
    match (&cli.model, &cli.tokenizer) {
        (Some(m), Some(t)) => {
            let device = resolve_device(cli.device_or_default().into())?;
            let model = QuantizedLlama::load(m, t, device)
                .with_context(|| format!("loading GGUF model {}", m.display()))?;
            Ok(Box::new(model))
        }
        (Some(_), None) => {
            anyhow::bail!("--model requires --tokenizer");
        }
        _ => Ok(Box::new(StubTextModel::new())),
    }
}

/// Seed a small in-memory knowledge base for the RAG / Ask demos.
fn demo_graph() -> InMemoryGraph {
    let g = InMemoryGraph::new();
    g.ingest(
        "/kb/rust.md",
        "The Rust Programming Language",
        "markdown",
        "Rust is a systems programming language focused on safety, speed, and concurrency. \
         Ownership and borrowing let Rust guarantee memory safety without a garbage collector. \
         Cargo is Rust's build system and package manager.",
    );
    g.ingest(
        "/kb/graphdb.md",
        "Graph Database Engine",
        "markdown",
        "The graphdb engine stores documents as nodes and chunks connected by edges. \
         It supports semantic search, retrieval-augmented generation, and tool calling. \
         Vector similarity is used to rank chunks against a query.",
    );
    g.ingest(
        "/kb/candle.md",
        "Candle ML Framework",
        "markdown",
        "Candle is a minimalist ML framework for Rust with CPU and GPU (CUDA, Metal) backends. \
         It powers quantized Llama, Whisper, and CLIP models in this project.",
    );
    g
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    match &cli.command {
        Commands::Generate {
            prompt,
            max_tokens,
            temperature,
        } => {
            let mut model = build_model(&cli)?;
            let cfg = GenerationConfig::default()
                .with_max_tokens(*max_tokens)
                .with_temperature(*temperature);
            let out = model.generate(prompt, &cfg)?;
            println!("{}", out);
        }
        Commands::Embed { text, engine } => {
            let cfg = PipelineConfig::all((*engine).into());
            let rag = RagPipeline::new(4, cfg.embedding_engine);
            let mut model = build_model(&cli)?;
            let emb = rag.embed(text, Some(model.as_mut()))?;
            println!(
                "{} dims={} (engine={})",
                "embedding".green().bold(),
                emb.len(),
                EngineKind::from(*engine)
            );
            let preview: Vec<String> = emb.iter().take(8).map(|x| format!("{x:.4}")).collect();
            println!("[{}, ...]", preview.join(", "));
        }
        Commands::Rag { query, k, engine } => {
            let graph = demo_graph();
            let rag = RagPipeline::new(*k, (*engine).into()).with_rerank(true);
            let mut model = build_model(&cli)?;
            let ans = rag
                .answer(&graph, model.as_mut(), query, &GenerationConfig::default())
                .await?;
            println!("{}", "Answer:".green().bold());
            println!("{}\n", ans.answer);
            println!("{}", "Sources:".cyan().bold());
            for (i, r) in ans.context.iter().enumerate() {
                println!("  {}. {} (score {:.3})", i + 1, r.chunk.source_path, r.score);
            }
        }
        Commands::Extract {
            text,
            schema,
            engine,
        } => {
            let schema_json = resolve_schema(schema);
            let extractor = StructuredExtractor::new((*engine).into());
            let mut model = build_model(&cli)?;
            let value = extractor.extract(text, &schema_json, Some(model.as_mut()))?;
            println!("{}", serde_json::to_string_pretty(&value)?);
        }
        Commands::Ask { query, engine } => {
            let graph = demo_graph();
            let orch = Orchestrator::new(PipelineConfig {
                tool_engine: (*engine).into(),
                ..Default::default()
            });
            let mut model = build_model(&cli)?;
            let turn = orch
                .run(&graph, model.as_mut(), query, &GenerationConfig::default())
                .await?;
            if let Some(tool) = &turn.tool_name {
                println!("{} {}", "Tool:".yellow().bold(), tool);
            }
            println!("{}", "Answer:".green().bold());
            println!("{}", turn.answer);
        }
        Commands::Transcribe { path } => {
            let bytes = std::fs::read(path)
                .with_context(|| format!("reading {}", path.display()))?;
            let samples: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            use model_hub::models::audio::StubAudioModel;
            use model_hub::AudioModel;
            let mut audio = StubAudioModel::new();
            println!("{}", audio.transcribe(&samples)?);
        }
        Commands::Info => print_info(&cli),
        Commands::Agent { context_dir } => {
            // Load settings.json (or built-in defaults).
            let (settings, source) = settings::AgentSettings::load(cli.settings.as_deref())?;

            // Merge with precedence: CLI flag > settings.json > default.
            let model_path = cli.model.clone().or_else(|| settings.model_path());
            let tokenizer_path = cli.tokenizer.clone().or_else(|| settings.tokenizer_path());
            let device: DeviceKind = match cli.device {
                Some(d) => d.into(),
                None => match &settings.model.device {
                    Some(s) => parse_device(s)?.into(),
                    None => DeviceKind::Cpu,
                },
            };
            let context_dir = context_dir.clone().or_else(|| settings.context_dir_path());

            agent::run(agent::AgentConfig {
                context_dir,
                model_path,
                tokenizer_path,
                device,
                max_tokens: settings.generation.max_tokens,
                temperature: settings.generation.temperature,
                skills_dir: settings.skills_dir_path(),
                history_file: settings.history_path(),
                settings_source: source,
            })
            .await?;
        }
    }

    Ok(())
}

/// Resolve a schema argument: a built-in name or an inline JSON schema.
fn resolve_schema(schema: &str) -> String {
    match schema {
        "entities" => r#"{"type":"object","properties":{"name":{"type":"string"},"organization":{"type":"string"},"date":{"type":"string"}}}"#.to_string(),
        other => other.to_string(),
    }
}

/// Print build/system information.
fn print_info(cli: &Cli) {
    println!("{}", "graphdb model hub".green().bold());
    println!("  version:   {}", env!("CARGO_PKG_VERSION"));
    println!("  device:    {:?}", DeviceKind::from(cli.device_or_default()));
    let backend = if cli.model.is_some() {
        "quantized-llama (GGUF)"
    } else {
        "stub-text (offline)"
    };
    println!("  text model:{backend}");
    println!(
        "  gpu features: cuda={} metal={} mkl={} flash-attn={}",
        cfg!(feature = "cuda"),
        cfg!(feature = "metal"),
        cfg!(feature = "mkl"),
        cfg!(feature = "flash-attn"),
    );
    println!("  engines:   llm | needle (selectable for tools, extraction, embeddings)");
    println!("{}", "Registered tools:".cyan().bold());
    for schema in graphdb_tool_schemas() {
        println!("  - {}: {}", schema.name, schema.description);
    }
    // Reference ToolExecutor so the import is always used.
    let _ = std::any::type_name::<ToolExecutor>();
}
