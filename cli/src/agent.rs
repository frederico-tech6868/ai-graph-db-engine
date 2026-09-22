//! # Claude-Code-style agent REPL
//!
//! Implements the interactive `graphdb-cli agent` sub-command: a terminal REPL
//! modelled on Claude Code's UX — a diamond prompt `◆`, visible
//! `tool_use → tool_result` traces, selected-skill annotations, difficulty
//! routing, and slash commands for meta-operations.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     REPL loop (rustyline)                        │
//! │  ◆ <user input>                                                  │
//! │       │                                                           │
//! │       ├─ /slash command  ─► handle_command()                     │
//! │       │                                                           │
//! │       └─ task  ──► DifficultyRouter::classify()                  │
//! │                          │                                        │
//! │              ┌───────────┴───────────┐                           │
//! │           Simple                Moderate / Complex                │
//! │        Needle only           SkillRegistry::select()             │
//! │              │                      │                             │
//! │              │              run_agent_turn()                      │
//! │              │            ┌─────────┴──────────┐                 │
//! │              │       select_tool()         compose                │
//! │              │       tool_use trace        with LLM               │
//! │              │       ToolExecutor::execute                        │
//! │              │       tool_result trace                            │
//! │              │                                                     │
//! │              └────────────────► print answer                      │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Slash commands
//!
//! | Command | Effect |
//! |---------|--------|
//! | `/help` | Print this table |
//! | `/skills` | List loaded skills (name, description, cues) |
//! | `/tools` | List available tools |
//! | `/context` | Show how many documents/chunks are in the KB |
//! | `/ingest <path> <title>` | Read a file and add it to the KB |
//! | `/clear` | Clear the terminal screen |
//! | `/quit` / `/exit` | Exit the REPL |

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use colored::Colorize;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use serde_json::Value;

use model_hub::models::text::{QuantizedLlama, StubTextModel};
use model_hub::pipeline::tools::select_tool;
use model_hub::{
    graphdb_tool_schemas, resolve_device, DeviceKind, EngineKind, GenerationConfig, InMemoryGraph,
    TextModel, ToolExecutor,
};

// ─────────────────────────────── skills ────────────────────────────────────

/// A reusable instruction bundle loaded from a `SKILL.md` file on disk.
///
/// Mirrors Claude Code's Agent Skills: a named folder whose `SKILL.md` has a
/// YAML-ish frontmatter block (`name`, `description`, `cues`) followed by an
/// instruction body that is injected into the LLM prompt when the skill fires.
#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub description: String,
    /// Lowercase trigger phrases — any match makes the skill eligible.
    pub cues: Vec<String>,
    /// Instructions injected into the LLM prompt.
    pub instructions: String,
}

impl Skill {
    /// Parse a `SKILL.md` with `---` frontmatter then an instruction body.
    pub fn parse(md: &str) -> Option<Self> {
        let rest = md.strip_prefix("---")?;
        let end = rest.find("\n---")?;
        let (front, body) = rest.split_at(end);
        let body = body.trim_start_matches("\n---").trim().to_string();

        let (mut name, mut description, mut cues) = (String::new(), String::new(), Vec::new());
        for line in front.lines() {
            let Some((k, v)) = line.split_once(':') else { continue };
            match k.trim() {
                "name" => name = v.trim().to_string(),
                "description" => description = v.trim().to_string(),
                "cues" => {
                    cues = v.trim().split(',').map(|c| c.trim().to_lowercase())
                        .filter(|c| !c.is_empty()).collect()
                }
                _ => {}
            }
        }
        if name.is_empty() || body.is_empty() { return None; }
        Some(Skill { name, description, cues, instructions: body })
    }

    /// Number of cue matches in `task_lower`.
    pub fn score(&self, task_lower: &str) -> usize {
        self.cues.iter().filter(|c| task_lower.contains(c.as_str())).count()
    }
}

/// Discovers skills from `<manifest_dir>/examples/skills/*/SKILL.md` at
/// startup, falling back to a baked-in set when the folder is absent.
pub struct SkillRegistry {
    skills: Vec<Skill>,
}

impl SkillRegistry {
    /// Load skills. Tries the on-disk folder first; falls back to builtins.
    pub fn load() -> Self {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()          // workspace root
            .unwrap_or(Path::new("."))
            .join("model-hub")
            .join("examples")
            .join("skills");
        let mut skills = Self::load_dir(&dir);
        if skills.is_empty() { skills = builtin_skills(); }
        skills.sort_by(|a, b| a.name.cmp(&b.name));
        Self { skills }
    }

    fn load_dir(dir: &Path) -> Vec<Skill> {
        let mut out = Vec::new();
        let Ok(entries) = fs::read_dir(dir) else { return out };
        for entry in entries.flatten() {
            let skill_md = entry.path().join("SKILL.md");
            if let Ok(text) = fs::read_to_string(&skill_md) {
                if let Some(s) = Skill::parse(&text) { out.push(s); }
            }
        }
        out
    }

    /// Best-matching skill for `task`, or `None` if no cue matches.
    pub fn select(&self, task: &str) -> Option<&Skill> {
        let q = task.to_lowercase();
        self.skills.iter()
            .map(|s| (s, s.score(&q)))
            .filter(|(_, n)| *n > 0)
            .max_by_key(|(_, n)| *n)
            .map(|(s, _)| s)
    }

    /// Pretty multi-line catalog for `/skills`.
    pub fn catalog(&self) -> String {
        self.skills.iter().map(|s| {
            format!(
                "  {} {}\n    {}\n    cues: {}",
                "◈".cyan(),
                s.name.bold(),
                s.description,
                s.cues.join(", ").dimmed()
            )
        }).collect::<Vec<_>>().join("\n\n")
    }

    /// Comma-separated list of skill names.
    pub fn names(&self) -> String {
        self.skills.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(", ")
    }
}

fn builtin_skills() -> Vec<Skill> {
    [
        ("codebase-navigator",
         "Locate files, list documents, and fetch chunks.",
         "list, show, find, where, which file, locate, chunk, sections of, what documents",
         "Use the tools to locate documents and chunks; never guess paths. Return exact file \
          paths and be terse: a short list or a single path, no prose padding."),
        ("code-explainer",
         "Explain how a piece of the codebase works.",
         "explain, how, why, what does, walk me through, describe, understand",
         "Ground every claim in the retrieved chunks and reference the file paths. Lead with a \
          one-sentence summary, then 2–4 supporting points. Never invent APIs not in context."),
        ("refactor-planner",
         "Plan an implementation or refactor.",
         "refactor, implement, design, add, change, migrate, rewrite, introduce, build",
         "Respond with a numbered, step-by-step plan. Name the affected files per step and end \
          with a short Risks / trade-offs note. Be specific; do not write the full code."),
        ("api-comparator",
         "Compare options or summarize trade-offs.",
         "compare, versus, vs, trade-off, tradeoff, difference, summarize, pros and cons",
         "Give a compact comparison of pros and cons for each option, then a one-line \
          recommendation with the condition under which it applies."),
    ].into_iter().map(|(name, desc, cues, instr)| Skill {
        name: name.to_string(),
        description: desc.to_string(),
        cues: cues.split(',').map(|c| c.trim().to_lowercase()).collect(),
        instructions: instr.to_string(),
    }).collect()
}

// ─────────────────────────── difficulty router ─────────────────────────────

/// Task difficulty — determines which engine(s) handle a turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Difficulty {
    /// Deterministic tool intent — Needle only, no LLM.
    Simple,
    /// Focused lookup — Needle routes the tool, LLM writes the answer.
    Moderate,
    /// Open-ended reasoning — LLM routes the tool and writes the answer.
    Complex,
}

impl Difficulty {
    pub fn label(self) -> &'static str {
        match self {
            Self::Simple => "SIMPLE",
            Self::Moderate => "MODERATE",
            Self::Complex => "COMPLEX",
        }
    }

    pub fn color_label(self) -> colored::ColoredString {
        match self {
            Self::Simple => self.label().green().bold(),
            Self::Moderate => self.label().yellow().bold(),
            Self::Complex => self.label().red().bold(),
        }
    }
}

/// Classify a task by heuristic keyword matching — no model needed.
pub fn classify(task: &str) -> (Difficulty, &'static str) {
    let q = task.to_lowercase();
    let words = q.split_whitespace().count();

    let is_list = (q.contains("list") || q.contains("show all") || q.contains("what documents"))
        && q.contains("document");
    let is_chunks = q.contains("chunk") || q.contains("sections of");
    if is_list || is_chunks {
        return (Difficulty::Simple, "deterministic tool intent (list/chunks)");
    }

    const COMPLEX: [&str; 14] = [
        "explain", "why", "how ", "design", "refactor", "compare", "implement", "write",
        "generate", "summarize", "architecture", "trade-off", "tradeoff", "walk me through",
    ];
    if COMPLEX.iter().any(|c| q.contains(c)) {
        return (Difficulty::Complex, "reasoning/generation cue");
    }
    if words > 18 || q.matches('?').count() > 1 {
        return (Difficulty::Complex, "long / multi-part request");
    }

    (Difficulty::Moderate, "focused factual lookup")
}

// ───────────────────────── tool-calling turn ────────────────────────────────

/// Run one agent turn with a **printed** Claude-Code-style tool-call trace.
///
/// Returns `(tool_name, answer)`.
async fn run_agent_turn(
    graph: &InMemoryGraph,
    model: &mut dyn TextModel,
    task: &str,
    skill: Option<&Skill>,
    route_with_llm: bool,
    compose_with_llm: bool,
    gen: &GenerationConfig,
) -> model_hub::Result<(Option<String>, String)> {
    let schemas = graphdb_tool_schemas();

    // ── 1. Route: pick a tool. ───────────────────────────────────────────────
    // Inline the reborrow so the &mut borrow of `model` ends here.
    let call = if route_with_llm {
        select_tool(EngineKind::Llm, task, &schemas, Some(&mut *model))?
    } else {
        select_tool(EngineKind::Needle, task, &schemas, None)?
    };

    // ── 2. Execute and print the trace. ─────────────────────────────────────
    let (tool_name, tool_result) = match &call {
        Some(c) => {
            println!(
                "  {} {}  {}({})",
                "●".cyan(),
                "tool_use".cyan().bold(),
                c.name.bold(),
                compact_json(&c.arguments).dimmed(),
            );
            let result = ToolExecutor::execute(graph, c).await?;
            let preview = truncate(&compact_json(&result), 120);
            println!("  {} {}  {}", "└".cyan(), "tool_result".cyan(), preview.dimmed());
            (Some(c.name.clone()), result)
        }
        None => {
            println!("  {} {}  {}", "●".dimmed(), "tool_use".dimmed(), "(no tool selected)".dimmed());
            (None, Value::Null)
        }
    };

    // ── 3. Compose the answer. ───────────────────────────────────────────────
    let answer = if compose_with_llm {
        let prompt = build_prompt(skill, task, &tool_result);
        model.generate(&prompt, gen)?
    } else {
        format_result_deterministic(tool_name.as_deref(), &tool_result)
    };

    Ok((tool_name, answer))
}

fn build_prompt(skill: Option<&Skill>, task: &str, result: &Value) -> String {
    let mut p = String::from("You are a coding assistant answering questions about a codebase.\n");
    if let Some(s) = skill {
        p.push_str("\n# Skill: ");
        p.push_str(&s.name);
        p.push('\n');
        p.push_str(&s.instructions);
        p.push('\n');
    }
    p.push_str("\n# Tool result (JSON)\n");
    p.push_str(&truncate(&compact_json(result), 1200));
    p.push_str("\n\n# User request\n");
    p.push_str(task);
    p.push_str("\n\n# Answer\n");
    p
}

fn format_result_deterministic(tool: Option<&str>, result: &Value) -> String {
    match tool {
        Some("list_documents") => {
            let arr = result.as_array().cloned().unwrap_or_default();
            let mut s = format!("{} document(s):", arr.len());
            for d in &arr {
                let title = d.get("title").and_then(|t| t.as_str()).unwrap_or("?");
                let path  = d.get("source_path").and_then(|t| t.as_str()).unwrap_or("?");
                s.push_str(&format!("\n  - {title} ({path})"));
            }
            s
        }
        Some("get_document_chunks") => {
            let n = result.as_array().map(|a| a.len()).unwrap_or(0);
            format!("{n} chunk(s) retrieved")
        }
        Some("search_knowledge_base") => {
            let n = result.as_array().map(|a| a.len()).unwrap_or(0);
            format!("{n} relevant chunk(s) found")
        }
        _ => "no matching tool result".to_string(),
    }
}

// ────────────────────────────── REPL ───────────────────────────────────────

/// Configuration passed from the `Agent` CLI subcommand.
pub struct AgentConfig {
    /// Optional path to a directory whose `.rs`/`.md` files are ingested.
    pub context_dir: Option<PathBuf>,
    /// Optional GGUF model weights path.
    pub model_path: Option<PathBuf>,
    /// Matching tokenizer.json.
    pub tokenizer_path: Option<PathBuf>,
    /// Compute device.
    pub device: DeviceKind,
}

/// Entry point for `graphdb-cli agent`. Builds the graph/model, then runs the
/// REPL until the user quits.
pub async fn run(cfg: AgentConfig) -> Result<()> {
    // ── Build model. ────────────────────────────────────────────────────────
    let mut model: Box<dyn TextModel> = match (&cfg.model_path, &cfg.tokenizer_path) {
        (Some(m), Some(t)) => {
            let device = resolve_device(cfg.device)?;
            let llm = QuantizedLlama::load(m, t, device)
                .with_context(|| format!("loading GGUF {}", m.display()))?;
            Box::new(llm)
        }
        (Some(_), None) => anyhow::bail!("--model requires --tokenizer"),
        _ => Box::new(StubTextModel::new()),
    };

    // ── Build knowledge graph. ───────────────────────────────────────────────
    let graph = InMemoryGraph::new();
    seed_demo_context(&graph);
    let mut ingested_by_dir = 0usize;
    if let Some(dir) = &cfg.context_dir {
        ingested_by_dir = ingest_dir(&graph, dir);
    }

    // ── Load skills. ────────────────────────────────────────────────────────
    let skills = SkillRegistry::load();

    // ── Print banner. ────────────────────────────────────────────────────────
    print_banner(&graph, &skills, model.name(), ingested_by_dir);

    // ── REPL. ────────────────────────────────────────────────────────────────
    let gen = GenerationConfig::deterministic().with_max_tokens(256);
    let mut rl = DefaultEditor::new().context("failed to initialize readline")?;

    // Try to load persistent history from home dir.
    let history_path = dirs_for_history();
    if let Some(ref hp) = history_path {
        let _ = rl.load_history(hp);
    }

    loop {
        let prompt = format!("{} ", "◆".bright_magenta().bold());
        match rl.readline(&prompt) {
            Ok(raw) => {
                let line = raw.trim().to_string();
                if line.is_empty() { continue; }
                let _ = rl.add_history_entry(&line);

                // Slash commands.
                if line.starts_with('/') {
                    if handle_slash(&line, &graph, &skills) == SlashResult::Quit {
                        break;
                    }
                    continue;
                }

                // Agent turn.
                println!();
                if let Err(e) = agent_turn(&graph, model.as_mut(), &line, &skills, &gen).await {
                    eprintln!("{} {e}", "error:".red().bold());
                }
                println!();
            }
            Err(ReadlineError::Interrupted) => {
                println!("{}", "(^C — type /quit to exit)".dimmed());
            }
            Err(ReadlineError::Eof) => {
                println!("{}", "Bye!".dimmed());
                break;
            }
            Err(e) => {
                eprintln!("{} {e}", "readline error:".red());
                break;
            }
        }
    }

    if let Some(ref hp) = history_path {
        let _ = rl.save_history(hp);
    }
    Ok(())
}

/// Run one REPL turn: classify → pick skill → tool loop → answer.
async fn agent_turn(
    graph: &InMemoryGraph,
    model: &mut dyn TextModel,
    task: &str,
    skills: &SkillRegistry,
    gen: &GenerationConfig,
) -> model_hub::Result<()> {
    let (difficulty, reason) = classify(task);
    let skill = skills.select(task);

    let route_llm   = matches!(difficulty, Difficulty::Complex);
    let compose_llm = !matches!(difficulty, Difficulty::Simple);

    // ── Print turn header. ──────────────────────────────────────────────────
    println!(
        "  {} {}  {}",
        "→".dimmed(),
        difficulty.color_label(),
        reason.dimmed()
    );
    if let Some(s) = skill {
        println!(
            "  {} skill  {}",
            "✦".magenta(),
            s.name.magenta().bold()
        );
    }
    println!();

    // ── Tool loop + compose. ────────────────────────────────────────────────
    let (tool_name, answer) = run_agent_turn(
        graph, model, task, skill, route_llm, compose_llm, gen,
    ).await?;

    // ── Engine label. ───────────────────────────────────────────────────────
    let tool = tool_name.as_deref().unwrap_or("none");
    match difficulty {
        Difficulty::Simple => println!(
            "  {}  {} → {} route → deterministic",
            "handled by".dimmed(), "needle".green(), tool.green()
        ),
        Difficulty::Moderate => println!(
            "  {}  {} route → {} → {} compose",
            "handled by".dimmed(), "needle".green(), tool.green(), model.name().yellow()
        ),
        Difficulty::Complex => println!(
            "  {}  {} route → {} → {} compose",
            "handled by".dimmed(), model.name().yellow(), tool.yellow(), model.name().yellow()
        ),
    };
    println!();

    // ── Answer. ─────────────────────────────────────────────────────────────
    for line in answer.lines() {
        println!("  {line}");
    }

    Ok(())
}

// ─────────────────────────── slash commands ─────────────────────────────────

#[derive(PartialEq)]
enum SlashResult {
    Continue,
    Quit,
}

fn handle_slash(line: &str, graph: &InMemoryGraph, skills: &SkillRegistry) -> SlashResult {
    let parts: Vec<&str> = line.splitn(4, ' ').collect();
    match parts[0] {
        "/quit" | "/exit" | "/q" => {
            println!("{}", "Bye!".dimmed());
            return SlashResult::Quit;
        }
        "/help" => print_help(),
        "/skills" => {
            println!("{}", "\nLoaded skills:".bold());
            println!("{}\n", skills.catalog());
        }
        "/tools" => {
            println!("{}", "\nAvailable tools:".bold());
            for t in graphdb_tool_schemas() {
                println!("  {} {}", "◈".cyan(), t.name.bold());
                println!("    {}", t.description.dimmed());
                if !t.parameters.is_empty() {
                    let pnames: Vec<_> = t.parameters.iter().map(|p| p.name.as_str()).collect();
                    println!("    params: {}", pnames.join(", ").dimmed());
                }
            }
            println!();
        }
        "/context" => {
            println!(
                "\n  knowledge base: {} documents / {} chunks\n",
                graph.document_count().to_string().bold(),
                graph.chunk_count().to_string().bold()
            );
        }
        "/ingest" => {
            // /ingest <path> [title]
            if parts.len() < 2 {
                println!("{}", "usage: /ingest <path> [title]".yellow());
            } else {
                let path = parts[1];
                let title = if parts.len() >= 3 { parts[2..].join(" ") }
                            else { Path::new(path).file_stem()
                                       .map(|s| s.to_string_lossy().to_string())
                                       .unwrap_or_else(|| path.to_string()) };
                match fs::read_to_string(path) {
                    Ok(content) => {
                        let doc_type = if path.ends_with(".rs") { "code" } else { "markdown" };
                        graph.ingest(path, &title, doc_type, &content);
                        println!(
                            "  {} ingested {} → {} chunks total",
                            "✓".green(),
                            title.bold(),
                            graph.chunk_count().to_string().bold()
                        );
                    }
                    Err(e) => eprintln!("{} {e}", "error reading file:".red()),
                }
            }
        }
        "/clear" => {
            // ANSI clear screen + cursor home.
            print!("\x1B[2J\x1B[H");
        }
        other => {
            println!("{} unknown command {}. Type /help.", "?".yellow(), other.bold());
        }
    }
    SlashResult::Continue
}

fn print_help() {
    println!();
    println!("{}", "Slash commands:".bold());
    let cmds = [
        ("/help",              "show this table"),
        ("/skills",            "list loaded skills (name, description, cues)"),
        ("/tools",             "list registered tools with parameters"),
        ("/context",           "show documents and chunks in the knowledge base"),
        ("/ingest <path> [title]", "read a file and add it to the knowledge base"),
        ("/clear",             "clear the terminal screen"),
        ("/quit  /exit  /q",   "exit the REPL"),
    ];
    for (cmd, desc) in &cmds {
        println!("  {:<28} {}", cmd.cyan().to_string(), desc.dimmed());
    }
    println!();
    println!("{}", "Routing:".bold());
    println!("  {:<12} {}", "SIMPLE".green().bold().to_string(),   "Needle tool only — no LLM");
    println!("  {:<12} {}", "MODERATE".yellow().bold().to_string(),"Needle route + LLM compose + skill");
    println!("  {:<12} {}", "COMPLEX".red().bold().to_string(),    "LLM route + LLM compose + skill");
    println!();
}

// ─────────────────────────── helpers ────────────────────────────────────────

fn print_banner(
    graph: &InMemoryGraph,
    skills: &SkillRegistry,
    model_name: &str,
    dir_ingested: usize,
) {
    let border = "─".repeat(51);
    println!("\n{}", format!("╭{border}╮").bright_magenta());
    println!("{}", "│   graphdb agent  •  Claude-Code-style REPL  🤖   │".bright_magenta());
    println!("{}\n", format!("╰{border}╯").bright_magenta());

    println!("  {:<12} {}", "model".dimmed(), model_name.bold());
    println!("  {:<12} {}", "skills".dimmed(), skills.names().bold());
    println!(
        "  {:<12} search_knowledge_base, list_documents, get_document_chunks",
        "tools".dimmed()
    );
    println!(
        "  {:<12} {} documents / {} chunks{}",
        "context".dimmed(),
        graph.document_count().to_string().bold(),
        graph.chunk_count().to_string().bold(),
        if dir_ingested > 0 {
            format!("  ({dir_ingested} from --context-dir)").dimmed().to_string()
        } else {
            String::new()
        }
    );
    println!("\n  {}\n", "Type /help for commands, /quit to exit.".dimmed());
}

/// Seed a compact codebase description as the default knowledge base.
fn seed_demo_context(g: &InMemoryGraph) {
    g.ingest(
        "/model-hub/src/pipeline/orchestrator.rs", "Orchestrator", "code",
        "The Orchestrator runs a full agentic turn: it selects a tool via the configured engine \
         (LLM or Needle), executes it against a GraphBackend, then has the text model compose a \
         final answer grounded in the tool result. Engine selection is per-capability through \
         PipelineConfig.",
    );
    g.ingest(
        "/model-hub/src/needle/mod.rs", "Needle Engine", "code",
        "Needle is a lightweight, deterministic, fully-offline engine that replaces an LLM for \
         tool selection, structured extraction, and text embeddings. It uses regex and heuristics \
         instead of neural weights, so it runs instantly. NeedleOrchestrator drives a complete \
         tool-calling turn with no LLM.",
    );
    g.ingest(
        "/model-hub/src/models/mod.rs", "Model Traits", "code",
        "Three modality traits decouple pipelines from implementations: TextModel (generate + \
         embed), AudioModel (transcribe), ImageModel (embed_image/embed_text). Each has a real \
         candle-transformers implementation and an offline stub, selected at load time.",
    );
    g.ingest(
        "/model-hub/src/pipeline/rag.rs", "RAG Pipeline", "code",
        "RagPipeline retrieves top-k chunks from a GraphBackend, optionally re-ranks them by \
         embedding similarity, builds a grounded prompt, and generates an answer with a TextModel.",
    );
    g.ingest(
        "/rust/src/jepa.rs", "JEPA World Model", "code",
        "The Graph-JEPA world model has context/target encoders and a predictor trained with the \
         VICReg loss. JEPAGraphRAG combines it with GraphRAG for latent search over communities.",
    );
}

/// Ingest all `.rs` and `.md` files found (non-recursively) under `dir`.
/// Returns the number of files ingested.
fn ingest_dir(graph: &InMemoryGraph, dir: &Path) -> usize {
    let Ok(entries) = fs::read_dir(dir) else {
        eprintln!("{} cannot read {:?}", "warning:".yellow(), dir);
        return 0;
    };
    let mut n = 0usize;
    for entry in entries.flatten() {
        let p = entry.path();
        let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !matches!(ext, "rs" | "md" | "txt") { continue; }
        let Ok(content) = fs::read_to_string(&p) else { continue };
        let title = p.file_stem().map(|s| s.to_string_lossy().to_string())
                     .unwrap_or_else(|| p.to_string_lossy().to_string());
        let doc_type = if ext == "rs" { "code" } else { "markdown" };
        let path_str = p.to_string_lossy().to_string();
        graph.ingest(&path_str, &title, doc_type, &content);
        n += 1;
    }
    n
}

/// Try to find a writable history file.
fn dirs_for_history() -> Option<PathBuf> {
    let base = std::env::var("HOME").ok().map(PathBuf::from)
        .or_else(|| std::env::var("USERPROFILE").ok().map(PathBuf::from))?;
    Some(base.join(".graphdb_agent_history"))
}

// ─────────────────────────── small helpers ──────────────────────────────────

fn compact_json(v: &Value) -> String {
    serde_json::to_string(v).unwrap_or_else(|_| "<json>".to_string())
}

fn truncate(s: &str, n: usize) -> String {
    let t: String = s.chars().take(n).collect();
    if s.chars().count() > n { format!("{t}…") } else { t }
}
