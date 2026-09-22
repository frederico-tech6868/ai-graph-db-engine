//! # Claude-Code-style multi-agent coding assistant
//!
//! A coding-assistant example (in the spirit of "Claude Code") that answers
//! questions **grounded in the graph-db-engine codebase**. It combines three
//! ideas from Claude Code:
//!
//! 1. **A main orchestrator that delegates by task difficulty** across two
//!    engines — a deterministic [`Needle`](model_hub::needle) engine for cheap
//!    tasks and a local LLM ([`TextModel`]) for hard ones.
//! 2. **Tool calling** — an explicit, visible tool-use loop. The agent picks a
//!    tool from a registry ([`graphdb_tool_schemas`]), executes it against the
//!    graph, and shows the `tool_use` / `tool_result` trace, just like a coding
//!    agent narrating its actions.
//! 3. **Skills** — reusable instruction "folders" on disk
//!    (`examples/skills/<name>/SKILL.md`). Each skill has trigger cues and a set
//!    of instructions that are injected into the LLM prompt when the skill
//!    matches the task — the same pattern as Claude Code's Agent Skills.
//!
//! ## The difficulty router (the "main orchestrator")
//!
//! Every task is classified into a [`Difficulty`] by
//! [`DifficultyRouter::classify`]. The router then delegates:
//!
//! | Difficulty | Route tool with | Compose answer with | Skill injected |
//! |------------|-----------------|---------------------|----------------|
//! | `Simple`   | Needle          | (deterministic — no LLM) | no        |
//! | `Moderate` | Needle          | local LLM               | yes        |
//! | `Complex`  | local LLM       | local LLM               | yes        |
//!
//! The cheap engine handles what it can; only genuinely hard tasks pay for LLM
//! inference. Every turn prints the chosen difficulty, the reason, the selected
//! skill, and the tool-call trace.
//!
//! ## Selecting a local model
//! By default the LLM is the offline [`StubTextModel`] (no weights, no network).
//! Point it at a local GGUF file to run a real quantized Llama.
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
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

use model_hub::models::text::{QuantizedLlama, StubTextModel};
use model_hub::pipeline::tools::select_tool;
use model_hub::{
    graphdb_tool_schemas, resolve_device, DeviceKind, EngineKind, GenerationConfig, InMemoryGraph,
    TextModel, ToolExecutor,
};

// ───────────────────────────── skills ──────────────────────────────────────

/// A reusable instruction bundle, loaded from `examples/skills/<name>/SKILL.md`.
///
/// This mirrors Claude Code's "Agent Skills": a named folder whose `SKILL.md`
/// carries a description, trigger cues, and a body of instructions. When a
/// skill's cues match the task, its instructions are injected into the LLM
/// prompt to steer the answer.
#[derive(Debug, Clone)]
struct Skill {
    /// Skill identifier (folder name / `name:` field).
    name: String,
    /// One-line summary of what the skill is for.
    description: String,
    /// Lowercase trigger phrases; a match makes the skill eligible.
    cues: Vec<String>,
    /// The instruction body injected into the prompt.
    instructions: String,
}

impl Skill {
    /// Parse a `SKILL.md` file with a simple `---` frontmatter block
    /// (`name`, `description`, `cues`) followed by the instruction body.
    fn parse(md: &str) -> Option<Skill> {
        let rest = md.strip_prefix("---")?;
        let end = rest.find("\n---")?;
        let (front, body) = rest.split_at(end);
        let body = body.trim_start_matches("\n---").trim().to_string();

        let (mut name, mut description, mut cues) = (String::new(), String::new(), Vec::new());
        for line in front.lines() {
            let Some((key, val)) = line.split_once(':') else {
                continue;
            };
            let val = val.trim();
            match key.trim() {
                "name" => name = val.to_string(),
                "description" => description = val.to_string(),
                "cues" => {
                    cues = val
                        .split(',')
                        .map(|c| c.trim().to_lowercase())
                        .filter(|c| !c.is_empty())
                        .collect()
                }
                _ => {}
            }
        }
        if name.is_empty() || body.is_empty() {
            return None;
        }
        Some(Skill {
            name,
            description,
            cues,
            instructions: body,
        })
    }

    /// How strongly this skill matches `task` (number of cue hits).
    fn score(&self, task_lower: &str) -> usize {
        self.cues.iter().filter(|c| task_lower.contains(*c)).count()
    }
}

/// The set of skills discovered on disk, with fallbacks baked in.
struct SkillRegistry {
    skills: Vec<Skill>,
}

impl SkillRegistry {
    /// Load every `skills/*/SKILL.md` under the example directory. Falls back to
    /// a built-in copy if the folder is missing (e.g. running from a package).
    fn load() -> Self {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("examples")
            .join("skills");
        let mut skills = Self::load_dir(&dir);
        if skills.is_empty() {
            skills = builtin_skills();
        }
        skills.sort_by(|a, b| a.name.cmp(&b.name));
        Self { skills }
    }

    fn load_dir(dir: &Path) -> Vec<Skill> {
        let mut out = Vec::new();
        let Ok(entries) = fs::read_dir(dir) else {
            return out;
        };
        for entry in entries.flatten() {
            let skill_md = entry.path().join("SKILL.md");
            if let Ok(text) = fs::read_to_string(&skill_md) {
                if let Some(skill) = Skill::parse(&text) {
                    out.push(skill);
                }
            }
        }
        out
    }

    /// Pick the best-matching skill for a task, if any cue matches.
    fn select(&self, task: &str) -> Option<&Skill> {
        let q = task.to_lowercase();
        self.skills
            .iter()
            .map(|s| (s, s.score(&q)))
            .filter(|(_, n)| *n > 0)
            .max_by_key(|(_, n)| *n)
            .map(|(s, _)| s)
    }

    /// A multi-line catalog of loaded skills (name — description), for display.
    fn catalog(&self) -> String {
        self.skills
            .iter()
            .map(|s| format!("  - {}: {}", s.name, s.description))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Built-in fallback skills (used when the on-disk folder is unavailable).
fn builtin_skills() -> Vec<Skill> {
    [
        (
            "codebase-navigator",
            "Locate files, list documents, and fetch chunks.",
            "list, show, find, where, which file, locate, chunk, sections of, what documents",
            "Use the tools to locate documents and chunks; never guess paths. Return exact \
             file paths and be terse.",
        ),
        (
            "code-explainer",
            "Explain how a piece of the codebase works.",
            "explain, how, why, what does, walk me through, describe, understand",
            "Ground every claim in the retrieved chunks and reference the file paths. Lead with \
             a one-sentence summary, then 2-4 supporting points.",
        ),
        (
            "refactor-planner",
            "Plan an implementation or refactor.",
            "refactor, implement, design, add, change, migrate, rewrite, introduce, build",
            "Respond with a numbered plan; name the affected files per step and end with a short \
             Risks / trade-offs note.",
        ),
        (
            "api-comparator",
            "Compare options or summarize trade-offs.",
            "compare, versus, vs, trade-off, tradeoff, difference, summarize, pros and cons",
            "Give a compact comparison of pros and cons for each option, then a one-line \
             recommendation.",
        ),
    ]
    .into_iter()
    .map(|(name, description, cues, instructions)| Skill {
        name: name.to_string(),
        description: description.to_string(),
        cues: cues.split(',').map(|c| c.trim().to_lowercase()).collect(),
        instructions: instructions.to_string(),
    })
    .collect()
}

// ─────────────────────────── difficulty router ─────────────────────────────

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

/// The main orchestrator: classifies a task and delegates to the right engine,
/// selecting a skill and running an explicit tool-calling loop.
struct DifficultyRouter {
    /// Generation parameters passed to the LLM when it is used.
    gen_config: GenerationConfig,
    /// Skills available for injection.
    skills: SkillRegistry,
}

/// The outcome of handling one task, for display.
struct TurnReport {
    difficulty: Difficulty,
    reason: &'static str,
    skill: Option<String>,
    engine_desc: String,
    answer: String,
}

impl DifficultyRouter {
    fn new(skills: SkillRegistry) -> Self {
        Self {
            gen_config: GenerationConfig::deterministic().with_max_tokens(160),
            skills,
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
            "explain", "why", "how ", "design", "refactor", "compare", "implement", "write",
            "generate", "summarize", "architecture", "trade-off", "tradeoff", "walk me through",
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

    /// Handle one task end to end: classify, pick a skill, and run the tool loop.
    async fn handle(
        &self,
        graph: &InMemoryGraph,
        model: &mut dyn TextModel,
        task: &str,
    ) -> model_hub::Result<TurnReport> {
        let (difficulty, reason) = self.classify(task);
        let skill = self.skills.select(task);

        // Route tool with the cheap engine unless the task is complex.
        let route_with_llm = matches!(difficulty, Difficulty::Complex);
        // Only Simple tasks skip the LLM entirely.
        let compose_with_llm = !matches!(difficulty, Difficulty::Simple);

        let (tool_name, answer) = run_agent(
            graph,
            model,
            task,
            skill,
            route_with_llm,
            compose_with_llm,
            &self.gen_config,
        )
        .await?;

        let engine_desc = match difficulty {
            Difficulty::Simple => format!(
                "needle route: {} → deterministic format (no LLM)",
                tool_name.as_deref().unwrap_or("none")
            ),
            Difficulty::Moderate => format!(
                "needle route: {} → {} compose",
                tool_name.as_deref().unwrap_or("none"),
                model.name()
            ),
            Difficulty::Complex => format!(
                "{} route: {} + compose",
                model.name(),
                tool_name.as_deref().unwrap_or("none")
            ),
        };

        Ok(TurnReport {
            difficulty,
            reason,
            skill: skill.map(|s| s.name.clone()),
            engine_desc,
            answer,
        })
    }
}

// ─────────────────────── explicit tool-calling loop ────────────────────────

/// Run one Claude-Code-style agent turn with a **visible tool-call trace**:
///
/// 1. **route** — pick a tool from [`graphdb_tool_schemas`] using either the
///    Needle heuristics or the LLM ([`select_tool`]).
/// 2. **tool_use / tool_result** — execute the tool against the graph
///    ([`ToolExecutor`]) and print the call + a preview of the result.
/// 3. **compose** — either format the tool result deterministically (no LLM)
///    or have the LLM write the final answer, with the selected skill's
///    instructions injected into the prompt.
///
/// Returns the tool name that ran (if any) and the final answer text.
async fn run_agent(
    graph: &InMemoryGraph,
    model: &mut dyn TextModel,
    task: &str,
    skill: Option<&Skill>,
    route_with_llm: bool,
    compose_with_llm: bool,
    gen_config: &GenerationConfig,
) -> model_hub::Result<(Option<String>, String)> {
    let schemas = graphdb_tool_schemas();

    // 1. Route to a tool. Inline the reborrow so the mutable borrow of `model`
    //    ends with the call and `model` is free again for composition below.
    let call = if route_with_llm {
        select_tool(EngineKind::Llm, task, &schemas, Some(&mut *model))?
    } else {
        select_tool(EngineKind::Needle, task, &schemas, None)?
    };

    // 2. Execute the tool, narrating the trace.
    let (tool_name, tool_result) = match &call {
        Some(c) => {
            println!("    ● tool_use    {}({})", c.name, compact_json(&c.arguments));
            let result = ToolExecutor::execute(graph, c).await?;
            println!(
                "    └ tool_result {}",
                truncate(&compact_json(&result), 100)
            );
            (Some(c.name.clone()), result)
        }
        None => {
            println!("    ● tool_use    (no tool selected)");
            (None, Value::Null)
        }
    };

    // 3. Compose the answer.
    let answer = if compose_with_llm {
        let prompt = build_prompt(skill, task, &tool_result);
        model.generate(&prompt, gen_config)?
    } else {
        format_tool_result(tool_name.as_deref(), &tool_result)
    };

    Ok((tool_name, answer))
}

/// Assemble the LLM prompt, injecting the selected skill's instructions.
fn build_prompt(skill: Option<&Skill>, task: &str, tool_result: &Value) -> String {
    let mut p = String::new();
    p.push_str("You are a coding assistant answering questions about a codebase.\n");
    if let Some(s) = skill {
        p.push_str("\n# Skill: ");
        p.push_str(&s.name);
        p.push('\n');
        p.push_str(&s.instructions);
        p.push('\n');
    }
    p.push_str("\n# Tool result (JSON)\n");
    p.push_str(&truncate(&compact_json(tool_result), 1200));
    p.push_str("\n\n# User request\n");
    p.push_str(task);
    p.push_str("\n\n# Answer\n");
    p
}

/// Format a tool result deterministically, with no LLM (the Simple tier).
fn format_tool_result(tool: Option<&str>, result: &Value) -> String {
    match tool {
        Some("list_documents") => {
            let Some(arr) = result.as_array() else {
                return compact_json(result);
            };
            let mut s = format!("{} document(s):", arr.len());
            for d in arr {
                let title = d.get("title").and_then(|t| t.as_str()).unwrap_or("?");
                let path = d.get("source_path").and_then(|t| t.as_str()).unwrap_or("?");
                s.push_str(&format!("\n  - {title} ({path})"));
            }
            s
        }
        Some("get_document_chunks") => {
            let n = result.as_array().map(|a| a.len()).unwrap_or(0);
            format!("retrieved {n} chunk(s) from the requested document")
        }
        Some("search_knowledge_base") => {
            let n = result.as_array().map(|a| a.len()).unwrap_or(0);
            format!("found {n} relevant chunk(s)")
        }
        _ => "no tool matched this request".to_string(),
    }
}

// ──────────────────────────── model + context ──────────────────────────────

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
        "code context: {} files / {} chunks ingested",
        graph.document_count(),
        graph.chunk_count()
    );

    let skills = SkillRegistry::load();
    println!("skills loaded:\n{}", skills.catalog());

    let mut model = load_model()?;
    let router = DifficultyRouter::new(skills);
    println!(
        "main orchestrator: difficulty router  |  simple->needle  moderate->needle+llm  complex->llm"
    );
    println!("tools available: search_knowledge_base, list_documents, get_document_chunks\n");

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
        println!("────────────────────────────────────────────────────────");
        println!("task {}: {task}", i + 1);
        let report = router.handle(&graph, model.as_mut(), task).await?;
        println!(
            "  routed as : {} ({})",
            report.difficulty.label(),
            report.reason
        );
        println!("  skill     : {}", report.skill.as_deref().unwrap_or("(none)"));
        println!("  handled by: {}", report.engine_desc);
        println!("  answer    : {}", truncate(&report.answer, 320));
        println!();
    }

    println!("Tip: set GRAPHDB_MODEL and GRAPHDB_TOKENIZER to run a real local GGUF model.");
    Ok(())
}

// ─────────────────────────────── helpers ───────────────────────────────────

/// Compact single-line JSON for tracing.
fn compact_json(v: &Value) -> String {
    serde_json::to_string(v).unwrap_or_else(|_| "<json>".to_string())
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
