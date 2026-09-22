//! # `settings.json` — declarative configuration for the agent REPL
//!
//! Makes `graphdb-cli agent` customizable **without recompiling**. A JSON file
//! sets the model, device, generation parameters, skills folder, startup
//! context directory, and history file. Command-line flags always **override**
//! whatever the file says, and the file overrides the built-in defaults:
//!
//! ```text
//! CLI flag  >  settings.json  >  built-in default
//! ```
//!
//! ## Discovery order
//!
//! The first file found is used (unless `--settings <path>` is given, which is
//! taken verbatim):
//!
//! 1. `--settings <path>` (explicit)
//! 2. `$GRAPHDB_AGENT_SETTINGS` (environment variable)
//! 3. `./graphdb-agent.settings.json` (current directory)
//! 4. `~/.config/graphdb/agent.settings.json` (user config)
//!
//! If none exist, built-in defaults are used (offline stub model, CPU).
//!
//! ## Example
//!
//! ```json
//! {
//!   "model": {
//!     "path": "~/models/llama-3.2-3b-instruct-q4_k_m.gguf",
//!     "tokenizer": "~/models/llama-3.2-3b-tokenizer.json",
//!     "device": "cpu"
//!   },
//!   "generation": { "max_tokens": 512, "temperature": 0.2 },
//!   "skills_dir": "./my-skills",
//!   "context_dir": "./src",
//!   "history_file": "~/.graphdb_agent_history"
//! }
//! ```

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

/// Top-level `settings.json` schema. Every field is optional; missing fields
/// fall back to [`Default`], so a partial file is always valid.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AgentSettings {
    /// Text-model configuration (GGUF path, tokenizer, device).
    pub model: ModelSettings,
    /// Text-generation parameters.
    pub generation: GenerationSettings,
    /// Directory to load `*/SKILL.md` skills from. `None` → built-in location.
    pub skills_dir: Option<String>,
    /// Directory whose `.rs`/`.md`/`.txt` files are ingested at startup.
    pub context_dir: Option<String>,
    /// Persistent readline history file.
    pub history_file: Option<String>,
}

/// Model selection loaded from `settings.json`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelSettings {
    /// Path to a GGUF text-model file. `None` → offline stub model.
    pub path: Option<String>,
    /// Path to the matching `tokenizer.json` (required when `path` is set).
    pub tokenizer: Option<String>,
    /// Compute device: `"cpu"`, `"cuda"`, or `"metal"`.
    pub device: Option<String>,
}

/// Generation parameters loaded from `settings.json`.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GenerationSettings {
    /// Maximum new tokens per answer.
    pub max_tokens: usize,
    /// Sampling temperature (0.0 = greedy / deterministic).
    pub temperature: f64,
}

impl Default for GenerationSettings {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.0,
        }
    }
}

impl AgentSettings {
    /// Load settings, honouring the discovery order. `explicit` comes from the
    /// `--settings` flag. Returns `(settings, source)` where `source` names the
    /// file used (or `"built-in defaults"`).
    pub fn load(explicit: Option<&Path>) -> Result<(Self, String)> {
        if let Some(p) = explicit {
            let s = Self::from_file(p)
                .with_context(|| format!("loading --settings {}", p.display()))?;
            return Ok((s, p.display().to_string()));
        }
        for candidate in Self::candidate_paths() {
            if candidate.is_file() {
                let s = Self::from_file(&candidate)
                    .with_context(|| format!("loading {}", candidate.display()))?;
                return Ok((s, candidate.display().to_string()));
            }
        }
        Ok((Self::default(), "built-in defaults".to_string()))
    }

    /// Parse a settings file (JSON, comments not supported).
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let s: Self = serde_json::from_str(&text)
            .with_context(|| format!("parsing JSON in {}", path.display()))?;
        Ok(s)
    }

    fn candidate_paths() -> Vec<PathBuf> {
        let mut v = Vec::new();
        if let Ok(env) = std::env::var("GRAPHDB_AGENT_SETTINGS") {
            if !env.is_empty() {
                v.push(PathBuf::from(env));
            }
        }
        v.push(PathBuf::from("graphdb-agent.settings.json"));
        if let Some(home) = home_dir() {
            v.push(home.join(".config").join("graphdb").join("agent.settings.json"));
        }
        v
    }

    /// Model path with `~` expanded, if set.
    pub fn model_path(&self) -> Option<PathBuf> {
        self.model.path.as_deref().map(expand_tilde)
    }

    /// Tokenizer path with `~` expanded, if set.
    pub fn tokenizer_path(&self) -> Option<PathBuf> {
        self.model.tokenizer.as_deref().map(expand_tilde)
    }

    /// Skills dir with `~` expanded, if set.
    pub fn skills_dir_path(&self) -> Option<PathBuf> {
        self.skills_dir.as_deref().map(expand_tilde)
    }

    /// Context dir with `~` expanded, if set.
    pub fn context_dir_path(&self) -> Option<PathBuf> {
        self.context_dir.as_deref().map(expand_tilde)
    }

    /// History file with `~` expanded, if set.
    pub fn history_path(&self) -> Option<PathBuf> {
        self.history_file.as_deref().map(expand_tilde)
    }
}

/// Expand a leading `~` to the user's home directory.
pub fn expand_tilde(s: &str) -> PathBuf {
    if let Some(rest) = s.strip_prefix("~/") {
        if let Some(home) = home_dir() {
            return home.join(rest);
        }
    }
    if s == "~" {
        if let Some(home) = home_dir() {
            return home;
        }
    }
    PathBuf::from(s)
}

fn home_dir() -> Option<PathBuf> {
    std::env::var("HOME")
        .ok()
        .or_else(|| std::env::var("USERPROFILE").ok())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
}
