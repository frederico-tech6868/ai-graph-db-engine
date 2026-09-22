//! Lightweight settings reader for `graphdb-tui`.
//!
//! Reads the same `graphdb-agent.settings.json` discovery order used by
//! `graphdb-cli agent`:
//!
//! 1. `$GRAPHDB_AGENT_SETTINGS` environment variable
//! 2. `./graphdb-agent.settings.json` (current directory)
//! 3. `~/.config/graphdb/agent.settings.json` (user config)
//!
//! Only the `model` and `generation` sections are consumed. All other keys in
//! the JSON (skills_dir, context_dir, history_file, …) are silently ignored,
//! so `graphdb-tui` and `graphdb-cli agent` can share the same settings file
//! without any conflicts.

use std::path::PathBuf;

use serde::Deserialize;

/// Top-level settings schema — the `model` and `generation` sub-objects only.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct TuiSettings {
    /// Text-model configuration (GGUF path, tokenizer, device).
    pub model: ModelSettings,
    /// Text-generation parameters.
    pub generation: GenerationSettings,
}

/// Model-selection settings.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct ModelSettings {
    /// Path to a GGUF text-model file. `None` → offline stub model.
    pub path: Option<String>,
    /// Path to the matching `tokenizer.json`.
    pub tokenizer: Option<String>,
    /// Compute device: `"cpu"`, `"cuda"`, or `"metal"`.
    pub device: Option<String>,
}

/// Text-generation parameters.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct GenerationSettings {
    pub max_tokens: usize,
    pub temperature: f64,
}

impl Default for GenerationSettings {
    fn default() -> Self {
        Self { max_tokens: 256, temperature: 0.0 }
    }
}

impl TuiSettings {
    /// Load settings using the standard discovery order.
    ///
    /// Returns `(settings, human-readable source label)`. On any parse/IO
    /// error the built-in defaults are returned with a descriptive label so
    /// the TUI always starts successfully.
    pub fn load() -> (Self, String) {
        for (path, label) in Self::candidates() {
            if !path.is_file() {
                continue;
            }
            let text = match std::fs::read_to_string(&path) {
                Ok(t) => t,
                Err(_) => continue,
            };
            // No `deny_unknown_fields` → CLI-only keys are silently dropped.
            match serde_json::from_str::<Self>(&text) {
                Ok(s) => return (s, label),
                Err(e) => {
                    eprintln!(
                        "warning: graphdb-tui could not parse {}: {}; using defaults",
                        label, e
                    );
                    continue;
                }
            }
        }
        (Self::default(), "built-in defaults".to_string())
    }

    fn candidates() -> Vec<(PathBuf, String)> {
        let mut v = Vec::new();
        if let Ok(env) = std::env::var("GRAPHDB_AGENT_SETTINGS") {
            if !env.is_empty() {
                v.push((PathBuf::from(&env), env));
            }
        }
        let cwd = PathBuf::from("graphdb-agent.settings.json");
        v.push((cwd.clone(), cwd.display().to_string()));
        if let Some(home) = home_dir() {
            let p = home.join(".config").join("graphdb").join("agent.settings.json");
            v.push((p.clone(), p.display().to_string()));
        }
        v
    }

    /// Model path with `~` expanded, if configured.
    pub fn model_path(&self) -> Option<PathBuf> {
        self.model.path.as_deref().map(expand_tilde)
    }

    /// Tokenizer path with `~` expanded, if configured.
    pub fn tokenizer_path(&self) -> Option<PathBuf> {
        self.model.tokenizer.as_deref().map(expand_tilde)
    }

    /// Compute device from the settings string. Defaults to `Cpu`.
    pub fn device_kind(&self) -> model_hub::DeviceKind {
        match self
            .model
            .device
            .as_deref()
            .unwrap_or("cpu")
            .trim()
            .to_lowercase()
            .as_str()
        {
            "cuda" | "gpu" => model_hub::DeviceKind::Cuda,
            "metal" => model_hub::DeviceKind::Metal,
            _ => model_hub::DeviceKind::Cpu,
        }
    }
}

fn expand_tilde(s: &str) -> PathBuf {
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
