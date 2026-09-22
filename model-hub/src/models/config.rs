//! Model & generation configuration types.

use serde::{Deserialize, Serialize};

/// Which compute device to run models on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum DeviceKind {
    /// CPU (always available).
    #[default]
    Cpu,
    /// CUDA GPU (NVIDIA) — requires the `cuda` feature.
    Cuda,
    /// Metal GPU (Apple) — requires the `metal` feature.
    Metal,
}

/// Sampling / generation parameters for text models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate.
    pub max_tokens: usize,
    /// Sampling temperature. `0.0` means greedy/argmax.
    pub temperature: f64,
    /// Nucleus sampling probability (top-p). `None` disables it.
    pub top_p: Option<f64>,
    /// Repeat penalty applied over the recent context.
    pub repeat_penalty: f32,
    /// How many trailing tokens the repeat penalty considers.
    pub repeat_last_n: usize,
    /// RNG seed for reproducible sampling.
    pub seed: u64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_p: Some(0.9),
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            seed: 299792458,
        }
    }
}

impl GenerationConfig {
    /// A deterministic (greedy) configuration.
    pub fn deterministic() -> Self {
        Self {
            temperature: 0.0,
            top_p: None,
            ..Default::default()
        }
    }

    /// Builder-style setter for `max_tokens`.
    pub fn with_max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    /// Builder-style setter for `temperature`.
    pub fn with_temperature(mut self, t: f64) -> Self {
        self.temperature = t;
        self
    }
}

/// Where model weights come from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelSource {
    /// Download from the Hugging Face Hub (`repo`, optional `revision`, `file`).
    HuggingFace {
        /// Repository id, e.g. `"TheBloke/Llama-2-7B-Chat-GGUF"`.
        repo: String,
        /// Optional git revision / branch.
        revision: Option<String>,
        /// The weight filename within the repo.
        file: String,
    },
    /// A local file path already on disk.
    LocalPath(String),
}
