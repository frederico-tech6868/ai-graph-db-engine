//! Text generation models: quantized Llama (GGUF) via candle-transformers,
//! and an offline stub.

use std::path::Path;

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::{
    quantized_gemma3, quantized_glm4, quantized_llama, quantized_phi, quantized_phi3,
    quantized_qwen2, quantized_qwen3,
};
use candle_transformers::utils::apply_repeat_penalty;
use tokenizers::Tokenizer;

use super::{GenerationConfig, TextModel};
use crate::error::{ModelHubError, Result};

/// Fixed embedding dimension used by the stub and by hashed pooling.
pub const EMBED_DIM: usize = 384;

/// A quantized decoder-only LLM loaded from a GGUF file. Despite the historical
/// name, this dispatches to the correct candle-transformers parser based on the
/// `general.architecture` field stored in the GGUF metadata, so it can load
/// Llama, Qwen2/Qwen3, Gemma, Phi/Phi-3 and GLM-4 checkpoints — not only Llama.
enum LoadedModel {
    Llama(quantized_llama::ModelWeights),
    Qwen2(quantized_qwen2::ModelWeights),
    Qwen3(quantized_qwen3::ModelWeights),
    Gemma3(quantized_gemma3::ModelWeights),
    Phi3(quantized_phi3::ModelWeights),
    Phi(quantized_phi::ModelWeights),
    Glm4(quantized_glm4::ModelWeights),
}

impl LoadedModel {
    /// Run a forward pass. Every wrapped parser exposes the same
    /// `forward(&Tensor, offset) -> Result<Tensor>` shape.
    fn forward(&mut self, input: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
        match self {
            LoadedModel::Llama(m) => m.forward(input, index_pos),
            LoadedModel::Qwen2(m) => m.forward(input, index_pos),
            LoadedModel::Qwen3(m) => m.forward(input, index_pos),
            LoadedModel::Gemma3(m) => m.forward(input, index_pos),
            LoadedModel::Phi3(m) => m.forward(input, index_pos),
            LoadedModel::Phi(m) => m.forward(input, index_pos),
            LoadedModel::Glm4(m) => m.forward(input, index_pos),
        }
    }
}

/// A quantized LLM loaded from a GGUF file, with a Hugging Face tokenizer,
/// running on candle.
pub struct QuantizedLlama {
    model: LoadedModel,
    tokenizer: Tokenizer,
    device: Device,
    name: String,
    architecture: String,
    eos_token: u32,
    /// Maximum context window in tokens read from GGUF metadata, if present.
    ctx_len: Option<usize>,
}

impl QuantizedLlama {
    /// Load a GGUF model file and its tokenizer.
    ///
    /// The correct model parser is selected automatically from the GGUF's
    /// `general.architecture` metadata field. If that field is missing, we fall
    /// back to the Llama parser (the most common layout).
    pub fn load(
        gguf_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        device: Device,
    ) -> Result<Self> {
        let gguf_path = gguf_path.as_ref();
        let mut file = std::fs::File::open(gguf_path)
            .map_err(|e| ModelHubError::ModelLoad(format!("open gguf: {e}")))?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| ModelHubError::ModelLoad(format!("read gguf: {e}")))?;

        // Discover the architecture so we can pick the matching parser rather
        // than blindly using the Llama one (which fails with e.g.
        // "cannot find llama.attention.head_count in metadata" on Qwen/Gemma/Phi GGUFs).
        let architecture = content
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .cloned()
            .unwrap_or_else(|| "llama".to_string());

        // Read context window size BEFORE from_gguf consumes `content`.
        // Stored under "{arch}.context_length" (e.g. "qwen2.context_length").
        let ctx_len = content
            .metadata
            .get(&format!("{architecture}.context_length"))
            .and_then(|v| v.to_u64().ok())
            .map(|n| n as usize);

        let model = match architecture.as_str() {
            "llama" | "mistral" | "mixtral" | "stablelm" | "starcoder2" => {
                quantized_llama::ModelWeights::from_gguf(content, &mut file, &device)
                    .map(LoadedModel::Llama)
            }
            "qwen2" => quantized_qwen2::ModelWeights::from_gguf(content, &mut file, &device)
                .map(LoadedModel::Qwen2),
            "qwen3" => quantized_qwen3::ModelWeights::from_gguf(content, &mut file, &device)
                .map(LoadedModel::Qwen3),
            "gemma" | "gemma2" | "gemma3" => {
                quantized_gemma3::ModelWeights::from_gguf(content, &mut file, &device)
                    .map(LoadedModel::Gemma3)
            }
            "phi3" => {
                quantized_phi3::ModelWeights::from_gguf(false, content, &mut file, &device)
                    .map(LoadedModel::Phi3)
            }
            "phi2" | "phi" => quantized_phi::ModelWeights::from_gguf(content, &mut file, &device)
                .map(LoadedModel::Phi),
            "glm4" | "chatglm" => {
                quantized_glm4::ModelWeights::from_gguf(content, &mut file, &device, DType::F32)
                    .map(LoadedModel::Glm4)
            }
            // Unknown architecture: try the Llama parser as a best effort.
            _ => quantized_llama::ModelWeights::from_gguf(content, &mut file, &device)
                .map(LoadedModel::Llama),
        }
        .map_err(|e| {
            let known = matches!(
                architecture.as_str(),
                "llama"
                    | "mistral"
                    | "mixtral"
                    | "stablelm"
                    | "starcoder2"
                    | "qwen2"
                    | "qwen3"
                    | "gemma"
                    | "gemma2"
                    | "gemma3"
                    | "phi3"
                    | "phi2"
                    | "phi"
                    | "glm4"
                    | "chatglm"
            );
            if known {
                ModelHubError::ModelLoad(format!("from_gguf ({architecture}): {e}"))
            } else {
                ModelHubError::ModelLoad(format!(
                    "unsupported GGUF architecture {architecture:?}; the Llama fallback parser \
                     also failed: {e}. Supported architectures: llama, mistral, qwen2, qwen3, \
                     gemma/gemma2/gemma3, phi, phi3, glm4."
                ))
            }
        })?;

        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| ModelHubError::Tokenizer(e.to_string()))?;

        // Resolve a reasonable EOS token id, covering the common special-token
        // conventions across Llama, Qwen, Gemma and Phi tokenizers.
        let eos_token = tokenizer
            .token_to_id("<|im_end|>") // Qwen / ChatML
            .or_else(|| tokenizer.token_to_id("<end_of_turn>")) // Gemma
            .or_else(|| tokenizer.token_to_id("<|eot_id|>")) // Llama-3
            .or_else(|| tokenizer.token_to_id("</s>")) // Llama-2 / Mistral
            .or_else(|| tokenizer.token_to_id("<|endoftext|>")) // Phi / GPT-style
            .unwrap_or(2);

        let name = gguf_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("llama-gguf")
            .to_string();

        Ok(Self {
            model,
            tokenizer,
            device,
            name,
            architecture,
            eos_token,
            ctx_len,
        })
    }

    /// The GGUF `general.architecture` this model was loaded as
    /// (e.g. `"llama"`, `"qwen2"`, `"gemma3"`).
    pub fn architecture(&self) -> &str {
        &self.architecture
    }
}

impl TextModel for QuantizedLlama {
    fn generate(&mut self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| ModelHubError::Tokenizer(e.to_string()))?;
        let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
        if tokens.is_empty() {
            return Ok(String::new());
        }

        let temperature = if config.temperature <= 0.0 {
            None
        } else {
            Some(config.temperature)
        };
        let mut logits_processor = LogitsProcessor::new(config.seed, temperature, config.top_p);

        let mut generated: Vec<u32> = Vec::new();
        let mut index_pos = 0usize;

        for step in 0..config.max_tokens {
            // On the first step feed the whole prompt; afterwards a single token.
            let (context, ctxt_len) = if step == 0 {
                (tokens.as_slice(), tokens.len())
            } else {
                (&tokens[tokens.len() - 1..], 1)
            };
            let input = Tensor::new(context, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, index_pos)?;
            let logits = logits.squeeze(0)?;
            // Take the logits for the last position.
            let logits = if logits.rank() == 2 {
                logits.get(logits.dim(0)? - 1)?
            } else {
                logits
            };
            let logits = logits.to_dtype(candle_core::DType::F32)?;

            // Apply repeat penalty over the recent context.
            let logits = if config.repeat_penalty == 1.0 {
                logits
            } else {
                let start = generated.len().saturating_sub(config.repeat_last_n);
                apply_repeat_penalty(&logits, config.repeat_penalty, &generated[start..])?
            };

            let next = logits_processor.sample(&logits)?;
            index_pos += ctxt_len;
            if next == self.eos_token {
                break;
            }
            tokens.push(next);
            generated.push(next);
        }

        let text = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| ModelHubError::Tokenizer(e.to_string()))?;
        Ok(text)
    }

    fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        // Mean-pool the token embeddings would require exposing internals; we use
        // a stable hashed bag-of-tokens embedding for a lightweight vector that
        // is consistent for identical inputs.
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| ModelHubError::Tokenizer(e.to_string()))?;
        Ok(hashed_embedding(encoding.get_ids()))
    }

    fn name(&self) -> &str {
        &self.name
    }

    /// Token count using the real tokenizer (not the char/4 approximation).
    fn count_tokens(&self, text: &str) -> usize {
        self.tokenizer
            .encode(text, false)
            .map(|enc| enc.len())
            .unwrap_or_else(|_| (text.len() + 3) / 4)
    }

    /// Context window as reported by the GGUF `{arch}.context_length` metadata.
    fn context_length(&self) -> Option<usize> {
        self.ctx_len
    }
}

/// Deterministic hashed embedding from token ids into a fixed-size vector.
fn hashed_embedding(ids: &[u32]) -> Vec<f32> {
    let mut v = vec![0.0f32; EMBED_DIM];
    for &id in ids {
        // Simple splitmix-style hash for bucket + sign.
        let mut h = id as u64;
        h = (h ^ (h >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        h = (h ^ (h >> 27)).wrapping_mul(0x94d049bb133111eb);
        h ^= h >> 31;
        let bucket = (h as usize) % EMBED_DIM;
        let sign = if (h >> 33) & 1 == 0 { 1.0 } else { -1.0 };
        v[bucket] += sign;
    }
    // L2 normalize.
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    v
}

/// An offline stub text model. It never needs weights and produces
/// deterministic, context-aware placeholder responses. Ideal for tests and for
/// running the full pipeline without downloading models.
pub struct StubTextModel {
    name: String,
}

impl Default for StubTextModel {
    fn default() -> Self {
        Self {
            name: "stub-text".to_string(),
        }
    }
}

impl StubTextModel {
    /// Create a new stub text model.
    pub fn new() -> Self {
        Self::default()
    }
}

impl TextModel for StubTextModel {
    fn generate(&mut self, prompt: &str, config: &GenerationConfig) -> Result<String> {
        // Echo a concise, deterministic answer that reflects the prompt so that
        // downstream pipelines (RAG, extraction) behave sensibly offline.
        let trimmed = prompt.trim();
        let preview: String = trimmed.chars().take(280).collect();
        let out = format!(
            "[stub-text] Based on the prompt, here is a generated response (max_tokens={}). \
             Prompt preview: \"{}\"",
            config.max_tokens, preview
        );
        Ok(out)
    }

    fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        // Hash whitespace tokens into the embedding space.
        let ids: Vec<u32> = text
            .split_whitespace()
            .map(|w| {
                let mut h: u32 = 2166136261;
                for b in w.bytes() {
                    h ^= b as u32;
                    h = h.wrapping_mul(16777619);
                }
                h
            })
            .collect();
        Ok(hashed_embedding(&ids))
    }

    fn name(&self) -> &str {
        &self.name
    }
}
