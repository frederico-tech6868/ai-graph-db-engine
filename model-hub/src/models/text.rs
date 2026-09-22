//! Text generation models: quantized Llama (GGUF) via candle-transformers,
//! and an offline stub.

use std::path::Path;

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama::ModelWeights;
use candle_transformers::utils::apply_repeat_penalty;
use tokenizers::Tokenizer;

use super::{GenerationConfig, TextModel};
use crate::error::{ModelHubError, Result};

/// Fixed embedding dimension used by the stub and by hashed pooling.
pub const EMBED_DIM: usize = 384;

/// A quantized Llama model loaded from a GGUF file, with a Hugging Face
/// tokenizer, running on candle.
pub struct QuantizedLlama {
    model: ModelWeights,
    tokenizer: Tokenizer,
    device: Device,
    name: String,
    eos_token: u32,
}

impl QuantizedLlama {
    /// Load a GGUF model file and its tokenizer.
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
        let model = ModelWeights::from_gguf(content, &mut file, &device)
            .map_err(|e| ModelHubError::ModelLoad(format!("from_gguf: {e}")))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| ModelHubError::Tokenizer(e.to_string()))?;

        // Resolve a reasonable EOS token id.
        let eos_token = tokenizer
            .token_to_id("</s>")
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .or_else(|| tokenizer.token_to_id("<|eot_id|>"))
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
            eos_token,
        })
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
