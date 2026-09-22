//! Audio (speech-to-text) models: Whisper via candle-transformers, and an
//! offline stub.

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as whisper, model::Whisper, Config};
use tokenizers::Tokenizer;

use super::AudioModel;
use crate::error::{ModelHubError, Result};

/// A Whisper model loaded via candle-transformers.
///
/// Loading constructs the real candle `Whisper` model from safetensors. Full
/// transcription requires the Whisper mel-filter bank (`melfilters.bytes`),
/// supplied at load time; the audio encoder produces embeddings from log-mel
/// spectrograms.
pub struct WhisperModel {
    model: Whisper,
    tokenizer: Tokenizer,
    config: Config,
    mel_filters: Vec<f32>,
    device: Device,
    name: String,
}

impl WhisperModel {
    /// Load Whisper weights, config, tokenizer, and mel-filter bank.
    ///
    /// * `weights_path`     - safetensors weights.
    /// * `config_path`      - `config.json` for the model.
    /// * `tokenizer_path`   - `tokenizer.json`.
    /// * `mel_filters_path` - the `melfilters.bytes` (80-bin) asset.
    pub fn load(
        weights_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        mel_filters_path: impl AsRef<Path>,
        device: Device,
    ) -> Result<Self> {
        let config: Config = serde_json::from_slice(
            &std::fs::read(config_path.as_ref())
                .map_err(|e| ModelHubError::ModelLoad(format!("whisper config: {e}")))?,
        )?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path.as_ref()], DType::F32, &device)
                .map_err(|e| ModelHubError::ModelLoad(format!("whisper weights: {e}")))?
        };
        let model = Whisper::load(&vb, config.clone())
            .map_err(|e| ModelHubError::ModelLoad(format!("whisper model: {e}")))?;
        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| ModelHubError::Tokenizer(e.to_string()))?;

        // mel filters stored as little-endian f32.
        let bytes = std::fs::read(mel_filters_path.as_ref())
            .map_err(|e| ModelHubError::ModelLoad(format!("mel filters: {e}")))?;
        let mut mel_filters = vec![0f32; bytes.len() / 4];
        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            mel_filters[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }

        Ok(Self {
            model,
            tokenizer,
            config,
            mel_filters,
            device,
            name: "whisper".to_string(),
        })
    }

    /// Compute the log-mel spectrogram tensor `[1, n_mels, n_frames]`.
    fn mel_tensor(&self, samples: &[f32]) -> Result<Tensor> {
        let mel = whisper::audio::pcm_to_mel(&self.config, samples, &self.mel_filters);
        let n_mels = self.config.num_mel_bins;
        let n_frames = mel.len() / n_mels;
        let t = Tensor::from_vec(mel, (1, n_mels, n_frames), &self.device)?;
        Ok(t)
    }
}

impl AudioModel for WhisperModel {
    fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
        // Greedy decoding over a single 30s window.
        let mel = self.mel_tensor(samples)?;
        let audio_features = self
            .model
            .encoder
            .forward(&mel, true)
            .map_err(|e| ModelHubError::Inference(format!("whisper encoder: {e}")))?;

        let sot = self
            .tokenizer
            .token_to_id("<|startoftranscript|>")
            .unwrap_or(50258);
        let eot = self.tokenizer.token_to_id("<|endoftext|>").unwrap_or(50257);
        let transcribe_tok = self.tokenizer.token_to_id("<|transcribe|>").unwrap_or(50359);
        let no_ts = self
            .tokenizer
            .token_to_id("<|notimestamps|>")
            .unwrap_or(50363);

        let mut tokens: Vec<u32> = vec![sot, transcribe_tok, no_ts];
        let max_steps = self.config.max_target_positions.min(224);
        for _ in 0..max_steps {
            let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            // Re-feed the full sequence each step and flush the KV cache; this is
            // slower than incremental decoding but avoids cache-position bugs.
            let ys = self
                .model
                .decoder
                .forward(&input, &audio_features, true)
                .map_err(|e| ModelHubError::Inference(format!("whisper decoder: {e}")))?;
            let (_b, seq_len, _n) = ys.dims3()?;
            let last = ys.i((.., seq_len - 1..seq_len))?;
            let logits = self.model.decoder.final_linear(&last)?;
            let logits = logits.i((0, 0))?;
            let next = logits.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()?;
            if next == eot {
                break;
            }
            tokens.push(next);
        }

        let text = self
            .tokenizer
            .decode(&tokens[3..], true)
            .map_err(|e| ModelHubError::Tokenizer(e.to_string()))?;
        Ok(text.trim().to_string())
    }

    fn embed(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        let mel = self.mel_tensor(samples)?;
        let features = self
            .model
            .encoder
            .forward(&mel, true)
            .map_err(|e| ModelHubError::Inference(format!("whisper encoder: {e}")))?;
        // Mean-pool over the time dimension.
        let pooled = features.mean(1)?.flatten_all()?.to_dtype(DType::F32)?;
        Ok(pooled.to_vec1::<f32>()?)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Offline stub audio model producing deterministic placeholder text/embeddings.
pub struct StubAudioModel {
    name: String,
}

impl Default for StubAudioModel {
    fn default() -> Self {
        Self {
            name: "stub-audio".to_string(),
        }
    }
}

impl StubAudioModel {
    /// Create a new stub audio model.
    pub fn new() -> Self {
        Self::default()
    }
}

impl AudioModel for StubAudioModel {
    fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
        let secs = samples.len() as f32 / whisper::SAMPLE_RATE as f32;
        Ok(format!(
            "[stub-audio] transcription placeholder for {:.1}s of audio ({} samples)",
            secs,
            samples.len()
        ))
    }

    fn embed(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        let mut v = vec![0f32; 128];
        for (i, s) in samples.iter().enumerate() {
            v[i % 128] += s.abs();
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
        Ok(v)
    }

    fn name(&self) -> &str {
        &self.name
    }
}
