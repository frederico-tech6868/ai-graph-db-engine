//! # Transcription example (speech-to-text)
//!
//! Demonstrates the [`AudioModel`] trait using the offline [`StubAudioModel`].
//! The stub needs no weights, so this runs anywhere; the code path is identical
//! for the real Whisper model.
//!
//! What it demonstrates:
//! * synthesizing a mono 16 kHz PCM signal,
//! * `transcribe` (audio -> text),
//! * `embed` (audio -> fixed-length vector).
//!
//! Run it:
//! ```bash
//! cargo run -p model-hub --example transcription
//! ```
//!
//! ## Using the real Whisper model
//! Load real weights and the code below works unchanged:
//! ```ignore
//! use model_hub::models::audio::WhisperModel;
//! use model_hub::models::resolve_device;
//! use model_hub::DeviceKind;
//!
//! let device = resolve_device(DeviceKind::Cpu)?;
//! let mut model = WhisperModel::load(
//!     "weights.safetensors",
//!     "config.json",
//!     "tokenizer.json",
//!     "melfilters.bytes",
//!     device,
//! )?;
//! let text = model.transcribe(&samples)?;
//! ```

use model_hub::models::audio::StubAudioModel;
use model_hub::models::AudioModel;
use model_hub::models::config::DeviceKind;

/// Whisper operates on 16 kHz mono audio.
const SAMPLE_RATE: usize = 16_000;

fn main() -> model_hub::Result<()> {
    // Note: DeviceKind is shown here for parity with the real model path.
    let _device = DeviceKind::Cpu;

    // 1. Synthesize ~2 seconds of a 220 Hz tone as mono 16 kHz PCM.
    let seconds = 2.0f32;
    let n = (SAMPLE_RATE as f32 * seconds) as usize;
    let freq = 220.0f32;
    let samples: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            0.6 * (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect();

    // 2. Load the (offline) audio model.
    let mut model = StubAudioModel::new();
    println!("model: {}", model.name());
    println!("audio: {} samples ({:.1}s @ {} Hz)\n", samples.len(), seconds, SAMPLE_RATE);

    // 3. Transcribe.
    let text = model.transcribe(&samples)?;
    println!("transcription:\n  {text}\n");

    // 4. Embed (useful for audio similarity / retrieval).
    let emb = model.embed(&samples)?;
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("embedding: dim={}, L2 norm={:.4}", emb.len(), norm);
    println!("first 8 dims: {:?}", &emb[..emb.len().min(8)]);

    Ok(())
}
