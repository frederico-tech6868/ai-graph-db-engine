//! Image + text embedding models: CLIP via candle-transformers, and an
//! offline stub.

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip::{ClipConfig, ClipModel};
use tokenizers::Tokenizer;

use super::ImageModel;
use crate::error::{ModelHubError, Result};

/// CLIP image size (ViT-B/32).
const IMAGE_SIZE: usize = 224;

/// A CLIP model (ViT-B/32) loaded from safetensors via candle-transformers.
pub struct ClipImageModel {
    model: ClipModel,
    tokenizer: Tokenizer,
    device: Device,
    name: String,
}

impl ClipImageModel {
    /// Load CLIP weights (safetensors) and a tokenizer.
    pub fn load(
        weights_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        device: Device,
    ) -> Result<Self> {
        let config = ClipConfig::vit_base_patch32();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path.as_ref()], DType::F32, &device)
                .map_err(|e| ModelHubError::ModelLoad(format!("clip weights: {e}")))?
        };
        let model = ClipModel::new(vb, &config)
            .map_err(|e| ModelHubError::ModelLoad(format!("clip model: {e}")))?;
        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|e| ModelHubError::Tokenizer(e.to_string()))?;
        Ok(Self {
            model,
            tokenizer,
            device,
            name: "clip-vit-base-patch32".to_string(),
        })
    }

    /// Build a normalized CLIP pixel tensor `[1, 3, 224, 224]` from raw RGB.
    fn pixel_tensor(&self, rgb: &[u8], width: usize, height: usize) -> Result<Tensor> {
        if width == 0 || height == 0 || rgb.len() < width * height * 3 {
            return Err(ModelHubError::Inference("invalid image dimensions".into()));
        }
        // Nearest-neighbour resize into 224x224x3, in CHW float order.
        let mut data = vec![0f32; 3 * IMAGE_SIZE * IMAGE_SIZE];
        // CLIP normalization constants.
        let mean = [0.48145466f32, 0.4578275, 0.40821073];
        let std = [0.26862954f32, 0.26130258, 0.27577711];
        for y in 0..IMAGE_SIZE {
            let sy = y * height / IMAGE_SIZE;
            for x in 0..IMAGE_SIZE {
                let sx = x * width / IMAGE_SIZE;
                let src = (sy * width + sx) * 3;
                for c in 0..3 {
                    let val = rgb[src + c] as f32 / 255.0;
                    let norm = (val - mean[c]) / std[c];
                    data[c * IMAGE_SIZE * IMAGE_SIZE + y * IMAGE_SIZE + x] = norm;
                }
            }
        }
        let t = Tensor::from_vec(data, (1, 3, IMAGE_SIZE, IMAGE_SIZE), &self.device)?;
        Ok(t)
    }
}

fn to_vec(t: &Tensor) -> Result<Vec<f32>> {
    let t = t.flatten_all()?.to_dtype(DType::F32)?;
    Ok(t.to_vec1::<f32>()?)
}

fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

impl ImageModel for ClipImageModel {
    fn embed_image(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<Vec<f32>> {
        let pixels = self.pixel_tensor(rgb, width, height)?;
        let features = self
            .model
            .get_image_features(&pixels)
            .map_err(|e| ModelHubError::Inference(format!("clip image: {e}")))?;
        let mut v = to_vec(&features)?;
        l2_normalize(&mut v);
        Ok(v)
    }

    fn embed_text(&mut self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| ModelHubError::Tokenizer(e.to_string()))?;
        let ids: Vec<u32> = encoding.get_ids().to_vec();
        let input = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let features = self
            .model
            .get_text_features(&input)
            .map_err(|e| ModelHubError::Inference(format!("clip text: {e}")))?;
        let mut v = to_vec(&features)?;
        l2_normalize(&mut v);
        Ok(v)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Offline stub CLIP model: deterministic hashed embeddings in a shared space.
pub struct StubImageModel {
    name: String,
}

impl Default for StubImageModel {
    fn default() -> Self {
        Self {
            name: "stub-image".to_string(),
        }
    }
}

impl StubImageModel {
    /// Create a new stub image model.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Embedding dimension for the stub CLIP space.
const STUB_DIM: usize = 512;

impl ImageModel for StubImageModel {
    fn embed_image(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<Vec<f32>> {
        let mut v = vec![0f32; STUB_DIM];
        // Coarse color histogram -> embedding.
        let px = (width * height).max(1);
        for i in 0..px {
            let base = (i * 3) % rgb.len().max(3);
            if base + 2 < rgb.len() {
                let r = rgb[base] as usize;
                let g = rgb[base + 1] as usize;
                let b = rgb[base + 2] as usize;
                v[r % STUB_DIM] += 1.0;
                v[(g + 128) % STUB_DIM] += 1.0;
                v[(b + 256) % STUB_DIM] += 1.0;
            }
        }
        l2_normalize(&mut v);
        Ok(v)
    }

    fn embed_text(&mut self, text: &str) -> Result<Vec<f32>> {
        let mut v = vec![0f32; STUB_DIM];
        for w in text.split_whitespace() {
            let mut h: usize = 5381;
            for b in w.bytes() {
                h = h.wrapping_mul(33).wrapping_add(b as usize);
            }
            v[h % STUB_DIM] += 1.0;
        }
        l2_normalize(&mut v);
        Ok(v)
    }

    fn name(&self) -> &str {
        &self.name
    }
}
