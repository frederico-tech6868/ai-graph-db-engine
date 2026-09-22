//! Vision capabilities: **text recognition (OCR)**, **image generation**, and
//! **image segmentation**.
//!
//! Each capability is expressed as a trait so a real, candle-transformers-backed
//! model and a lightweight **offline stub** are interchangeable. The stubs are
//! fully deterministic and need no weights, so every example in this crate runs
//! without a network connection. To go from a demo to production, implement the
//! trait on top of a real model:
//!
//! | Trait               | Stub                    | Suggested real model (candle)                 |
//! |---------------------|-------------------------|-----------------------------------------------|
//! | [`TextRecognizer`]  | [`StubTextRecognizer`]  | TrOCR (`candle_transformers::models::trocr`)  |
//! | [`ImageGenerator`]  | [`StubImageGenerator`]  | Stable Diffusion (`...::models::stable_diffusion`) |
//! | [`ImageSegmenter`]  | [`StubImageSegmenter`]  | Segment Anything (`...::models::segment_anything`) |
//!
//! All pixel buffers are raw interleaved RGB (`height * width * 3` bytes, row
//! major), matching [`crate::models::ImageModel::embed_image`].

use crate::error::{ModelHubError, Result};

// ---------------------------------------------------------------------------
// Shared geometry
// ---------------------------------------------------------------------------

/// An axis-aligned bounding box in pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BBox {
    /// Left edge (pixels from the left).
    pub x: usize,
    /// Top edge (pixels from the top).
    pub y: usize,
    /// Box width in pixels.
    pub width: usize,
    /// Box height in pixels.
    pub height: usize,
}

/// Compute the luminance (0..=255) of an RGB triple (Rec. 601 weights).
fn luma(r: u8, g: u8, b: u8) -> u8 {
    ((0.299 * r as f32) + (0.587 * g as f32) + (0.114 * b as f32)).round() as u8
}

/// Validate that `rgb` holds at least `width * height * 3` bytes.
fn check_dims(rgb: &[u8], width: usize, height: usize) -> Result<()> {
    if width == 0 || height == 0 || rgb.len() < width * height * 3 {
        return Err(ModelHubError::Inference(format!(
            "invalid image: {width}x{height} needs {} bytes, got {}",
            width * height * 3,
            rgb.len()
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Text recognition (OCR)
// ---------------------------------------------------------------------------

/// A single recognized line of text with its location and confidence.
#[derive(Debug, Clone)]
pub struct OcrLine {
    /// The recognized text of this line.
    pub text: String,
    /// Model confidence in `[0.0, 1.0]`.
    pub confidence: f32,
    /// Bounding box of the line within the source image.
    pub bbox: BBox,
}

/// Optical Character Recognition: extract text lines from an image.
pub trait TextRecognizer {
    /// Recognize text in a raw RGB image (`height * width * 3` bytes).
    ///
    /// Returns one [`OcrLine`] per detected text line, in top-to-bottom order.
    fn recognize(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<Vec<OcrLine>>;

    /// A short human-readable identifier for the recognizer.
    fn name(&self) -> &str;
}

/// Offline stub OCR engine.
///
/// It performs a genuinely useful piece of work — **text-line detection** — by
/// scanning horizontal bands of dark pixels (ink) on a lighter background, then
/// emits a deterministic placeholder string per detected band. Swap in TrOCR to
/// turn the detected bands into real transcriptions.
pub struct StubTextRecognizer {
    name: String,
    /// Luminance below this threshold counts as "ink".
    ink_threshold: u8,
}

impl Default for StubTextRecognizer {
    fn default() -> Self {
        Self {
            name: "stub-ocr".to_string(),
            ink_threshold: 96,
        }
    }
}

impl StubTextRecognizer {
    /// Create a new stub OCR engine with the default ink threshold.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the luminance threshold (0..=255) below which a pixel is "ink".
    pub fn with_ink_threshold(mut self, threshold: u8) -> Self {
        self.ink_threshold = threshold;
        self
    }
}

impl TextRecognizer for StubTextRecognizer {
    fn recognize(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<Vec<OcrLine>> {
        check_dims(rgb, width, height)?;

        // Per-row ink counts.
        let mut ink_per_row = vec![0usize; height];
        for y in 0..height {
            let mut count = 0usize;
            for x in 0..width {
                let i = (y * width + x) * 3;
                if luma(rgb[i], rgb[i + 1], rgb[i + 2]) < self.ink_threshold {
                    count += 1;
                }
            }
            ink_per_row[y] = count;
        }

        // A row is "text" if at least 2% of its pixels are ink.
        let min_ink = (width as f32 * 0.02).ceil() as usize;
        let mut lines = Vec::new();
        let mut band_start: Option<usize> = None;
        for y in 0..height {
            let is_text = ink_per_row[y] >= min_ink;
            match (band_start, is_text) {
                (None, true) => band_start = Some(y),
                (Some(start), false) => {
                    lines.push(self.make_line(start, y, width, &ink_per_row));
                    band_start = None;
                }
                _ => {}
            }
        }
        if let Some(start) = band_start {
            lines.push(self.make_line(start, height, width, &ink_per_row));
        }
        Ok(lines)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl StubTextRecognizer {
    /// Build a deterministic [`OcrLine`] for a detected text band `[start, end)`.
    fn make_line(&self, start: usize, end: usize, width: usize, ink_per_row: &[usize]) -> OcrLine {
        let h = end - start;
        let total_ink: usize = ink_per_row[start..end].iter().sum();
        // Rough "characters" estimate from ink density and band height.
        let approx_chars = ((total_ink as f32) / (h.max(1) as f32) / 6.0).round() as usize;
        let confidence = ((total_ink as f32) / (width as f32 * h as f32)).clamp(0.05, 0.99);
        OcrLine {
            text: format!("[stub-ocr] line of ~{} characters", approx_chars.max(1)),
            confidence,
            bbox: BBox {
                x: 0,
                y: start,
                width,
                height: h,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Image generation
// ---------------------------------------------------------------------------

/// Parameters controlling image generation.
#[derive(Debug, Clone)]
pub struct ImageGenConfig {
    /// Output width in pixels.
    pub width: usize,
    /// Output height in pixels.
    pub height: usize,
    /// RNG seed for reproducible output.
    pub seed: u64,
    /// Number of refinement steps (ignored by the stub, used by real models).
    pub steps: usize,
}

impl Default for ImageGenConfig {
    fn default() -> Self {
        Self {
            width: 256,
            height: 256,
            seed: 42,
            steps: 20,
        }
    }
}

/// A generated raster image (raw interleaved RGB).
#[derive(Debug, Clone)]
pub struct GeneratedImage {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub rgb: Vec<u8>,
    /// Image height in pixels.
    pub height: usize,
}

impl GeneratedImage {
    /// Encode the image as a binary PPM (P6) byte buffer, easy to write to disk
    /// and viewable by most image tools without any extra dependency.
    pub fn to_ppm(&self) -> Vec<u8> {
        let header = format!("P6\n{} {}\n255\n", self.width, self.height);
        let mut out = Vec::with_capacity(header.len() + self.rgb.len());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&self.rgb);
        out
    }
}

/// Text-to-image generation.
pub trait ImageGenerator {
    /// Generate an image for `prompt` using `config`.
    fn generate(&mut self, prompt: &str, config: &ImageGenConfig) -> Result<GeneratedImage>;

    /// A short human-readable identifier for the generator.
    fn name(&self) -> &str;
}

/// Offline stub image generator.
///
/// Produces a deterministic, prompt-derived procedural image (a smooth plasma /
/// gradient field seeded from the prompt hash and `config.seed`). The same
/// prompt + seed always yields the same picture, which is ideal for tests and
/// demos. Swap in Stable Diffusion for photorealistic output.
pub struct StubImageGenerator {
    name: String,
}

impl Default for StubImageGenerator {
    fn default() -> Self {
        Self {
            name: "stub-imagegen".to_string(),
        }
    }
}

impl StubImageGenerator {
    /// Create a new stub image generator.
    pub fn new() -> Self {
        Self::default()
    }
}

/// FNV-1a hash of a string into a 64-bit seed.
fn hash_prompt(prompt: &str) -> u64 {
    let mut h: u64 = 1469598103934665603;
    for b in prompt.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

impl ImageGenerator for StubImageGenerator {
    fn generate(&mut self, prompt: &str, config: &ImageGenConfig) -> Result<GeneratedImage> {
        if config.width == 0 || config.height == 0 {
            return Err(ModelHubError::Inference("zero image dimensions".into()));
        }
        let seed = hash_prompt(prompt) ^ config.seed.rotate_left(17);
        // Derive a handful of frequency/phase/color parameters from the seed.
        let fx = 1.0 + (seed & 0x7) as f32 * 0.6;
        let fy = 1.0 + ((seed >> 3) & 0x7) as f32 * 0.6;
        let phase = ((seed >> 6) & 0xff) as f32 / 40.0;
        let cr = ((seed >> 8) & 0xff) as f32 / 255.0;
        let cg = ((seed >> 16) & 0xff) as f32 / 255.0;
        let cb = ((seed >> 24) & 0xff) as f32 / 255.0;

        let (w, h) = (config.width, config.height);
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            let ny = y as f32 / h as f32;
            for x in 0..w {
                let nx = x as f32 / w as f32;
                let v = (std::f32::consts::PI * (fx * nx + phase)).sin()
                    * (std::f32::consts::PI * (fy * ny + phase)).cos();
                let t = (v + 1.0) * 0.5; // 0..1
                let i = (y * w + x) * 3;
                rgb[i] = ((cr + t) * 0.5 * 255.0).clamp(0.0, 255.0) as u8;
                rgb[i + 1] = ((cg + (1.0 - t)) * 0.5 * 255.0).clamp(0.0, 255.0) as u8;
                rgb[i + 2] = ((cb + t * (1.0 - nx)) * 0.5 * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
        Ok(GeneratedImage {
            width: w,
            height: h,
            rgb,
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ---------------------------------------------------------------------------
// Image segmentation
// ---------------------------------------------------------------------------

/// A dense segmentation mask: one label per pixel (row-major).
#[derive(Debug, Clone)]
pub struct SegmentationMask {
    /// Mask width in pixels.
    pub width: usize,
    /// Mask height in pixels.
    pub height: usize,
    /// Per-pixel label ids, length `width * height` (row major).
    pub labels: Vec<u32>,
    /// Number of distinct labels (segments) in the mask.
    pub num_labels: u32,
}

impl SegmentationMask {
    /// Count how many pixels belong to `label`.
    pub fn area_of(&self, label: u32) -> usize {
        self.labels.iter().filter(|&&l| l == label).count()
    }
}

/// Semantic / region segmentation of an image.
pub trait ImageSegmenter {
    /// Segment a raw RGB image into per-pixel labels.
    fn segment(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<SegmentationMask>;

    /// A short human-readable identifier for the segmenter.
    fn name(&self) -> &str;
}

/// Offline stub segmenter.
///
/// Partitions the image into `k` regions by quantizing per-pixel luminance into
/// `k` equal bands — a fast, deterministic approximation of thresholding-based
/// segmentation. Swap in Segment Anything (SAM) for prompt-driven instance
/// masks.
pub struct StubImageSegmenter {
    name: String,
    /// Number of luminance bands / segments.
    k: u32,
}

impl Default for StubImageSegmenter {
    fn default() -> Self {
        Self {
            name: "stub-segment".to_string(),
            k: 4,
        }
    }
}

impl StubImageSegmenter {
    /// Create a segmenter producing `k` luminance-band segments (clamped to 1..=256).
    pub fn new(k: u32) -> Self {
        Self {
            name: "stub-segment".to_string(),
            k: k.clamp(1, 256),
        }
    }
}

impl ImageSegmenter for StubImageSegmenter {
    fn segment(&mut self, rgb: &[u8], width: usize, height: usize) -> Result<SegmentationMask> {
        check_dims(rgb, width, height)?;
        let k = self.k.max(1);
        let mut labels = vec![0u32; width * height];
        for (p, label) in labels.iter_mut().enumerate() {
            let i = p * 3;
            let l = luma(rgb[i], rgb[i + 1], rgb[i + 2]) as u32;
            // Map 0..=255 luminance into 0..k-1 bands.
            *label = (l * k / 256).min(k - 1);
        }
        Ok(SegmentationMask {
            width,
            height,
            labels,
            num_labels: k,
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
}
