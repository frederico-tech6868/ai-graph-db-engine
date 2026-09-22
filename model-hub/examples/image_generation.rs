//! # Image Generation example (text-to-image)
//!
//! Demonstrates the [`ImageGenerator`] trait with the offline
//! [`StubImageGenerator`], which turns a prompt + seed into a deterministic
//! procedural image (a smooth plasma field). The same prompt and seed always
//! produce the same picture.
//!
//! The generated image is written to disk as a binary PPM (P6) file — a format
//! most image viewers and ImageMagick/`ffmpeg` understand, with no extra crates.
//!
//! Run it:
//! ```bash
//! cargo run -p model-hub --example image_generation
//! # -> writes generated.ppm in the current directory
//! ```
//!
//! ## Going to real generation
//! Implement [`ImageGenerator`] on top of Stable Diffusion
//! (`candle_transformers::models::stable_diffusion`), keeping the
//! `generate(prompt, &ImageGenConfig) -> GeneratedImage` signature.

use std::fs;

use model_hub::{ImageGenConfig, ImageGenerator, StubImageGenerator};

fn main() -> model_hub::Result<()> {
    let prompt = "a serene mountain lake at sunrise, watercolor";

    // 1. Configure the output.
    let config = ImageGenConfig {
        width: 256,
        height: 256,
        seed: 7,
        steps: 24,
    };

    // 2. Generate.
    let mut generator = StubImageGenerator::new();
    println!("engine: {}", generator.name());
    println!("prompt: \"{prompt}\"");
    let image = generator.generate(prompt, &config)?;
    println!("generated {}x{} image ({} bytes RGB)", image.width, image.height, image.rgb.len());

    // 3. Determinism check: same prompt + seed -> identical pixels.
    let again = generator.generate(prompt, &config)?;
    println!("deterministic: {}", again.rgb == image.rgb);

    // 4. Save as PPM.
    let out = "generated.ppm";
    fs::write(out, image.to_ppm())?;
    println!("wrote {out}  (convert with: `convert {out} generated.png`)");

    Ok(())
}
