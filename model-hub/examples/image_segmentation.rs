//! # Image Segmentation example
//!
//! Demonstrates the [`ImageSegmenter`] trait with the offline
//! [`StubImageSegmenter`], which partitions an image into `k` regions by
//! quantizing per-pixel luminance into `k` bands — a fast, deterministic
//! approximation of threshold-based segmentation.
//!
//! We build a synthetic image with a dark→light horizontal gradient plus a
//! bright square, segment it into 4 regions, and print each region's pixel area
//! and an ASCII preview of the mask.
//!
//! Run it:
//! ```bash
//! cargo run -p model-hub --example image_segmentation
//! ```
//!
//! ## Going to real segmentation
//! Implement [`ImageSegmenter`] on top of Segment Anything (SAM,
//! `candle_transformers::models::segment_anything`) for prompt-driven instance
//! masks, keeping the `segment(rgb, w, h) -> SegmentationMask` signature.

use model_hub::{ImageSegmenter, StubImageSegmenter};

fn main() -> model_hub::Result<()> {
    let (width, height) = (64usize, 32usize);
    let mut rgb = vec![0u8; width * height * 3];

    // 1. Horizontal dark->light gradient.
    for y in 0..height {
        for x in 0..width {
            let v = (x * 255 / (width - 1)) as u8;
            let i = (y * width + x) * 3;
            rgb[i] = v;
            rgb[i + 1] = v;
            rgb[i + 2] = v;
        }
    }
    // 2. A bright square in the upper-left quadrant.
    for y in 4..14 {
        for x in 4..14 {
            let i = (y * width + x) * 3;
            rgb[i] = 250;
            rgb[i + 1] = 250;
            rgb[i + 2] = 250;
        }
    }

    // 3. Segment into 4 luminance bands.
    let mut seg = StubImageSegmenter::new(4);
    println!("engine: {}", seg.name());
    let mask = seg.segment(&rgb, width, height)?;
    println!("mask: {}x{}, {} segments\n", mask.width, mask.height, mask.num_labels);

    // 4. Report each region's area.
    for label in 0..mask.num_labels {
        println!("  segment {label}: {} pixels", mask.area_of(label));
    }

    // 5. ASCII preview (one glyph per label, subsampled to keep it compact).
    let glyphs = [' ', '.', '+', '#', '@', '%', '*', 'o'];
    println!("\nmask preview:");
    let step_y = (height / 16).max(1);
    let step_x = (width / 48).max(1);
    for y in (0..height).step_by(step_y) {
        let mut row = String::new();
        for x in (0..width).step_by(step_x) {
            let label = mask.labels[y * width + x] as usize;
            row.push(glyphs[label % glyphs.len()]);
        }
        println!("  {row}");
    }

    Ok(())
}
