//! # Text Recognition example (OCR)
//!
//! Demonstrates the [`TextRecognizer`] trait with the offline
//! [`StubTextRecognizer`], which detects horizontal bands of "ink" (dark pixels
//! on a light background) and reports one [`OcrLine`] per detected text line.
//!
//! We render a tiny synthetic "page" — a white canvas with a few dark bars that
//! stand in for lines of text — then run OCR line detection over it.
//!
//! Run it:
//! ```bash
//! cargo run -p model-hub --example text_recognition
//! ```
//!
//! ## Going to real OCR
//! Implement [`TextRecognizer`] on top of TrOCR
//! (`candle_transformers::models::trocr`): keep the same
//! `recognize(rgb, w, h) -> Vec<OcrLine>` signature and every caller keeps
//! working.

use model_hub::{StubTextRecognizer, TextRecognizer};

fn main() -> model_hub::Result<()> {
    let (width, height) = (240usize, 120usize);

    // 1. Build a white RGB canvas.
    let mut rgb = vec![255u8; width * height * 3];

    // 2. Paint three dark horizontal "text lines" at different rows/lengths.
    paint_line(&mut rgb, width, 20, 30, 10, 200); // y 20..30, x 10..200
    paint_line(&mut rgb, width, 50, 60, 10, 150); // y 50..60, x 10..150
    paint_line(&mut rgb, width, 85, 95, 10, 120); // y 85..95, x 10..120

    // 3. Run OCR line detection.
    let mut ocr = StubTextRecognizer::new();
    println!("engine: {}", ocr.name());
    let lines = ocr.recognize(&rgb, width, height)?;

    println!("detected {} text line[s]:\n", lines.len());
    for (i, line) in lines.iter().enumerate() {
        println!(
            "  line {}: conf={:.2}  bbox=({}, {}, {}x{})  \"{}\"",
            i + 1,
            line.confidence,
            line.bbox.x,
            line.bbox.y,
            line.bbox.width,
            line.bbox.height,
            line.text
        );
    }

    Ok(())
}

/// Fill a solid dark rectangle (an ink bar) into the RGB buffer.
fn paint_line(rgb: &mut [u8], width: usize, y0: usize, y1: usize, x0: usize, x1: usize) {
    for y in y0..y1 {
        for x in x0..x1 {
            let i = (y * width + x) * 3;
            rgb[i] = 20;
            rgb[i + 1] = 20;
            rgb[i + 2] = 20;
        }
    }
}
