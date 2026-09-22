//! # Web Scraping example
//!
//! Demonstrates the [`WebScraper`]: turn raw HTML into clean text, pull out the
//! title and links, then ingest the readable content straight into an
//! [`InMemoryGraph`] and query it — an end-to-end "scrape → knowledge base →
//! search" flow.
//!
//! Parsing is pure Rust and fully offline, so this example uses an embedded
//! HTML string. Fetching a live URL is intentionally out of the default build
//! (keeps the crate offline/lean); see [`WebScraper::fetch`] for the pattern.
//!
//! Run it:
//! ```bash
//! cargo run -p model-hub --example web_scraping
//! ```

use model_hub::{GraphBackend, InMemoryGraph, WebScraper};

/// A small sample HTML page (stands in for a fetched URL).
const SAMPLE_HTML: &str = r#"
<!DOCTYPE html>
<html>
  <head>
    <title>GraphDB &amp; AI — Overview</title>
    <style>.hidden { display:none; }</style>
  </head>
  <body>
    <h1>Retrieval Augmented Generation</h1>
    <p>The engine combines a property <b>graph</b> with semantic search so an
       LLM can answer questions grounded in your own documents.</p>
    <p>It also supports community detection and a JEPA world model for latent
       retrieval.</p>
    <a href="/docs/rag">RAG docs</a>
    <a href="https://example.com/whisper">Whisper</a>
    <script>console.log("this should be stripped");</script>
  </body>
</html>
"#;

#[tokio::main]
async fn main() -> model_hub::Result<()> {
    let url = "https://example.com/overview";
    let scraper = WebScraper::new();

    // 1. Extract title, links, and readable text.
    let title = scraper.extract_title(SAMPLE_HTML).unwrap_or_default();
    let links = scraper.extract_links(SAMPLE_HTML, Some(url));
    let text = scraper.extract_text(SAMPLE_HTML);

    println!("title: {title}");
    println!("links:");
    for l in &links {
        println!("  - {l}");
    }
    println!("\nclean text:\n  {text}\n");

    // 2. Ingest the page into a knowledge base.
    let graph = InMemoryGraph::new();
    let chunks = scraper.ingest_into(&graph, url, SAMPLE_HTML);
    println!("ingested {chunks} chunk[s] from {url}");

    // 3. Query the freshly-scraped content.
    println!("\n== search: \"latent retrieval world model\" ==");
    for hit in graph.search("latent retrieval world model", 2, None).await? {
        let preview: String = hit.chunk.text.chars().take(80).collect();
        println!("  {:.3}  {}", hit.score, preview);
    }

    Ok(())
}
