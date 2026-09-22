//! # GraphBackend example
//!
//! Shows the storage abstraction at the heart of the pipeline layer. Every AI
//! pipeline (RAG, tool calling) talks only to the [`GraphBackend`] trait, never
//! to a concrete engine. Here we use the bundled [`InMemoryGraph`] reference
//! implementation.
//!
//! What it demonstrates:
//! * ingesting documents (they are auto-chunked),
//! * `search` with and without a `doc_type` filter,
//! * `list_documents` and `get_document_chunks`.
//!
//! Run it:
//! ```bash
//! cargo run -p model-hub --example graph_backend
//! ```
//!
//! To back the pipelines with the real graph engine instead, implement
//! `GraphBackend` on your own store and pass it wherever an `InMemoryGraph` is
//! used — nothing else changes.

use model_hub::{GraphBackend, InMemoryGraph};

#[tokio::main]
async fn main() -> model_hub::Result<()> {
    // 1. Build a knowledge base and ingest a few documents.
    let graph = InMemoryGraph::new();
    graph.ingest(
        "/kb/rust.md",
        "The Rust Language",
        "markdown",
        "Rust is a systems programming language focused on safety, speed, and \
         concurrency. Its ownership model guarantees memory safety without a \
         garbage collector.",
    );
    graph.ingest(
        "/kb/graphdb.md",
        "Graph Database Engine",
        "markdown",
        "The graph database engine stores nodes and edges and supports semantic \
         search, community detection, and retrieval augmented generation over a \
         property graph.",
    );
    graph.ingest(
        "/kb/whisper.txt",
        "Speech To Text",
        "text",
        "Whisper is an automatic speech recognition model that transcribes audio \
         into text across many languages.",
    );

    println!(
        "Ingested {} documents / {} chunks\n",
        graph.document_count(),
        graph.chunk_count()
    );

    // 2. List everything in the store.
    println!("== Documents ==");
    for doc in graph.list_documents().await? {
        println!(
            "  {:<14} [{}] \"{}\" ({} chunk[s])",
            doc.source_path, doc.doc_type, doc.title, doc.chunk_count
        );
    }
    println!();

    // 3. Semantic search across all document types.
    println!("== search: \"memory safety\" (all types) ==");
    for hit in graph.search("memory safety", 3, None).await? {
        println!("  {:.3}  {}  {}", hit.score, hit.chunk.source_path, snippet(&hit.chunk.text));
    }
    println!();

    // 4. The same query, filtered to a single document type.
    println!("== search: \"semantic search\" (doc_type = markdown) ==");
    for hit in graph.search("semantic search", 3, Some("markdown")).await? {
        println!("  {:.3}  {}  {}", hit.score, hit.chunk.source_path, snippet(&hit.chunk.text));
    }
    println!();

    // 5. Pull back the raw chunks of one document.
    println!("== chunks of /kb/graphdb.md ==");
    for chunk in graph.get_document_chunks("/kb/graphdb.md").await? {
        println!("  #{}  {}", chunk.index, snippet(&chunk.text));
    }

    Ok(())
}

/// Truncate text for tidy console output.
fn snippet(text: &str) -> String {
    let s: String = text.chars().take(70).collect();
    if text.chars().count() > 70 {
        format!("{s}…")
    } else {
        s
    }
}
