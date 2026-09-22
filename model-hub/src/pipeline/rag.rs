//! Retrieval-augmented generation (RAG).

use crate::backend::{GraphBackend, SearchResult};
use crate::error::Result;
use crate::models::{cosine_similarity, GenerationConfig, TextModel};
use crate::needle::needle_embed;
use crate::pipeline::EngineKind;

/// A retrieval + generation pipeline over a [`GraphBackend`].
pub struct RagPipeline {
    /// Number of chunks to retrieve.
    pub top_k: usize,
    /// Which engine produces embeddings used for optional re-ranking.
    pub embedding_engine: EngineKind,
    /// Whether to re-rank retrieved chunks by embedding similarity.
    pub rerank: bool,
}

impl Default for RagPipeline {
    fn default() -> Self {
        Self {
            top_k: 5,
            embedding_engine: EngineKind::default(),
            rerank: false,
        }
    }
}

impl RagPipeline {
    /// Create a RAG pipeline retrieving `top_k` chunks, using `embedding_engine`
    /// for optional re-ranking.
    pub fn new(top_k: usize, embedding_engine: EngineKind) -> Self {
        Self {
            top_k,
            embedding_engine,
            rerank: false,
        }
    }

    /// Enable embedding-based re-ranking (builder style).
    pub fn with_rerank(mut self, rerank: bool) -> Self {
        self.rerank = rerank;
        self
    }

    /// Retrieve chunks for `query`, optionally re-ranking them by embedding
    /// similarity computed with the selected engine.
    pub async fn retrieve<G: GraphBackend>(
        &self,
        graph: &G,
        query: &str,
        model: Option<&mut dyn TextModel>,
    ) -> Result<Vec<SearchResult>> {
        let mut results = graph.search(query, self.top_k, None).await?;
        if self.rerank && !results.is_empty() {
            let q_emb = self.embed(query, model)?;
            // Note: this re-ranks using the Needle embedder for chunk vectors to
            // avoid needing a mutable model borrow per chunk.
            for r in results.iter_mut() {
                let c_emb = needle_embed(&r.chunk.text);
                let sim = cosine_similarity(&q_emb, &c_emb);
                // Blend graph score with embedding similarity.
                r.score = 0.5 * r.score + 0.5 * sim;
            }
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        Ok(results)
    }

    /// Compute an embedding for `text` using the configured embedding engine.
    ///
    /// `EngineKind::Needle` uses the deterministic hashed embedder;
    /// `EngineKind::Llm` uses the model's `embed` (falling back to Needle if no
    /// model is provided).
    pub fn embed(&self, text: &str, model: Option<&mut dyn TextModel>) -> Result<Vec<f32>> {
        match self.embedding_engine {
            EngineKind::Needle => Ok(needle_embed(text)),
            EngineKind::Llm => match model {
                Some(m) => m.embed(text),
                None => Ok(needle_embed(text)),
            },
        }
    }

    /// Build a grounded prompt from retrieved context and the user question.
    pub fn build_prompt(&self, query: &str, context: &[SearchResult]) -> String {
        let mut ctx = String::new();
        for (i, r) in context.iter().enumerate() {
            ctx.push_str(&format!(
                "[{}] (source: {}, score: {:.3})\n{}\n\n",
                i + 1,
                r.chunk.source_path,
                r.score,
                r.chunk.text
            ));
        }
        format!(
            "You are a helpful assistant. Answer the question using ONLY the context below. \
             If the answer is not in the context, say so.\n\n\
             Context:\n{ctx}\nQuestion: {query}\nAnswer:"
        )
    }

    /// Full RAG turn: retrieve, build a grounded prompt, and generate an answer.
    pub async fn answer<G: GraphBackend>(
        &self,
        graph: &G,
        model: &mut dyn TextModel,
        query: &str,
        gen_config: &GenerationConfig,
    ) -> Result<RagAnswer> {
        // Retrieve first (no model borrow needed for the default path).
        let context = graph.search(query, self.top_k, None).await?;
        let prompt = self.build_prompt(query, &context);
        let answer = model.generate(&prompt, gen_config)?;
        Ok(RagAnswer { answer, context })
    }
}

/// The result of a RAG turn.
#[derive(Debug, Clone)]
pub struct RagAnswer {
    /// The generated answer.
    pub answer: String,
    /// The retrieved context that grounded the answer.
    pub context: Vec<SearchResult>,
}
