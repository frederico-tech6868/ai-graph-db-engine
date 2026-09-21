//! Graph-JEPA world model: context/target encoders, predictor, VICReg loss and
//! the unified [`JEPAGraphRAG`] search struct.
//!
//! Pure-Rust port of the numpy backend in `ai_memory/jepa_graphrag.py`, using
//! `ndarray` for the small MLPs. Weight initialisation is deterministic (a
//! fixed-seed xorshift PRNG + He scaling) so results are reproducible.

use std::collections::HashMap;

use ndarray::{Array1, Array2, Axis};

use crate::error::Result;
use crate::graphrag::GraphRAGRetriever;
use crate::store::GraphStore;

// ---------------------------------------------------------------------------
// Deterministic PRNG + primitives
// ---------------------------------------------------------------------------

/// xorshift64 step.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Next pseudo-random `f32` in `[-1.0, 1.0)`.
fn next_f32(state: &mut u64) -> f32 {
    let r = xorshift64(state);
    let unit = ((r >> 40) as f32) / ((1u64 << 24) as f32); // [0, 1)
    unit * 2.0 - 1.0
}

/// A dense linear layer `y = x . W + b`.
struct Linear {
    w: Array2<f32>,
    b: Array1<f32>,
}

impl Linear {
    /// He-initialised layer (`scale = sqrt(2 / in_dim)`) using a fixed seed.
    fn new(in_dim: usize, out_dim: usize) -> Self {
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let scale = (2.0_f32 / in_dim as f32).sqrt();
        let mut w = Array2::<f32>::zeros((in_dim, out_dim));
        for v in w.iter_mut() {
            *v = next_f32(&mut state) * scale;
        }
        let b = Array1::<f32>::zeros(out_dim);
        Self { w, b }
    }

    /// Forward pass: `x` is `(batch, in_dim)`, result is `(batch, out_dim)`.
    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut out = x.dot(&self.w);
        for mut row in out.rows_mut() {
            row += &self.b;
        }
        out
    }

    /// EMA update: `self = alpha * self + (1 - alpha) * source`.
    fn update_ema(&mut self, source: &Linear, alpha: f32) {
        self.w = &self.w * alpha + &source.w * (1.0 - alpha);
        self.b = &self.b * alpha + &source.b * (1.0 - alpha);
    }

    /// Copy weights from another layer.
    fn copy_from(&mut self, source: &Linear) {
        self.w = source.w.clone();
        self.b = source.b.clone();
    }
}

/// Element-wise ReLU.
fn relu(x: Array2<f32>) -> Array2<f32> {
    x.mapv(|v| v.max(0.0))
}

/// L2-normalize each row (eps 1e-8).
fn l2_normalize_rows(mut m: Array2<f32>) -> Array2<f32> {
    for mut row in m.rows_mut() {
        let norm = row.dot(&row).sqrt();
        let denom = norm + 1e-8;
        row.mapv_inplace(|v| v / denom);
    }
    m
}

// ---------------------------------------------------------------------------
// Encoders / predictor
// ---------------------------------------------------------------------------

/// Context encoder: `input -> 256 -> latent`, L2-normalised output.
pub struct ContextEncoder {
    pub input_dim: usize,
    pub latent_dim: usize,
    fc1: Linear,
    fc2: Linear,
}

impl ContextEncoder {
    pub fn new(input_dim: usize, latent_dim: usize) -> Self {
        Self {
            input_dim,
            latent_dim,
            fc1: Linear::new(input_dim, 256),
            fc2: Linear::new(256, latent_dim),
        }
    }

    pub fn encode(&self, x: &Array2<f32>) -> Array2<f32> {
        let h = relu(self.fc1.forward(x));
        let z = self.fc2.forward(&h);
        l2_normalize_rows(z)
    }

    pub fn encode_one(&self, x: &[f32]) -> Vec<f32> {
        let arr = Array2::from_shape_vec((1, x.len()), x.to_vec())
            .expect("encode_one: shape mismatch");
        let z = self.encode(&arr);
        z.row(0).to_vec()
    }
}

/// Target encoder (EMA-updated copy of the context encoder).
pub struct TargetEncoder {
    pub input_dim: usize,
    pub latent_dim: usize,
    fc1: Linear,
    fc2: Linear,
}

impl TargetEncoder {
    pub fn new(input_dim: usize, latent_dim: usize) -> Self {
        Self {
            input_dim,
            latent_dim,
            fc1: Linear::new(input_dim, 256),
            fc2: Linear::new(256, latent_dim),
        }
    }

    pub fn encode(&self, x: &Array2<f32>) -> Array2<f32> {
        let h = relu(self.fc1.forward(x));
        let z = self.fc2.forward(&h);
        l2_normalize_rows(z)
    }

    pub fn encode_one(&self, x: &[f32]) -> Vec<f32> {
        let arr = Array2::from_shape_vec((1, x.len()), x.to_vec())
            .expect("encode_one: shape mismatch");
        let z = self.encode(&arr);
        z.row(0).to_vec()
    }

    /// EMA-update each layer from the source context encoder.
    pub fn update_ema(&mut self, source: &ContextEncoder, alpha: f32) {
        self.fc1.update_ema(&source.fc1, alpha);
        self.fc2.update_ema(&source.fc2, alpha);
    }

    /// Copy weights from the source context encoder (used for init).
    pub fn copy_from(&mut self, source: &ContextEncoder) {
        self.fc1.copy_from(&source.fc1);
        self.fc2.copy_from(&source.fc2);
    }
}

/// Predictor in latent space: `latent -> latent -> latent`, L2-normalised.
pub struct Predictor {
    pub latent_dim: usize,
    fc1: Linear,
    fc2: Linear,
}

impl Predictor {
    pub fn new(latent_dim: usize) -> Self {
        Self {
            latent_dim,
            fc1: Linear::new(latent_dim, latent_dim),
            fc2: Linear::new(latent_dim, latent_dim),
        }
    }

    pub fn predict(&self, z_ctx: &Array2<f32>) -> Array2<f32> {
        let h = relu(self.fc1.forward(z_ctx));
        let z = self.fc2.forward(&h);
        l2_normalize_rows(z)
    }

    pub fn predict_one(&self, z_ctx: &[f32]) -> Vec<f32> {
        let arr = Array2::from_shape_vec((1, z_ctx.len()), z_ctx.to_vec())
            .expect("predict_one: shape mismatch");
        let z = self.predict(&arr);
        z.row(0).to_vec()
    }
}

// ---------------------------------------------------------------------------
// VICReg loss
// ---------------------------------------------------------------------------

/// The four components of the VICReg loss.
pub struct VICRegLoss {
    pub total: f32,
    pub sim: f32,
    pub var: f32,
    pub cov: f32,
}

/// Subtract the column mean from every row (center the columns).
fn center_columns(m: &Array2<f32>) -> Array2<f32> {
    let mean = m.mean_axis(Axis(0)).unwrap_or_else(|| Array1::zeros(m.ncols()));
    let mut out = m.clone();
    for mut row in out.rows_mut() {
        row -= &mean;
    }
    out
}

/// Sum of squares of the off-diagonal entries of a square matrix.
fn off_diag_sq_sum(m: &Array2<f32>) -> f32 {
    let mut s = 0.0_f32;
    for ((i, j), &v) in m.indexed_iter() {
        if i != j {
            s += v * v;
        }
    }
    s
}

/// VICReg loss (invariance + variance + covariance).
pub fn vicreg_loss(
    z_pred: &Array2<f32>,
    z_target: &Array2<f32>,
    sim_coeff: f32,
    var_coeff: f32,
    cov_coeff: f32,
    eps: f32,
) -> VICRegLoss {
    // 1. Invariance (MSE).
    let diff = z_pred - z_target;
    let sim_loss = diff.mapv(|v| v * v).mean().unwrap_or(0.0);

    // Center columns.
    let z_pred_c = center_columns(z_pred);
    let z_target_c = center_columns(z_target);

    // 2. Variance (hinge on per-dimension std).
    let var_pred = z_pred_c.var_axis(Axis(0), 0.0);
    let var_target = z_target_c.var_axis(Axis(0), 0.0);
    let std_pred = var_pred.mapv(|v| (v + eps).sqrt());
    let std_target = var_target.mapv(|v| (v + eps).sqrt());
    let var_loss = (std_pred.mapv(|s| (1.0 - s).max(0.0)).mean().unwrap_or(0.0)
        + std_target.mapv(|s| (1.0 - s).max(0.0)).mean().unwrap_or(0.0))
        / 2.0;

    // 3. Covariance (off-diagonal energy).
    let batch = z_pred.nrows();
    let latent_dim = z_pred.ncols().max(1);
    let cov_loss = if batch > 1 {
        let denom = batch as f32 - 1.0;
        let cov_pred = z_pred_c.t().dot(&z_pred_c) / denom;
        let cov_target = z_target_c.t().dot(&z_target_c) / denom;
        (off_diag_sq_sum(&cov_pred) + off_diag_sq_sum(&cov_target)) / (2.0 * latent_dim as f32)
    } else {
        0.0
    };

    let total = sim_coeff * sim_loss + var_coeff * var_loss + cov_coeff * cov_loss;
    VICRegLoss {
        total,
        sim: sim_loss,
        var: var_loss,
        cov: cov_loss,
    }
}

// ---------------------------------------------------------------------------
// Unified JEPA-GraphRAG
// ---------------------------------------------------------------------------

/// Result of a hybrid search across all retrieval modes.
pub struct HybridSearchResult {
    /// `(node_id, score)` descending.
    pub local: Option<Vec<(String, f32)>>,
    /// `(community_id, node_ids, score)` descending.
    pub global: Option<Vec<(usize, Vec<String>, f32)>>,
    /// `(community_id, l2_distance)` ascending.
    pub latent_communities: Option<Vec<(usize, f32)>>,
    /// `(node_id, l2_distance)` ascending.
    pub latent_nodes: Option<Vec<(String, f32)>>,
}

/// Unified JEPA-GraphRAG system combining the world model with GraphRAG.
pub struct JEPAGraphRAG {
    pub latent_dim: usize,
    pub context_encoder: ContextEncoder,
    pub target_encoder: TargetEncoder,
    pub predictor: Predictor,
    pub retriever: GraphRAGRetriever,
    community_embeddings: HashMap<usize, Vec<f32>>,
}

impl JEPAGraphRAG {
    pub fn new(input_dim: usize, latent_dim: usize, resolution: f64) -> Self {
        let context_encoder = ContextEncoder::new(input_dim, latent_dim);
        let mut target_encoder = TargetEncoder::new(input_dim, latent_dim);
        target_encoder.copy_from(&context_encoder);
        let predictor = Predictor::new(latent_dim);
        let retriever = GraphRAGRetriever::new(resolution);
        Self {
            latent_dim,
            context_encoder,
            target_encoder,
            predictor,
            retriever,
            community_embeddings: HashMap::new(),
        }
    }

    /// Single training step: encode, compute VICReg loss, EMA-update target.
    pub fn train_step(
        &mut self,
        contexts: Array2<f32>,
        targets: Array2<f32>,
        ema_alpha: f32,
    ) -> VICRegLoss {
        let z_ctx = self.context_encoder.encode(&contexts);
        let z_pred = self.predictor.predict(&z_ctx);
        let z_tgt = self.target_encoder.encode(&targets);
        let loss = vicreg_loss(&z_pred, &z_tgt, 25.0, 25.0, 1.0, 1e-4);
        self.target_encoder.update_ema(&self.context_encoder, ema_alpha);
        loss
    }

    /// Precompute mean-pooled latent embeddings for each community.
    pub fn precompute_community_embeddings(&mut self, store: &GraphStore) {
        let partition = self.retriever.communities(store).clone();

        // Group node embeddings by community.
        let mut comm_nodes: HashMap<usize, Vec<Vec<f32>>> = HashMap::new();
        for node in store.all_nodes() {
            if let Some(&comm_id) = partition.get(&node.id) {
                if let Some(emb) = &node.embedding {
                    comm_nodes.entry(comm_id).or_default().push(emb.clone());
                }
            }
        }

        self.community_embeddings.clear();
        for (comm_id, embs) in comm_nodes.iter() {
            if embs.is_empty() {
                continue;
            }
            let rows = embs.len();
            let cols = embs[0].len();
            let flat: Vec<f32> = embs.iter().flatten().copied().collect();
            let arr = match Array2::from_shape_vec((rows, cols), flat) {
                Ok(a) => a,
                Err(_) => continue,
            };
            let z = self.target_encoder.encode(&arr);
            if let Some(mean) = z.mean_axis(Axis(0)) {
                self.community_embeddings.insert(*comm_id, mean.to_vec());
            }
        }
    }

    /// Latent search over communities: `(community_id, l2_distance)` ascending.
    pub fn latent_search_communities(&self, query_emb: &[f32], k: usize) -> Vec<(usize, f32)> {
        let z_ctx = self.context_encoder.encode_one(query_emb);
        let z_pred = self.predictor.predict_one(&z_ctx);

        let mut results: Vec<(usize, f32)> = Vec::new();
        for (comm_id, z_comm) in self.community_embeddings.iter() {
            results.push((*comm_id, l2_distance(&z_pred, z_comm)));
        }
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Latent search over nodes: `(node_id, l2_distance)` ascending.
    pub fn latent_search_nodes(
        &self,
        store: &GraphStore,
        query_emb: &[f32],
        k: usize,
    ) -> Vec<(String, f32)> {
        let z_ctx = self.context_encoder.encode_one(query_emb);
        let z_pred = self.predictor.predict_one(&z_ctx);

        let mut results: Vec<(String, f32)> = Vec::new();
        for node in store.all_nodes() {
            if let Some(emb) = &node.embedding {
                let z_node = self.target_encoder.encode_one(emb);
                results.push((node.id.clone(), l2_distance(&z_pred, &z_node)));
            }
        }
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Hybrid search combining local, global and latent retrieval modes.
    pub fn hybrid_search(
        &mut self,
        store: &GraphStore,
        query_emb: &[f32],
        k: usize,
        use_local: bool,
        use_global: bool,
        use_latent: bool,
    ) -> Result<HybridSearchResult> {
        let local = if use_local {
            Some(self.retriever.local_search(store, query_emb, k, 2, None)?)
        } else {
            None
        };

        let global = if use_global {
            Some(self.retriever.global_search(store, query_emb, k, None)?)
        } else {
            None
        };

        let (latent_communities, latent_nodes) = if use_latent {
            if self.community_embeddings.is_empty() {
                self.precompute_community_embeddings(store);
            }
            (
                Some(self.latent_search_communities(query_emb, k)),
                Some(self.latent_search_nodes(store, query_emb, k)),
            )
        } else {
            (None, None)
        };

        Ok(HybridSearchResult {
            local,
            global,
            latent_communities,
            latent_nodes,
        })
    }
}

/// Euclidean (L2) distance between two equal-length vectors.
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}
