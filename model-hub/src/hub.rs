//! Model hub: resolves [`ModelSource`]s to on-disk paths (downloading from the
//! Hugging Face Hub when needed) and constructs concrete models.

use std::path::PathBuf;

use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};

use crate::error::{ModelHubError, Result};
use crate::models::config::{DeviceKind, ModelSource};
use crate::models::{resolve_device, TextModel};

/// Resolves and caches model weights.
pub struct ModelHub {
    device_kind: DeviceKind,
}

impl Default for ModelHub {
    fn default() -> Self {
        Self {
            device_kind: DeviceKind::Cpu,
        }
    }
}

impl ModelHub {
    /// Create a new hub targeting the given device.
    pub fn new(device_kind: DeviceKind) -> Self {
        Self { device_kind }
    }

    /// The device kind this hub loads models onto.
    pub fn device_kind(&self) -> DeviceKind {
        self.device_kind
    }

    /// Resolve a [`ModelSource`] to a local file path, downloading if needed.
    pub fn resolve(&self, source: &ModelSource) -> Result<PathBuf> {
        match source {
            ModelSource::LocalPath(p) => {
                let path = PathBuf::from(p);
                if !path.exists() {
                    return Err(ModelHubError::ModelNotFound(p.clone()));
                }
                Ok(path)
            }
            ModelSource::HuggingFace {
                repo,
                revision,
                file,
            } => self.download(repo, revision.as_deref(), file),
        }
    }

    /// Download a single file from a Hugging Face repo, returning its cached path.
    pub fn download(&self, repo: &str, revision: Option<&str>, file: &str) -> Result<PathBuf> {
        let api = ApiBuilder::new()
            .build()
            .map_err(|e| ModelHubError::ModelLoad(format!("hf-hub api: {e}")))?;
        let repo_handle = match revision {
            Some(rev) => api.repo(Repo::with_revision(
                repo.to_string(),
                RepoType::Model,
                rev.to_string(),
            )),
            None => api.model(repo.to_string()),
        };
        let path = repo_handle
            .get(file)
            .map_err(|e| ModelHubError::ModelNotFound(format!("{repo}/{file}: {e}")))?;
        Ok(path)
    }

    /// Load a quantized Llama GGUF text model from a [`ModelSource`] plus its
    /// tokenizer source.
    pub fn load_llama(
        &self,
        weights: &ModelSource,
        tokenizer: &ModelSource,
    ) -> Result<Box<dyn TextModel>> {
        let gguf = self.resolve(weights)?;
        let tok = self.resolve(tokenizer)?;
        let device = resolve_device(self.device_kind)?;
        let model = crate::models::text::QuantizedLlama::load(gguf, tok, device)?;
        Ok(Box::new(model))
    }
}
