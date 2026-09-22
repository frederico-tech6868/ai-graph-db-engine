//! Structured extraction: pull JSON matching a schema out of free text, using
//! either the LLM or the Needle engine.

use serde_json::Value;

use crate::error::Result;
use crate::models::{GenerationConfig, TextModel};
use crate::needle::agent::NeedleAgent;
use crate::pipeline::EngineKind;

/// Extracts structured JSON from text.
pub struct StructuredExtractor {
    engine: EngineKind,
    agent: NeedleAgent,
}

impl StructuredExtractor {
    /// Create an extractor using the given engine.
    pub fn new(engine: EngineKind) -> Self {
        Self {
            engine,
            agent: NeedleAgent::new(),
        }
    }

    /// The engine backing this extractor.
    pub fn engine(&self) -> EngineKind {
        self.engine
    }

    /// Extract structured data from `text` conforming to `schema_json`.
    ///
    /// With `EngineKind::Llm`, `model` must be supplied; the model is prompted to
    /// emit JSON, which is then parsed (falling back to Needle on failure).
    /// With `EngineKind::Needle`, extraction is rule-based and needs no model.
    pub fn extract(
        &self,
        text: &str,
        schema_json: &str,
        model: Option<&mut dyn TextModel>,
    ) -> Result<Value> {
        match self.engine {
            EngineKind::Needle => Ok(self.agent.extract(text, schema_json)),
            EngineKind::Llm => match model {
                None => Ok(self.agent.extract(text, schema_json)),
                Some(model) => {
                    let prompt = format!(
                        "Extract structured information from the text as JSON that conforms to the \
                         schema. Respond with ONLY the JSON (optionally in a ```json code block).\n\
                         Schema:\n{schema_json}\nText:\n{text}\nJSON:"
                    );
                    let response = model.generate(&prompt, &GenerationConfig::deterministic())?;
                    match parse_json_block(&response) {
                        Some(v) => Ok(v),
                        None => Ok(self.agent.extract(text, schema_json)),
                    }
                }
            },
        }
    }
}

/// Find and parse the first JSON object/array in `text`, tolerating code fences.
fn parse_json_block(text: &str) -> Option<Value> {
    // Prefer a fenced ```json ... ``` block.
    if let Ok(re) = regex::Regex::new(r"(?s)```(?:json)?\s*([\[{].*?[\]}])\s*```") {
        if let Some(cap) = re.captures(text) {
            if let Ok(v) = serde_json::from_str::<Value>(&cap[1]) {
                return Some(v);
            }
        }
    }
    // Otherwise grab the first balanced-looking object or array.
    if let Ok(re) = regex::Regex::new(r"(?s)[\[{].*[\]}]") {
        if let Some(m) = re.find(text) {
            if let Ok(v) = serde_json::from_str::<Value>(m.as_str()) {
                return Some(v);
            }
        }
    }
    None
}
