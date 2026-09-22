//! Tool calling: schemas compatible with the Python `GRAPHDB_TOOL_SCHEMAS`,
//! tool-call parsing, execution against a [`GraphBackend`], and engine-selectable
//! tool selection (LLM or Needle).

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::backend::GraphBackend;
use crate::error::{ModelHubError, Result};
use crate::models::{GenerationConfig, TextModel};
use crate::needle::agent::NeedleAgent;
use crate::pipeline::EngineKind;

/// A single tool parameter definition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolParam {
    /// Parameter name.
    pub name: String,
    /// JSON schema type (e.g. `"string"`, `"integer"`).
    pub param_type: String,
    /// Human-readable description.
    pub description: String,
    /// Whether the parameter is required.
    pub required: bool,
}

/// A tool/function schema. Serializes to the same structure the Python
/// `GRAPHDB_TOOL_SCHEMAS` uses (OpenAI-style function definitions).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolSchema {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// Ordered parameter definitions.
    pub parameters: Vec<ToolParam>,
}

impl ToolSchema {
    /// Create a new tool schema with no parameters.
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            parameters: Vec::new(),
        }
    }

    /// Add a parameter (builder style).
    pub fn with_param(
        mut self,
        name: &str,
        param_type: &str,
        description: &str,
        required: bool,
    ) -> Self {
        self.parameters.push(ToolParam {
            name: name.to_string(),
            param_type: param_type.to_string(),
            description: description.to_string(),
            required,
        });
        self
    }

    /// Render the OpenAI-/Needle-compatible JSON function schema.
    pub fn to_function_json(&self) -> Value {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();
        for p in &self.parameters {
            properties.insert(
                p.name.clone(),
                json!({ "type": p.param_type, "description": p.description }),
            );
            if p.required {
                required.push(Value::String(p.name.clone()));
            }
        }
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        })
    }
}

/// The canonical graphdb tool schemas. Mirrors the Python `GRAPHDB_TOOL_SCHEMAS`
/// exactly: `search_knowledge_base`, `list_documents`, `get_document_chunks`.
pub fn graphdb_tool_schemas() -> Vec<ToolSchema> {
    vec![
        ToolSchema::new(
            "search_knowledge_base",
            "Search the graph knowledge base for chunks relevant to a natural language query.",
        )
        .with_param("query", "string", "natural language question", true)
        .with_param("k", "integer", "number of results to return", false)
        .with_param("doc_type", "string", "restrict results to a document type", false),
        ToolSchema::new(
            "list_documents",
            "List all documents that have been ingested into the knowledge base.",
        ),
        ToolSchema::new(
            "get_document_chunks",
            "Retrieve all chunks belonging to a specific document.",
        )
        .with_param("source_path", "string", "absolute file path of the document", true),
    ]
}

/// A parsed tool call: which tool to invoke and with what arguments.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    /// Tool name.
    pub name: String,
    /// Arguments as a JSON object.
    pub arguments: Value,
}

impl ToolCall {
    /// Parse a tool call from raw model output, tolerating code fences and
    /// surrounding prose. Expects a JSON object with `name`/`tool` and
    /// `arguments`/`parameters`.
    pub fn parse(raw: &str) -> Result<Self> {
        let re = regex::Regex::new(r"(?s)\{.*\}").map_err(|e| ModelHubError::Tool(e.to_string()))?;
        let json_str = re
            .find(raw)
            .map(|m| m.as_str())
            .ok_or_else(|| ModelHubError::Tool("no JSON object in tool output".into()))?;
        let v: Value = serde_json::from_str(json_str)?;
        let name = v
            .get("name")
            .or_else(|| v.get("tool"))
            .and_then(|n| n.as_str())
            .ok_or_else(|| ModelHubError::Tool("tool call missing 'name'".into()))?
            .to_string();
        let arguments = v
            .get("arguments")
            .or_else(|| v.get("parameters"))
            .or_else(|| v.get("args"))
            .cloned()
            .unwrap_or_else(|| json!({}));
        Ok(ToolCall { name, arguments })
    }
}

/// Executes tool calls against a [`GraphBackend`].
pub struct ToolExecutor;

impl ToolExecutor {
    /// Execute a tool call, returning its result as JSON.
    pub async fn execute<G: GraphBackend>(graph: &G, call: &ToolCall) -> Result<Value> {
        match call.name.as_str() {
            "search_knowledge_base" => {
                let query = call
                    .arguments
                    .get("query")
                    .and_then(|q| q.as_str())
                    .ok_or_else(|| ModelHubError::Tool("search: missing 'query'".into()))?;
                let k = call
                    .arguments
                    .get("k")
                    .and_then(|k| k.as_u64())
                    .unwrap_or(5) as usize;
                let doc_type = call.arguments.get("doc_type").and_then(|d| d.as_str());
                let results = graph.search(query, k, doc_type).await?;
                Ok(serde_json::to_value(results)?)
            }
            "list_documents" => {
                let docs = graph.list_documents().await?;
                Ok(serde_json::to_value(docs)?)
            }
            "get_document_chunks" => {
                let path = call
                    .arguments
                    .get("source_path")
                    .and_then(|p| p.as_str())
                    .ok_or_else(|| {
                        ModelHubError::Tool("get_document_chunks: missing 'source_path'".into())
                    })?;
                let chunks = graph.get_document_chunks(path).await?;
                Ok(serde_json::to_value(chunks)?)
            }
            other => Err(ModelHubError::Tool(format!("unknown tool '{other}'"))),
        }
    }
}

/// Select a tool call for `query`, using either the LLM or the Needle engine.
///
/// * `EngineKind::Needle` uses [`NeedleAgent`] heuristics (no model needed).
/// * `EngineKind::Llm` prompts `model` to emit a JSON tool call and parses it,
///   falling back to Needle heuristics if the model output cannot be parsed.
pub fn select_tool(
    engine: EngineKind,
    query: &str,
    schemas: &[ToolSchema],
    model: Option<&mut dyn TextModel>,
) -> Result<Option<ToolCall>> {
    let agent = NeedleAgent::new();
    match engine {
        EngineKind::Needle => Ok(agent.detect_tool(query, schemas)),
        EngineKind::Llm => {
            let Some(model) = model else {
                return Ok(agent.detect_tool(query, schemas));
            };
            let tools_json: Vec<Value> = schemas.iter().map(|s| s.to_function_json()).collect();
            let prompt = format!(
                "You are a tool router. Given the tools below and the user request, respond with \
                 ONLY a JSON object of the form {{\"name\": <tool>, \"arguments\": {{...}}}}.\n\
                 Tools: {}\nUser request: {}\nJSON:",
                serde_json::to_string(&tools_json)?,
                query
            );
            let out = model.generate(&prompt, &GenerationConfig::deterministic())?;
            match ToolCall::parse(&out) {
                Ok(call) => Ok(Some(call)),
                Err(_) => Ok(agent.detect_tool(query, schemas)),
            }
        }
    }
}
