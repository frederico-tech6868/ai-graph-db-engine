//! Needle orchestrator: a full, LLM-free tool-calling turn.

use serde_json::Value;

use crate::backend::GraphBackend;
use crate::error::Result;
use crate::needle::agent::NeedleAgent;
use crate::pipeline::tools::{graphdb_tool_schemas, ToolExecutor, ToolSchema};

/// Drives a complete tool-calling turn using only the Needle engine: detect a
/// tool from the query, execute it against the graph, and format a plain-text
/// answer from the structured result.
pub struct NeedleOrchestrator {
    agent: NeedleAgent,
    schemas: Vec<ToolSchema>,
}

impl Default for NeedleOrchestrator {
    fn default() -> Self {
        Self {
            agent: NeedleAgent::new(),
            schemas: graphdb_tool_schemas(),
        }
    }
}

impl NeedleOrchestrator {
    /// Create a new orchestrator with the default graphdb tool schemas.
    pub fn new() -> Self {
        Self::default()
    }

    /// The tool schemas this orchestrator can dispatch to.
    pub fn schemas(&self) -> &[ToolSchema] {
        &self.schemas
    }

    /// Run one turn: choose a tool, execute it, and summarize the result.
    pub async fn run<G: GraphBackend>(&self, graph: &G, query: &str) -> Result<String> {
        let Some(call) = self.agent.detect_tool(query, &self.schemas) else {
            return Ok("Needle could not map the request to a known tool.".to_string());
        };
        let result = ToolExecutor::execute(graph, &call).await?;
        Ok(format_result(&call.name, &result))
    }

    /// Run one turn and return the raw tool result as JSON.
    pub async fn run_json<G: GraphBackend>(&self, graph: &G, query: &str) -> Result<Value> {
        match self.agent.detect_tool(query, &self.schemas) {
            Some(call) => ToolExecutor::execute(graph, &call).await,
            None => Ok(Value::Null),
        }
    }
}

/// Format a tool result into a concise human-readable summary.
fn format_result(tool: &str, result: &Value) -> String {
    match tool {
        "search_knowledge_base" => {
            let hits = result.as_array().map(|a| a.len()).unwrap_or(0);
            let mut s = format!("Found {hits} relevant chunk(s):\n");
            if let Some(arr) = result.as_array() {
                for (i, hit) in arr.iter().enumerate().take(5) {
                    let text = hit
                        .get("chunk")
                        .and_then(|c| c.get("text"))
                        .and_then(|t| t.as_str())
                        .unwrap_or("");
                    let score = hit.get("score").and_then(|s| s.as_f64()).unwrap_or(0.0);
                    let preview: String = text.chars().take(160).collect();
                    s.push_str(&format!("  {}. (score {:.3}) {}\n", i + 1, score, preview));
                }
            }
            s
        }
        "list_documents" => {
            let docs = result.as_array().map(|a| a.len()).unwrap_or(0);
            let mut s = format!("{docs} document(s):\n");
            if let Some(arr) = result.as_array() {
                for doc in arr {
                    let title = doc.get("title").and_then(|t| t.as_str()).unwrap_or("?");
                    let path = doc.get("source_path").and_then(|t| t.as_str()).unwrap_or("?");
                    s.push_str(&format!("  - {title} ({path})\n"));
                }
            }
            s
        }
        "get_document_chunks" => {
            let n = result.as_array().map(|a| a.len()).unwrap_or(0);
            format!("Retrieved {n} chunk(s) from the requested document.")
        }
        _ => result.to_string(),
    }
}
