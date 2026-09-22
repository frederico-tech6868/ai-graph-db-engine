//! Orchestrator: combines tool calling, retrieval, and generation into a single
//! turn, honoring per-capability engine selection ([`PipelineConfig`]).

use serde_json::Value;

use crate::backend::GraphBackend;
use crate::error::Result;
use crate::models::{GenerationConfig, TextModel};
use crate::pipeline::tools::{graphdb_tool_schemas, select_tool, ToolExecutor, ToolSchema};
use crate::pipeline::{EngineKind, PipelineConfig};

/// Orchestrates a full agentic turn: pick a tool (via the configured engine),
/// execute it against the graph, then have the LLM compose a final answer from
/// the tool result.
pub struct Orchestrator {
    schemas: Vec<ToolSchema>,
    config: PipelineConfig,
}

impl Default for Orchestrator {
    fn default() -> Self {
        Self {
            schemas: graphdb_tool_schemas(),
            config: PipelineConfig::default(),
        }
    }
}

impl Orchestrator {
    /// Create an orchestrator with the given per-capability engine config.
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            schemas: graphdb_tool_schemas(),
            config,
        }
    }

    /// The active pipeline configuration.
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// The tool schemas available to the orchestrator.
    pub fn schemas(&self) -> &[ToolSchema] {
        &self.schemas
    }

    /// Run a full turn. Returns the final answer plus the raw tool result.
    ///
    /// Tool selection uses `config.tool_engine`. If that engine is `Llm`, the
    /// `model` is used both for tool routing and for the final answer; if it is
    /// `Needle`, routing is heuristic and the model only composes the answer.
    pub async fn run<G: GraphBackend>(
        &self,
        graph: &G,
        model: &mut dyn TextModel,
        query: &str,
        gen_config: &GenerationConfig,
    ) -> Result<OrchestratorTurn> {
        // 1. Select the tool. For the LLM engine we pass the model in.
        let call = match self.config.tool_engine {
            EngineKind::Llm => select_tool(EngineKind::Llm, query, &self.schemas, Some(model))?,
            EngineKind::Needle => select_tool(EngineKind::Needle, query, &self.schemas, None)?,
        };

        let Some(call) = call else {
            let answer = model.generate(query, gen_config)?;
            return Ok(OrchestratorTurn {
                answer,
                tool_name: None,
                tool_result: Value::Null,
            });
        };

        // 2. Execute the tool against the graph.
        let tool_result = ToolExecutor::execute(graph, &call).await?;

        // 3. Compose a final answer grounded in the tool result.
        let prompt = format!(
            "You are a helpful assistant with access to a knowledge base. A tool was called to \
             help answer the user.\nTool: {}\nTool result (JSON):\n{}\n\nUser question: {}\n\
             Using the tool result, write a concise, accurate answer.\nAnswer:",
            call.name,
            serde_json::to_string_pretty(&tool_result)?,
            query
        );
        let answer = model.generate(&prompt, gen_config)?;

        Ok(OrchestratorTurn {
            answer,
            tool_name: Some(call.name),
            tool_result,
        })
    }
}

/// The outcome of an orchestrated turn.
#[derive(Debug, Clone)]
pub struct OrchestratorTurn {
    /// The final natural-language answer.
    pub answer: String,
    /// The tool that was invoked, if any.
    pub tool_name: Option<String>,
    /// The raw JSON result of the tool call.
    pub tool_result: Value,
}
