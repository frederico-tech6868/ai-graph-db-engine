//! Integration tests for the model-hub pipelines (offline / stub only).

use model_hub::models::text::StubTextModel;
use model_hub::needle::NeedleOrchestrator;
use model_hub::{
    graphdb_tool_schemas, EngineKind, GenerationConfig, InMemoryGraph, Orchestrator,
    PipelineConfig, RagPipeline, StructuredExtractor, TextModel, ToolCall, ToolExecutor,
};

fn demo_graph() -> InMemoryGraph {
    let g = InMemoryGraph::new();
    g.ingest(
        "/kb/rust.md",
        "Rust",
        "markdown",
        "Rust is a systems programming language focused on safety and concurrency.",
    );
    g.ingest(
        "/kb/graphdb.md",
        "GraphDB",
        "markdown",
        "The graph database engine supports semantic search and retrieval augmented generation.",
    );
    g
}

#[tokio::test]
async fn search_tool_executes() {
    let g = demo_graph();
    let call = ToolCall {
        name: "search_knowledge_base".to_string(),
        arguments: serde_json::json!({ "query": "memory safety", "k": 2 }),
    };
    let out = ToolExecutor::execute(&g, &call).await.unwrap();
    assert!(out.as_array().is_some());
    assert!(!out.as_array().unwrap().is_empty());
}

#[tokio::test]
async fn needle_orchestrator_lists_documents() {
    let g = demo_graph();
    let orch = NeedleOrchestrator::new();
    let out = orch.run(&g, "list all documents").await.unwrap();
    assert!(out.contains("document"));
}

#[test]
fn tool_schemas_match_expected_names() {
    let schemas = graphdb_tool_schemas();
    let names: Vec<&str> = schemas.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "search_knowledge_base",
            "list_documents",
            "get_document_chunks"
        ]
    );
    // search_knowledge_base must expose query/k/doc_type.
    let search = &schemas[0];
    let params: Vec<&str> = search.parameters.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(params, vec!["query", "k", "doc_type"]);
}

#[test]
fn extraction_needle_finds_entities() {
    let extractor = StructuredExtractor::new(EngineKind::Needle);
    let schema = r#"{"type":"object","properties":{"name":{"type":"string"}}}"#;
    let v = extractor
        .extract("Apple was founded by Steve Jobs.", schema, None)
        .unwrap();
    let entities = v.get("entities").and_then(|e| e.as_array()).unwrap();
    assert!(entities.iter().any(|e| e.as_str() == Some("Apple")));
}

#[tokio::test]
async fn rag_answer_with_stub_model() {
    let g = demo_graph();
    let rag = RagPipeline::new(2, EngineKind::Needle);
    let mut model: Box<dyn TextModel> = Box::new(StubTextModel::new());
    let ans = rag
        .answer(&g, model.as_mut(), "What is Rust?", &GenerationConfig::default())
        .await
        .unwrap();
    assert!(!ans.context.is_empty());
    assert!(!ans.answer.is_empty());
}

#[tokio::test]
async fn orchestrator_needle_tool_then_answer() {
    let g = demo_graph();
    let orch = Orchestrator::new(PipelineConfig::all(EngineKind::Needle));
    let mut model: Box<dyn TextModel> = Box::new(StubTextModel::new());
    let turn = orch
        .run(&g, model.as_mut(), "search for concurrency", &GenerationConfig::default())
        .await
        .unwrap();
    assert_eq!(turn.tool_name.as_deref(), Some("search_knowledge_base"));
    assert!(!turn.answer.is_empty());
}

#[test]
fn stub_embeddings_are_deterministic() {
    let mut a = StubTextModel::new();
    let e1 = a.embed("hello graph world").unwrap();
    let e2 = a.embed("hello graph world").unwrap();
    assert_eq!(e1, e2);
    assert_eq!(e1.len(), 384);
}
