//! The Needle agent: rule-based intent detection and structured extraction.

use regex::Regex;
use serde_json::{json, Map, Value};

use crate::pipeline::tools::{ToolCall, ToolSchema};

/// A deterministic, rules-based agent.
#[derive(Debug, Default, Clone)]
pub struct NeedleAgent;

impl NeedleAgent {
    /// Create a new Needle agent.
    pub fn new() -> Self {
        Self
    }

    /// Choose an appropriate tool call for `query` given the available schemas.
    ///
    /// Uses keyword heuristics: listing intents map to `list_documents`, chunk
    /// requests to `get_document_chunks` (extracting a file path), and
    /// everything else to `search_knowledge_base`.
    pub fn detect_tool(&self, query: &str, schemas: &[ToolSchema]) -> Option<ToolCall> {
        let has = |name: &str| schemas.iter().any(|s| s.name == name);
        let q = query.to_lowercase();

        // get_document_chunks — look for a file-path-like token.
        if has("get_document_chunks") && (q.contains("chunk") || q.contains("sections of")) {
            if let Some(path) = extract_path(query) {
                return Some(ToolCall {
                    name: "get_document_chunks".to_string(),
                    arguments: json!({ "source_path": path }),
                });
            }
        }

        // list_documents — listing / inventory intents.
        if has("list_documents")
            && (q.contains("list") || q.contains("what documents") || q.contains("show all"))
            && q.contains("document")
        {
            return Some(ToolCall {
                name: "list_documents".to_string(),
                arguments: json!({}),
            });
        }

        // Default: search the knowledge base.
        if has("search_knowledge_base") {
            let mut args = Map::new();
            args.insert("query".to_string(), Value::String(query.to_string()));
            if let Some(k) = extract_k(&q) {
                args.insert("k".to_string(), json!(k));
            }
            return Some(ToolCall {
                name: "search_knowledge_base".to_string(),
                arguments: Value::Object(args),
            });
        }

        None
    }

    /// Extract structured information from `text`.
    ///
    /// Produces a JSON object with detected entities (proper nouns), numbers,
    /// emails, dates, and — when the `schema_json` declares object properties —
    /// a best-effort mapping of those property names to matching values.
    pub fn extract(&self, text: &str, schema_json: &str) -> Value {
        let entities = extract_entities(text);
        let emails = extract_all(text, r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}");
        let numbers = extract_all(text, r"\b\d+(?:\.\d+)?\b");
        let dates = extract_all(
            text,
            r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b",
        );

        let mut obj = Map::new();
        obj.insert("entities".to_string(), json!(entities));
        obj.insert("emails".to_string(), json!(emails));
        obj.insert("numbers".to_string(), json!(numbers));
        obj.insert("dates".to_string(), json!(dates));

        // If the schema declares specific properties, attempt to fill them.
        if let Ok(schema) = serde_json::from_str::<Value>(schema_json) {
            if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
                let mut mapped = Map::new();
                for (key, _) in props {
                    let value = best_effort_field(key, text, &entities, &emails, &dates, &numbers);
                    mapped.insert(key.clone(), value);
                }
                obj.insert("fields".to_string(), Value::Object(mapped));
            }
        }

        Value::Object(obj)
    }
}

/// Extract capitalized multi-word proper nouns as candidate entities.
fn extract_entities(text: &str) -> Vec<String> {
    let re = Regex::new(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\b").unwrap();
    let mut seen = Vec::new();
    for cap in re.captures_iter(text) {
        let s = cap[1].trim().to_string();
        // Skip single very short tokens and sentence-initial noise words.
        if s.len() >= 2 && !seen.contains(&s) {
            seen.push(s);
        }
    }
    seen
}

/// Return all matches of `pattern` in `text`.
fn extract_all(text: &str, pattern: &str) -> Vec<String> {
    let re = match Regex::new(pattern) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for m in re.find_iter(text) {
        let s = m.as_str().to_string();
        if !out.contains(&s) {
            out.push(s);
        }
    }
    out
}

/// Extract a file-path-like token from a query.
fn extract_path(text: &str) -> Option<String> {
    let re = Regex::new(r#"["']?((?:/|[A-Za-z]:\\)[^\s"']+)["']?"#).ok()?;
    re.captures(text).map(|c| c[1].to_string())
}

/// Extract a "top N" / "k results" count from a query.
fn extract_k(query: &str) -> Option<usize> {
    let re = Regex::new(r"\b(?:top|first|best)\s+(\d{1,3})\b").ok()?;
    re.captures(query)
        .and_then(|c| c[1].parse::<usize>().ok())
}

/// Best-effort mapping of a schema field name to a value found in the text.
fn best_effort_field(
    key: &str,
    _text: &str,
    entities: &[String],
    emails: &[String],
    dates: &[String],
    numbers: &[String],
) -> Value {
    let k = key.to_lowercase();
    let pick = |v: &[String]| v.first().cloned().map(Value::String).unwrap_or(Value::Null);
    if k.contains("email") {
        pick(emails)
    } else if k.contains("date") || k.contains("time") || k.contains("year") {
        pick(dates)
    } else if k.contains("count") || k.contains("amount") || k.contains("number") || k.contains("age")
    {
        numbers
            .first()
            .and_then(|n| n.parse::<f64>().ok())
            .map(|f| json!(f))
            .unwrap_or(Value::Null)
    } else if k.contains("name")
        || k.contains("person")
        || k.contains("org")
        || k.contains("company")
        || k.contains("entity")
        || k.contains("title")
    {
        pick(entities)
    } else {
        Value::Null
    }
}
