//! Web scraping: turn raw HTML into clean text, extract links, and ingest the
//! result straight into a [`crate::graph::InMemoryGraph`] knowledge base.
//!
//! The parsing here is **pure Rust and dependency-light** (it uses the `regex`
//! crate already vendored by this workspace), so it runs fully offline against
//! an HTML string you already have. Actually *fetching* a URL over the network
//! is intentionally left out of the default build to keep it lean and offline;
//! see [`WebScraper::fetch`] for the recommended `reqwest`-based pattern.

use regex::Regex;

use crate::graph::InMemoryGraph;

/// A pure-Rust HTML scraper.
///
/// Compiles its regexes once on construction and reuses them across calls.
pub struct WebScraper {
    script_style: Regex,
    tag: Regex,
    whitespace: Regex,
    title: Regex,
    href: Regex,
}

impl Default for WebScraper {
    fn default() -> Self {
        Self::new()
    }
}

impl WebScraper {
    /// Construct a scraper with its parsing regexes precompiled.
    pub fn new() -> Self {
        Self {
            // Drop <script>...</script> and <style>...</style> blocks entirely.
            script_style: Regex::new(r"(?is)<(script|style)[^>]*>.*?</\s*(script|style)\s*>")
                .expect("valid regex"),
            // Any remaining HTML tag.
            tag: Regex::new(r"(?s)<[^>]+>").expect("valid regex"),
            // Runs of whitespace.
            whitespace: Regex::new(r"\s+").expect("valid regex"),
            // The document <title>.
            title: Regex::new(r"(?is)<title[^>]*>(.*?)</\s*title\s*>").expect("valid regex"),
            // href="..." attributes.
            href: Regex::new(r#"(?is)<a\b[^>]*?href\s*=\s*["']([^"'#]+)["']"#).expect("valid regex"),
        }
    }

    /// Strip tags and collapse whitespace, returning the readable text content.
    pub fn extract_text(&self, html: &str) -> String {
        let no_scripts = self.script_style.replace_all(html, " ");
        let no_tags = self.tag.replace_all(&no_scripts, " ");
        let decoded = decode_entities(&no_tags);
        self.whitespace.replace_all(&decoded, " ").trim().to_string()
    }

    /// Extract the document `<title>`, if present.
    pub fn extract_title(&self, html: &str) -> Option<String> {
        self.title.captures(html).map(|c| {
            let raw = c.get(1).map(|m| m.as_str()).unwrap_or_default();
            decode_entities(raw).trim().to_string()
        })
    }

    /// Extract hyperlink targets from `<a href="...">` tags.
    ///
    /// Relative links are resolved against `base` when it is provided.
    pub fn extract_links(&self, html: &str, base: Option<&str>) -> Vec<String> {
        let mut out = Vec::new();
        for cap in self.href.captures_iter(html) {
            let raw = cap.get(1).map(|m| m.as_str()).unwrap_or_default();
            let resolved = resolve_url(raw, base);
            if !out.contains(&resolved) {
                out.push(resolved);
            }
        }
        out
    }

    /// Parse `html` and ingest its readable text into `graph` as one document.
    ///
    /// Returns the number of chunks created. The document `title` is taken from
    /// the HTML `<title>` (falling back to `url`) and the `doc_type` is `"web"`.
    pub fn ingest_into(&self, graph: &InMemoryGraph, url: &str, html: &str) -> usize {
        let text = self.extract_text(html);
        let title = self.extract_title(html).unwrap_or_else(|| url.to_string());
        graph.ingest(url, &title, "web", &text)
    }

    /// Fetch a live URL over HTTP. **Not compiled by default.**
    ///
    /// Networking is deliberately excluded from the default build so this crate
    /// stays offline and lean. To enable live fetching, add `reqwest` and a
    /// small wrapper in your binary, for example:
    ///
    /// ```ignore
    /// let html = reqwest::blocking::get(url)?.text()?;
    /// let chunks = scraper.ingest_into(&graph, url, &html);
    /// ```
    ///
    /// This stub always returns an error explaining how to enable fetching.
    pub fn fetch(&self, url: &str) -> crate::error::Result<String> {
        Err(crate::error::ModelHubError::Other(format!(
            "live fetching is disabled in the default build; \
             fetch {url} yourself (e.g. with reqwest) and pass the HTML to \
             extract_text/ingest_into"
        )))
    }
}

/// Resolve a possibly-relative URL against an optional base URL.
fn resolve_url(link: &str, base: Option<&str>) -> String {
    if link.starts_with("http://") || link.starts_with("https://") {
        return link.to_string();
    }
    match base {
        Some(base) => {
            let base = base.trim_end_matches('/');
            if let Some(stripped) = link.strip_prefix('/') {
                // Absolute path: attach to the scheme+host of `base`.
                if let Some(scheme_end) = base.find("://") {
                    let after = &base[scheme_end + 3..];
                    let host_end = after.find('/').map(|i| scheme_end + 3 + i).unwrap_or(base.len());
                    format!("{}/{}", &base[..host_end], stripped)
                } else {
                    format!("{base}/{stripped}")
                }
            } else {
                format!("{base}/{link}")
            }
        }
        None => link.to_string(),
    }
}

/// Decode the handful of HTML entities that commonly appear in body text.
fn decode_entities(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ")
}
