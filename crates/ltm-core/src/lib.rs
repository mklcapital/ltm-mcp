//! ltm-core — shared types, configuration, errors, and the **trait contracts**
//! that the other crates implement. This is the frozen interface for the
//! parallel port: `ltm-db` implements [`Store`], `ltm-embed` implements
//! [`EmbeddingBackend`], `ltm-rerank` implements [`Reranker`], `ltm-tools`
//! consumes all three, and `ltm-server` wires concrete implementations.
//!
//! Design decision (documented so the parallel crates agree): the data,
//! embedding, and rerank traits are **synchronous/blocking**. The async MCP
//! transport lives only in `ltm-server`, which bridges to these blocking calls
//! via `spawn_blocking`. This keeps `ltm-db`/`ltm-embed`/`ltm-rerank` free to
//! use simple blocking clients (e.g. the `postgres` crate, `ureq`/blocking
//! `reqwest`) without forcing an async runtime across the whole tree.

use std::collections::BTreeMap;
use std::fmt;

use serde::Serialize;

pub type Result<T> = std::result::Result<T, Error>;

/// Unified error type. Each layer maps its own failures into a variant.
#[derive(Debug)]
pub enum Error {
    Config(String),
    Db(String),
    Embed(String),
    Rerank(String),
    NotFound(String),
    Other(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Config(m) => write!(f, "config error: {m}"),
            Error::Db(m) => write!(f, "db error: {m}"),
            Error::Embed(m) => write!(f, "embed error: {m}"),
            Error::Rerank(m) => write!(f, "rerank error: {m}"),
            Error::NotFound(m) => write!(f, "not found: {m}"),
            Error::Other(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for Error {}

// ---------------------------------------------------------------------------
// Configuration (env-driven; mirrors server.py plus a pluggable backend)
// ---------------------------------------------------------------------------

/// Which embedding backend to use. The maintainer requirement: this must be
/// swappable (Ollama today; OpenRouter / LM Studio / any OpenAI-compatible
/// endpoint / cloud provider) without code changes — only config.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EmbedBackendKind {
    /// Local Ollama (`/api/embeddings`).
    Ollama,
    /// Any OpenAI-compatible `/v1/embeddings` endpoint (OpenRouter, LM Studio,
    /// on-prem, cloud). Distinguished only by base URL + key.
    OpenAiCompatible,
}

#[derive(Clone, Debug)]
pub struct Config {
    // Postgres
    pub pg_host: String,
    pub pg_port: u16,
    pub pg_user: String,
    pub pg_database: String,

    // Embedding backend (pluggable)
    pub embed_backend: EmbedBackendKind,
    pub embed_model: String,
    /// Ollama embeddings URL (when embed_backend == Ollama).
    pub ollama_url: String,
    /// Base URL for an OpenAI-compatible backend (when OpenAiCompatible).
    pub embed_base_url: Option<String>,
    /// API key for the OpenAI-compatible backend, if required.
    pub embed_api_key: Option<String>,
    /// The mxbai asymmetric-retrieval query prefix.
    pub query_prefix: String,

    // Rerank (optional)
    pub rerank_python: Option<String>,
    pub rerank_script: Option<String>,
    pub rerank_candidates: usize,
    pub rerank_timeout_s: u64,

    // Confidence gates
    pub low_rerank: f64,
    pub high_rerank: f64,
    pub low_cosine: f64,
    pub high_cosine: f64,

    // HTTP transport
    pub http_host: String,
    pub http_port: u16,
    pub http_token_file: String,
    pub public_host: Option<String>,
}

impl Config {
    /// Load configuration from the `LTM_*` environment variables, matching
    /// server.py's defaults. (Implemented here so every crate shares it.)
    pub fn from_env() -> Self {
        fn var(k: &str) -> Option<String> {
            std::env::var(k).ok().filter(|s| !s.is_empty())
        }
        fn f(k: &str, d: f64) -> f64 {
            var(k).and_then(|s| s.parse().ok()).unwrap_or(d)
        }
        let backend = match var("LTM_EMBED_BACKEND").as_deref() {
            Some("openai") | Some("openai-compatible") | Some("openrouter") | Some("lmstudio") => {
                EmbedBackendKind::OpenAiCompatible
            }
            _ => EmbedBackendKind::Ollama,
        };
        Config {
            pg_host: var("LTM_PGHOST").unwrap_or_else(|| "127.0.0.1".into()),
            pg_port: var("LTM_PGPORT").and_then(|s| s.parse().ok()).unwrap_or(5433),
            pg_user: var("LTM_PGUSER").unwrap_or_else(|| "postgres".into()),
            pg_database: var("LTM_PGDATABASE").unwrap_or_else(|| "brain".into()),

            embed_backend: backend,
            embed_model: var("LTM_EMBED_MODEL").unwrap_or_else(|| "mxbai-embed-large".into()),
            ollama_url: var("LTM_OLLAMA_URL")
                .unwrap_or_else(|| "http://127.0.0.1:11434/api/embeddings".into()),
            embed_base_url: var("LTM_EMBED_BASE_URL"),
            embed_api_key: var("LTM_EMBED_API_KEY"),
            query_prefix: "Represent this sentence for searching relevant passages: ".into(),

            rerank_python: var("LTM_RERANK_PYTHON"),
            rerank_script: var("LTM_RERANK_SCRIPT"),
            rerank_candidates: var("LTM_RERANK_CANDIDATES").and_then(|s| s.parse().ok()).unwrap_or(50),
            rerank_timeout_s: var("LTM_RERANK_TIMEOUT_S").and_then(|s| s.parse().ok()).unwrap_or(300),

            low_rerank: f("LTM_LOW_RERANK", 3.0),
            high_rerank: f("LTM_HIGH_RERANK", 6.0),
            low_cosine: f("LTM_LOW_COSINE", 0.78),
            high_cosine: f("LTM_HIGH_COSINE", 0.855),

            http_host: var("LTM_HTTP_HOST").unwrap_or_else(|| "127.0.0.1".into()),
            http_port: var("LTM_HTTP_PORT").and_then(|s| s.parse().ok()).unwrap_or(8322),
            http_token_file: var("LTM_HTTP_TOKEN_FILE").unwrap_or_else(|| {
                let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
                format!("{home}/.ltm/http-token")
            }),
            public_host: var("LTM_PUBLIC_HOST"),
        }
    }
}

// ---------------------------------------------------------------------------
// Shared data types
// ---------------------------------------------------------------------------

/// One candidate chunk row returned by [`Store::search`]. `ltm-tools` turns
/// these into [`Hit`]s and (when reranking) feeds `chunk_text` to the reranker.
#[derive(Clone, Debug)]
pub struct ChunkRow {
    pub chunk_id: i64,
    pub doc_id: i64,
    pub doc_type: Option<String>,
    pub title: Option<String>,
    pub source_label: Option<String>,
    pub root: Option<String>,
    pub relpath: Option<String>,
    pub doc_date: Option<String>,
    pub text_method: Option<String>,
    pub cosine: f64,
    pub chunk_text: String,
}

/// A search hit as returned to the MCP client (serialized to JSON).
#[derive(Clone, Debug, Serialize)]
pub struct Hit {
    pub path: String,
    pub doc_id: i64,
    pub chunk_id: i64,
    pub title: Option<String>,
    pub doc_type: String,
    pub source: Option<String>,
    pub doc_date: Option<String>,
    pub extraction: Option<String>,
    pub cosine: f64,
    pub snippet: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank_score: Option<f64>,
}

/// Result of a document lookup by id / path / fragment.
#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum DocLookup {
    Found(Box<Document>),
    Ambiguous { candidates: Vec<DocCandidate> },
    NotFound,
}

#[derive(Clone, Debug, Serialize)]
pub struct Document {
    pub doc_id: i64,
    pub path: String,
    pub source: Option<String>,
    pub mime: Option<String>,
    pub doc_type: String,
    pub title: Option<String>,
    pub doc_date: Option<String>,
    pub bytes: Option<i64>,
    pub chunk_count: Option<i64>,
    pub char_count: Option<i64>,
    pub extraction: Option<String>,
    pub ocr: bool,
    pub ingested_at: Option<String>,
    pub text: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct DocCandidate {
    pub doc_id: i64,
    pub path: String,
    pub doc_type: String,
    pub title: Option<String>,
    pub char_count: Option<i64>,
}

/// Corpus statistics returned by [`Store::stats`] (the DB-derived portion;
/// `ltm-tools` adds embed/rerank health).
#[derive(Clone, Debug, Serialize)]
pub struct Stats {
    pub documents: i64,
    pub chunks: i64,
    pub mxbai_embeddings: i64,
    pub documents_by_type: BTreeMap<String, i64>,
    pub hnsw_ready: bool,
    pub table_sizes: BTreeMap<String, String>,
}

/// Filters for a search.
#[derive(Clone, Debug, Default)]
pub struct SearchFilter {
    pub doc_type: Option<String>,
    pub path_contains: Option<String>,
}

/// How a document is addressed in `ltm_get_document`.
#[derive(Clone, Debug)]
pub enum DocKey {
    Id(i64),
    PathOrFragment(String),
}

// ---------------------------------------------------------------------------
// Trait contracts (FROZEN interface for parallel implementation)
// ---------------------------------------------------------------------------

/// Read-only corpus data access (Postgres + pgvector). Implemented by `ltm-db`.
pub trait Store: Send + Sync {
    /// Cosine top-`fetch_k` over `emb_mxbai`, honoring optional filters.
    fn search(&self, query_embedding: &[f32], fetch_k: usize, filter: &SearchFilter) -> Result<Vec<ChunkRow>>;
    /// Fetch a document by id / exact path / fragment (ambiguous → candidates).
    fn get_document(&self, key: &DocKey, max_chars: usize) -> Result<DocLookup>;
    /// Corpus counts, sizes, and HNSW index status.
    fn stats(&self) -> Result<Stats>;
    /// Whether a valid HNSW index exists (fast path vs. seq scan).
    fn hnsw_ready(&self) -> Result<bool>;
}

/// Pluggable embedding backend. Implemented by `ltm-embed` (Ollama +
/// OpenAI-compatible adapters selected via [`Config`]).
pub trait EmbeddingBackend: Send + Sync {
    /// Embed a query (the implementation applies the asymmetric query prefix).
    fn embed_query(&self, text: &str) -> Result<Vec<f32>>;
    /// Human-readable backend id, e.g. "ollama(mxbai-embed-large)".
    fn describe(&self) -> String;
    /// Health string for `ltm_stats` (e.g. "ok (model present)").
    fn health(&self) -> String;
}

/// Optional cross-encoder reranker. Implemented by `ltm-rerank`.
pub trait Reranker: Send + Sync {
    /// Returns (order into `texts` best-first, absolute scores).
    fn rerank(&self, query: &str, texts: &[String], top_k: usize) -> Result<(Vec<usize>, Vec<f64>)>;
    fn available(&self) -> bool;
}

// ---------------------------------------------------------------------------
// Shared helpers (used by ltm-tools; defined here so behavior is consistent)
// ---------------------------------------------------------------------------

pub fn verdict_from_rerank(top: f64, cfg: &Config) -> String {
    if top >= cfg.high_rerank {
        "high".into()
    } else if top >= cfg.low_rerank {
        "medium".into()
    } else {
        "LOW — answer may not be in the corpus".into()
    }
}

pub fn verdict_from_cosine(top: f64, cfg: &Config) -> String {
    if top >= cfg.high_cosine {
        "high (cosine only — enable rerank for a calibrated verdict)".into()
    } else if top >= cfg.low_cosine {
        "medium (cosine only — enable rerank for a calibrated verdict)".into()
    } else {
        "LOW — answer may not be in the corpus (cosine only)".into()
    }
}

/// Collapse whitespace and truncate — used for snippets.
pub fn clean_snippet(text: &str, limit: usize) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    collapsed.chars().take(limit).collect()
}
