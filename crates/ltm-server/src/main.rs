//! ltm-server — the MCP server binary (`ltm-mcp`). Wires the concrete
//! implementations from `ltm-db`, `ltm-embed`, and `ltm-rerank` behind the
//! `ltm-tools` pipeline and exposes them over MCP.
//!
//! Two transports (mirroring `server.py`):
//!   - **stdio** (default): `ltm-mcp` — for local clients (Claude Code/Desktop,
//!     LM Studio, ...). The MCP JSON-RPC stream owns stdout; all logging goes
//!     to stderr.
//!   - **streamable HTTP** (`ltm-mcp --http`): bearer-gated streamable HTTP on
//!     `http_host:http_port`. A 32-byte urlsafe token is minted (0600) at
//!     `http_token_file` and is never logged. Requests authenticate with
//!     `Authorization: Bearer <token>` OR a path-embedded `/t/<token>/mcp`
//!     (rewritten to `/mcp`). `/healthz` is open and returns `ok`. Comparisons
//!     are constant-time and HTTP access logging is disabled so the token never
//!     leaks. `public_host` is allowed through the Host (DNS-rebinding) checks.
//!
//! The `ltm-core` traits are blocking, so the async MCP handlers bridge to them
//! via `tokio::task::spawn_blocking`, sharing the constructed `Store`,
//! `EmbeddingBackend`, and `Reranker` behind `Arc`s.

use std::sync::Arc;

use base64::Engine as _;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Implementation, ServerCapabilities, ServerInfo};
use rmcp::transport::stdio;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::{StreamableHttpServerConfig, StreamableHttpService};
use rmcp::{tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler, ServiceExt};
use rmcp::{schemars, serde};

use ltm_core::{Config, DocKey, EmbeddingBackend, Reranker, Store};
use ltm_db::PgStore;
use ltm_embed::make_backend;
use ltm_rerank::make_reranker;
use ltm_tools::Tools;

use serde::Deserialize;

// ---------------------------------------------------------------------------
// Shared, thread-safe application state (built once at startup).
// ---------------------------------------------------------------------------

struct AppState {
    cfg: Config,
    store: Arc<dyn Store>,
    embed: Arc<dyn EmbeddingBackend>,
    rerank: Arc<dyn Reranker>,
}

impl AppState {
    /// Build a transport-agnostic [`Tools`] view over the shared, blocking
    /// backends. Cheap: only borrows the `Arc`-held implementations.
    fn tools(&self) -> Tools<'_> {
        Tools {
            cfg: &self.cfg,
            store: self.store.as_ref(),
            embed: self.embed.as_ref(),
            rerank: self.rerank.as_ref(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tool argument schemas (mirroring server.py's signatures + docstrings).
// ---------------------------------------------------------------------------

fn default_k() -> i64 {
    8
}
fn default_true() -> bool {
    true
}
fn default_max_chars() -> i64 {
    20_000
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(crate = "rmcp::serde")]
struct SearchArgs {
    /// Natural-language search query.
    query: String,
    /// Number of results to return (1-50, default 8).
    #[serde(default = "default_k")]
    #[schemars(default = "default_k")]
    k: i64,
    /// Rerank top-50 with a local cross-encoder (default True; better ordering
    /// + calibrated confidence; first call loads the model, ~15-30 s extra).
    #[serde(default = "default_true")]
    #[schemars(default = "default_true")]
    rerank: bool,
    /// Optional filter. One of: bank statement, lease, invoice, legal, email,
    /// insurance, tax return, loan, appraisal, financials.
    #[serde(default)]
    doc_type: Option<String>,
    /// Optional case-insensitive substring filter on the source file path
    /// (e.g. "Taxes" or "Archive").
    #[serde(default)]
    path_contains: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(crate = "rmcp::serde")]
struct GetDocumentArgs {
    /// A numeric document id (as returned by ltm_search's doc_id), an exact
    /// relative path, or a path fragment. If a fragment matches several
    /// documents, a candidate list is returned instead of text — pick one and
    /// call again with its id.
    path_or_id: String,
    /// Cap on returned text (default 20000). The response says whether it was
    /// truncated and what the full length is.
    #[serde(default = "default_max_chars")]
    #[schemars(default = "default_max_chars")]
    max_chars: i64,
}

// ---------------------------------------------------------------------------
// MCP server: registers the three tools, dispatching to the blocking pipeline.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct LtmServer {
    state: Arc<AppState>,
    router: ToolRouter<LtmServer>,
}

impl LtmServer {
    /// `compat` additionally registers the ChatGPT-connector `search`/`fetch`
    /// aliases (see [`compat_tools_enabled`]).
    fn new(state: Arc<AppState>, compat: bool) -> Self {
        let mut router = Self::tool_router();
        if compat {
            router += Self::tool_router_compat();
        }
        Self { state, router }
    }
}

#[tool_router]
impl LtmServer {
    #[tool(
        name = "ltm_search",
        description = "Semantic search over the Long-Term Memory corpus (~100K local documents: \
property management, loans, legal, banking, leases, taxes, email, ...).\n\n\
Vector search (mxbai-embed-large + pgvector cosine), then an optional local \
cross-encoder rerank of the top 50 candidates. Each hit includes the source \
file path, doc type, snippet, and scores. `confidence` is an honest gate: \
\"LOW\" means the top hit scored so poorly the answer is likely NOT in the \
corpus — say so rather than forcing an answer from weak hits."
    )]
    async fn ltm_search(
        &self,
        Parameters(args): Parameters<SearchArgs>,
    ) -> Result<CallToolResult, McpError> {
        let state = self.state.clone();
        let value = tokio::task::spawn_blocking(move || {
            state.tools().ltm_search(
                &args.query,
                args.k.max(0) as usize,
                args.rerank,
                args.doc_type.as_deref(),
                args.path_contains.as_deref(),
            )
        })
        .await
        .map_err(join_err)?;
        Ok(CallToolResult::structured(value))
    }

    #[tool(
        name = "ltm_get_document",
        description = "Fetch a document's full extracted text from the LTM database (the text \
layer stored at ingestion — not the raw file on disk).\n\n\
Args:\n    \
path_or_id: A numeric document id (as returned by ltm_search's doc_id), an \
exact relative path, or a path fragment. If a fragment matches several \
documents, a candidate list is returned instead of text — pick one and call \
again with its id.\n    \
max_chars: Cap on returned text (default 20000). The response says whether it \
was truncated and what the full length is."
    )]
    async fn ltm_get_document(
        &self,
        Parameters(args): Parameters<GetDocumentArgs>,
    ) -> Result<CallToolResult, McpError> {
        let state = self.state.clone();
        let value = tokio::task::spawn_blocking(move || {
            let key = parse_doc_key(&args.path_or_id);
            state.tools().ltm_get_document(&key, args.max_chars.max(0) as usize)
        })
        .await
        .map_err(join_err)?;
        Ok(CallToolResult::structured(value))
    }

    #[tool(
        name = "ltm_stats",
        description = "Corpus health: document counts by type, chunk/embedding coverage, and \
vector index status (whether searches are fast HNSW or slow seq scan)."
    )]
    async fn ltm_stats(&self) -> Result<CallToolResult, McpError> {
        let state = self.state.clone();
        let value = tokio::task::spawn_blocking(move || state.tools().ltm_stats())
            .await
            .map_err(join_err)?;
        Ok(CallToolResult::structured(value))
    }
}

// ---------------------------------------------------------------------------
// ChatGPT-connector compatibility: OpenAI's connector / Deep Research contract
// requires tools named EXACTLY `search` and `fetch`, each taking a single
// string (search -> {"results": [{id, title, text, url}]}, fetch -> {id,
// title, text, url, metadata}). Thin aliases over the same read-only pipeline
// (see ltm-tools compat_search/compat_fetch) — no new capability. Registered
// in HTTP mode by default (cloud connectors arrive over HTTP); local stdio
// clients keep the clean 3-tool surface unless LTM_COMPAT_TOOLS=on.
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(crate = "rmcp::serde")]
struct CompatSearchArgs {
    /// Natural-language search query.
    query: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
#[serde(crate = "rmcp::serde")]
struct CompatFetchArgs {
    /// Document id from a `search` result.
    id: String,
}

#[tool_router(router = tool_router_compat)]
impl LtmServer {
    #[tool(
        name = "search",
        description = "Search the user's private Long-Term Memory corpus (~100K local documents: \
property management, loans, legal, banking, leases, taxes, email, ...). Returns matching \
documents with an id usable with the `fetch` tool.\n\n\
The `confidence` field is an honest gate: \"LOW\" means the top hit scored so poorly the \
answer is likely NOT in the corpus — report that rather than forcing an answer from weak hits."
    )]
    async fn compat_search(
        &self,
        Parameters(args): Parameters<CompatSearchArgs>,
    ) -> Result<CallToolResult, McpError> {
        let state = self.state.clone();
        let value =
            tokio::task::spawn_blocking(move || state.tools().compat_search(&args.query))
                .await
                .map_err(join_err)?;
        Ok(CallToolResult::structured(value))
    }

    #[tool(
        name = "fetch",
        description = "Fetch the full extracted text of one document from the Long-Term Memory \
corpus by its id (as returned by `search`)."
    )]
    async fn compat_fetch(
        &self,
        Parameters(args): Parameters<CompatFetchArgs>,
    ) -> Result<CallToolResult, McpError> {
        let state = self.state.clone();
        let value = tokio::task::spawn_blocking(move || {
            let key = parse_doc_key(&args.id);
            state.tools().compat_fetch(&key)
        })
        .await
        .map_err(join_err)?;
        if value.get("error").is_some() && value.get("id").is_none() {
            return Ok(CallToolResult::structured_error(value));
        }
        Ok(CallToolResult::structured(value))
    }
}

/// Whether to register the `search`/`fetch` aliases: LTM_COMPAT_TOOLS
/// "on"/"off" forces, anything else ("auto") follows the transport — on for
/// HTTP (where cloud connectors arrive), off for stdio.
fn compat_tools_enabled(http: bool) -> bool {
    match std::env::var("LTM_COMPAT_TOOLS")
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "on" | "1" | "true" | "yes" => true,
        "off" | "0" | "false" | "no" => false,
        _ => http,
    }
}

#[tool_handler(router = self.router)]
impl ServerHandler for LtmServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("ltm", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Read-only semantic search over a local Long-Term Memory document corpus. \
Tools: ltm_search, ltm_get_document, ltm_stats.",
            )
    }
}

/// Mirror server.py's `path_or_id` dispatch: an all-digits string is a numeric
/// id, otherwise it is treated as a path or fragment.
fn parse_doc_key(raw: &str) -> DocKey {
    let key = raw.trim();
    if !key.is_empty() && key.bytes().all(|b| b.is_ascii_digit()) {
        if let Ok(id) = key.parse::<i64>() {
            return DocKey::Id(id);
        }
    }
    DocKey::PathOrFragment(key.to_string())
}

fn join_err(e: tokio::task::JoinError) -> McpError {
    McpError::internal_error(format!("tool task failed: {e}"), None)
}

// ---------------------------------------------------------------------------
// Entry point + transports.
// ---------------------------------------------------------------------------

fn main() {
    // stderr-only logging: stdout belongs to the stdio MCP stream. Matches
    // server.py's `logging.basicConfig(stream=sys.stderr, ...)`.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ltm_mcp=info,rmcp=warn".into()),
        )
        .with_ansi(false)
        .init();

    let http = std::env::args().any(|a| a == "--http");
    let compat = compat_tools_enabled(http);
    let cfg = Config::from_env();

    let store = match PgStore::connect(&cfg) {
        Ok(s) => Arc::new(s) as Arc<dyn Store>,
        Err(e) => {
            eprintln!("ltm-mcp: failed to connect to Postgres: {e}");
            std::process::exit(1);
        }
    };
    let embed: Arc<dyn EmbeddingBackend> = Arc::from(make_backend(&cfg));
    let rerank: Arc<dyn Reranker> = Arc::from(make_reranker(&cfg));

    tracing::info!(
        "ltm-mcp (Rust): transport={} embed={} rerank_available={} compat_tools={}",
        if http { "http" } else { "stdio" },
        embed.describe(),
        rerank.available(),
        compat,
    );

    let state = Arc::new(AppState { cfg, store, embed, rerank });

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");

    // Keep an owning `state` ref in `main` and hand clones to the transports.
    // A concrete `Store` (e.g. `ltm-db`'s blocking `postgres` client) drives its
    // own internal runtime in `Drop`, which panics if it runs inside our async
    // runtime context. Retaining this ref ensures the final `Store` drop happens
    // here on the main thread, after the runtime is torn down — never inside a
    // running runtime.
    let result = if http {
        rt.block_on(run_http(state.clone(), compat))
    } else {
        rt.block_on(run_stdio(state.clone(), compat))
    };

    drop(rt);
    drop(state);

    if let Err(e) = result {
        eprintln!("ltm-mcp: fatal: {e}");
        std::process::exit(1);
    }
}

/// Default stdio transport for local MCP clients.
async fn run_stdio(state: Arc<AppState>, compat: bool) -> Result<(), Box<dyn std::error::Error>> {
    let server = LtmServer::new(state, compat);
    let service = server.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}

/// Bearer-gated streamable HTTP transport (mirrors server.py `run_http`).
async fn run_http(state: Arc<AppState>, compat: bool) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = &state.cfg;
    let token = Arc::new(load_or_create_token(&cfg.http_token_file)?);

    // DNS-rebinding protection: loopback is always allowed; the public tunnel
    // hostname (if configured) is allowed through too.
    let mut allowed_hosts: Vec<String> =
        vec!["localhost".into(), "127.0.0.1".into(), "::1".into()];
    if let Some(h) = cfg.public_host.clone() {
        allowed_hosts.push(h);
    }

    // Stateless: every request is self-contained, so the server survives
    // tunnel/edge reconnects and suits cloud connectors.
    let mut http_config = StreamableHttpServerConfig::default();
    http_config.legacy_session_mode = false;
    http_config.allowed_hosts = allowed_hosts;

    let factory_state = state.clone();
    let mcp_service = StreamableHttpService::new(
        move || Ok(LtmServer::new(factory_state.clone(), compat)),
        Arc::new(LocalSessionManager::default()),
        http_config,
    );

    // `/healthz` is handled inside the gate; MCP lives under `/mcp`.
    let router = axum::Router::new().nest_service("/mcp", mcp_service);

    // The gate must run BEFORE routing so it can rewrite `/t/<token>/mcp` to
    // `/mcp` (middleware added via `Router::layer` runs *after* routing and
    // cannot affect it). Wrapping the whole `Router` in the layer achieves this.
    use axum::ServiceExt as _; // for `into_make_service`
    use tower::Layer as _;
    let gate = axum::middleware::from_fn_with_state(token.clone(), bearer_gate);
    let app = gate.layer(router);

    let addr = format!("{}:{}", cfg.http_host, cfg.http_port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    // Token file path is logged; the token value itself never is.
    tracing::info!(
        "LTM MCP streamable-HTTP on {} (bearer required; token file: {})",
        addr,
        cfg.http_token_file,
    );
    // No access-log layer on purpose: request URLs may carry the path-embedded
    // token, and it must never be written to logs.
    axum::serve(listener, app.into_make_service()).await?;
    Ok(())
}

/// Pure-middleware auth gate in front of the MCP streamable-HTTP service.
///
/// Accepts EITHER `Authorization: Bearer <token>` OR a token embedded as the
/// first path segment (`/t/<token>/mcp` → routed as `/mcp`) for connector UIs
/// that cannot send custom headers. Everything else gets 401. `/healthz` is
/// open (returns "ok", no info). Comparisons are constant-time.
async fn bearer_gate(
    axum::extract::State(token): axum::extract::State<Arc<String>>,
    mut req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::http::{header, StatusCode};
    use axum::response::IntoResponse;

    let path = req.uri().path().to_string();

    if path == "/healthz" {
        return (StatusCode::OK, "ok").into_response();
    }

    let mut ok = false;

    if let Some(v) = req.headers().get(header::AUTHORIZATION) {
        if let Ok(s) = v.to_str() {
            if let Some(rest) = s.strip_prefix("Bearer ") {
                ok = ct_eq(rest.trim(), &token);
            }
        }
    }

    if !ok {
        if let Some(rest) = path.strip_prefix("/t/") {
            if let Some(slash) = rest.find('/') {
                let (tok, tail) = rest.split_at(slash); // tail keeps leading '/'
                if ct_eq(tok, &token) {
                    ok = true;
                    rewrite_path(&mut req, tail);
                }
            }
        }
    }

    if !ok {
        return unauthorized();
    }

    next.run(req).await
}

/// Rewrite the request's path (preserving any query string), used to strip the
/// `/t/<token>` prefix before the request reaches MCP routing.
fn rewrite_path(req: &mut axum::extract::Request, new_path: &str) {
    let mut pq = new_path.to_string();
    if let Some(q) = req.uri().query() {
        pq.push('?');
        pq.push_str(q);
    }
    let mut parts = req.uri().clone().into_parts();
    if let Ok(p) = pq.parse() {
        parts.path_and_query = Some(p);
        if let Ok(uri) = axum::http::Uri::from_parts(parts) {
            *req.uri_mut() = uri;
        }
    }
}

fn unauthorized() -> axum::response::Response {
    use axum::http::{header, StatusCode};
    use axum::response::IntoResponse;
    (
        StatusCode::UNAUTHORIZED,
        [
            (header::CONTENT_TYPE, "application/json"),
            (header::WWW_AUTHENTICATE, "Bearer"),
        ],
        r#"{"error":"unauthorized"}"#,
    )
        .into_response()
}

/// Constant-time string comparison (subtle's `ct_eq` short-circuits only on
/// length, matching Python's `secrets.compare_digest` behavior).
fn ct_eq(a: &str, b: &str) -> bool {
    use subtle::ConstantTimeEq;
    a.as_bytes().ct_eq(b.as_bytes()).into()
}

/// Read the bearer token, minting a 32-byte urlsafe one (0600) on first run.
/// The token value itself is NEVER logged — only the file path.
fn load_or_create_token(path: &str) -> std::io::Result<String> {
    use std::io::Write;
    use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};

    let p = std::path::Path::new(path);
    if let Some(dir) = p.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::DirBuilder::new()
                .recursive(true)
                .mode(0o700)
                .create(dir)?;
        }
    }

    if let Ok(existing) = std::fs::read_to_string(p) {
        let tok = existing.trim().to_string();
        if !tok.is_empty() {
            return Ok(tok);
        }
    }

    let mut buf = [0u8; 32];
    rand::fill(&mut buf);
    let tok = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(buf);

    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(p)?;
    f.write_all(tok.as_bytes())?;
    f.write_all(b"\n")?;
    tracing::info!("minted new bearer token at {path}");
    Ok(tok)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_router_has_the_three_tools() {
        let names: Vec<String> = LtmServer::tool_router()
            .list_all()
            .into_iter()
            .map(|t| t.name.to_string())
            .collect();
        assert_eq!(names.len(), 3);
        for n in ["ltm_search", "ltm_get_document", "ltm_stats"] {
            assert!(names.contains(&n.to_string()), "missing {n}");
        }
    }

    #[test]
    fn compat_router_has_exactly_search_and_fetch() {
        let mut names: Vec<String> = LtmServer::tool_router_compat()
            .list_all()
            .into_iter()
            .map(|t| t.name.to_string())
            .collect();
        names.sort();
        assert_eq!(names, vec!["fetch".to_string(), "search".to_string()]);
    }

    #[test]
    fn combined_router_lists_all_five() {
        let mut router = LtmServer::tool_router();
        router += LtmServer::tool_router_compat();
        assert_eq!(router.list_all().len(), 5);
    }
}
