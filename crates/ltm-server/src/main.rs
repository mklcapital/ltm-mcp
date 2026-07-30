//! ltm-server — the MCP server binary (`ltm-mcp`). Wires the concrete
//! implementations from `ltm-db`, `ltm-embed`, and `ltm-rerank` behind the
//! `ltm-tools` pipeline and exposes them over MCP.
//!
//! STUB: this currently just loads config and constructs the pieces (proving
//! the wiring type-checks). The owning agent implements the transports (see
//! PORTING.md → ltm-server):
//!   - stdio transport (`ltm-mcp`), default;
//!   - streamable HTTP (`ltm-mcp --http`) on 127.0.0.1:8322, bearer-gated
//!     (token minted 0600 at `http_token_file`; `/healthz` open; path-embedded
//!     `/t/<token>/mcp` accepted; constant-time compare; access log off);
//!   - register the three tools with the schemas server.py exposes.
//!
//! Because the ltm-core traits are blocking, the async transport bridges to
//! them via `spawn_blocking`.

use ltm_core::{Config, DocKey};
use ltm_db::PgStore;
use ltm_embed::make_backend;
use ltm_rerank::make_reranker;
use ltm_tools::Tools;

fn main() {
    let http = std::env::args().any(|a| a == "--http");
    let cfg = Config::from_env();

    let store = match PgStore::connect(&cfg) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ltm-mcp: failed to connect to Postgres: {e}");
            std::process::exit(1);
        }
    };
    let embed = make_backend(&cfg);
    let rerank = make_reranker(&cfg);
    let tools = Tools { cfg: &cfg, store: &store, embed: embed.as_ref(), rerank: rerank.as_ref() };

    // Smoke: exercise the wiring so the whole graph type-checks end to end.
    let _ = tools.ltm_stats();
    let _ = tools.ltm_search("hello", 8, true, None, None);
    let _ = tools.ltm_get_document(&DocKey::PathOrFragment("lease".into()), 20_000);

    eprintln!(
        "ltm-mcp (Rust) scaffold: transport={} embed={} rerank_available={}",
        if http { "http" } else { "stdio" },
        embed.describe(),
        rerank.available(),
    );
    eprintln!("transports not yet implemented — see PORTING.md (ltm-server).");
}
