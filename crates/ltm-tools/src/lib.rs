//! ltm-tools — the three MCP tools (`ltm_search`, `ltm_get_document`,
//! `ltm_stats`), orchestrating a [`Store`], an [`EmbeddingBackend`], and an
//! optional [`Reranker`] from `ltm-core`. Transport-agnostic: returns
//! `serde_json::Value` payloads identical in shape to server.py's, so
//! `ltm-server` only has to expose them.
//!
//! STUB: functions return an `unimplemented` marker. The owning agent wires the
//! pipeline (see PORTING.md → ltm-tools):
//!   search: embed query → Store::search(fetch_k) → optional rerank of the
//!           top candidates → top-k Hits + confidence verdict + hnsw note;
//!   get_document: Store::get_document → Found/Ambiguous/NotFound shape;
//!   stats: Store::stats + embed/rerank health.

use ltm_core::{Config, DocKey, EmbeddingBackend, Reranker, Store};
use serde_json::{json, Value};

pub struct Tools<'a> {
    pub cfg: &'a Config,
    pub store: &'a dyn Store,
    pub embed: &'a dyn EmbeddingBackend,
    pub rerank: &'a dyn Reranker,
}

impl<'a> Tools<'a> {
    pub fn ltm_search(
        &self,
        _query: &str,
        _k: usize,
        _rerank: bool,
        _filter_doc_type: Option<&str>,
        _filter_path_contains: Option<&str>,
    ) -> Value {
        json!({"error": "ltm-tools::ltm_search unimplemented"})
    }

    pub fn ltm_get_document(&self, _key: &DocKey, _max_chars: usize) -> Value {
        json!({"error": "ltm-tools::ltm_get_document unimplemented"})
    }

    pub fn ltm_stats(&self) -> Value {
        json!({"error": "ltm-tools::ltm_stats unimplemented"})
    }
}
