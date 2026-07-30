//! ltm-db — read-only Postgres + pgvector data layer. Implements
//! [`ltm_core::Store`].
//!
//! STUB: `PgStore` currently returns `Error::Db("unimplemented")`. The owning
//! agent implements the real queries (see PORTING.md → ltm-db):
//!   - persistent read-only connection (`SET default_transaction_read_only = on`,
//!     `SET statement_timeout = '180s'`), transparent reconnect;
//!   - cosine top-k over `emb_mxbai` with optional `doc_type` / `path_contains`
//!     filters and HNSW `ef_search` / iterative-scan tuning when the index
//!     exists (mirror server.py's SQL exactly, parameterized);
//!   - `get_document` by id / exact path / fragment (ambiguous → candidates);
//!   - `stats` (counts, by-type, table sizes, hnsw status).

use ltm_core::{ChunkRow, Config, DocKey, DocLookup, Error, Result, SearchFilter, Stats, Store};

pub struct PgStore {
    #[allow(dead_code)]
    cfg: Config,
}

impl PgStore {
    /// Open a connection using the config (host/port/user/database).
    pub fn connect(cfg: &Config) -> Result<Self> {
        // TODO(ltm-db agent): establish the read-only connection here.
        Ok(PgStore { cfg: cfg.clone() })
    }
}

impl Store for PgStore {
    fn search(&self, _emb: &[f32], _fetch_k: usize, _filter: &SearchFilter) -> Result<Vec<ChunkRow>> {
        Err(Error::Db("ltm-db::search unimplemented".into()))
    }
    fn get_document(&self, _key: &DocKey, _max_chars: usize) -> Result<DocLookup> {
        Err(Error::Db("ltm-db::get_document unimplemented".into()))
    }
    fn stats(&self) -> Result<Stats> {
        Err(Error::Db("ltm-db::stats unimplemented".into()))
    }
    fn hnsw_ready(&self) -> Result<bool> {
        Err(Error::Db("ltm-db::hnsw_ready unimplemented".into()))
    }
}
