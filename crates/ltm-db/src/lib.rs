//! ltm-db — read-only Postgres + pgvector data layer. Implements
//! [`ltm_core::Store`].
//!
//! Mirrors `server.py`'s data access:
//!   - a persistent read-only connection (`SET default_transaction_read_only =
//!     on`, `SET statement_timeout = '180s'`) that transparently reconnects if
//!     the socket drops;
//!   - cosine top-`fetch_k` search over `emb_mxbai` joined to `doc_chunks` and
//!     `documents`, with optional `doc_type` / `path_contains` filters and HNSW
//!     `ef_search` / iterative-scan tuning when the index is present;
//!   - `get_document` by id / exact path / fragment (ambiguous -> candidates);
//!   - `stats` (counts, by-type, table/db sizes, hnsw status).
//!
//! Every query is a parameterized `SELECT`; user input is never interpolated
//! into SQL.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use ltm_core::{
    ChunkRow, Config, DocCandidate, DocKey, DocLookup, Document, Error, Result, SearchFilter,
    Stats, Store,
};
use pgvector::Vector;
use postgres::types::ToSql;
use postgres::{Client, NoTls, Row};

/// How much chunk text a search hit carries (mirrors `RERANK_TEXT_CHARS`).
const RERANK_TEXT_CHARS: i32 = 1300;

pub struct PgStore {
    cfg: Config,
    /// Persistent connection, lazily (re)opened. `postgres::Client` is `Send`
    /// but not `Sync`, so it lives behind a `Mutex` to satisfy `Store: Sync`.
    client: Mutex<Option<Client>>,
    /// Sticky once a valid HNSW index is observed (matches server.py's
    /// `_hnsw_ready` global: cached once true, re-checked while false).
    hnsw_cached: AtomicBool,
}

impl PgStore {
    /// Open a read-only connection using the config (host/port/user/database).
    pub fn connect(cfg: &Config) -> Result<Self> {
        let client = Self::open(cfg)?;
        Ok(PgStore {
            cfg: cfg.clone(),
            client: Mutex::new(Some(client)),
            hnsw_cached: AtomicBool::new(false),
        })
    }

    /// Establish a fresh read-only session.
    fn open(cfg: &Config) -> Result<Client> {
        let conn = format!(
            "host={} port={} user={} dbname={}",
            cfg.pg_host, cfg.pg_port, cfg.pg_user, cfg.pg_database
        );
        let mut client = Client::connect(&conn, NoTls).map_err(|e| Error::Db(e.to_string()))?;
        // Hard read-only: any write attempted in this session errors out.
        client
            .batch_execute(
                "SET default_transaction_read_only = on; SET statement_timeout = '180s';",
            )
            .map_err(|e| Error::Db(e.to_string()))?;
        Ok(client)
    }

    /// Run `f` against a live client, transparently reconnecting if the
    /// connection has dropped. Retries once on a fresh connection.
    fn with_client<T, F>(&self, f: F) -> Result<T>
    where
        F: Fn(&mut Client) -> std::result::Result<T, postgres::Error>,
    {
        let mut guard = self
            .client
            .lock()
            .map_err(|e| Error::Db(format!("connection mutex poisoned: {e}")))?;
        let mut last_err: Option<postgres::Error> = None;
        for attempt in 0..2 {
            if guard.is_none() {
                *guard = Some(Self::open(&self.cfg)?);
            } else if attempt == 0 {
                // Cheap liveness probe; reconnect if the socket is dead.
                if guard
                    .as_mut()
                    .unwrap()
                    .is_valid(Duration::from_secs(5))
                    .is_err()
                {
                    *guard = Some(Self::open(&self.cfg)?);
                }
            }
            let client = guard.as_mut().unwrap();
            match f(client) {
                Ok(v) => return Ok(v),
                Err(e) => {
                    // Drop the (possibly broken) connection so the next attempt
                    // reconnects from scratch.
                    *guard = None;
                    last_err = Some(e);
                }
            }
        }
        Err(Error::Db(
            last_err
                .map(|e| e.to_string())
                .unwrap_or_else(|| "query failed".into()),
        ))
    }
}

/// Columns selected by `get_document`, in a fixed order (see `row_to_document`).
const DOC_SELECT: &str = "SELECT id::bigint, source_label, root, relpath, mime, doc_type, title, \
     doc_date::text, bytes, chunk_count::bigint, char_count::bigint, text_method, ocr, \
     ingested_at::text, text FROM documents ";

fn join_path(root: Option<&str>, relpath: Option<&str>) -> String {
    format!("{}/{}", root.unwrap_or(""), relpath.unwrap_or(""))
}

fn or_untyped(doc_type: Option<String>) -> String {
    match doc_type {
        Some(s) if !s.is_empty() => s,
        _ => "untyped".to_string(),
    }
}

fn row_to_document(row: &Row, max_chars: usize) -> Document {
    let root: Option<String> = row.get(2);
    let relpath: Option<String> = row.get(3);
    let text: String = row.get::<_, Option<String>>(14).unwrap_or_default();
    // Truncate by unicode scalar values, matching Python's str slicing.
    let text = text.chars().take(max_chars).collect::<String>();
    Document {
        doc_id: row.get(0),
        path: join_path(root.as_deref(), relpath.as_deref()),
        source: row.get(1),
        mime: row.get(4),
        doc_type: or_untyped(row.get(5)),
        title: row.get(6),
        doc_date: row.get(7),
        bytes: row.get(8),
        chunk_count: row.get(9),
        char_count: row.get(10),
        extraction: row.get(11),
        ocr: row.get::<_, Option<bool>>(12).unwrap_or(false),
        ingested_at: row.get(13),
        text,
    }
}

fn row_to_candidate(row: &Row) -> DocCandidate {
    let root: Option<String> = row.get(2);
    let relpath: Option<String> = row.get(3);
    DocCandidate {
        doc_id: row.get(0),
        path: join_path(root.as_deref(), relpath.as_deref()),
        doc_type: or_untyped(row.get(5)),
        title: row.get(6),
        char_count: row.get(10),
    }
}

impl Store for PgStore {
    fn search(&self, emb: &[f32], fetch_k: usize, filter: &SearchFilter) -> Result<Vec<ChunkRow>> {
        // Resolve HNSW status first (own connection lock), before we take the
        // lock for the search itself — the std Mutex is not reentrant.
        let indexed = self.hnsw_ready()?;
        let filtered = filter.doc_type.is_some() || filter.path_contains.is_some();

        let qvec = Vector::from(emb.to_vec());
        let limit = fetch_k as i64;
        let pc_pattern = filter.path_contains.as_ref().map(|p| format!("%{p}%"));
        let doc_type = filter.doc_type.clone();

        self.with_client(move |client| {
            if indexed {
                client.batch_execute("SET hnsw.ef_search = 200")?;
                // With WHERE filters an HNSW scan can under-return; iterative
                // scan (pgvector >= 0.8) keeps scanning until LIMIT is met.
                let scan = if filtered { "relaxed_order" } else { "off" };
                client.batch_execute(&format!("SET hnsw.iterative_scan = {scan}"))?;
            }

            // $1 is always the query vector (referenced twice). Filter params
            // and the LIMIT are appended and numbered dynamically.
            let mut params: Vec<&(dyn ToSql + Sync)> = vec![&qvec];
            let mut clauses: Vec<String> = Vec::new();
            let mut n = 2;
            if let Some(dt) = doc_type.as_ref() {
                clauses.push(format!("d.doc_type ILIKE ${n}"));
                params.push(dt);
                n += 1;
            }
            if let Some(pc) = pc_pattern.as_ref() {
                clauses.push(format!("(d.root || '/' || d.relpath) ILIKE ${n}"));
                params.push(pc);
                n += 1;
            }
            let where_sql = if clauses.is_empty() {
                String::new()
            } else {
                format!("WHERE {} ", clauses.join(" AND "))
            };
            params.push(&limit);

            let sql = format!(
                "SELECT c.id::bigint, d.id::bigint, d.doc_type, d.title, d.source_label, \
                 d.root, d.relpath, d.doc_date::text, d.text_method, \
                 1 - (e.embedding <=> $1::vector) AS cosine, \
                 left(c.text, {RERANK_TEXT_CHARS}) AS chunk_text \
                 FROM emb_mxbai e \
                 JOIN doc_chunks c ON c.id = e.chunk_id \
                 JOIN documents d ON d.id = c.doc_id \
                 {where_sql}\
                 ORDER BY e.embedding <=> $1::vector \
                 LIMIT ${n}"
            );

            let rows = client.query(&sql, &params)?;
            Ok(rows
                .iter()
                .map(|r| ChunkRow {
                    chunk_id: r.get(0),
                    doc_id: r.get(1),
                    doc_type: r.get(2),
                    title: r.get(3),
                    source_label: r.get(4),
                    root: r.get(5),
                    relpath: r.get(6),
                    doc_date: r.get(7),
                    text_method: r.get(8),
                    cosine: r.get(9),
                    chunk_text: r.get::<_, Option<String>>(10).unwrap_or_default(),
                })
                .collect())
        })
    }

    fn get_document(&self, key: &DocKey, max_chars: usize) -> Result<DocLookup> {
        let rows = self.with_client(|client| match key {
            DocKey::Id(id) => client.query(&format!("{DOC_SELECT}WHERE id = $1::bigint"), &[id]),
            DocKey::PathOrFragment(raw) => {
                let key = raw.trim().to_string();
                let exact = client.query(
                    &format!("{DOC_SELECT}WHERE relpath = $1 OR (root || '/' || relpath) = $1"),
                    &[&key],
                )?;
                if exact.is_empty() {
                    let pat = format!("%{key}%");
                    client.query(
                        &format!(
                            "{DOC_SELECT}WHERE (root || '/' || relpath) ILIKE $1 \
                             ORDER BY char_count DESC NULLS LAST LIMIT 10"
                        ),
                        &[&pat],
                    )
                } else {
                    Ok(exact)
                }
            }
        })?;

        match rows.len() {
            0 => Ok(DocLookup::NotFound),
            1 => Ok(DocLookup::Found(Box::new(row_to_document(&rows[0], max_chars)))),
            _ => Ok(DocLookup::Ambiguous {
                candidates: rows.iter().map(row_to_candidate).collect(),
            }),
        }
    }

    fn stats(&self) -> Result<Stats> {
        let indexed = self.hnsw_ready()?;
        self.with_client(move |client| {
            let documents: i64 = client.query_one("SELECT count(*) FROM documents", &[])?.get(0);
            let chunks: i64 = client.query_one("SELECT count(*) FROM doc_chunks", &[])?.get(0);
            let mxbai_embeddings: i64 =
                client.query_one("SELECT count(*) FROM emb_mxbai", &[])?.get(0);

            let by_type_rows = client.query(
                "SELECT COALESCE(NULLIF(doc_type, ''), 'untyped'), count(*) \
                 FROM documents GROUP BY 1 ORDER BY 2 DESC",
                &[],
            )?;
            let documents_by_type = by_type_rows
                .iter()
                .map(|r| (r.get::<_, String>(0), r.get::<_, i64>(1)))
                .collect();

            let sizes = client.query_one(
                "SELECT pg_size_pretty(pg_total_relation_size('documents')), \
                 pg_size_pretty(pg_total_relation_size('doc_chunks')), \
                 pg_size_pretty(pg_total_relation_size('emb_mxbai')), \
                 pg_size_pretty(pg_database_size(current_database()))",
                &[],
            )?;
            let mut table_sizes = std::collections::BTreeMap::new();
            table_sizes.insert("documents".to_string(), sizes.get::<_, String>(0));
            table_sizes.insert("doc_chunks".to_string(), sizes.get::<_, String>(1));
            table_sizes.insert("emb_mxbai".to_string(), sizes.get::<_, String>(2));
            table_sizes.insert("database".to_string(), sizes.get::<_, String>(3));

            Ok(Stats {
                documents,
                chunks,
                mxbai_embeddings,
                documents_by_type,
                hnsw_ready: indexed,
                table_sizes,
            })
        })
    }

    fn hnsw_ready(&self) -> Result<bool> {
        if self.hnsw_cached.load(Ordering::Relaxed) {
            return Ok(true);
        }
        let ready = self.with_client(|client| {
            let rows = client.query(
                "SELECT idx.indisvalid FROM pg_index idx \
                 JOIN pg_class i ON i.oid = idx.indexrelid \
                 JOIN pg_class t ON t.oid = idx.indrelid \
                 JOIN pg_namespace n ON n.oid = t.relnamespace \
                 JOIN pg_am am ON am.oid = i.relam \
                 WHERE n.nspname = 'public' AND t.relname = 'emb_mxbai' AND am.amname = 'hnsw'",
                &[],
            )?;
            Ok(rows
                .iter()
                .any(|r| r.get::<_, Option<bool>>(0).unwrap_or(false)))
        })?;
        if ready {
            self.hnsw_cached.store(true, Ordering::Relaxed);
        }
        Ok(ready)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a test config from `LTM_*` env (so `LTM_PGPORT=5432` targets the
    /// live dev DB). Skips the test gracefully when the DB is unreachable.
    fn store_or_skip() -> Option<PgStore> {
        match PgStore::connect(&Config::from_env()) {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("skipping ltm-db live test — no Postgres: {e}");
                None
            }
        }
    }

    #[test]
    fn live_stats_and_search() {
        let Some(store) = store_or_skip() else {
            return;
        };

        // stats: counts should be internally consistent and non-negative.
        let stats = store.stats().expect("stats query");
        eprintln!(
            "documents={} chunks={} embeddings={} hnsw_ready={}",
            stats.documents, stats.chunks, stats.mxbai_embeddings, stats.hnsw_ready
        );
        assert!(stats.documents >= 0);
        assert!(stats.chunks >= 0);
        assert!(stats.table_sizes.contains_key("database"));

        // search: a hand-made 1024-dim unit-ish vector is enough to exercise
        // the join + cosine ordering path without needing Ollama.
        let qvec: Vec<f32> = (0..1024).map(|i| ((i % 7) as f32) * 0.01 + 0.1).collect();
        let rows = store
            .search(&qvec, 5, &SearchFilter::default())
            .expect("search query");
        eprintln!("search returned {} rows", rows.len());
        // Cosine should be within [-1, 1] and rows ordered best-first (desc).
        for w in rows.windows(2) {
            assert!(w[0].cosine >= w[1].cosine - 1e-9, "rows not cosine-ordered");
        }
        for r in &rows {
            assert!(r.cosine <= 1.0 + 1e-6 && r.cosine >= -1.0 - 1e-6);
            assert!(r.chunk_text.chars().count() <= RERANK_TEXT_CHARS as usize);
        }

        // A filter that matches nothing must yield zero rows (parameterized).
        let none = store
            .search(
                &qvec,
                5,
                &SearchFilter {
                    doc_type: Some("this-type-does-not-exist-xyz".into()),
                    path_contains: None,
                },
            )
            .expect("filtered search query");
        assert!(none.is_empty(), "bogus doc_type filter should match nothing");

        // get_document: an id surfaced by search must resolve to Found; a
        // nonsense fragment must resolve to NotFound.
        if let Some(first) = rows.first() {
            match store
                .get_document(&DocKey::Id(first.doc_id), 20_000)
                .expect("get_document by id")
            {
                DocLookup::Found(doc) => {
                    assert_eq!(doc.doc_id, first.doc_id);
                    eprintln!("get_document Found doc_id={} path={}", doc.doc_id, doc.path);
                }
                other => panic!("expected Found, got {other:?}"),
            }
        }
        match store
            .get_document(
                &DocKey::PathOrFragment("no-such-fragment-zzz-qqq".into()),
                20_000,
            )
            .expect("get_document by fragment")
        {
            DocLookup::NotFound => {}
            other => panic!("expected NotFound, got {other:?}"),
        }
    }
}
