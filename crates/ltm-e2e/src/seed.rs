//! Seed utility: (re)create the schema and insert the verification corpus.
//!
//! Talks to Postgres directly (a normal read/write connection, since the
//! runtime server's `PgStore` is read-only) and embeds each chunk RAW via
//! Ollama (see [`crate::embed`]). Idempotent: it drops and recreates the three
//! tables on every run, so re-seeding always yields a clean corpus.

use ltm_core::Config;

use crate::corpus::{self};
use crate::embed;

/// What the seed produced.
pub struct SeedReport {
    pub documents: i64,
    pub chunks: i64,
    pub embeddings: i64,
}

/// Why seeding could not complete.
pub enum SeedError {
    /// Postgres or Ollama was unreachable — the harness treats this as SKIP.
    ServiceDown(String),
    /// A genuine failure (bad SQL, unexpected state) — the harness FAILS.
    Other(String),
}

impl std::fmt::Display for SeedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SeedError::ServiceDown(m) => write!(f, "service down: {m}"),
            SeedError::Other(m) => write!(f, "{m}"),
        }
    }
}

/// Postgres connection string from `Config` (trust auth: no password).
pub fn conn_string(cfg: &Config) -> String {
    format!(
        "host={} port={} user={} dbname={} connect_timeout=5",
        cfg.pg_host, cfg.pg_port, cfg.pg_user, cfg.pg_database
    )
}

const SCHEMA_SQL: &str = "\
DROP TABLE IF EXISTS emb_mxbai;
DROP TABLE IF EXISTS doc_chunks;
DROP TABLE IF EXISTS documents;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE documents (
    id           serial PRIMARY KEY,
    source_label text,
    root         text,
    relpath      text,
    mime         text,
    doc_type     text,
    title        text,
    doc_date     date,
    bytes        bigint,
    chunk_count  integer,
    char_count   integer,
    text_method  text,
    ocr          boolean,
    ingested_at  timestamptz DEFAULT now(),
    text         text
);
CREATE TABLE doc_chunks (
    id     serial PRIMARY KEY,
    doc_id integer REFERENCES documents(id),
    text   text
);
CREATE TABLE emb_mxbai (
    chunk_id  integer PRIMARY KEY REFERENCES doc_chunks(id),
    embedding vector(1024)
);";

/// (Re)create the schema and insert the corpus, embedding each chunk via Ollama.
pub fn seed(cfg: &Config) -> Result<SeedReport, SeedError> {
    // 1) Embed every chunk FIRST so an Ollama outage is detected before we
    //    touch the schema (leaving the DB in a clean, already-seeded state).
    let mut vectors: Vec<String> = Vec::with_capacity(corpus::CORPUS.len());
    for doc in corpus::CORPUS {
        let emb = embed::embed_document(cfg, doc.text)
            .map_err(|e| SeedError::ServiceDown(format!("ollama unreachable: {e}")))?;
        if emb.len() != 1024 {
            return Err(SeedError::Other(format!(
                "expected 1024-dim embedding, got {} for {}",
                emb.len(),
                doc.path()
            )));
        }
        vectors.push(embed::to_vector_literal(&emb));
    }

    // 2) Connect (read/write) and rebuild the schema.
    let mut client = postgres::Client::connect(&conn_string(cfg), postgres::NoTls)
        .map_err(|e| SeedError::ServiceDown(format!("postgres unreachable: {e}")))?;

    client
        .batch_execute(SCHEMA_SQL)
        .map_err(|e| SeedError::Other(format!("schema (re)creation failed: {e}")))?;

    // 3) Insert documents, chunks, and embeddings.
    for (doc, vec_lit) in corpus::CORPUS.iter().zip(vectors.iter()) {
        let char_count = doc.text.chars().count() as i32;
        let bytes = doc.text.len() as i64;
        let chunk_count: i32 = 1;
        let ocr = false;

        let doc_id: i32 = client
            .query_one(
                "INSERT INTO documents \
                 (source_label, root, relpath, mime, doc_type, title, doc_date, \
                  bytes, chunk_count, char_count, text_method, ocr, text) \
                 VALUES ($1,$2,$3,$4,$5,$6, CAST($7::text AS date), $8,$9,$10,$11,$12,$13) \
                 RETURNING id",
                &[
                    &doc.source_label,
                    &doc.root,
                    &doc.relpath,
                    &doc.mime,
                    &doc.doc_type,
                    &doc.title,
                    &doc.doc_date,
                    &bytes,
                    &chunk_count,
                    &char_count,
                    &doc.text_method,
                    &ocr,
                    &doc.text,
                ],
            )
            .map_err(|e| SeedError::Other(format!("insert documents failed: {e}")))?
            .get(0);

        let chunk_id: i32 = client
            .query_one(
                "INSERT INTO doc_chunks (doc_id, text) VALUES ($1,$2) RETURNING id",
                &[&doc_id, &doc.text],
            )
            .map_err(|e| SeedError::Other(format!("insert doc_chunks failed: {e}")))?
            .get(0);

        client
            .execute(
                "INSERT INTO emb_mxbai (chunk_id, embedding) VALUES ($1, CAST($2::text AS vector))",
                &[&chunk_id, &vec_lit],
            )
            .map_err(|e| SeedError::Other(format!("insert emb_mxbai failed: {e}")))?;
    }

    // 4) Build the HNSW index so `hnsw_ready()` reports the fast path (the
    //    corpus is tiny, but this exercises the same code path as production).
    client
        .batch_execute(
            "CREATE INDEX IF NOT EXISTS emb_mxbai_hnsw \
             ON emb_mxbai USING hnsw (embedding vector_cosine_ops);",
        )
        .map_err(|e| SeedError::Other(format!("hnsw index creation failed: {e}")))?;

    // 5) Report counts.
    let counts = client
        .query_one(
            "SELECT (SELECT count(*) FROM documents), \
                    (SELECT count(*) FROM doc_chunks), \
                    (SELECT count(*) FROM emb_mxbai)",
            &[],
        )
        .map_err(|e| SeedError::Other(format!("count query failed: {e}")))?;

    Ok(SeedReport {
        documents: counts.get(0),
        chunks: counts.get(1),
        embeddings: counts.get(2),
    })
}
