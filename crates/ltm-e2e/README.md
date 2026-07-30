# ltm-e2e — integration / end-to-end harness

The integration harness for the Rust LTM-MCP port. It verifies the **whole
retrieval pipeline** against the **live** services (Postgres + pgvector and
Ollama), mirroring how the Python `server.py` was verified. This is the harness
other crates' implementations are checked against.

It is **not** part of the runtime server (`publish = false`, not a dependency of
`ltm-server`).

## What it does

1. **Seed** (`src/seed.rs` + `src/corpus.rs`): drops and (re)creates the
   `documents` / `doc_chunks` / `emb_mxbai(vector(1024))` schema and inserts a
   small 4-document corpus (loan / lease / tax / insurance style text). Each
   chunk is embedded **raw** (no query prefix) via Ollama `mxbai-embed-large`
   into a 1024-dim vector, and an HNSW cosine index is built. Idempotent —
   re-running yields a clean corpus.

2. **End-to-end assertions** (`src/harness.rs`, `tests/e2e.rs`): builds the real
   `ltm_db::PgStore` + the Ollama `EmbeddingBackend` from `ltm_embed` +
   `ltm_rerank::NoopReranker`, wires them into `ltm_tools::Tools`, and asserts:
   - `ltm_stats` reports the seeded counts (`documents`/`chunks`/
     `mxbai_embeddings` = 4), `ollama` health starting with `ok`, and
     `read_only: true`;
   - `ltm_search("when should I order the appraisal for the loan")` ranks the
     **loan** document first with a cosine clearly higher (margin ≥ 0.1) than
     the top hit of a nonsense query;
   - `ltm_get_document` resolves the loan document both by the unambiguous path
     **fragment** (`mortgage_appraisal_timeline`) and by numeric `doc_id`.

### Design notes

- **Why embed documents directly (not via `ltm-embed`)?** The
  `EmbeddingBackend` trait only exposes `embed_query`, which prepends the mxbai
  asymmetric-retrieval **query** prefix. Corpus documents must be embedded
  **raw** (server.py's asymmetric setup: documents raw, queries prefixed), so
  the seed talks to Ollama directly. The query side still goes through
  `ltm-embed` (with the prefix) via `Tools`.
- **Skip, don't fail.** Every live check is guarded. If a sibling crate is still
  a stub (returns the `unimplemented` marker) or a service is unreachable, the
  harness prints a clear `SKIPPED (...)` message and exits 0 / passes the test
  instead of failing — so it is green both before and after integration.

## Prerequisites

- **Postgres 18** with **pgvector 0.8.5** on `127.0.0.1:5432`, database `brain`,
  user `postgres` (trust auth).
- **Ollama** on `127.0.0.1:11434` with `mxbai-embed-large` pulled.

The server defaults to `LTM_PGPORT=5433`, so set `LTM_PGPORT=5432` for this box
(all other defaults in `ltm_core::Config` already match). Relevant env vars:
`LTM_PGHOST`, `LTM_PGPORT`, `LTM_PGUSER`, `LTM_PGDATABASE`, `LTM_OLLAMA_URL`,
`LTM_EMBED_MODEL`.

## Running

```sh
# 1. Seed only (re)creates the schema + corpus and prints counts:
LTM_PGPORT=5432 cargo run -p ltm-e2e -- seed

# 2. Full harness (seeds, then runs every assertion):
LTM_PGPORT=5432 cargo run -p ltm-e2e            # or: -- e2e

# 3. As a cargo integration test (self-seeds; --nocapture shows PASS/SKIP):
LTM_PGPORT=5432 cargo test -p ltm-e2e -- --nocapture
```

Exit codes for the binary: `0` on **PASS** or **SKIP**, `1` on **FAIL**.

## Build / lint

```sh
cargo build  -p ltm-e2e
cargo clippy -p ltm-e2e --all-targets -- -D warnings
```
