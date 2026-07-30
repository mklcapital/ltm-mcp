# LTM-MCP → Rust: parallel porting guide

`server.py` is the **behavioral spec**. The Rust port is a Cargo workspace
(`Cargo.toml` → `crates/*`) split so several agents can work in parallel. Each
agent owns exactly one crate and implements the code behind the **frozen trait
contracts in `ltm-core`**. Do not change `ltm-core`'s public interface without
flagging it (it is the shared contract); everything else is yours to build.

## Ground rules

- Keep the app **read-only** and local-first, exactly like `server.py`.
- The `ltm-core` traits (`Store`, `EmbeddingBackend`, `Reranker`) are
  **synchronous/blocking**. Only `ltm-server` is async and bridges via
  `spawn_blocking`.
- Match the JSON shapes `server.py` returns (field names, verdict strings).
- Add only the dependencies your crate needs; do not edit sibling crates.
- Verify with `cargo build -p <your-crate>` and `cargo clippy -p <your-crate>`.
- A live Postgres 18 (`brain`, pgvector 0.8.5) and Ollama (`mxbai-embed-large`)
  are available on the dev box for integration; unit-test against them where
  practical, but keep unit tests hermetic where you can.

## Crate assignments

### 1. `ltm-core` (owner: coordinator) — DONE / FROZEN
Types, `Config::from_env`, error type, verdict/snippet helpers, and the trait
contracts. Change only by coordination.

### 2. `ltm-db` — implement `Store` (Postgres + pgvector)
Spec: `server.py` `db()`, `hnsw_ready()`, and the SQL in `ltm_search`,
`ltm_get_document`, `ltm_stats`.
- Persistent read-only connection: `SET default_transaction_read_only = on`,
  `SET statement_timeout = '180s'`; transparent reconnect on drop.
- `search`: cosine top-`fetch_k` over `emb_mxbai` joined to `doc_chunks` +
  `documents`; optional `doc_type ILIKE` and `(root||'/'||relpath) ILIKE`
  filters; when `hnsw_ready`, `SET hnsw.ef_search = 200` and
  `hnsw.iterative_scan` (`relaxed_order` when filtered else `off`). Return
  `ChunkRow`s including `cosine = 1 - (embedding <=> query)` and `chunk_text`
  (`left(c.text, 1300)`). **Parameterized queries only.**
- `get_document`: numeric id / exact `relpath` or `root||'/'||relpath` / else
  `ILIKE '%frag%'` (≤10 → candidates). Map to `DocLookup`.
- `stats`: counts, `documents_by_type`, `pg_size_pretty` table/db sizes, hnsw.
- Suggested deps: `postgres` (blocking) + `pgvector`, or `tokio-postgres` in a
  small blocking wrapper.

### 3. `ltm-embed` — implement `EmbeddingBackend` (pluggable)
Spec: `server.py` `embed_query`, plus the maintainer's pluggable-backend rule.
- `OllamaBackend`: POST `{model, prompt: query_prefix + text}` to `ollama_url`;
  read `.embedding` → `Vec<f32>`. `health()` hits `/api/tags`.
- `OpenAiBackend`: POST `{model, input: query_prefix + text}` to
  `{embed_base_url}/v1/embeddings` with `Authorization: Bearer {embed_api_key}`;
  read `.data[0].embedding`. Covers OpenRouter / LM Studio / cloud.
- `make_backend` already selects by `Config`. Suggested dep: `ureq`.

### 4. `ltm-rerank` — implement `Reranker` (optional)
Spec: `server.py` persistent rerank worker (`_RERANK_WORKER_SRC`, `run_rerank`).
- `WorkerReranker`: spawn `rerank_python -u -c <src>` importing
  `rerank.Reranker`; JSONL request/response over stdin/stdout; wall-clock
  timeout (`rerank_timeout_s`); kill+retry once on death. `available()` true
  only when `rerank_python`+`rerank_script` are set and exist.
- Keep `NoopReranker` as the default. Suggested deps: none beyond std/serde_json.

### 5. `ltm-tools` — implement the three tools over the traits
Spec: `server.py` `ltm_search`, `ltm_get_document`, `ltm_stats`.
- `ltm_search`: clamp `k` to 1..=50; `use_rerank = rerank && reranker.available()`;
  `fetch_k = max(rerank_candidates, k)` when reranking else `k`; embed → search
  → (rerank top candidates, reorder, scores) → top-k `Hit`s. Confidence via
  `verdict_from_rerank`/`verdict_from_cosine` (already in `ltm-core`). Include
  `reranked`, `top_score`, `hnsw_index`, and `note` on rerank fallback. Empty →
  the LOW/no-match shape.
- `ltm_get_document`: clamp `max_chars` to 200..=500_000; return the
  Found/Ambiguous/NotFound shapes.
- `ltm_stats`: `Store::stats` + `EmbeddingBackend::health` + reranker
  availability + `read_only: true`, `embedding_coverage`, etc.

### 6. `ltm-server` — implement the MCP transports + tool registration
Spec: `server.py` `mcp.run()` (stdio) and `run_http()` (streamable HTTP).
- Register `ltm_search` / `ltm_get_document` / `ltm_stats` with the same
  arg schemas/descriptions; call `ltm-tools` (blocking) via `spawn_blocking`.
- stdio by default; `--http` → streamable HTTP on `http_host:http_port`,
  bearer-gated (`_load_or_create_token` at `http_token_file`, 0600; `/healthz`
  open; accept `Authorization: Bearer` **or** path-embedded `/t/<token>/mcp`;
  constant-time compare; access log off; allow `public_host`).
- Suggested deps: `rmcp` (official Rust MCP SDK) + `tokio` + `axum`/`hyper`.

## Integration

Each crate is merged onto the workspace branch; `ltm-server` wiring already
compiles against the stubs, so integration is: land crates, `cargo build`,
`cargo clippy`, then end-to-end against the live Postgres + Ollama using the
same seed corpus the Python server was verified with.
