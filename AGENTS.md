# AGENTS.md

## Cursor Cloud specific instructions

`ltm-mcp` is a read-only MCP server (`server.py`) that does semantic search over
a local Postgres + pgvector corpus, embedding queries with `mxbai-embed-large`
via a local Ollama. Tool/config reference is in `README.md`.

Dependencies (venv at `./venv`) are installed by the startup update script. Run
modes: stdio `venv/bin/python server.py`; HTTP `venv/bin/python server.py --http`
(binds `127.0.0.1:8322`, bearer token minted to `~/.ltm/http-token`, `/healthz`
is open and unauthenticated).

Non-obvious notes for this environment:

- **Pin `mcp<2`.** `requirements.txt` says `mcp>=1.28`, but `mcp` 2.0 removed the
  `mcp.server.fastmcp` import path the code uses. The update script installs
  `mcp<2` (1.29.x) after `requirements.txt` — do not "upgrade" it to 2.x.
- **Postgres is not auto-started (no systemd).** Start it with
  `sudo pg_ctlcluster 16 main start`. The cluster listens on **5432**, but the
  server defaults to **5433**, so run it with `LTM_PGPORT=5432`.
- **Loopback auth is set to `trust`** in `pg_hba.conf` for local dev, because the
  code's `pg8000` connection sends no password. DB is `brain`, user `postgres`,
  with the `vector` extension enabled.
- **Ollama is not auto-started.** Run `ollama serve` (in the background); the
  `mxbai-embed-large` model is already pulled. `ltm_stats` reports Ollama/model
  health.
- **You must seed data yourself.** The ingestion pipeline that builds the corpus
  is a separate, unpublished project; this repo assumes the tables already exist.
  For local verification, create `documents(id, root, relpath, doc_type, title,
  text, source_label, doc_date, text_method, mime, bytes, chunk_count,
  char_count, ocr, ingested_at)`, `doc_chunks(id, doc_id, text)`, and
  `emb_mxbai(chunk_id, embedding vector(1024))`, then insert a few docs whose
  chunk text is embedded through Ollama.
- **The reranker is optional.** With `LTM_RERANK_PYTHON`/`LTM_RERANK_SCRIPT`
  unset, `ltm_search` returns cosine order and the coarser cosine confidence
  gate (raw mxbai cosines cluster lower, so a relevant hit ~0.6 still reads
  "LOW (cosine only)" while nonsense scores ~0.28 — the discrimination is real).

### Project direction: Rust/Swift only

Per the maintainer: this Python server is a **prototype**. The long-term goal is
to port everything to Rust (and Swift for any Apple-platform UI), targeting the
latest iPhone/iPad/macOS technologies. Treat `server.py` as the behavioral spec
for the Rust port; write new functionality in Rust/Swift with a short rationale
for the language choice rather than extending the Python.
