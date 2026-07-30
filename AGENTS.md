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
  `sudo pg_ctlcluster 18 main start`. The cluster listens on **5432**, but the
  server defaults to **5433**, so run it with `LTM_PGPORT=5432`.
- **Use the latest stable PostgreSQL.** This env runs **PostgreSQL 18** (18.x,
  the current stable major; 19 is still beta until ~Sept 2026) with
  **pgvector 0.8.5**, installed from the official PGDG apt repo — not the older
  distro default (16). Postgres has no "LTS" track; every major gets 5 years of
  support.
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

### Project direction (per maintainer)

- **These are Linux server apps and must be Rust only.** `ltm-mcp` drifting into
  Python was a mistake — it was always meant to be Rust ("Rust only from day
  one"), and **the MCP server in particular must be Rust**. Treat `server.py` as
  the behavioral spec for a Rust reimplementation; write new functionality in
  Rust, not Python.
- **Swift is only for genuinely native Apple client apps** (macOS / iPhone /
  iPad / Apple Silicon viewers) that talk to these servers — never for the
  servers themselves. All heavy lifting, including GUI rendering, stays
  server-side and is exposed over web tech (and optional Apple-native
  viewers/browsers).
- **The model backend must be pluggable/configurable.** Local Ollama
  (llama.cpp) is one option, but OpenRouter, an LM Studio-hosted server, on-prem
  Macs, or cloud model providers must be swappable without disruption. Do not
  hard-wire Ollama into the Rust port.
- **The vector store is PostgreSQL + pgvector and stays** (unless something more
  powerful is warranted), on the latest stable PostgreSQL.
