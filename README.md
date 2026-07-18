# LTM MCP Server

An MCP server that lets any MCP-capable AI — Claude Code, Claude Desktop,
LM Studio, and (via a tunnel) cloud AIs like claude.ai and ChatGPT — search
your own local **Long-Term Memory**: a Postgres + pgvector corpus of your
documents, embedded with `mxbai-embed-large` and optionally reranked by a
local cross-encoder.

**Local-only. Read-only. No secrets.** The server talks to Postgres and
Ollama on 127.0.0.1; the Postgres session is forced into read-only
transactions, every query is a parameterized SELECT, and nothing leaves the
machine unless you deliberately put a tunnel in front of it.

## Why

I run a real-estate operation and got tired of my own filing system beating me:
statements, leases, loan docs, taxes, email — terabytes of it, spread across
decades. So I built a local "brain": every document OCR'd, extracted, chunked,
and embedded into Postgres + pgvector (~100K documents in my deployment). This
repo is the connective tissue — the MCP server that lets the AI tools I already
use query that corpus directly, with honest confidence scores so the AI says
"that isn't in the corpus" instead of guessing.

**Status:** in-progress, shared to show the approach. It runs daily against my
corpus; the ingestion pipeline that builds the database is a separate (not yet
published) project — this repo assumes you already have the tables described
below.

## Architecture

```
AI client (Claude Code / Desktop / LM Studio / cloud via tunnel)
        │  MCP over stdio  —or—  streamable HTTP (bearer-gated)
        ▼
server.py
        │ 1. embed query        → Ollama mxbai-embed-large (query prefix)
        │ 2. cosine top-50      → Postgres, emb_mxbai vector(1024) (+HNSW)
        │ 3. rerank to top-k    → mxbai-rerank-base-v2 (persistent worker,
        ▼                          optional — see LTM_RERANK_*)
results + honest confidence verdict
```

Expected schema: `documents` (id, root, relpath, doc_type, title, text, ...),
`doc_chunks` (id, doc_id, text), `emb_mxbai` (chunk_id, embedding
vector(1024)). An HNSW index on `emb_mxbai` is detected per-query — searches
work without it (sequential scan, slower) and speed up the moment it exists.

## Tools

### `ltm_search(query, k=8, rerank=true, doc_type=null, path_contains=null)`
Semantic search. Returns hits with source **path**, **doc_type**, **snippet**,
**cosine** similarity, **rerank_score**, and a **confidence** verdict derived
from the reranker's *absolute* top score:

| top rerank score | verdict |
|---|---|
| ≥ 6 | `high` |
| 3 – 6 | `medium` |
| < 3 | `LOW — answer may not be in the corpus` |

The LOW gate is real: nonsense queries score ~2.6 and get flagged, while a
query the corpus actually answers scores 10+. When a search comes back LOW,
the right behavior is "this doesn't appear to be in the corpus" — not forcing
an answer from weak hits. With `rerank=false` the verdict falls back to a
coarser cosine-based gate and is labeled as such.

### `ltm_get_document(path_or_id, max_chars=20000)`
Returns a document's extracted text **from the database** (the ingestion text
layer, not the raw file) plus metadata. Accepts a numeric `doc_id`, an exact
path, or a path fragment — ambiguous fragments return a candidate list.

### `ltm_stats()`
Corpus counts by type, chunk/embedding coverage, HNSW index status, Ollama
and reranker availability, table sizes.

## Quickstart (stdio, local clients)

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt

# Claude Code:
claude mcp add ltm --scope user -- /path/to/venv/bin/python /path/to/server.py

# Claude Desktop / LM Studio — same shape in their mcpServers config:
#   { "mcpServers": { "ltm": { "command": "/path/to/venv/bin/python",
#                              "args": ["/path/to/server.py"] } } }
```

Prereqs: Postgres with pgvector and the schema above; Ollama running with
`mxbai-embed-large` pulled.

## HTTP mode (remote / cloud AIs)

```bash
python server.py --http    # streamable HTTP on 127.0.0.1:8322
```

Bearer-token gated: a token is minted on first run into `~/.ltm/http-token`
(0600; `LTM_HTTP_TOKEN_FILE` overrides). Clients send
`Authorization: Bearer <token>`, or — for connector UIs with no header field
(claude.ai, ChatGPT) — embed it in the URL path: `https://host/t/<token>/mcp`.
Access logging is disabled so the token is never logged; comparisons are
constant-time; `/healthz` is open and returns only "ok".

The server binds 127.0.0.1 only. Expose it exclusively through an
outbound-only tunnel (Cloudflare Tunnel, Tailscale Funnel) — never a
port-forward. Set `LTM_PUBLIC_HOST` to your tunnel hostname so it passes the
transport-security host check. Hookup steps for claude.ai / ChatGPT / Gemini
CLI: [docs/CONNECT-CLOUD-AIS.md](docs/CONNECT-CLOUD-AIS.md).

## Config (env, all optional)

| Variable | Default | Purpose |
|---|---|---|
| `LTM_PGHOST` / `LTM_PGPORT` | 127.0.0.1 / 5433 | Postgres |
| `LTM_PGUSER` / `LTM_PGDATABASE` | postgres / brain | Postgres |
| `LTM_OLLAMA_URL` | http://127.0.0.1:11434/api/embeddings | embedder |
| `LTM_EMBED_MODEL` | mxbai-embed-large | embedding model |
| `LTM_RERANK_PYTHON` / `LTM_RERANK_SCRIPT` | unset (rerank off) | cross-encoder worker |
| `LTM_RERANK_CANDIDATES` | 50 | rerank pool size |
| `LTM_LOW_RERANK` / `LTM_HIGH_RERANK` | 3 / 6 | confidence gates |
| `LTM_HTTP_HOST` / `LTM_HTTP_PORT` | 127.0.0.1 / 8322 | HTTP mode |
| `LTM_HTTP_TOKEN_FILE` | ~/.ltm/http-token | bearer token file |
| `LTM_PUBLIC_HOST` | unset | public tunnel hostname (HTTP mode) |

## Security posture

- **Read-only by construction**: `default_transaction_read_only = on`; every
  query is a parameterized SELECT; three tools only; no SQL passthrough, no
  file reads, document text capped at `max_chars` ≤ 500K.
- **Bearer required** on everything except `/healthz`. 401 otherwise.
- **No open ports** in the tunnel deployment: the server binds 127.0.0.1; the
  tunnel daemon makes an outbound connection to its edge.
- **Token hygiene**: 0600 token files, access log off, secrets never printed.

## License

MIT — see [LICENSE](LICENSE).
