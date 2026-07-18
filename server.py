#!/usr/bin/env python3
"""LTM MCP server — read-only semantic search over a local "Long-Term Memory"
document corpus.

Exposes a local corpus (Postgres + pgvector, mxbai-embed-large 1024-dim
embeddings via local Ollama) to any MCP-capable AI client (Claude Code,
Claude Desktop, LM Studio, ...) over stdio.

Tools:
    ltm_search(query, k=8, rerank=True, doc_type=None, path_contains=None)
    ltm_get_document(path_or_id, max_chars=20000)
    ltm_stats()

Strictly READ-ONLY: the Postgres session is forced into read-only transactions
(`default_transaction_read_only = on`), every query is a SELECT, and all user
input goes through parameterized SQL (pg8000 native :params) — never string
interpolation.

Retrieval pipeline:
    1. Embed the query with mxbai-embed-large via local Ollama, using the
       model-card query prefix ("Represent this sentence for searching
       relevant passages: ..."). Documents were embedded raw (asymmetric).
    2. pgvector cosine search over emb_mxbai (vector(1024)). If the HNSW index
       on public.emb_mxbai exists and is valid, hnsw.ef_search is set (and
       iterative scan enabled when filters are applied); otherwise the query
       still works via sequential scan, just slower (~10 s).
    3. Optional cross-encoder rerank: top-RERANK_CANDIDATES (50) candidates are
       rescored by mxbai-rerank-base-v2 (a rerank.py worker in its own venv;
       see LTM_RERANK_* env vars) and the top k returned. The reranker's
       ABSOLUTE top score
       gates confidence: below ~LOW_RERANK the answer is likely not in the
       corpus, and the result says so honestly.

Everything is local: Postgres on 127.0.0.1, Ollama on 127.0.0.1:11434,
reranker as a local subprocess. Nothing leaves the machine. No secrets.

Transports:
    python server.py           stdio (local clients: Claude Code/Desktop, LM Studio)
    python server.py --http    streamable HTTP on 127.0.0.1:8322 (LTM_HTTP_HOST/PORT),
                               bearer-token gated, for remote/cloud AIs via a
                               Cloudflare Tunnel or similar. The token lives in
                               ~/.ltm/http-token (auto-generated 0600;
                               LTM_HTTP_TOKEN_FILE overrides). Clients present it
                               either as  Authorization: Bearer <token>  or, for
                               connector UIs with no header field, embedded in the
                               URL path:  https://host/t/<token>/mcp
                               (the prefix is stripped before MCP routing; access
                               logging is disabled so the token is never logged).
"""

import json
import logging
import os
import re
import subprocess
import sys
import urllib.request

import pg8000.native
from mcp.server.fastmcp import FastMCP

# ----------------------------------------------------------------------------
# Configuration (env-overridable; defaults match the brain deployment)
# ----------------------------------------------------------------------------
PGHOST = os.environ.get("LTM_PGHOST", "127.0.0.1")
PGPORT = int(os.environ.get("LTM_PGPORT", "5433"))
PGUSER = os.environ.get("LTM_PGUSER", "postgres")
PGDATABASE = os.environ.get("LTM_PGDATABASE", "brain")

OLLAMA_EMBED_URL = os.environ.get(
    "LTM_OLLAMA_URL", "http://127.0.0.1:11434/api/embeddings")
EMBED_MODEL = os.environ.get("LTM_EMBED_MODEL", "mxbai-embed-large")
# mxbai model card: documents embedded raw, queries get this exact prefix.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Optional cross-encoder reranker: point these at a python interpreter (in a
# venv with the rerank model deps) and a rerank.py exposing class Reranker.
# Leave unset to run without reranking (cosine order + coarser confidence).
RERANK_PYTHON = os.environ.get("LTM_RERANK_PYTHON", "")
RERANK_SCRIPT = os.environ.get("LTM_RERANK_SCRIPT", "")
RERANK_CANDIDATES = int(os.environ.get("LTM_RERANK_CANDIDATES", "50"))
RERANK_TIMEOUT_S = int(os.environ.get("LTM_RERANK_TIMEOUT_S", "300"))

# Confidence gates. Reranker scores are absolute relevance logits from
# mxbai-rerank-base-v2: below LOW_RERANK the best hit is probably noise —
# the answer is likely NOT in the corpus. Cosine similarity (no rerank) is
# far less discriminative (everything clusters ~0.75-0.90), so its gate is
# coarser and the verdict is labeled accordingly.
LOW_RERANK = float(os.environ.get("LTM_LOW_RERANK", "3.0"))
HIGH_RERANK = float(os.environ.get("LTM_HIGH_RERANK", "6.0"))
LOW_COSINE = float(os.environ.get("LTM_LOW_COSINE", "0.78"))
HIGH_COSINE = float(os.environ.get("LTM_HIGH_COSINE", "0.855"))

SNIPPET_CHARS = 350          # what a search hit shows
RERANK_TEXT_CHARS = 1300     # how much chunk text the cross-encoder sees
MAX_K = 50

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="ltm-mcp %(levelname)s %(message)s")
log = logging.getLogger("ltm")

mcp = FastMCP("ltm")

# ----------------------------------------------------------------------------
# Postgres (read-only, persistent connection with reconnect)
# ----------------------------------------------------------------------------
_con = None
_hnsw_ready = False  # sticky once True; re-checked while False (auto-build soon)


def db():
    """Persistent read-only connection; transparently reconnects if dropped."""
    global _con
    if _con is not None:
        try:
            _con.run("SELECT 1")
            return _con
        except Exception:
            _con = None
    con = pg8000.native.Connection(
        host=PGHOST, port=PGPORT, user=PGUSER, database=PGDATABASE)
    # Hard read-only: any write attempted in this session errors out.
    con.run("SET default_transaction_read_only = on")
    con.run("SET statement_timeout = '180s'")
    _con = con
    return con


def hnsw_ready(con):
    """True once a valid HNSW index exists on public.emb_mxbai.

    The index may not exist yet (auto-builds soon); until then searches fall
    back to a sequential scan, which is correct but slow (~10 s).
    """
    global _hnsw_ready
    if _hnsw_ready:
        return True
    rows = con.run(
        "SELECT idx.indisvalid FROM pg_index idx "
        "JOIN pg_class i ON i.oid = idx.indexrelid "
        "JOIN pg_class t ON t.oid = idx.indrelid "
        "JOIN pg_namespace n ON n.oid = t.relnamespace "
        "JOIN pg_am am ON am.oid = i.relam "
        "WHERE n.nspname = 'public' AND t.relname = 'emb_mxbai' "
        "AND am.amname = 'hnsw'")
    _hnsw_ready = any(r[0] for r in rows)
    return _hnsw_ready


# ----------------------------------------------------------------------------
# Embedding + rerank helpers
# ----------------------------------------------------------------------------
def embed_query(text):
    """Embed a query with the asymmetric-retrieval prefix (mxbai model card)."""
    body = json.dumps({"model": EMBED_MODEL,
                       "prompt": QUERY_PREFIX + text}).encode()
    req = urllib.request.Request(
        OLLAMA_EMBED_URL, data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        emb = json.load(r)["embedding"]
    return "[" + ",".join(f"{x:.6f}" for x in emb) + "]"


def rerank_available():
    return bool(RERANK_PYTHON) and bool(RERANK_SCRIPT) \
        and os.path.exists(RERANK_PYTHON) and os.path.exists(RERANK_SCRIPT)


# Persistent rerank worker: a long-lived subprocess in the rerank venv that
# imports rerank.Reranker ONCE (model load ~15-30 s) and then serves JSONL
# requests over its stdin/stdout for the life of this MCP server. Without
# this, every search would pay the model load again (40-80 s per call).
_RERANK_WORKER_SRC = r"""
import json, sys
sys.path.insert(0, {rerank_dir!r})
from rerank import Reranker
rr = Reranker()
rr._load()
print(json.dumps({{"ready": True}}), flush=True)
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        order, scores = rr.rerank(req["query"], req["texts"], req.get("top_k"))
        print(json.dumps({{"order": order, "scores": scores}}), flush=True)
    except Exception as exc:
        print(json.dumps({{"error": str(exc)}}), flush=True)
"""

_rr_proc = None


def _rr_read_line(proc, timeout):
    """Read one JSONL response, enforcing a wall-clock timeout via select."""
    import select
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    if not ready:
        raise TimeoutError(f"reranker did not answer within {timeout}s")
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("reranker worker exited unexpectedly")
    return json.loads(line)


def _rr_worker():
    global _rr_proc
    if _rr_proc is not None and _rr_proc.poll() is None:
        return _rr_proc
    src = _RERANK_WORKER_SRC.format(
        rerank_dir=os.path.dirname(os.path.abspath(RERANK_SCRIPT)))
    env = dict(os.environ, TQDM_DISABLE="1")
    log.info("starting rerank worker (model load takes ~15-30 s)...")
    proc = subprocess.Popen(
        [RERANK_PYTHON, "-u", "-c", src],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=sys.stderr, text=True, env=env)
    hello = _rr_read_line(proc, RERANK_TIMEOUT_S)
    if not hello.get("ready"):
        proc.kill()
        raise RuntimeError(f"rerank worker failed to start: {hello}")
    log.info("rerank worker ready")
    _rr_proc = proc
    return proc


def run_rerank(query, texts, top_k):
    """Cross-encode (query, text) pairs on the persistent rerank worker.

    Returns (order, scores): indices into `texts` best-first + absolute
    relevance scores. Retries once on a fresh worker if the old one died.
    """
    global _rr_proc
    payload = json.dumps({"query": query, "texts": texts, "top_k": top_k})
    for attempt in (1, 2):
        proc = _rr_worker()
        try:
            proc.stdin.write(payload + "\n")
            proc.stdin.flush()
            res = _rr_read_line(proc, RERANK_TIMEOUT_S)
            if "error" in res:
                raise RuntimeError(res["error"])
            return res["order"], res["scores"]
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
            _rr_proc = None
            if attempt == 2:
                raise
            log.warning("rerank worker died, retrying on a fresh one")


def verdict_from_rerank(top_score):
    if top_score >= HIGH_RERANK:
        return "high"
    if top_score >= LOW_RERANK:
        return "medium"
    return "LOW — answer may not be in the corpus"


def verdict_from_cosine(top_score):
    if top_score >= HIGH_COSINE:
        return "high (cosine only — enable rerank for a calibrated verdict)"
    if top_score >= LOW_COSINE:
        return "medium (cosine only — enable rerank for a calibrated verdict)"
    return "LOW — answer may not be in the corpus (cosine only)"


def clean_snippet(text, limit=SNIPPET_CHARS):
    return re.sub(r"\s+", " ", text or "").strip()[:limit]


# ----------------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------------
@mcp.tool()
def ltm_search(query: str, k: int = 8, rerank: bool = True,
               doc_type: str | None = None,
               path_contains: str | None = None) -> dict:
    """Semantic search over the Long-Term Memory corpus (~100K local documents:
    property management, loans, legal, banking, leases, taxes, email, ...).

    Vector search (mxbai-embed-large + pgvector cosine), then an optional local
    cross-encoder rerank of the top 50 candidates. Each hit includes the source
    file path, doc type, snippet, and scores. `confidence` is an honest gate:
    "LOW" means the top hit scored so poorly the answer is likely NOT in the
    corpus — say so rather than forcing an answer from weak hits.

    Args:
        query: Natural-language search query.
        k: Number of results to return (1-50, default 8).
        rerank: Rerank top-50 with a local cross-encoder (default True;
            better ordering + calibrated confidence; first call loads the
            model, ~15-30 s extra).
        doc_type: Optional filter. One of: bank statement, lease, invoice,
            legal, email, insurance, tax return, loan, appraisal, financials.
        path_contains: Optional case-insensitive substring filter on the
            source file path (e.g. "Taxes" or "Archive").
    """
    k = max(1, min(int(k), MAX_K))
    con = db()
    qvec = embed_query(query)

    use_rerank = bool(rerank) and rerank_available()
    rerank_note = None
    if rerank and not use_rerank:
        rerank_note = ("rerank requested but reranker unavailable "
                       "(LTM_RERANK_PYTHON / LTM_RERANK_SCRIPT not set or "
                       "not found) — cosine order only")

    fetch_k = max(RERANK_CANDIDATES, k) if use_rerank else k
    filtered = doc_type is not None or path_contains is not None

    indexed = hnsw_ready(con)
    if indexed:
        con.run("SET hnsw.ef_search = 200")
        # With WHERE filters an HNSW scan can under-return; iterative scan
        # (pgvector >= 0.8) keeps scanning until LIMIT is satisfied.
        con.run("SET hnsw.iterative_scan = "
                + ("relaxed_order" if filtered else "off"))

    where, params = [], {"e": qvec, "k": fetch_k}
    if doc_type is not None:
        where.append("d.doc_type ILIKE :dt")
        params["dt"] = doc_type
    if path_contains is not None:
        where.append("(d.root || '/' || d.relpath) ILIKE :pc")
        params["pc"] = "%" + path_contains + "%"
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    rows = con.run(
        "SELECT c.id, d.id, d.doc_type, d.title, d.source_label, "
        "d.root, d.relpath, d.doc_date, d.text_method, "
        "1 - (e.embedding <=> CAST(:e AS vector)) AS cosine, "
        "left(c.text, " + str(RERANK_TEXT_CHARS) + ") "
        "FROM emb_mxbai e "
        "JOIN doc_chunks c ON c.id = e.chunk_id "
        "JOIN documents d ON d.id = c.doc_id "
        + where_sql + " "
        "ORDER BY e.embedding <=> CAST(:e AS vector) "
        "LIMIT :k", **params)

    if not rows:
        return {"query": query, "results": [],
                "confidence": "LOW — no matches (filters may be too narrow)",
                "hnsw_index": "ready" if indexed else "not built yet (seq scan)"}

    rr_scores = None
    if use_rerank:
        try:
            order, scores = run_rerank(query, [r[10] or "" for r in rows], k)
            rows = [rows[i] for i in order]
            rr_scores = scores
        except Exception as exc:  # degrade, don't fail the search
            log.warning("rerank failed, falling back to cosine order: %s", exc)
            rerank_note = f"rerank failed ({type(exc).__name__}) — cosine order only"
            use_rerank = False

    rows = rows[:k]
    results = []
    for i, r in enumerate(rows):
        (chunk_id, doc_id, dtype, title, src_label, root, relpath,
         doc_date, text_method, cosine, chunk_text) = r
        item = {
            "path": (root or "") + "/" + (relpath or ""),
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "title": title,
            "doc_type": dtype or "untyped",
            "source": src_label,
            "doc_date": str(doc_date) if doc_date else None,
            "extraction": text_method,
            "cosine": round(float(cosine), 4),
            "snippet": clean_snippet(chunk_text),
        }
        if rr_scores is not None:
            item["rerank_score"] = rr_scores[i]
        results.append(item)

    if rr_scores is not None:
        confidence = verdict_from_rerank(rr_scores[0])
        top = {"rerank_score": rr_scores[0]}
    else:
        confidence = verdict_from_cosine(float(rows[0][9]))
        top = {"cosine": round(float(rows[0][9]), 4)}

    out = {"query": query, "reranked": rr_scores is not None,
           "confidence": confidence, "top_score": top,
           "hnsw_index": "ready" if indexed else "not built yet (seq scan, slower)",
           "results": results}
    if rerank_note:
        out["note"] = rerank_note
    return out


@mcp.tool()
def ltm_get_document(path_or_id: str, max_chars: int = 20000) -> dict:
    """Fetch a document's full extracted text from the LTM database (the text
    layer stored at ingestion — not the raw file on disk).

    Args:
        path_or_id: A numeric document id (as returned by ltm_search's
            doc_id), an exact relative path, or a path fragment. If a fragment
            matches several documents, a candidate list is returned instead
            of text — pick one and call again with its id.
        max_chars: Cap on returned text (default 20000). The response says
            whether it was truncated and what the full length is.
    """
    max_chars = max(200, min(int(max_chars), 500_000))
    con = db()
    sel = ("SELECT id, source_label, root, relpath, mime, doc_type, title, "
           "doc_date, bytes, chunk_count, char_count, text_method, ocr, "
           "ingested_at, text FROM documents ")

    key = str(path_or_id).strip()
    if key.isdigit():
        rows = con.run(sel + "WHERE id = :v", v=int(key))
    else:
        rows = con.run(
            sel + "WHERE relpath = :v OR (root || '/' || relpath) = :v",
            v=key)
        if not rows:
            rows = con.run(
                sel + "WHERE (root || '/' || relpath) ILIKE :v "
                "ORDER BY char_count DESC NULLS LAST LIMIT 10",
                v="%" + key + "%")

    if not rows:
        return {"found": False,
                "error": f"no document matches {key!r} (id, path, or fragment)"}

    if len(rows) > 1:
        return {"found": False,
                "ambiguous": True,
                "candidates": [
                    {"doc_id": r[0],
                     "path": (r[2] or "") + "/" + (r[3] or ""),
                     "doc_type": r[5] or "untyped",
                     "title": r[6],
                     "char_count": r[10]} for r in rows],
                "hint": "multiple matches — call again with one doc_id"}

    (doc_id, src, root, relpath, mime, dtype, title, doc_date, nbytes,
     chunks, chars, method, ocr, ingested, text) = rows[0]
    text = text or ""
    truncated = len(text) > max_chars
    return {
        "found": True,
        "doc_id": doc_id,
        "path": (root or "") + "/" + (relpath or ""),
        "source": src,
        "mime": mime,
        "doc_type": dtype or "untyped",
        "title": title,
        "doc_date": str(doc_date) if doc_date else None,
        "bytes": nbytes,
        "chunk_count": chunks,
        "char_count": chars,
        "extraction": method,
        "ocr": bool(ocr),
        "ingested_at": str(ingested) if ingested else None,
        "truncated": truncated,
        "text": text[:max_chars],
    }


@mcp.tool()
def ltm_stats() -> dict:
    """Corpus health: document counts by type, chunk/embedding coverage, and
    vector index status (whether searches are fast HNSW or slow seq scan)."""
    con = db()
    n_docs = con.run("SELECT count(*) FROM documents")[0][0]
    n_chunks = con.run("SELECT count(*) FROM doc_chunks")[0][0]
    n_emb = con.run("SELECT count(*) FROM emb_mxbai")[0][0]
    by_type = con.run(
        "SELECT COALESCE(NULLIF(doc_type, ''), 'untyped'), count(*) "
        "FROM documents GROUP BY 1 ORDER BY 2 DESC")
    sizes = con.run(
        "SELECT pg_size_pretty(pg_total_relation_size('documents')), "
        "pg_size_pretty(pg_total_relation_size('doc_chunks')), "
        "pg_size_pretty(pg_total_relation_size('emb_mxbai')), "
        "pg_size_pretty(pg_database_size(current_database()))")[0]
    indexed = hnsw_ready(con)

    ollama = "unreachable"
    try:
        with urllib.request.urlopen(
                OLLAMA_EMBED_URL.rsplit("/api/", 1)[0] + "/api/tags",
                timeout=5) as r:
            models = [m["name"] for m in json.load(r)["models"]]
            ollama = ("ok (" + EMBED_MODEL + " present)"
                      if any(m.startswith(EMBED_MODEL) for m in models)
                      else "ok, but " + EMBED_MODEL + " NOT pulled")
    except Exception as exc:
        ollama = f"unreachable ({type(exc).__name__})"

    return {
        "documents": n_docs,
        "chunks": n_chunks,
        "mxbai_embeddings": n_emb,
        "embedding_coverage": f"{(n_emb / n_chunks * 100):.2f}%" if n_chunks else "n/a",
        "documents_by_type": {t: c for t, c in by_type},
        "hnsw_index": ("ready — fast approximate search"
                       if indexed else
                       "not built yet — searches fall back to sequential "
                       "scan (~10 s each); auto-build pending"),
        "embed_model": EMBED_MODEL,
        "ollama": ollama,
        "reranker": ("available" if rerank_available()
                     else "unavailable (set LTM_RERANK_PYTHON / "
                          "LTM_RERANK_SCRIPT)"),
        "table_sizes": {"documents": sizes[0], "doc_chunks": sizes[1],
                        "emb_mxbai": sizes[2], "database": sizes[3]},
        "read_only": True,
    }


# ----------------------------------------------------------------------------
# HTTP transport (Phase 2): streamable HTTP on localhost, bearer-token gated.
# Exposed to the internet ONLY through a Cloudflare Tunnel that points at
# 127.0.0.1:8322 — the server itself never binds a public interface.
# ----------------------------------------------------------------------------
HTTP_HOST = os.environ.get("LTM_HTTP_HOST", "127.0.0.1")
HTTP_PORT = int(os.environ.get("LTM_HTTP_PORT", "8322"))
TOKEN_FILE = os.environ.get(
    "LTM_HTTP_TOKEN_FILE", os.path.expanduser("~/.ltm/http-token"))


def _load_or_create_token():
    """Read the bearer token; mint a 32-byte one (0600) on first run.

    The token value itself is NEVER logged — only the file path.
    """
    import secrets
    d = os.path.dirname(TOKEN_FILE)
    os.makedirs(d, mode=0o700, exist_ok=True)
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE) as f:
            tok = f.read().strip()
        if tok:
            return tok
    tok = secrets.token_urlsafe(32)  # 32 random bytes, urlsafe-encoded
    fd = os.open(TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(tok + "\n")
    log.info("minted new bearer token at %s", TOKEN_FILE)
    return tok


class _BearerGate:
    """Pure-ASGI auth gate in front of the MCP streamable-HTTP app.

    Accepts EITHER
        Authorization: Bearer <token>
    OR a token embedded as the first path segment (for connector UIs that
    cannot send custom headers, e.g. claude.ai custom connectors):
        /t/<token>/mcp   ->  routed as  /mcp
    Everything else gets 401. /healthz is open (returns "ok", no info).
    Comparisons are constant-time. The token never appears in logs
    (uvicorn access log is disabled in run_http()).
    """

    def __init__(self, app, token):
        self._app = app
        self._token = token

    async def _deny(self, send, code=401, msg=b'{"error":"unauthorized"}'):
        await send({"type": "http.response.start", "status": code,
                    "headers": [(b"content-type", b"application/json"),
                                (b"www-authenticate", b"Bearer")]})
        await send({"type": "http.response.body", "body": msg})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self._app(scope, receive, send)
        import secrets as _s
        path = scope.get("path", "") or "/"
        if path == "/healthz":
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"text/plain")]})
            await send({"type": "http.response.body", "body": b"ok"})
            return
        authz = b""
        for k, v in scope.get("headers", []):
            if k.lower() == b"authorization":
                authz = v
                break
        ok = False
        if authz.startswith(b"Bearer "):
            ok = _s.compare_digest(authz[7:].decode("latin1").strip(),
                                   self._token)
        if not ok and path.startswith("/t/"):
            tok, _, tail = path[3:].partition("/")
            if _s.compare_digest(tok, self._token):
                ok = True
                scope = dict(scope)
                scope["path"] = "/" + tail
                scope["raw_path"] = scope["path"].encode()
        if not ok:
            return await self._deny(send)
        return await self._app(scope, receive, send)


def run_http():
    import uvicorn
    token = _load_or_create_token()
    # Stateless: every request is self-contained — survives tunnel/edge
    # reconnects and suits cloud connectors (claude.ai / ChatGPT).
    mcp.settings.stateless_http = True
    mcp.settings.host = HTTP_HOST
    mcp.settings.port = HTTP_PORT
    # Allow the public tunnel hostname through FastMCP's DNS-rebinding
    # protection (the bearer token + the tunnel provider is the real gate;
    # requests over the tunnel arrive with Host: <your public hostname>).
    # Set LTM_PUBLIC_HOST to your tunnel hostname, e.g. "ltm.example.com".
    from mcp.server.transport_security import TransportSecuritySettings
    allowed_hosts = ["127.0.0.1", f"127.0.0.1:{HTTP_PORT}",
                     "localhost", f"localhost:{HTTP_PORT}"]
    allowed_origins = []
    public_host = os.environ.get("LTM_PUBLIC_HOST", "").strip()
    if public_host:
        allowed_hosts.append(public_host)
        allowed_origins.append("https://" + public_host)
    mcp.settings.transport_security = TransportSecuritySettings(
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
    )
    app = _BearerGate(mcp.streamable_http_app(), token)
    log.info("LTM MCP streamable-HTTP on %s:%s (bearer required; "
             "token file: %s)", HTTP_HOST, HTTP_PORT, TOKEN_FILE)
    # access_log=False on purpose: URLs may carry the path-embedded token.
    uvicorn.run(app, host=HTTP_HOST, port=HTTP_PORT,
                access_log=False, log_level="warning")


if __name__ == "__main__":
    if "--http" in sys.argv:
        run_http()
    else:
        mcp.run()  # stdio transport
